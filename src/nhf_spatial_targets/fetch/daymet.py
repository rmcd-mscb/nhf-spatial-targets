"""Register operator-staged Daymet V4 R1 regional zarr stores.

Daymet V4 R1 is distributed by ORNL DAAC as three regional zarr stores —
North America (na), Hawaii (hi), and Puerto Rico (pr) — totalling hundreds
of gigabytes per region. We do not download or copy them: the operator
pre-stages the zarrs at a shared filesystem location, and this module
fingerprints each region (sha256 over the zarr's structural metadata
files: ``.zattrs``, ``.zgroup``, and every per-array ``.zarray`` and
``.zattrs``, walked in sorted order — plus the total on-disk byte
count) and records the result in ``manifest.json`` for downstream
provenance. The fingerprint is intentionally cheap; a full-data hash
would take many hours per region and add no defensive value beyond what
the metadata already encodes (every chunk rewrite changes either a
``.zarray`` shape/dtype/chunks block, an attribute file, or the
directory's total byte count).

**Residual risk in the fingerprint.** A chunk file rewritten in place
with **identical compressed length** but different content bytes (e.g.
the operator re-ran the publisher's converter and got a different
floating-point shuffle, or an ``rsync --inplace`` overlaid a
same-length corruption) is invisible to both the metadata hash and
the directory byte count. In practice this is improbable for V4 R1
(blosc-shuffled chunks producing the exact same compressed length
from different input is statistically rare), and zarr v2 records no
per-chunk checksums. Operators who suspect chunk-level tampering
should re-stage the zarrs from the ORNL DAAC distribution.

Concurrent invocation: ``fetch daymet --region <r>`` is safe to run
in parallel across regions; the per-region manifest update takes an
exclusive ``flock`` on a ``manifest.json.lock`` sibling so two
processes never lose each other's region records.

See ``docs/sources/daymet.md`` for the operator workflow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl as _fcntl

    _HAVE_FLOCK = True
except ImportError:  # Windows fallback (not used on HPC).
    _HAVE_FLOCK = False

import xarray as xr

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import (
    parse_period,
    period_bounds,
    years_in_period,
)
from nhf_spatial_targets.workspace import load as _load_project

logger = logging.getLogger(__name__)

_SOURCE_KEY = "daymet"
_REGIONS = ("na", "hi", "pr")
# Stability window for detecting concurrent writes: stat the .zmetadata
# file twice, sleep, compare. Tunable for tests.
_STABILITY_SECONDS = 0.5
# Variables every Daymet regional zarr is expected to expose. Mirrors
# the catalog `variables:` block; checked at registration time.
_REQUIRED_VARS = {"swe", "prcp", "tmax", "tmin", "srad", "vp"}


def _metadata_files(zroot: Path) -> list[Path]:
    """Return all zarr structural metadata files in deterministic order.

    Includes ``.zgroup`` / ``.zattrs`` at the top level and every
    ``.zarray`` / ``.zattrs`` / ``.zgroup`` underneath. If a
    ``.zmetadata`` (consolidated metadata) file is present, it is
    included too — it is the cheapest single-file authoritative
    summary. Walking metadata is fast (kilobytes per region) so this
    cost is negligible compared to a chunk hash.
    """
    names = {".zgroup", ".zattrs", ".zarray", ".zmetadata"}
    out: list[Path] = []
    for child in zroot.rglob("*"):
        if child.is_file() and child.name in names:
            out.append(child)
    out.sort(key=lambda p: str(p.relative_to(zroot)))
    return out


def _verify_zarr_stable(zroot: Path) -> None:
    """Raise RuntimeError if any structural metadata file is being written."""
    files = _metadata_files(zroot)
    if not files:
        raise RuntimeError(
            f"{zroot}: zarr has no .zgroup / .zarray / .zattrs metadata "
            f"files; this is not a valid zarr v2 store."
        )
    snap = [(p, p.stat().st_size, p.stat().st_mtime) for p in files]
    time.sleep(_STABILITY_SECONDS)
    for p, size, mtime in snap:
        st = p.stat()
        if st.st_size != size or st.st_mtime != mtime:
            raise RuntimeError(
                f"{p} is being modified concurrently (size or mtime "
                f"changed during the {_STABILITY_SECONDS:.2f}s stability "
                f"window). Wait for the writer to finish, then re-run."
            )


def _hash_zarr_metadata(zroot: Path) -> str:
    """sha256 over the zarr's structural metadata files, sorted by path.

    The hash incorporates each file's relative path AND content so that
    adding/removing arrays changes the digest even when the per-file
    bytes would otherwise reshuffle.
    """
    sha = hashlib.sha256()
    for p in _metadata_files(zroot):
        sha.update(str(p.relative_to(zroot)).encode("utf-8"))
        sha.update(b"\0")
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(64 * 1024), b""):
                sha.update(chunk)
        sha.update(b"\0")
    return sha.hexdigest()


def _zarr_directory_size(zroot: Path) -> int:
    """Total on-disk byte count of all files inside the zarr store.

    Used alongside the metadata sha256 as a defensive fingerprint: a
    chunk file rewritten without touching ``.zarray`` / ``.zattrs`` is
    unusual but possible (e.g. rsync --inplace), and the directory size
    will diverge in that case.
    """
    total = 0
    for child in zroot.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _read_manifest_entry(workdir: Path) -> dict | None:
    """Return the daymet manifest entry, or None if absent."""
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manifest.json in {workdir} is corrupt and cannot be parsed. "
            f"Delete it and re-run the fetch step. Original error: {exc}"
        ) from exc
    return manifest.get("sources", {}).get(_SOURCE_KEY)


def _validate_zarr(zroot: Path, region: str) -> dict:
    """Open the zarr, assert required vars + dims, return time-range info.

    Returns ``{"time_min": iso, "time_max": iso, "time_steps": int}``.

    Raises RuntimeError if the zarr is missing the SWE variable, the
    canonical CF coords (time/x/y), or the CRS spec
    (``lambert_conformal_conic``).
    """
    try:
        # Real Daymet zarrs are v2 without consolidated metadata.
        ds = xr.open_zarr(zroot, consolidated=False)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot open zarr {zroot}: {exc}. The store may be incomplete "
            f"or corrupt; verify the operator staging step completed."
        ) from exc
    try:
        # Required vars must be data_vars, not coords. A coord coincidentally
        # named "swe" should not mask a missing data variable.
        present_data_vars = set(ds.data_vars)
        missing = _REQUIRED_VARS - present_data_vars
        if missing:
            raise RuntimeError(
                f"{zroot} (region={region}) is missing required variables "
                f"{sorted(missing)}. Daymet V4 R1 should expose "
                f"{sorted(_REQUIRED_VARS)} on every regional zarr. "
                f"Re-stage the publisher's zarr before re-running."
            )
        if "time" not in ds.coords and "time" not in ds.dims:
            raise RuntimeError(
                f"{zroot}: zarr has no 'time' coordinate; cannot use as a "
                f"calibration source."
            )
        for axis in ("x", "y"):
            if axis not in ds.coords and axis not in ds.dims:
                raise RuntimeError(
                    f"{zroot}: zarr has no '{axis}' coordinate; "
                    f"Daymet uses (time, y, x) projected coords."
                )
        # The CRS spec is a scalar var that can show up under data_vars or
        # coords depending on how the zarr was written; check both.
        if "lambert_conformal_conic" not in (present_data_vars | set(ds.coords)):
            logger.warning(
                "%s (region=%s): missing CRS variable "
                "'lambert_conformal_conic'. Aggregation steps will likely "
                "fail to identify the projection.",
                zroot,
                region,
            )
        times = ds["time"].values
        if len(times) == 0:
            raise RuntimeError(f"{zroot}: time coord is empty.")
        return {
            "time_min": str(times[0]),
            "time_max": str(times[-1]),
            "time_steps": int(len(times)),
        }
    finally:
        ds.close()


def _resolve_root(
    workdir: Path,
    source_path: Path | None,
) -> Path:
    """Return the absolute path to the directory containing the three zarrs.

    Order of precedence: ``source_path`` arg → ``config.yml: daymet_root``
    → raise ``FileNotFoundError`` with operator instructions. The catalog
    `access.stores` values use a ``{daymet_root}`` template placeholder
    that is substituted with the resolved root.
    """
    if source_path is not None:
        return Path(source_path).resolve()
    ws = _load_project(workdir)
    cfg_root = ws.config.get("daymet_root")
    if cfg_root:
        return Path(cfg_root).resolve()
    raise FileNotFoundError(
        "Daymet zarr root is not configured. Either pass "
        "--source-path <dir> on the CLI, or add `daymet_root: /path/to/...` "
        "to config.yml. The directory must contain daymet_na.zarr, "
        "daymet_hi.zarr, and/or daymet_pr.zarr. See docs/sources/daymet.md."
    )


def _resolve_region_paths(root: Path, regions: tuple[str, ...]) -> dict[str, Path]:
    """Apply the ``{daymet_root}`` template from the catalog to ``root``."""
    meta = _catalog.source(_SOURCE_KEY)
    template = meta["access"]["stores"]
    out: dict[str, Path] = {}
    for region in regions:
        try:
            tmpl = template[region]
        except KeyError as exc:
            raise ValueError(
                f"region {region!r} is not declared in catalog "
                f"sources.yml[daymet].access.stores; known: "
                f"{sorted(template.keys())}"
            ) from exc
        out[region] = Path(tmpl.format(daymet_root=str(root)))
    return out


def fetch_daymet(
    workdir: Path,
    period: str,
    *,
    source_path: Path | None = None,
    region: str = "all",
) -> dict:
    """Register pre-staged Daymet V4 R1 regional zarr stores in manifest.json.

    The three regional zarrs (NA, HI, PR) live on a shared filesystem; this
    module verifies each one opens cleanly, fingerprints it, and records a
    per-region entry under ``manifest["sources"]["daymet"]["regions"]``.
    No download and no copy: the catalog's ``access.stores`` map carries
    a ``{daymet_root}`` template that is filled with the resolved root.

    Parameters
    ----------
    workdir : Path
        Project directory.
    period : str
        Temporal window ``"YYYY/YYYY"``. Validated against the publisher-
        usable window (1980/2024) and recorded in the manifest entry.
    source_path : Path or None
        Directory containing the regional zarrs. If ``None``, falls back
        to ``config.yml -> daymet_root``.
    region : str
        One of ``"na" | "hi" | "pr" | "all"``. ``"all"`` registers every
        region present at ``source_path``; missing regions are reported
        but do not raise (operators frequently stage only NA).

    Returns
    -------
    dict
        Provenance summary with one entry per registered region.

    Raises
    ------
    FileNotFoundError
        Neither ``source_path`` nor ``daymet_root`` resolves to an
        existing directory, or none of the requested regional zarrs
        are present.
    ValueError
        Period falls outside the publisher window or ``region`` is
        not one of the documented values.
    RuntimeError
        A zarr is being written concurrently, fails to open, or is
        missing required variables.
    """
    parse_period(period)
    meta = _catalog.source(_SOURCE_KEY)
    data_lo, data_hi = period_bounds(meta["period"])
    for y in years_in_period(period):
        if y < data_lo or y > data_hi:
            raise ValueError(
                f"Year {y} is outside the Daymet V4 R1 publisher window "
                f"({data_lo}-{data_hi}, from catalog `sources.yml[{_SOURCE_KEY}]"
                f".period`). Adjust --period."
            )

    if region not in (*_REGIONS, "all"):
        raise ValueError(
            f"region={region!r} is not recognised; expected one of "
            f"{(*_REGIONS, 'all')}."
        )

    root = _resolve_root(workdir, source_path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(
            f"Daymet zarr root does not exist or is not a directory: {root}. "
            f"Pass --source-path or set `daymet_root:` in config.yml to "
            f"a directory containing daymet_na.zarr / daymet_hi.zarr / "
            f"daymet_pr.zarr."
        )

    requested = _REGIONS if region == "all" else (region,)
    region_paths = _resolve_region_paths(root, requested)
    present: dict[str, Path] = {
        r: p for r, p in region_paths.items() if p.exists() and p.is_dir()
    }
    missing = [r for r in requested if r not in present]
    if not present:
        raise FileNotFoundError(
            f"None of the requested Daymet regional zarrs exist under {root}. "
            f"Looked for {[str(p) for p in region_paths.values()]}. "
            f"See docs/sources/daymet.md for the staging procedure."
        )
    for r in missing:
        logger.warning(
            "daymet: region %r not present at %s; skipping.",
            r,
            region_paths[r],
        )

    license_str = meta.get("license", "unknown")
    now_utc = datetime.now(timezone.utc).isoformat()

    # Idempotency fast path: if every requested-and-present region's
    # manifest entry already matches (zmetadata sha256 + directory size),
    # return cached records without re-hashing.
    existing = _read_manifest_entry(workdir) or {}
    existing_regions = existing.get("regions", {}) if isinstance(existing, dict) else {}

    region_records: dict[str, dict] = {}
    fingerprinted_any = False
    for r, zroot in present.items():
        # Quick structural check: the zarr must have a .zgroup at the
        # root, otherwise it's not a valid zarr v2 hierarchy.
        if not (zroot / ".zgroup").exists():
            raise RuntimeError(
                f"{zroot}: not a valid zarr v2 store (missing .zgroup at "
                f"root). Re-stage the publisher zarr."
            )
        size_bytes = _zarr_directory_size(zroot)
        recorded = existing_regions.get(r) or {}
        if (
            recorded.get("size_bytes") == size_bytes
            and recorded.get("zmetadata_sha256")
            and _hash_zarr_metadata(zroot) == recorded["zmetadata_sha256"]
        ):
            logger.info(
                "daymet: region %r matches manifest (size + metadata "
                "sha256); skipping re-fingerprint.",
                r,
            )
            region_records[r] = recorded
            continue

        _verify_zarr_stable(zroot)
        sha_hex = _hash_zarr_metadata(zroot)
        time_info = _validate_zarr(zroot, region=r)
        region_records[r] = {
            "region": r,
            "path": str(zroot),
            "size_bytes": size_bytes,
            # Stored under the legacy key ``zmetadata_sha256`` for
            # operator-facing continuity; the value is a hash over all
            # zarr structural metadata, not just ``.zmetadata``.
            "zmetadata_sha256": sha_hex,
            "time_min": time_info["time_min"],
            "time_max": time_info["time_max"],
            "time_steps": time_info["time_steps"],
            "registered_utc": now_utc,
            "manual_staging": True,
        }
        fingerprinted_any = True

    if fingerprinted_any:
        _update_manifest(workdir, period, meta, license_str, region_records)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "doi": meta.get("doi"),
        "license": license_str,
        "variables": [v["name"] for v in meta["variables"]],
        "period": period,
        "spatial_extent": meta.get("spatial_extent"),
        "download_timestamp": now_utc if fingerprinted_any else None,
        "regions": region_records,
        "missing_regions": missing,
    }


def _update_manifest(
    workdir: Path,
    period: str,
    meta: dict,
    license_str: str,
    region_records: dict[str, dict],
) -> None:
    """Merge daymet provenance into manifest.json (per-region entries).

    Operators may invoke ``fetch daymet --region na`` and
    ``fetch daymet --region hi`` concurrently; the read-merge-write
    cycle takes an exclusive ``flock`` on a ``manifest.json.lock``
    sibling so parallel processes never lose each other's region
    records.
    """
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    def _do_update() -> None:
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"manifest.json in {workdir} is corrupt and cannot be "
                    f"parsed. Delete it and re-run the fetch step. "
                    f"Original error: {exc}"
                ) from exc
        else:
            manifest = {"sources": {}, "steps": []}

        manifest.setdefault("sources", {})
        entry = manifest["sources"].get(_SOURCE_KEY, {})
        existing_regions = entry.get("regions", {}) or {}
        # Merge: keep regions we didn't touch this run, overwrite the ones we did.
        merged_regions = {**existing_regions, **region_records}
        access = meta["access"]
        entry.update(
            {
                "source_key": _SOURCE_KEY,
                "access_url": access["url"],
                "doi": meta.get("doi"),
                "license": license_str,
                "period": period,
                "spatial_extent": meta.get("spatial_extent"),
                "variables": [v["name"] for v in meta["variables"]],
                "regions": merged_regions,
            }
        )
        manifest["sources"][_SOURCE_KEY] = entry

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=manifest_path.parent, suffix=".json.tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(manifest, f, indent=2)
            Path(tmp_path).replace(manifest_path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    if _HAVE_FLOCK:
        with open(lock_path, "a") as _lock_f:
            _fcntl.flock(_lock_f, _fcntl.LOCK_EX)
            try:
                _do_update()
            finally:
                _fcntl.flock(_lock_f, _fcntl.LOCK_UN)
    else:
        _do_update()
    logger.info(
        "Updated manifest.json with daymet provenance for regions %s",
        sorted(region_records.keys()),
    )
