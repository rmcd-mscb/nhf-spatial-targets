"""Register a manually-placed USGS MWBM (ClimGrid-forced) NetCDF.

The publisher (ScienceBase item 64c948dbd34e70357a34c11e) gates the
~7.5 GB ``ClimGrid_WBM.nc`` download behind a CAPTCHA / "I'm not a
robot" check, so automated retrieval via ``sciencebasepy`` is not
viable. Instead, the operator downloads the file once via a browser
and drops it into the project's datastore; this module fingerprints
the file (sha256 + size), validates CF metadata, and writes a manifest
entry for downstream provenance.

See ``docs/sources/mwbm_climgrid.md`` for the manual-download
procedure that this module assumes has already been completed.
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

import xarray as xr

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import parse_period, years_in_period
from nhf_spatial_targets.workspace import load as _load_project

logger = logging.getLogger(__name__)

_SOURCE_KEY = "mwbm_climgrid"
_DATA_PERIOD = (1900, 2020)  # publisher's usable window; 1895-1899 is spinup
# Stability-window seconds: stat the file, sleep, stat again. If size
# or mtime changed, an operator rsync/cp is still in flight and we
# refuse to fingerprint a half-written copy. Tunable for tests.
_STABILITY_SECONDS = 0.5


def _verify_file_stable(path: Path) -> None:
    """Raise RuntimeError if the file is being modified concurrently.

    Stats the file twice across `_STABILITY_SECONDS`; if size or
    mtime changed in that window, another process is writing to it and
    we must not fingerprint a half-written copy. Defends against an
    in-progress operator rsync/cp/browser-download finalize.
    """
    s_initial = path.stat()
    time.sleep(_STABILITY_SECONDS)
    s_after = path.stat()
    if s_after.st_size != s_initial.st_size or s_after.st_mtime != s_initial.st_mtime:
        raise RuntimeError(
            f"{path} is being modified concurrently (size or mtime "
            f"changed during the {_STABILITY_SECONDS:.2f}s "
            f"stability window). Wait for the writer to finish, then "
            f"re-run."
        )


def _hash_file(path: Path) -> str:
    """sha256 of `path`, streamed in 8 MB chunks."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _read_manifest_entry(workdir: Path) -> dict | None:
    """Return the mwbm_climgrid entry from manifest.json, or None if absent.

    Raises ValueError if manifest.json exists but cannot be parsed —
    surfaces corruption fast rather than letting the caller mistake a
    corrupt manifest for a missing one and trigger a wasteful 7.5 GB
    re-hash before the corruption is finally detected by `_update_manifest`.
    """
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


def _validate_nc(nc_path: Path, meta: dict) -> None:
    """Verify variable presence and cell_methods match the catalog declaration.

    Raises RuntimeError with a clear message if the placed file diverges
    from what the catalog declares, so downstream targets don't silently
    consume mis-aggregated data. Validation failures leave the file on
    disk for inspection; every error message tells the user to delete
    the file and re-download before re-running so the next invocation
    doesn't loop on the same divergence.
    """
    declared = {v["name"]: v.get("cell_methods") for v in meta["variables"]}
    # File-open failures get the dedicated "truncated/corrupt" wrapper.
    # netCDF4/HDF5 corruption can surface as RuntimeError or KeyError as
    # well as OSError/ValueError, so catch broadly here. The internal
    # validation `raise RuntimeError(...)` calls live OUTSIDE this except
    # and propagate cleanly.
    try:
        ds = xr.open_dataset(nc_path, decode_times=False)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot open {nc_path}: {exc}. The file may be truncated "
            f"or corrupt; delete it and re-download from ScienceBase "
            f"(see docs/sources/mwbm_climgrid.md)."
        ) from exc
    try:
        present = set(ds.data_vars)
        missing = set(declared) - present
        if missing:
            raise RuntimeError(
                f"{nc_path.name} is missing variables {sorted(missing)}. "
                f"Catalog declares {sorted(declared)} but file only "
                f"contains {sorted(present)}. Publisher may have "
                f"reorganized the dataset; check the ScienceBase page. "
                f"Delete {nc_path} before re-running, or this error "
                f"will repeat indefinitely."
            )
        for name, expected in declared.items():
            if expected is None:
                continue
            actual = ds[name].attrs.get("cell_methods")
            if actual != expected:
                raise RuntimeError(
                    f"{nc_path.name}: variable {name!r} has "
                    f"cell_methods={actual!r}, catalog declares "
                    f"{expected!r}. Publisher metadata diverged from "
                    f"this catalog version; do not trust this file "
                    f"for aggregation until the catalog is updated. "
                    f"Delete {nc_path} before re-running, or this error "
                    f"will repeat indefinitely."
                )
    finally:
        ds.close()


def fetch_mwbm_climgrid(workdir: Path, period: str) -> dict:
    """Register a manually-placed ClimGrid_WBM.nc in manifest.json.

    The ScienceBase distribution is gated by a CAPTCHA, so this function
    does **not** download. The operator must manually download
    ``ClimGrid_WBM.nc`` from the publisher (see
    ``docs/sources/mwbm_climgrid.md``) and place it at
    ``<datastore>/mwbm_climgrid/ClimGrid_WBM.nc`` before invoking this.

    Idempotent: if the manifest already records a matching size + sha256
    for the file on disk, returns immediately without re-hashing past
    the size check. Otherwise stability-checks, hashes, validates CF
    metadata, and writes a fresh manifest entry. Raises FileNotFoundError
    with download instructions if the file is absent.

    Parameters
    ----------
    workdir : Path
        Project directory.
    period : str
        Temporal range as "YYYY/YYYY" — validated against the
        publisher-usable window (1900-2020) and recorded in the
        manifest entry. The publisher distributes a single file
        covering all years, so `period` does not select a subset of
        bytes; it gates the call (rejecting out-of-window years before
        any work) and provides provenance.

    Returns
    -------
    dict
        Provenance record for manifest.json.

    Raises
    ------
    FileNotFoundError
        The expected file is not present in the datastore.
    ValueError
        The requested period falls outside the publisher-usable range
        or manifest.json is corrupt.
    RuntimeError
        The file is concurrently being written, is zero bytes, fails
        to open as a NetCDF, or its CF metadata diverges from the
        catalog declaration.
    """
    parse_period(period)
    requested_years = years_in_period(period)
    for y in requested_years:
        if y < _DATA_PERIOD[0] or y > _DATA_PERIOD[1]:
            raise ValueError(
                f"Year {y} is outside the MWBM-ClimGrid data range "
                f"({_DATA_PERIOD[0]}-{_DATA_PERIOD[1]}). The 1895-1899 "
                f"period is publisher-flagged spinup. Adjust --period."
            )

    ws = _load_project(workdir)
    meta = _catalog.source(_SOURCE_KEY)
    access = meta["access"]
    filename = access["filename"]
    license_str = meta.get("license", "unknown")

    output_dir = ws.raw_dir(_SOURCE_KEY)
    output_dir.mkdir(parents=True, exist_ok=True)
    nc_path = output_dir / filename

    if not nc_path.exists():
        raise FileNotFoundError(
            f"Expected manually-placed file not found: {nc_path}\n\n"
            f"ScienceBase gates this ~7.5 GB download behind a CAPTCHA, "
            f"so automated retrieval is not possible. Download "
            f"{filename!r} from {access['url']} via a browser, then "
            f"place it at the path above and re-run.\n\n"
            f"See docs/sources/mwbm_climgrid.md for full instructions."
        )

    size_bytes = nc_path.stat().st_size
    if size_bytes == 0:
        raise RuntimeError(
            f"{nc_path} is zero bytes. The download likely failed or "
            f"is still in progress. Delete it and re-download from "
            f"ScienceBase (see docs/sources/mwbm_climgrid.md)."
        )

    # Fast path: manifest agrees on size + sha256 → no-op (skip the
    # 7.5 GB rehash). The size check uses cheap stat data only; the
    # full hash is computed lazily and only when sizes match, so a
    # tampered file with a different size is detected without paying
    # the streaming-hash cost on every invocation.
    manifest_entry = _read_manifest_entry(workdir)
    if manifest_entry is not None:
        recorded = manifest_entry.get("file", {})
        if (
            recorded.get("size_bytes") == size_bytes
            and recorded.get("sha256")
            and _hash_file(nc_path) == recorded["sha256"]
        ):
            logger.info(
                "mwbm_climgrid: file matches manifest (size + sha256); "
                "skipping re-fingerprint."
            )
            return {
                "source_key": _SOURCE_KEY,
                "access_url": access["url"],
                "doi": meta["doi"],
                "license": license_str,
                "variables": [v["name"] for v in meta["variables"]],
                "period": period,
                "spatial_extent": meta.get("spatial_extent", "CONUS"),
                "download_timestamp": recorded.get("downloaded_utc"),
                "file": recorded,
            }
        logger.info(
            "mwbm_climgrid: file size or sha256 differs from manifest; "
            "re-fingerprinting."
        )

    # File exists but either has no manifest entry, or its fingerprint
    # disagrees with the recorded one (operator replaced the file).
    # Verify it isn't currently being written, then hash + validate.
    _verify_file_stable(nc_path)
    sha_hex = _hash_file(nc_path)
    _validate_nc(nc_path, meta)
    now_utc = datetime.now(timezone.utc).isoformat()
    file_record = {
        "path": str(nc_path),
        "size_bytes": size_bytes,
        "sha256": sha_hex,
        "registered_utc": now_utc,
        "manual_download": True,
    }
    _update_manifest(workdir, period, meta, license_str, file_record)
    return {
        "source_key": _SOURCE_KEY,
        "access_url": access["url"],
        "doi": meta["doi"],
        "license": license_str,
        "variables": [v["name"] for v in meta["variables"]],
        "period": period,
        "spatial_extent": meta.get("spatial_extent", "CONUS"),
        "download_timestamp": now_utc,
        "file": file_record,
    }


def _update_manifest(
    workdir: Path,
    period: str,
    meta: dict,
    license_str: str,
    file_record: dict,
) -> None:
    """Merge mwbm_climgrid provenance into manifest.json."""
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
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
    access = meta["access"]
    entry = manifest["sources"].get(_SOURCE_KEY, {})
    entry.update(
        {
            "source_key": _SOURCE_KEY,
            "access_url": access["url"],
            "doi": meta["doi"],
            "license": license_str,
            "period": period,
            "spatial_extent": meta.get("spatial_extent", "CONUS"),
            "variables": [v["name"] for v in meta["variables"]],
            "file": file_record,
        }
    )
    manifest["sources"][_SOURCE_KEY] = entry

    tmp_fd, tmp_path = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp_path).replace(manifest_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    logger.info("Updated manifest.json with mwbm_climgrid provenance")
