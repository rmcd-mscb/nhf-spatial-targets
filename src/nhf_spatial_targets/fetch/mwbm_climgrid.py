"""Fetch USGS MWBM (ClimGrid-forced) monthly outputs from ScienceBase.

Single ~7.5 GB CF-conformant NetCDF (ClimGrid_WBM.nc); the fetch is
purely a download — no consolidation step. sha256 + size are persisted
in manifest.json for idempotency and corruption detection.
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
# Stability-window seconds for the repair branch: stat the file, sleep,
# stat again. If size or mtime changed, something is writing to it and
# we refuse to fingerprint a mid-copy file. Tunable for tests.
_REPAIR_STABILITY_SECONDS = 0.5


def _verify_file_stable(path: Path) -> None:
    """Raise RuntimeError if the file is being modified concurrently.

    Stats the file twice across `_REPAIR_STABILITY_SECONDS`; if size or
    mtime changed in that window, another process is writing to it and
    we must not fingerprint a half-written copy.

    Used by the repair branch (where the file came from operator action,
    not our own download) to defend against in-flight rsync/cp. The
    download path doesn't need this — sciencebasepy completes the write
    before we hash, and there is no other writer in the single-threaded
    CLI.
    """
    s_initial = path.stat()
    time.sleep(_REPAIR_STABILITY_SECONDS)
    s_after = path.stat()
    if s_after.st_size != s_initial.st_size or s_after.st_mtime != s_initial.st_mtime:
        raise RuntimeError(
            f"{path} is being modified concurrently (size or mtime "
            f"changed during the {_REPAIR_STABILITY_SECONDS:.2f}s "
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


def _validate_downloaded_nc(nc_path: Path, meta: dict) -> None:
    """Verify variable presence and cell_methods match the catalog declaration.

    Raises RuntimeError with a clear message if the publisher-distributed
    file diverges from what the catalog declares, so downstream targets
    don't silently consume mis-aggregated data. Validation failures leave
    the file on disk for inspection; every error message tells the user
    to delete the file before re-running so the next fetch invocation
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
            f"Cannot open downloaded NetCDF {nc_path}: {exc}. "
            f"The file may be truncated or corrupt; delete it and re-run."
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
    """Download ClimGrid_WBM.nc to <datastore>/mwbm_climgrid/.

    Idempotent: skips download if the file is present AND its size +
    sha256 match the values recorded in manifest.json. After download
    (or on a manifest-repair pass over an existing file), sha256 is
    computed by streaming the saved file in 8 MB chunks and persisted
    to manifest.json alongside the file size. Subsequent runs use the
    same streaming hash to verify the file before trusting the
    manifest. Validates expected variables and CF metadata after
    download.

    Parameters
    ----------
    workdir : Path
        Project directory.
    period : str
        Temporal range as "YYYY/YYYY" — validated against the
        publisher-usable window (1900-2020) and recorded in the
        manifest entry. The publisher distributes a single file
        covering all years, so `period` does not select a subset of
        bytes to download; it gates the call (rejecting out-of-window
        years before any network I/O) and provides provenance.

    Returns
    -------
    dict
        Provenance record for manifest.json.
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
    item_id = access["item_id"]
    filename = access["filename"]
    license_str = meta.get("license", "unknown")

    output_dir = ws.raw_dir(_SOURCE_KEY)
    output_dir.mkdir(parents=True, exist_ok=True)
    nc_path = output_dir / filename

    # Fast-path: file present, manifest agrees on size + sha256 → no-op.
    manifest_entry = _read_manifest_entry(workdir)
    if nc_path.exists() and manifest_entry is not None:
        recorded = manifest_entry.get("file", {})
        if recorded.get("size_bytes") == nc_path.stat().st_size and recorded.get(
            "sha256"
        ):
            actual_sha = _hash_file(nc_path)
            if actual_sha == recorded["sha256"]:
                logger.info(
                    "mwbm_climgrid: file matches manifest (size + sha256); "
                    "skipping download."
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
            logger.warning(
                "mwbm_climgrid: file size matches manifest but sha256 "
                "does not. Re-downloading."
            )

    elif nc_path.exists() and manifest_entry is None:
        logger.info(
            "mwbm_climgrid: file present but no manifest entry. "
            "Hashing existing file to reconstruct provenance."
        )
        size_bytes = nc_path.stat().st_size
        if size_bytes == 0:
            raise RuntimeError(
                f"Existing file {nc_path} is zero bytes. Delete it and "
                f"re-run to download fresh."
            )
        # Guard against an in-progress rsync/cp/operator writer: the
        # repair branch fingerprints whatever is on disk, so we must
        # ensure it isn't being mutated underneath us before trusting
        # the hash.
        _verify_file_stable(nc_path)
        sha_hex = _hash_file(nc_path)
        _validate_downloaded_nc(nc_path, meta)
        now_utc = datetime.now(timezone.utc).isoformat()
        file_record = {
            "path": str(nc_path),
            "size_bytes": size_bytes,
            "sha256": sha_hex,
            "downloaded_utc": now_utc,
            "reconstructed": True,
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

    now_utc = datetime.now(timezone.utc).isoformat()

    from sciencebasepy import SbSession

    logger.info("Connecting to ScienceBase (item %s)...", item_id)
    sb = SbSession()
    item = sb.get_item(item_id)
    if not item or "id" not in item:
        raise RuntimeError(
            f"ScienceBase item {item_id} returned an invalid response. "
            f"The item may have been deleted or moved."
        )

    file_infos = sb.get_item_file_info(item)
    file_info_record = next((fi for fi in file_infos if fi["name"] == filename), None)
    if file_info_record is None:
        raise RuntimeError(
            f"File {filename!r} not found in ScienceBase item {item_id}. "
            f"Available files: {sorted(fi['name'] for fi in file_infos)}"
        )

    try:
        sb.download_file(file_info_record["url"], filename, str(output_dir))
    except Exception as exc:
        nc_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ScienceBase download failed for {filename!r}: {exc}"
        ) from exc

    if not nc_path.exists() or nc_path.stat().st_size == 0:
        nc_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download of {filename!r} produced no file or a zero-byte "
            f"file. ScienceBase may be experiencing issues. Try again later."
        )

    size_bytes = nc_path.stat().st_size
    publisher_size = file_info_record.get("size", 0)
    if publisher_size and size_bytes != publisher_size:
        nc_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download size mismatch for {filename!r}: got {size_bytes}, "
            f"publisher reported {publisher_size}. The download may have "
            f"been truncated; re-run."
        )

    sha_hex = _hash_file(nc_path)
    _validate_downloaded_nc(nc_path, meta)

    file_record = {
        "path": str(nc_path),
        "size_bytes": size_bytes,
        "sha256": sha_hex,
        "downloaded_utc": now_utc,
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
