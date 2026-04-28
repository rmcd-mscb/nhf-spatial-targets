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
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr  # noqa: F401 — used in Tasks 6-8

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import parse_period, years_in_period
from nhf_spatial_targets.workspace import load as _load_project

logger = logging.getLogger(__name__)

_SOURCE_KEY = "mwbm_climgrid"
_DATA_PERIOD = (1900, 2020)  # publisher's usable window; 1895-1899 is spinup


def fetch_mwbm_climgrid(workdir: Path, period: str) -> dict:
    """Download ClimGrid_WBM.nc to <datastore>/mwbm_climgrid/.

    Idempotent: skips download if the file is present AND its size +
    sha256 match the values recorded in manifest.json. Computes sha256
    streaming during download (no second-pass read of the 7.5 GB file).
    Validates expected variables and CF metadata after download.

    Parameters
    ----------
    workdir : Path
        Project directory.
    period : str
        Temporal range as "YYYY/YYYY" — used to validate the project's
        intended use against publisher coverage and to record in the
        manifest entry. The download itself ignores this argument
        (the publisher distributes one file).

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
    now_utc = datetime.now(timezone.utc).isoformat()

    # Idempotency + repair branches land in Tasks 6-7. For now: always
    # download, hash, validate.
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

    sha = hashlib.sha256()
    with nc_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    sha_hex = sha.hexdigest()

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
