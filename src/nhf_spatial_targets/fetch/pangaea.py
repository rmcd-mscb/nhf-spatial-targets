"""Fetch datasets hosted on PANGAEA (WaterGAP 2.2d)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xarray as xr

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import parse_period
from nhf_spatial_targets.workspace import load as _load_workspace

logger = logging.getLogger(__name__)

_SOURCE_KEY = "watergap22d"


def _cf_fixup(raw_path: Path, output_path: Path) -> Path:
    """Fix CF compliance issues in a WaterGAP NC4 file.

    Addresses:
    - Time encoding: reconstructs 'months since 1901-01-01' offsets as datetime64
    - CF-1.6 metadata: delegates to ``apply_cf_metadata`` for grid mapping,
      coordinate attrs, time_bnds, and Conventions

    Writes to a temporary file and atomically renames to avoid leaving
    partial/corrupt output on failure.

    Parameters
    ----------
    raw_path : Path
        Path to the original (non-CF-compliant) NetCDF file.
    output_path : Path
        Path to write the corrected NetCDF file.

    Returns
    -------
    Path
        Path to the written CF-compliant file (same as output_path).
    """
    ds = xr.open_dataset(raw_path, decode_times=False)
    tmp_path = output_path.with_suffix(".nc.tmp")
    try:
        # --- Reconstruct time coordinate ---
        time_offsets = ds.time.values.astype(int)
        base_year = 1901
        base_month = 1
        dates = []
        for offset in time_offsets:
            total_months = base_month - 1 + offset
            year = base_year + total_months // 12
            month = 1 + total_months % 12
            dates.append(pd.Timestamp(year=int(year), month=int(month), day=1))
        new_time = pd.DatetimeIndex(dates)
        ds = ds.assign_coords(time=new_time)

        # --- Apply CF-1.6 metadata via shared helper ---
        from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

        ds = apply_cf_metadata(ds, "watergap22d", "monthly")

        # --- Write atomically ---
        ds.to_netcdf(tmp_path, format="NETCDF4")
        tmp_path.rename(output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        ds.close()

    logger.info("Wrote CF-compliant file: %s", output_path)
    return output_path


def fetch_watergap22d(workdir: Path, period: str) -> dict:
    """Download WaterGAP 2.2d diffuse groundwater recharge from PANGAEA.

    Downloads the single NC4 file via pangaeapy, applies CF compliance
    fixes, and updates manifest.json. Skips download if the CF-corrected
    file already exists.

    Parameters
    ----------
    workdir : Path
        Workspace directory.
    period : str
        Temporal range as ``"YYYY/YYYY"``. Used for provenance only —
        the downloaded file covers the full 1901-2016 period.

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    ws = _load_workspace(workdir)
    meta = _catalog.source(_SOURCE_KEY)
    parse_period(period)  # validate format

    access = meta["access"]
    raw_filename = access["file"]
    dataset_id = int(access["doi"].rsplit(".", 1)[-1])
    file_index = access["file_index"]
    license_str = meta.get("license", "unknown")
    cf_filename = f"{_SOURCE_KEY}_qrdif_cf.nc"

    bbox = ws.fabric["bbox_buffered"]

    output_dir = ws.raw_dir(_SOURCE_KEY)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / raw_filename
    cf_path = output_dir / cf_filename
    now_utc = datetime.now(timezone.utc).isoformat()

    if cf_path.exists():
        logger.info(
            "CF-corrected file already exists, skipping download: %s",
            cf_path,
        )
    else:
        # Download from PANGAEA
        if not raw_path.exists():
            from pangaeapy import PanDataSet

            logger.info(
                "Downloading WaterGAP 2.2d from PANGAEA (dataset %d)...",
                dataset_id,
            )
            cache_dir = output_dir / ".pangaea_cache"
            cache_dir.mkdir(exist_ok=True)
            try:
                pan_ds = PanDataSet(dataset_id, cachedir=str(cache_dir))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to connect to PANGAEA dataset {dataset_id}: "
                    f"{exc}. Check network connectivity and PANGAEA "
                    f"availability."
                ) from exc

            # Validate file index before downloading.
            # PANGAEA's "File name" column omits the extension (format is
            # in a separate "File format" column), so compare against the
            # filename stem.
            actual_name = pan_ds.data.loc[file_index, "File name"]
            expected_stem = Path(raw_filename).stem
            if actual_name != expected_stem:
                raise RuntimeError(
                    f"PANGAEA file index {file_index} contains "
                    f"'{actual_name}', expected '{expected_stem}'. "
                    f"The dataset listing may have changed."
                )

            try:
                downloaded = pan_ds.download(indices=[file_index])
            except Exception as exc:
                raise RuntimeError(
                    f"PANGAEA download failed for dataset {dataset_id}: {exc}"
                ) from exc

            if not downloaded:
                raise RuntimeError(
                    f"pangaeapy returned no files for dataset "
                    f"{dataset_id}, index {file_index}."
                )

            cached_path = Path(downloaded[0])
            shutil.move(str(cached_path), str(raw_path))
            logger.info("Moved %s -> %s", cached_path, raw_path)

        # CF compliance fix-up
        logger.info("Applying CF compliance fixes...")
        _cf_fixup(raw_path, cf_path)

    if not cf_path.exists():
        raise RuntimeError(
            f"CF-corrected file was not created at {cf_path}. "
            f"The CF fix-up step may have failed."
        )

    # Build provenance
    file_info = {
        "path": str(cf_path),
        "raw_path": (str(raw_path) if raw_path.exists() else None),
        "size_bytes": cf_path.stat().st_size,
        "downloaded_utc": now_utc,
    }

    _update_manifest(workdir, period, bbox, meta, license_str, file_info)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": access["url"],
        "doi": access["doi"],
        "license": license_str,
        "variables": meta["variables"],
        "period": period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "file": file_info,
        "cf_corrected_file": str(cf_path),
    }


def _update_manifest(
    workdir: Path,
    period: str,
    bbox: dict,
    meta: dict,
    license_str: str,
    file_info: dict,
) -> None:
    """Merge WaterGAP 2.2d provenance into manifest.json."""
    ws = _load_workspace(workdir)
    manifest_path = ws.manifest_path
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {workdir} is corrupt and cannot be "
                f"parsed. You may need to delete it and re-run the fetch "
                f"step. Original error: {exc}"
            ) from exc
    else:
        manifest = {"sources": {}, "steps": []}

    if "sources" not in manifest:
        manifest["sources"] = {}

    access = meta["access"]
    entry = manifest["sources"].get(_SOURCE_KEY, {})
    entry.update(
        {
            "source_key": _SOURCE_KEY,
            "access_url": access["url"],
            "doi": access["doi"],
            "license": license_str,
            "period": period,
            "bbox": bbox,
            "variables": [v["name"] for v in meta["variables"]],
            "file": file_info,
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
    logger.info("Updated manifest.json with WaterGAP 2.2d provenance")
