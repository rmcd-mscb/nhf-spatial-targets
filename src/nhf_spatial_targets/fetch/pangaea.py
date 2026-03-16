"""Fetch datasets hosted on PANGAEA (WaterGAP 2.2d)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import parse_period

logger = logging.getLogger(__name__)

_SOURCE_KEY = "watergap22d"
_PANGAEA_DATASET_ID = 918447
_PANGAEA_FILE_INDEX = 30  # row index for qrdif file in PanDataSet.data
_RAW_FILENAME = "watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4"
_CF_FILENAME = "watergap22d_qrdif_cf.nc"


def _cf_fixup(raw_path: Path, output_path: Path) -> Path:
    """Fix CF compliance issues in a WaterGAP NC4 file.

    Addresses:
    - Time encoding: reconstructs 'months since 1901-01-01' offsets as datetime64
    - Grid mapping: adds WGS84 crs variable and grid_mapping attr on data vars
    - Conventions: sets to CF-1.6

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
    ds.time.attrs = {"standard_name": "time", "long_name": "time", "axis": "T"}

    # --- Add CRS variable with WGS84 grid mapping ---
    crs = xr.DataArray(
        np.int32(0),
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
            "longitude_of_prime_meridian": 0.0,
            "crs_wkt": (
                'GEOGCS["WGS 84",'
                'DATUM["WGS_1984",'
                'SPHEROID["WGS 84",6378137,298.257223563]],'
                'PRIMEM["Greenwich",0],'
                'UNIT["degree",0.0174532925199433]]'
            ),
        },
    )
    ds["crs"] = crs

    # --- Set grid_mapping on data variables ---
    for var in ds.data_vars:
        if var != "crs":
            ds[var].attrs["grid_mapping"] = "crs"

    # --- Set Conventions ---
    ds.attrs["Conventions"] = "CF-1.6"
    # Remove the old non-standard conventions attr if present
    ds.attrs.pop("conventions", None)

    # --- Write ---
    ds.to_netcdf(output_path, format="NETCDF4")
    ds.close()

    logger.info("Wrote CF-compliant file: %s", output_path)
    return output_path


def fetch_watergap22d(run_dir: Path, period: str) -> dict:
    """Download WaterGAP 2.2d diffuse groundwater recharge from PANGAEA.

    Downloads the single NC4 file via pangaeapy, applies CF compliance
    fixes, and updates manifest.json. Skips download if the CF-corrected
    file already exists.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory.
    period : str
        Temporal range as ``"YYYY/YYYY"``. Used for provenance only —
        the downloaded file covers the full 1901–2016 period.

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    meta = _catalog.source(_SOURCE_KEY)
    parse_period(period)  # validate format

    fabric_path = run_dir / "fabric.json"
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"fabric.json not found in {run_dir}. "
            f"Run 'nhf-targets init' to create a run workspace first."
        )
    try:
        fabric = json.loads(fabric_path.read_text())
        bbox = fabric["bbox_buffered"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(
            f"fabric.json in {run_dir} is malformed or missing required "
            f"fields (bbox_buffered.{{minx,miny,maxx,maxy}}). "
            f"Re-run 'nhf-targets init' to regenerate it."
        ) from exc

    output_dir = run_dir / "data" / "raw" / _SOURCE_KEY
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / _RAW_FILENAME
    cf_path = output_dir / _CF_FILENAME
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
                _PANGAEA_DATASET_ID,
            )
            try:
                cache_dir = output_dir / ".pangaea_cache"
                cache_dir.mkdir(exist_ok=True)
                pan_ds = PanDataSet(
                    _PANGAEA_DATASET_ID,
                    cachedir=str(cache_dir),
                )

                # Validate file index before downloading
                actual_name = pan_ds.data.loc[_PANGAEA_FILE_INDEX, "File name"]
                if actual_name != _RAW_FILENAME:
                    raise RuntimeError(
                        f"PANGAEA file index {_PANGAEA_FILE_INDEX} contains "
                        f"'{actual_name}', expected '{_RAW_FILENAME}'. "
                        f"The dataset listing may have changed."
                    )

                downloaded = pan_ds.download(indices=[_PANGAEA_FILE_INDEX])
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download WaterGAP 2.2d from PANGAEA "
                    f"(dataset {_PANGAEA_DATASET_ID}): {exc}. "
                    f"Check network connectivity and PANGAEA availability."
                ) from exc

            if not downloaded:
                raise RuntimeError(
                    f"pangaeapy returned no files for dataset "
                    f"{_PANGAEA_DATASET_ID}, index {_PANGAEA_FILE_INDEX}."
                )

            cached_path = Path(downloaded[0])
            shutil.move(str(cached_path), str(raw_path))
            logger.info("Moved %s -> %s", cached_path, raw_path)

        # CF compliance fix-up
        logger.info("Applying CF compliance fixes...")
        _cf_fixup(raw_path, cf_path)

    # Build provenance
    file_info = {
        "path": str(cf_path.relative_to(run_dir)),
        "raw_path": (str(raw_path.relative_to(run_dir)) if raw_path.exists() else None),
        "size_bytes": cf_path.stat().st_size,
        "downloaded_utc": now_utc,
    }

    _update_manifest(run_dir, period, bbox, meta, file_info)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "doi": meta["access"]["doi"],
        "license": "CC BY-NC 4.0",
        "variables": meta["variables"],
        "period": period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "file": file_info,
        "cf_corrected_file": str(cf_path.relative_to(run_dir)),
    }


def _update_manifest(
    run_dir: Path,
    period: str,
    bbox: dict,
    meta: dict,
    file_info: dict,
) -> None:
    """Merge WaterGAP 2.2d provenance into manifest.json."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"sources": {}, "steps": []}

    if "sources" not in manifest:
        manifest["sources"] = {}

    entry = manifest["sources"].get(_SOURCE_KEY, {})
    entry.update(
        {
            "source_key": _SOURCE_KEY,
            "access_url": meta["access"]["url"],
            "doi": meta["access"]["doi"],
            "license": "CC BY-NC 4.0",
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
