"""Fetch datasets hosted on PANGAEA (WaterGAP 2.2d)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

_SOURCE_KEY = "watergap22d"


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
