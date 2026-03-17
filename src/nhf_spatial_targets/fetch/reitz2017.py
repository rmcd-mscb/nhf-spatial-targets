"""Fetch Reitz et al. (2017) recharge estimates from USGS ScienceBase."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr

from nhf_spatial_targets.fetch._period import years_in_period

logger = logging.getLogger(__name__)

_SOURCE_KEY = "reitz2017"
_CONSOLIDATED_FILENAME = "reitz2017_consolidated.nc"
_DATA_PERIOD = (2000, 2013)


def _year_from_filename(path: Path) -> int:
    """Extract year from filenames like TotalRecharge_2005.tif."""
    stem = path.stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot extract year from filename: {path.name}")
    try:
        return int(parts[1])
    except ValueError:
        raise ValueError(f"Cannot extract year from filename: {path.name}") from None


def _consolidate(output_dir: Path, period: str) -> Path:
    """Read annual GeoTIFFs and consolidate into a single NetCDF.

    Parameters
    ----------
    output_dir : Path
        Directory containing TotalRecharge_YYYY.tif and EffRecharge_YYYY.tif files.
    period : str
        Period as "YYYY/YYYY" — only files within this range are included.

    Returns
    -------
    Path
        Path to the consolidated NetCDF file.
    """
    years = years_in_period(period)
    nc_path = output_dir / _CONSOLIDATED_FILENAME
    tmp_path = nc_path.with_suffix(".nc.tmp")

    var_configs = [
        ("TotalRecharge", "total_recharge"),
        ("EffRecharge", "eff_recharge"),
    ]

    # Collect per-variable DataArrays keyed by year
    var_arrays: dict[str, dict[int, xr.DataArray]] = {
        ds_name: {} for _, ds_name in var_configs
    }

    for file_prefix, ds_name in var_configs:
        for tif_path in sorted(output_dir.glob(f"{file_prefix}_*.tif")):
            year = _year_from_filename(tif_path)
            if year not in years:
                continue
            da = rioxarray.open_rasterio(tif_path, masked=True)
            da = da.squeeze("band", drop=True)
            var_arrays[ds_name][year] = da

    # Validate pairing: every year must have both variables
    total_years = set(var_arrays["total_recharge"].keys())
    eff_years = set(var_arrays["eff_recharge"].keys())
    if total_years != eff_years:
        only_total = total_years - eff_years
        only_eff = eff_years - total_years
        raise RuntimeError(
            f"Mismatched years between TotalRecharge and EffRecharge. "
            f"Only in TotalRecharge: {sorted(only_total)}. "
            f"Only in EffRecharge: {sorted(only_eff)}."
        )

    if not total_years:
        raise RuntimeError(
            f"No GeoTIFF files found in {output_dir} for period {period}."
        )

    sorted_years = sorted(total_years)
    time_coords = pd.to_datetime([f"{y}-07-01" for y in sorted_years])

    # Stack into Dataset
    ds = xr.Dataset()
    for _, ds_name in var_configs:
        stacked = xr.concat(
            [var_arrays[ds_name][y] for y in sorted_years],
            dim="time",
        )
        stacked = stacked.assign_coords(time=time_coords)
        ds[ds_name] = stacked

    # Write atomically with compression
    encoding = {ds_name: {"zlib": True, "complevel": 4} for _, ds_name in var_configs}
    try:
        ds.to_netcdf(tmp_path, format="NETCDF4", encoding=encoding)
        tmp_path.rename(nc_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        ds.close()

    logger.info("Wrote consolidated file: %s", nc_path)
    return nc_path
