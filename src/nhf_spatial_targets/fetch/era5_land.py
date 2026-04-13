"""Fetch ERA5-Land hourly runoff from Copernicus CDS.

Downloads hourly accumulated runoff variables (ro, sro, ssro) for the
CONUS+contributing-watersheds bbox, then aggregates hourly→daily and
daily→monthly. Both daily and monthly consolidated NetCDFs are written
to the shared datastore.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# CDS area parameter [N, W, S, E], snapped to ERA5-Land 0.1° grid.
# Encompasses CONUS contributing watersheds (Canada/Mexico) + ~10 km buffer.
BBOX_NWSE = [53.0, -125.0, 24.7, -66.0]

VARIABLES = ("ro", "sro", "ssro")


def hourly_to_daily(da: xr.DataArray) -> xr.DataArray:
    """Aggregate ERA5-Land hourly accumulated runoff to daily totals.

    ERA5-Land accumulated fields (ro, sro, ssro) reset at 00 UTC each day
    and represent meters of water equivalent accumulated since 00 UTC.
    Daily total = sum of hourly increments computed via .diff('time'),
    then resampled to daily sums. The diff approach is robust to the
    midnight-reset boundary and to missing hours within a day.

    The first hourly step of each day, after the 00 UTC reset, equals the
    accumulation over the 23->00 hour of the prior day. We discard the
    pre-first-midnight increment (which is meaningless without a prior
    23 UTC value) by requiring the result to come from `.diff` with
    matching valid timestamps.

    Parameters
    ----------
    da : xr.DataArray
        Hourly accumulated runoff with a 'time' dimension. Time stamps
        must be regular hourly.

    Returns
    -------
    xr.DataArray
        Daily-summed runoff. Time coordinate is the date (00:00) of each
        complete day. Original attrs are preserved.
    """
    incr = da.diff("time", label="upper")
    # Negative jumps occur at the 00 UTC reset; where the diff is negative,
    # replace it with the raw value at that timestamp (which equals the
    # accumulation since the midnight reset, i.e. the true hourly increment).
    incr = xr.where(incr >= 0, incr, da.isel(time=slice(1, None)))
    # Shift the timestamp back by 1 hour so that the 00 UTC increment
    # (which is the 23->00 accumulation) lands inside the prior day.
    incr = incr.assign_coords(time=incr.time - pd.Timedelta(hours=1))
    daily = incr.resample(time="1D").sum()
    daily.attrs = dict(da.attrs)
    return daily


def daily_to_monthly(da: xr.DataArray) -> xr.DataArray:
    """Sum daily totals to monthly totals.

    Uses month-end frequency ('1ME') so the time coordinate marks the
    last day of each month — consistent with other monthly products in
    this codebase. Original attrs are preserved.
    """
    monthly = da.resample(time="1ME").sum()
    monthly.attrs = dict(da.attrs)
    return monthly


# Map short variable name to the CDS request name
_VARIABLE_REQUEST_NAME = {
    "ro": "runoff",
    "sro": "surface_runoff",
    "ssro": "sub_surface_runoff",
}


def _cds_client():
    """Construct a cdsapi.Client. Separated for test injection."""
    import cdsapi

    return cdsapi.Client()


def download_year_variable(
    year: int,
    variable: str,
    output_path: Path,
) -> Path:
    """Download one year of one ERA5-Land variable to ``output_path``.

    Idempotent: if ``output_path`` already exists, returns immediately.
    Submits a single CDS request covering all 12 months × all hours of
    the given year, clipped to ``BBOX_NWSE``.

    Parameters
    ----------
    year : int
    variable : {"ro", "sro", "ssro"}
    output_path : Path
        Target NetCDF file. Parent directory is created if missing.

    Returns
    -------
    Path
        ``output_path`` (for caller convenience).
    """
    if variable not in _VARIABLE_REQUEST_NAME:
        raise ValueError(f"Unknown ERA5-Land variable: {variable!r}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.info("Skipping existing ERA5-Land file: %s", output_path)
        return output_path

    request = {
        "variable": _VARIABLE_REQUEST_NAME[variable],
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": BBOX_NWSE,
        "format": "netcdf",
    }
    client = _cds_client()
    logger.info("Submitting CDS request for %s %d → %s", variable, year, output_path)
    client.retrieve("reanalysis-era5-land", request, str(output_path))
    return output_path
