"""Normalization and range-bound construction methods."""

from __future__ import annotations


def normalize_0_1(da, dim: str = "time") -> object:
    """Normalize an xarray DataArray to [0, 1] over the given dimension."""
    # (x - min) / (max - min)
    raise NotImplementedError


def normalize_by_calendar_month(da) -> object:
    """Normalize per calendar month: each month normalized independently."""
    # For each month m in {1..12}:
    #   subset = da.sel(time=da.time.dt.month == m)
    #   normalized[m] = (subset - subset.min()) / (subset.max() - subset.min())
    raise NotImplementedError


def multi_source_minmax(datasets: list) -> tuple:
    """
    Compute lower/upper bounds as min/max across a list of DataArrays.

    Returns (lower_bound, upper_bound) DataArrays.
    """
    raise NotImplementedError


def modis_ci_bounds(sca, ci, ci_threshold: float = 0.70) -> tuple:
    """
    Construct SCA calibration bounds from MODIS confidence interval.

    Parameters
    ----------
    sca : DataArray of fractional snow cover (0-1)
    ci  : DataArray of confidence interval (0-1)
    ci_threshold : minimum CI to include a cell/day

    Returns
    -------
    (lower_bound, upper_bound) DataArrays
    NOTE: exact formula TBD — verify against PRMSobjfun.f
    """
    raise NotImplementedError
