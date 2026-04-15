"""Unit tests for per-year streaming aggregation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _write_nc(path: Path, times: pd.DatetimeIndex) -> None:
    xr.Dataset(
        {"v": (["time", "lat", "lon"], np.ones((len(times), 2, 2)))},
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "lat": ("lat", [0.0, 1.0], {"standard_name": "latitude"}),
            "lon": ("lon", [0.0, 1.0], {"standard_name": "longitude"}),
        },
    ).to_netcdf(path)


def test_enumerate_years_per_year_files(tmp_path):
    from nhf_spatial_targets.aggregate._driver import enumerate_years

    f2000 = tmp_path / "src_2000_consolidated.nc"
    f2001 = tmp_path / "src_2001_consolidated.nc"
    _write_nc(f2000, pd.date_range("2000-01-01", periods=12, freq="MS"))
    _write_nc(f2001, pd.date_range("2001-01-01", periods=12, freq="MS"))

    year_files = enumerate_years([f2000, f2001])
    assert year_files == [(2000, f2000), (2001, f2001)]


def test_enumerate_years_single_multi_year_file(tmp_path):
    from nhf_spatial_targets.aggregate._driver import enumerate_years

    f = tmp_path / "src_consolidated.nc"
    _write_nc(f, pd.date_range("2000-01-01", periods=36, freq="MS"))

    year_files = enumerate_years([f])
    assert year_files == [(2000, f), (2001, f), (2002, f)]


def test_enumerate_years_raises_on_year_overlap(tmp_path):
    from nhf_spatial_targets.aggregate._driver import enumerate_years

    fa = tmp_path / "src_a_consolidated.nc"
    fb = tmp_path / "src_b_consolidated.nc"
    _write_nc(fa, pd.date_range("2001-01-01", periods=12, freq="MS"))
    _write_nc(fb, pd.date_range("2001-06-01", periods=12, freq="MS"))

    with pytest.raises(ValueError, match="overlap"):
        enumerate_years([fa, fb])


def test_enumerate_years_raises_when_no_time_coord(tmp_path):
    from nhf_spatial_targets.aggregate._driver import enumerate_years

    f = tmp_path / "no_time_consolidated.nc"
    xr.Dataset({"v": (["y", "x"], np.zeros((2, 2)))}).to_netcdf(f)
    with pytest.raises(ValueError, match="time"):
        enumerate_years([f])
