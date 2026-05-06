"""Tests for shared multi-source-minmax target machinery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nhf_spatial_targets.workspace import load
from tests.conftest import make_minimal_project, write_year_nc


def test_read_aggregated_source_concats_per_year_nc(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    write_year_nc(src_dir / f"{src}_2000_agg.nc", 2000, var)
    write_year_nc(src_dir / f"{src}_2001_agg.nc", 2001, var)

    project = load(workdir)
    da = read_aggregated_source(
        project, src, var, period=("2000-01-01", "2001-12-31"), chunks={"time": 12}
    )
    assert da.dims == ("time", "nhm_id")
    assert len(da.time) == 24
    assert da.time.values[0] == np.datetime64("2000-01-01")
    assert da.time.values[-1] == np.datetime64("2001-12-01")


def test_read_aggregated_source_slices_to_period(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    for y in (1999, 2000, 2001, 2002):
        write_year_nc(src_dir / f"{src}_{y}_agg.nc", y, var)

    project = load(workdir)
    da = read_aggregated_source(
        project, src, var, period=("2000-06-01", "2001-06-30"), chunks={"time": 12}
    )
    # months 2000-06 .. 2001-06 inclusive -> 13 months
    assert len(da.time) == 13


def test_read_aggregated_source_raises_when_dir_empty(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    project = load(workdir)
    with pytest.raises(FileNotFoundError, match="No aggregated NC files found"):
        read_aggregated_source(
            project, "era5_land", "ro", period=("2000-01-01", "2001-12-31")
        )


def test_read_aggregated_source_raises_when_period_outside_coverage(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    write_year_nc(src_dir / f"{src}_2000_agg.nc", 2000, var)

    project = load(workdir)
    with pytest.raises(ValueError, match="entirely outside source coverage"):
        read_aggregated_source(project, src, var, period=("2010-01-01", "2010-12-31"))


def test_read_aggregated_source_raises_diagnostic_on_missing_var(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    src = "era5_land"
    src_dir = workdir / "data" / "aggregated" / src
    write_year_nc(src_dir / f"{src}_2000_agg.nc", 2000, "ro")

    project = load(workdir)
    with pytest.raises(KeyError, match="Available variables"):
        read_aggregated_source(
            project, src, "not_a_var", period=("2000-01-01", "2000-12-31")
        )


def _da_with_time(times, hrus=(1, 2, 3), values=None) -> xr.DataArray:
    if values is None:
        values = np.arange(len(times) * len(hrus), dtype=np.float32).reshape(
            len(times), len(hrus)
        )
    return xr.DataArray(
        values,
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(times), "nhm_id": list(hrus)},
    )


def test_reindex_to_month_start_maps_eom_to_ms():
    from nhf_spatial_targets.targets._common import reindex_to_month_start

    eom = _da_with_time(["2000-01-31", "2000-02-29", "2000-03-31"])
    master = pd.date_range("2000-01-01", "2000-03-01", freq="MS")
    reindexed = reindex_to_month_start(eom, master)
    assert list(reindexed.time.values) == list(master.values)
    np.testing.assert_array_equal(reindexed.values, eom.values)


def test_reindex_to_month_start_maps_mid_month_to_ms():
    from nhf_spatial_targets.targets._common import reindex_to_month_start

    mid = _da_with_time(["2000-01-15", "2000-02-15", "2000-03-15"])
    master = pd.date_range("2000-01-01", "2000-03-01", freq="MS")
    reindexed = reindex_to_month_start(mid, master)
    np.testing.assert_array_equal(reindexed.values, mid.values)


def test_reindex_to_month_start_pads_missing_months_with_nan():
    """Months in master_index but absent from the source come out as NaN."""
    from nhf_spatial_targets.targets._common import reindex_to_month_start

    partial = _da_with_time(["2000-01-01", "2000-02-01"])
    master = pd.date_range("2000-01-01", "2000-04-01", freq="MS")
    reindexed = reindex_to_month_start(partial, master)
    assert len(reindexed.time) == 4
    assert np.isnan(reindexed.values[2:]).all()
    np.testing.assert_array_equal(reindexed.values[:2], partial.values)


def test_reindex_to_month_start_already_ms_is_idempotent():
    from nhf_spatial_targets.targets._common import reindex_to_month_start

    ms = _da_with_time(["2000-01-01", "2000-02-01", "2000-03-01"])
    master = pd.date_range("2000-01-01", "2000-03-01", freq="MS")
    reindexed = reindex_to_month_start(ms, master)
    np.testing.assert_array_equal(reindexed.values, ms.values)


def test_multi_source_nanminmax_three_finite_sources():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(["2000-01-01"], values=np.array([[10, 20, 30]], dtype=np.float32))
    b = _da_with_time(["2000-01-01"], values=np.array([[15, 25, 35]], dtype=np.float32))
    c = _da_with_time(["2000-01-01"], values=np.array([[20, 30, 40]], dtype=np.float32))
    lower, upper, n = multi_source_nanminmax({"a": a, "b": b, "c": c})
    np.testing.assert_array_equal(lower.values, [[10, 20, 30]])
    np.testing.assert_array_equal(upper.values, [[20, 30, 40]])
    np.testing.assert_array_equal(n.values, [[3, 3, 3]])


def test_multi_source_nanminmax_partial_nan_uses_finite_only():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(
        ["2000-01-01"], values=np.array([[10.0, np.nan, 30.0]], dtype=np.float32)
    )
    b = _da_with_time(
        ["2000-01-01"], values=np.array([[15.0, 25.0, np.nan]], dtype=np.float32)
    )
    lower, upper, n = multi_source_nanminmax({"a": a, "b": b})
    np.testing.assert_array_equal(lower.values, [[10, 25, 30]])
    np.testing.assert_array_equal(upper.values, [[15, 25, 30]])
    np.testing.assert_array_equal(n.values, [[2, 1, 1]])


def test_multi_source_nanminmax_all_nan_returns_nan_and_zero():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(
        ["2000-01-01"], hrus=(1,), values=np.array([[np.nan]], dtype=np.float32)
    )
    b = _da_with_time(
        ["2000-01-01"], hrus=(1,), values=np.array([[np.nan]], dtype=np.float32)
    )
    lower, upper, n = multi_source_nanminmax({"a": a, "b": b})
    assert np.isnan(lower.values[0, 0])
    assert np.isnan(upper.values[0, 0])
    assert n.values[0, 0] == 0


def test_multi_source_nanminmax_n_sources_is_int8():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(["2000-01-01"], values=np.array([[10.0, 20.0, 30.0]], np.float32))
    _, _, n = multi_source_nanminmax({"a": a})
    assert n.dtype == np.int8


def test_multi_source_nanminmax_raises_on_hru_mismatch():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(["2000-01-01"], hrus=(1, 2, 3))
    b = _da_with_time(["2000-01-01"], hrus=(1, 2, 4))
    with pytest.raises(ValueError, match="HRU coords differ"):
        multi_source_nanminmax({"a": a, "b": b})
