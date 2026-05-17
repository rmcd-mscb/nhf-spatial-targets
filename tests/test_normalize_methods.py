"""Tests for the per-HRU normalization primitives in normalize.methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _da(times, values, dims=("time", "nhm_id"), hrus=(1, 2, 3)) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype=np.float32),
        dims=dims,
        coords={"time": pd.DatetimeIndex(times), "nhm_id": list(hrus)},
        attrs={"units": "mm"},
    )


# ---------------------------------------------------------------------------
# normalize_0_1
# ---------------------------------------------------------------------------


def test_normalize_0_1_simple_range():
    """Linear input on [0, 10] over 3 times normalizes to [0.0, 0.5, 1.0]."""
    from nhf_spatial_targets.normalize.methods import normalize_0_1

    da = _da(
        ["2000-01-01", "2000-02-01", "2000-03-01"],
        [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [10.0, 10.0, 10.0]],
    )
    out = normalize_0_1(da, dim="time")
    np.testing.assert_allclose(
        out.values, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], rtol=1e-6
    )
    assert out.attrs["units"] == "1"


def test_normalize_0_1_per_hru_min_max():
    """Each HRU is normalized independently using its own min/max."""
    from nhf_spatial_targets.normalize.methods import normalize_0_1

    # HRU 1 swings 0..100, HRU 2 swings 10..30, HRU 3 swings 5..5 (constant).
    da = _da(
        ["2000-01-01", "2000-02-01", "2000-03-01"],
        [[0.0, 10.0, 5.0], [50.0, 20.0, 5.0], [100.0, 30.0, 5.0]],
    )
    out = normalize_0_1(da, dim="time")
    # HRU 1: (0,50,100) -> (0,0.5,1)
    np.testing.assert_allclose(out.values[:, 0], [0.0, 0.5, 1.0], rtol=1e-6)
    # HRU 2: (10,20,30) -> (0,0.5,1)
    np.testing.assert_allclose(out.values[:, 1], [0.0, 0.5, 1.0], rtol=1e-6)
    # HRU 3: constant series -> all NaN (zero range)
    assert np.isnan(out.values[:, 2]).all()


def test_normalize_0_1_handles_nan_skipna_for_minmax():
    """NaNs along the dim are ignored when computing min/max."""
    from nhf_spatial_targets.normalize.methods import normalize_0_1

    da = _da(
        ["2000-01-01", "2000-02-01", "2000-03-01", "2000-04-01"],
        [
            [10.0, 0.0, 0.0],
            [np.nan, 5.0, 0.0],
            [20.0, np.nan, 0.0],
            [30.0, 10.0, 0.0],
        ],
    )
    out = normalize_0_1(da, dim="time")
    # HRU 1: min=10, max=30 (NaN ignored) -> 10->0, NaN->NaN, 20->0.5, 30->1
    np.testing.assert_allclose(out.values[0, 0], 0.0, rtol=1e-6)
    assert np.isnan(out.values[1, 0])
    np.testing.assert_allclose(out.values[2, 0], 0.5, rtol=1e-6)
    np.testing.assert_allclose(out.values[3, 0], 1.0, rtol=1e-6)
    # HRU 3: constant 0 -> all NaN (zero range)
    assert np.isnan(out.values[:, 2]).all()


def test_normalize_0_1_all_nan_yields_all_nan():
    """All-NaN cell stays NaN (range is NaN, propagates)."""
    from nhf_spatial_targets.normalize.methods import normalize_0_1

    da = _da(
        ["2000-01-01", "2000-02-01"],
        [[np.nan, 1.0], [np.nan, 2.0]],
        hrus=(1, 2),
    )
    out = normalize_0_1(da, dim="time")
    assert np.isnan(out.values[:, 0]).all()
    # HRU 2 normalizes as usual
    np.testing.assert_allclose(out.values[:, 1], [0.0, 1.0], rtol=1e-6)


def test_normalize_0_1_raises_on_unknown_dim():
    from nhf_spatial_targets.normalize.methods import normalize_0_1

    da = _da(["2000-01-01"], [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="not in DataArray dims"):
        normalize_0_1(da, dim="not_a_dim")


def test_normalize_0_1_preserves_non_units_attrs():
    """Custom attrs (e.g. long_name) survive; only `units` is replaced."""
    from nhf_spatial_targets.normalize.methods import normalize_0_1

    da = _da(["2000-01-01", "2000-02-01"], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    da.attrs["long_name"] = "monthly recharge"
    da.attrs["source"] = "synthetic"
    out = normalize_0_1(da, dim="time")
    assert out.attrs["long_name"] == "monthly recharge"
    assert out.attrs["source"] == "synthetic"
    assert out.attrs["units"] == "1"


# ---------------------------------------------------------------------------
# normalize_0_1_by_calendar_month
# ---------------------------------------------------------------------------


def test_normalize_0_1_by_calendar_month_independence_across_months():
    """Each calendar month is normalized with its own min/max.

    Two Januaries (year A: 0, year B: 10) normalize to [0, 1]. Two Julys
    with the same swing (200 to 300) also normalize to [0, 1] — the Jan
    min of 0 must NOT pull Jul's normalization toward zero.
    """
    from nhf_spatial_targets.normalize.methods import normalize_0_1_by_calendar_month

    times = pd.to_datetime(["2000-01-01", "2000-07-01", "2001-01-01", "2001-07-01"])
    # Values per (year, month) pair for HRU 1:
    #   2000-01: 0     2000-07: 200
    #   2001-01: 10    2001-07: 300
    da = xr.DataArray(
        np.array([[0.0], [200.0], [10.0], [300.0]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
        attrs={"units": "mm"},
    )
    out = normalize_0_1_by_calendar_month(da)
    # 2000-01 norm-jan: (0-0)/(10-0) = 0
    np.testing.assert_allclose(out.sel(time="2000-01-01").values, [0.0], rtol=1e-6)
    # 2001-01 norm-jan: (10-0)/(10-0) = 1
    np.testing.assert_allclose(out.sel(time="2001-01-01").values, [1.0], rtol=1e-6)
    # 2000-07 norm-jul: (200-200)/(300-200) = 0
    np.testing.assert_allclose(out.sel(time="2000-07-01").values, [0.0], rtol=1e-6)
    # 2001-07 norm-jul: (300-200)/(300-200) = 1
    np.testing.assert_allclose(out.sel(time="2001-07-01").values, [1.0], rtol=1e-6)
    assert out.attrs["units"] == "1"


def test_normalize_0_1_by_calendar_month_preserves_input_order():
    """Output is on the original (monotonic) time index, not regrouped."""
    from nhf_spatial_targets.normalize.methods import normalize_0_1_by_calendar_month

    times = pd.date_range("2000-01-01", "2001-12-01", freq="MS")
    values = np.arange(len(times) * 1, dtype=np.float32).reshape(-1, 1)
    da = xr.DataArray(
        values,
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
        attrs={"units": "mm"},
    )
    out = normalize_0_1_by_calendar_month(da)
    assert list(out.time.values) == list(times.values)
    assert out.shape == da.shape


def test_normalize_0_1_by_calendar_month_constant_month_yields_nan():
    """A calendar month with constant value across years -> NaN (zero range)."""
    from nhf_spatial_targets.normalize.methods import normalize_0_1_by_calendar_month

    times = pd.to_datetime(
        ["2000-01-01", "2001-01-01", "2002-01-01", "2000-07-01", "2001-07-01"]
    )
    # All Januaries equal 5.0; Julys swing 100..200
    values = np.array([[5.0], [5.0], [5.0], [100.0], [200.0]], dtype=np.float32)
    da = xr.DataArray(
        values,
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
        attrs={"units": "mm"},
    )
    out = normalize_0_1_by_calendar_month(da)
    # Three constant Januaries -> NaN
    jans = out.sel(time=out.time.dt.month == 1).values
    assert np.isnan(jans).all()
    # Two-element July normalization -> 0, 1
    juls = out.sel(time=out.time.dt.month == 7).values
    np.testing.assert_allclose(juls.flatten(), [0.0, 1.0], rtol=1e-6)


def test_normalize_0_1_by_calendar_month_raises_on_missing_time_dim():
    from nhf_spatial_targets.normalize.methods import normalize_0_1_by_calendar_month

    da = xr.DataArray(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        dims=("nhm_id",),
        coords={"nhm_id": [1, 2, 3]},
    )
    with pytest.raises(ValueError, match="expected 'time' dim"):
        normalize_0_1_by_calendar_month(da)


# ---------------------------------------------------------------------------
# normalize_0_1_over_window
# ---------------------------------------------------------------------------


def test_normalize_0_1_over_window_uses_window_minmax():
    """Window's min/max drive the normalization; values outside may exceed [0, 1]."""
    from nhf_spatial_targets.normalize.methods import normalize_0_1_over_window

    # da spans 5 years; window is years 2000-2002 (min=10, max=30).
    # Year 2003 = 50 should normalize to (50-10)/(30-10) = 2.0 (visibly out of range).
    times = pd.date_range("2000-01-01", "2004-01-01", freq="YS")
    da = xr.DataArray(
        np.array([[10.0], [20.0], [30.0], [50.0], [80.0]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
        attrs={"units": "mm"},
    )
    window = da.sel(time=slice("2000-01-01", "2002-12-31"))
    out = normalize_0_1_over_window(da, window)
    np.testing.assert_allclose(
        out.values, [[0.0], [0.5], [1.0], [2.0], [3.5]], rtol=1e-6
    )
    assert out.attrs["units"] == "1"


def test_normalize_0_1_over_window_equals_normalize_0_1_when_window_is_da():
    """When window == da, the window variant matches the no-window primitive."""
    from nhf_spatial_targets.normalize.methods import (
        normalize_0_1,
        normalize_0_1_over_window,
    )

    times = pd.date_range("2000-01-01", "2004-01-01", freq="YS")
    da = xr.DataArray(
        np.array([[0.0, 5.0], [10.0, 15.0], [20.0, 25.0], [30.0, 35.0], [40.0, 45.0]]),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1, 2]},
        attrs={"units": "mm"},
    )
    np.testing.assert_allclose(
        normalize_0_1_over_window(da, da).values,
        normalize_0_1(da, dim="time").values,
        rtol=1e-6,
    )


def test_normalize_0_1_over_window_zero_range_yields_nan():
    """Constant series over window → NaN (zero range)."""
    from nhf_spatial_targets.normalize.methods import normalize_0_1_over_window

    times = pd.date_range("2000-01-01", "2003-01-01", freq="YS")
    da = xr.DataArray(
        np.array([[5.0], [5.0], [5.0], [10.0]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
    )
    window = da.sel(time=slice("2000-01-01", "2002-12-31"))
    out = normalize_0_1_over_window(da, window)
    assert np.isnan(out.values).all()


def test_normalize_0_1_over_window_raises_on_unknown_dim():
    from nhf_spatial_targets.normalize.methods import normalize_0_1_over_window

    da = xr.DataArray(
        np.array([1.0, 2.0]),
        dims=("nhm_id",),
        coords={"nhm_id": [1, 2]},
    )
    with pytest.raises(ValueError, match="not in DataArray dims"):
        normalize_0_1_over_window(da, da, dim="time")


# ---------------------------------------------------------------------------
# normalize_0_1_by_calendar_month_over_window
# ---------------------------------------------------------------------------


def test_normalize_0_1_by_calendar_month_over_window_uses_window_per_month():
    """Each calendar month is normalized using only that month's window min/max."""
    from nhf_spatial_targets.normalize.methods import (
        normalize_0_1_by_calendar_month_over_window,
    )

    # 4 years × 2 calendar months. Window = years 2000-2001 only.
    #   Jan window: 0, 10 → min=0, max=10
    #   Jul window: 100, 200 → min=100, max=200
    # da's 2002-2003 values are normalized using the window's min/max.
    times = pd.to_datetime(
        [
            "2000-01-01",
            "2000-07-01",
            "2001-01-01",
            "2001-07-01",
            "2002-01-01",
            "2002-07-01",
            "2003-01-01",
            "2003-07-01",
        ]
    )
    values = np.array(
        [[0.0], [100.0], [10.0], [200.0], [20.0], [300.0], [5.0], [150.0]],
        dtype=np.float32,
    )
    da = xr.DataArray(
        values,
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
        attrs={"units": "mm"},
    )
    window = da.sel(time=slice("2000-01-01", "2001-12-31"))
    out = normalize_0_1_by_calendar_month_over_window(da, window)
    # Jan: window min=0, max=10
    #   2000-01: (0-0)/(10-0) = 0
    #   2001-01: (10-0)/(10-0) = 1
    #   2002-01: (20-0)/(10-0) = 2.0 (above window max — by design)
    #   2003-01: (5-0)/(10-0) = 0.5
    # Jul: window min=100, max=200
    #   2000-07: (100-100)/(200-100) = 0
    #   2001-07: (200-100)/(200-100) = 1
    #   2002-07: (300-100)/(200-100) = 2.0
    #   2003-07: (150-100)/(200-100) = 0.5
    np.testing.assert_allclose(
        out.values,
        [[0.0], [0.0], [1.0], [1.0], [2.0], [2.0], [0.5], [0.5]],
        rtol=1e-6,
    )
    assert out.attrs["units"] == "1"


def test_normalize_0_1_by_calendar_month_over_window_equals_no_window():
    """When window == da, the variant matches the no-window primitive."""
    from nhf_spatial_targets.normalize.methods import (
        normalize_0_1_by_calendar_month,
        normalize_0_1_by_calendar_month_over_window,
    )

    times = pd.to_datetime(["2000-01-01", "2000-07-01", "2001-01-01", "2001-07-01"])
    da = xr.DataArray(
        np.array([[0.0], [100.0], [10.0], [200.0]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
    )
    out_with = normalize_0_1_by_calendar_month_over_window(da, da)
    out_no = normalize_0_1_by_calendar_month(da)
    np.testing.assert_allclose(out_with.values, out_no.values, rtol=1e-6)


def test_normalize_0_1_by_calendar_month_over_window_raises_on_missing_time():
    from nhf_spatial_targets.normalize.methods import (
        normalize_0_1_by_calendar_month_over_window,
    )

    da = xr.DataArray(
        np.array([1.0, 2.0]),
        dims=("nhm_id",),
        coords={"nhm_id": [1, 2]},
    )
    with pytest.raises(ValueError, match="expected 'time' dim"):
        normalize_0_1_by_calendar_month_over_window(da, da)


def test_normalize_0_1_by_calendar_month_over_window_raises_when_window_lacks_time():
    from nhf_spatial_targets.normalize.methods import (
        normalize_0_1_by_calendar_month_over_window,
    )

    times = pd.to_datetime(["2000-01-01", "2000-07-01"])
    da = xr.DataArray(
        np.array([[1.0], [2.0]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
    )
    bad_window = xr.DataArray(np.array([1.0]), dims=("nhm_id",), coords={"nhm_id": [1]})
    with pytest.raises(ValueError, match="window must also have 'time' dim"):
        normalize_0_1_by_calendar_month_over_window(da, bad_window)
