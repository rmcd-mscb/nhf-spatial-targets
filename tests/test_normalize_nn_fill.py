"""Tests for HRU-space NN-fill of multi-source-minmax bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def _bounds_dataset(values_lower, values_upper, hrus, times):
    return xr.Dataset(
        {
            "lower_bound": (("time", "nhm_id"), np.asarray(values_lower, np.float32)),
            "upper_bound": (("time", "nhm_id"), np.asarray(values_upper, np.float32)),
        },
        coords={"time": pd.DatetimeIndex(times), "nhm_id": list(hrus)},
    )


def test_nn_fill_bounds_fills_isolated_nan_with_nearest_finite():
    from nhf_spatial_targets.normalize.methods import nn_fill_bounds

    # 3 HRUs in a row at x = 0, 1, 5; HRU 2 (at x=1) is NaN. Nearest finite
    # neighbor is HRU 1 (at x=0, distance 1) over HRU 3 (at x=5, distance 4).
    centroids_xy = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    ds = _bounds_dataset(
        values_lower=[[10.0, np.nan, 50.0]],
        values_upper=[[20.0, np.nan, 60.0]],
        hrus=[1, 2, 3],
        times=["2000-01-01"],
    )
    filled, diag = nn_fill_bounds(ds, centroids_xy, max_candidates=10)
    assert filled["lower_bound"].values[0, 1] == 10.0
    assert filled["upper_bound"].values[0, 1] == 20.0
    assert diag.values[0, 1] == 1
    assert diag.values[0, 0] == 0  # not filled


def test_nn_fill_bounds_walks_when_nearest_is_also_nan():
    from nhf_spatial_targets.normalize.methods import nn_fill_bounds

    # HRU 2 is NaN. Nearest is HRU 1 at distance 1, but HRU 1 is also NaN.
    # Walk to HRU 3 at distance 4.
    centroids_xy = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    ds = _bounds_dataset(
        values_lower=[[np.nan, np.nan, 50.0]],
        values_upper=[[np.nan, np.nan, 60.0]],
        hrus=[1, 2, 3],
        times=["2000-01-01"],
    )
    filled, diag = nn_fill_bounds(ds, centroids_xy, max_candidates=10)
    assert filled["lower_bound"].values[0, 1] == 50.0
    assert filled["upper_bound"].values[0, 1] == 60.0
    assert diag.values[0, 1] == 1


def test_nn_fill_bounds_max_candidates_cap_leaves_nan():
    from nhf_spatial_targets.normalize.methods import nn_fill_bounds

    # All NaN at this time step -> no donor will ever be finite; cell stays NaN.
    centroids_xy = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    ds = _bounds_dataset(
        values_lower=[[np.nan, np.nan, np.nan]],
        values_upper=[[np.nan, np.nan, np.nan]],
        hrus=[1, 2, 3],
        times=["2000-01-01"],
    )
    filled, diag = nn_fill_bounds(ds, centroids_xy, max_candidates=2)
    assert np.isnan(filled["lower_bound"].values).all()
    assert (diag.values == 0).all()


def test_nn_fill_bounds_per_time_step_independence():
    from nhf_spatial_targets.normalize.methods import nn_fill_bounds

    # HRU 2 is NaN at t=0 (filled from HRU 1 = 10.0) but finite at t=1 (untouched).
    centroids_xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    ds = _bounds_dataset(
        values_lower=[[10.0, np.nan], [99.0, 88.0]],
        values_upper=[[20.0, np.nan], [199.0, 188.0]],
        hrus=[1, 2],
        times=["2000-01-01", "2000-02-01"],
    )
    filled, diag = nn_fill_bounds(ds, centroids_xy, max_candidates=10)
    assert filled["lower_bound"].values[0, 1] == 10.0
    assert filled["lower_bound"].values[1, 1] == 88.0
    assert diag.values[0, 1] == 1
    assert diag.values[1, 1] == 0


def test_nn_fill_bounds_preserves_finite_cells():
    from nhf_spatial_targets.normalize.methods import nn_fill_bounds

    centroids_xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    ds = _bounds_dataset(
        values_lower=[[10.0, 20.0]],
        values_upper=[[100.0, 200.0]],
        hrus=[1, 2],
        times=["2000-01-01"],
    )
    filled, diag = nn_fill_bounds(ds, centroids_xy, max_candidates=10)
    np.testing.assert_array_equal(
        filled["lower_bound"].values, ds["lower_bound"].values
    )
    np.testing.assert_array_equal(
        filled["upper_bound"].values, ds["upper_bound"].values
    )
    assert (diag.values == 0).all()


def test_nn_fill_bounds_does_not_fill_asymmetric_nan():
    """If only one bound is NaN at a cell, leave both alone (preserves the
    valid finite bound). Per spec: fill only when *both* are NaN."""
    from nhf_spatial_targets.normalize.methods import nn_fill_bounds

    centroids_xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    ds = _bounds_dataset(
        values_lower=[[10.0, np.nan]],  # HRU 2 lower is NaN
        values_upper=[[20.0, 30.0]],  # HRU 2 upper is finite
        hrus=[1, 2],
        times=["2000-01-01"],
    )
    filled, diag = nn_fill_bounds(ds, centroids_xy, max_candidates=10)
    # HRU 2's bounds must be left unchanged because upper is finite:
    assert np.isnan(filled["lower_bound"].values[0, 1])
    assert filled["upper_bound"].values[0, 1] == 30.0
    assert diag.values[0, 1] == 0  # not filled
