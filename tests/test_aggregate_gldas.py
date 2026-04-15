"""Tests for GLDAS aggregation adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def test_adapter_declares_runoff_vars():
    from nhf_spatial_targets.aggregate.gldas import ADAPTER

    assert ADAPTER.source_key == "gldas_noah_v21_monthly"
    assert ADAPTER.output_name == "gldas_agg.nc"
    assert set(ADAPTER.variables) == {"Qs_acc", "Qsb_acc", "runoff_total"}


def test_derive_runoff_total():
    from nhf_spatial_targets.aggregate.gldas import _derive_runoff_total

    ds = xr.Dataset(
        {
            "Qs_acc": (["time"], np.array([1.0, 2.0])),
            "Qsb_acc": (["time"], np.array([3.0, 4.0])),
        },
        coords={"time": pd.date_range("2000-01-01", periods=2, freq="MS")},
    )
    out = _derive_runoff_total(ds)
    np.testing.assert_array_equal(out["runoff_total"].values, np.array([4.0, 6.0]))
    assert out["runoff_total"].attrs["derived_from"] == "Qs_acc + Qsb_acc"


def test_derive_runoff_total_preserves_nan():
    """NaN in either component propagates to runoff_total."""
    from nhf_spatial_targets.aggregate.gldas import _derive_runoff_total

    ds = xr.Dataset(
        {
            "Qs_acc": (["time", "lat", "lon"], np.array([[[1.0, np.nan], [1.0, 1.0]]])),
            "Qsb_acc": (
                ["time", "lat", "lon"],
                np.array([[[2.0, 2.0], [np.nan, 2.0]]]),
            ),
        },
        coords={
            "time": pd.date_range("2000-01-01", periods=1, freq="MS"),
            "lat": [0.25, 0.75],
            "lon": [0.5, 1.5],
        },
    )
    out = _derive_runoff_total(ds)
    rt = out["runoff_total"].isel(time=0).values
    assert np.isnan(rt[0, 1])  # Qs NaN -> total NaN
    assert np.isnan(rt[1, 0])  # Qsb NaN -> total NaN
    assert rt[0, 0] == 3.0
    assert rt[1, 1] == 3.0


def test_adapter_sets_files_glob_and_pre_hook():
    from nhf_spatial_targets.aggregate.gldas import ADAPTER, _derive_runoff_total

    assert ADAPTER.files_glob == "gldas_noah_v21_monthly*.nc"
    assert ADAPTER.pre_aggregate_hook is _derive_runoff_total
