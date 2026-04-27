"""Tests for MOD10C1 tier-2 aggregator (CI masking)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nhf_spatial_targets.aggregate.mod10c1 import build_masked_source


@pytest.fixture()
def raw_mod10c1():
    times = pd.date_range("2000-01-01", periods=1, freq="D")
    snow = np.array([[[50.0, 50.0], [50.0, 50.0]]])  # day, y, x
    ci = np.array([[[80.0, 60.0], [30.0, 100.0]]])  # CI in percent
    return xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["time", "lat", "lon"], snow),
            "Day_CMG_Clear_Index": (["time", "lat", "lon"], ci),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )


def test_build_masked_source_variables_present(raw_mod10c1):
    out = build_masked_source(raw_mod10c1)
    assert set(["sca", "ci", "valid_mask"]).issubset(out.data_vars)


def test_sca_is_nan_where_ci_below_threshold(raw_mod10c1):
    out = build_masked_source(raw_mod10c1)
    sca = out["sca"].isel(time=0).values
    # Cells with CI 80, 100 pass (>70); CI 60, 30 fail.
    assert np.isclose(sca[0, 0], 0.5)  # CI=80  -> keep, 50/100=0.5
    assert np.isnan(sca[0, 1])  # CI=60  -> drop
    assert np.isnan(sca[1, 0])  # CI=30  -> drop
    assert np.isclose(sca[1, 1], 0.5)  # CI=100 -> keep


def test_ci_passes_through_unmasked_below_threshold(raw_mod10c1):
    out = build_masked_source(raw_mod10c1)
    ci = out["ci"].isel(time=0).values
    # ci = Day_CMG_Clear_Index / 100; not gated on the 0.70 threshold,
    # so cells where SCA was masked still carry their fractional CI.
    np.testing.assert_allclose(ci, np.array([[0.8, 0.6], [0.3, 1.0]]))


def test_valid_mask_is_zero_one_float(raw_mod10c1):
    out = build_masked_source(raw_mod10c1)
    vm = out["valid_mask"].isel(time=0).values
    np.testing.assert_array_equal(vm, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert out["valid_mask"].dtype.kind == "f"


def test_build_masked_source_strict_threshold_at_0_70():
    """CI exactly at 0.70 must fail the filter (strict >, not >=)."""
    times = pd.date_range("2000-01-01", periods=1, freq="D")
    snow = np.array([[[50.0]]])
    ci = np.array([[[70.0]]])  # exactly 70% -> ci == 0.70 -> fail
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["time", "lat", "lon"], snow),
            "Day_CMG_Clear_Index": (["time", "lat", "lon"], ci),
        },
        coords={"time": times, "lat": [0.25], "lon": [0.5]},
    )
    out = build_masked_source(ds)
    assert np.isnan(out["sca"].isel(time=0).values[0, 0])
    assert out["valid_mask"].isel(time=0).values[0, 0] == 0.0


def test_build_masked_source_day_with_all_low_ci_yields_all_nan_sca():
    """If every cell fails the CI filter, sca is entirely NaN and valid_mask is 0."""
    times = pd.date_range("2000-01-01", periods=1, freq="D")
    snow = np.ones((1, 2, 2)) * 80.0
    ci = np.ones((1, 2, 2)) * 50.0  # CI=0.5 everywhere, all fail
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["time", "lat", "lon"], snow),
            "Day_CMG_Clear_Index": (["time", "lat", "lon"], ci),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )
    out = build_masked_source(ds)
    assert np.isnan(out["sca"].values).all()
    np.testing.assert_array_equal(out["valid_mask"].values, np.zeros((1, 2, 2)))


def test_build_masked_source_masks_snow_flag_values():
    """Day_CMG_Snow_Cover values >100 (107=lake, 237=water, 239=ocean, etc.)
    must be masked to NaN before scaling so they don't contaminate the
    weighted mean even when their CI happens to pass the filter.
    """
    times = pd.date_range("2000-01-01", periods=1, freq="D")
    # All cells have CI=100 (pass), but snow values are flag codes
    snow = np.array([[[50.0, 107.0], [239.0, 253.0]]])
    ci = np.ones((1, 2, 2)) * 100.0
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["time", "lat", "lon"], snow),
            "Day_CMG_Clear_Index": (["time", "lat", "lon"], ci),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )
    out = build_masked_source(ds)
    sca = out["sca"].isel(time=0).values
    assert np.isclose(sca[0, 0], 0.5)  # 50 -> 0.5, valid
    assert np.isnan(sca[0, 1])  # 107 (lake ice flag)
    assert np.isnan(sca[1, 0])  # 239 (ocean flag)
    assert np.isnan(sca[1, 1])  # 253 (not mapped flag)


def test_build_masked_source_masks_ci_flag_values():
    """Day_CMG_Clear_Index values >100 are flag codes; cells with such CI
    must fail the pass_mask, regardless of snow value.
    """
    times = pd.date_range("2000-01-01", periods=1, freq="D")
    snow = np.ones((1, 2, 2)) * 50.0
    ci = np.array([[[100.0, 239.0], [255.0, 80.0]]])  # 239 ocean, 255 fill
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["time", "lat", "lon"], snow),
            "Day_CMG_Clear_Index": (["time", "lat", "lon"], ci),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )
    out = build_masked_source(ds)
    sca = out["sca"].isel(time=0).values
    assert np.isclose(sca[0, 0], 0.5)  # CI=100 passes
    assert np.isnan(sca[0, 1])  # CI=239 -> NaN -> fails
    assert np.isnan(sca[1, 0])  # CI=255 -> NaN -> fails
    assert np.isclose(sca[1, 1], 0.5)  # CI=80 passes


def test_log_low_valid_coverage_warns_above_threshold(caplog):
    import logging

    from nhf_spatial_targets.aggregate.mod10c1 import _log_low_valid_coverage

    times = pd.date_range("2000-01-01", periods=10, freq="D")
    vaf_data = np.zeros((10, 10))
    vaf_data[0, 0] = 1.0  # 1 nonzero, 99 zero, 0 NaN -> 99% zero
    year_ds = xr.Dataset(
        {"valid_area_fraction": (["time", "hru_id"], vaf_data)},
        coords={"time": times, "hru_id": range(10)},
    )
    with caplog.at_level(
        logging.WARNING, logger="nhf_spatial_targets.aggregate.mod10c1"
    ):
        _log_low_valid_coverage(year_ds, year=2000)
    warnings = [rec.message for rec in caplog.records]
    assert any("zero valid-area" in m for m in warnings), warnings
    assert any("year=2000" in m for m in warnings), warnings


def test_log_low_valid_coverage_silent_below_threshold(caplog):
    import logging

    from nhf_spatial_targets.aggregate.mod10c1 import _log_low_valid_coverage

    times = pd.date_range("2000-01-01", periods=10, freq="D")
    # 99 nonzero, 1 zero -> 1% zero, below 10% threshold
    vaf_data = np.ones((10, 10)) * 0.8
    vaf_data[0, 0] = 0.0
    year_ds = xr.Dataset(
        {"valid_area_fraction": (["time", "hru_id"], vaf_data)},
        coords={"time": times, "hru_id": range(10)},
    )
    with caplog.at_level(
        logging.WARNING, logger="nhf_spatial_targets.aggregate.mod10c1"
    ):
        _log_low_valid_coverage(year_ds, year=2000)
    assert not any("zero valid-area" in rec.message for rec in caplog.records)


def test_mod10c1_adapter_wires_hooks_and_variables():
    from nhf_spatial_targets.aggregate.mod10c1 import ADAPTER, build_masked_source

    assert ADAPTER.source_key == "mod10c1_v061"
    assert ADAPTER.output_name == "mod10c1_agg.nc"
    assert set(ADAPTER.variables) == {"sca", "ci", "valid_mask"}
    assert ADAPTER.grid_variable == "sca"
    assert ADAPTER.pre_aggregate_hook is build_masked_source
    assert ADAPTER.post_aggregate_hook is not None
