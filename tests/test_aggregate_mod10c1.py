"""Tests for MOD10C1 tier-2 aggregator (CI masking).

The aggregator follows the A2 transformation policy
(``docs/architecture/transformation-pipeline.md``): native variable
names and units pass through, pixel-defined operations (flag masking,
CI gating) happen pre-aggregation, ``÷ 100`` and other linear rescales
live downstream in ``targets/`` / ``normalize/``.
"""

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
    ci = np.array([[[80.0, 60.0], [30.0, 100.0]]])  # CI in native 0–100 scale
    return xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["time", "lat", "lon"], snow),
            "Day_CMG_Clear_Index": (["time", "lat", "lon"], ci),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )


def test_build_masked_source_variables_present(raw_mod10c1):
    out = build_masked_source(raw_mod10c1)
    assert {"Day_CMG_Snow_Cover", "Day_CMG_Clear_Index", "valid_mask"}.issubset(
        out.data_vars
    )


def test_snow_cover_nan_where_ci_below_threshold(raw_mod10c1):
    """A2: pixel-level CI gate sets Day_CMG_Snow_Cover to NaN where CI ≤ 70.
    Native 0–100 scale is preserved (no ÷ 100 in the aggregator)."""
    out = build_masked_source(raw_mod10c1)
    snow = out["Day_CMG_Snow_Cover"].isel(time=0).values
    # Cells with CI 80, 100 pass (>70); CI 60, 30 fail.
    assert np.isclose(snow[0, 0], 50.0)  # CI=80  -> keep, native 50
    assert np.isnan(snow[0, 1])  # CI=60  -> drop
    assert np.isnan(snow[1, 0])  # CI=30  -> drop
    assert np.isclose(snow[1, 1], 50.0)  # CI=100 -> keep


def test_clear_index_passes_through_unmasked_below_threshold(raw_mod10c1):
    """A2: Day_CMG_Clear_Index is only flag-masked, not CI-gated.
    Cells where snow was masked still carry their native CI value."""
    out = build_masked_source(raw_mod10c1)
    ci = out["Day_CMG_Clear_Index"].isel(time=0).values
    np.testing.assert_allclose(ci, np.array([[80.0, 60.0], [30.0, 100.0]]))


def test_valid_mask_is_zero_one_float(raw_mod10c1):
    out = build_masked_source(raw_mod10c1)
    vm = out["valid_mask"].isel(time=0).values
    np.testing.assert_array_equal(vm, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert out["valid_mask"].dtype.kind == "f"


def test_build_masked_source_strict_threshold_at_70():
    """CI exactly at 70 must fail the filter (strict >, not >=)."""
    times = pd.date_range("2000-01-01", periods=1, freq="D")
    snow = np.array([[[50.0]]])
    ci = np.array([[[70.0]]])  # exactly 70 -> fail
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["time", "lat", "lon"], snow),
            "Day_CMG_Clear_Index": (["time", "lat", "lon"], ci),
        },
        coords={"time": times, "lat": [0.25], "lon": [0.5]},
    )
    out = build_masked_source(ds)
    assert np.isnan(out["Day_CMG_Snow_Cover"].isel(time=0).values[0, 0])
    assert out["valid_mask"].isel(time=0).values[0, 0] == 0.0


def test_build_masked_source_day_with_all_low_ci_yields_all_nan_snow_cover():
    """If every cell fails the CI filter, Day_CMG_Snow_Cover is entirely NaN
    and valid_mask is 0."""
    times = pd.date_range("2000-01-01", periods=1, freq="D")
    snow = np.ones((1, 2, 2)) * 80.0
    ci = np.ones((1, 2, 2)) * 50.0  # CI=50 everywhere, all fail
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["time", "lat", "lon"], snow),
            "Day_CMG_Clear_Index": (["time", "lat", "lon"], ci),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )
    out = build_masked_source(ds)
    assert np.isnan(out["Day_CMG_Snow_Cover"].values).all()
    np.testing.assert_array_equal(out["valid_mask"].values, np.zeros((1, 2, 2)))


def test_build_masked_source_masks_snow_flag_values():
    """Day_CMG_Snow_Cover values >100 (107=lake, 237=water, 239=ocean, etc.)
    must be masked to NaN before aggregation so they don't contaminate the
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
    snow = out["Day_CMG_Snow_Cover"].isel(time=0).values
    assert np.isclose(snow[0, 0], 50.0)  # 50 -> kept, native scale preserved
    assert np.isnan(snow[0, 1])  # 107 (lake ice flag)
    assert np.isnan(snow[1, 0])  # 239 (ocean flag)
    assert np.isnan(snow[1, 1])  # 253 (not mapped flag)


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
    snow = out["Day_CMG_Snow_Cover"].isel(time=0).values
    assert np.isclose(snow[0, 0], 50.0)  # CI=100 passes
    assert np.isnan(snow[0, 1])  # CI=239 -> flag -> NaN -> fails
    assert np.isnan(snow[1, 0])  # CI=255 -> flag -> NaN -> fails
    assert np.isclose(snow[1, 1], 50.0)  # CI=80 passes


def test_native_units_preserved_no_division_by_100():
    """A2 contract: aggregator must preserve the source's native 0–100
    integer scale. Any value at the input must round-trip unchanged through
    the pre-hook for cells that pass the filter (the ÷100 rescale is the
    target builder's responsibility)."""
    times = pd.date_range("2000-01-01", periods=1, freq="D")
    snow = np.array([[[1.0, 25.0, 73.0, 99.0]]])  # day, y, x — y has 1 row
    ci = np.ones((1, 1, 4)) * 100.0  # all pass
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["time", "lat", "lon"], snow),
            "Day_CMG_Clear_Index": (["time", "lat", "lon"], ci),
        },
        coords={"time": times, "lat": [0.25], "lon": [0.5, 1.5, 2.5, 3.5]},
    )
    out = build_masked_source(ds)
    np.testing.assert_allclose(
        out["Day_CMG_Snow_Cover"].isel(time=0).values[0],
        np.array([1.0, 25.0, 73.0, 99.0]),
    )
    np.testing.assert_allclose(
        out["Day_CMG_Clear_Index"].isel(time=0).values[0],
        np.array([100.0, 100.0, 100.0, 100.0]),
    )


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
    assert set(ADAPTER.variables) == {
        "Day_CMG_Snow_Cover",
        "Day_CMG_Clear_Index",
        "valid_mask",
    }
    assert ADAPTER.grid_variable == "Day_CMG_Snow_Cover"
    assert ADAPTER.raw_grid_variable == "Day_CMG_Snow_Cover"
    assert ADAPTER.pre_aggregate_hook is build_masked_source
    assert ADAPTER.post_aggregate_hook is not None
