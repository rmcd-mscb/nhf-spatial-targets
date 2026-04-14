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
    qa = np.array([[[80.0, 60.0], [30.0, 100.0]]])  # CI in percent
    return xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["time", "lat", "lon"], snow),
            "Snow_Spatial_QA": (["time", "lat", "lon"], qa),
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


def test_ci_passes_through_unmasked(raw_mod10c1):
    out = build_masked_source(raw_mod10c1)
    ci = out["ci"].isel(time=0).values
    # ci is raw QA / 100 — no NaNs even where SCA was masked
    np.testing.assert_allclose(ci, np.array([[0.8, 0.6], [0.3, 1.0]]))


def test_valid_mask_is_zero_one_float(raw_mod10c1):
    out = build_masked_source(raw_mod10c1)
    vm = out["valid_mask"].isel(time=0).values
    np.testing.assert_array_equal(vm, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert out["valid_mask"].dtype.kind == "f"
