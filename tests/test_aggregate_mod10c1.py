"""Tests for MOD10C1 tier-2 aggregator (CI masking)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nhf_spatial_targets.aggregate.mod10c1 import _open, build_masked_source


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


def test_build_masked_source_strict_threshold_at_0_70():
    """CI exactly at 0.70 must fail the filter (strict >, not >=)."""
    times = pd.date_range("2000-01-01", periods=1, freq="D")
    snow = np.array([[[50.0]]])
    qa = np.array([[[70.0]]])  # exactly 70% -> ci == 0.70 -> fail
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["time", "lat", "lon"], snow),
            "Snow_Spatial_QA": (["time", "lat", "lon"], qa),
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
    qa = np.ones((1, 2, 2)) * 50.0  # CI=0.5 everywhere, all fail
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["time", "lat", "lon"], snow),
            "Snow_Spatial_QA": (["time", "lat", "lon"], qa),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )
    out = build_masked_source(ds)
    assert np.isnan(out["sca"].values).all()
    np.testing.assert_array_equal(out["valid_mask"].values, np.zeros((1, 2, 2)))


def test_log_low_valid_coverage_warns_above_threshold(caplog):
    import logging

    from nhf_spatial_targets.aggregate.mod10c1 import _log_low_valid_coverage

    times = pd.date_range("2000-01-01", periods=10, freq="D")
    vaf_data = np.zeros((10, 10))
    vaf_data[0, 0] = 1.0  # 1 nonzero, 99 zero, 0 NaN -> 99% zero
    combined = xr.Dataset(
        {"valid_area_fraction": (["time", "hru_id"], vaf_data)},
        coords={"time": times, "hru_id": range(10)},
    )
    with caplog.at_level(
        logging.WARNING, logger="nhf_spatial_targets.aggregate.mod10c1"
    ):
        _log_low_valid_coverage(combined)
    assert any("zero valid-area" in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]


def test_log_low_valid_coverage_silent_below_threshold(caplog):
    import logging

    from nhf_spatial_targets.aggregate.mod10c1 import _log_low_valid_coverage

    times = pd.date_range("2000-01-01", periods=10, freq="D")
    # 99 nonzero, 1 zero -> 1% zero, below 10% threshold
    vaf_data = np.ones((10, 10)) * 0.8
    vaf_data[0, 0] = 0.0
    combined = xr.Dataset(
        {"valid_area_fraction": (["time", "hru_id"], vaf_data)},
        coords={"time": times, "hru_id": range(10)},
    )
    with caplog.at_level(
        logging.WARNING, logger="nhf_spatial_targets.aggregate.mod10c1"
    ):
        _log_low_valid_coverage(combined)
    assert not any("zero valid-area" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# _open: consolidated NC selection
# ---------------------------------------------------------------------------


def _make_consolidated_nc(path, times):
    """Write a minimal consolidated MOD10C1 NC with a time coordinate."""
    snow = np.ones((len(times), 2, 2)) * 50.0
    qa = np.ones((len(times), 2, 2)) * 80.0
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["time", "lat", "lon"], snow),
            "Snow_Spatial_QA": (["time", "lat", "lon"], qa),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )
    ds.to_netcdf(path)


def test_open_loads_single_consolidated_nc(tmp_path):
    """`_open` returns a Dataset with a 'time' coordinate from one consolidated file."""
    raw_dir = tmp_path / "mod10c1_v061"
    raw_dir.mkdir()
    times = pd.date_range("2000-03-01", periods=3, freq="D")
    _make_consolidated_nc(raw_dir / "mod10c1_v061_2000_consolidated.nc", times)
    # Also put a .conus.nc file in the dir to confirm it is ignored
    (raw_dir / "MOD10C1.A2000061.061.conus.nc").write_bytes(b"not a real nc")

    class _FakeProject:
        def raw_dir(self, key):
            return raw_dir

    result = _open(_FakeProject())
    assert "time" in result.coords
    assert len(result["time"]) == 3


def test_open_concatenates_multiple_consolidated_ncs(tmp_path):
    """`_open` merges multi-year consolidated files into one Dataset along time."""
    raw_dir = tmp_path / "mod10c1_v061"
    raw_dir.mkdir()
    times_2000 = pd.date_range("2000-01-01", periods=2, freq="D")
    times_2001 = pd.date_range("2001-01-01", periods=2, freq="D")
    _make_consolidated_nc(raw_dir / "mod10c1_v061_2000_consolidated.nc", times_2000)
    _make_consolidated_nc(raw_dir / "mod10c1_v061_2001_consolidated.nc", times_2001)

    class _FakeProject:
        def raw_dir(self, key):
            return raw_dir

    result = _open(_FakeProject())
    assert "time" in result.coords
    assert len(result["time"]) == 4
    # Verify time is sorted (2000 before 2001)
    assert result["time"].values[0] < result["time"].values[-1]
    # Both years must be represented (catches a regression that drops a file)
    years_present = set(pd.DatetimeIndex(result["time"].values).year)
    assert years_present == {2000, 2001}


def test_open_concat_sorts_when_filename_order_disagrees_with_time(tmp_path):
    """sortby('time') must run even when filename glob order is non-chronological."""
    raw_dir = tmp_path / "mod10c1_v061"
    raw_dir.mkdir()

    # partA covers 2001, partB covers 2000 — filename order is reversed
    times_partA = pd.date_range("2001-01-01", periods=2, freq="D")
    times_partB = pd.date_range("2000-01-01", periods=2, freq="D")
    _make_consolidated_nc(raw_dir / "mod10c1_v061_partA_consolidated.nc", times_partA)
    _make_consolidated_nc(raw_dir / "mod10c1_v061_partB_consolidated.nc", times_partB)

    class _FakeProject:
        def raw_dir(self, key):
            return raw_dir

    result = _open(_FakeProject())
    times = pd.DatetimeIndex(result["time"].values)
    assert list(times.year) == [2000, 2000, 2001, 2001]


def test_open_raises_when_no_consolidated_nc_found(tmp_path):
    """`_open` raises FileNotFoundError when only .conus.nc files exist."""
    raw_dir = tmp_path / "mod10c1_v061"
    raw_dir.mkdir()
    (raw_dir / "MOD10C1.A2000061.061.conus.nc").write_bytes(b"not a real nc")

    class _FakeProject:
        def raw_dir(self, key):
            return raw_dir

    with pytest.raises(FileNotFoundError, match="consolidated"):
        _open(_FakeProject())


def test_open_raises_on_duplicate_time_across_consolidated_ncs(tmp_path):
    """Overlapping-time mod10c1 consolidated NCs must raise."""
    raw_dir = tmp_path / "mod10c1_v061"
    raw_dir.mkdir()

    times_a = pd.date_range("2001-01-01", periods=10, freq="D")
    times_b = pd.date_range("2001-01-05", periods=10, freq="D")  # overlaps
    for name, times in (
        ("mod10c1_v061_2001a_consolidated.nc", times_a),
        ("mod10c1_v061_2001b_consolidated.nc", times_b),
    ):
        xr.Dataset(
            {
                "Day_CMG_Snow_Cover": (["time", "lat", "lon"], np.ones((10, 2, 2))),
                "Snow_Spatial_QA": (["time", "lat", "lon"], np.ones((10, 2, 2)) * 90),
            },
            coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
        ).to_netcdf(raw_dir / name)

    class _FakeProject:
        def raw_dir(self, key):
            return raw_dir

    with pytest.raises(ValueError, match="Duplicate"):
        _open(_FakeProject())
