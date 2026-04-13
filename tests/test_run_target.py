from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nhf_spatial_targets.targets.run import (
    _M3_PER_DAY_TO_CFS,
    _validate_alignment,
    era5_to_mm_per_month,
    gldas_to_mm_per_month,
    mm_per_month_to_cfs,
    multi_source_runoff_bounds,
)

HRU_AREA_M2 = 1.0e8  # 100 km²


def _series(values, units):
    times = pd.date_range("2020-01-01", periods=len(values), freq="1MS")
    da = xr.DataArray(values, dims=("time",), coords={"time": times})
    da.attrs["units"] = units
    return da


def test_era5_meters_to_mm():
    da = _series([0.05, 0.10], "m")
    out = era5_to_mm_per_month(da)
    np.testing.assert_allclose(out.values, [50.0, 100.0])
    assert out.attrs["units"] == "mm"


def test_gldas_kgm2_passthrough_to_mm():
    da = _series([20.0, 40.0], "kg m-2")
    out = gldas_to_mm_per_month(da)
    np.testing.assert_allclose(out.values, [20.0, 40.0])
    assert out.attrs["units"] == "mm"


def test_mm_to_cfs_uses_days_in_month():
    # 31 mm in January = 0.001 m/day over 100 km² = 100 m³/day = ~0.0409 cfs
    da = _series([31.0], "mm")
    out = mm_per_month_to_cfs(da, hru_area_m2=HRU_AREA_M2)
    expected_m3_per_day = 0.001 * HRU_AREA_M2  # 100 000 m³/day
    expected_cfs = expected_m3_per_day / 86400.0 * 35.3147
    np.testing.assert_allclose(out.values, [expected_cfs], rtol=1e-4)
    assert out.attrs["units"] == "cfs"


def test_multi_source_minmax():
    a = _series([10.0, 20.0, 30.0], "cfs")
    b = _series([15.0, 18.0, 32.0], "cfs")
    lower, upper = multi_source_runoff_bounds([a, b])
    np.testing.assert_allclose(lower.values, [10.0, 18.0, 30.0])
    np.testing.assert_allclose(upper.values, [15.0, 20.0, 32.0])


def test_build_writes_lower_upper_bounds(tmp_path):
    from nhf_spatial_targets.targets.run import build

    times = pd.date_range("2020-01-01", periods=3, freq="1MS")
    hrus = np.arange(3)

    def _save(path: Path, var: str, vals_m_or_kg, units: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        da = xr.DataArray(
            vals_m_or_kg,
            dims=("time", "hru"),
            coords={"time": times, "hru": hrus},
            name=var,
        )
        da.attrs["units"] = units
        ds = da.to_dataset()
        ds.to_netcdf(path)

    agg = tmp_path / "data" / "aggregated"
    _save(agg / "era5_land" / "ro.nc", "ro", np.full((3, 3), 0.05), "m")
    _save(
        agg / "gldas_noah_v21_monthly" / "runoff_total.nc",
        "runoff_total",
        np.full((3, 3), 30.0),
        "kg m-2",
    )

    out = tmp_path / "targets" / "runoff_target.nc"
    hru_area = xr.DataArray(np.full(3, 1.0e8), dims=("hru",), coords={"hru": hrus})

    build(
        config={"aggregated_dir": str(agg), "hru_area_m2": hru_area},
        output_path=str(out),
    )

    assert out.exists()
    ds = xr.open_dataset(out)
    try:
        assert "lower_bound" in ds.data_vars
        assert "upper_bound" in ds.data_vars
        assert ds["lower_bound"].dims == ("time", "hru")
        assert (ds["lower_bound"].values <= ds["upper_bound"].values).all()
        # CF attrs on variables, not dataset
        assert ds["lower_bound"].attrs.get("units") == "cfs"
        assert ds["upper_bound"].attrs.get("units") == "cfs"
        assert "units" not in ds.attrs, "Dataset-level 'units' should not exist"
        assert "cell_methods" not in ds.attrs, (
            "Dataset-level 'cell_methods' should not exist"
        )
        assert ds.attrs.get("Conventions") == "CF-1.6"
    finally:
        ds.close()


def test_build_numeric_correctness(tmp_path):
    """Verify exact cfs values for ERA5=0.05 m and GLDAS=30 kg/m² in January."""
    from nhf_spatial_targets.targets.run import build

    HRU_AREA = 1.0e8  # m²
    ERA5_M = 0.05  # m/month
    GLDAS_KGM2 = 30.0  # kg m-2 = mm

    times = pd.date_range("2020-01-01", periods=1, freq="1MS")
    hrus = np.arange(1)

    def _save(path: Path, var: str, val, units: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        da = xr.DataArray(
            np.full((1, 1), val),
            dims=("time", "hru"),
            coords={"time": times, "hru": hrus},
            name=var,
            attrs={"units": units},
        )
        da.to_dataset().to_netcdf(path)

    agg = tmp_path / "data" / "aggregated"
    _save(agg / "era5_land" / "ro.nc", "ro", ERA5_M, "m")
    _save(
        agg / "gldas_noah_v21_monthly" / "runoff_total.nc",
        "runoff_total",
        GLDAS_KGM2,
        "kg m-2",
    )

    out = tmp_path / "targets" / "runoff_target.nc"
    hru_area = xr.DataArray(np.full(1, HRU_AREA), dims=("hru",), coords={"hru": hrus})
    build(
        config={"aggregated_dir": str(agg), "hru_area_m2": hru_area},
        output_path=str(out),
    )

    ds = xr.open_dataset(out)
    try:
        # January 2020 has 31 days
        days_jan = 31
        # ERA5: 0.05 m → 50 mm; GLDAS: 30 mm → lower=30, upper=50
        for mm_val, bound_name in [(30.0, "lower_bound"), (50.0, "upper_bound")]:
            expected_m3_per_day = (mm_val * 1e-3 / days_jan) * HRU_AREA
            expected_cfs = expected_m3_per_day * _M3_PER_DAY_TO_CFS
            np.testing.assert_allclose(
                ds[bound_name].values.squeeze(),
                expected_cfs,
                rtol=1e-4,
                err_msg=f"{bound_name}: expected {expected_cfs:.6f} cfs",
            )
    finally:
        ds.close()


def test_build_disjoint_time_ranges_raises(tmp_path):
    """build() raises ValueError when ERA5 and GLDAS have no time overlap."""
    from nhf_spatial_targets.targets.run import build

    hrus = np.arange(2)
    hru_area = xr.DataArray(np.full(2, 1.0e8), dims=("hru",), coords={"hru": hrus})

    def _save(path: Path, var: str, times):
        path.parent.mkdir(parents=True, exist_ok=True)
        da = xr.DataArray(
            np.ones((len(times), 2)),
            dims=("time", "hru"),
            coords={"time": times, "hru": hrus},
            name=var,
            attrs={"units": "m"},
        )
        da.to_dataset().to_netcdf(path)

    agg = tmp_path / "data" / "aggregated"
    era5_times = pd.date_range("2000-01-01", periods=3, freq="1MS")
    gldas_times = pd.date_range("2010-01-01", periods=3, freq="1MS")
    _save(agg / "era5_land" / "ro.nc", "ro", era5_times)
    _save(
        agg / "gldas_noah_v21_monthly" / "runoff_total.nc",
        "runoff_total",
        gldas_times,
    )

    out = tmp_path / "targets" / "runoff_target.nc"
    with pytest.raises(ValueError, match="do not overlap"):
        build(
            config={"aggregated_dir": str(agg), "hru_area_m2": hru_area},
            output_path=str(out),
        )


def test_validate_alignment_disjoint_raises():
    """_validate_alignment raises ValueError for disjoint time ranges."""
    times_a = pd.date_range("2000-01-01", periods=3, freq="1MS")
    times_b = pd.date_range("2010-01-01", periods=3, freq="1MS")
    da_a = xr.DataArray(np.ones((3,)), dims=("time",), coords={"time": times_a})
    da_b = xr.DataArray(np.ones((3,)), dims=("time",), coords={"time": times_b})
    with pytest.raises(ValueError, match="do not overlap"):
        _validate_alignment(da_a, da_b)
