from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.targets.run import (
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
        fabric_path="unused",
        output_path=str(out),
    )

    assert out.exists()
    ds = xr.open_dataset(out)
    try:
        assert "lower_bound" in ds.data_vars
        assert "upper_bound" in ds.data_vars
        assert ds["lower_bound"].dims == ("time", "hru")
        assert (ds["lower_bound"].values <= ds["upper_bound"].values).all()
    finally:
        ds.close()
