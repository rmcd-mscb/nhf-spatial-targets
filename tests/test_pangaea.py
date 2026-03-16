"""Tests for PANGAEA fetch module (WaterGAP 2.2d)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def _make_watergap_nc(path: Path, n_times: int = 24) -> Path:
    """Create a synthetic WaterGAP-style NC4 with non-CF time encoding.

    Mimics the real file: time as float offsets with
    units='months since 1901-01-01', calendar='proleptic_gregorian'.
    """
    time_vals = np.arange(n_times, dtype=np.float32)
    lat = np.array([89.75, 45.25, 0.25, -44.75], dtype=np.float32)
    lon = np.array([-179.75, -90.25, 0.25, 89.75], dtype=np.float32)
    data = np.random.rand(n_times, len(lat), len(lon)).astype(np.float32)

    ds = xr.Dataset(
        {"qrdif": (["time", "lat", "lon"], data)},
        coords={
            "time": xr.Variable(
                "time",
                time_vals,
                attrs={
                    "units": "months since 1901-01-01",
                    "calendar": "proleptic_gregorian",
                    "standard_name": "time",
                },
            ),
            "lat": xr.Variable(
                "lat",
                lat,
                attrs={
                    "units": "degrees_north",
                    "standard_name": "latitude",
                },
            ),
            "lon": xr.Variable(
                "lon",
                lon,
                attrs={
                    "units": "degrees_east",
                    "standard_name": "longitude",
                },
            ),
        },
        attrs={
            "conventions": "partly ALMA, CF and ISIMIP2b protocol",
            "title": "Test WaterGAP",
        },
    )
    ds["qrdif"].attrs.update(
        {
            "standard_name": "qrdif",
            "long_name": "diffuse groundwater recharge",
            "units": "kg m-2 s-1",
        }
    )
    ds.to_netcdf(path, format="NETCDF4")
    return path


def test_cf_fixup_reconstructs_time(tmp_path: Path):
    """Time coordinate must be proper datetime64 after CF fix-up."""
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    raw = _make_watergap_nc(tmp_path / "raw.nc4", n_times=24)
    fixed = _cf_fixup(raw, tmp_path / "fixed.nc")

    ds = xr.open_dataset(fixed)
    assert ds.time.dtype == np.dtype("datetime64[ns]")
    # 24 months from 1901-01 = 1901-01 through 1902-12
    assert str(ds.time.values[0])[:7] == "1901-01"
    assert str(ds.time.values[12])[:7] == "1902-01"
    assert str(ds.time.values[23])[:7] == "1902-12"
    ds.close()


def test_cf_fixup_adds_grid_mapping(tmp_path: Path):
    """CF fix-up must add crs variable and grid_mapping attr to qrdif."""
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    raw = _make_watergap_nc(tmp_path / "raw.nc4")
    fixed = _cf_fixup(raw, tmp_path / "fixed.nc")

    ds = xr.open_dataset(fixed)
    assert "crs" in ds.data_vars
    assert ds["qrdif"].attrs.get("grid_mapping") == "crs"
    assert ds["crs"].attrs.get("grid_mapping_name") == "latitude_longitude"
    ds.close()


def test_cf_fixup_sets_conventions(tmp_path: Path):
    """CF fix-up must set Conventions to CF-1.6."""
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    raw = _make_watergap_nc(tmp_path / "raw.nc4")
    fixed = _cf_fixup(raw, tmp_path / "fixed.nc")

    ds = xr.open_dataset(fixed)
    assert ds.attrs.get("Conventions") == "CF-1.6"
    ds.close()


def test_cf_fixup_preserves_data(tmp_path: Path):
    """CF fix-up must not alter the qrdif data values."""
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    raw = _make_watergap_nc(tmp_path / "raw.nc4", n_times=12)
    raw_ds = xr.open_dataset(raw, decode_times=False)
    raw_values = raw_ds["qrdif"].values.copy()
    raw_ds.close()

    fixed = _cf_fixup(raw, tmp_path / "fixed.nc")
    fixed_ds = xr.open_dataset(fixed)
    np.testing.assert_array_equal(fixed_ds["qrdif"].values, raw_values)
    fixed_ds.close()
