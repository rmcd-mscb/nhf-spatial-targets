"""Tests for ERA5-Land aggregation adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nhf_spatial_targets.aggregate.era5_land import ADAPTER, _open_monthly


@pytest.fixture()
def monthly_nc(tmp_path):
    ds_dir = tmp_path / "era5_land"
    ds_dir.mkdir()
    times = pd.date_range("2000-01-01", periods=3, freq="MS")
    ds = xr.Dataset(
        {
            "ro": (["time", "lat", "lon"], np.ones((3, 2, 2))),
            "sro": (["time", "lat", "lon"], np.ones((3, 2, 2)) * 0.5),
            "ssro": (["time", "lat", "lon"], np.ones((3, 2, 2)) * 0.5),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )
    # Distinguishable from any daily neighbour by the "monthly" token.
    ds.to_netcdf(ds_dir / "era5_land_monthly_2000_2002.nc")
    ds.to_netcdf(ds_dir / "era5_land_daily_2000_2002.nc")
    return ds_dir


def test_adapter_declares_runoff_vars():
    assert ADAPTER.source_key == "era5_land"
    assert ADAPTER.output_name == "era5_land_agg.nc"
    assert set(ADAPTER.variables) == {"ro", "sro", "ssro"}


def test_open_monthly_selects_monthly_nc(monthly_nc):
    project = MagicMock()
    project.raw_dir.return_value = monthly_nc
    ds = _open_monthly(project)
    assert "ro" in ds
    assert ds.sizes["time"] == 3
