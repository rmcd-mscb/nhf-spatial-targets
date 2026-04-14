"""Tests for GLDAS aggregation adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nhf_spatial_targets.aggregate.gldas import ADAPTER, _open


@pytest.fixture()
def gldas_nc(tmp_path):
    ds_dir = tmp_path / "gldas_noah_v21_monthly"
    ds_dir.mkdir()
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    xr.Dataset(
        {
            "Qs_acc": (["time", "lat", "lon"], np.ones((2, 2, 2))),
            "Qsb_acc": (["time", "lat", "lon"], np.ones((2, 2, 2)) * 3.0),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    ).to_netcdf(ds_dir / "gldas_noah_v21_monthly.nc")
    return ds_dir


def test_adapter_declares_runoff_vars():
    assert ADAPTER.source_key == "gldas_noah_v21_monthly"
    assert ADAPTER.output_name == "gldas_agg.nc"
    assert set(ADAPTER.variables) == {"Qs_acc", "Qsb_acc", "runoff_total"}


def test_open_adds_runoff_total(gldas_nc):
    project = MagicMock()
    project.raw_dir.return_value = gldas_nc
    ds = _open(project)
    assert "runoff_total" in ds
    np.testing.assert_allclose(ds["runoff_total"].values, 4.0)
    assert ds["runoff_total"].attrs["units"] == "kg m-2"


def test_open_preserves_nan_in_runoff_total(tmp_path):
    """If either Qs_acc or Qsb_acc is NaN at a cell, runoff_total is NaN there."""
    ds_dir = tmp_path / "gldas_noah_v21_monthly"
    ds_dir.mkdir()
    times = pd.date_range("2000-01-01", periods=1, freq="MS")
    qs = np.array([[[1.0, np.nan], [1.0, 1.0]]])
    qsb = np.array([[[2.0, 2.0], [np.nan, 2.0]]])
    xr.Dataset(
        {
            "Qs_acc": (["time", "lat", "lon"], qs),
            "Qsb_acc": (["time", "lat", "lon"], qsb),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    ).to_netcdf(ds_dir / "gldas_noah_v21_monthly.nc")

    project = MagicMock()
    project.raw_dir.return_value = ds_dir
    ds = _open(project)
    rt = ds["runoff_total"].isel(time=0).values
    assert np.isnan(rt[0, 1])  # Qs NaN -> total NaN
    assert np.isnan(rt[1, 0])  # Qsb NaN -> total NaN
    assert rt[0, 0] == 3.0
    assert rt[1, 1] == 3.0
