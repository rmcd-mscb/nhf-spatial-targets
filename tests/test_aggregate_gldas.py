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
