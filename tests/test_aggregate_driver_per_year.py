"""Unit tests for per-year streaming aggregation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from shapely.geometry import box

from nhf_spatial_targets.workspace import load as load_project


def _write_nc(path: Path, times: pd.DatetimeIndex) -> None:
    xr.Dataset(
        {"v": (["time", "lat", "lon"], np.ones((len(times), 2, 2)))},
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "lat": ("lat", [0.0, 1.0], {"standard_name": "latitude"}),
            "lon": ("lon", [0.0, 1.0], {"standard_name": "longitude"}),
        },
    ).to_netcdf(path)


def test_enumerate_years_per_year_files(tmp_path):
    from nhf_spatial_targets.aggregate._driver import enumerate_years

    f2000 = tmp_path / "src_2000_consolidated.nc"
    f2001 = tmp_path / "src_2001_consolidated.nc"
    _write_nc(f2000, pd.date_range("2000-01-01", periods=12, freq="MS"))
    _write_nc(f2001, pd.date_range("2001-01-01", periods=12, freq="MS"))

    year_files = enumerate_years([f2000, f2001])
    assert year_files == [(2000, f2000), (2001, f2001)]


def test_enumerate_years_single_multi_year_file(tmp_path):
    from nhf_spatial_targets.aggregate._driver import enumerate_years

    f = tmp_path / "src_consolidated.nc"
    _write_nc(f, pd.date_range("2000-01-01", periods=36, freq="MS"))

    year_files = enumerate_years([f])
    assert year_files == [(2000, f), (2001, f), (2002, f)]


def test_enumerate_years_raises_on_year_overlap(tmp_path):
    from nhf_spatial_targets.aggregate._driver import enumerate_years

    fa = tmp_path / "src_a_consolidated.nc"
    fb = tmp_path / "src_b_consolidated.nc"
    _write_nc(fa, pd.date_range("2001-01-01", periods=12, freq="MS"))
    _write_nc(fb, pd.date_range("2001-06-01", periods=12, freq="MS"))

    with pytest.raises(ValueError, match="overlap"):
        enumerate_years([fa, fb])


def test_enumerate_years_raises_when_no_time_coord(tmp_path):
    from nhf_spatial_targets.aggregate._driver import enumerate_years

    f = tmp_path / "no_time_consolidated.nc"
    xr.Dataset({"v": (["y", "x"], np.zeros((2, 2)))}).to_netcdf(f)
    with pytest.raises(ValueError, match="time"):
        enumerate_years([f])


@pytest.fixture()
def project(tmp_path):
    datastore = tmp_path / "datastore"
    datastore.mkdir()
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": "", "id_col": "hru_id"},
                "datastore": str(datastore),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "abc"}))
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    (tmp_path / "data" / "aggregated").mkdir(parents=True)
    (tmp_path / "weights").mkdir()
    return load_project(tmp_path)


@pytest.fixture()
def tiny_batched_fabric():
    polys = [box(i, 0, i + 1, 1) for i in range(2)]
    gdf = gpd.GeoDataFrame(
        {"hru_id": range(2), "batch_id": [0, 0]}, geometry=polys, crs="EPSG:4326"
    )
    return gdf


def test_per_year_output_path(project):
    from nhf_spatial_targets.aggregate._driver import per_year_output_path

    p = per_year_output_path(project, "foo", 2005)
    assert p == project.workdir / "data" / "aggregated" / "_by_year" / "foo_2005_agg.nc"


def test_aggregate_year_skips_when_intermediate_exists(project, tiny_batched_fabric):
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import (
        aggregate_year,
        per_year_output_path,
    )

    out_path = per_year_output_path(project, "merra2", 2005)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2005-01-01", periods=1, freq="MS")
    xr.Dataset(
        {"GWETTOP": (["time", "hru_id"], np.ones((1, 2)))},
        coords={"time": times, "hru_id": [0, 1]},
    ).to_netcdf(out_path)

    src_dir = project.raw_dir("merra2")
    src_dir.mkdir(parents=True, exist_ok=True)
    src_file = src_dir / "src_2005_consolidated.nc"
    _write_nc(src_file, pd.date_range("2005-01-01", periods=12, freq="MS"))

    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=["v"]
    )
    with patch(
        "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch"
    ) as mock_batch:
        path = aggregate_year(
            adapter, project, 2005, src_file, tiny_batched_fabric, "hru_id"
        )
    assert path == out_path
    assert not mock_batch.called, "existing intermediate must skip aggregation"


def test_aggregate_year_writes_intermediate_when_missing(project, tiny_batched_fabric):
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import (
        aggregate_year,
        per_year_output_path,
    )

    src_dir = project.raw_dir("merra2")
    src_dir.mkdir(parents=True, exist_ok=True)
    src_file = src_dir / "src_2005_consolidated.nc"
    _write_nc(src_file, pd.date_range("2005-01-01", periods=12, freq="MS"))

    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=["v"]
    )

    fake_weights = pd.DataFrame({"i": [0], "j": [0], "wght": [1.0], "hru_id": [0]})
    fake_year_ds = xr.Dataset(
        {"v": (["time", "hru_id"], np.ones((1, 2)))},
        coords={
            "time": (
                "time",
                pd.date_range("2005-01-01", periods=1, freq="MS"),
                {"standard_name": "time"},
            ),
            "hru_id": [0, 1],
        },
    )
    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=fake_weights,
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            return_value=fake_year_ds,
        ) as mock_batch,
    ):
        path = aggregate_year(
            adapter, project, 2005, src_file, tiny_batched_fabric, "hru_id"
        )

    assert path == per_year_output_path(project, "merra2", 2005)
    assert path.exists()
    call = mock_batch.call_args
    assert call.kwargs["period"] == ("2005-01-01", "2005-12-31")
