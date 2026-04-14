"""Tests for the shared aggregation driver."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from shapely.geometry import box

from nhf_spatial_targets.aggregate._driver import update_manifest
from nhf_spatial_targets.workspace import load as load_project


@pytest.fixture()
def project(tmp_path):
    datastore = tmp_path / "datastore"
    datastore.mkdir()
    config = {
        "fabric": {"path": "", "id_col": "hru_id"},
        "datastore": str(datastore),
    }
    (tmp_path / "config.yml").write_text(yaml.dump(config))
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "abc123"}))
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    return load_project(tmp_path)


def test_update_manifest_writes_source_entry(project):
    update_manifest(
        project=project,
        source_key="foo",
        access={"type": "nasa_gesdisc", "short_name": "FOO"},
        period="2000-01-01/2009-12-31",
        output_file="data/aggregated/foo_agg.nc",
        weight_files=["weights/foo_batch0.csv"],
    )
    manifest = json.loads((project.workdir / "manifest.json").read_text())
    entry = manifest["sources"]["foo"]
    assert entry["source_key"] == "foo"
    assert entry["access_type"] == "nasa_gesdisc"
    assert entry["short_name"] == "FOO"
    assert entry["period"] == "2000-01-01/2009-12-31"
    assert entry["fabric_sha256"] == "abc123"
    assert entry["output_file"] == "data/aggregated/foo_agg.nc"
    assert entry["weight_files"] == ["weights/foo_batch0.csv"]
    assert "timestamp" in entry


def test_update_manifest_preserves_existing_sources(project):
    manifest_path = project.workdir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"sources": {"bar": {"source_key": "bar"}}, "steps": []})
    )
    update_manifest(
        project=project,
        source_key="foo",
        access={"type": "local"},
        period="2000/2001",
        output_file="data/aggregated/foo_agg.nc",
        weight_files=[],
    )
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["sources"].keys()) == {"foo", "bar"}


def test_source_adapter_defaults():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP", "GWETROOT"],
    )
    assert adapter.source_crs == "EPSG:4326"
    assert adapter.x_coord == "lon"
    assert adapter.y_coord == "lat"
    assert adapter.time_coord == "time"
    assert adapter.open_hook is None


def test_source_adapter_open_hook_invocable(project):
    import xarray as xr

    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    def _open(proj):
        return xr.Dataset({"a": (("time",), [1.0])}, coords={"time": [0]})

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP"],
        open_hook=_open,
    )
    ds = adapter.open_hook(project)
    assert "a" in ds


@pytest.fixture()
def tiny_fabric(tmp_path):
    polys = [box(i, 0, i + 1, 1) for i in range(4)]
    gdf = gpd.GeoDataFrame(
        {"hru_id": range(4)},
        geometry=polys,
        crs="EPSG:4326",
    )
    path = tmp_path / "fabric.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def test_load_and_batch_fabric_single_batch(tiny_fabric):
    from nhf_spatial_targets.aggregate._driver import load_and_batch_fabric

    batched = load_and_batch_fabric(tiny_fabric, batch_size=500)
    assert "batch_id" in batched.columns
    assert batched["batch_id"].nunique() == 1


def _fake_user_data():
    return MagicMock()


def _fake_weights():
    return pd.DataFrame(
        {"i": [0, 1], "j": [0, 0], "wght": [0.5, 0.5], "hru_id": [0, 0]}
    )


def _fake_agg_result(var_name, hru_ids):
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    data = np.array([[1.0] * len(hru_ids), [2.0] * len(hru_ids)])
    ds = xr.Dataset(
        {var_name: (["time", "hru_id"], data)},
        coords={"time": times, "hru_id": hru_ids},
    )
    gdf = gpd.GeoDataFrame({"hru_id": hru_ids})
    return gdf, ds


def test_aggregate_variables_for_batch_merges_variables(tiny_fabric):
    from nhf_spatial_targets.aggregate._driver import (
        aggregate_variables_for_batch,
    )

    batch_gdf = gpd.read_file(tiny_fabric)
    batch_gdf["batch_id"] = 0
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    source_ds = xr.Dataset(
        {
            "a": (["time", "lat", "lon"], np.ones((2, 2, 2))),
            "b": (["time", "lat", "lon"], np.ones((2, 2, 2)) * 2),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )

    with (
        patch("nhf_spatial_targets.aggregate._driver.AggGen") as mock_agg,
        patch("nhf_spatial_targets.aggregate._driver.UserCatData") as mock_ucd,
    ):
        mock_ucd.return_value = _fake_user_data()
        agg_instance = MagicMock()
        mock_agg.return_value = agg_instance
        agg_instance.calculate_agg.side_effect = [
            _fake_agg_result("a", [0, 1, 2, 3]),
            _fake_agg_result("b", [0, 1, 2, 3]),
        ]

        result = aggregate_variables_for_batch(
            batch_gdf=batch_gdf,
            source_ds=source_ds,
            variables=["a", "b"],
            source_crs="EPSG:4326",
            x_coord="lon",
            y_coord="lat",
            time_coord="time",
            id_col="hru_id",
            weights=_fake_weights(),
        )
    assert set(result.data_vars) == {"a", "b"}
    assert result.sizes["hru_id"] == 4


def test_aggregate_source_writes_multi_var_nc_and_manifest(tmp_path, tiny_fabric):
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    # --- minimal project ---
    datastore = tmp_path / "datastore"
    (datastore / "merra2").mkdir(parents=True)
    # write a placeholder consolidated NC so default open_hook has something
    src_nc = datastore / "merra2" / "merra2.nc"
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    xr.Dataset(
        {
            "a": (["time", "lat", "lon"], np.ones((2, 2, 2))),
            "b": (["time", "lat", "lon"], np.ones((2, 2, 2)) * 2.0),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    ).to_netcdf(src_nc)

    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": str(tiny_fabric), "id_col": "hru_id"},
                "datastore": str(datastore),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    (tmp_path / "data" / "aggregated").mkdir(parents=True)
    (tmp_path / "weights").mkdir()

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["a", "b"],
    )

    # Patch catalog.source to supply access metadata for the manifest
    fake_meta = {"access": {"type": "local_nc"}}

    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.catalog_source",
            return_value=fake_meta,
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=_fake_weights(),
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch"
        ) as mock_agg_batch,
    ):
        times = pd.date_range("2000-01-01", periods=2, freq="MS")
        mock_agg_batch.return_value = xr.Dataset(
            {
                "a": (["time", "hru_id"], np.ones((2, 4))),
                "b": (["time", "hru_id"], np.ones((2, 4)) * 2.0),
            },
            coords={"time": times, "hru_id": [0, 1, 2, 3]},
        )
        out = aggregate_source(
            adapter,
            fabric_path=tiny_fabric,
            id_col="hru_id",
            workdir=tmp_path,
            batch_size=500,
        )

    assert set(out.data_vars) == {"a", "b"}
    output_nc = tmp_path / "data" / "aggregated" / "merra2_agg.nc"
    assert output_nc.exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "merra2" in manifest["sources"]
    assert manifest["sources"]["merra2"]["output_file"] == (
        "data/aggregated/merra2_agg.nc"
    )


def test_source_adapter_rejects_empty_variables():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    with pytest.raises(ValueError, match="non-empty"):
        SourceAdapter(source_key="merra2", output_name="m.nc", variables=[])


def test_source_adapter_rejects_path_in_output_name():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    with pytest.raises(ValueError, match="bare filename"):
        SourceAdapter(
            source_key="merra2",
            output_name="subdir/foo.nc",
            variables=["GWETTOP"],
        )


def test_source_adapter_rejects_unknown_source_key():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    with pytest.raises(ValueError, match="catalog"):
        SourceAdapter(
            source_key="not_a_real_source",
            output_name="foo.nc",
            variables=["x"],
        )


def test_source_adapter_coerces_list_to_tuple():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    adapter = SourceAdapter(
        source_key="merra2", output_name="m.nc", variables=["GWETTOP"]
    )
    assert isinstance(adapter.variables, tuple)
