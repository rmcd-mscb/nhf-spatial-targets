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
        output_files=[
            "data/aggregated/foo/foo_2000_agg.nc",
            "data/aggregated/foo/foo_2001_agg.nc",
        ],
        weight_files=["weights/foo_batch0.csv"],
    )
    manifest = json.loads((project.workdir / "manifest.json").read_text())
    entry = manifest["sources"]["foo"]
    assert entry["source_key"] == "foo"
    assert entry["access_type"] == "nasa_gesdisc"
    assert entry["short_name"] == "FOO"
    assert entry["period"] == "2000-01-01/2009-12-31"
    assert entry["fabric_sha256"] == "abc123"
    assert entry["output_files"] == [
        "data/aggregated/foo/foo_2000_agg.nc",
        "data/aggregated/foo/foo_2001_agg.nc",
    ]
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
        output_files=["data/aggregated/foo/foo_2000_agg.nc"],
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
    assert adapter.x_coord is None
    assert adapter.y_coord is None
    assert adapter.time_coord is None
    assert adapter.pre_aggregate_hook is None
    assert adapter.post_aggregate_hook is None


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
            period=("2000-01-01", "2000-02-01"),
        )
    assert set(result.data_vars) == {"a", "b"}
    assert result.sizes["hru_id"] == 4


def test_aggregate_source_writes_multi_var_nc_and_manifest(tmp_path, tiny_fabric):
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    datastore = tmp_path / "datastore"
    (datastore / "merra2").mkdir(parents=True)
    src_nc = datastore / "merra2" / "merra2_2000_consolidated.nc"
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    xr.Dataset(
        {
            "a": (["time", "lat", "lon"], np.ones((12, 2, 2))),
            "b": (["time", "lat", "lon"], np.ones((12, 2, 2)) * 2.0),
        },
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "lat": ("lat", [0.25, 0.75], {"standard_name": "latitude"}),
            "lon": ("lon", [0.5, 1.5], {"standard_name": "longitude"}),
        },
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

    fake_meta = {"access": {"type": "local_nc"}}
    fake_year_ds = xr.Dataset(
        {
            "a": (["time", "hru_id"], np.ones((12, 4))),
            "b": (["time", "hru_id"], np.ones((12, 4)) * 2.0),
        },
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "hru_id": [0, 1, 2, 3],
        },
    )

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
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            return_value=fake_year_ds,
        ),
    ):
        aggregate_source(
            adapter,
            fabric_path=tiny_fabric,
            id_col="hru_id",
            workdir=tmp_path,
            batch_size=500,
        )

    # No consolidated file is produced anymore.
    legacy_consolidated = tmp_path / "data" / "aggregated" / "merra2_agg.nc"
    assert not legacy_consolidated.exists()
    per_year = tmp_path / "data" / "aggregated" / "merra2" / "merra2_2000_agg.nc"
    assert per_year.exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "merra2" in manifest["sources"]
    assert manifest["sources"]["merra2"]["output_files"] == [
        "data/aggregated/merra2/merra2_2000_agg.nc"
    ]


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


def test_source_adapter_defaults_grid_variable_to_first():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP", "GWETROOT"],
    )
    assert adapter.grid_variable == "GWETTOP"


def test_source_adapter_explicit_grid_variable_must_be_in_variables():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    with pytest.raises(ValueError, match="grid_variable"):
        SourceAdapter(
            source_key="merra2",
            output_name="merra2_agg.nc",
            variables=["GWETTOP", "GWETROOT"],
            grid_variable="BOGUS",
        )


def test_source_adapter_accepts_valid_grid_variable():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP", "GWETROOT"],
        grid_variable="GWETROOT",
    )
    assert adapter.grid_variable == "GWETROOT"


def test_source_adapter_raw_grid_variable_defaults_to_grid_variable():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=("GWETTOP", "GWETROOT"),
        grid_variable="GWETROOT",
    )
    assert adapter.raw_grid_variable == "GWETROOT"


def test_source_adapter_raw_grid_variable_can_differ_from_declared_vars():
    """Hooked adapters whose declared variables are all derived must be able to
    name a raw-NC variable for the grid-shape invariant."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    adapter = SourceAdapter(
        source_key="mod10c1_v061",
        output_name="mod10c1_agg.nc",
        variables=("sca", "ci", "valid_mask"),
        grid_variable="sca",
        raw_grid_variable="Day_CMG_Snow_Cover",
        pre_aggregate_hook=lambda ds: ds,
    )
    assert adapter.raw_grid_variable == "Day_CMG_Snow_Cover"
    assert "Day_CMG_Snow_Cover" not in adapter.variables


def test_mod10c1_adapter_declares_raw_grid_variable():
    """Regression: MOD10C1 must set raw_grid_variable so the grid-drift check
    is not silently skipped (it used to short-circuit on ``sca not in raw``)."""
    from nhf_spatial_targets.aggregate.mod10c1 import ADAPTER

    assert ADAPTER.raw_grid_variable == "Day_CMG_Snow_Cover"
    assert ADAPTER.raw_grid_variable not in ADAPTER.variables


def test_source_adapter_coerces_list_to_tuple():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    adapter = SourceAdapter(
        source_key="merra2", output_name="m.nc", variables=["GWETTOP"]
    )
    assert isinstance(adapter.variables, tuple)


def test_aggregate_source_raises_when_variable_missing(tmp_path, tiny_fabric):
    """Adapter declaring a variable absent from the source NC must raise early."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    datastore = tmp_path / "datastore"
    (datastore / "merra2").mkdir(parents=True)
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    # NC only has "GWETTOP"; adapter will ask for "BOGUS_VAR" too.
    xr.Dataset(
        {"GWETTOP": (["time", "lat", "lon"], np.ones((2, 2, 2)))},
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    ).to_netcdf(datastore / "merra2" / "merra2_2000_consolidated.nc")

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

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP", "BOGUS_VAR"],
        x_coord="lon",
        y_coord="lat",
        time_coord="time",
    )
    with pytest.raises(ValueError, match="BOGUS_VAR"):
        aggregate_source(
            adapter,
            fabric_path=tiny_fabric,
            id_col="hru_id",
            workdir=tmp_path,
        )


def test_compute_or_load_weights_writes_cache_on_miss(tmp_path, tiny_fabric):
    """First call computes weights via WeightGen; second call loads from cache."""
    from nhf_spatial_targets.aggregate._driver import compute_or_load_weights

    (tmp_path / "weights").mkdir()
    batch_gdf = gpd.read_file(tiny_fabric)
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    source_ds = xr.Dataset(
        {"a": (["time", "lat", "lon"], np.ones((2, 2, 2)))},
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )
    fake = _fake_weights()

    with (
        patch("nhf_spatial_targets.aggregate._driver.WeightGen") as mock_wg,
        patch("nhf_spatial_targets.aggregate._driver.UserCatData"),
    ):
        inst = MagicMock()
        inst.calculate_weights.return_value = fake
        mock_wg.return_value = inst
        weights = compute_or_load_weights(
            batch_gdf=batch_gdf,
            source_ds=source_ds,
            source_var="a",
            source_crs="EPSG:4326",
            x_coord="lon",
            y_coord="lat",
            time_coord="time",
            id_col="hru_id",
            source_key="toy",
            batch_id=0,
            workdir=tmp_path,
            period=("2000-01-01", "2000-02-01"),
        )
    assert mock_wg.called
    cache_path = tmp_path / "weights" / "toy_batch0.csv"
    assert cache_path.exists()
    pd.testing.assert_frame_equal(weights, fake)


def test_compute_or_load_weights_uses_cache_on_hit(tmp_path, tiny_fabric):
    """Preexisting cache CSV must be loaded without invoking WeightGen."""
    from nhf_spatial_targets.aggregate._driver import compute_or_load_weights

    (tmp_path / "weights").mkdir()
    batch_gdf = gpd.read_file(tiny_fabric)
    source_ds = xr.Dataset(
        {"a": (["time", "lat", "lon"], np.ones((1, 2, 2)))},
        coords={
            "time": pd.date_range("2000-01-01", periods=1, freq="MS"),
            "lat": [0.25, 0.75],
            "lon": [0.5, 1.5],
        },
    )
    cached = _fake_weights()
    cache_path = tmp_path / "weights" / "toy_batch0.csv"
    cached.to_csv(cache_path, index=False)

    with patch("nhf_spatial_targets.aggregate._driver.WeightGen") as mock_wg:
        weights = compute_or_load_weights(
            batch_gdf=batch_gdf,
            source_ds=source_ds,
            source_var="a",
            source_crs="EPSG:4326",
            x_coord="lon",
            y_coord="lat",
            time_coord="time",
            id_col="hru_id",
            source_key="toy",
            batch_id=0,
            workdir=tmp_path,
            period=("2000-01-01", "2000-02-01"),
        )
    assert not mock_wg.called
    pd.testing.assert_frame_equal(weights, cached)


def test_update_manifest_raises_on_corrupt_json(project):
    from nhf_spatial_targets.aggregate._driver import update_manifest

    (project.workdir / "manifest.json").write_text("{not json")
    with pytest.raises(ValueError, match="corrupt"):
        update_manifest(
            project=project,
            source_key="foo",
            access={"type": "local"},
            period="2000/2001",
            output_files=["data/aggregated/foo/foo_2000_agg.nc"],
            weight_files=[],
        )


def test_source_adapter_rejects_invalid_source_crs():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    with pytest.raises(ValueError, match="source_crs"):
        SourceAdapter(
            source_key="merra2",
            output_name="merra2_agg.nc",
            variables=["GWETTOP"],
            source_crs="EPSG4326",  # missing colon
        )


def test_source_adapter_accepts_valid_epsg():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    # Default EPSG:4326 is implicitly tested by all other adapter tests.
    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP"],
        source_crs="EPSG:5070",
    )
    assert adapter.source_crs == "EPSG:5070"


def test_compute_or_load_weights_ignores_stray_tmp_from_crashed_run(
    tmp_path, tiny_fabric
):
    """A leftover .tmp file from a prior crashed run must not be loaded
    as a valid cache; the final-name CSV must be written cleanly."""
    from nhf_spatial_targets.aggregate._driver import compute_or_load_weights

    (tmp_path / "weights").mkdir()
    # Simulate a crashed prior run that left behind a partial tmp file.
    stray = tmp_path / "weights" / "toy_batch0.csv.tmp"
    stray.write_text("garbage,partial\n1,2\n")

    batch_gdf = gpd.read_file(tiny_fabric)
    source_ds = xr.Dataset(
        {"a": (["time", "lat", "lon"], np.ones((1, 2, 2)))},
        coords={
            "time": pd.date_range("2000-01-01", periods=1, freq="MS"),
            "lat": [0.25, 0.75],
            "lon": [0.5, 1.5],
        },
    )
    fake = _fake_weights()

    with (
        patch("nhf_spatial_targets.aggregate._driver.WeightGen") as mock_wg,
        patch("nhf_spatial_targets.aggregate._driver.UserCatData"),
    ):
        inst = MagicMock()
        inst.calculate_weights.return_value = fake
        mock_wg.return_value = inst
        weights = compute_or_load_weights(
            batch_gdf=batch_gdf,
            source_ds=source_ds,
            source_var="a",
            source_crs="EPSG:4326",
            x_coord="lon",
            y_coord="lat",
            time_coord="time",
            id_col="hru_id",
            source_key="toy",
            batch_id=0,
            workdir=tmp_path,
            period=("2000-01-01", "2000-02-01"),
        )

    # Final-name CSV present and equals the fresh computation.
    final = tmp_path / "weights" / "toy_batch0.csv"
    assert final.exists()
    pd.testing.assert_frame_equal(weights, fake)
    # WeightGen was invoked — cache was NOT short-circuited by the stray tmp.
    assert mock_wg.called


def test_source_adapter_accepts_hooks():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    def _pre(ds):
        return ds

    def _post(ds):
        return ds

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP"],
        pre_aggregate_hook=_pre,
        post_aggregate_hook=_post,
    )
    assert adapter.pre_aggregate_hook is _pre
    assert adapter.post_aggregate_hook is _post


def test_aggregate_variables_for_batch_passes_period_through(tiny_fabric):
    from nhf_spatial_targets.aggregate._driver import aggregate_variables_for_batch

    batch_gdf = gpd.read_file(tiny_fabric)
    batch_gdf["batch_id"] = 0
    times = pd.date_range("2005-01-01", periods=12, freq="MS")
    source_ds = xr.Dataset(
        {"a": (["time", "lat", "lon"], np.ones((12, 2, 2)))},
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )

    with (
        patch("nhf_spatial_targets.aggregate._driver.AggGen") as mock_agg,
        patch("nhf_spatial_targets.aggregate._driver.UserCatData") as mock_ucd,
    ):
        mock_ucd.return_value = _fake_user_data()
        inst = MagicMock()
        mock_agg.return_value = inst
        inst.calculate_agg.side_effect = [_fake_agg_result("a", [0, 1, 2, 3])]

        aggregate_variables_for_batch(
            batch_gdf=batch_gdf,
            source_ds=source_ds,
            variables=["a"],
            source_crs="EPSG:4326",
            x_coord="lon",
            y_coord="lat",
            time_coord="time",
            id_col="hru_id",
            weights=_fake_weights(),
            period=("2005-01-01", "2005-12-01"),
        )

    call_kwargs = mock_ucd.call_args.kwargs
    assert call_kwargs["period"] == ["2005-01-01", "2005-12-01"]


def _setup_aggregate_source_project(tmp_path, tiny_fabric, source_key):
    """Helper: create project + datastore dir for aggregate_source integration tests."""
    datastore = tmp_path / "datastore"
    (datastore / source_key).mkdir(parents=True)
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
    return datastore / source_key


def test_aggregate_source_invokes_pre_aggregate_hook(tmp_path, tiny_fabric):
    """pre_aggregate_hook must run before the missing-variable check."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    src_dir = _setup_aggregate_source_project(
        tmp_path, tiny_fabric, "gldas_noah_v21_monthly"
    )
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    # Raw NC lacks 'derived' — hook injects it. Missing-var check must be skipped.
    xr.Dataset(
        {"raw": (["time", "lat", "lon"], np.ones((12, 2, 2)))},
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "lat": ("lat", [0.25, 0.75], {"standard_name": "latitude"}),
            "lon": ("lon", [0.5, 1.5], {"standard_name": "longitude"}),
        },
    ).to_netcdf(src_dir / "gldas_noah_v21_monthly.nc")

    hook_calls = []

    def hook(ds):
        hook_calls.append(1)
        return ds.assign(derived=ds["raw"] * 2.0)

    adapter = SourceAdapter(
        source_key="gldas_noah_v21_monthly",
        output_name="gldas_agg.nc",
        variables=("derived",),
        raw_grid_variable="raw",
        files_glob="gldas_noah_v21_monthly*.nc",
        pre_aggregate_hook=hook,
    )

    fake_year_ds = xr.Dataset(
        {"derived": (["time", "hru_id"], np.ones((12, 4)) * 2.0)},
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "hru_id": [0, 1, 2, 3],
        },
    )
    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.catalog_source",
            return_value={"access": {"type": "local_nc"}},
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=_fake_weights(),
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            return_value=fake_year_ds,
        ),
    ):
        aggregate_source(
            adapter, fabric_path=tiny_fabric, id_col="hru_id", workdir=tmp_path
        )

    assert hook_calls, "pre_aggregate_hook was not invoked"
    per_year = (
        tmp_path
        / "data"
        / "aggregated"
        / "gldas_noah_v21_monthly"
        / "gldas_noah_v21_monthly_2000_agg.nc"
    )
    with xr.open_dataset(per_year) as written:
        assert "derived" in written.data_vars


def test_aggregate_source_invokes_post_aggregate_hook(tmp_path, tiny_fabric):
    """post_aggregate_hook must run on the concatenated dataset before write."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    src_dir = _setup_aggregate_source_project(tmp_path, tiny_fabric, "merra2")
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    xr.Dataset(
        {"a": (["time", "lat", "lon"], np.ones((12, 2, 2)))},
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "lat": ("lat", [0.25, 0.75], {"standard_name": "latitude"}),
            "lon": ("lon", [0.5, 1.5], {"standard_name": "longitude"}),
        },
    ).to_netcdf(src_dir / "merra2_2000_consolidated.nc")

    def post_hook(ds):
        return ds.rename({"a": "a_renamed"})

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=("a",),
        post_aggregate_hook=post_hook,
    )
    fake_year_ds = xr.Dataset(
        {"a": (["time", "hru_id"], np.ones((12, 4)))},
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "hru_id": [0, 1, 2, 3],
        },
    )
    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.catalog_source",
            return_value={"access": {"type": "local_nc"}},
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=_fake_weights(),
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            return_value=fake_year_ds,
        ),
    ):
        aggregate_source(
            adapter, fabric_path=tiny_fabric, id_col="hru_id", workdir=tmp_path
        )

    per_year = tmp_path / "data" / "aggregated" / "merra2" / "merra2_2000_agg.nc"
    with xr.open_dataset(per_year) as written:
        assert "a_renamed" in written.data_vars
        assert "a" not in written.data_vars


def test_aggregate_source_raises_on_grid_drift_across_files(tmp_path, tiny_fabric):
    """Differing grid shape across per-year files must fail before aggregation."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    src_dir = _setup_aggregate_source_project(tmp_path, tiny_fabric, "merra2")
    t0 = pd.date_range("2000-01-01", periods=12, freq="MS")
    t1 = pd.date_range("2001-01-01", periods=12, freq="MS")
    xr.Dataset(
        {"a": (["time", "lat", "lon"], np.ones((12, 2, 2)))},
        coords={
            "time": ("time", t0, {"standard_name": "time"}),
            "lat": ("lat", [0.25, 0.75], {"standard_name": "latitude"}),
            "lon": ("lon", [0.5, 1.5], {"standard_name": "longitude"}),
        },
    ).to_netcdf(src_dir / "merra2_2000_consolidated.nc")
    # Second year has a different grid shape (3x3 instead of 2x2).
    xr.Dataset(
        {"a": (["time", "lat", "lon"], np.ones((12, 3, 3)))},
        coords={
            "time": ("time", t1, {"standard_name": "time"}),
            "lat": ("lat", [0.2, 0.5, 0.8], {"standard_name": "latitude"}),
            "lon": ("lon", [0.2, 0.5, 0.8], {"standard_name": "longitude"}),
        },
    ).to_netcdf(src_dir / "merra2_2001_consolidated.nc")

    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=("a",)
    )
    with patch(
        "nhf_spatial_targets.aggregate._driver.catalog_source",
        return_value={"access": {"type": "local_nc"}},
    ):
        with pytest.raises(ValueError, match="grid shape drift"):
            aggregate_source(
                adapter, fabric_path=tiny_fabric, id_col="hru_id", workdir=tmp_path
            )


def test_aggregate_source_raises_when_raw_grid_variable_missing(tmp_path, tiny_fabric):
    """Hooked adapter whose raw_grid_variable is absent from the raw NC must
    raise — the prior silent-skip left the cross-year grid invariant un-enforced."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    src_dir = _setup_aggregate_source_project(tmp_path, tiny_fabric, "merra2")
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    xr.Dataset(
        {"raw_present": (["time", "lat", "lon"], np.ones((12, 2, 2)))},
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "lat": ("lat", [0.25, 0.75], {"standard_name": "latitude"}),
            "lon": ("lon", [0.5, 1.5], {"standard_name": "longitude"}),
        },
    ).to_netcdf(src_dir / "merra2_2000_consolidated.nc")

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=("derived",),
        raw_grid_variable="raw_missing",
        pre_aggregate_hook=lambda ds: ds.assign(derived=ds["raw_present"] * 2.0),
    )
    with patch(
        "nhf_spatial_targets.aggregate._driver.catalog_source",
        return_value={"access": {"type": "local_nc"}},
    ):
        with pytest.raises(ValueError, match="raw_grid_variable.*raw_missing"):
            aggregate_source(
                adapter, fabric_path=tiny_fabric, id_col="hru_id", workdir=tmp_path
            )


def test_aggregate_source_missing_var_check_scans_all_files(tmp_path, tiny_fabric):
    """Missing variable in any input file (not just files[0]) must raise."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    src_dir = _setup_aggregate_source_project(tmp_path, tiny_fabric, "merra2")
    t0 = pd.date_range("2000-01-01", periods=12, freq="MS")
    t1 = pd.date_range("2001-01-01", periods=12, freq="MS")
    # First file has both a and b; second file is missing b.
    xr.Dataset(
        {
            "a": (["time", "lat", "lon"], np.ones((12, 2, 2))),
            "b": (["time", "lat", "lon"], np.ones((12, 2, 2))),
        },
        coords={
            "time": ("time", t0, {"standard_name": "time"}),
            "lat": ("lat", [0.25, 0.75], {"standard_name": "latitude"}),
            "lon": ("lon", [0.5, 1.5], {"standard_name": "longitude"}),
        },
    ).to_netcdf(src_dir / "merra2_2000_consolidated.nc")
    xr.Dataset(
        {"a": (["time", "lat", "lon"], np.ones((12, 2, 2)))},
        coords={
            "time": ("time", t1, {"standard_name": "time"}),
            "lat": ("lat", [0.25, 0.75], {"standard_name": "latitude"}),
            "lon": ("lon", [0.5, 1.5], {"standard_name": "longitude"}),
        },
    ).to_netcdf(src_dir / "merra2_2001_consolidated.nc")

    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=("a", "b")
    )
    with patch(
        "nhf_spatial_targets.aggregate._driver.catalog_source",
        return_value={"access": {"type": "local_nc"}},
    ):
        with pytest.raises(ValueError, match="b.*missing.*2001"):
            aggregate_source(
                adapter, fabric_path=tiny_fabric, id_col="hru_id", workdir=tmp_path
            )


def test_attach_cf_global_attrs_appends_to_existing_history():
    """Existing history attr on consolidated NC must be preserved, not overwritten."""
    from nhf_spatial_targets.aggregate._driver import _attach_cf_global_attrs

    ds = xr.Dataset({"v": (["x"], [1.0])})
    ds.attrs["history"] = "2024-01-01: fetched from provider"
    _attach_cf_global_attrs(ds, "src", {"access": {}})
    assert "2024-01-01: fetched from provider" in ds.attrs["history"]
    assert "aggregated to HRU fabric" in ds.attrs["history"]


def test_find_time_coord_name_detects_non_standard_name():
    """Time dim named anything (not literally 'time') is found via axis='T'."""
    from nhf_spatial_targets.aggregate._driver import _find_time_coord_name

    times = pd.date_range("2000-01-01", periods=3, freq="MS")
    ds = xr.Dataset(
        {"v": (["valid_time", "y"], np.ones((3, 2)))},
        coords={
            "valid_time": ("valid_time", times, {"axis": "T"}),
            "y": [0.0, 1.0],
        },
    )
    assert _find_time_coord_name(ds) == "valid_time"


def test_aggregate_source_grid_drift_check_uses_cf_time_detection(
    tmp_path, tiny_fabric
):
    """Grid-drift check must exclude the CF-detected time dim regardless of name.

    Regression: if the filter used literal "time" / adapter.time_coord to
    exclude the time axis, a source whose time dim is named ``valid_time``
    (ERA5-Land) with different per-file timestep counts would be reported
    as a spurious grid shape drift.
    """
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    src_dir = _setup_aggregate_source_project(tmp_path, tiny_fabric, "merra2")
    t0 = pd.date_range("2000-01-01", periods=11, freq="MS")  # 11 timesteps
    t1 = pd.date_range("2001-01-01", periods=12, freq="MS")  # 12 timesteps
    # Same spatial grid (2x2), different time lengths, CF time named 'valid_time'.
    xr.Dataset(
        {"a": (["valid_time", "lat", "lon"], np.ones((11, 2, 2)))},
        coords={
            "valid_time": ("valid_time", t0, {"axis": "T"}),
            "lat": ("lat", [0.25, 0.75], {"standard_name": "latitude"}),
            "lon": ("lon", [0.5, 1.5], {"standard_name": "longitude"}),
        },
    ).to_netcdf(src_dir / "merra2_2000_consolidated.nc")
    xr.Dataset(
        {"a": (["valid_time", "lat", "lon"], np.ones((12, 2, 2)))},
        coords={
            "valid_time": ("valid_time", t1, {"axis": "T"}),
            "lat": ("lat", [0.25, 0.75], {"standard_name": "latitude"}),
            "lon": ("lon", [0.5, 1.5], {"standard_name": "longitude"}),
        },
    ).to_netcdf(src_dir / "merra2_2001_consolidated.nc")

    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=("a",)
    )

    def _year_ds(times):
        return xr.Dataset(
            {"a": (["valid_time", "hru_id"], np.ones((len(times), 4)))},
            coords={
                "valid_time": ("valid_time", times, {"axis": "T"}),
                "hru_id": [0, 1, 2, 3],
            },
        )

    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.catalog_source",
            return_value={"access": {"type": "local_nc"}},
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=_fake_weights(),
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            side_effect=[_year_ds(t0), _year_ds(t1)],
        ),
    ):
        # Must NOT raise "grid shape drift" — the name-based filter would have
        # incorrectly included valid_time in the shape tuple.
        aggregate_source(
            adapter, fabric_path=tiny_fabric, id_col="hru_id", workdir=tmp_path
        )


def test_aggregate_source_multi_year_output_files_order_and_period(
    tmp_path, tiny_fabric
):
    """Regression: two-year aggregation. Pins output_files ordering, verifies
    _derive_period straddles both years, and confirms per-year files land in
    the new per-source subdir with no consolidated file.

    Covers three integration seams at once:
      1. aggregate_source invokes the per-year loop in order.
      2. _derive_period uses first/last files (not first twice).
      3. manifest output_files is a list in year order.
    """
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    src_dir = _setup_aggregate_source_project(tmp_path, tiny_fabric, "merra2")
    t2000 = pd.date_range("2000-01-01", periods=12, freq="MS")
    t2001 = pd.date_range("2001-01-01", periods=12, freq="MS")
    for year, times in [(2000, t2000), (2001, t2001)]:
        xr.Dataset(
            {"a": (["time", "lat", "lon"], np.ones((12, 2, 2)))},
            coords={
                "time": ("time", times, {"standard_name": "time"}),
                "lat": ("lat", [0.25, 0.75], {"standard_name": "latitude"}),
                "lon": ("lon", [0.5, 1.5], {"standard_name": "longitude"}),
            },
        ).to_netcdf(src_dir / f"merra2_{year}_consolidated.nc")

    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=("a",)
    )

    def _year_ds(times):
        return xr.Dataset(
            {"a": (["time", "hru_id"], np.ones((len(times), 4)))},
            coords={
                "time": ("time", times, {"standard_name": "time"}),
                "hru_id": [0, 1, 2, 3],
            },
        )

    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.catalog_source",
            return_value={"access": {"type": "local_nc"}},
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=_fake_weights(),
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            side_effect=[_year_ds(t2000), _year_ds(t2001)],
        ),
    ):
        aggregate_source(
            adapter, fabric_path=tiny_fabric, id_col="hru_id", workdir=tmp_path
        )

    agg_dir = tmp_path / "data" / "aggregated"
    assert (agg_dir / "merra2" / "merra2_2000_agg.nc").exists()
    assert (agg_dir / "merra2" / "merra2_2001_agg.nc").exists()
    assert not (agg_dir / "merra2_agg.nc").exists()  # no consolidated file

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["sources"]["merra2"]
    assert entry["output_files"] == [
        "data/aggregated/merra2/merra2_2000_agg.nc",
        "data/aggregated/merra2/merra2_2001_agg.nc",
    ]
    # Period must span both years: first-of-first .. last-of-last.
    assert entry["period"].startswith("2000-01-01/")
    assert entry["period"].endswith("/2001-12-01")


def test_aggregate_source_surfaces_year_coverage_gap(tmp_path, tiny_fabric):
    """Regression: _verify_year_coverage must fire when a year is missing.
    Pins the integration seam between the per-year loop and the coverage
    check — e.g., a future refactor that silences the check or reorders it
    would slip through other tests."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    src_dir = _setup_aggregate_source_project(tmp_path, tiny_fabric, "merra2")
    t2000 = pd.date_range("2000-01-01", periods=12, freq="MS")
    t2002 = pd.date_range("2002-01-01", periods=12, freq="MS")
    # 2001 skipped on purpose.
    for year, times in [(2000, t2000), (2002, t2002)]:
        xr.Dataset(
            {"a": (["time", "lat", "lon"], np.ones((12, 2, 2)))},
            coords={
                "time": ("time", times, {"standard_name": "time"}),
                "lat": ("lat", [0.25, 0.75], {"standard_name": "latitude"}),
                "lon": ("lon", [0.5, 1.5], {"standard_name": "longitude"}),
            },
        ).to_netcdf(src_dir / f"merra2_{year}_consolidated.nc")

    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=("a",)
    )

    def _year_ds(times):
        return xr.Dataset(
            {"a": (["time", "hru_id"], np.ones((len(times), 4)))},
            coords={
                "time": ("time", times, {"standard_name": "time"}),
                "hru_id": [0, 1, 2, 3],
            },
        )

    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.catalog_source",
            return_value={"access": {"type": "local_nc"}},
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=_fake_weights(),
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            side_effect=[_year_ds(t2000), _year_ds(t2002)],
        ),
    ):
        with pytest.raises(ValueError, match=r"missing=\[2001\]"):
            aggregate_source(
                adapter,
                fabric_path=tiny_fabric,
                id_col="hru_id",
                workdir=tmp_path,
            )

    # Per-year files are on disk from the per-year loop (that's fine); the
    # manifest must NOT record this source as complete because the coverage
    # check raised before update_manifest ran.
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "merra2" not in manifest["sources"]


def test_aggregate_source_migrates_legacy_layout_on_startup(tmp_path, tiny_fabric):
    """Regression: aggregate_source must call _migrate_legacy_layout before
    the per-year loop. A legacy _by_year/ file and a stale consolidated file
    must both be handled on the next aggregation pass, reusing the moved file
    rather than re-aggregating."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    src_dir = _setup_aggregate_source_project(tmp_path, tiny_fabric, "merra2")
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    xr.Dataset(
        {"a": (["time", "lat", "lon"], np.ones((12, 2, 2)))},
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "lat": ("lat", [0.25, 0.75], {"standard_name": "latitude"}),
            "lon": ("lon", [0.5, 1.5], {"standard_name": "longitude"}),
        },
    ).to_netcdf(src_dir / "merra2_2000_consolidated.nc")

    # Seed legacy state: a _by_year intermediate and a stale consolidated NC.
    legacy_by_year = tmp_path / "data" / "aggregated" / "_by_year"
    legacy_by_year.mkdir(parents=True, exist_ok=True)
    legacy_file = legacy_by_year / "merra2_2000_agg.nc"
    # Write a valid per-year shape so aggregate_year's idempotent skip fires.
    xr.Dataset(
        {"a": (["time", "hru_id"], np.ones((12, 4)))},
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "hru_id": [0, 1, 2, 3],
        },
    ).to_netcdf(legacy_file)
    stale = tmp_path / "data" / "aggregated" / "merra2_agg.nc"
    stale.write_bytes(b"placeholder")

    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=("a",)
    )
    with patch(
        "nhf_spatial_targets.aggregate._driver.catalog_source",
        return_value={"access": {"type": "local_nc"}},
    ):
        aggregate_source(
            adapter, fabric_path=tiny_fabric, id_col="hru_id", workdir=tmp_path
        )

    # Legacy location emptied; stale consolidated removed; canonical file lives.
    assert not legacy_file.exists()
    assert not stale.exists()
    canonical = tmp_path / "data" / "aggregated" / "merra2" / "merra2_2000_agg.nc"
    assert canonical.exists()
