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
    assert adapter.x_coord is None
    assert adapter.y_coord is None
    assert adapter.time_coord is None
    assert adapter.open_hook is None
    assert adapter.pre_aggregate_hook is None
    assert adapter.post_aggregate_hook is None


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
            period=("2000-01-01", "2000-02-01"),
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
    src_nc = datastore / "merra2" / "merra2_consolidated.nc"
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
        x_coord="lon",
        y_coord="lat",
        time_coord="time",
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
    ).to_netcdf(datastore / "merra2" / "merra2_consolidated.nc")

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


def test_default_open_hook_single_consolidated_nc(tmp_path):
    """Single *_consolidated.nc file is opened and returned with time coord."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import _default_open_hook

    src_dir = tmp_path / "datastore" / "merra2"
    src_dir.mkdir(parents=True)
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    xr.Dataset(
        {"GWETTOP": (["time", "lat", "lon"], np.ones((2, 2, 2)))},
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    ).to_netcdf(src_dir / "merra2_consolidated.nc")
    # Non-consolidated file should be ignored
    (src_dir / "stale.nc").write_bytes(b"not a real nc")

    class _FakeProject:
        def raw_dir(self, key):
            return src_dir

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP"],
    )
    result = _default_open_hook(_FakeProject(), adapter)
    assert "time" in result.coords
    assert len(result["time"]) == 2


def test_default_open_hook_concatenates_multiple_consolidated_ncs(tmp_path):
    """Multiple *_consolidated.nc files (one per year) are concatenated on time."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import _default_open_hook

    src_dir = tmp_path / "datastore" / "merra2"
    src_dir.mkdir(parents=True)
    for year, t0 in ((2000, "2000-01-01"), (2001, "2001-01-01")):
        times = pd.date_range(t0, periods=2, freq="MS")
        xr.Dataset(
            {"GWETTOP": (["time", "lat", "lon"], np.ones((2, 2, 2)))},
            coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
        ).to_netcdf(src_dir / f"merra2_{year}_consolidated.nc")

    class _FakeProject:
        def raw_dir(self, key):
            return src_dir

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP"],
    )
    result = _default_open_hook(_FakeProject(), adapter)
    assert "time" in result.coords
    assert len(result["time"]) == 4
    assert result["time"].values[0] < result["time"].values[-1]
    # Both input years must be represented (catches a regression that drops
    # a file silently); use pandas to extract the year from the time coord.
    years_present = set(pd.DatetimeIndex(result["time"].values).year)
    assert years_present == {2000, 2001}


def test_default_open_hook_concat_sorts_when_filename_order_disagrees_with_time(
    tmp_path,
):
    """sortby('time') must run even when filename glob order is non-chronological."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import _default_open_hook

    src_dir = tmp_path / "datastore" / "merra2"
    src_dir.mkdir(parents=True)

    # Filenames sort partA < partB lexically, but partA covers 2001 and
    # partB covers 2000 — chronological order disagrees with glob order.
    times_partA = pd.date_range("2001-01-01", periods=2, freq="MS")
    times_partB = pd.date_range("2000-01-01", periods=2, freq="MS")
    for name, times in (
        ("merra2_partA_consolidated.nc", times_partA),
        ("merra2_partB_consolidated.nc", times_partB),
    ):
        xr.Dataset(
            {"GWETTOP": (["time"], np.ones(2))},
            coords={"time": times},
        ).to_netcdf(src_dir / name)

    class _FakeProject:
        def raw_dir(self, key):
            return src_dir

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP"],
    )
    result = _default_open_hook(_FakeProject(), adapter)
    times = pd.DatetimeIndex(result["time"].values)
    assert list(times.year) == [2000, 2000, 2001, 2001]


def test_default_open_hook_returned_dataset_detached_from_disk(tmp_path):
    """Returned Dataset must be in-memory; deleting source NCs must not break it.

    Guards against a future refactor that drops .load() and returns a lazy
    view tied to the on-disk handle (per memory feedback_rioxarray_close.md).
    """
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import _default_open_hook

    src_dir = tmp_path / "datastore" / "merra2"
    src_dir.mkdir(parents=True)
    times = pd.date_range("2001-01-01", periods=3, freq="MS")
    nc_path = src_dir / "merra2_2001_consolidated.nc"
    xr.Dataset(
        {"GWETTOP": (["time"], np.array([0.1, 0.2, 0.3]))},
        coords={"time": times},
    ).to_netcdf(nc_path)

    class _FakeProject:
        def raw_dir(self, key):
            return src_dir

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP"],
    )
    result = _default_open_hook(_FakeProject(), adapter)
    nc_path.unlink()  # delete source file; the in-memory copy must survive
    np.testing.assert_array_equal(result["GWETTOP"].values, [0.1, 0.2, 0.3])


def test_default_open_hook_single_non_consolidated_nc(tmp_path):
    """Single *.nc with no _consolidated suffix is opened directly (watergap22d pattern)."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import _default_open_hook

    src_dir = tmp_path / "datastore" / "watergap22d"
    src_dir.mkdir(parents=True)
    times = pd.date_range("1901-01-01", periods=3, freq="MS")
    xr.Dataset(
        {"qrdif": (["time", "lat", "lon"], np.ones((3, 2, 2)))},
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    ).to_netcdf(src_dir / "watergap22d_qrdif_cf.nc")

    class _FakeProject:
        def raw_dir(self, key):
            return src_dir

    adapter = SourceAdapter(
        source_key="watergap22d",
        output_name="watergap22d_agg.nc",
        variables=["qrdif"],
    )
    result = _default_open_hook(_FakeProject(), adapter)
    assert "time" in result.coords
    assert len(result["time"]) == 3


def test_default_open_hook_raises_when_multiple_ncs_none_consolidated(tmp_path):
    """FileNotFoundError when multiple NCs exist but none match *_consolidated.nc."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import _default_open_hook

    src_dir = tmp_path / "datastore" / "merra2"
    src_dir.mkdir(parents=True)
    times = pd.date_range("2000-01-01", periods=1, freq="MS")
    for name in ("raw_2000.nc", "raw_2001.nc"):
        xr.Dataset(
            {"GWETTOP": (["time", "lat", "lon"], np.ones((1, 2, 2)))},
            coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
        ).to_netcdf(src_dir / name)

    class _FakeProject:
        def raw_dir(self, key):
            return src_dir

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP"],
    )
    with pytest.raises(FileNotFoundError, match="consolidated"):
        _default_open_hook(_FakeProject(), adapter)


def test_default_open_hook_raises_when_no_nc_at_all(tmp_path):
    """FileNotFoundError when the raw directory is empty."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import _default_open_hook

    src_dir = tmp_path / "datastore" / "merra2"
    src_dir.mkdir(parents=True)

    class _FakeProject:
        def raw_dir(self, key):
            return src_dir

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP"],
    )
    with pytest.raises(FileNotFoundError):
        _default_open_hook(_FakeProject(), adapter)


def test_default_open_hook_raises_on_duplicate_time_across_consolidated_ncs(tmp_path):
    """Overlapping per-year consolidated NCs must raise, not silently concat."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import _default_open_hook

    src_dir = tmp_path / "datastore" / "merra2"
    src_dir.mkdir(parents=True)

    # Two NCs whose time ranges overlap at 2001-06
    times_a = pd.date_range("2001-01-01", periods=12, freq="MS")
    times_b = pd.date_range("2001-06-01", periods=12, freq="MS")
    for name, times in (
        ("merra2_2001a_consolidated.nc", times_a),
        ("merra2_2001b_consolidated.nc", times_b),
    ):
        xr.Dataset(
            {"GWETTOP": (["time"], np.ones(12))},
            coords={"time": times},
        ).to_netcdf(src_dir / name)

    class _FakeProject:
        def raw_dir(self, key):
            return src_dir

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP"],
    )
    with pytest.raises(ValueError, match="Duplicate"):
        _default_open_hook(_FakeProject(), adapter)


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
            output_file="foo_agg.nc",
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
