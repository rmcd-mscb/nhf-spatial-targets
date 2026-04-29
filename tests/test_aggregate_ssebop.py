"""Tests for SSEBop aggregation orchestration."""

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

from nhf_spatial_targets.aggregate.ssebop import aggregate_ssebop


@pytest.fixture()
def workdir(tmp_path):
    """Create a minimal workspace for aggregation tests."""
    (tmp_path / "weights").mkdir()
    datastore = tmp_path / "datastore"
    datastore.mkdir()
    config = {
        "fabric": {"path": "", "id_col": "hru_id"},
        "datastore": str(datastore),
        "dir_mode": "2775",
    }
    (tmp_path / "config.yml").write_text(yaml.dump(config))
    fabric_meta = {"sha256": "abc123", "id_col": "hru_id"}
    (tmp_path / "fabric.json").write_text(json.dumps(fabric_meta))
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    (tmp_path / "data" / "aggregated").mkdir(parents=True)
    return tmp_path


@pytest.fixture()
def tiny_fabric(tmp_path):
    """Create a 4-polygon fabric GeoPackage."""
    polys = [box(i, 0, i + 1, 1) for i in range(4)]
    gdf = gpd.GeoDataFrame(
        {"hru_id": range(4)},
        geometry=polys,
        crs="EPSG:4326",
    )
    path = tmp_path / "fabric.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def _make_mock_agg_result(hru_ids, year=2000, n_times=12):
    """Build a fake (gdf, Dataset) return from AggGen.calculate_agg."""
    times = pd.date_range(f"{year}-01-01", periods=n_times, freq="MS")
    data = np.random.default_rng(42).random((n_times, len(hru_ids)))
    ds = xr.Dataset(
        {"et": (["time", "hru_id"], data)},
        coords={"time": times, "hru_id": hru_ids},
    )
    gdf = gpd.GeoDataFrame({"hru_id": hru_ids})
    return gdf, ds


@patch("nhf_spatial_targets.aggregate.ssebop.get_stac_collection")
@patch("nhf_spatial_targets.aggregate.ssebop.WeightGen")
@patch("nhf_spatial_targets.aggregate.ssebop.AggGen")
@patch("nhf_spatial_targets.aggregate.ssebop.NHGFStacZarrData")
def test_aggregate_writes_per_year_nc(
    mock_stac_data,
    mock_agg_gen,
    mock_weight_gen,
    mock_get_col,
    workdir,
    tiny_fabric,
):
    """A per-year NC should land in data/aggregated/ssebop/ and weights in weights/."""
    mock_get_col.return_value = MagicMock()

    mock_wg_instance = MagicMock()
    mock_wg_instance.calculate_weights.return_value = pd.DataFrame(
        {"src_idx": [0, 1], "tgt_idx": [0, 1], "weight": [0.5, 0.5]}
    )
    mock_weight_gen.return_value = mock_wg_instance

    mock_agg_instance = MagicMock()
    mock_agg_instance.calculate_agg.return_value = _make_mock_agg_result(
        [0, 1, 2, 3], year=2000
    )
    mock_agg_gen.return_value = mock_agg_instance

    aggregate_ssebop(
        fabric_path=tiny_fabric,
        id_col="hru_id",
        period="2000/2000",
        workdir=workdir,
    )

    out_path = workdir / "data" / "aggregated" / "ssebop" / "ssebop_2000_agg.nc"
    assert out_path.is_file()
    with xr.open_dataset(out_path) as ds:
        assert "et" in ds.data_vars
        assert "time" in ds.dims
        assert ds.sizes["hru_id"] == 4

    weight_files = list((workdir / "weights").glob("ssebop_batch*.csv"))
    assert len(weight_files) >= 1


@patch("nhf_spatial_targets.aggregate.ssebop.get_stac_collection")
@patch("nhf_spatial_targets.aggregate.ssebop.WeightGen")
@patch("nhf_spatial_targets.aggregate.ssebop.AggGen")
@patch("nhf_spatial_targets.aggregate.ssebop.NHGFStacZarrData")
def test_multi_year_emits_one_nc_per_year(
    mock_stac_data,
    mock_agg_gen,
    mock_weight_gen,
    mock_get_col,
    workdir,
    tiny_fabric,
):
    """A 3-year period should emit 3 per-year NCs and one consolidated manifest entry."""
    mock_get_col.return_value = MagicMock()

    mock_wg_instance = MagicMock()
    mock_wg_instance.calculate_weights.return_value = pd.DataFrame(
        {"src_idx": [0, 1], "tgt_idx": [0, 1], "weight": [0.5, 0.5]}
    )
    mock_weight_gen.return_value = mock_wg_instance

    # AggGen.calculate_agg is called once per (year, batch). With batch_size
    # default, all 4 polygons fall in one batch — so 3 calls total.
    call_count = {"i": 0}

    def _fake_calc_agg():
        year = 2000 + call_count["i"]
        call_count["i"] += 1
        return _make_mock_agg_result([0, 1, 2, 3], year=year)

    mock_agg_instance = MagicMock()
    mock_agg_instance.calculate_agg.side_effect = _fake_calc_agg
    mock_agg_gen.return_value = mock_agg_instance

    aggregate_ssebop(
        fabric_path=tiny_fabric,
        id_col="hru_id",
        period="2000/2002",
        workdir=workdir,
    )

    per_source_dir = workdir / "data" / "aggregated" / "ssebop"
    yearly = sorted(per_source_dir.glob("ssebop_*_agg.nc"))
    assert [p.name for p in yearly] == [
        "ssebop_2000_agg.nc",
        "ssebop_2001_agg.nc",
        "ssebop_2002_agg.nc",
    ]

    manifest = json.loads((workdir / "manifest.json").read_text())
    entry = manifest["sources"]["ssebop"]
    assert entry["output_files"] == [
        "data/aggregated/ssebop/ssebop_2000_agg.nc",
        "data/aggregated/ssebop/ssebop_2001_agg.nc",
        "data/aggregated/ssebop/ssebop_2002_agg.nc",
    ]


@patch("nhf_spatial_targets.aggregate.ssebop.get_stac_collection")
@patch("nhf_spatial_targets.aggregate.ssebop.WeightGen")
@patch("nhf_spatial_targets.aggregate.ssebop.AggGen")
@patch("nhf_spatial_targets.aggregate.ssebop.NHGFStacZarrData")
def test_existing_per_year_nc_is_skipped(
    mock_stac_data,
    mock_agg_gen,
    mock_weight_gen,
    mock_get_col,
    workdir,
    tiny_fabric,
):
    """Per-year files that already exist should not trigger AggGen."""
    mock_get_col.return_value = MagicMock()

    # Pre-seed the per-year NC for 2000.
    seeded_dir = workdir / "data" / "aggregated" / "ssebop"
    seeded_dir.mkdir(parents=True, exist_ok=True)
    _gdf, ds_seed = _make_mock_agg_result([0, 1, 2, 3], year=2000)
    ds_seed.to_netcdf(seeded_dir / "ssebop_2000_agg.nc")

    mock_wg_instance = MagicMock()
    mock_wg_instance.calculate_weights.return_value = pd.DataFrame(
        {"src_idx": [0, 1], "tgt_idx": [0, 1], "weight": [0.5, 0.5]}
    )
    mock_weight_gen.return_value = mock_wg_instance

    mock_agg_instance = MagicMock()
    mock_agg_instance.calculate_agg.return_value = _make_mock_agg_result(
        [0, 1, 2, 3], year=2000
    )
    mock_agg_gen.return_value = mock_agg_instance

    aggregate_ssebop(
        fabric_path=tiny_fabric,
        id_col="hru_id",
        period="2000/2000",
        workdir=workdir,
    )

    # The file existed, so the heavy machinery should not have been hit.
    mock_agg_gen.assert_not_called()
    mock_weight_gen.assert_not_called()


@patch("nhf_spatial_targets.aggregate.ssebop.get_stac_collection")
@patch("nhf_spatial_targets.aggregate.ssebop.WeightGen")
@patch("nhf_spatial_targets.aggregate.ssebop.AggGen")
@patch("nhf_spatial_targets.aggregate.ssebop.NHGFStacZarrData")
def test_legacy_consolidated_file_is_removed(
    mock_stac_data,
    mock_agg_gen,
    mock_weight_gen,
    mock_get_col,
    workdir,
    tiny_fabric,
):
    """Stale data/aggregated/ssebop_agg_aet.nc should be deleted on first run."""
    mock_get_col.return_value = MagicMock()

    legacy = workdir / "data" / "aggregated" / "ssebop_agg_aet.nc"
    legacy.write_bytes(b"legacy")
    assert legacy.is_file()

    mock_wg_instance = MagicMock()
    mock_wg_instance.calculate_weights.return_value = pd.DataFrame(
        {"src_idx": [0, 1], "tgt_idx": [0, 1], "weight": [0.5, 0.5]}
    )
    mock_weight_gen.return_value = mock_wg_instance

    mock_agg_instance = MagicMock()
    mock_agg_instance.calculate_agg.return_value = _make_mock_agg_result(
        [0, 1, 2, 3], year=2000
    )
    mock_agg_gen.return_value = mock_agg_instance

    aggregate_ssebop(
        fabric_path=tiny_fabric,
        id_col="hru_id",
        period="2000/2000",
        workdir=workdir,
    )

    assert not legacy.exists()


@patch("nhf_spatial_targets.aggregate.ssebop.get_stac_collection")
@patch("nhf_spatial_targets.aggregate.ssebop.WeightGen")
@patch("nhf_spatial_targets.aggregate.ssebop.AggGen")
@patch("nhf_spatial_targets.aggregate.ssebop.NHGFStacZarrData")
def test_cached_weights_skips_recompute(
    mock_stac_data,
    mock_agg_gen,
    mock_weight_gen,
    mock_get_col,
    workdir,
    tiny_fabric,
):
    """When weight CSV exists, WeightGen should not be called."""
    mock_get_col.return_value = MagicMock()

    weights_df = pd.DataFrame(
        {"src_idx": [0, 1], "tgt_idx": [0, 1], "weight": [0.5, 0.5]}
    )
    weights_df.to_csv(workdir / "weights" / "ssebop_batch0.csv", index=False)

    mock_agg_instance = MagicMock()
    mock_agg_instance.calculate_agg.return_value = _make_mock_agg_result(
        [0, 1, 2, 3], year=2000
    )
    mock_agg_gen.return_value = mock_agg_instance

    aggregate_ssebop(
        fabric_path=tiny_fabric,
        id_col="hru_id",
        period="2000/2000",
        workdir=workdir,
    )

    mock_weight_gen.assert_not_called()


@patch("nhf_spatial_targets.aggregate.ssebop.get_stac_collection")
@patch("nhf_spatial_targets.aggregate.ssebop.WeightGen")
@patch("nhf_spatial_targets.aggregate.ssebop.AggGen")
@patch("nhf_spatial_targets.aggregate.ssebop.NHGFStacZarrData")
def test_manifest_updated(
    mock_stac_data,
    mock_agg_gen,
    mock_weight_gen,
    mock_get_col,
    workdir,
    tiny_fabric,
):
    """Manifest should be updated with SSEBop provenance after aggregation."""
    mock_get_col.return_value = MagicMock()

    mock_wg_instance = MagicMock()
    mock_wg_instance.calculate_weights.return_value = pd.DataFrame(
        {"src_idx": [0, 1], "tgt_idx": [0, 1], "weight": [0.5, 0.5]}
    )
    mock_weight_gen.return_value = mock_wg_instance

    mock_agg_instance = MagicMock()
    mock_agg_instance.calculate_agg.return_value = _make_mock_agg_result(
        [0, 1, 2, 3], year=2000
    )
    mock_agg_gen.return_value = mock_agg_instance

    aggregate_ssebop(
        fabric_path=tiny_fabric,
        id_col="hru_id",
        period="2000/2000",
        workdir=workdir,
    )

    manifest = json.loads((workdir / "manifest.json").read_text())
    assert "ssebop" in manifest["sources"]
    entry = manifest["sources"]["ssebop"]
    assert entry["access_type"] == "usgs_gdp_stac"
    assert entry["collection_id"] == "ssebopeta_monthly"
    assert entry["doi"] == "10.5066/P9L2YMV"
    assert entry["output_files"] == ["data/aggregated/ssebop/ssebop_2000_agg.nc"]


@pytest.mark.parametrize("bad", ["2000", "2000-2001", "2000/2001/2002", "abcd/efgh"])
def test_invalid_period_raises(workdir, tiny_fabric, bad):
    """Period must be 'YYYY/YYYY'."""
    with pytest.raises(ValueError, match="period"):
        aggregate_ssebop(
            fabric_path=tiny_fabric,
            id_col="hru_id",
            period=bad,
            workdir=workdir,
        )


@pytest.mark.integration
def test_integration_tiny_fabric(workdir, tiny_fabric):
    """End-to-end test with real STAC endpoint (4 HRUs, 1 year)."""
    aggregate_ssebop(
        fabric_path=tiny_fabric,
        id_col="hru_id",
        period="2020/2020",
        workdir=workdir,
    )
    out_path = workdir / "data" / "aggregated" / "ssebop" / "ssebop_2020_agg.nc"
    assert out_path.is_file()
    with xr.open_dataset(out_path) as ds:
        assert "et" in ds.data_vars
        assert ds["et"].shape[0] > 0
