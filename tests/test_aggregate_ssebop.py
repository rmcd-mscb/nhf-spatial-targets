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


def _make_mock_agg_result(hru_ids, n_times=2):
    """Build a fake (gdf, Dataset) return from AggGen.calculate_agg."""
    times = pd.date_range("2000-01-01", periods=n_times, freq="MS")
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
def test_aggregate_produces_dataset(
    mock_stac_data,
    mock_agg_gen,
    mock_weight_gen,
    mock_get_col,
    workdir,
    tiny_fabric,
):
    """Full orchestration with mocked gdptools returns a valid Dataset."""
    mock_get_col.return_value = MagicMock()

    mock_wg_instance = MagicMock()
    mock_wg_instance.calculate_weights.return_value = pd.DataFrame(
        {"src_idx": [0, 1], "tgt_idx": [0, 1], "weight": [0.5, 0.5]}
    )
    mock_weight_gen.return_value = mock_wg_instance

    mock_agg_instance = MagicMock()
    mock_agg_instance.calculate_agg.return_value = _make_mock_agg_result([0, 1, 2, 3])
    mock_agg_gen.return_value = mock_agg_instance

    ds = aggregate_ssebop(
        fabric_path=tiny_fabric,
        id_col="hru_id",
        period="2000/2000",
        workdir=workdir,
    )

    assert isinstance(ds, xr.Dataset)
    assert "et" in ds.data_vars
    assert "time" in ds.dims
    weight_files = list((workdir / "weights").glob("ssebop_batch*.csv"))
    assert len(weight_files) >= 1


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
    mock_agg_instance.calculate_agg.return_value = _make_mock_agg_result([0, 1, 2, 3])
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
    mock_agg_instance.calculate_agg.return_value = _make_mock_agg_result([0, 1, 2, 3])
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


@pytest.mark.integration
def test_integration_tiny_fabric(workdir, tiny_fabric):
    """End-to-end test with real STAC endpoint (4 HRUs, 1 year)."""
    ds = aggregate_ssebop(
        fabric_path=tiny_fabric,
        id_col="hru_id",
        period="2020/2020",
        workdir=workdir,
    )
    assert isinstance(ds, xr.Dataset)
    assert "et" in ds.data_vars
    assert ds["et"].shape[0] > 0
