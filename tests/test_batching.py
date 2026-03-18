"""Tests for spatial batching of HRU fabric."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from nhf_spatial_targets.aggregate.batching import spatial_batch


def _make_grid_fabric(n_cols: int, n_rows: int) -> gpd.GeoDataFrame:
    """Create a grid of box polygons for testing."""
    geoms = []
    for row in range(n_rows):
        for col in range(n_cols):
            geoms.append(box(col, row, col + 1, row + 1))
    return gpd.GeoDataFrame(
        {"nhm_id": range(len(geoms))},
        geometry=geoms,
        crs="EPSG:4326",
    )


def test_single_batch_when_all_fit():
    """25 features with batch_size=50 should produce 1 batch."""
    gdf = _make_grid_fabric(5, 5)
    result = spatial_batch(gdf, batch_size=50)
    assert "batch_id" in result.columns
    assert result["batch_id"].nunique() == 1
    assert (result["batch_id"] == 0).all()


def test_multiple_batches_created():
    """400 features with batch_size=50 should produce multiple batches."""
    gdf = _make_grid_fabric(20, 20)
    result = spatial_batch(gdf, batch_size=50)
    assert "batch_id" in result.columns
    assert result["batch_id"].nunique() > 1
    # Each batch should respect approximate size bounds
    for _, group in result.groupby("batch_id"):
        assert len(group) <= 100  # no batch bigger than 2x batch_size


def test_batches_are_spatially_contiguous():
    """Each batch bounding box should be less than 50% of total area."""
    gdf = _make_grid_fabric(20, 20)
    result = spatial_batch(gdf, batch_size=50)
    total_bounds = result.total_bounds  # minx, miny, maxx, maxy
    total_area = (total_bounds[2] - total_bounds[0]) * (
        total_bounds[3] - total_bounds[1]
    )
    for _, group in result.groupby("batch_id"):
        b = group.total_bounds
        batch_area = (b[2] - b[0]) * (b[3] - b[1])
        assert batch_area < 0.5 * total_area


def test_empty_geodataframe():
    """Empty GeoDataFrame should return empty with batch_id column."""
    gdf = gpd.GeoDataFrame(
        {"nhm_id": []},
        geometry=[],
        crs="EPSG:4326",
    )
    result = spatial_batch(gdf)
    assert "batch_id" in result.columns
    assert len(result) == 0


def test_degenerate_single_axis():
    """100 features along one axis with batch_size=20 should still split."""
    gdf = _make_grid_fabric(100, 1)
    result = spatial_batch(gdf, batch_size=20)
    assert "batch_id" in result.columns
    assert result["batch_id"].nunique() > 1


def test_preserves_original_columns():
    """Extra columns from the input GeoDataFrame should be preserved."""
    gdf = _make_grid_fabric(5, 5)
    gdf["extra_col"] = np.arange(len(gdf))
    gdf["name"] = [f"hru_{i}" for i in range(len(gdf))]
    result = spatial_batch(gdf, batch_size=50)
    assert "extra_col" in result.columns
    assert "name" in result.columns
    assert list(result["extra_col"]) == list(range(25))
