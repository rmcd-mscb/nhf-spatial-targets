"""KD-tree recursive bisection to partition fabric HRUs into spatial batches."""

from __future__ import annotations

import logging
import warnings
from math import ceil, log2

import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)


def _recursive_bisect(
    centroids: np.ndarray,
    indices: np.ndarray,
    depth: int,
    max_depth: int,
    min_batch_size: int,
) -> list[np.ndarray]:
    """Recursively bisect indices by alternating x/y median splits.

    Parameters
    ----------
    centroids
        Shape (N, 2) array of centroid coordinates (x, y).
    indices
        1-D integer array of indices into the original GeoDataFrame.
    depth
        Current recursion depth.
    max_depth
        Maximum recursion depth.
    min_batch_size
        Stop splitting when a partition has fewer than this many features.

    Returns
    -------
    list[np.ndarray]
        List of index arrays, one per leaf batch.
    """
    if depth >= max_depth or len(indices) <= min_batch_size:
        return [indices]

    # Alternate splitting axis: 0 = x, 1 = y
    axis = depth % 2
    coords = centroids[indices, axis]
    median = np.median(coords)

    left_mask = coords <= median
    right_mask = ~left_mask

    # If all points fall on one side, don't split (degenerate case)
    if left_mask.all() or right_mask.all():
        return [indices]

    left_indices = indices[left_mask]
    right_indices = indices[right_mask]

    left_batches = _recursive_bisect(
        centroids, left_indices, depth + 1, max_depth, min_batch_size
    )
    right_batches = _recursive_bisect(
        centroids, right_indices, depth + 1, max_depth, min_batch_size
    )
    return left_batches + right_batches


def spatial_batch(gdf: gpd.GeoDataFrame, batch_size: int = 500) -> gpd.GeoDataFrame:
    """Partition fabric HRUs into spatially contiguous batches.

    Uses KD-tree-style recursive bisection of centroid coordinates to
    create batches of approximately ``batch_size`` features each.

    Parameters
    ----------
    gdf
        GeoDataFrame of HRU polygons.
    batch_size
        Target number of features per batch.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of *gdf* with an added ``batch_id`` integer column.
    """
    result = gdf.copy()

    if len(result) == 0:
        result["batch_id"] = np.array([], dtype=int)
        return result

    if len(result) <= batch_size:
        result["batch_id"] = 0
        return result

    # Compute centroids; suppress warning for geographic CRS
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Geometry is in a geographic CRS.*",
            category=UserWarning,
        )
        centroids_geom = result.geometry.centroid

    centroids = np.column_stack([centroids_geom.x, centroids_geom.y])
    n_features = len(result)
    max_depth = ceil(log2(n_features / batch_size))
    min_batch_size = batch_size // 2
    indices = np.arange(n_features)

    batches = _recursive_bisect(centroids, indices, 0, max_depth, min_batch_size)

    batch_ids = np.empty(n_features, dtype=int)
    for batch_id, batch_indices in enumerate(batches):
        batch_ids[batch_indices] = batch_id

    result["batch_id"] = batch_ids

    sizes = [len(b) for b in batches]
    logger.info(
        "Partitioned %d features into %d batches (min=%d, max=%d, mean=%.0f)",
        n_features,
        len(batches),
        min(sizes),
        max(sizes),
        np.mean(sizes),
    )

    return result
