# SSEBop Aggregation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggregate SSEBop monthly AET from the USGS NHGF STAC catalog to HRU fabric polygons using gdptools, with spatially contiguous batching and weight caching.

**Architecture:** Remote-only aggregation via `NHGFStacZarrData` → `WeightGen` → `AggGen`. Fabric HRUs are partitioned into spatially contiguous batches using KD-tree recursive bisection. Per-batch weight CSVs are cached for reuse. No local fetch/download step.

**Tech Stack:** gdptools (NHGFStacZarrData, WeightGen, AggGen), geopandas, xarray, numpy, pystac

**Spec:** `docs/superpowers/specs/2026-03-17-ssebop-aggregation-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `catalog/sources.yml` | Update SSEBop entry to `usgs_gdp_stac` access type |
| Delete | `src/nhf_spatial_targets/fetch/ssebop.py` | Remove obsolete fetch stub |
| Create | `src/nhf_spatial_targets/aggregate/batching.py` | Spatial batching via KD-tree recursive bisection |
| Create | `src/nhf_spatial_targets/aggregate/ssebop.py` | SSEBop aggregation orchestrator |
| Modify | `src/nhf_spatial_targets/cli.py` | Add `agg` sub-app with `ssebop` command |
| Create | `tests/test_batching.py` | Spatial batching unit tests |
| Create | `tests/test_aggregate_ssebop.py` | Aggregation orchestration unit + integration tests |

---

## Task 1: Update catalog and delete fetch stub

**Files:**
- Modify: `catalog/sources.yml:88-106`
- Delete: `src/nhf_spatial_targets/fetch/ssebop.py`
- Modify: `tests/test_catalog.py`

- [ ] **Step 1: Update SSEBop entry in sources.yml**

Replace the existing `ssebop` block (lines 88–106) with:

```yaml
  ssebop:
    name: SSEBop Actual Evapotranspiration
    description: >
      Operational Simplified Surface Energy Balance (SSEBop) AET estimates.
      Used as one of three sources for AET calibration target range.
      Remote dataset accessed via USGS GDP STAC catalog (no local download).
    citations:
      - "Senay, G.B., and others, 2013"
    doi: "10.5066/P9L2YMV"
    access:
      type: usgs_gdp_stac
      collection_id: ssebopeta_monthly
      endpoint: https://usgs.osn.mghpcc.org/
      zarr_store: s3://mdmf/gdp/ssebopeta_monthly.zarr/
    variables:
      - actual_et
    time_step: monthly
    period: "2000/2023"
    spatial_extent: CONUS
    spatial_resolution: 1km
    units: mm/month
    status: current
```

- [ ] **Step 2: Add catalog test for SSEBop access type**

In `tests/test_catalog.py`, add:

```python
def test_ssebop_source_is_usgs_gdp_stac():
    s = source("ssebop")
    assert s["access"]["type"] == "usgs_gdp_stac"
    assert s["access"]["collection_id"] == "ssebopeta_monthly"
    assert s["status"] == "current"
```

- [ ] **Step 3: Run catalog tests**

Run: `pixi run -e dev test tests/test_catalog.py -v`
Expected: All tests pass, including the new one.

- [ ] **Step 4: Delete the fetch stub**

```bash
git rm src/nhf_spatial_targets/fetch/ssebop.py
```

- [ ] **Step 5: Commit**

```bash
git add catalog/sources.yml tests/test_catalog.py
git commit -m "Update SSEBop catalog to usgs_gdp_stac, remove fetch stub

SSEBop is a remote dataset accessed via NHGF STAC catalog — no local
download step needed. Updated access type, collection ID, DOI, and
period. Deleted the obsolete fetch/ssebop.py stub."
```

---

## Task 2: Spatial batching module

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/batching.py`
- Create: `tests/test_batching.py`

- [ ] **Step 1: Write failing tests for spatial_batch**

Create `tests/test_batching.py`:

```python
"""Tests for spatial batching via KD-tree recursive bisection."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from nhf_spatial_targets.aggregate.batching import spatial_batch


def _make_grid_fabric(n_cols: int, n_rows: int) -> gpd.GeoDataFrame:
    """Create a simple grid of square polygons for testing."""
    polys = []
    for r in range(n_rows):
        for c in range(n_cols):
            polys.append(box(c, r, c + 1, r + 1))
    return gpd.GeoDataFrame(
        {"hru_id": range(len(polys))},
        geometry=polys,
        crs="EPSG:4326",
    )


def test_single_batch_when_all_fit():
    gdf = _make_grid_fabric(5, 5)  # 25 features
    result = spatial_batch(gdf, batch_size=50)
    assert "batch_id" in result.columns
    assert result["batch_id"].nunique() == 1
    assert (result["batch_id"] == 0).all()


def test_multiple_batches_created():
    gdf = _make_grid_fabric(20, 20)  # 400 features
    result = spatial_batch(gdf, batch_size=50)
    assert "batch_id" in result.columns
    n_batches = result["batch_id"].nunique()
    assert n_batches > 1
    # Each feature assigned exactly once
    assert len(result) == 400


def test_batches_are_spatially_contiguous():
    gdf = _make_grid_fabric(20, 20)  # 400 features
    result = spatial_batch(gdf, batch_size=50)
    # For each batch, the bounding box area should be much smaller
    # than the total extent (20x20 = 400 sq units)
    total_area = 20 * 20
    for bid in result["batch_id"].unique():
        batch = result[result["batch_id"] == bid]
        bounds = batch.total_bounds  # minx, miny, maxx, maxy
        bbox_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        # Each batch bbox should be a fraction of total
        assert bbox_area < total_area * 0.5


def test_empty_geodataframe():
    gdf = gpd.GeoDataFrame(
        {"hru_id": []},
        geometry=[],
        crs="EPSG:4326",
    )
    result = spatial_batch(gdf, batch_size=50)
    assert "batch_id" in result.columns
    assert len(result) == 0


def test_degenerate_single_axis():
    """Features aligned along a single axis (all same y)."""
    polys = [box(i, 0, i + 1, 1) for i in range(100)]
    gdf = gpd.GeoDataFrame(
        {"hru_id": range(100)},
        geometry=polys,
        crs="EPSG:4326",
    )
    result = spatial_batch(gdf, batch_size=20)
    assert "batch_id" in result.columns
    assert len(result) == 100
    # Should still produce batches (splitting on x-axis works)
    assert result["batch_id"].nunique() >= 2


def test_preserves_original_columns():
    gdf = _make_grid_fabric(10, 10)
    gdf["extra_col"] = "hello"
    result = spatial_batch(gdf, batch_size=20)
    assert "extra_col" in result.columns
    assert "hru_id" in result.columns
    assert (result["extra_col"] == "hello").all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test tests/test_batching.py -v`
Expected: FAIL — `ImportError: cannot import name 'spatial_batch'`

- [ ] **Step 3: Implement spatial batching module**

Create `src/nhf_spatial_targets/aggregate/batching.py`:

```python
"""Spatial batching: assign features to spatially contiguous groups.

Group polygon features into spatially contiguous batches using KD-tree
recursive bisection. Each batch's bounding box is compact, which is
critical for efficient spatial subsetting when aggregating from remote
gridded data via gdptools.

Adapted from hydro-param (github.com/rmcd-mscb/hydro-param).
"""

from __future__ import annotations

import logging
import warnings

import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)


def _recursive_bisect(
    centroids: np.ndarray,
    indices: np.ndarray,
    depth: int = 0,
    max_depth: int = 7,
    min_batch_size: int = 50,
) -> list[np.ndarray]:
    """Recursively bisect features along alternating axes (KD-tree style).

    Parameters
    ----------
    centroids : np.ndarray
        Shape ``(N, 2)`` array of centroid coordinates (x, y).
    indices : np.ndarray
        1-D integer array of indices into the original GeoDataFrame.
    depth : int
        Current recursion depth (0 = split on x, 1 = y, ...).
    max_depth : int
        Maximum recursion depth. Produces up to ``2^max_depth`` batches.
    min_batch_size : int
        Stop splitting below this threshold.

    Returns
    -------
    list[np.ndarray]
        List of index arrays, one per batch.
    """
    if depth >= max_depth or len(indices) <= min_batch_size:
        return [indices]

    axis = depth % 2
    coords = centroids[indices, axis]
    median = np.median(coords)
    left_mask = coords <= median
    right_mask = ~left_mask

    if not left_mask.any() or not right_mask.any():
        return [indices]

    left = _recursive_bisect(
        centroids, indices[left_mask], depth + 1, max_depth, min_batch_size
    )
    right = _recursive_bisect(
        centroids, indices[right_mask], depth + 1, max_depth, min_batch_size
    )
    return left + right


def spatial_batch(
    gdf: gpd.GeoDataFrame,
    batch_size: int = 500,
) -> gpd.GeoDataFrame:
    """Assign spatially contiguous batch IDs via KD-tree recursive bisection.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Target fabric with polygon geometries.
    batch_size : int
        Target number of features per batch.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of input with a ``batch_id`` column added.
    """
    if gdf.empty:
        result = gdf.copy()
        result["batch_id"] = np.array([], dtype=int)
        return result

    if len(gdf) <= batch_size:
        result = gdf.copy()
        result["batch_id"] = 0
        logger.info(
            "Spatial batching: %d features -> 1 batch (all fit in batch_size=%d)",
            len(gdf),
            batch_size,
        )
        return result

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*geographic CRS.*centroid.*")
        centroids = np.column_stack(
            [gdf.geometry.centroid.x.values, gdf.geometry.centroid.y.values]
        )

    n_batches = max(1, len(gdf) // batch_size)
    max_depth = max(1, int(np.ceil(np.log2(n_batches))))

    batches = _recursive_bisect(
        centroids,
        np.arange(len(gdf)),
        max_depth=max_depth,
        min_batch_size=max(1, batch_size // 2),
    )

    batch_ids = np.empty(len(gdf), dtype=int)
    for batch_id, indices in enumerate(batches):
        batch_ids[indices] = batch_id

    result = gdf.copy()
    result["batch_id"] = batch_ids

    logger.info(
        "Spatial batching: %d features -> %d batches "
        "(target size=%d, actual range=%d-%d)",
        len(gdf),
        len(batches),
        batch_size,
        min(len(b) for b in batches),
        max(len(b) for b in batches),
    )

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev test tests/test_batching.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Run lint and format**

Run: `pixi run -e dev fmt && pixi run -e dev lint`

- [ ] **Step 6: Commit**

```bash
git add src/nhf_spatial_targets/aggregate/batching.py tests/test_batching.py
git commit -m "Add spatial batching module with KD-tree recursive bisection

Partitions fabric HRUs into spatially contiguous batches so that each
batch's bounding box is compact. Critical for memory-efficient remote
data access via gdptools.

Adapted from hydro-param (github.com/rmcd-mscb/hydro-param)."
```

---

## Task 3: SSEBop aggregation module

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/ssebop.py`
- Create: `tests/test_aggregate_ssebop.py`

- [ ] **Step 1: Write failing unit tests**

Create `tests/test_aggregate_ssebop.py`:

```python
"""Tests for SSEBop aggregation orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from nhf_spatial_targets.aggregate.ssebop import aggregate_ssebop


@pytest.fixture()
def run_dir(tmp_path):
    """Create a minimal run workspace."""
    (tmp_path / "weights").mkdir()
    (tmp_path / "output").mkdir()
    fabric_meta = {"sha256": "abc123", "id_col": "hru_id"}
    (tmp_path / "fabric.json").write_text(json.dumps(fabric_meta))
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
    mock_stac_data, mock_agg_gen, mock_weight_gen, mock_get_col,
    run_dir, tiny_fabric,
):
    """Full orchestration with mocked gdptools returns a valid Dataset."""
    mock_get_col.return_value = MagicMock()

    # WeightGen returns a DataFrame
    mock_wg_instance = MagicMock()
    mock_wg_instance.calculate_weights.return_value = pd.DataFrame(
        {"src_idx": [0, 1], "tgt_idx": [0, 1], "weight": [0.5, 0.5]}
    )
    mock_weight_gen.return_value = mock_wg_instance

    # AggGen returns (gdf, Dataset) for the single batch (4 HRUs < default batch_size)
    mock_agg_instance = MagicMock()
    mock_agg_instance.calculate_agg.return_value = _make_mock_agg_result([0, 1, 2, 3])
    mock_agg_gen.return_value = mock_agg_instance

    ds = aggregate_ssebop(
        fabric_path=tiny_fabric,
        id_col="hru_id",
        period="2000/2000",
        run_dir=run_dir,
    )

    assert isinstance(ds, xr.Dataset)
    assert "et" in ds.data_vars
    assert "time" in ds.dims
    # Should have written weights CSV
    weight_files = list((run_dir / "weights").glob("ssebop_batch*.csv"))
    assert len(weight_files) >= 1


@patch("nhf_spatial_targets.aggregate.ssebop.get_stac_collection")
@patch("nhf_spatial_targets.aggregate.ssebop.WeightGen")
@patch("nhf_spatial_targets.aggregate.ssebop.AggGen")
@patch("nhf_spatial_targets.aggregate.ssebop.NHGFStacZarrData")
def test_cached_weights_skips_recompute(
    mock_stac_data, mock_agg_gen, mock_weight_gen, mock_get_col,
    run_dir, tiny_fabric,
):
    """When weight CSV exists, WeightGen should not be called."""
    mock_get_col.return_value = MagicMock()

    # Pre-create cached weights
    weights_df = pd.DataFrame(
        {"src_idx": [0, 1], "tgt_idx": [0, 1], "weight": [0.5, 0.5]}
    )
    weights_df.to_csv(run_dir / "weights" / "ssebop_batch0.csv", index=False)

    mock_agg_instance = MagicMock()
    mock_agg_instance.calculate_agg.return_value = _make_mock_agg_result([0, 1, 2, 3])
    mock_agg_gen.return_value = mock_agg_instance

    aggregate_ssebop(
        fabric_path=tiny_fabric,
        id_col="hru_id",
        period="2000/2000",
        run_dir=run_dir,
    )

    # WeightGen should never have been instantiated
    mock_weight_gen.assert_not_called()


@patch("nhf_spatial_targets.aggregate.ssebop.get_stac_collection")
@patch("nhf_spatial_targets.aggregate.ssebop.WeightGen")
@patch("nhf_spatial_targets.aggregate.ssebop.AggGen")
@patch("nhf_spatial_targets.aggregate.ssebop.NHGFStacZarrData")
def test_manifest_updated(
    mock_stac_data, mock_agg_gen, mock_weight_gen, mock_get_col,
    run_dir, tiny_fabric,
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
        run_dir=run_dir,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "ssebop" in manifest["sources"]
    entry = manifest["sources"]["ssebop"]
    assert entry["access_type"] == "usgs_gdp_stac"
    assert entry["collection_id"] == "ssebopeta_monthly"
    assert entry["doi"] == "10.5066/P9L2YMV"


@pytest.mark.integration
def test_integration_tiny_fabric(run_dir, tiny_fabric):
    """End-to-end test with real STAC endpoint (2-3 HRUs, 1 month)."""
    ds = aggregate_ssebop(
        fabric_path=tiny_fabric,
        id_col="hru_id",
        period="2020/2020",
        run_dir=run_dir,
    )
    assert isinstance(ds, xr.Dataset)
    assert "et" in ds.data_vars
    assert ds["et"].shape[0] > 0  # has time steps
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test tests/test_aggregate_ssebop.py -v -m "not integration"`
Expected: FAIL — `ImportError: cannot import name 'aggregate_ssebop'`

- [ ] **Step 3: Implement the aggregation module**

Create `src/nhf_spatial_targets/aggregate/ssebop.py`:

```python
"""Aggregate SSEBop AET from NHGF STAC catalog to HRU fabric polygons."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
from gdptools import AggGen, NHGFStacZarrData, WeightGen
from gdptools.helpers import get_stac_collection

from nhf_spatial_targets import catalog
from nhf_spatial_targets.aggregate.batching import spatial_batch

logger = logging.getLogger(__name__)

_SOURCE_KEY = "ssebop"
_SOURCE_VAR = "et"
_WEIGHT_GEN_CRS = 5070  # NAD83 / CONUS Albers (EPSG:5070)


def _parse_period(period: str) -> list[str]:
    """Convert 'YYYY/YYYY' to ['YYYY-01-01', 'YYYY-12-31']."""
    parts = period.split("/")
    return [f"{parts[0]}-01-01", f"{parts[-1]}-12-31"]


def _weight_path(run_dir: Path, batch_id: int) -> Path:
    return run_dir / "weights" / f"ssebop_batch{batch_id}.csv"


def _process_batch(
    batch_gdf: gpd.GeoDataFrame,
    batch_id: int,
    collection,
    id_col: str,
    time_period: list[str],
    run_dir: Path,
) -> xr.Dataset:
    """Process a single spatial batch: compute/load weights, aggregate."""
    wp = _weight_path(run_dir, batch_id)

    if wp.exists():
        logger.info("Batch %d: loading cached weights from %s", batch_id, wp)
        weights = pd.read_csv(wp)
    else:
        logger.info("Batch %d: computing weights (%d HRUs)", batch_id, len(batch_gdf))
        stac_data = NHGFStacZarrData(
            source_collection=collection,
            source_var=_SOURCE_VAR,
            target_gdf=batch_gdf,
            target_id=id_col,
            source_time_period=time_period,
        )
        wg = WeightGen(
            user_data=stac_data,
            method="serial",
            weight_gen_crs=_WEIGHT_GEN_CRS,
        )
        weights = wg.calculate_weights()
        wp.parent.mkdir(parents=True, exist_ok=True)
        weights.to_csv(wp, index=False)
        logger.info("Batch %d: weights saved to %s", batch_id, wp)

    # Need a fresh NHGFStacZarrData for aggregation (WeightGen may have
    # consumed the prior one)
    stac_data = NHGFStacZarrData(
        source_collection=collection,
        source_var=_SOURCE_VAR,
        target_gdf=batch_gdf,
        target_id=id_col,
        source_time_period=time_period,
    )
    agg = AggGen(
        user_data=stac_data,
        stat_method="masked_mean",
        agg_engine="serial",
        agg_writer="none",
        weights=weights,
    )
    _gdf, ds = agg.calculate_agg()
    logger.info("Batch %d: aggregation complete", batch_id)
    return ds


def aggregate_ssebop(
    fabric_path: str | Path,
    id_col: str,
    period: str,
    run_dir: str | Path,
    batch_size: int = 500,
) -> xr.Dataset:
    """Aggregate SSEBop monthly AET to fabric HRU polygons.

    Parameters
    ----------
    fabric_path : str | Path
        Path to the HRU fabric GeoPackage.
    id_col : str
        Column name for HRU identifiers in the fabric.
    period : str
        Temporal range as 'YYYY/YYYY'.
    run_dir : str | Path
        Run workspace directory.
    batch_size : int
        Target number of HRUs per spatial batch.

    Returns
    -------
    xr.Dataset
        Aggregated SSEBop AET with dimensions (time, <id_col>).
    """
    run_dir = Path(run_dir)
    fabric_path = Path(fabric_path)

    # 1. Load source metadata and STAC collection
    meta = catalog.source(_SOURCE_KEY)
    collection_id = meta["access"]["collection_id"]
    logger.info("Resolving STAC collection: %s", collection_id)
    collection = get_stac_collection(collection_id)

    # 2. Load and batch the fabric
    logger.info("Loading fabric: %s", fabric_path)
    gdf = gpd.read_file(fabric_path)
    batched = spatial_batch(gdf, batch_size=batch_size)
    n_batches = batched["batch_id"].nunique()
    logger.info("Fabric split into %d spatial batches", n_batches)

    # 3. Process each batch
    time_period = _parse_period(period)
    datasets = []
    for bid in sorted(batched["batch_id"].unique()):
        batch_gdf = batched[batched["batch_id"] == bid].copy()
        batch_gdf = batch_gdf.drop(columns=["batch_id"])
        ds = _process_batch(
            batch_gdf, bid, collection, id_col, time_period, run_dir
        )
        datasets.append(ds)

    # 4. Concatenate batches
    combined = xr.concat(datasets, dim=id_col)
    logger.info(
        "Combined dataset: %s time steps x %s HRUs",
        combined.sizes.get("time", "?"),
        combined.sizes.get(id_col, "?"),
    )

    # 5. Write output
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ssebop_aet.nc"
    combined.to_netcdf(output_path)
    logger.info("Output written to %s", output_path)

    # 6. Update manifest
    _update_manifest(run_dir, period, meta, n_batches)

    return combined


def _update_manifest(
    run_dir: Path,
    period: str,
    meta: dict,
    n_batches: int,
) -> None:
    """Merge SSEBop aggregation provenance into manifest.json."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {run_dir} is corrupt: {exc}"
            ) from exc
    else:
        manifest = {"sources": {}, "steps": []}

    if "sources" not in manifest:
        manifest["sources"] = {}

    # Read fabric sha256 if available
    fabric_json = run_dir / "fabric.json"
    fabric_sha = ""
    if fabric_json.exists():
        fabric_meta = json.loads(fabric_json.read_text())
        fabric_sha = fabric_meta.get("sha256", "")

    access = meta["access"]
    time_period = _parse_period(period)
    weight_files = [
        str(Path("weights") / f"ssebop_batch{i}.csv") for i in range(n_batches)
    ]

    manifest["sources"][_SOURCE_KEY] = {
        "source_key": _SOURCE_KEY,
        "access_type": access["type"],
        "collection_id": access["collection_id"],
        "doi": meta.get("doi", ""),
        "period": f"{time_period[0]}/{time_period[1]}",
        "fabric_sha256": fabric_sha,
        "output_file": "output/ssebop_aet.nc",
        "weight_files": weight_files,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=manifest_path.parent, suffix=".json.tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp_path).replace(manifest_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    logger.info("Updated manifest.json with SSEBop aggregation provenance")
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `pixi run -e dev test tests/test_aggregate_ssebop.py -v -m "not integration"`
Expected: All 3 unit tests PASS.

- [ ] **Step 5: Run lint and format**

Run: `pixi run -e dev fmt && pixi run -e dev lint`

- [ ] **Step 6: Commit**

```bash
git add src/nhf_spatial_targets/aggregate/ssebop.py tests/test_aggregate_ssebop.py
git commit -m "Add SSEBop aggregation module

Orchestrates NHGFStacZarrData -> WeightGen -> AggGen per spatial batch.
Caches weight CSVs for reuse. Writes output NetCDF in native units
(mm/month). Updates manifest.json with provenance."
```

---

## Task 4: CLI integration

**Files:**
- Modify: `src/nhf_spatial_targets/cli.py`

- [ ] **Step 1: Add agg sub-app and ssebop command to cli.py**

After the `fetch_app` and `catalog_app` declarations (line 28), add:

```python
agg_app = App(name="agg", help="Aggregate source datasets to HRU fabric polygons.")
app.command(agg_app)
```

Then add the SSEBop aggregation command (after the last fetch command, before the catalog commands):

```python
@agg_app.command(name="ssebop")
def agg_ssebop_cmd(
    run_dir: Annotated[
        Path,
        Parameter(
            name=["--run-dir", "-r"],
            help="Run workspace created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
    id_col: Annotated[
        str,
        Parameter(name="--id-col", help="HRU ID column name in the fabric."),
    ] = "nhm_id",
    batch_size: Annotated[
        int,
        Parameter(name="--batch-size", help="Target HRUs per spatial batch."),
    ] = 500,
):
    """Aggregate SSEBop monthly AET to HRU fabric polygons.

    Reads SSEBop data from the USGS NHGF STAC catalog (Zarr), computes
    area-weighted means per HRU, and writes the result to NetCDF.
    """
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.aggregate.ssebop import aggregate_ssebop

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(2)

    # Read fabric path from the run workspace config
    config_path = run_dir / "config.yml"
    if not config_path.exists():
        print(f"Error: config.yml not found in {run_dir}", file=sys.stderr)
        sys.exit(2)
    cfg = yaml.safe_load(config_path.read_text())
    fabric_path = cfg["fabric"]["path"]

    console = Console()
    console.print(
        f"[bold]Aggregating SSEBop AET for period {period} "
        f"(batch_size={batch_size})...[/bold]"
    )

    try:
        ds = aggregate_ssebop(
            fabric_path=fabric_path,
            id_col=id_col,
            period=period,
            run_dir=run_dir,
            batch_size=batch_size,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during SSEBop aggregation")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print(
        f"[green]SSEBop aggregation complete: "
        f"{ds.sizes.get('time', '?')} time steps x "
        f"{ds.sizes.get(id_col, '?')} HRUs[/green]"
    )
    console.print(f"[green]Output: {run_dir / 'output' / 'ssebop_aet.nc'}[/green]")
```

- [ ] **Step 2: Verify CLI wiring**

Run: `pixi run nhf-targets agg --help`
Expected: Shows the `agg` sub-app with `ssebop` listed.

Run: `pixi run nhf-targets agg ssebop --help`
Expected: Shows `--run-dir`, `--period`, `--id-col`, `--batch-size` options.

- [ ] **Step 3: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All tests pass, no lint errors.

- [ ] **Step 4: Commit**

```bash
git add src/nhf_spatial_targets/cli.py
git commit -m "Add 'nhf-targets agg ssebop' CLI command

Standalone command to run SSEBop aggregation against a run workspace.
Follows the existing CLI pattern (agg sub-app parallels fetch sub-app)."
```

---

## Task 5: Final quality gate

- [ ] **Step 1: Run full quality checks**

```bash
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
```

Expected: All green.

- [ ] **Step 2: Verify file structure matches plan**

```
catalog/sources.yml                            # modified
src/nhf_spatial_targets/fetch/ssebop.py        # deleted
src/nhf_spatial_targets/aggregate/batching.py   # created
src/nhf_spatial_targets/aggregate/ssebop.py     # created
src/nhf_spatial_targets/cli.py                  # modified
tests/test_batching.py                          # created
tests/test_aggregate_ssebop.py                  # created
```

- [ ] **Step 3: Review git log**

Run: `git log --oneline -5`
Expected: 4 commits matching the tasks above.
