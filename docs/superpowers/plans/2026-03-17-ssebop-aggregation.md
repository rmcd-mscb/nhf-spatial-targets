# SSEBop Aggregation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggregate SSEBop monthly AET from the USGS NHGF STAC catalog to HRU fabric polygons using gdptools, with spatially contiguous batching and weight caching.

**Architecture:** Remote-only aggregation via `NHGFStacZarrData` → `WeightGen` → `AggGen`. Fabric HRUs are partitioned into spatially contiguous batches using KD-tree recursive bisection. Per-batch weight CSVs are cached for reuse. No local fetch/download step.

**Tech Stack:** gdptools (NHGFStacZarrData, WeightGen, AggGen), geopandas, xarray, numpy, pystac

**Spec:** `docs/superpowers/specs/2026-03-17-ssebop-aggregation-design.md`

**Workspace note:** This plan uses the workspace refactor from PR #33. All modules use `workdir: Path` + `workspace.load(workdir)` for path resolution. Output goes to `ws.aggregated_dir()`, weights to `workdir / "weights"`, manifest to `ws.manifest_path`.

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

Run: `pixi run -e dev test -- tests/test_catalog.py`
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
    assert len(result) == 400


def test_batches_are_spatially_contiguous():
    gdf = _make_grid_fabric(20, 20)  # 400 features
    result = spatial_batch(gdf, batch_size=50)
    total_area = 20 * 20
    for bid in result["batch_id"].unique():
        batch = result[result["batch_id"] == bid]
        bounds = batch.total_bounds
        bbox_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
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

Run: `pixi run -e dev test -- tests/test_batching.py`
Expected: FAIL — `ImportError: cannot import name 'spatial_batch'`

- [ ] **Step 3: Implement spatial batching module**

Create `src/nhf_spatial_targets/aggregate/batching.py` with:
- `_recursive_bisect(centroids, indices, depth, max_depth, min_batch_size)` — KD-tree recursive bisection
- `spatial_batch(gdf, batch_size=500) -> gpd.GeoDataFrame` — assigns `batch_id` column

See spec for full implementation details.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev test -- tests/test_batching.py`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

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

Create `tests/test_aggregate_ssebop.py`. Key fixtures:

```python
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
    (tmp_path / "manifest.json").write_text(
        json.dumps({"sources": {}, "steps": []})
    )
    (tmp_path / "data" / "aggregated").mkdir(parents=True)
    return tmp_path
```

Test cases (all mock gdptools classes):
1. `test_aggregate_produces_dataset` — full orchestration returns `xr.Dataset` with `et` variable
2. `test_cached_weights_skips_recompute` — pre-created weight CSV means `WeightGen` is never called
3. `test_manifest_updated` — manifest has ssebop entry with `access_type`, `collection_id`, `doi`
4. `test_integration_tiny_fabric` — `@pytest.mark.integration`, real STAC endpoint

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -- tests/test_aggregate_ssebop.py`
Expected: FAIL — `ImportError: cannot import name 'aggregate_ssebop'`

- [ ] **Step 3: Implement the aggregation module**

Create `src/nhf_spatial_targets/aggregate/ssebop.py` with:

```python
def aggregate_ssebop(
    fabric_path: str | Path,
    id_col: str,
    period: str,
    workdir: str | Path,
    batch_size: int = 500,
) -> xr.Dataset:
```

Key differences from original plan (workspace refactor):
- Takes `workdir` not `run_dir`
- Uses `workspace.load(workdir)` to get `Workspace` object
- Output written to `ws.aggregated_dir() / "ssebop_aet.nc"` (not `run_dir / "output"`)
- Weights stored at `workdir / "weights" / "ssebop_batch<N>.csv"`
- Manifest read/written via `ws.manifest_path`
- `_update_manifest` uses `ws.manifest_path` and catches `JSONDecodeError`

Flow:
1. Load source metadata via `catalog.source("ssebop")`
2. Get STAC collection via `get_stac_collection(collection_id)`
3. Load and spatially batch the fabric
4. For each batch: compute/load weights, create NHGFStacZarrData, run AggGen
5. Concatenate batch datasets along `id_col`
6. Write output NetCDF to `ws.aggregated_dir() / "ssebop_aet.nc"`
7. Update manifest

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `pixi run -e dev test -- tests/test_aggregate_ssebop.py`
Expected: All 3 unit tests PASS.

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Add agg sub-app and ssebop command**

After `fetch_app` and `catalog_app` declarations, add:

```python
agg_app = App(name="agg", help="Aggregate source datasets to HRU fabric polygons.")
app.command(agg_app)
```

Add `ssebop` command using the workspace pattern:

```python
@agg_app.command(name="ssebop")
def agg_ssebop_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--workdir", "-w"],
            help="Workspace created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
    batch_size: Annotated[
        int,
        Parameter(name="--batch-size", help="Target HRUs per spatial batch."),
    ] = 500,
):
```

The command:
- Checks workdir exists and fabric.json present (validated)
- Reads fabric path and id_col from workspace config
- Calls `aggregate_ssebop(fabric_path, id_col, period, workdir, batch_size)`
- Has two-tier error handling matching the fetch command pattern

- [ ] **Step 2: Verify CLI wiring**

Run: `nhf-targets agg --help`
Run: `nhf-targets agg ssebop --help`

- [ ] **Step 3: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/nhf_spatial_targets/cli.py
git commit -m "Add 'nhf-targets agg ssebop' CLI command

Standalone command to run SSEBop aggregation against a workspace.
Follows the existing CLI pattern (agg sub-app parallels fetch sub-app)."
```

---

## Task 5: Final quality gate

- [ ] **Step 1: Run full quality checks**

```bash
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
```

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
