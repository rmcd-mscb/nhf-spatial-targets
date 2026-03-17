# SSEBop Aggregation Design

## Summary

Aggregate SSEBop monthly actual evapotranspiration from the USGS NHGF STAC
catalog to HRU fabric polygons using gdptools. SSEBop is a remote dataset —
no local download step. Data is accessed directly from a Zarr store on the
Open Storage Network and area-weighted to the target fabric.

## Source Details

- **Collection ID:** `ssebopeta_monthly`
- **Zarr store:** `s3://mdmf/gdp/ssebopeta_monthly.zarr/`
- **S3 endpoint:** `https://usgs.osn.mghpcc.org/`
- **Variable:** `et` (Actual Evapotranspiration, mm/month)
- **Resolution:** ~0.009° (~1 km), WGS 84
- **Temporal extent:** 2000-01 to 2023-12, monthly
- **DOI:** 10.5066/P9L2YMV
- **Access:** anonymous S3 (no credentials required)

## Catalog Changes

Update `catalog/sources.yml` SSEBop entry:

```yaml
ssebop:
  name: SSEBop Actual Evapotranspiration
  description: >
    Operational Simplified Surface Energy Balance (SSEBop) AET estimates.
    Used as one of three sources for AET calibration target range.
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

Note: `access.endpoint` and `access.zarr_store` are informational metadata for
human reference. The code uses `access.collection_id` with
`gdptools.helpers.get_stac_collection()`, which resolves the Zarr asset
location from the STAC catalog automatically.

Key changes from existing entry:
- `access.type`: `usgs_water` → `usgs_gdp_stac`
- Added `access.collection_id`, `access.endpoint`, `access.zarr_store`
- Added `doi` field
- `status`: `needs_version_verification` → `current`
- `period`: `2000/2010` → `2000/2023` (full available range)

## Existing Fetch Stub

The existing `src/nhf_spatial_targets/fetch/ssebop.py` stub will be deleted.
SSEBop is a remote dataset accessed via gdptools STAC interface — no local
download step. This is an intentional pattern variation for `usgs_gdp_stac`
sources compared to the fetch-then-aggregate pattern used for other sources.

## Spatial Batching

**File:** `src/nhf_spatial_targets/aggregate/batching.py`

Adapted from `hydro-param` (github.com/rmcd-mscb/hydro-param). Uses KD-tree
recursive bisection to partition fabric HRUs into spatially contiguous batches.

**Function:** `spatial_batch(gdf, batch_size=500) -> gpd.GeoDataFrame`

- Computes polygon centroids, then recursively bisects along alternating x/y
  axes at the median (KD-tree style).
- Returns a copy of the GeoDataFrame with a `batch_id` column (int, 0-indexed).
- Recursion depth: `ceil(log2(n_features / batch_size))`.
- `min_batch_size = batch_size // 2` prevents excessive fragmentation.
- Suppresses geographic CRS centroid warnings (approximate centroids are
  sufficient for spatial grouping).

Spatial contiguity is critical: naive batching (e.g., by row order) would
produce batches with large bounding boxes, causing gdptools to fetch far more
source data than needed and risking OOM on high-resolution grids.

## Aggregate Module

**File:** `src/nhf_spatial_targets/aggregate/ssebop.py`

All new modules use `from __future__ import annotations`.

**Function:** `aggregate_ssebop(fabric_path, id_col, period, run_dir) -> xr.Dataset`

### Flow

1. Load source metadata via `catalog.source("ssebop")` — get collection ID.
2. Obtain the STAC collection object via
   `gdptools.helpers.get_stac_collection(collection_id)`.
3. Load fabric GeoDataFrame from the GeoPackage.
4. Spatially batch the fabric via `spatial_batch(gdf, batch_size)` — produces
   a `batch_id` column with spatially contiguous groups.
5. For each batch (grouped by `batch_id`):
   a. Check for cached weights at `<run_dir>/weights/ssebop_batch<N>.csv`.
   b. If no cache: create `NHGFStacZarrData` with `source_collection`,
      batch GeoDataFrame, `source_var="et"`, and time period. Run
      `WeightGen.calculate_weights(method="serial")`, save CSV.
   c. If cached: load weights DataFrame from CSV.
   d. Create `AggGen` with `stat_method="masked_mean"`, `agg_engine="serial"`,
      `agg_writer="none"`, pass weights DataFrame.
   e. Call `calculate_agg()` → `(gdf, xr.Dataset)`.
   f. Collect the Dataset.
6. Concatenate batch Datasets along the HRU dimension.
7. Write consolidated NetCDF to `<run_dir>/output/ssebop_aet.nc`.
8. Return the Dataset.

### Key Parameters

- `source_var="et"` (string, single variable)
- `source_time_period` derived from `period` config (e.g., `["2000-01-01", "2010-12-31"]`)
- `weight_gen_crs=5070` (NAD83 / CONUS Albers, EPSG:5070, equal-area for
  accurate intersection areas)
- `weight_gen_method="serial"` (matches `agg_engine`)
- `batch_size=500` (configurable, target HRUs per batch)
- Weight CSVs named by source key + batch index for identification and reuse

### Weight Caching

Weights are keyed by (source grid, fabric). As long as the source and target
are the same, weights can be reused across runs. Weight CSVs are stored at
`<run_dir>/weights/ssebop_batch<N>.csv`.

### Output

- **File:** `<run_dir>/output/ssebop_aet.nc`
- **Dimensions:** `(time, <id_col>)` where `id_col` is the fabric's ID column
- **Variable:** `et` in native units (mm/month) — no unit conversion at this
  stage. All unit conversions deferred to calibration file writer.
- **Attributes:** CF-1.6 compliant, includes DOI, collection ID, aggregation
  method metadata.

## Unit Conversion

SSEBop native units are mm/month. The AET calibration target requires
inches/day. This conversion is **not** performed here — it is deferred to the
calibration file writer that combines all three AET sources. Keeping aggregated
outputs in native units avoids double-conversion and simplifies the aggregation
layer.

## CLI Integration

Add `nhf-targets agg ssebop` command for standalone aggregation, following the
existing CLI pattern (`nhf-targets fetch <source>`). This allows running SSEBop
aggregation independently for testing and incremental work, without requiring
the full AET target builder.

## Manifest / Provenance

After aggregation completes, write a provenance entry to
`<run_dir>/manifest.json`:

```json
{
  "ssebop": {
    "source_key": "ssebop",
    "collection_id": "ssebopeta_monthly",
    "doi": "10.5066/P9L2YMV",
    "access_type": "usgs_gdp_stac",
    "period": "2000-01-01/2010-12-31",
    "fabric_sha256": "<from fabric.json>",
    "output_file": "output/ssebop_aet.nc",
    "weight_files": ["weights/ssebop_batch0.csv", "..."],
    "timestamp": "<ISO 8601>"
  }
}
```

## Testing

**File:** `tests/test_aggregate_ssebop.py`

### Unit Tests

**`tests/test_batching.py`** — spatial batching module:
- Verify spatially contiguous batch assignment on a synthetic GeoDataFrame.
- Edge cases: empty GeoDataFrame, fewer features than batch_size (single batch),
  features aligned along one axis (degenerate median split).

**`tests/test_aggregate_ssebop.py`** — aggregation orchestration:
- Mock `NHGFStacZarrData`, `WeightGen`, and `AggGen` to verify orchestration:
  batching logic, weight cache check/save/reload, dataset concatenation,
  manifest writing.
- Test that cached weights CSV is loaded instead of recomputed.
- Test that batch results are concatenated correctly along HRU dimension.

### Integration Tests (`pytest.mark.integration`)

- Hit the real STAC endpoint with a tiny fabric (2–3 HRUs) and a short period
  (1 month) to verify end-to-end flow.
