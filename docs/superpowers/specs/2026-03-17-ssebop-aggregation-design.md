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

Key changes from existing entry:
- `access.type`: `usgs_water` → `usgs_gdp_stac`
- Added `access.collection_id`, `access.endpoint`, `access.zarr_store`
- Added `doi` field
- `status`: `needs_version_verification` → `current`
- `period`: `2000/2010` → `2000/2023` (full available range)

## Aggregate Module

**File:** `src/nhf_spatial_targets/aggregate/ssebop.py`

**Function:** `aggregate_ssebop(fabric_path, id_col, period, run_dir) -> xr.Dataset`

### Flow

1. Load source metadata via `catalog.source("ssebop")` — get collection ID and
   endpoint.
2. Load fabric GeoDataFrame from the GeoPackage.
3. Chunk the fabric into batches of HRUs (configurable chunk size).
4. For each chunk:
   a. Check for cached weights at `<run_dir>/weights/ssebop_chunk<N>.csv`.
   b. If no cache: create `NHGFStacZarrData` with the chunk GeoDataFrame,
      run `WeightGen.calculate_weights()`, save CSV.
   c. If cached: load weights DataFrame from CSV.
   d. Create `AggGen` with `stat_method="masked_mean"`, `agg_engine="serial"`,
      `agg_writer="none"`, pass weights DataFrame.
   e. Call `calculate_agg()` → `(gdf, xr.Dataset)`.
   f. Collect the Dataset.
5. Concatenate chunk Datasets along the HRU dimension.
6. Write consolidated NetCDF to `<run_dir>/output/ssebop_aet.nc`.
7. Return the Dataset.

### Key Parameters

- `source_var=["et"]`
- `source_time_period` derived from `period` config (e.g., `["2000-01-01", "2010-12-31"]`)
- `weight_gen_crs=6931` (NAD83 / CONUS Albers, equal-area for accurate
  intersection areas)
- Weight CSVs named by source key + chunk index for identification and reuse

### Weight Caching

Weights are keyed by (source grid, fabric). As long as the source and target
are the same, weights can be reused across runs. Weight CSVs are stored at
`<run_dir>/weights/ssebop_chunk<N>.csv`.

### Output

- **File:** `<run_dir>/output/ssebop_aet.nc`
- **Dimensions:** `(time, hru_id)`
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

Add `nhf-targets run-ssebop-agg` command for standalone aggregation. This
allows running SSEBop aggregation independently for testing and incremental
work, without requiring the full AET target builder.

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
    "weight_files": ["weights/ssebop_chunk0.csv", "..."],
    "timestamp": "<ISO 8601>"
  }
}
```

## Testing

**File:** `tests/test_aggregate_ssebop.py`

### Unit Tests

- Mock `NHGFStacZarrData`, `WeightGen`, and `AggGen` to verify orchestration:
  chunking logic, weight cache check/save/reload, dataset concatenation,
  manifest writing.
- Test that cached weights CSV is loaded instead of recomputed.
- Test that chunk results are concatenated correctly along HRU dimension.

### Integration Tests (`pytest.mark.integration`)

- Hit the real STAC endpoint with a tiny fabric (2–3 HRUs) and a short period
  (1 month) to verify end-to-end flow.
