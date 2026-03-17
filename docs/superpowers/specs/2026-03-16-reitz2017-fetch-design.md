# Reitz 2017 Recharge Fetch — Design Spec

## Goal

Fetch annual gridded recharge estimates (total and effective) from Reitz et al. (2017) via USGS ScienceBase, consolidate into a single NetCDF, and update the run manifest with provenance.

## Background

Reitz et al. (2017) provide empirical regression-based estimates of groundwater recharge for CONUS at 800m resolution, 2000–2013. This is one of two sources (alongside WaterGAP 2.2d) for the recharge calibration target in TM 6-B10. The recharge target uses normalized min-max over the 2000–2009 window.

The data release (doi:10.5066/F7PN93P0) is hosted on USGS ScienceBase:
- **Parent item:** `56c49126e4b0946c65219231` — contains ET, recharge, and runoff children
- **Recharge child item:** `55d383a9e4b0518e35468e58` — contains annual zipped GeoTIFFs

Two recharge products are available:
- **Total Recharge** (`TotalRecharge_YYYY.zip`) — total groundwater recharge
- **Effective Recharge** (`EffRecharge_YYYY.zip`) — base flow component

Each zip contains a single GeoTIFF at 800m CONUS resolution, ~35–55 MB per file.

## Architecture

### Approach

Use `sciencebasepy` to download zipped GeoTIFFs from the child item, unzip to raw GeoTIFFs, read with `rioxarray`, and consolidate into a single time-dimensioned NetCDF. Incremental download — skip years already in manifest.

No spatial clipping needed: the data is already CONUS-extent, and downstream gdptools aggregation handles subsetting to HRU polygons.

### Files

- **Create:** `src/nhf_spatial_targets/fetch/reitz2017.py` — fetch module
- **Create:** `tests/test_reitz2017.py` — unit + integration tests
- **Modify:** `catalog/sources.yml` — replace existing placeholder entry with full access details, structured variables
- **Modify:** `src/nhf_spatial_targets/cli.py` — add `fetch reitz2017` subcommand
- **Modify:** `pixi.toml` — add `fetch-reitz2017` task

### Module Structure

`src/nhf_spatial_targets/fetch/reitz2017.py`:

- `fetch_reitz2017(run_dir: Path, period: str) -> dict` — main entry point
  - Reads catalog metadata (child_item_id, file_patterns)
  - Validates fabric.json exists (confirms run workspace is initialized)
  - Downloads zips via sciencebasepy for each year in period, logging progress (`logger.info` per year: "Downloading year YYYY (N/M)...")
  - Unzips to GeoTIFFs (globs for `*.tif` within each zip to locate the raster), deletes zips
  - Calls `_consolidate()` to build NetCDF
  - Updates manifest.json via `_update_manifest()`
  - Returns provenance dict

- `_consolidate(output_dir: Path, period: str) -> Path` — internal
  - Globs `output_dir` for `TotalRecharge_*.tif` and `EffRecharge_*.tif` separately
  - For each variable, reads matching GeoTIFFs with `rioxarray.open_rasterio(path, masked=True)`
  - Squeezes the `band` dimension (`.squeeze("band", drop=True)`) since each GeoTIFF is single-band
  - The GeoTIFFs use a projected CRS (likely Albers Equal Area CONUS); preserve native CRS in the consolidated file — reprojection is deferred to the aggregation step
  - Nodata pixels (ocean, borders) are preserved via `masked=True` (NaN in float output)
  - Extracts year from filename (e.g. `TotalRecharge_2005.tif` → 2005)
  - Assigns annual time coordinate (mid-year: YYYY-07-01)
  - Pairs files by year: if a year has `TotalRecharge` but not `EffRecharge` (or vice versa), raises `RuntimeError` — both must be present
  - Builds xarray Dataset with two DataArrays: `total_recharge` and `eff_recharge`, dims `(time, y, x)`
  - Writes `reitz2017_consolidated.nc` atomically (`.nc.tmp` then rename), with zlib compression (`encoding={var: {"zlib": True, "complevel": 4}}`) to keep file size manageable (~1.2 GB uncompressed across 14 years × 2 vars)

- `_update_manifest(run_dir, period, meta, license_str, file_info)` — merges provenance into manifest.json. `meta` is the catalog dict from `_catalog.source("reitz2017")` (same pattern as pangaea.py). `license_str` is read from `meta.get("license", "unknown")`. Includes `spatial_extent: CONUS` in the manifest entry (no bbox clipping is applied since the data is already CONUS-only; bbox is not read from fabric.json)

### Data Flow

```
catalog/sources.yml (child_item_id, file_patterns)
        │
        ▼
fetch_reitz2017(run_dir, period)
        │
        ├── 1. Read catalog metadata + validate fabric.json exists
        │
        ├── 2. sciencebasepy: connect to child item 55d383a9e4b0518e35468e58
        │
        ├── 3. For each year in period (within 2000–2013):
        │      ├── Skip if GeoTIFF already exists (incremental)
        │      ├── Download TotalRecharge_YYYY.zip + EffRecharge_YYYY.zip
        │      ├── Unzip → GeoTIFF
        │      └── Delete zip, keep GeoTIFF
        │
        ├── 4. Consolidate: read all GeoTIFFs with rioxarray,
        │      assign time coord, write reitz2017_consolidated.nc
        │
        └── 5. Update manifest.json with provenance
```

### Output Directory

`<run_dir>/data/raw/reitz2017/`:
- `TotalRecharge_2000.tif` through `TotalRecharge_2013.tif` (raw provenance)
- `EffRecharge_2000.tif` through `EffRecharge_2013.tif`
- `reitz2017_consolidated.nc` — both variables with time dimension

### Catalog Entry Updates

The existing `reitz2017` entry in `sources.yml` is a placeholder with a single flat `variables: [recharge]` list. This replaces it entirely with structured access details and two explicit variables (`total_recharge`, `eff_recharge`) matching the actual data products on ScienceBase.

```yaml
reitz2017:
  name: Reitz et al. (2017) Empirical Recharge Estimates
  description: >
    Annual empirical regression-based estimates of groundwater recharge,
    quick-flow runoff, and ET for CONUS. Used as one of two sources for
    the recharge calibration target range (normalized min-max method).
    Targets relative year-to-year change, not absolute magnitude.
  citations:
    - "Reitz, M., Sanford, W.E., Senay, G.B., and Cazenas, J., 2017, doi:10.1111/1752-1688.12546"
    - "Data release: doi:10.5066/F7PN93P0"
  access:
    type: sciencebase
    item_id: "56c49126e4b0946c65219231"
    child_item_id: "55d383a9e4b0518e35468e58"
    doi: "10.5066/F7PN93P0"
    url: https://www.sciencebase.gov/catalog/item/56c49126e4b0946c65219231
    file_patterns:
      total_recharge: "TotalRecharge_{year}.zip"
      eff_recharge: "EffRecharge_{year}.zip"
  variables:
    - name: total_recharge
      file_variable: TotalRecharge
      long_name: total groundwater recharge
      units: inches/year
    - name: eff_recharge
      file_variable: EffRecharge
      long_name: effective groundwater recharge (base flow component)
      units: inches/year
  time_step: annual
  period: "2000/2013"
  spatial_extent: CONUS
  spatial_resolution: 800m raster
  units: inches/year
  license: public domain (USGS)
  status: current
```

### CLI

```python
@fetch_app.command(name="reitz2017")
def fetch_reitz2017_cmd(run_dir: Annotated[Path, ...], period: Annotated[str, ...]):
    ...
```

`period` is required (no default), consistent with all other fetch CLI commands.

pixi task: `fetch-reitz2017`

## Error Handling

- **ScienceBase unavailable:** Catch connection errors from sciencebasepy, raise `RuntimeError` with clear message
- **Missing zip file:** If a year's zip isn't found in the ScienceBase file listing, raise `RuntimeError`
- **Corrupt zip:** Let `zipfile.BadZipFile` propagate with context
- **Unexpected zip contents:** Glob for `*.tif` within the zip; raise `RuntimeError` if no `.tif` found
- **Mismatched variables:** If a year has only one of TotalRecharge/EffRecharge, raise `RuntimeError`
- **Incremental download:** Track downloaded GeoTIFFs on disk; skip years already present. If consolidated NC exists and all years downloaded, skip entirely
- **Atomic writes:** Consolidated NC written to `.tmp` then renamed
- **Period validation:** Only years within 2000–2013 are valid; raise `ValueError` if period extends beyond data range
- **Corrupt manifest:** Catch `json.JSONDecodeError`, raise `ValueError` with guidance
- **No auth required:** ScienceBase public items — no credentials needed

## Testing

### Test Helpers

- `_make_reitz_tif(path, year)` — creates a small (4×4) synthetic GeoTIFF with CRS and transform mimicking Reitz output
- `_mock_sciencebasepy` fixture — injects fake `sciencebasepy` module via `sys.modules`, controls `SbSession` mock

### Unit Tests

| Test | What it verifies |
|------|-----------------|
| `test_consolidate_builds_nc` | Reads synthetic GeoTIFFs, correct time coord, both variables, shape |
| `test_missing_fabric_raises` | FileNotFoundError when fabric.json absent |
| `test_skips_existing` | Consolidated NC exists → sciencebasepy not called |
| `test_incremental_skips_downloaded_years` | Some GeoTIFFs on disk → only missing years downloaded |
| `test_downloads_and_updates_manifest` | Full flow with mocked SB, manifest has correct fields |
| `test_manifest_preserves_existing_sources` | Merge doesn't overwrite other sources |
| `test_corrupt_manifest_raises` | ValueError on corrupt manifest.json |
| `test_download_failure_raises` | RuntimeError on sciencebasepy error |
| `test_period_out_of_range_raises` | ValueError for period beyond 2000/2013 |
| `test_zip_no_tif_raises` | RuntimeError when zip contains no .tif file |
| `test_cli_nonexistent_run_dir` | CLI exits with error code 2 |

### Integration Test

`@pytest.mark.integration` — downloads a single year (e.g. 2005) of both TotalRecharge and EffRecharge from real ScienceBase, verifies GeoTIFFs are valid and readable with rioxarray, runs consolidation on the single year, and checks the resulting NetCDF has both variables with correct time coordinate.
