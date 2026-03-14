# MODIS Fetch Module Design

**Date:** 2026-03-14
**Status:** Approved

## Overview

Implement `fetch_mod16a2()` and `fetch_mod10c1()` in `src/nhf_spatial_targets/fetch/modis.py` to download MODIS AET and snow cover data via earthaccess, following the existing MERRA-2 fetch pattern.

## Products

| Product | Purpose | Resolution | Temporal | Source Key |
|---------|---------|------------|----------|------------|
| MOD16A2 v061 | AET (8-day composite) | 500m, sinusoidal tiles | 2000–2010 | `mod16a2_v061` |
| MOD10C1 v061 | Snow cover (daily) | 0.05° CMG (global) | 2000–2014 | `mod10c1_v061` |

## Architecture

### Two independent fetch functions

Both follow the MERRA-2 pattern (`run_dir` + `fabric.json` bbox), with product-specific differences in download and consolidation.

**`fetch_mod16a2(run_dir: Path, period: str) -> dict`**
1. Load catalog (`mod16a2_v061`), read `fabric.json` bbox
2. Authenticate via `earthdata_login()`
3. For each needed year: `earthaccess.search_data(short_name, bbox, temporal)` → download CONUS tiles
4. Consolidate per year: mosaic tiles per timestep, concat along time → `mod16a2_v061_YYYY.nc`
5. Update `manifest.json`, return provenance dict

**`fetch_mod10c1(run_dir: Path, period: str) -> dict`**
1. Load catalog (`mod10c1_v061`), read `fabric.json` bbox
2. Authenticate via `earthdata_login()`
3. For each needed year: search and download daily global CMG files
4. **Subset on the fly**: open each global HDF, clip to CONUS bbox, save subset NetCDF, delete original HDF
5. Consolidate per year: concat daily subsets → `mod10c1_v061_YYYY.nc`
6. Update `manifest.json`, return provenance dict

### Memory management

MOD10C1 global files at 0.05° × 365 days/year ≈ 38 GB/year uncompressed. To stay within 32 GB RAM:
- Subset each global file to CONUS immediately after download (25x reduction)
- Only store CONUS subsets on disk
- Consolidation uses dask-backed xarray as a safety net

MOD16A2 tiles are already CONUS-scoped via bbox filtering; ~46 composites/year at 500m is manageable.

### earthaccess search parameters

- **MOD16A2**: `short_name` TBD (likely `MOD16A2GF` or `MOD16A2`), bbox from `fabric.json`, returns only CONUS tiles
- **MOD10C1**: `short_name="MOD10C1"`, bbox passed but returns global CMG files regardless — subset after download

### Incremental downloads

Track by **year**. Check `manifest.json` for years already downloaded; skip those. Use `years_in_period()` from `_period.py`.

### File organization

```
<run_dir>/data/raw/mod16a2_v061/
  MOD16A2*.hdf              # raw tiles
  mod16a2_v061_2005.nc      # consolidated per year

<run_dir>/data/raw/mod10c1_v061/
  MOD10C1_2005001_conus.nc  # CONUS subsets (originals deleted)
  mod10c1_v061_2005.nc      # consolidated per year
```

### Manifest structure

Per-year consolidated files stored as a dict:

```json
{
  "sources": {
    "mod16a2_v061": {
      "source_key": "mod16a2_v061",
      "access_url": "...",
      "period": "2000/2010",
      "bbox": {"minx": ..., "miny": ..., "maxx": ..., "maxy": ...},
      "variables": ["ET_500m", "ET_QC_500m"],
      "files": [{"path": "...", "year": 2005, "size_bytes": 123, "downloaded_utc": "..."}],
      "consolidated_ncs": {"2005": "data/raw/.../mod16a2_v061_2005.nc"},
      "last_consolidated_utc": "..."
    }
  }
}
```

## Catalog changes

Add `short_name` to both v061 entries in `catalog/sources.yml`:

```yaml
mod16a2_v061:
  access:
    short_name: MOD16A2GF  # verify during implementation

mod10c1_v061:
  access:
    short_name: MOD10C1
```

## CLI commands

Two new subcommands under `nhf-targets fetch`:
- `nhf-targets fetch mod16a2 --run-dir <path> --period YYYY/YYYY`
- `nhf-targets fetch mod10c1 --run-dir <path> --period YYYY/YYYY`

## Testing

`tests/test_modis.py`:

**Unit tests (mocked earthaccess):**
- Filename year extraction
- Missing/malformed `fabric.json` errors
- Superseded source key warning
- earthaccess called with correct parameters
- No granules found error
- Partial download warning
- Year-based incremental skip
- Manifest structure validation
- MOD10C1 CONUS subsetting
- Provenance dict completeness

**Integration tests (`@pytest.mark.integration`):**
- Single-granule real download per product
- End-to-end download → subset → consolidate → manifest

## Future work

- Refactor all fetch modules to use shared data directory with fixed CONUS bbox instead of per-run-dir `fabric.json` bbox (see GitHub issue)
- Extract shared earthaccess fetch helpers once a third earthaccess product is added
