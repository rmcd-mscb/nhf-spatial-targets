> **SUPERSEDED:** Kerchunk references in this document are outdated — consolidation now uses direct xarray merge + NetCDF (PR #10). Retained for historical context.

# Soil Moisture Fetch Modules Design

**Date:** 2026-03-12
**Status:** Draft
**Scope:** Implement fetch modules for NCEP/NCAR, NLDAS-2 MOSAIC, and NLDAS-2 NOAH soil moisture sources, plus shared earthdata credential handling.

## Context

The soil moisture calibration target requires four reanalysis/LSM sources normalized to a common 0-1 range. MERRA-2 fetch is already implemented. This design covers the remaining three sources and a shared authentication utility.

## Design Decisions

1. **NLDAS variable names confirmed:** `SoilM_0_10cm`, `SoilM_10_40cm`, `SoilM_40_200cm` for both MOSAIC and NOAH. All three layers fetched; selection/interpolation deferred to calibration.
2. **NCEP/NCAR fetches both layers:** `soilw.0-10cm.gauss` and `soilw.10-200cm.gauss`.
3. **NLDAS MOSAIC and NOAH share a single module** (`fetch/nldas.py`) with a shared private helper and two public entry points. The logic is identical except for the source key.
4. **NCEP/NCAR daily-to-monthly aggregation** happens at download time. Monthly NetCDFs are the raw artifact stored on disk. Daily files are not retained.
5. **Incremental download** pattern (track progress in manifest, skip on re-run) used for all three sources, consistent with MERRA-2.
6. **Shared earthdata auth** (`fetch/_auth.py`) used by MERRA-2, NLDAS MOSAIC, and NLDAS NOAH. Reads credentials from the existing `.credentials.yml` template created by `init`.
7. **Per-source consolidation functions** in `consolidate.py` — no premature generic abstraction.

## Credential Handling (`fetch/_auth.py`)

### `earthdata_login(run_dir: Path) -> earthaccess.Auth`

```python
from nhf_spatial_targets.fetch._auth import earthdata_login
```

1. Read `run_dir/.credentials.yml` (already created by `nhf-targets init`).
2. Look for `nasa_earthdata.username` and `nasa_earthdata.password`.
3. If both non-empty: set `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` env vars, call `earthaccess.login(strategy="environment")`.
4. If missing, empty, or file not found: fall back to `earthaccess.login()` (tries netrc, interactive prompt, etc.).
5. Check `auth.authenticated` — if `False` or `auth is None`, raise `RuntimeError`. Callers do not need to check `.authenticated` themselves.
6. Return the authenticated `earthaccess.Auth` object.

Existing MERRA-2 module updated: replace inline `earthaccess.login()` + `.authenticated` check with `earthdata_login(run_dir)`. Mock path in `test_merra2.py` changes to `nhf_spatial_targets.fetch._auth.earthdata_login`.

## NLDAS Module (`fetch/nldas.py`)

### Public API

```python
def fetch_nldas_mosaic(run_dir: Path, period: str) -> dict:
    """Download NLDAS-2 MOSAIC monthly soil moisture."""
    return _fetch_nldas("nldas_mosaic", run_dir, period)

def fetch_nldas_noah(run_dir: Path, period: str) -> dict:
    """Download NLDAS-2 NOAH monthly soil moisture."""
    return _fetch_nldas("nldas_noah", run_dir, period)
```

### `_fetch_nldas(source_key, run_dir, period)` Flow

1. Load catalog metadata via `_catalog.source(source_key)`.
2. Check superseded status, warn if applicable.
3. Authenticate via `earthdata_login(run_dir)`.
4. Read `fabric.json` for bbox — use `fabric["bbox_buffered"]` with keys `minx/miny/maxx/maxy` (same as MERRA-2).
5. Incremental fetch: read manifest for already-downloaded months, compute needed months.
6. `earthaccess.search_data(short_name=..., bounding_box=..., temporal=...)`.
7. Filter granules to needed months.
8. `earthaccess.download()` to `run_dir/data/raw/<source_key>/`.
9. Build file inventory from disk.
10. Consolidate via `consolidate_nldas(run_dir, source_key, variables)`.
11. Update manifest.
12. Return provenance dict.

### Catalog Metadata (from `sources.yml`)

| Source | `short_name` | Version |
|--------|-------------|---------|
| NLDAS MOSAIC | `NLDAS_MOS0125_M` | 2.0 |
| NLDAS NOAH | `NLDAS_NOAH0125_M` | 2.0 |

### Variables (both sources)

| Variable | Depth | Units |
|----------|-------|-------|
| `SoilM_0_10cm` | 0-10 cm | kg/m² |
| `SoilM_10_40cm` | 10-40 cm | kg/m² |
| `SoilM_40_200cm` | 40-200 cm | kg/m² |

### Target Catalog YAML (`sources.yml` update for each NLDAS source)

```yaml
  variables:
    - name: SoilM_0_10cm
      long_name: "soil moisture 0-10 cm"
      layer_depth_m: "0.00-0.10"
      units: kg/m2
    - name: SoilM_10_40cm
      long_name: "soil moisture 10-40 cm"
      layer_depth_m: "0.10-0.40"
      units: kg/m2
    - name: SoilM_40_200cm
      long_name: "soil moisture 40-200 cm"
      layer_depth_m: "0.40-2.00"
      units: kg/m2
  status: current
```

Replaces the single `SOILM_UNKNOWN` entry and changes status from `needs_variable_verification` to `current`.

## NCEP/NCAR Module (`fetch/ncep_ncar.py`)

### Public API

```python
def fetch_ncep_ncar(run_dir: Path, period: str) -> dict:
    """Download NCEP/NCAR Reanalysis soil moisture."""
```

### Flow

1. Load catalog metadata via `_catalog.source("ncep_ncar")`.
2. Check superseded status.
3. Read `fabric.json` for bbox (stored in provenance only — no spatial subsetting at download).
4. Parse period via `_parse_period(period)` — same helper as MERRA-2/NLDAS (extract to a shared `fetch/_period.py` utility used by all three modules).
5. Incremental fetch: read manifest for already-downloaded years, compute needed years.
6. For each needed year, for each variable in `meta["variables"]`:
   - Build URL from variable's `file_pattern` with `{year}` substitution.
   - Download via `urllib.request.urlretrieve` to `run_dir/data/raw/ncep_ncar/`. On HTTP error (`urllib.error.HTTPError`), raise `RuntimeError` with the URL and status code. No retry logic — user re-runs to resume (incremental fetch skips completed years).
   - Open with xarray, resample to monthly means (`ds.resample(time="1ME").mean()`).
   - Write monthly NetCDF to `run_dir/data/raw/ncep_ncar/{file_variable}.{year}.monthly.nc`.
   - Delete the raw daily file after monthly means are written successfully.
7. Build file inventory from monthly files on disk.
8. Consolidate via `consolidate_ncep_ncar(run_dir, variables)`.
9. Update manifest.
10. Return provenance dict.

### Period Parsing (shared `fetch/_period.py`)

Extract `_parse_period(period)` and `_months_in_period(period)` from `merra2.py` into `fetch/_period.py`. All three fetch modules import from there. NCEP/NCAR also adds a `_years_in_period(period)` helper that returns `list[int]`.

### Catalog Metadata

- Access type: `noaa_psl` (direct HTTPS, no auth)
- URL pattern per variable: each variable entry has its own `file_pattern` with `{year}` placeholder
- Incremental tracking by year (not year-month)
- Bbox from `fabric.json` is stored in provenance only — NCEP/NCAR files are global grids, no spatial subsetting at download time. Spatial clipping happens during aggregation.

### Variables

Each variable entry in `catalog/sources.yml` includes its own `file_pattern`:

| Variable | `file_variable` | `file_pattern` | Depth | Units |
|----------|----------------|----------------|-------|-------|
| `soilw` | `soilw.0-10cm.gauss` | `https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface_gauss/soilw.0-10cm.gauss.{year}.nc` | 0-10 cm | kg/m² |
| `soilw` | `soilw.10-200cm.gauss` | `https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface_gauss/soilw.10-200cm.gauss.{year}.nc` | 10-200 cm | kg/m² |

The fetch module iterates `meta["variables"]` and substitutes `{year}` into each variable's `file_pattern` to build download URLs. The top-level `access.file_pattern` is removed in favor of per-variable patterns.

### Target Catalog YAML (`sources.yml` update)

```yaml
ncep_ncar:
  # ... existing fields unchanged ...
  access:
    type: noaa_psl
    url: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html
    notes: Monthly means derived from daily Gaussian grid files.
  variables:
    - name: soilw
      file_variable: soilw.0-10cm.gauss
      file_pattern: "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface_gauss/soilw.0-10cm.gauss.{year}.nc"
      long_name: "volumetric soil moisture 0-10 cm"
      layer_depth_m: "0.00-0.10"
      units: kg/m2
    - name: soilw
      file_variable: soilw.10-200cm.gauss
      file_pattern: "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface_gauss/soilw.10-200cm.gauss.{year}.nc"
      long_name: "volumetric soil moisture 10-200 cm"
      layer_depth_m: "0.10-2.00"
      units: kg/m2
```

## Consolidation (`consolidate.py`)

### `consolidate_nldas(run_dir, source_key, variables) -> dict`

Shared for both MOSAIC and NOAH. Same Kerchunk pattern as `consolidate_merra2`:
- Scan `.nc`/`.nc4` files in `run_dir/data/raw/<source_key>/`
- Build single-file Kerchunk references using `kerchunk.hdf.SingleHdf5ToZarr` (NLDAS GES DISC files are HDF5/NetCDF4)
- Combine on time dimension, filter to requested variables
- Apply time fixes if needed (check whether NLDAS monthly files already have correct timestamps)
- Write `run_dir/data/raw/<source_key>/<source_key>_refs.json`
- Return provenance dict with keys: `kerchunk_ref`, `last_consolidated_utc`, `n_files`, `variables`

### `consolidate_ncep_ncar(run_dir, variables) -> dict`

Operates on monthly `.nc` files only (daily files not retained):
- Scan `run_dir/data/raw/ncep_ncar/*.monthly.nc`
- Build single-file Kerchunk references using `kerchunk.netCDF3.NetCDF3ToZarr` (NCEP/NCAR Reanalysis files from NOAA PSL are NetCDF-3 classic format)
- Combine on time dimension, filter to requested variables
- Write `run_dir/data/raw/ncep_ncar/ncep_ncar_refs.json`
- Return provenance dict with keys: `kerchunk_ref`, `last_consolidated_utc`, `n_files`, `variables`

## CLI Integration (`cli.py`)

Three new subcommands under `fetch_app`:

```
nhf-targets fetch nldas-mosaic --run-dir <path> --period YYYY/YYYY
nhf-targets fetch nldas-noah   --run-dir <path> --period YYYY/YYYY
nhf-targets fetch ncep-ncar    --run-dir <path> --period YYYY/YYYY
```

Same pattern as existing `fetch merra2` — lazy import, call fetch function, log results.

## Testing

### `tests/test_auth.py`

- Credentials file with valid creds: sets env vars, calls `earthaccess.login(strategy="environment")`
- Credentials file with empty creds: falls back to `earthaccess.login()`
- No credentials file: falls back to `earthaccess.login()`
- All strategies fail: raises `RuntimeError`

### `tests/test_nldas.py`

Covers both MOSAIC and NOAH:
- Auth: login called, login failure raises
- Search params: correct `short_name` per source, bbox, temporal
- No granules: raises ValueError
- Output directory: `data/raw/nldas_mosaic/` or `data/raw/nldas_noah/`
- Provenance record: required keys present
- Superseded status warning
- Period validation: missing slash, non-numeric, reversed
- Missing fabric.json: raises FileNotFoundError
- Empty download: raises RuntimeError
- Incremental fetch: skips existing months
- Manifest: updated with provenance, preserves download timestamps

### `tests/test_ncep_ncar.py`

- URL construction: correct URL from catalog pattern + year
- Download failure: HTTP error raises
- Daily-to-monthly aggregation: verify monthly means from synthetic daily data
- Output directory: `data/raw/ncep_ncar/`
- Provenance, manifest, period validation tests
- Incremental fetch: skips existing years
- Missing fabric.json: raises FileNotFoundError

### `tests/test_consolidate.py` (extend)

- `consolidate_nldas`: variable filtering, time handling, relative paths, provenance return
- `consolidate_ncep_ncar`: same categories

All unit tests mock external I/O. Integration tests marked with `@pytest.mark.integration`.

## Files Changed/Created

| File | Action |
|------|--------|
| `catalog/sources.yml` | Edit — fix NLDAS variables (3 per source), add NCEP/NCAR second layer with per-variable `file_pattern`, update statuses to `current` |
| `catalog/variables.yml` | Edit — update range_notes with confirmed variable names |
| `CLAUDE.md` | Edit — remove NLDAS variable gap from "Still open" |
| `src/nhf_spatial_targets/fetch/_auth.py` | Create — shared `earthdata_login(run_dir)`, handles `.credentials.yml` + fallback |
| `src/nhf_spatial_targets/fetch/_period.py` | Create — shared `_parse_period`, `_months_in_period`, `_years_in_period` extracted from merra2.py |
| `src/nhf_spatial_targets/fetch/nldas.py` | Rewrite — shared `_fetch_nldas` helper + two public functions |
| `src/nhf_spatial_targets/fetch/ncep_ncar.py` | Rewrite — HTTPS download + daily→monthly + Kerchunk consolidation |
| `src/nhf_spatial_targets/fetch/consolidate.py` | Edit — add `consolidate_nldas` (HDF5), `consolidate_ncep_ncar` (NetCDF3) |
| `src/nhf_spatial_targets/fetch/merra2.py` | Edit — use `earthdata_login(run_dir)`, import period helpers from `_period.py` |
| `src/nhf_spatial_targets/cli.py` | Edit — add `fetch nldas-mosaic`, `fetch nldas-noah`, `fetch ncep-ncar` subcommands |
| `tests/test_auth.py` | Create — credential handling + fallback tests |
| `tests/test_nldas.py` | Create — NLDAS MOSAIC + NOAH tests (both sources) |
| `tests/test_ncep_ncar.py` | Create — NCEP/NCAR tests including daily→monthly aggregation |
| `tests/test_consolidate.py` | Edit — add tests for new consolidation functions |
| `tests/test_merra2.py` | Edit — update login and period mock paths |
