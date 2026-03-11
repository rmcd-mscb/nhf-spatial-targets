# MERRA-2 Soil Moisture Fetch Module

**Date:** 2026-03-11
**Status:** Approved

## Goal

Implement the MERRA-2 fetch module to download monthly soil moisture data via `earthaccess`, subsetted to the run workspace's fabric bounding box and time period.

## Design

### Fetch module (`src/nhf_spatial_targets/fetch/merra2.py`)

Replaces the existing stub (`fetch_merra2_soilm`). The old function name is removed — no callers exist yet (all target builders are stubs).

- **Authentication**: `earthaccess.login()` — uses cached `~/.netrc` credentials or prompts interactively. If login fails, raise `RuntimeError` with a message pointing to https://urs.earthdata.nasa.gov/users/new.
- **Product**: `M2TMNXLND` (MERRA-2 tavg1_2d_lnd_Nx, monthly mean land surface diagnostics)
- **Variables**: `SFMC` (surface soil moisture) and `GWETROOT` (root zone wetness). MERRA-2 monthly files contain all variables per granule; variable subsetting happens after download when opening with xarray, not at download time.
- **Spatial subset**: Bounding box read from `run_dir/fabric.json` (`bbox_buffered`), converted to the tuple format `earthaccess.search_data()` expects: `(minx, miny, maxx, maxy)`.
- **Temporal subset**: Period as `"YYYY/YYYY"` (year granularity, start/end inclusive). The catalog entry for `merra2` shows `period: "1980/present"` — the actual period passed to the function is determined by the caller (pipeline config or target builder), typically `"1982/2010"` for soil moisture calibration.
- **Output**: Raw NetCDF4 files saved to `<run_dir>/data/raw/merra2/`, preserving original filenames from GES DISC.
- **Provenance**: Returns a dict suitable for writing to `manifest.json`. File paths are relative to `run_dir`.
- **Metadata**: Source URL and variable names read from `catalog.py` (via `from nhf_spatial_targets.catalog import source`), not hardcoded.
- **Superseded warning**: If `catalog.source("merra2")["status"] == "superseded"`, emit `warnings.warn("...", DeprecationWarning)`. Not triggered for `merra2` (status: `current`) but present for consistency with the provenance-reviewer convention.

### Interface

```python
def fetch_merra2(run_dir: Path, period: str) -> dict:
```

**Parameters:**
- `run_dir`: Path to the run workspace. The function reads `run_dir/fabric.json` to get `bbox_buffered` and writes files to `run_dir/data/raw/merra2/`.
- `period`: Temporal range as `"YYYY/YYYY"` string (start/end years inclusive, year granularity only).

**Returns:** Provenance dict:
```python
{
    "source_key": "merra2",
    "access_url": "<from catalog>",
    "variables": ["SFMC", "GWETROOT"],
    "period": "1982/2010",
    "bbox": {"minx": ..., "miny": ..., "maxx": ..., "maxy": ...},
    "download_timestamp": "2026-03-11T22:30:00Z",
    "files": [
        {"path": "data/raw/merra2/MERRA2_..._198201.nc4", "size_bytes": 12345},
        ...
    ]
}
```

File paths in `files` are relative to `run_dir`.

### Error handling

- **Auth failure**: `earthaccess.login()` failure raises `RuntimeError` with registration URL.
- **No results**: If `earthaccess.search_data()` returns zero granules, raise `ValueError` with the search parameters for debugging.
- **Download failure**: `earthaccess.download()` failures propagate as-is (typically `requests` exceptions). No retry logic — the caller can re-run the fetch; already-downloaded files in `data/raw/merra2/` are not re-downloaded if they exist.
- **Partial downloads**: Not supported. If the function raises mid-download, the caller should re-invoke; the function skips files that already exist in the output directory.

### Testing (`tests/test_merra2.py`)

**Unit tests** (no network, no credentials):
- Mock `earthaccess.login()`, `earthaccess.search_data()`, `earthaccess.download()`
- Verify `search_data()` called with correct short name (`M2TMNXLND`), bounding box as `(minx, miny, maxx, maxy)` tuple, and temporal range
- Verify output directory created at `run_dir/data/raw/merra2/`
- Verify provenance dict has all required keys (`source_key`, `access_url`, `variables`, `period`, `bbox`, `download_timestamp`, `files`) and correct types
- Verify `DeprecationWarning` emitted when catalog status is mocked as `superseded`
- Verify `RuntimeError` raised when `earthaccess.login()` returns `None` or raises
- Verify `ValueError` raised when `search_data()` returns empty list

**Integration test** (`@pytest.mark.integration`):
- Real `earthaccess.login()` and download for a single year (`"2010/2010"`, downloads 12 monthly files)
- Verify downloaded files exist in `run_dir/data/raw/merra2/`
- Open one file with xarray, verify it contains `SFMC` and `GWETROOT` variables

### Files

- Modify: `src/nhf_spatial_targets/fetch/merra2.py` (replace stub with implementation)
- Create: `tests/test_merra2.py`

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Auth method | `earthaccess.login()` | Standard, handles token caching in `~/.netrc` |
| Spatial subset | Fabric bbox from `fabric.json` | Follows CLAUDE.md convention, smaller downloads |
| bbox source | Read inside function from `run_dir/fabric.json` | Keeps interface simple; `run_dir` already passed |
| Variables | Both SFMC and GWETROOT | Uncertain which matches PRMS soil_rechr best |
| Variable subsetting | Post-download (xarray) | MERRA-2 granules contain all variables per file |
| Metadata source | `catalog.py` | Provenance-reviewer convention; no hardcoded URLs |
| File naming | Preserve original | Traceability back to GES DISC source |
| File paths in provenance | Relative to `run_dir` | Portable across machines |
| Period granularity | Year only (`"YYYY/YYYY"`) | Matches catalog convention |
| Superseded warning | `DeprecationWarning` | Standard Python category for deprecated resources |
| Skip existing files | Yes | Allows safe re-invocation after partial failure |
