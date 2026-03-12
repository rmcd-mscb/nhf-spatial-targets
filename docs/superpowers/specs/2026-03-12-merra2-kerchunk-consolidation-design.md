> **SUPERSEDED:** Kerchunk was replaced with direct xarray merge + NetCDF consolidation in PR #10. This document is retained for historical context only.

# MERRA-2 Kerchunk Consolidation Design

## Goal

After fetching MERRA-2 monthly NetCDF files, create a virtual Zarr store via Kerchunk that extracts only the relevant soil moisture variables, applies CF-compliant time representation, and supports incremental updates when the period of record changes.

## Context

MERRA-2 M2TMNXLND monthly files each contain ~100 variables. Only three soil moisture wetness variables are needed for calibration targets. The files are global (not spatially subsetted) — gdptools handles spatial aggregation downstream. A full period-of-record fetch (1980–present) produces ~500 files; Kerchunk avoids duplicating that data into a merged NetCDF.

## Variables

Three MERRA-2 soil wetness variables at nested depth intervals:

| Variable | Long name | Depth | Units |
|----------|-----------|-------|-------|
| GWETTOP | surface soil wetness | 0–0.05 m | dimensionless (0–1) |
| GWETROOT | root zone soil wetness | 0–1.0 m | dimensionless (0–1) |
| GWETPROF | average profile soil moisture | 0–full profile (~3.5 m) | dimensionless (0–1) |

All three are dimensionless fractions of saturation on the same 0.5 x 0.625 degree grid. The nested depths enable weighted interpolation to arbitrary target depth intervals. SFMC (volumetric, m3/m3) is removed — redundant with GWETTOP at the same layer.

## Catalog Update

Update `catalog/sources.yml` `merra2.variables`:
- Remove SFMC
- Add GWETROOT (root zone, 0–1.0 m)
- Add GWETPROF (full profile, ~3.5 m)
- Keep GWETTOP as preferred

## Incremental Fetch

Current `fetch_merra2()` downloads all granules for the requested period. New behavior:

1. Read `manifest.json` to find which months are already downloaded
2. Compute delta — only search/download months not yet on disk
3. After downloading, update manifest with per-file records:
   - filename, relative path, size_bytes, year_month, download timestamp
4. On network failure, next run picks up where it left off
5. If period extends (e.g., `1980/2020` → `1980/2023`), only new months download

Manifest `sources.merra2` structure:

```json
{
  "source_key": "merra2",
  "period": "1980/2020",
  "variables": ["GWETTOP", "GWETROOT", "GWETPROF"],
  "files": [
    {"path": "data/raw/merra2/MERRA2_100.tavgM_2d_lnd_Nx.198001.nc4",
     "year_month": "1980-01", "size_bytes": 83000000,
     "downloaded_utc": "2026-03-12T00:30:00+00:00"}
  ],
  "kerchunk_ref": "data/raw/merra2/merra2_refs.json",
  "last_consolidated_utc": "2026-03-12T01:00:00+00:00"
}
```

## Kerchunk Consolidation

Runs automatically at end of `fetch_merra2()`:

1. **Scan** each NetCDF via `kerchunk.hdf.SingleHdf5ToZarr`
2. **Filter variables** — drop unwanted keys from each reference dict, keeping only GWETTOP, GWETROOT, GWETPROF plus coordinate variables (time, lat, lon)
3. **Combine** along time via `kerchunk.combine.MultiZarrToZarr`, sorted by time
4. **Fix time representation:**
   - Shift timestamps from `YYYY-01-01T00:30:00` to `YYYY-01-15T00:00:00` (mid-month)
   - Add `time_bnds`: `[[YYYY-01-01, YYYY-02-01], ...]` (first-of-month bounds)
   - Set `time:bounds = "time_bnds"` and `time:cell_methods = "time: mean"`
5. **Add global attributes:**
   - `history`: "Kerchunk virtual Zarr created by nhf-spatial-targets v{version}"
   - `source`: "NASA MERRA-2 M2TMNXLND v5.12.4"
   - `time_modification_note`: "Original timestamps (YYYY-01-01T00:30:00) shifted to mid-month (15th) for consistency. See time_bnds for exact averaging periods."
   - `references`: DOI from catalog
   - `Conventions`: "CF-1.8"
6. **Write** `merra2_refs.json` with **relative paths** (e.g., `./MERRA2_300.tavgM_2d_lnd_Nx.201001.nc4`)
7. **Update manifest** with `kerchunk_ref` path and `last_consolidated_utc`

Variable filtering happens at scan time (option a) to produce a clean, small reference file — not deferred to read time.

## Module Structure

**New file:** `src/nhf_spatial_targets/fetch/consolidate.py`
- `consolidate_merra2(run_dir: Path, variables: list[str]) -> dict`
- Builds Kerchunk reference store, returns provenance dict for manifest

**Modified:** `src/nhf_spatial_targets/fetch/merra2.py`
- Incremental fetch logic: read manifest, compute delta, download missing months
- Calls `consolidate_merra2()` after download
- Updates manifest with file records and consolidation metadata

**New dependency:** `kerchunk`, `ujson` in `pixi.toml`

**New test file:** `tests/test_consolidate.py`
- Variable filtering from reference dicts
- Time coordinate adjustment (mid-month, bounds)
- Relative path handling
- Incremental rebuild (existing refs + new files)

## CLI

No new commands. Consolidation runs as part of `nhf-targets fetch merra2`.

## Consumer Pattern

```python
import fsspec
import xarray as xr

fs = fsspec.filesystem("reference", fo="data/raw/merra2/merra2_refs.json")
ds = xr.open_zarr(fs.get_mapper(""), consolidated=False)
# ds has GWETTOP, GWETROOT, GWETPROF with CF-compliant time
```

## Future: Cloud Serving

Kerchunk JSON references can be served on OSN pods by updating relative paths to `s3://` URLs. If versioned/transactional access is needed, convert to VirtualiZarr/Icechunk at deployment time — no wasted work since VirtualiZarr can ingest Kerchunk references directly.
