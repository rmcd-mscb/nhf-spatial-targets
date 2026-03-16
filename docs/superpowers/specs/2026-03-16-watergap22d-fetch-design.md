# WaterGAP 2.2d Fetch Module Design

## Summary

Implement the WaterGAP 2.2d groundwater recharge fetch module, downloading from PANGAEA via `pangaeapy` and verifying/fixing CF compliance. Update the catalog entry, variable definitions, and related documentation.

## Data Source

- **Product:** WaterGAP v2.2d — diffuse groundwater recharge (Rg)
- **DOI:** 10.1594/PANGAEA.918447
- **Citation:** Müller Schmied, H., and others, 2021, doi:10.5194/gmd-14-1037-2021
- **File:** `watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4` (~217 MB)
- **Confirmed download URL:** `https://hs.pangaea.de/model/WaterGAP_v2-2d/watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4`
- **Filename verified:** via `pangaeapy.PanDataSet(918447).data` — row 30, `File name` column
- **Access:** Open access via PANGAEA; CC BY-NC 4.0 license (non-commercial; USGS research use compliant)
- **Variable:** `qrdif` (diffuse groundwater recharge), units: `kg m-2 s-1` (flux)
- **Dimensions:** time (1392 monthly steps, 1901–2016), lat (360, 0.5°), lon (720, 0.5°)
- **Spatial extent:** Global, 0.5° resolution
- **Role in pipeline:** One of two sources for the recharge calibration target (alongside Reitz 2017). Both normalized 0–1 over the 2000–2009 window; min/max across sources defines the target range.

## Known CF Compliance Issues (verified from file inspection)

The downloaded NC4 file has several CF compliance problems that must be fixed:

1. **Time encoding:** `units: "months since 1901-01-01"` with `calendar: proleptic_gregorian` — xarray cannot decode this (CF only allows "months since" with 360_day calendar). Must reconstruct time coordinate as proper datetime values.
2. **No `grid_mapping`:** The `qrdif` variable has no `grid_mapping` attribute. Must add a `crs` variable with WGS84 grid mapping attributes and set `grid_mapping = "crs"` on `qrdif`.
3. **`Conventions` attribute:** Listed as `"partly ALMA, CF and ISIMIP2b protocol for naming and units"` — not a standard CF value. Should be set to `"CF-1.6"` after fixes.
4. **Units:** Stored as `kg m-2 s-1` (flux). The catalog lists `mm/month`. Unit conversion from flux to depth/time is: 1 kg m-2 s-1 × seconds_in_month = mm/month (since 1 kg/m² water = 1 mm depth). Note: conversion happens at aggregation time, not in the fetch module — fetch preserves original units.

## Approach

Use `pangaeapy.PanDataSet(918447).download(indices=[30])` to fetch the single `qrdif` NC4 file. The file downloads to `~/.pangaeapy_cache/` — move it to `data/raw/watergap22d/`. Open with xarray (`decode_times=False`), fix CF compliance issues, and write a corrected copy alongside the original. No consolidation step.

## Catalog Update (`catalog/sources.yml`)

Update the `watergap22d` entry:

- `access.type`: `pangaea`
- `access.doi`: `10.1594/PANGAEA.918447`
- `access.url`: `https://doi.pangaea.de/10.1594/PANGAEA.918447`
- `access.file`: `watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4`
- Remove `access.notes` about verifying variable name
- `variables`: convert to dict format for consistency:
  ```yaml
  variables:
    - name: groundwater_recharge
      file_variable: qrdif
      long_name: diffuse groundwater recharge
      units: kg m-2 s-1
  ```
- `time_step`: change from `annual` to `monthly`
- `period`: `1901/2016`
- `units`: `kg m-2 s-1` (original; converted to mm/month at aggregation time)

## Variable Definition Update (`catalog/variables.yml`)

Update the `recharge` variable:
- Change `sources` list: `watergap22a` → `watergap22d`
- Remove the comment `# or watergap22d (open-access substitute on PANGAEA)`

## Documentation Updates

- `CLAUDE.md`: Move WaterGAP from "Still open" to "Resolved"
- `README.md`: Update WaterGAP task status

## Dependency

Add `pangaeapy` as an explicit dependency in `pixi.toml`. Already added during design exploration.

Note: `pangaeapy` is GPL v3 licensed. This project uses it as a runtime dependency (not linking), which is compatible with USGS open-source research use. The dependency is isolated to the PANGAEA fetch module.

## Fetch Module (`src/nhf_spatial_targets/fetch/pangaea.py`)

Module named `pangaea.py` since the `pangaeapy` access pattern could serve future PANGAEA-hosted sources (consistent with `sciencebase.py` which also groups by provider).

All modules use `from __future__ import annotations`.

### Structure

```
_SOURCE_KEY = "watergap22d"

fetch_watergap22d(run_dir: Path, period: str) -> dict
```

The `period` parameter is kept for interface consistency with other fetch modules but is only used for provenance recording — the single file covers the full 1901–2016 period.

### Flow

1. Read catalog metadata via `_catalog.source("watergap22d")`
2. Read `fabric.json` for bbox (provenance only — no spatial subsetting on global 0.5° data)
3. Check if CF-corrected file already exists in `data/raw/watergap22d/` — skip download if present
4. Use `pangaeapy.PanDataSet(918447).download(indices=[30])` to fetch the `qrdif` NC4 file (downloads to `~/.pangaeapy_cache/`)
5. Move/copy from cache to `data/raw/watergap22d/`
6. CF compliance fix-up:
   - Open with `xr.open_dataset(path, decode_times=False)`
   - Reconstruct time coordinate: convert "months since 1901-01-01" integer offsets to proper `datetime64` values (year = 1901 + month_offset // 12, month = 1 + month_offset % 12)
   - Add `crs` variable with WGS84 grid mapping attributes
   - Set `grid_mapping = "crs"` on `qrdif`
   - Set `Conventions = "CF-1.6"`
   - Write corrected file as `watergap22d_qrdif_cf.nc` (preserve original as audit artifact)
7. Update `manifest.json` with provenance (including license: CC BY-NC 4.0)

### No consolidation

The single downloaded file is the final artifact. No per-year splitting or merging needed.

### Error handling

- PANGAEA unreachable or download failure: raise `RuntimeError` with actionable message
- File already exists: log and skip to CF check + manifest update
- CF fix-up failure: raise with details, preserve original file

### Manifest update

`_update_manifest()` follows the same pattern as other fetch modules: atomic write via tempfile, merge into existing manifest structure. Include `license: "CC BY-NC 4.0"` in the provenance record.

### Provenance return

Returns dict with: `source_key`, `access_url`, `doi`, `license`, `variables`, `period`, `bbox`, `download_timestamp`, `file`, `cf_corrected_file`.

## CLI Wiring (`cli.py`)

Add `@fetch_app.command(name="watergap22d")` following the existing pattern:

- Parameters: `--run-dir`, `--period` (default from pipeline.yml)
- Calls `fetch_watergap22d(run_dir, period)`
- Prints result JSON

## Tests (`tests/test_pangaea.py`)

### Unit tests (no network)

- `test_fetch_watergap22d_skips_existing_file` — mock pangaeapy, verify skip when CF-corrected file exists
- `test_fetch_watergap22d_cf_time_reconstruction` — create synthetic NC4 with "months since" time encoding, verify proper datetime reconstruction
- `test_fetch_watergap22d_cf_grid_mapping` — verify `grid_mapping` and `crs` variable added
- `test_fetch_watergap22d_missing_fabric_json` — verify `FileNotFoundError`
- `test_fetch_watergap22d_manifest_update` — verify manifest structure including license field
- `test_fetch_watergap22d_download_failure` — mock pangaeapy raising, verify `RuntimeError`
- `test_fetch_watergap22d_preserves_original` — verify both original and CF-corrected files exist

### Integration test (marked `pytest.mark.integration`)

- `test_fetch_watergap22d_real_download` — fetch from real PANGAEA, verify file exists with correct variable name and CF-compliant time coordinate

## Relation to Issue #13

Issue #13 proposes a shared data directory with fixed CONUS bbox. This module downloads the full global dataset (no subsetting), consistent with issue #13's direction. When issue #13 lands, only the output directory location changes.
