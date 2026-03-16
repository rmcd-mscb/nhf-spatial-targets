# WaterGAP 2.2d Fetch Module Design

## Summary

Implement the WaterGAP 2.2d groundwater recharge fetch module, downloading from PANGAEA via `pangaeapy` and verifying CF compliance. Update the catalog entry, variable definitions, and related documentation.

## Data Source

- **Product:** WaterGAP v2.2d — diffuse groundwater recharge (Rg)
- **DOI:** 10.1594/PANGAEA.918447
- **Citation:** Müller Schmied, H., and others, 2021, doi:10.5194/essd-13-2791-2021
- **File:** `watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4` (~222 MB)
- **Download URL pattern:** `https://hs.pangaea.de/model/WaterGAP_v2-2d/<filename>.nc4`
- **Access:** Open access via PANGAEA; CC BY-NC 4.0 license
- **Variable:** `qrdif` (diffuse groundwater recharge), units: mm per time step
- **Dimensions:** time (monthly, 1901–2016), lat, lon
- **Spatial extent:** Global, 0.5° resolution
- **Role in pipeline:** One of two sources for the recharge calibration target (alongside Reitz 2017). Both normalized 0–1 over the 2000–2009 window; min/max across sources defines the target range.

## Approach

Use `pangaeapy.PanDataSet(918447).download()` to fetch the single `qrdif` NC4 file. Verify CF compliance (Conventions, grid_mapping, coordinate attrs) and fix up if needed. No consolidation step — the downloaded file is the final artifact. Single-file skip logic (if file exists, skip download).

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
      units: mm
  ```
- `period`: `1901/2016`
- `units`: `mm/month`

## Variable Definition Update (`catalog/variables.yml`)

Update the `recharge` variable's `sources` list: change `watergap22a` to `watergap22d`.

## Documentation Updates

- `CLAUDE.md`: Move WaterGAP from "Still open" to "Resolved"
- `README.md`: Update WaterGAP task status

## Dependency

Add `pangaeapy` as an explicit dependency in `pixi.toml`. Currently not in the environment.

## Fetch Module (`src/nhf_spatial_targets/fetch/pangaea.py`)

Module named `pangaea.py` since the `pangaeapy` access pattern could serve future PANGAEA-hosted sources.

All modules use `from __future__ import annotations`.

### Structure

```
_SOURCE_KEY = "watergap22d"

fetch_watergap22d(run_dir: Path, period: str) -> dict
```

### Flow

1. Read catalog metadata via `_catalog.source("watergap22d")`
2. Read `fabric.json` for bbox (provenance only — no spatial subsetting on global 0.5° data)
3. Check if file already exists in `data/raw/watergap22d/` — skip download if present
4. Use `pangaeapy.PanDataSet(918447).download()` to fetch the `qrdif` NC4 file to `data/raw/watergap22d/`
5. Open with xarray, verify/ensure CF compliance:
   - `Conventions` attribute present
   - `qrdif` variable has `grid_mapping` attribute pointing to CRS variable
   - Coordinate variables have `units`, `axis` attrs
   - If attributes are missing, add them and write a corrected copy
6. Update `manifest.json` with provenance

### No consolidation

The single downloaded file is the final artifact. No per-year splitting or merging needed.

### Error handling

- PANGAEA unreachable or download failure: raise `RuntimeError` with actionable message
- File already exists: log and skip to CF check + manifest update
- CF attributes missing: fix in place, log what was added

### Manifest update

`_update_manifest()` follows the same pattern as other fetch modules: atomic write via tempfile, merge into existing manifest structure.

### Provenance return

Returns dict with: `source_key`, `access_url`, `doi`, `variables`, `period`, `bbox`, `download_timestamp`, `file`.

## CLI Wiring (`cli.py`)

Add `@fetch_app.command(name="watergap22d")` following the existing pattern:

- Parameters: `--run-dir`, `--period` (default from pipeline.yml)
- Calls `fetch_watergap22d(run_dir, period)`
- Prints result JSON

## Tests (`tests/test_pangaea.py`)

### Unit tests (no network)

- `test_fetch_watergap22d_skips_existing_file` — mock pangaeapy, verify skip when file exists
- `test_fetch_watergap22d_cf_compliance` — create synthetic NC4 missing CF attrs, verify they get added (especially `grid_mapping`)
- `test_fetch_watergap22d_missing_fabric_json` — verify `FileNotFoundError`
- `test_fetch_watergap22d_manifest_update` — verify manifest structure after fetch
- `test_fetch_watergap22d_download_failure` — mock pangaeapy raising, verify `RuntimeError`

### Integration test (marked `pytest.mark.integration`)

- `test_fetch_watergap22d_real_download` — fetch from real PANGAEA, verify file exists with correct variable name

## Relation to Issue #13

Issue #13 proposes a shared data directory with fixed CONUS bbox. This module downloads the full global dataset (no subsetting), consistent with issue #13's direction. When issue #13 lands, only the output directory location changes.
