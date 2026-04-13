# ERA5-Land + GLDAS Runoff Replacement Design

**Date:** 2026-04-13
**Status:** Draft — pending implementation plan

## Summary

Remove the NHM Monthly Water Balance Model (`nhm_mwbm`) as a data source and
replace it with two reanalysis-based runoff sources:

- **ERA5-Land** (Copernicus, hourly, 0.1°) — primary new source
- **GLDAS-2.1 NOAH monthly** (NASA GES DISC, 0.25°) — second source

The runoff target switches from a single-source `mwbm_uncertainty` envelope
to a `multi_source_minmax` range across the two new sources (matching the
AET pattern). ERA5-Land hourly data is aggregated to daily (and saved) and
to monthly. ERA5-Land sub-surface runoff (`ssro`) is also added as a third
source for the recharge calibration target.

## Motivation

`nhm_mwbm` was never implemented as a fetch module. Replacing it with
modern reanalysis sources gives:

- A reproducible, openly available data path (no dependency on a frozen
  USGS data release)
- A `multi_source_minmax` range structure consistent with the AET target
- Daily ERA5-Land runoff in the datastore for any future daily-resolution
  work

## Scope of Changes

### Remove

- `nhm_mwbm` entry in `catalog/sources.yml`
- `nhm_mwbm` from the `aet` source list in `catalog/variables.yml`
- Stub references to MWBM in `targets/run.py`
- MWBM-specific notes in `CLAUDE.md` "Resolved" gaps

### Add

- `era5_land` source in `catalog/sources.yml`
- `gldas_noah_v21_monthly` source in `catalog/sources.yml`
- `src/nhf_spatial_targets/fetch/era5_land.py` — CDS API client; hourly
  download; hourly→daily aggregation; daily→monthly aggregation
- `src/nhf_spatial_targets/fetch/gldas.py` — earthaccess client for
  `GLDAS_NOAH025_M` v2.1
- `tests/test_era5_land.py`, `tests/test_gldas.py`, `tests/test_run_target.py`
- `cdsapi` dependency in `pixi.toml`

### Modify

- `catalog/variables.yml`:
  - `runoff` block — `range_method: multi_source_minmax`,
    `sources: [era5_land, gldas_noah_v21_monthly]`,
    drop `mwbm_uncertainty` notes
  - `recharge` block — add `era5_land` (using `ssro`) as third source
  - `aet` block — drop `nhm_mwbm`
- `src/nhf_spatial_targets/targets/run.py` — implement multi-source minmax
- `src/nhf_spatial_targets/targets/aet.py` — drop MWBM source
- `src/nhf_spatial_targets/targets/rch.py` — add ERA5-Land source
- `src/nhf_spatial_targets/validate.py` — check for CDS credentials and
  `cdsapi` import
- `tests/test_catalog.py` — drop MWBM assertions, add new sources
- `CLAUDE.md` — update Known Gaps section
- `README.md` — update runoff source description

## Geographic Bounds

A single bbox is used for both new sources, encompassing CONUS plus
contributing watersheds in Canada and Mexico, with ~10 km buffer
(snapped to the ERA5-Land 0.1° grid):

- **Lon/Lat (minx, miny, maxx, maxy):** `(-125.0, 24.7, -66.0, 53.0)`
- **CDS area parameter (N, W, S, E):** `[53.0, -125.0, 24.7, -66.0]`

This constant lives in `fetch/era5_land.py` and is documented in the
catalog `notes:` field for both sources.

## ERA5-Land Fetch Module

### Access

- Library: `cdsapi`
- Credentials in project `.credentials.yml`:
  ```yaml
  cds:
    url: https://cds.climate.copernicus.eu/api
    key: <uid>:<key>
  ```
- `validate.py` checks the CDS section is present and that `cdsapi` imports.

### Variables

All three runoff variables are downloaded and stored:

| Short name | Long name              | Units                  | Notes                              |
|------------|------------------------|------------------------|------------------------------------|
| `ro`       | total runoff           | m of water equivalent  | Used for runoff target             |
| `sro`      | surface runoff         | m of water equivalent  | Stored for future use              |
| `ssro`     | sub-surface runoff     | m of water equivalent  | Used as recharge proxy             |

ERA5-Land accumulated fields reset at 00 UTC each day.

### Period

1979 through "present minus 3 months" (CDS lag for ERA5-Land).

### Hourly → Daily

Implementation: load hourly, compute hourly increments via `.diff('time')`
to handle the 00 UTC accumulation reset cleanly, then `.resample(time='1D').sum()`.
The diff-based approach avoids edge cases around the daily reset boundary
better than a "value at 00 UTC of next day" snapshot.

### Daily → Monthly

`.resample(time='1ME').sum()` on the daily file.

### Datastore Layout

```
<datastore>/era5_land/
  hourly/
    era5_land_<var>_<year>.nc          # raw downloads, kept for reproducibility
  daily/
    era5_land_daily_<year>.nc          # all 3 vars in one file per year
  monthly/
    era5_land_monthly_<start>_<end>.nc # all 3 vars, full record
```

### Submission Strategy

CDS jobs are queued and may take minutes to hours. The fetch module
submits per-year, per-variable requests. Yearly granularity balances job
runtime vs. resilience to interruption. Successful per-year hourly files
trigger immediate consolidation into the daily file for that year, so
partial progress is durable.

### CF Compliance

Daily and monthly NCs follow the same CF-1.6 patterns as `watergap22d`
and `reitz2017` (per project memory): explicit CRS, `grid_mapping` on
each variable, time `units`/`calendar` attributes, atomic writes via
temp-file-and-rename.

### Incremental Behavior

Skip download for any year whose hourly file already exists; skip
consolidation if the corresponding daily/monthly file is current.

## GLDAS Fetch Module

### Access

- Library: `earthaccess` (NASA EDL credentials already configured)
- Product: `GLDAS_NOAH025_M` v2.1, monthly, 0.25° global, 2000-present

### Variables

`Qs_acc` (storm surface runoff, kg/m²) and `Qsb_acc`
(baseflow-groundwater runoff, kg/m²) are downloaded. A derived variable
`runoff_total = Qs_acc + Qsb_acc` is computed at consolidation time and
stored alongside the originals.

### Download

Granules are global, ~few MB each. Server-side spatial subsetting via
earthaccess is not used (matching the existing MERRA-2 pattern). Granules
are downloaded in full and clipped to the project bbox at consolidation
time.

### Datastore Layout

```
<datastore>/gldas_noah_v21/
  raw/                              # original monthly global granules
  gldas_noah_v21_monthly.nc         # consolidated, bbox-clipped, CF-compliant
```

## Aggregation

Existing `aggregate/gdptools_agg.py` patterns are reused. Both new
monthly NCs are aggregated to HRU polygons via `gdptools` area-weighting,
producing per-HRU monthly time series. Weight caches land in the standard
`<project>/weights/` directory.

## Runoff Target Builder

`src/nhf_spatial_targets/targets/run.py` implements the multi-source
minmax target:

1. Load HRU-aggregated monthly runoff for ERA5-Land (`ro`) and GLDAS
   (`runoff_total`).
2. Harmonize units to mm/month, then convert to cfs using HRU area and
   days-in-month:
   - ERA5-Land `ro` (m water-eq) × 1000 → mm/month
   - GLDAS `Qs_acc + Qsb_acc` (kg/m² over month) ≡ mm/month directly
3. Per HRU, per month: `lower_bound = min(era5, gldas)`,
   `upper_bound = max(era5, gldas)`.
4. Write `runoff_target.nc` with `lower_bound` and `upper_bound`
   variables, dims `(hru, time)`, CF-compliant.

## Recharge Target Update

`targets/rch.py` gains ERA5-Land as a third source. `ssro` (m water-eq)
is summed monthly→annual, then normalized 0-1 over 2000-2009 alongside
Reitz2017 and WaterGAP 2.2d, after which `multi_source_minmax` runs over
the three normalized series. Catalog `recharge` sources list becomes
`[reitz2017, watergap22d, era5_land]`.

## AET Target Update

`targets/aet.py` source list drops `nhm_mwbm`. Now MOD16A2_v061 + SSEBop
only. Range method unchanged (`multi_source_minmax`).

## Validation

`validate.py` additions:

- Check `.credentials.yml` has a `cds:` section with non-empty `key`
- Check `cdsapi` is importable
- Catalog presence checks for `era5_land` and `gldas_noah_v21_monthly`

## Tests

| File                       | Coverage                                                                  |
|----------------------------|---------------------------------------------------------------------------|
| `tests/test_era5_land.py`  | hourly→daily diff/resample logic, daily→monthly, bbox/grid snap, atomic write |
| `tests/test_gldas.py`      | bbox clip, `runoff_total` derivation, CF metadata                         |
| `tests/test_run_target.py` | unit harmonization, multi-source minmax over synthetic inputs             |
| `tests/test_catalog.py`    | drop MWBM assertions, add new-source assertions                           |

Network-dependent tests are marked `@pytest.mark.integration`.

## Build Sequence

1. Catalog edits (`sources.yml`, `variables.yml`) + `CLAUDE.md` update
2. `cdsapi` dependency in `pixi.toml`; credentials handling in `validate.py`
3. `fetch/era5_land.py` (download → daily → monthly) + tests
4. `fetch/gldas.py` (download → consolidation) + tests
5. Aggregation wiring for both new sources
6. `targets/run.py` implementation + tests
7. `targets/aet.py` source-list edit
8. `targets/rch.py` ERA5-Land integration

## Open Items

- Confirm the exact "present minus N months" lag for ERA5-Land at
  implementation time (CDS publishes the current lag in their docs;
  typically 2-3 months).
- Decide final target period for the runoff target. The recommended
  default is the intersection of available data: 2000-present (matches
  GLDAS v2.1 start). Final value set in `variables.yml` `runoff.period`.
