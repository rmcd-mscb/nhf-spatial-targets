# Plan: SWE Calibration Target — Fetch-Layer Scaffolding

> Tracking issue: #99
> Plan saved 2026-05-12.

## Context

The nhf-spatial-targets pipeline currently builds five calibration targets
(runoff, AET, recharge, soil moisture, snow-covered area). We are adding a
sixth: **snow water equivalent (SWE)**. Four sources are in scope:

1. **Daymet** — V4 R1 daily 1 km zarr stores, already on disk at
   `/caldera/hovenweep/projects/usgs/water/impd/nhgf/data_creation/daymet/source/zarr/{daymet_na.zarr,daymet_hi.zarr,daymet_pr.zarr}`.
   Each zarr exposes `swe` plus prcp/srad/tmax/tmin/vp.
2. **SNODAS** — NSIDC G02158, daily CONUS 1 km, via `earthaccess`.
3. **ERA5-Land `sd`** — snow depth water equivalent. Extend the *existing*
   `fetch/era5_land.py`; same CDS, same module, additional variable.
4. **Margulis Western US Snow Reanalysis** (NSIDC-0719) — daily 90 m posterior
   SWE, water years 1985–2021. **Oregon-fabric only** (carries a new
   `fabric_scope:` catalog field; enforcement deferred to the target builder).

**This PR scope: fetch routines + catalog wiring + tests only.** Aggregate,
target builder, defaults block, and full target wiring are deferred to four
follow-up issues filed at merge time. Per the user direction: "start with the
fetch routines and then create separate issues to finish the workflow for
each dataset."

## Critical files to modify

- `catalog/sources.yml` — extend `era5_land`, add `daymet`, `snodas`,
  `margulis_wus_sr`.
- `catalog/variables.yml` — add `snow_water_equivalent` target entry.
- `src/nhf_spatial_targets/fetch/era5_land.py` — add `sd` variable +
  instantaneous-vs-accumulated dispatch.
- `src/nhf_spatial_targets/fetch/daymet.py` — **new**, verify-only.
- `src/nhf_spatial_targets/fetch/snodas.py` — **new**, earthaccess.
- `src/nhf_spatial_targets/fetch/margulis_wus_sr.py` — **new**, earthaccess.
- `src/nhf_spatial_targets/cli.py` — imports + `fetch all` tuple + 3 new
  per-source `@fetch_app.command(...)` blocks + skip list.
- `tests/test_fetch_daymet.py`, `tests/test_fetch_snodas.py`,
  `tests/test_fetch_margulis_wus_sr.py` — **new**.
- `tests/test_era5_land.py`, `tests/test_catalog.py`, `tests/test_cli.py` —
  extend.

## Reference patterns to reuse (file paths)

- **Verify-only** (Daymet): mirror
  [`src/nhf_spatial_targets/fetch/mwbm_climgrid.py`](../../../src/nhf_spatial_targets/fetch/mwbm_climgrid.py)
  — stability check, sha256 fingerprint, manifest registration, idempotency
  fast-path.
- **earthaccess + per-year consolidation + flock manifest** (SNODAS,
  Margulis): mirror
  [`src/nhf_spatial_targets/fetch/era5_land.py`](../../../src/nhf_spatial_targets/fetch/era5_land.py)
  (manifest flock `_update_manifest` lines 710–810) and
  [`src/nhf_spatial_targets/fetch/gldas.py`](../../../src/nhf_spatial_targets/fetch/gldas.py)
  (earthaccess search/download).
- **Auth**: `from nhf_spatial_targets.fetch._auth import earthdata_login`.
- **Period parsing/clamping**:
  `nhf_spatial_targets.fetch._period.years_in_period`.
- **CF metadata + write helpers**: `fetch/consolidate.py` —
  `apply_cf_metadata`, `resolve_license`, `_write_netcdf`.
- **SourceAdapter / catalog**: `from nhf_spatial_targets import catalog as
  _catalog; _catalog.source(key)` — never read YAML directly elsewhere.

## Implementation steps

### Step 1 — Catalog edits

**`catalog/variables.yml`** — append `snow_water_equivalent` block after
`snow_covered_area` (after line 183). `range_method: multi_source_minmax`,
`normalize: false`, `time_step: daily`, sources list = `[daymet, snodas,
era5_land, margulis_wus_sr]`. Document in `range_notes:` that Margulis WUS-SR
is fabric-scoped to Oregon and excluded for non-OR fabrics at target build
time. PRMS variable: `pkwater_equiv` (confirm in target-builder follow-up).

**`catalog/sources.yml`** — four edits:

1. Add `SNOW WATER EQUIVALENT` section banner (style mirrors lines 8–10).
2. **Extend `era5_land`** (lines 12–62): append a fourth variable `sd` with
   `long_name: snow depth water equivalent`, `cf_units: "m"`,
   `cell_methods: "time: point"`, and a note that this is instantaneous
   (NOT accumulated; uses the `.mean()` reducer, not diff-of-accumulations).
3. **Add `daymet`** with `access.type: zarr_verify`, three regional store
   paths under `stores: {na, hi, pr}` keyed via `{daymet_root}` template,
   `period: "1980/2023"`, all six variables (`swe, prcp, tmax, tmin, srad,
   vp`). Operator-staged location, no download.
4. **Add `snodas`** with `access.type: nasa_nsidc`, `short_name: G02158`,
   `version: "1"`, `period: "2003/present"`, bbox NWSE clipped to
   CONUS+contributing. Single variable `swe`.
5. **Add `margulis_wus_sr`** with `access.type: nasa_nsidc`,
   `short_name: WUS_UCLA_SR` (verify at impl time),
   `doi: 10.5067/PP7T2GBI52I2`, `period: "1985/2021"`. **New optional field
   `fabric_scope: {fabrics: [or], notes: ...}`** documenting Oregon-only
   scope; enforcement is at target-build time.

### Step 2 — ERA5-Land `sd` extension (small, isolated)

In `fetch/era5_land.py`:

- Line 41: `VARIABLES = ("ro", "sro", "ssro", "sd")`.
- Lines 104–108: add `"sd": "snow_depth_water_equivalent"` to
  `_VARIABLE_REQUEST_NAME`.
- Add `_VARIABLE_KIND = {"ro": "accumulated", "sro": "accumulated",
  "ssro": "accumulated", "sd": "instantaneous"}` after that table.
- Add `hourly_to_daily_instantaneous(da)`: `da.resample(time="1D").mean()`,
  preserve attrs. Short docstring contrasts with the accumulated path.
- Refactor `daily_to_monthly` to take `kind: str = "accumulated"`; branch
  `.sum()` vs `.mean()`. Update the existing call site in `consolidate_year`
  to pass `kind=_VARIABLE_KIND[var]`.
- In `consolidate_year` (lines 408–429), dispatch
  `hourly_to_daily` vs `hourly_to_daily_instantaneous` by `_VARIABLE_KIND[var]`.
- Update title strings at lines 434 / 477 to "ERA5-Land daily runoff and snow".
- **No CLI changes** — `nhf-targets fetch era5-land` picks up `sd`
  automatically because it iterates `VARIABLES`.

### Step 3 — `fetch/daymet.py` (new, verify-only)

```python
def fetch_daymet(
    workdir: Path,
    period: str,
    *,
    source_path: Path | None = None,
    region: str = "all",  # "na" | "hi" | "pr" | "all"
) -> dict
```

Resolution order for zarr root: `source_path` arg → `config.yml: daymet_root`
→ raise `FileNotFoundError` with operator instructions. Per region: open
zarr via `xarray.open_zarr`, assert `swe`/`time`/`x`/`y`/
`lambert_conformal_conic` present, validate time coord within `period`,
compute cheap fingerprint (sha256 of `.zmetadata` + total directory byte
count — NOT a full-data hash, the zarrs are 100s of GB), record under
`manifest["sources"]["daymet"]["regions"][region]`. Idempotent: identical
fingerprint = no-op. Stability check pattern from mwbm_climgrid.

### Step 4 — `fetch/snodas.py` (new, earthaccess)

```python
def fetch_snodas(
    workdir: Path,
    period: str,
    *,
    worker_index: int = 0,
    n_workers: int = 1,
) -> dict
```

`earthdata_login` → year-sharded round-robin (copy era5_land's helper
pattern) → `earthaccess.search_data(short_name=G02158, temporal=..., bounding_box=bbox)`
→ download tarballs to `<datastore>/snodas/raw/<year>/` → extract SWE
binary+`.Hdr`, build `xr.DataArray` (with scale + fill mask per SNODAS data
dictionary) → daily NC `<datastore>/snodas/daily/snodas_daily_{year}.nc` →
monthly NC `<datastore>/snodas/monthly/snodas_monthly_{year}.nc`. Use
`apply_cf_metadata` + atomic write. **Flock-protected `_update_manifest`**
(copy from era5_land).

> The SNODAS native format (int16 binary + ENVI-style `.Hdr`) needs
> characterisation in a notebook before consolidation logic is finalised —
> per CLAUDE.md "characterise the data first". The fetch PR may stage just
> the search/download/manifest path and defer consolidation to a follow-up
> sub-step within the same PR, with the notebook checked into
> `notebooks/`.

### Step 5 — `fetch/margulis_wus_sr.py` (new, earthaccess)

```python
def fetch_margulis_wus_sr(workdir: Path, period: str) -> dict
```

`earthdata_login` → `parse_period` clamped to `(1985, 2021)` →
`earthaccess.search_data(short_name="WUS_UCLA_SR", bounding_box=ws.fabric["bbox_buffered"], temporal=...)`
→ download to `<datastore>/margulis_wus_sr/raw/`, drop zero-byte files →
per-year NC at `<datastore>/margulis_wus_sr/daily/margulis_wus_sr_daily_{year}.nc`.
Flock-protected manifest update. Module docstring + manifest entry record
`fabric_scope` per the catalog. Fetch does not enforce the scope (raw
downloads are reusable); the target builder will.

### Step 6 — CLI wiring (`src/nhf_spatial_targets/cli.py`)

- Imports near lines 372–382 (alphabetical): `fetch_daymet`,
  `fetch_margulis_wus_sr`, `fetch_snodas`.
- `fetch all` sources tuple (lines 385–397): add `("daymet", "daymet",
  fetch_daymet)`, `("snodas", "snodas", fetch_snodas)`,
  `("margulis-wus-sr", "margulis_wus_sr", fetch_margulis_wus_sr)`.
- Skip-list at ~line 432 (currently `mwbm-climgrid`): add `daymet` and
  `margulis-wus-sr` so `FileNotFoundError` yields a yellow "skipped" instead
  of failing `fetch all` (relevant for non-OR projects and Daymet not yet
  staged).
- Three new `@fetch_app.command(...)` blocks following the
  `fetch_mwbm_climgrid_cmd` pattern at lines 1021+:
  - `fetch daymet` — `--source-path`, `--region`.
  - `fetch snodas` — `--worker-index`, `--n-workers`.
  - `fetch wus-sr` (CLI name kebab; catalog key `margulis_wus_sr`).
- `fetch era5-land` docstring (line 809) updated to mention `sd`.

### Step 7 — Tests

All hermetic — no network, no real Earthdata creds. Mock `cdsapi`,
`earthaccess`, and any file-format readers.

- **`tests/test_fetch_daymet.py`** (mirrors test_fetch_mwbm_climgrid.py):
  missing-zarr error message, period rejection, region=all/single,
  initial-registration, idempotency on second call, re-fingerprint after
  replacement, missing `swe` variable rejected, `--source-path` overrides
  config, corrupt manifest fails fast.
- **`tests/test_fetch_snodas.py`** (mirrors test_gldas.py + test_modis.py):
  empty-search error, partial-download error, per-year daily+monthly
  produced, consolidation idempotent on mtime, period<2003 rejected,
  manifest accumulates years, worker partitioning.
- **`tests/test_fetch_margulis_wus_sr.py`**: period clamping, empty search,
  partial download, zero-byte drop, per-year NC, manifest records
  `fabric_scope: ["or"]`, auth failure surfaces cleanly.
- **`tests/test_era5_land.py`** (extend): `hourly_to_daily_instantaneous`
  means-per-day + attrs, `consolidate_year` kind dispatch produces sum for
  ro and mean for sd, `daily_to_monthly` kind branch, defensive check that
  every entry in `VARIABLES` is keyed in `_VARIABLE_KIND`.
- **`tests/test_catalog.py`** (extend): `snow_water_equivalent` variable
  present with 4 sources; `daymet`, `snodas`, `margulis_wus_sr` source
  entries with expected fields; `era5_land.variables` now contains `sd`
  with `cell_methods: "time: point"`; `margulis_wus_sr.fabric_scope.fabrics
  == ["or"]`.
- **`tests/test_cli.py`** (extend if pattern exists): `--help` smoke for the
  three new commands.

## Verification

```bash
pixi run -e dev fmt
pixi run -e dev lint
pixi run -e dev test

# Catalog inspection
pixi run catalog-sources   | grep -E "daymet|snodas|margulis|sd"
pixi run catalog-variables | grep -i snow_water

# CLI help smoke (no network)
pixi run nhf-targets fetch daymet --help
pixi run nhf-targets fetch snodas --help
pixi run nhf-targets fetch wus-sr --help
pixi run nhf-targets fetch era5-land --help   # docstring now mentions sd
```

Manual end-to-end smoke on a real project (separate from automated tests):

```bash
# In an existing nhf-runs project with credentials materialized:
pixi run nhf-targets fetch daymet --project-dir /path/to/project \
    --source-path /caldera/hovenweep/projects/usgs/water/impd/nhgf/data_creation/daymet/source/zarr \
    --region na --period 1980/2023
# Verify: manifest.json gains a sources.daymet.regions.na entry.

pixi run nhf-targets fetch era5-land --project-dir /path/to/project --period 2020/2020
# Verify: <datastore>/era5_land/daily/era5_land_daily_2020.nc and
# <datastore>/era5_land/monthly/era5_land_monthly_2020.nc each contain
# variables ro, sro, ssro, sd.
```

## Follow-up issues to file at merge

1. **SWE: Daymet aggregate + target wiring.** `aggregate/daymet.py` (3
   regional zarrs → per-region per-year HRU NCs, concat in target builder),
   `defaults.py` SWE block, `cli.py` agg dispatch.
2. **SWE: SNODAS aggregate + target wiring.**
3. **SWE: ERA5-Land `sd` aggregate + target wiring.** Extend
   `aggregate/era5_land.py` to emit `sd` alongside `ro/sro/ssro`.
4. **SWE: Margulis WUS-SR aggregate + target wiring with `fabric_scope`
   enforcement.** Wire Oregon-only check (skip-or-raise for non-OR fabrics).
5. **SWE target builder (`targets/swe.py`).** Multi-source minmax over the
   four sources; honours `fabric_scope`; emits `swe.nc` and (optional)
   `swe_nn_filled.nc`. Open question whether normalisation matches
   `runoff` (none) or `soil_moisture` (calendar-month) — confirm against
   TM 6-B10 §SWE during this issue.

## Risks / open questions

- **SNODAS short_name/version + masked-vs-unmasked.** Confirm `G02158` vs
  `G02158_unmasked` against NSIDC before SNODAS consolidation lands.
- **SNODAS native format.** Int16 binary + ENVI `.Hdr` — consolidation code
  is non-trivial. Characterise in a notebook before finalising.
- **Margulis short_name.** `WUS_UCLA_SR` is a best guess from the NSIDC-0719
  page; verify via earthaccess CMR query.
- **Daymet fingerprint scope.** Sha256 of `.zmetadata` is cheap but could
  miss a structure-only change. Acceptable for verify-only; documented in
  module docstring.
- **ERA5-Land `sd` daily reducer = mean.** Defensible for slowly-varying
  snowpack; some operators may prefer 12 UTC point sampling. Document in
  catalog `notes`; revisit if the target builder needs different semantics.
- **`fabric_scope` enforcement is deferred.** This PR introduces the field
  and records it in manifests but does not enforce it on aggregate/target
  paths. The Margulis follow-up issue (#4) carries the enforcement.
- **`fetch all` silent skips.** Adding daymet + margulis-wus-sr to the
  "missing-file = skip" list means a misconfigured datastore proceeds with
  fewer sources than intended. Mitigation: log a summary tally at the end
  of `fetch all` distinguishing successes from skips.
- **No new dependencies expected.** xarray (existing) handles zarr;
  earthaccess (existing) handles NSIDC. If SNODAS binary parsing requires
  features not already in our rasterio stack, raise during implementation
  rather than this scaffolding PR.
