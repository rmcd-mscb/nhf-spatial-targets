# Plan: Margulis WUS-SR fetch-time consolidation into per-year CF NetCDFs

> Tracking issue: #111
> Template PR: #109 (`feat(snodas): decode raw .tars into per-year daily CF NetCDFs`)
> Plan saved 2026-05-13.

## Context

PR #109 (`feat(snodas): decode raw .tars into per-year daily CF NetCDFs`) added
a "consolidation" stage to the SNODAS fetcher: after raw downloads land for a
year, the fetcher decodes them into a single per-year CF-compliant NetCDF at
`<datastore>/snodas/daily/snodas_daily_<year>.nc`. That product is the file the
downstream aggregate adapter will eventually open with
`fetch.consolidate.open_consolidated`.

The Margulis Western US Snow Reanalysis (NSIDC-0719) fetcher
(`src/nhf_spatial_targets/fetch/margulis_wus_sr.py`) currently stops at raw
downloads — the module docstring at lines 13-14 explicitly defers
consolidation. The umbrella issue #101 groups consolidation, aggregate
adapter, target wiring, and `fabric_scope` enforcement into a single Margulis
sub-task (task 4 of 7).

Goal: pull **just the consolidation step** out of #101 task 4 into this issue
(#111) / branch / PR, mirroring the SNODAS PR #109 shape. Aggregate adapter,
target wiring, and `fabric_scope` enforcement stay in #101 for a follow-up.

## Decisions confirmed with user

1. **Scope**: consolidation **+ a characterization notebook** that opens a real
   NSIDC-0719 granule and validates the consolidated NC against source tiles.
2. **Spatial layout**: **mosaic to the full WUS source domain** (not clipped to
   fabric bbox). Keeps consolidated NCs fabric-independent so they're reusable
   across projects sharing a datastore — same property the SNODAS consolidated
   NCs have.
3. **Time axis**: **rebuild to calendar year** (Jan–Dec) to match the SNODAS
   `snodas_daily_<year>.nc` layout. Calendar year *X* is assembled by reading
   the Oct–Dec portion of water-year *X* and the Jan–Sep portion of water-year
   *X+1*.

## Approach

### 1. Branch + plan commit

`feature/111-margulis-consolidation`, branched from `main`. Commit this plan
to `docs/superpowers/plans/2026-05-13-margulis-consolidation.md` as the first
commit on the branch.

### 2. Update umbrella #101

Add a checkbox noting that the consolidation portion of task 4 is split out
to #111; the aggregate adapter / target wiring / `fabric_scope` enforcement
portions stay in #101.

### 3. Characterization notebook (do this *first*)

`notebooks/inspect_margulis_wus_sr.ipynb` — modeled on the existing
`visualize_*` notebooks. Per CLAUDE.md "characterise the data first" guidance:

- Open one real NSIDC-0719 granule, dump `ds.info()`, confirm:
  - actual variable name (`SWE_Post` vs `SWE` — catalog declares `SWE` but
    NSIDC docs use `SWE_Post`; the on-disk var may differ. Read the on-disk
    name then map to the catalog's declared name during consolidation —
    catalog wins).
  - units (`m`), coordinate names, fill-value convention.
  - tile geometry: lat/lon extents, resolution (90 m ≈ 0.000833°), how tiles
    are named, whether tiles share a single regular grid or each carries its
    own offset.
- Confirm one water-year granule covers Oct 1 – Sep 30 with **365/366 daily
  time steps**; verify the calendar-year rebuild boundary (Oct–Dec from WY X
  joined with Jan–Sep from WY X+1) by index-counting.
- Validate CONUS-mean SWE magnitude (~tens to low hundreds of mm peak;
  near-zero summer) before trusting catalog metadata.

The notebook output informs the consolidator's implementation details (var
rename map, dtype, fill value). If characterization reveals a tile-grid
mismatch we haven't anticipated, we revisit the design before writing code.

### 4. Consolidator implementation

Add the consolidation stage to
`src/nhf_spatial_targets/fetch/margulis_wus_sr.py`, **directly mirroring** the
SNODAS pattern in `src/nhf_spatial_targets/fetch/snodas.py:434`.

New functions in `fetch/margulis_wus_sr.py`:

- `consolidate_calendar_year_margulis_wus_sr(year, raw_root, daily_dir) -> Path`
  - Locates the relevant water-year granules in
    `<datastore>/margulis_wus_sr/raw/{year}/` and
    `<datastore>/margulis_wus_sr/raw/{year+1}/`.
  - **Idempotency**: skip rebuild when `out_path` exists and is newer than
    every input granule (mtime check — same gate as `snodas.py:494-509`).
  - Opens tiles with `xr.open_mfdataset(..., combine="by_coords")` and slices
    out the SWE variable for the calendar-year window.
  - Boundary handling: WY *X* contributes Oct–Dec of calendar year *X*; WY
    *X+1* contributes Jan–Sep of calendar year *X*. If a needed adjacent
    water year is missing from `raw_root`, raise `FileNotFoundError` with a
    clear "run fetch for year *X+1* first" message rather than silently
    emitting a partial NC.
  - Renames any on-disk variable name to the catalog-declared `SWE`.
  - Calls `apply_cf_metadata(ds, "margulis_wus_sr", "daily")` from
    `fetch/consolidate.py:164` — same call SNODAS uses. **This is the
    CF-compliance gate**; see the explicit CF checklist below.
  - Writes via the shared `_write_netcdf` helper (atomic temp-file + rename),
    with zlib level 4. Use the native float dtype initially; only switch to
    int16+scale if the notebook shows the float output is unwieldy in size.
  - Returns `daily_dir / f"margulis_wus_sr_daily_{year}.nc"`.

Wire it into `fetch_margulis_wus_sr`:

- After the download loop, for each calendar year fully covered by the
  downloaded raw water years, call `consolidate_calendar_year_margulis_wus_sr`
  synchronously and update the per-year manifest record with `daily_path` /
  `consolidated_utc` (success) or `consolidate_error` (failure — log + record
  but **don't abort fetch**, mirroring `snodas.py:735-751`).
- Update `_completed_years_from_manifest` to require **both** `n_granules > 0`
  **and** `daily_path` existing on disk (SNODAS dual-gate completion check).
  NSIDC-0719's water-year granules naturally span two calendar years, so the
  manifest schema records both the source water-year `n_granules` count and
  the derived calendar-year `daily_path` — keep them on the same record keyed
  by calendar year, with `source_water_years: [WY_X, WY_X+1]`.

#### CF-1.6 compliance checklist (must hold for every consolidated NC)

Following the project's existing CF plan
(`docs/superpowers/plans/2026-03-17-cf-compliance.md`) and what
`apply_cf_metadata` already enforces — the consolidator must produce NCs
with **all** of these properties:

- **`Conventions = "CF-1.6"`** in global attrs (set by `apply_cf_metadata`).
- **Coordinate names normalized** to `time`, `lat`, `lon` with dim order
  `(time, lat, lon)`. NSIDC-0719 granules typically use
  `Latitude`/`Longitude`/`Day` — the consolidator must rename before calling
  `apply_cf_metadata` (which handles `latitude`/`longitude`/`valid_time` but
  not `Day` or capital-L variants — verify and rename in the notebook step).
- **`lat` attrs**: `standard_name="latitude"`, `units="degrees_north"`,
  `axis="Y"`. Latitude **descending** (north-to-south) per the existing
  convention used by SNODAS.
- **`lon` attrs**: `standard_name="longitude"`, `units="degrees_east"`,
  `axis="X"`. Longitude ascending.
- **`time` attrs**: `standard_name="time"`, `axis="T"`, encoded as
  `"days since 1970-01-01"` with calendar `"standard"`. Time monotonically
  ascending; `apply_cf_metadata`'s `time_step="daily"` branch leaves
  `time_bnds` off (point samples — matches SNODAS).
- **SWE variable attrs**: `units="m"` (from catalog `cf_units`),
  `long_name="posterior snow water equivalent"` (from catalog),
  `cell_methods="time: point"` (from catalog), `standard_name` added if the
  catalog declares one (Margulis does not — leave out rather than invent),
  `_FillValue` set, `grid_mapping="crs"`.
- **`crs` ancillary variable**: a scalar `int32` carrying WGS84 grid mapping
  attrs (`grid_mapping_name="latitude_longitude"`, full WKT in `crs_wkt`,
  plus the standard semi-major/inverse-flattening pair). Emitted by
  `apply_cf_metadata`'s default WGS84 branch.
- **Global provenance attrs**: `title`, `source` (catalog `name`),
  `institution`, `references` (DOI), and `history` (this-pipeline version
  stamp). `apply_cf_metadata` provides these.
- **Margulis-specific provenance** (added by the consolidator, mirroring
  `snodas_first_day_header`): a `margulis_source_water_years` global attr
  recording the WY range(s) that contributed to the calendar year. Use a
  JSON-encoded list so it round-trips through NetCDF text attrs cleanly.

A new compliance test
(`tests/test_fetch_margulis_wus_sr.py::test_consolidated_nc_is_cf_compliant`)
asserts each of the above on a synthetic-input consolidated NC. Optional:
spot-check with `cfchecks` / `compliance-checker` in the notebook step but do
not gate CI on an external tool.

No new shared helper is needed in `fetch/consolidate.py`: the existing
`apply_cf_metadata` + `_write_netcdf` cover the generic CF/atomic-write
concerns, and the Margulis-specific mosaic / water-year-to-calendar-year
logic lives in the source module (same factoring as `consolidate_year_snodas`).

### 5. Tests

`tests/test_fetch_margulis_wus_sr.py` (extend the existing file). New cases
mirror the SNODAS test set in `tests/test_fetch_snodas.py`:

- **Happy path** — pre-stage 2 synthetic water-year NetCDFs (small lat/lon
  grids, 365-day time axes) in a tmp datastore; call
  `consolidate_calendar_year_margulis_wus_sr` for the middle calendar year;
  assert variable name, units, dims, time axis sorted ascending, grid mapping
  present, source water years recorded in global attrs.
- **Idempotency** — second call when NC newer than inputs returns the same
  path without rewriting (compare mtime).
- **Missing adjacent water year** — only WY *X* on disk → consolidating CY *X*
  raises `FileNotFoundError` referencing the missing WY *X+1*.
- **Variable name remap** — synthetic granule with on-disk var `SWE_Post`
  consolidates to a dataset exposing `SWE` (catalog name).
- **CF-1.6 compliance** — `test_consolidated_nc_is_cf_compliant` asserts every
  item in the CF-1.6 checklist above (Conventions, coord names + attrs,
  variable attrs from catalog, `crs` ancillary var, time encoding,
  Margulis-specific provenance attrs).
- **Backfill workflow** — pre-stage raw granules + a manifest lacking
  `daily_path` for a calendar year; re-running `fetch_margulis_wus_sr` runs
  consolidation only (no re-download), updates manifest.

All tests stay offline — synthetic xarray Datasets written to tmp paths via
`ds.to_netcdf`, no `earthaccess.download` monkeypatching needed for the
consolidation cases (the existing tests already cover the download path).

### 6. PR

- Title: `feat(margulis): decode raw NSIDC-0719 granules into per-year CF NetCDFs`
- Body: closes #111, references PR #109 as the template, notes that aggregate
  adapter / target wiring / `fabric_scope` enforcement remain in #101 task 4.
- Pre-commit gate per CLAUDE.md:
  `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`.

## Critical files

- `src/nhf_spatial_targets/fetch/margulis_wus_sr.py` — add
  `consolidate_calendar_year_margulis_wus_sr` + wire into
  `fetch_margulis_wus_sr`.
- `src/nhf_spatial_targets/fetch/consolidate.py` — **read-only reuse** of
  `apply_cf_metadata` and `_write_netcdf`.
- `src/nhf_spatial_targets/fetch/snodas.py:434-609` — reference implementation
  pattern.
- `tests/test_fetch_margulis_wus_sr.py` — extend with consolidation cases
  (including `test_consolidated_nc_is_cf_compliant`).
- `notebooks/inspect_margulis_wus_sr.ipynb` — new characterization notebook.
- `catalog/sources.yml:790-846` — read-only (no catalog changes needed).
- `src/nhf_spatial_targets/cli.py:1196-1230` — read-only (existing
  `fetch margulis-wus-sr` command picks up consolidation automatically).

## Verification

1. **Unit tests pass**:
   `pixi run -e dev test tests/test_fetch_margulis_wus_sr.py -v`.
2. **Notebook validates**: open `notebooks/inspect_margulis_wus_sr.ipynb`, run
   end-to-end against one real granule pair on a project pointed at a real
   datastore. Confirm consolidated NC opens with
   `fetch.consolidate.open_consolidated`, has CF metadata, sensible
   spatial-mean SWE seasonal cycle (peak ~Feb–Apr, zero late summer).
3. **End-to-end fetch**: on a small calendar-year window
   (e.g. `--period 2000/2001`), run
   `pixi run nhf-targets fetch margulis-wus-sr --project-dir <test-project>`
   and confirm:
   - `<datastore>/margulis_wus_sr/raw/{2000,2001,2002}/` populated.
   - `<datastore>/margulis_wus_sr/daily/margulis_wus_sr_daily_2000.nc` and
     `..._2001.nc` written (2002 may stay raw-only if WY2003 isn't requested).
   - `<project>/manifest.json` shows `daily_path`, `consolidated_utc`,
     `source_water_years`, and the `fabric_scope` block per year.
4. **Idempotency**: rerun the same fetch; expect "daily NC up-to-date" log
   lines and no NC rewrites.
5. **Lint/format/test all clean**:
   `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`.
