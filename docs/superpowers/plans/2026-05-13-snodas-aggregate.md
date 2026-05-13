# SNODAS SWE aggregator (PR 1 of 2 for issue #101)

## Context

Issue #101 (umbrella) tracks SWE-category aggregate + target wiring. The
fetch side has landed: `fetch_snodas` now produces per-year consolidated
CF-1.6 NetCDFs at `<datastore>/snodas/daily/snodas_daily_<year>.nc`
(int16, `_FillValue=-9999`, daily chunks), and `fetch_daymet` verifies
pre-staged zarr stores at `{daymet_root}/daymet_{na,hi,pr}.zarr`.

The user has SNODAS daily NCs ready on disk and wants to start
aggregating. Per the decisions made when this plan was drafted:

- **This PR (PR 1)** wires up the SNODAS aggregator only. SNODAS fits the
  existing `SourceAdapter`/`aggregate_source` pattern cleanly (per-year
  NCs, single `swe` var, WGS84, no per-pixel masking — `_FillValue` is
  decoded to NaN at open time by xarray's `mask_and_scale=True`).
- **PR 2 (deferred)** will add `aggregate/daymet.py` (NA region only)
  using a custom zarr-aware loop that reuses the driver's helpers
  (`compute_or_load_weights`, `aggregate_variables_for_batch`,
  `_atomic_write_netcdf`, `sortby id_col`, `update_manifest`) but opens
  zarr stores instead of NCs. Splitting keeps SNODAS landable today and
  keeps the daymet review focused on the zarr machinery.

The SWE target builder (`targets/swe.py`) and the remaining SWE sources
(ERA5-Land `sd`, Margulis WUS-SR) stay tracked under #101 for separate
PRs after these two.

## Approach (SNODAS)

`aggregate/snodas.py` is a thin module that declares a `SourceAdapter`
and wraps `aggregate_source`. Everything else — per-year file
enumeration, weight caching, gdptools aggregation, CF attrs, id_col sort
on emission, manifest update — is handled by the shared driver at
[../../../src/nhf_spatial_targets/aggregate/_driver.py](../../../src/nhf_spatial_targets/aggregate/_driver.py).

Adapter settings:

- `source_key="snodas"`
- `variables=("swe",)` — single native variable; native units `kg m-2`
  (≡ mm), `cell_methods: "time: point"` per
  [catalog/sources.yml:773-781](../../../catalog/sources.yml).
- `source_crs="EPSG:4326"` — WGS84, 30 arcsec CONUS grid.
- `files_glob="daily/snodas_daily_*.nc"` — relative to
  `project.raw_dir("snodas")` = `<datastore>/snodas/`. The consolidated
  NCs live under the `daily/` subdir per `fetch.snodas`.
- `stat_method="mean"` — no `pre_aggregate_hook`, so the driver sees
  pixels with `_FillValue=-9999` already decoded to NaN by xarray. Per
  CLAUDE.md's "Aggregation Transformation Policy", that's the
  "geometric partial coverage / true upstream gaps" case — `mean` is
  correct, and HRUs that straddle the CONUS edge will be honestly NaN.
  NN-fill (if any) is a target-stage concern, not an aggregator
  concern.
- No `pre_aggregate_hook` or `post_aggregate_hook`. Coords (`time`,
  `lat`, `lon`) are CF-tagged by `fetch/consolidate.py:apply_cf_metadata`,
  so `detect_coords` picks them up without overrides.

The driver enforces grid-shape invariance across years
([_driver.py:734-774](../../../src/nhf_spatial_targets/aggregate/_driver.py))
and `id_col` canonical sort at emission
([_driver.py:439](../../../src/nhf_spatial_targets/aggregate/_driver.py))
— SNODAS's rows/cols are stable across years (the known sub-pixel drift
is within-year and absorbed at consolidation time per #114/#115), so
the invariance check will pass.

Output: one NC per year at
`<project>/data/aggregated/snodas/snodas_<year>_agg.nc`, with HRU dim
`id_col` ascending, native `swe` in `kg m-2`, CF-1.6 globals, and a
manifest entry merged into `manifest.json` keyed at `sources.snodas`.

## Files added / modified

**New:**

- `src/nhf_spatial_targets/aggregate/snodas.py` — `ADAPTER =
  SourceAdapter(...)` + thin `aggregate_snodas(fabric_path, id_col,
  workdir, batch_size=500, period=None)` wrapper that delegates to
  `aggregate_source`. Pattern: see
  `src/nhf_spatial_targets/aggregate/nldas_noah.py` (minimal no-hook
  adapter) and `src/nhf_spatial_targets/aggregate/gldas.py` (uses
  subdir in `files_glob`, precedent for `daily/...`).

- `tests/test_aggregate_snodas.py` — adapter-declaration assertions
  (`source_key`, `output_name`, `variables`, `files_glob`,
  `source_crs`, `stat_method`, hook-absence), catalog round-trip
  (`catalog.source("snodas")` resolves; declared variable in catalog
  variables), CLI registration check
  (`cli.aggregate_snodas is aggregate_snodas`), and signature check
  (`period` kwarg with default `None`). End-to-end driver coverage
  lives in `tests/test_aggregate_driver_per_year.py` and is
  source-agnostic.

- `notebooks/aggregated/inspect_aggregated_snodas.ipynb` — opens
  per-year aggregated NCs, converts `kg m-2 → inches` (× 1/25.4)
  inline for PRMS-units sanity-check, plots daily / monthly HRU
  choropleths in both unit systems, a histogram, multi-year
  representative-HRU timeseries, a NaN-coverage map, and a magnitude
  validation (winter-peak CONUS-mean SWE should land in the
  tens-of-mm range — memory `feedback_validate_magnitudes`). Mirrors
  `inspect_aggregated_snow_covered_area.ipynb` for layout / cells.

**Modified:**

- `src/nhf_spatial_targets/cli.py` — imports `aggregate_snodas`,
  adds an `agg snodas` subcommand (with optional `--period`), and
  registers SNODAS in the `agg all` sequence.

- `CLAUDE.md` — `pixi run nhf-targets agg snodas --project-dir ...`
  added to the command list under "Environment & Commands".

- `docs/superpowers/plans/2026-05-13-snodas-aggregate.md` (this file).

**Not changed:**

- `catalog/sources.yml` — the SNODAS entry is complete (variables,
  units, cell_methods, CRS, period).
- `aggregate/_driver.py` or `aggregate/_adapter.py` — the existing
  machinery covers this source.
- `defaults.py` — SNODAS is one of the four SWE sources, but per
  decision the SWE target block lands with `targets/swe.py` (a later
  PR), not in this PR.

## Verification

End-to-end on a project with SNODAS NCs already in the datastore:

```bash
pixi run nhf-targets agg snodas --project-dir <project-dir>
```

Expected: one `snodas_<year>_agg.nc` per year under
`<project>/data/aggregated/snodas/`; a re-run is a no-op (skips every
year because the per-year NCs exist). Then:

- Spot-check one year in the inspect notebook: open the agg NC, assert
  HRU dim monotonic ascending on `id_col`, plot a few HRUs' daily SWE
  timeseries, sanity-check the CONUS-area-weighted monthly mean against
  the SNODAS CONUS-mean published value (typical winter peak ≈ tens of
  mm averaged over CONUS).
- Verify `manifest.json` has `sources.snodas` with `output_files`
  enumerating the per-year NCs, `fabric_sha256` matching `fabric.json`,
  and `period` like `2003-10-01/2024-12-31`.

Automated:

```bash
pixi run -e dev fmt && pixi run -e dev lint
pixi run -e dev test -k snodas
pixi run -e dev test  # full suite at end of task
```

## What's NOT in this PR (deferred)

- `aggregate/daymet.py` (NA region) — PR 2; needs a custom zarr-aware
  loop because Daymet is a single multi-year zarr per region, not
  per-year NCs, and uses Lambert Conformal Conic (not WGS84). Will
  reuse driver helpers but not `aggregate_source` directly.
- ERA5-Land `sd` extension, Margulis WUS-SR aggregator,
  `targets/swe.py` — separate PRs under #101.
- Updates to `defaults.py` for the SWE target block — lands with
  `targets/swe.py`.
