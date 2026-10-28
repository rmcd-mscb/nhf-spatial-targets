# Margulis Western US Snow Reanalysis (NSIDC-0719)

The UCLA/Margulis Western US Snow Reanalysis is a daily 90 m posterior
SWE product over the Sierra Nevada, Cascades, Rockies and adjacent
ranges for water years 1985-2021 (Fang, Liu & Margulis, 2022;
doi:10.5067/PP7T2GBI52I2). NSIDC distributes it as collection
NSIDC-0719 (short_name `WUS_UCLA_SR`).

This source covers only the **Western US**. Since #309 there is no
catalog opt-in or `fabric.token` gate: the CMR search bbox is the
project fabric's buffered bbox, the aggregation driver skips fabric
batches outside the source grid (logging a per-source HRU-coverage
diagnostic, e.g. "overlaps 12 of 200 batches"), and the per-year
aggregated NCs carry honest NaN at uncovered HRUs. The SWE target's
NaN-aware combine then uses Margulis exactly where it has data — on a
CONUS fabric it tightens the bound at Western-US HRUs and drops out
elsewhere. A fabric with zero overlap skips the source entirely at agg
time with an INFO log. Raw downloads are reusable by any project
pointing at the same datastore.

Note the operational change from #309: `agg margulis-wus-sr` (and
`agg all`, which includes it) now requires fetched raw data like every
other source — without it the aggregator raises `FileNotFoundError`
instead of silently skipping. For a CONUS project this is a deliberate
results change: once fetched + aggregated, Margulis begins contributing
to Western-US HRUs in the SWE bound.

The fetch step ships the **search-and-download** path: granules land
in `<datastore>/margulis_wus_sr/raw/<year>/`. Concatenating the
per-water-year per-tile NetCDFs into a per-year CF NetCDF is handled
by the consolidator that runs after each year's downloads complete.

## Perennial-snow mask (#363)

Margulis holds glacier and perennial-snow pixels at multi-metre SWE through
every summer (Mt Hood about 2.1-2.3 m on Sep 30 each year; Rainier and Baker
about 100 m). PRMS cannot carry snow from one water year into the next, so at
those HRUs the raw product pulls the SWE target's upper bound to values the
model can never reach. This is a trait of the product, not a registration
error: the Margulis-to-HRU mapping was checked and is correct.

Perennial pixels are therefore removed **before** they are averaged to HRUs.

- **The rule** (catalog `sources.yml[margulis_wus_sr].perennial_snow_mask`):
  a pixel is perennial if its SWE is at least `swe_threshold_m` (0.1 m) on
  `check_month_day` (Sep 30) in **more than** `min_year_fraction` (0.5) of the
  years with a finite value there. One static mask covers the whole record.
- **The mask file** is
  `<datastore>/margulis_wus_sr/derived/perennial_snow_mask.nc`, with `perennial`
  (0/1), `n_years_held` and `n_years_valid` on the consolidated lat/lon grid.
  Its global attrs record the rule, the input years, the pixel count and area,
  `perennial_snow_mask_build_key` and `perennial_snow_mask_fingerprint`.
- **How it is applied.** `agg margulis-wus-sr` builds the mask if needed, then
  drops perennial cells from the gdptools weight table in memory
  (`SourceAdapter.weight_filter`; the cached weight CSVs are never rewritten
  and stay valid). It does not use `masked_mean`; see "Static pixel masks under
  `mean`" in `docs/architecture/transformation-pipeline.md`.
- **Effect on HRUs.**
  - An HRU with no perennial pixel in its footprint is **unchanged,
    bit for bit**.
  - A partly perennial HRU becomes the area-weighted mean of its remaining
    pixels.
  - A fully perennial (fully glaciated) HRU becomes NaN on every day, never
    0.0. Margulis therefore drops out of the multi-source bound there and
    `n_sources` falls by one.
  - Lake and coast HRUs are unaffected: the mask only flags grid pixels that
    hold snow, and an HRU Margulis does not cover stays NaN as before (#309).
- **Provenance.** Each per-year aggregated NC carries four
  `perennial_snow_mask_*` attrs (fingerprint, threshold, check day, year
  fraction), the SWE target NC carries
  `margulis_wus_sr_perennial_snow_mask_fingerprint`, and the manifest records
  the mask parameters.

### Rebuilds and self-healing

- **Two keys.** The *build key* (`perennial_snow_mask_build_key`, a hash of the
  rule plus the input years) decides when the mask file itself is rebuilt. The
  *fingerprint* (`perennial_snow_mask_fingerprint`, a hash of the rule, the grid
  and the perennial flags) is what is stamped on aggregated NCs and the SWE
  target, so downstream work re-runs exactly when the removed-pixel set or the
  rule changes.
- **Cost of a year-set change.** Adding or removing a Margulis year rebuilds
  the mask (it reads every Sep 30 slice) and changes the Margulis consolidate
  step in every project's manifest (run `maintenance rebuild-manifest`).
  Existing years re-aggregate, and the SWE target rebuilds, **only if a mask
  cell (or the rule) changed**; otherwise the fingerprint is unchanged and
  nothing downstream re-runs.
- **After re-consolidating a year Y with corrected data.** The aggregate skip is
  by file existence, so the operator deletes:
  1. `<datastore>/margulis_wus_sr/derived/perennial_snow_mask.nc`
  2. `<project>/data/aggregated/margulis_wus_sr/margulis_wus_sr_<Y>_agg.nc` in
     **every** project on the datastore
  3. `<project>/targets/.swe_intermediates/` in each such project (the SWE
     cache key changes only when the mask content changes)

  Then, per project:

  ```bash
  nhf-targets agg margulis-wus-sr --project-dir <project>
  nhf-targets run --target snow_water_equivalent --project-dir <project>   # pixi run run-swe
  nhf-targets maintenance rebuild-manifest --project-dir <project>
  nhf-targets validate --project-dir <project>
  ```

  If the rebuilt mask's cells changed, every year re-aggregates automatically.
- **Manifest drift elsewhere.** A mask build or rebuild in the shared datastore
  changes the Margulis consolidate step in **every** project's manifest,
  including projects that do not use Margulis (e.g. gfv2). Their publish gate
  reports drift until `nhf-targets maintenance rebuild-manifest` is run there.
- Pre-mask per-year aggregates (no fingerprint attr) and per-year aggregates
  stamped with an older fingerprint are re-aggregated on the next
  `agg margulis-wus-sr`. Stale SWE year intermediates rebuild on the next
  `run-swe` because the fingerprint is part of their cache key.
- A mixed pre-/post-mask set of per-year aggregates, or an entirely pre-mask
  set, makes the SWE build raise,
  naming the years that lack the fingerprint (or listing each year's
  fingerprint if they disagree). Until then `run-swe` cannot proceed; re-run `agg margulis-wus-sr` to fix it.
- **Manifest with a mixed set.** `rebuild-manifest` records a mask param on the
  Margulis aggregate step only when every per-year aggregate agrees on it. If
  some years are unstamped, or the fingerprints differ, the param is left out
  and a WARNING names each file's value (#366).
- **Publish gate (#366).** `release.publish._preflight_margulis_mask_current`
  refuses to publish (always fatal, no override) unless all four agree:
  1. every per-year aggregate is stamped, with one fingerprint;
  2. the stamped rule equals the current catalog `perennial_snow_mask` rule;
  3. that fingerprint equals the datastore mask NC's (a missing mask NC fails);
  4. every SWE target product built with Margulis (`margulis_wus_sr` in its
     `member_keys` or in the configured SWE `sources`) carries
     `margulis_wus_sr_perennial_snow_mask_fingerprint` with that same value.

  The usual cause of check 4 failing is re-aggregating Margulis without
  re-running `run-swe`. The fix is always the sequence above: `agg`, `run`,
  `rebuild-manifest`, `validate`. The gate is a no-op for projects with no
  Margulis aggregates (e.g. gfv2).
- Verification: the Perennial-snow mask section of
  `notebooks/aggregated/inspect_aggregated_swe.ipynb` (checks 1-5) and the
  `n_sources` section of `notebooks/targets/inspect_target_swe.ipynb`
  (check 6). Both compare against the pre-mask archive under
  `<project>/archive/margulis_premask_2026-10-28/` when it exists.

## Prerequisites

- NASA Earthdata account with credentials materialized into `~/.netrc`
  (`nhf-targets materialize-credentials --project-dir <project>`).
- Project's `fabric.json` present with a `bbox_buffered` covering the
  Oregon HRU set. The fetch reads this bbox and passes it as the CMR
  `bounding_box` constraint, so only granules overlapping the Oregon
  domain are downloaded.

## Procedure

```bash
nhf-targets fetch margulis-wus-sr \
    --project-dir <project> \
    --period 1985/2021
```

The command:

1. Logs in to NASA EDL via `earthaccess`.
2. Reads the publisher window from `sources.yml[margulis_wus_sr].period`
   and rejects out-of-range years before any network work.
3. Pre-filters years against the existing manifest (years with
   `n_granules > 0` are skipped on re-runs; zero-granule years are
   retried in case CMR coverage filled in).
4. For each pending year, searches CMR for granules overlapping the
   buffered fabric bbox and downloads them into
   `<datastore>/margulis_wus_sr/raw/<year>/`. Zero-byte downloads are
   dropped before the per-year tally is recorded.
5. The manifest update is flock-protected (parallel workers safe).

## Known follow-ups

- **CMR short_name verification.** `WUS_UCLA_SR` is the documented
  short_name on the NSIDC-0719 product page; confirm against an
  `earthaccess.search_data` smoke before the first production run.

## Troubleshooting

- `ValueError: fabric.json has no 'bbox_buffered' key` — the project's
  fabric metadata is stale. Re-run
  `nhf-targets validate --project-dir <project>` to regenerate
  `fabric.json`.

- `RuntimeError: partial download` — a granule failed mid-run. Re-run
  the same command to retry only the missing files.

- Repeated `no granules found for year YYYY` log entries — the
  project's fabric bbox does not overlap the Western US source domain.
  Expected for fabrics wholly outside the WUS; the aggregation stage
  will likewise skip the source (zero spatial overlap, #309).
