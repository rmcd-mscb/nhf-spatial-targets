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
  and `perennial_snow_mask_fingerprint`.
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

- The fingerprint covers **the rule plus the set of input years only, not file
  contents.** The mask is rebuilt automatically when the catalog rule changes
  or a Margulis year is added or removed.
- **Cost of a year-set change.** Adding or removing any Margulis year changes
  the fingerprint. Fetching WY2021, or moving a corrupt year aside, therefore
  re-aggregates **every** Margulis year in **every** project on the shared
  datastore and rebuilds the full SWE intermediates, even if no mask cell
  changes.
- **Manual recovery after re-consolidating a year with corrected data.**
  Because the fingerprint cannot see file contents, a rebuilt mask carries the
  *same* fingerprint. Per-year aggregates still match their stamp and are
  skipped, the re-consolidated year is skipped too (the aggregate skip is by
  file existence), and the SWE cache key is unchanged. Nothing re-runs
  automatically. Delete, then re-run:
  1. `<datastore>/margulis_wus_sr/derived/perennial_snow_mask.nc`
  2. `<project>/data/aggregated/margulis_wus_sr/` in **every** project that
     uses this datastore (the rebuilt mask may differ).
  3. `<project>/targets/.swe_intermediates/` in each such project.

  Then, per project:

  ```bash
  nhf-targets agg margulis-wus-sr --project-dir <project>
  nhf-targets run --target snow_water_equivalent --project-dir <project>   # pixi run run-swe
  nhf-targets maintenance rebuild-manifest --project-dir <project>
  nhf-targets validate --project-dir <project>
  ```
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
