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
