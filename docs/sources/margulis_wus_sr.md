# Margulis Western US Snow Reanalysis (NSIDC-0719)

The UCLA/Margulis Western US Snow Reanalysis is a daily 90 m posterior
SWE product over the Sierra Nevada, Cascades, Rockies and adjacent
ranges for water years 1985-2021 (Fang, Liu & Margulis, 2022;
doi:10.5067/PP7T2GBI52I2). NSIDC distributes it as collection
NSIDC-0719 (short_name `WUS_UCLA_SR`).

This source is **fabric-scoped to Oregon only** in this pipeline (see
[`catalog/sources.yml`](../../catalog/sources.yml)
`margulis_wus_sr.fabric_scope`). Raw downloads are reusable by any
project pointing at the same datastore, but the SWE target builder
excludes this source for non-Oregon fabrics — a Mississippi-fabric
calibration run will not consume Margulis WUS-SR data even if it
exists in the shared datastore. The fetch step warns when invoked
against a non-Oregon fabric so operators don't mistake the empty
search results for an upstream outage.

The fetch step ships the **search-and-download** path: granules land
in `<datastore>/margulis_wus_sr/raw/<year>/`. Concatenating the
per-water-year per-tile NetCDFs into a per-year CF NetCDF (and the
fabric_scope enforcement at target-build time) is **deferred** to the
Margulis aggregate follow-up issue.

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
3. If the project's fabric is not in
   `sources.yml[margulis_wus_sr].fabric_scope.fabrics` (currently
   `["or"]`), emits a one-line warning. Fetch proceeds anyway — raw
   data is datastore-shared and a future Oregon-fabric run can reuse
   it — but expect zero granules per year.
4. Pre-filters years against the existing manifest (years with
   `n_granules > 0` are skipped on re-runs; zero-granule years are
   retried in case CMR coverage filled in).
5. For each pending year, searches CMR for granules overlapping the
   buffered fabric bbox and downloads them into
   `<datastore>/margulis_wus_sr/raw/<year>/`. Zero-byte downloads are
   dropped before the per-year tally is recorded.
6. The manifest update is flock-protected (parallel workers safe).

## Known follow-ups

- **CMR short_name verification.** `WUS_UCLA_SR` is the documented
  short_name on the NSIDC-0719 product page; confirm against an
  `earthaccess.search_data` smoke before the first production run.
- **Aggregate adapter + fabric_scope enforcement** are filed as the
  Margulis aggregate follow-up issue.

## Troubleshooting

- `ValueError: fabric.json has no 'bbox_buffered' key` — the project's
  fabric metadata is stale. Re-run
  `nhf-targets validate --project-dir <project>` to regenerate
  `fabric.json`.

- `RuntimeError: partial download` — a granule failed mid-run. Re-run
  the same command to retry only the missing files.

- Repeated `no granules found for year YYYY` log entries with an
  accompanying scope warning — the project's fabric is not in
  `fabric_scope.fabrics`. This is expected for non-Oregon fabrics;
  the SWE target builder will exclude this source automatically.
