# SNODAS (NSIDC G02158)

The NOAA NOHRSC Snow Data Assimilation System (SNODAS) daily 1 km
CONUS snow products are distributed by NSIDC as collection
[G02158](https://nsidc.org/data/g02158)
(doi:10.7265/N5TB14TC). Daily tar/gz bundles contain flat int16
binary fields plus ENVI-style `.Hdr` headers. SWE is the variable of
interest for the snow water equivalent calibration target.

This PR ships the **fetch-only** path: download raw bundles via
`earthaccess` (NASA Earthdata login) into
`<datastore>/snodas/raw/<year>/` and record a per-year manifest
entry. Decoding the int16 binary into a CF NetCDF, and clipping or
re-projecting it for HRU aggregation, is **deferred** to the SNODAS
aggregate follow-up issue — the native format needs operator
characterisation in a notebook (per CLAUDE.md "characterise the data
first") before a robust parser is committed.

## Prerequisites

- NASA Earthdata account with credentials materialized into `~/.netrc`
  (run `nhf-targets materialize-credentials --project-dir <project>`
  after editing `.credentials.yml`).
- Project's `fabric.json` present with a `bbox_buffered` covering the
  CONUS footprint of interest. (The catalog also carries a CONUS-wide
  fallback `bbox_nwse` at `sources.yml[snodas].access.bbox_nwse`.)

## Procedure

```bash
nhf-targets fetch snodas \
    --project-dir <project> \
    --period 2003/2024 \
    [--worker-index 0 --n-workers 1]
```

The command:

1. Logs in to NASA EDL via `earthaccess`.
2. Reads `sources.yml[snodas].period` for the publisher window and
   rejects out-of-window years before any network work.
3. Pre-filters years against the existing manifest — years recorded
   with `n_granules > 0` are skipped on re-runs (years with
   `n_granules: 0` are retried because CMR coverage can fill in
   retroactively).
4. Round-robin assigns the remaining years to `--worker-index` of
   `--n-workers` so parallel SLURM array jobs cover the period
   without overlap or gaps.
5. For each assigned year, searches CMR (`short_name: G02158`,
   `version: "1"`) inside the catalog bbox, downloads matching
   granules into `<datastore>/snodas/raw/<year>/`, and records the
   per-year tally in `manifest.json` → `sources.snodas.years`.
6. The manifest update is flock-protected so parallel workers do not
   lose each other's year records.

## Known follow-ups

- **CMR short_name/version verification.** `G02158` / `"1"` is the
  best-guess pair; confirm against an `earthaccess.search_data` smoke
  test before the first production run.
- **Binary decoding.** Filed as the SNODAS aggregate follow-up issue:
  characterise the flat int16 + `.Hdr` layout in a notebook, then
  write the decoder, then write the aggregate adapter.

## Troubleshooting

- `RuntimeError: snodas: earthaccess.download returned no files` —
  Earthdata credentials are missing or wrong, or the CMR endpoint is
  unreachable. Re-run `nhf-targets materialize-credentials`.

- `RuntimeError: snodas: partial download` — a granule failed mid-run.
  Re-run the same command to retry only the missing files.

- `ValueError: outside the SNODAS publisher window` — the requested
  `--period` includes years before the catalog's start year. SNODAS
  daily archives begin 2003-09-30.
