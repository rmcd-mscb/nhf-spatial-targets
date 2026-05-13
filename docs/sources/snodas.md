# SNODAS (NSIDC G02158)

The NOAA NOHRSC Snow Data Assimilation System (SNODAS) daily 1 km
CONUS snow products are distributed by NSIDC as collection
[G02158](https://nsidc.org/data/g02158) (doi:10.7265/N5TB14TC). Daily
`.tar` bundles contain flat int16 binary fields plus ENVI-style `.Hdr`
headers; SWE is the variable of interest for the snow water equivalent
calibration target.

## Access path — direct HTTPS, not CMR

> **Heads-up:** despite the existence of a CMR collection record
> (`short_name=G02158, version=1, concept-id=C1386246263-NSIDCV0`),
> CMR holds **zero granule-level metadata** for this product. Verified
> in issue #107:
>
> ```python
> earthaccess.search_data(short_name='G02158')             # hits=0
> earthaccess.search_data(concept_id='C1386246263-NSIDCV0')  # hits=0
> ```
>
> The CMR record is a metadata-only stub pointing at NSIDC's external
> archive. NOAA NOHRSC products at NSIDC (SNODAS, NIC, etc.) are
> routinely distributed this way.

The real archive lives at:

```
https://noaadata.apps.nsidc.org/NOAA/G02158/masked/YYYY/MM_Mon/SNODAS_YYYYMMDD.tar
```

- Authenticated via Earthdata Login (`.netrc`-based; same credentials
  as the rest of the pipeline).
- `masked/` subtree (CONUS-trimmed): **2003 onward**, daily bundles.
- `unmasked/` subtree (~60 N to 24.95 N including parts of Canada/
  Mexico): 2009 onward. NOT used by this pipeline.

The fetch module constructs daily URLs from each date and streams the
`.tar` via the earthaccess HTTPS auth session
(`earthaccess.login().get_session()`), one file per calendar day.

## Prerequisites

- NASA Earthdata account with credentials materialized into `~/.netrc`
  (`nhf-targets materialize-credentials --project-dir <project>`).
- A few hundred GB of free space in `<datastore>/snodas/raw/` for the
  full 2003-present period (~8k .tars × 10-30 MB each).

## Procedure

```bash
nhf-targets fetch snodas \
    --project-dir <project> \
    --period 2003/2024 \
    [--worker-index 0 --n-workers 1]
```

Or via the SLURM array job:

```bash
sbatch fetch_snodas.slurm   # 4-worker array by default
```

The command:

1. Logs in to NASA EDL via `earthaccess.login(strategy='netrc')` and
   obtains an auth'd `requests.Session` via `auth.get_session()`.
2. Reads the publisher window from `sources.yml[snodas].period` and
   rejects out-of-window years before any HTTP work.
3. Pre-filters years against the existing manifest — years recorded
   with `n_granules > 0` are skipped on re-runs.
4. Round-robin assigns the remaining years to `--worker-index` of
   `--n-workers` so parallel SLURM array jobs cover the period
   without overlap or gaps.
5. For each assigned year, iterates over every calendar day, builds
   the URL `<archive_url>/<year>/MM_Mon/SNODAS_YYYYMMDD.tar`, and
   streams the bundle into
   `<datastore>/snodas/raw/<year>/SNODAS_YYYYMMDD.tar`.

Per-day status accounting (recorded in the manifest year entry):

| Status            | Meaning                                                    |
| ----------------- | ---------------------------------------------------------- |
| `downloaded`      | Streamed successfully this call.                            |
| `already_present` | File already on disk with non-zero size; skipped.           |
| `missing_404`     | Server returned 404; normal at year boundaries and gaps.   |
| `error`           | 5xx, connection reset, partial write, etc.; logged.         |

`n_granules` in the manifest entry is the cumulative `downloaded +
already_present` for the year. The manifest update is flock-protected
so parallel workers do not lose each other's year records.

## Resumability

- Pre-existing non-empty `.tar` files on disk are skipped — `n_already_present`
  in the manifest grows on subsequent runs without re-downloading.
- Years with `n_granules == 0` (e.g. all 404s, or all errors) are
  retried on the next run because NSIDC coverage can fill in
  retroactively.
- The atomic write (`.tar.tmp` → rename) means an interrupted download
  leaves no partial `.tar` for the next run to mistake for "already
  present".

## Known follow-ups

- **Binary decoding to CF NetCDF** is the SNODAS aggregate follow-up
  in umbrella issue #101: characterise the int16+`.Hdr` layout in a
  notebook, then write the decoder, then write the aggregate adapter.

## Known limitations

- **EDL token expiry on multi-hour runs.** The auth'd session is built
  once per worker via `earthaccess.login(strategy='netrc').get_session()`.
  Earthdata bearer tokens typically live 1-2 hours, refreshable up to a
  few days. For single-worker fetches of the full 2003-present archive
  (which can run 24+ hours), the token may expire mid-run. When that
  happens the next `session.get()` returns 401 and the day is recorded
  as `n_errors += 1`. Workarounds: (1) bump `--n-workers` so no worker's
  share exceeds ~1 hour, or (2) re-submit the job — failed days have
  no `out_path` on disk and are picked up cleanly on the next run.

## Troubleshooting

- `RuntimeError: earthaccess login failed for SNODAS` — Earthdata
  credentials are missing or wrong. Re-run
  `nhf-targets materialize-credentials --project-dir <project>` after
  editing `.credentials.yml`.

- All `n_errors` for a year, no `downloaded` or `already_present`, late
  in a long run — most likely an expired EDL token (see above). Re-run
  the job; the worker rebuilds its session and picks up where the
  errors started.

- `short read for <url> — wrote N of M declared bytes` log lines — NSIDC
  closed the connection mid-stream. The `.tar.tmp` is unlinked and the
  day is recorded as `error`. Re-run; transient.

- `ValueError: outside the SNODAS publisher window` — the requested
  `--period` includes years before the catalog's start year. SNODAS
  daily archives begin 2003-09-30; year 2003 will see ~270 404s for
  pre-September dates (recorded as `n_missing_404`, not errors).

- All `n_errors` for a year, no `downloaded`/`already_present` — the
  NSIDC archive is likely down. Confirm with
  `curl -I https://noaadata.apps.nsidc.org/NOAA/G02158/masked/` then
  retry (the manifest fast-path will replay only the failed year).
