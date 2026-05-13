# SNODAS (NSIDC G02158)

The NOAA NOHRSC Snow Data Assimilation System (SNODAS) daily 1 km
CONUS snow products are distributed by NSIDC as collection
[G02158](https://nsidc.org/data/g02158) (doi:10.7265/N5TB14TC). Daily
`.tar` bundles contain flat int16 binary fields plus ENVI-style `.Hdr`
headers; SWE is the variable of interest for the snow water equivalent
calibration target.

The fetch step runs in two phases per year: download raw `.tars` from
NSIDC's HTTPS archive, then decode product 1034 (SWE) from each day
into a single per-year CF NetCDF at
`<datastore>/snodas/daily/snodas_daily_<year>.nc`. Raw `.tars` are
preserved on disk after consolidation.

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
- **~24 GB RAM per worker** for the consolidation step. A full year of
  CONUS SNODAS is `(365, 3351, 6935)` int16 = ~17 GB held as a single
  numpy array while the year NC is being assembled; peak RSS during
  `to_netcdf` runs higher. The default `fetch_snodas.slurm` memory
  grant must accommodate this — bump `--mem` per worker if a SLURM
  array job OOM-kills during consolidation.

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
   with `n_granules > 0` **and** an existing `daily_path` are skipped
   on re-runs. Years that have been downloaded but not consolidated
   (manifest carries `n_granules > 0` but no `daily_path`) are
   re-entered to run the consolidation step only; see "Backfill"
   below.
4. Round-robin assigns the remaining years to `--worker-index` of
   `--n-workers` so parallel SLURM array jobs cover the period
   without overlap or gaps.
5. For each assigned year, iterates over every calendar day, builds
   the URL `<archive_url>/<year>/MM_Mon/SNODAS_YYYYMMDD.tar`, and
   streams the bundle into
   `<datastore>/snodas/raw/<year>/SNODAS_YYYYMMDD.tar`.
6. After the year's download loop, decodes the SWE product (NSIDC
   code 1034) out of every `.tar` and writes a single per-year CF
   NetCDF at `<datastore>/snodas/daily/snodas_daily_<year>.nc`.
   Consolidation failures are recorded on the per-year manifest
   record as `consolidate_error` rather than aborting the run — raw
   `.tars` are preserved and the consolidation can be retried.

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
- The daily NC write is also atomic (`.nc.tmp` → rename) and idempotent
  on mtime: if the daily NC exists and is newer than every input `.tar`,
  consolidation is skipped.

## Backfill (consolidation rerun without re-download)

Operator scenario after this PR merges: the 2003–2024 raw corpus is
already on disk from PRs #100 / #103 / #106 / #108 but no per-year
daily NCs exist. The next run of `fetch_snodas` (or
`sbatch fetch_snodas.slurm`) consolidates every year-already-on-disk
**without re-downloading**:

1. The completion check now requires both `n_granules > 0` AND a
   `daily_path` that exists on disk. Existing manifest entries have
   the former but not the latter, so every year is re-entered.
2. Inside `_fetch_year`, every per-day URL hits the `already_present`
   shortcut in `_download_tar` (the on-disk `.tar` is ≥ 1 MiB), so no
   bytes are re-downloaded.
3. `consolidate_year_snodas` runs against the raw dir and writes
   `<datastore>/snodas/daily/snodas_daily_<year>.nc`. Roughly one CPU
   per year file (~10 minutes per year on hovenweep), I/O bound on
   the raw read.
4. The manifest gets `daily_path` and `consolidated_utc` per year;
   subsequent runs are fast no-ops.

If a year's consolidation fails (e.g. a corrupt `.tar` from a partial
download), the manifest carries `consolidate_error` for that year and
the run continues. Operators triage by removing the offending `.tar`
from `<datastore>/snodas/raw/<year>/` and re-running.

## Daily NC layout

Each year file at `<datastore>/snodas/daily/snodas_daily_<year>.nc`
carries:

- `swe(time, lat, lon)` — native `int16` with `_FillValue=-9999`,
  zlib-compressed (level 4), chunked one day per chunk. xarray's
  default `mask_and_scale=True` converts the fill to NaN on read.
  Units `kg m-2` (≡ mm of water equivalent), `cell_methods: "time: point"`.
- `crs` ancillary variable — WGS84 geographic, set by
  `apply_cf_metadata`.
- `lat` descending (top-to-bottom rows), `lon` ascending. Cell-center
  coords derived from the first day's header.
- Global attributes: `Conventions=CF-1.6`, `title`, `institution`,
  `source`, `references`, `frequency=day`, `history`, and a
  `snodas_first_day_header` JSON string capturing the original ENVI
  grid metadata for cross-year drift forensics.

## For the SNODAS aggregator (sub-task 2b of #101)

When wiring the aggregate adapter, use `stat_method="masked_mean"` on
the `SourceAdapter` so the gdptools area-weighted mean **skips** the
NaN pixels at the CONUS analysis mask edge rather than letting them
poison every HRU that touches the mask. SNODAS deliberately ships
`-9999` for off-CONUS pixels (oceans, the Mexico/Canada portions of
the bbox, the Great Lakes); after consolidation those arrive at the
aggregator as NaN. Without `masked_mean` an HRU whose footprint
intersects even one masked pixel collapses to NaN. Same pattern as
[`aggregate/mod10c1.py`](../../src/nhf_spatial_targets/aggregate/mod10c1.py)
and [`aggregate/mod16a2.py`](../../src/nhf_spatial_targets/aggregate/mod16a2.py).

## Archive gaps — missing days inside otherwise-complete years

SNODAS's daily archive has **real day-gaps** even after the late-2003
launch. The first SLURM consolidation run on hovenweep (job 17553331,
2026-05-13) showed:

| Year | On disk | Calendar days | Missing | Cause |
| ---- | ------- | ------------- | ------- | ----- |
| 2003 | 93      | 92 (from Sep 30) | 0    | partial start year |
| 2004 | 363     | 366           | 3       | 02-25, 08-31, 09-27 |
| 2005 | 362     | 365           | 3       | (NOHRSC publishing gaps) |
| 2006 | 360     | 365           | 5       | (NOHRSC publishing gaps) |
| 2007 | 363     | 365           | 2       | (NOHRSC publishing gaps) |

These gaps trace to NSIDC HTTP 404s during the original fetch (PRs
#100 / #103 / #106 / #108) — NOHRSC's model didn't publish those
specific days, usually because of a processing outage, server
downtime, or convergence failure. The fetch layer records them as
`n_missing_404` in the manifest and the consolidator silently ingests
only the `.tars` that exist, so each year NC's `time` axis is
non-contiguous (e.g. 2004 has 363 daily timestamps, jumping over the
three missing dates rather than NaN-filling them).

**Implication for downstream consumers (aggregator and target):**

- The SNODAS aggregator (sub-task 2b of #101) must enumerate its time
  axis from the source NC's own `time` coord, not from a synthetic
  daily range — most gdptools-based aggregators do this already.
- Calibration targets that need a regular daily series must decide
  per-target whether to (a) drop the missing days from the time
  index, (b) NaN-fill them, or (c) interpolate. **TBD pending
  colleague consultation** (2026-05-13): the missing-day count is
  small (5 in the worst year of 2003–2007) and may be eligible for a
  cheap linear-in-time interpolation post-processor at the target
  stage. No decision is encoded in the consolidator itself — the year
  NCs are deliberately "honest about gaps" so any downstream policy
  is reversible.

## Partial-bundle days — 2003-10-30 case

Distinct from the gap problem: some early-SNODAS daily `.tars` are
present on the archive but ship **only a subset of the 8 product
pairs**. The first known case is `SNODAS_20031030.tar`, whose bundle
carries only product 1025 (snowpack layer temperature), missing
product 1034 (SWE).

Current behavior: `consolidate_year_snodas` raises a `ValueError`
("no SWE product (code 1034) member") the moment it hits such a day,
which the fetch layer catches and records as `consolidate_error` for
the year. **The whole year's daily NC is therefore not written** even
though 91 of the 92 valid 2003 days are decodable. This is the
correct fail-loud default for an unfamiliar data shape, but it is
also brittle in production — a single bad day in a 366-day year
should not block the other 365.

A follow-up consolidator change (likely sub-task 2c on #101) should
log-and-skip days that lack product 1034 the same way the download
layer treats 404s. Decision pending the same colleague consultation
that covers the missing-day fill policy.

## Grid drift across years

NSIDC has re-georeferenced the masked product at least twice
(between 2003 and 2004, and between 2013 and 2014). The shifts are
sub-pixel (a few µdeg) and never cause within-year header changes.
The consolidator's within-year tolerance is 1×10⁻⁶ deg, which
absorbs sub-µdeg text-representation noise while still rejecting a
real mid-year format change. **A future multi-year zarr stack will
need a snap-to-grid step at the year boundary**; per-year NCs are
the canonical "what each year actually shipped" record and do not
attempt cross-year alignment themselves.

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

- `consolidate_error` recorded on a year — one of the `.tar` files
  in that year is corrupt or the .tar's SWE binary disagrees with its
  declared shape. The raw `.tars` are untouched on disk so the
  download record is preserved; remove the offending `.tar` from
  `<datastore>/snodas/raw/<year>/` and re-run to recover.

- `ValueError: within-year grid mismatch` from `consolidate_year_snodas`
  — two `.tar` files for the same year declare different grids. Real
  cross-year shifts are expected, but mid-year changes indicate a
  corrupt or mismatched `.tar`. Compare the two header keys named in
  the error message and remove the day that does not match the rest of
  the year.
