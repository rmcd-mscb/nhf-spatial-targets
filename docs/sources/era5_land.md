# ERA5-Land (Copernicus CDS)

ECMWF [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land)
hourly reanalysis (Muñoz-Sabater et al., 2021;
doi:10.5194/essd-13-4349-2021). The pipeline downloads four variables
across CONUS + contributing watersheds (Canada/Mexico) at 0.1° native
resolution:

| Variable | Long name | Native units | cell_methods | Target |
| -------- | --------- | ------------ | ------------ | ------ |
| `ro` | total runoff | m | `time: sum` (accumulated) | runoff |
| `sro` | surface runoff | m | `time: sum` (accumulated) | reserved (future) |
| `ssro` | sub-surface runoff | m | `time: sum` (accumulated) | recharge (proxy) |
| `sd` | snow depth water equivalent | m | `time: point` (instantaneous) | SWE |

ERA5-Land is the largest fetch by far in the catalog: hourly fields
across ~12 GB/variable-year (CONUS+ bbox) and a 1979/present publisher
window. The pipeline writes both a daily and a monthly consolidated
NetCDF per year to the datastore (`<datastore>/era5_land/{daily,monthly}/era5_land_{daily,monthly}_<year>.nc`).

## Access path — Copernicus CDS API

Authenticated via `cdsapi.Client()`. Operators must:

1. Create a free Copernicus CDS account at
   <https://cds.climate.copernicus.eu/>.
2. Accept the **ERA5-Land licence** at
   <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences>
   — the API will fail with a licence-not-accepted error otherwise, and
   CDS does not surface this clearly in the failure message.
3. Place credentials (`url`, `key`) in `.credentials.yml` under the
   `cdsapi:` key, then run
   `nhf-targets materialize-credentials --project-dir <project>` to
   render `~/.cdsapirc`.

## Monthly-chunked CDS requests

Each year of each variable is split into **12 monthly CDS requests**
because a single annual all-hours-all-days request exceeds the CDS
per-request cost limit. The per-month chunks land alongside the year
file as
`<datastore>/era5_land/hourly/era5_land_<var>_<year>_<MM>.nc` and are
**preserved on disk** so re-runs are idempotent at the chunk level.

The monthly chunk validator (`_validate_chunk_time_coord` in
[`fetch/era5_land.py`](../../src/nhf_spatial_targets/fetch/era5_land.py))
guards against an observed-in-production rare CDS quirk: a chunk file
on disk whose `time` coord covers a different month than the chunk
filename implies. When detected, the bad chunk is deleted and
re-fetched. Without this guard, `open_mfdataset(combine="by_coords")`
silently inflates the year's time range and corrupts every downstream
daily/monthly NC.

## Hourly → daily → monthly aggregation

Two distinct reducers are dispatched by `_VARIABLE_KIND`, which reads
each variable's `cell_methods` from `catalog/sources.yml`:

- **Accumulated** (`ro`, `sro`, `ssro`): hourly values are
  midnight-resetting accumulations since 00 UTC. The hourly→daily step
  in `hourly_to_daily` (a) diffs the accumulation, (b) substitutes the
  raw accumulated value at the midnight reset where the diff is
  negative, (c) shifts timestamps back 1 hour so the 23→00 increment
  is credited to the prior day, then (d) sums hourly increments per
  calendar day. The daily→monthly step is a `.sum()`.
- **Instantaneous** (`sd`): values are point-in-time snowpack water
  equivalent. Hourly→daily is a `.mean()`; daily→monthly is also a
  `.mean()`. Applying the accumulated reducer here would yield
  physically meaningless values.

If you add a new ERA5-Land variable, set its catalog `cell_methods` to
either `time: sum` (accumulated) or `time: point` (instantaneous);
anything else makes `_derive_variable_kind` raise so the dispatch
table doesn't silently misclassify.

## Procedure

```bash
nhf-targets fetch era5-land \
    --project-dir <project> \
    --period 1979/2025 \
    [--worker-index 0 --n-workers 4]
```

Or via the sharded SLURM array (default 4 workers, respects CDS
per-user throttling):

```bash
sbatch slurm/shared/fetch_era5_land.slurm
```

Each worker takes a round-robin slice of `all_years`; the slice is
computed independently of sibling progress (the manifest fast-path that
caused gap-bugs in earlier versions was rewritten in #126).

## Manifest fast-path

A year is considered "complete" only when its manifest entry carries
both `daily_path` and `monthly_path` *and* both files exist on disk.
This means rerunning `fetch era5-land` after a hand-deletion of one of
the consolidated NCs picks the year back up, and rerunning after a new
worker pool size change picks up any years no previous slice covered.

## HPC memory/time notes

- CDS throttles each user to a few concurrent requests; the
  `slurm/shared/fetch_era5_land.slurm` script uses 4 parallel workers
  by default, which has converged on Hovenweep without
  rate-limit retries.
- A full 1979/2025 pipeline run takes many hours, sometimes days, with
  the CDS queue being the dominant cost.
- Memory is modest (~8-16 GB); the per-month chunks bound peak RSS even
  when concatenating into a year file.

## Aggregator wiring

`aggregate/era5_land.py` exposes **two adapters** for the one source:

- `ADAPTER` reads `<datastore>/era5_land/monthly/era5_land_monthly_*.nc`
  and emits the runoff variables (`ro`, `sro`, `ssro`). Output at
  `<project>/data/aggregated/era5_land/era5_land_<year>_agg.nc`.
- `ADAPTER_SD` reads `<datastore>/era5_land/daily/era5_land_daily_*.nc`
  and emits the snow depth water equivalent (`sd`). Output at
  `<project>/data/aggregated/era5_land_sd/era5_land_sd_<year>_agg.nc`.
  The synthetic source key `era5_land_sd` keeps the SWE adapter's
  weight cache, manifest entry, and aggregated subdir distinct from
  the monthly runoff adapter without duplicating catalog metadata.

Both adapters use `stat_method="mean"` (no per-pixel masking is done
upstream; CDS-delivered NaN pixels propagate honestly to NaN HRUs).

## Troubleshooting

- `Required licence not yet accepted` from `cdsapi.Client.retrieve` —
  see step 2 above; accept the ERA5-Land licence in the CDS web UI.
- A year repeatedly fails consolidation while individual monthly
  chunks are present — almost always a hand-corrupted chunk. Inspect
  `era5_land_<var>_<year>_<MM>.nc` files for non-NetCDF magic bytes,
  delete the offender, and re-run.
- Catalog reference: see [`catalog/sources.yml`](../../catalog/sources.yml)
  `era5_land:` block for variable units, the bbox (53°/-125° → 24.7°/-66°),
  and CF metadata that flows through `apply_cf_metadata`.
