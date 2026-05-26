# GLDAS-2.1 NOAH Monthly Runoff

NASA's Global Land Data Assimilation System version 2.1, NOAH land
surface model, monthly product
([GLDAS_NOAH025_M](https://disc.gsfc.nasa.gov/datasets/GLDAS_NOAH025_M_2.1/summary);
Rodell et al., 2004). Used as the second source for the runoff
calibration target (alongside ERA5-Land). The catalog key is
`gldas_noah_v21_monthly` — `GLDAS` alone is ambiguous between NOAH /
CLSM / VIC and between 2.0 / 2.1 / monthly / 3-hourly variants.

The pipeline downloads two variables and writes a derived third at
consolidation:

| Variable | Long name | Native units | cell_methods |
| -------- | --------- | ------------ | ------------ |
| `Qs_acc` | storm surface runoff | kg m-2 | `time: sum` |
| `Qsb_acc` | baseflow-groundwater runoff | kg m-2 | `time: sum` |
| `runoff_total` (derived) | `Qs_acc + Qsb_acc` | kg m-2 | `time: sum` |

The catalog `notes` field captures the GLDAS-2.1 `_acc` convention
gotcha: variables ending in `_acc` are stored as the **mean of
3-hourly accumulations across the month**, not as a monthly sum,
despite the misleading `cell_methods: time: sum`. To recover monthly
mm, multiply by `8 × days_in_month`. The runoff target builder
(`targets/run.py`) applies this conversion in `gldas_to_mm_per_month`;
do not apply it at fetch or aggregate time, where the native units are
preserved (see CLAUDE.md Aggregation Transformation Policy: linear
conversions live in `targets/`).

## Access path — NASA earthaccess

Authenticated via `earthaccess.login(strategy="netrc")` from
[`fetch/_auth.py`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/fetch/_auth.py). Requires
a NASA Earthdata Login account with the GES DISC application linked
(see <https://disc.gsfc.nasa.gov/earthdata-login>). Credentials are
materialized into `~/.netrc` by
`nhf-targets materialize-credentials --project-dir <project>`.

Granules are **global** monthly NetCDF-4 files (~few MB each). The
download happens full-globe; spatial clipping to the
CONUS+contributing-watersheds bbox (53°/-125° → 24.7°/-66°) happens at
consolidate time via `clip_to_bbox`, which handles both ascending and
descending latitude ordering and converts 0-360 longitude to -180-180
if needed.

## On-disk layout

```
<datastore>/gldas_noah_v21_monthly/
  raw/                                    # per-month granules from earthaccess
    GLDAS_NOAH025_M.A<YYYYMM>.021.nc4
    ...
  gldas_noah_v21_monthly.nc               # consolidated, clipped, CF-1.6
```

The consolidator (in [`fetch/gldas.py`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/fetch/gldas.py))
runs `derive_runoff_total → clip_to_bbox → apply_cf_metadata` (the
shared CF helper from `fetch/consolidate.py`) and writes atomically
via a `.nc.tmp` rename.

## Procedure

```bash
nhf-targets fetch gldas --project-dir <project> --period 2000/2025
```

Or via the general SLURM fetch array, index 1
([`slurm/shared/fetch_all.slurm`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/slurm/shared/fetch_all.slurm)).

The function fails fast (raises `ValueError`) if `earthaccess.search_data`
returns no granules, and raises `RuntimeError` on partial downloads —
the consolidated NC is regenerated from scratch on every call rather
than re-merging.

## HPC memory/time notes

- The general 128 GB / 24 h fetch SLURM allocation is overkill for
  GLDAS; runs typically finish in under an hour with peak RSS under a
  few GB. No source-specific tuning needed.

## Aggregator wiring

`aggregate/gldas.py` is a thin `SourceAdapter` (`monthly` cadence,
`stat_method="mean"` default). The pre-aggregate hook redundantly
re-derives `runoff_total` to defend against datastores that pre-date
the fetch-time derivation, but the consolidated NC already carries it.
Output at `<project>/data/aggregated/gldas_noah_v21_monthly/gldas_<year>_agg.nc`,
one per year.

## Troubleshooting

- `ValueError: No GLDAS granules found ...` — check Earthdata
  credentials and that GES DISC is linked to your EDL account.
- `ValueError: Clipping produced an empty dataset` — the bbox
  convention (lon/lat) and dataset longitude convention (-180/180 vs
  0/360) mismatch. The clipper converts 0/360→-180/180 automatically;
  if this raises anyway, file an issue with the offending file's
  lon/lat range from the error message.
- Catalog reference: see [`catalog/sources.yml`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/catalog/sources.yml)
  `gldas_noah_v21_monthly:` block for the `_acc` convention note and
  the runoff target's mm-conversion factor.
