# NCEP/NCAR Reanalysis Soil Moisture

NOAA's [NCEP/NCAR Reanalysis 1](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html)
soil moisture (Kalnay et al., 1996). Used as one of four sources for
the soil moisture (`som`) calibration target range. The publisher (NOAA
PSL) distributes per-year daily files; the pipeline resamples to
monthly means on the fly during the fetch.

The pipeline downloads two layers:

| Variable | Long name | Layer depth | Units (effective) |
| -------- | --------- | ----------- | ----------------- |
| `soilw_0_10cm` | volumetric soil moisture 0-10 cm | 0.00-0.10 m | `m3 m-3` (VWC) |
| `soilw_10_200cm` | volumetric soil moisture 10-200 cm | 0.10-2.00 m | `m3 m-3` (VWC) |

## The NCEP R1 `units: kg/m2` mislabel

> The upstream NetCDFs from NOAA PSL carry `units: kg/m2` on the
> `soilw.*.gauss` variables, but the long_name, var_desc, and
> valid_range `[0.0, 1.0]` (with observed actual_range ≈ `[0.10, 0.43]`)
> all confirm the values are **volumetric water content (m³/m³)**, not
> mass per area. This is a known mislabel in the NCEP R1 GRIB
> metadata.
>
> **Do NOT divide by `(layer_depth × ρ_water)` to convert to VWC; the
> data already is VWC.** The catalog `units:` field overrides the
> on-disk attribute, and `apply_cf_metadata` writes the corrected
> `units` into the consolidated NC.

The catalog `notes` field documents this explicitly. Always read units
from `catalog/sources.yml` not from the on-disk file (also a general
project convention; see CLAUDE.md → "Read source units from the
catalog, not from on-disk NetCDF attrs").

## Access path — direct HTTP (NOAA PSL)

No authentication required. The fetch module
[`fetch/ncep_ncar.py`](../../src/nhf_spatial_targets/fetch/ncep_ncar.py)
constructs annual URLs from the catalog `file_pattern`:

```
https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface_gauss/<file_variable>.{year}.nc
```

Each annual daily file is downloaded with `urllib.request.urlretrieve`,
opened with xarray, resampled to monthly means via `.resample(time="1ME").mean()`,
renamed to the catalog variable name (`soilw` → `soilw_0_10cm` or
`soilw_10_200cm`), and written as a monthly file alongside the daily
file. The daily file is then **deleted** — the source archive
preserves it for re-download if needed.

The fetch loop covers `(year, var)` pairs in a single `tqdm` progress
bar so partial-run resumption is per-(year, var), not per-year.

## On-disk layout

```
<datastore>/ncep_ncar/
  soilw.0-10cm.gauss.<year>.monthly.nc       # one per year-variable pair
  soilw.10-200cm.gauss.<year>.monthly.nc
  ncep_ncar_consolidated.nc                    # combined CF-1.6 NC
```

## Procedure

```bash
nhf-targets fetch ncep-ncar --project-dir <project> --period 1979/2025
```

Or via the general SLURM fetch array, index 5
([`slurm/shared/fetch_all.slurm`](../../slurm/shared/fetch_all.slurm)).

**Incremental**: years already recorded in `manifest.json` are skipped
before any HTTP work. Manifest writes are read-merge-write.

## Grid resolution and choropleth quirk

NCEP/NCAR R1 uses a **T62 Gaussian grid** at ~1.875° (~210 km). At
HRU sizes typical of NHM, many HRUs land inside the same source cell,
which produces visibly "blocky" aggregated choropleths — large
contiguous regions of HRUs with the same value because they share an
upstream cell. This is the visual evidence for the gfv2 fabric's
exclusion of `ncep_ncar` from `som` (see
[`docs/references/known-gaps-resolved.md`](../references/known-gaps-resolved.md)
if applicable, or the project memo
`project_coarse_source_blockiness`). It is not a pipeline bug.

## HPC memory/time notes

- 1979-2025 ≈ 47 years × 2 variables = ~94 annual daily downloads,
  each resampled to monthly in-memory. Peak RSS stays well under 8 GB.
- The general 128 GB / 24 h fetch SLURM allocation is overkill;
  full-period runs finish in ~1 hour, bottlenecked by the
  per-(year, var) sequential HTTP downloads.

## Aggregator wiring

`aggregate/ncep_ncar.py` is a thin `SourceAdapter` (monthly cadence,
`stat_method="mean"`). Output at
`<project>/data/aggregated/ncep_ncar/ncep_ncar_<year>_agg.nc`, one
per year, in native VWC `m3 m-3` (with the catalog's
units-correction in the variable attrs).

## Troubleshooting

- `RuntimeError: Failed to download <url>: HTTP 404` — NOAA PSL may
  have moved or renamed the dataset path. Inspect
  <https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html> and
  update `file_pattern` in `catalog/sources.yml` if so.
- The on-disk NetCDF `cf_units` attribute reads as something
  inconsistent with the catalog — trust the catalog
  (`catalog.source("ncep_ncar")["variables"][i]["units"]`); on-disk
  attrs drift after corrections. The aggregated NC is written by
  `apply_cf_metadata` and should agree with the catalog after the
  next aggregator run.
- Catalog reference: see [`catalog/sources.yml`](../../catalog/sources.yml)
  `ncep_ncar:` block for the `kg/m2`-mislabel notes block.
