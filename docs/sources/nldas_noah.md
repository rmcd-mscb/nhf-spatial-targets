# NLDAS-2 NOAH Soil Moisture

NASA's [North American Land Data Assimilation System phase 2](https://disc.gsfc.nasa.gov/datasets/NLDAS_NOAH0125_M_2.0/summary),
**NOAH** land surface model, monthly output (Xia et al., 2012). Used
as one of four sources for the soil moisture (`som`) calibration target
range. Distributed by NASA GES DISC as `NLDAS_NOAH0125_M` v2.0 on the
same 0.125° CONUS grid as the MOSAIC variant.

The pipeline downloads four soil moisture layers:

| Variable | Long name | Layer depth |
| -------- | --------- | ----------- |
| `SoilM_0_10cm` | soil moisture 0-10 cm | 0.00-0.10 m |
| `SoilM_10_40cm` | soil moisture 10-40 cm | 0.10-0.40 m |
| `SoilM_40_100cm` | soil moisture 40-100 cm | 0.40-1.00 m |
| `SoilM_100_200cm` | soil moisture 100-200 cm | 1.00-2.00 m |

All in native `kg m-2`. NOAH subdivides 40-200 cm into two layers
(40-100 and 100-200) while MOSAIC keeps it as a single 40-200 layer —
choose deliberately when constructing a `som` target stack and do not
sum NOAH's bottom two layers and compare to MOSAIC's bottom layer
without verifying the integration is meaningful.

## Access path — NASA earthaccess

Authenticated via `earthaccess.login(strategy="netrc")`. Requires a
NASA Earthdata Login account with **GES DISC linked**. Materialize
credentials with
`nhf-targets materialize-credentials --project-dir <project>`. CMR
search is bbox-filtered via `fabric.bbox_buffered`; granules are
already CONUS-clipped so no further spatial subsetting is needed.

Granule filenames embed `AYYYYMM` (e.g.
`NLDAS_NOAH0125_M.A200101.020.grb.SUB.nc4`). The shared NLDAS fetch
module ([`fetch/nldas.py`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/fetch/nldas.py))
handles both NOAH and MOSAIC variants behind the per-source key.

## On-disk layout

```
<datastore>/nldas_noah/
  NLDAS_NOAH0125_M.A<YYYYMM>.020.grb.SUB.nc4   # per-month granules
  nldas_noah_consolidated.nc                     # combined CF-1.6 NC
```

## Procedure

```bash
nhf-targets fetch nldas-noah --project-dir <project> --period 1979/2025
```

Or via the general SLURM fetch array, index 4
([`slurm/shared/fetch_all.slurm`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/slurm/shared/fetch_all.slurm)).

**Incremental**: months already recorded in `manifest.json` are
skipped at search-result filter time. Manifest writes are
read-merge-write.

## HPC memory/time notes

- 1979-2025 ≈ 564 monthly CONUS granules, ~2-4 GB raw total (see
  [README.md](../getting-started.md) §Datastore Storage Estimates).
- The general 128 GB / 24 h fetch SLURM allocation is far more than
  needed; full-period runs finish in 1-2 hours.

## Aggregator wiring

`aggregate/nldas_noah.py` is a thin `SourceAdapter` (monthly cadence,
`stat_method="mean"`). All four layers are aggregated; the `som`
target builder selects layers per the project's target config.
Output at
`<project>/data/aggregated/nldas_noah/nldas_noah_<year>_agg.nc`, one
per year, in native `kg m-2`.

## Troubleshooting

- `ValueError: No granules found for NLDAS_NOAH0125_M ...` — Earthdata
  credentials missing or GES DISC application not linked at
  <https://urs.earthdata.nasa.gov/profile> → Applications.
- For MOSAIC vs NOAH ambiguity: the `som` target builder reads the
  catalog `variables:` block by source key, so each
  `aggregate/nldas_*.py` adapter writes a separate aggregated NC and
  the multi-source min/max combination in `targets/som.py` keeps them
  distinct.
- Catalog reference: see [`catalog/sources.yml`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/catalog/sources.yml)
  `nldas_noah:` block for variable definitions and GES DISC links.
