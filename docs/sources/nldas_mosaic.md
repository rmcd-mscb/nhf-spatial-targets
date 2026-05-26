# NLDAS-2 MOSAIC Soil Moisture

NASA's [North American Land Data Assimilation System phase 2](https://disc.gsfc.nasa.gov/datasets/NLDAS_MOS0125_M_2.0/summary),
**MOSAIC** land surface model, monthly output (Xia et al., 2012). Used
as one of four sources for the soil moisture (`som`) calibration target
range. Distributed by NASA GES DISC as `NLDAS_MOS0125_M` v2.0 on a
0.125° CONUS grid.

The pipeline downloads three soil moisture layers:

| Variable | Long name | Layer depth |
| -------- | --------- | ----------- |
| `SoilM_0_10cm` | soil moisture 0-10 cm | 0.00-0.10 m |
| `SoilM_10_40cm` | soil moisture 10-40 cm | 0.10-0.40 m |
| `SoilM_40_200cm` | soil moisture 40-200 cm | 0.40-2.00 m |

All in native `kg m-2`; the `som` target normalizes to a dimensionless
[0, 1] range, so units differences across the four `som` sources
(MERRA-2 / NCEP-NCAR / NLDAS-MOSAIC / NLDAS-NOAH) wash out by design
(TM 6-B10). Note MOSAIC's third layer is **40-200 cm** — different
from NLDAS-2 NOAH, which subdivides the same range into 40-100 and
100-200 cm; see [`nldas_noah.md`](nldas_noah.md).

## Access path — NASA earthaccess

Authenticated via `earthaccess.login(strategy="netrc")`. Requires a
NASA Earthdata Login account with **GES DISC linked**. Materialize
credentials with
`nhf-targets materialize-credentials --project-dir <project>`. CMR
search is bbox-filtered via `fabric.bbox_buffered`; granules are
already CONUS-clipped at the source, so unlike GLDAS/MERRA-2 the
downloaded files do not require post-hoc spatial subsetting.

Granule filenames embed `AYYYYMM` (e.g.
`NLDAS_MOS0125_M.A200101.020.grb.SUB.nc4`). The shared NLDAS fetch
module ([`fetch/nldas.py`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/fetch/nldas.py))
handles both MOSAIC and NOAH variants behind the per-source key.

## On-disk layout

```
<datastore>/nldas_mosaic/
  NLDAS_MOS0125_M.A<YYYYMM>.020.grb.SUB.nc4   # per-month granules
  nldas_mosaic_consolidated.nc                  # combined CF-1.6 NC
```

## Procedure

```bash
nhf-targets fetch nldas-mosaic --project-dir <project> --period 1979/2025
```

Or via the general SLURM fetch array, index 3
([`slurm/shared/fetch_all.slurm`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/slurm/shared/fetch_all.slurm)).

**Incremental**: months already recorded in `manifest.json` are skipped
at search-result filter time, so re-running after adding a new period
fetches only the new months. Manifest writes are read-merge-write.

## HPC memory/time notes

- 1979-2025 ≈ 564 monthly CONUS granules, ~2-4 GB raw total (see
  [README.md](../getting-started.md) §Datastore Storage Estimates).
- The general 128 GB / 24 h fetch SLURM allocation is far more than
  needed; full-period runs finish in 1-2 hours on a warm network.

## Aggregator wiring

`aggregate/nldas_mosaic.py` is a thin `SourceAdapter` (monthly cadence,
`stat_method="mean"`, default `mean` policy because no per-pixel
masking happens upstream). Output at
`<project>/data/aggregated/nldas_mosaic/nldas_mosaic_<year>_agg.nc`,
one per year, in native `kg m-2`.

## Troubleshooting

- `ValueError: No granules found for NLDAS_MOS0125_M ...` — Earthdata
  credentials missing or GES DISC application not linked at
  <https://urs.earthdata.nasa.gov/profile> → Applications.
- Same `Partial download` warning as MERRA-2; re-run picks up missing
  months by filename.
- Catalog reference: see [`catalog/sources.yml`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/catalog/sources.yml)
  `nldas_mosaic:` block for variable definitions and GES DISC links.
