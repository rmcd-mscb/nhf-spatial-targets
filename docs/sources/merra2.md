# MERRA-2 Soil Moisture (M2TMNXLND)

NASA's [Modern-Era Retrospective analysis for Research and
Applications, Version 2](https://disc.gsfc.nasa.gov/datasets/M2TMNXLND_5.12.4/summary)
monthly land surface diagnostics (Gelaro et al., 2017;
Reichle et al., 2017). Used as one of four sources for the soil
moisture (`som`) calibration target range. The catalog short_name is
`M2TMNXLND`, version `5.12.4`, served by NASA GES DISC.

This is the **current replacement for MERRA-Land** (which was
discontinued 2016-02-29). All new runs use MERRA-2; the `merra_land`
catalog entry is preserved with `status: superseded` for documentary
provenance only.

The pipeline downloads three dimensionless soil-wetness variables:

| Variable | Long name | Layer depth | Units |
| -------- | --------- | ----------- | ----- |
| `GWETTOP` | surface_soil_wetness | 0.00-0.05 m (surface) | `1` (fraction of saturation, 0-1) |
| `GWETROOT` | root_zone_soil_wetness | 0.00-1.00 m (root zone) | `1` |
| `GWETPROF` | ave_prof_soil_moisture | spatially varying (surface to bedrock) | `1` |

`GWETTOP` is the **preferred variable** (marked `preferred: true` in
the catalog) for comparison with PRMS `soil_rechr`. All three are
already dimensionless (fraction of saturation 0-1) so no unit
conversion is needed before normalization. Note that magnitudes are
not expected to match across sources at different layer depths;
TM 6-B10 normalization makes this irrelevant for calibration. Layer
thickness fields (`dzsf`, `dzrz`, `dzpr`) live in the `M2CONXLND`
collection and are documented in the catalog `layer_depth_notes`
block.

## Access path — NASA earthaccess

Authenticated via `earthaccess.login(strategy="netrc")`. Requires a
NASA Earthdata Login account with **GES DISC linked** (a separate
opt-in from the base EDL account). Materialize credentials with
`nhf-targets materialize-credentials --project-dir <project>`.

Granules are global monthly NetCDF-4 files
(`MERRA2_<stream>.tavgM_2d_lnd_Nx.<YYYYMM>.nc4`). The CMR search uses
the project's `fabric.bbox_buffered`, but **earthaccess returns the
full global granule regardless** — bounding_box is a server-side
*overlap* test, not a subset operation. Spatial subsetting happens at
target-build time, not at fetch time.

## On-disk layout

```
<datastore>/merra2/
  MERRA2_*.tavgM_2d_lnd_Nx.<YYYYMM>.nc4    # one per year-month
  merra2_consolidated.nc                    # combined CF-1.6 NC (consolidate_merra2)
```

## Procedure

```bash
nhf-targets fetch merra2 --project-dir <project> --period 1980/2025
```

Or via the general SLURM fetch array, index 2
([`slurm/shared/fetch_all.slurm`](../../slurm/shared/fetch_all.slurm)).

Fully **incremental**: months already recorded in `manifest.json` are
skipped at search-result filter time, so a re-run after adding new
years downloads only the new months and re-runs `consolidate_merra2`
against the union of files on disk. The manifest is read-merge-write
to preserve sibling-source provenance (issue #97).

## HPC memory/time notes

- 1980-2025 ≈ 540 monthly granules at ~10 MB each (~5-8 GB total raw,
  see [README.md](../../README.md) §Datastore Storage Estimates).
- The general 128 GB / 24 h fetch SLURM allocation is comfortable;
  most runs finish within a few hours, dominated by network throughput
  to GES DISC.

## Aggregator wiring

`aggregate/merra2.py` is a thin `SourceAdapter` (`monthly` cadence,
`stat_method="mean"`). All three variables flow through area-weighted
aggregation onto the HRU fabric and out to
`<project>/data/aggregated/merra2/merra2_<year>_agg.nc`, one per year,
in native units (dimensionless 0-1). The `som` target consumes
`GWETTOP` by default; the other two are kept for diagnostic
cross-checks.

## Troubleshooting

- `ValueError: No granules found for M2TMNXLND ...` — Earthdata
  credentials missing or GES DISC application not linked. Visit
  <https://urs.earthdata.nasa.gov/profile> → Applications → Authorized
  Apps to confirm.
- `Partial download: got N of M granules` warning — earthaccess saw
  fewer files arrive than CMR listed. The consolidate step proceeds
  with the files on disk; re-run to fill the gap.
- Catalog reference: see [`catalog/sources.yml`](../../catalog/sources.yml)
  `merra2:` block for variable definitions, the `layer_depth_notes`
  block on layer thicknesses, and the GES DISC links.
