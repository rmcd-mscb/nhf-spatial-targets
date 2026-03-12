# nhf-spatial-targets

Curated calibration target datasets for the [National Hydrologic Model (NHM)](https://www.usgs.gov/mission-areas/water-resources/science/national-hydrologic-model-infrastructure).

Builds the five baseline calibration targets documented in [Hay and others (2022), USGS TM 6-B10](https://doi.org/10.3133/tm6B10) by spatially aggregating gridded source datasets to an NHM HRU fabric using [gdptools](https://github.com/rmcd-mscb/gdptools).

## Calibration Targets

| Target | PRMS Variable | Sources | Method | Time Step |
|---|---|---|---|---|
| Runoff | `basin_cfs` | NHM-MWBM | MWBM uncertainty | Monthly |
| AET | `hru_actet` | MWBM + MOD16A2 + SSEBop | Multi-source min/max | Monthly |
| Recharge | `recharge` | Reitz 2017 + WaterGAP 2.2a | Normalized min/max | Annual |
| Soil Moisture | `soil_rechr` | MERRA-2 + NCEP/NCAR + NLDAS-MOSAIC + NLDAS-NOAH | Normalized min/max | Monthly + Annual |
| Snow Cover | `snowcov_area` | MOD10C1 | MODIS CI bounds | Daily |

## Quick Start

```bash
# Install with pixi
pixi install

# Create a run workspace
pixi run init -- --fabric /data/gfv1.1.gpkg --id gfv11 --workdir /data/nhf-runs

# Fetch source datasets into the workspace
pixi run fetch-merra2 -- --run-dir /data/nhf-runs/2026-03-12_gfv11_v0.1.0 --period 1980/2025
pixi run fetch-nldas-mosaic -- --run-dir /data/nhf-runs/2026-03-12_gfv11_v0.1.0 --period 1980/2025
pixi run fetch-nldas-noah -- --run-dir /data/nhf-runs/2026-03-12_gfv11_v0.1.0 --period 1980/2025
pixi run fetch-ncep-ncar -- --run-dir /data/nhf-runs/2026-03-12_gfv11_v0.1.0 --period 1980/2025

# Run all targets
pixi run run -- --run-dir /data/nhf-runs/2026-03-12_gfv11_v0.1.0

# Run a single target
pixi run run-aet -- --run-dir /data/nhf-runs/2026-03-12_gfv11_v0.1.0

# Inspect the catalog
pixi run catalog-sources
pixi run catalog-variables
```

## Fetch & Consolidation Pipeline

Each `fetch` command downloads source granules and consolidates them into a single NetCDF file per source, stored in `<run_dir>/data/raw/<source_key>/`. Consolidation uses xarray to merge per-granule files, sort by time, and write a single `*_consolidated.nc`. Progress bars (tqdm + dask) are shown during download and consolidation.

| Source | Command | Granules | Consolidated Output |
|---|---|---|---|
| MERRA-2 | `fetch-merra2` | Monthly `.nc4` via earthaccess | `merra2_consolidated.nc` |
| NLDAS-2 MOSAIC | `fetch-nldas-mosaic` | Monthly `.nc4` via earthaccess | `nldas_mosaic_consolidated.nc` |
| NLDAS-2 NOAH | `fetch-nldas-noah` | Monthly `.nc` via earthaccess | `nldas_noah_consolidated.nc` |
| NCEP/NCAR | `fetch-ncep-ncar` | Annual daily `.nc` via HTTP | `ncep_ncar_consolidated.nc` |

All fetch commands support **incremental download** — months/years already in `manifest.json` are skipped. The manifest tracks provenance (download timestamps, file checksums, periods, bounding boxes).

## Configuration

Edit `config/pipeline.yml` to set:
- Fabric path and HRU ID column
- Period of record
- Which targets to enable
- Source dataset versions (e.g. MOD16A2 v006 vs v061)

## Repository Structure

```
nhf-spatial-targets/
├── catalog/
│   ├── sources.yml          # data source registry with access info
│   └── variables.yml        # calibration target definitions and range methods
├── config/
│   └── pipeline.yml         # run configuration
├── notebooks/
│   └── inspect_consolidated.ipynb  # visualize consolidated datasets
├── src/nhf_spatial_targets/
│   ├── fetch/               # per-source download & consolidation modules
│   │   ├── consolidate.py   # xarray merge into consolidated NetCDF
│   │   ├── merra2.py        # MERRA-2 via earthaccess
│   │   ├── nldas.py         # NLDAS-2 MOSAIC & NOAH via earthaccess
│   │   ├── ncep_ncar.py     # NCEP/NCAR Reanalysis via HTTP
│   │   ├── _auth.py         # Earthdata login helper
│   │   └── _period.py       # period parsing utilities
│   ├── aggregate/           # gdptools spatial aggregation
│   ├── normalize/           # normalization and CI bound construction
│   ├── targets/             # per-variable target builders
│   ├── catalog.py           # catalog interface
│   └── cli.py               # nhf-targets CLI (cyclopts)
├── tests/
├── pixi.toml
└── pyproject.toml
```

## Development

```bash
# Install dev environment
pixi install -e dev

# Set up pre-commit hooks (ruff, pytest, nbstripout)
pixi run -e dev pre-commit install

# Run tests / lint / format
pixi run -e dev test
pixi run -e dev lint
pixi run -e dev fmt
```

## Task Status

| Task | Status |
|---|---|
| Project infrastructure (pixi, ruff, pytest, pre-commit) | Done |
| CLI with cyclopts + structured logging | Done |
| Run workspace init (`nhf-targets init`) | Done |
| Catalog YAML registry (`sources.yml`, `variables.yml`) | Done |
| MERRA-2 fetch + consolidation | Done |
| NLDAS-2 MOSAIC fetch + consolidation | Done |
| NLDAS-2 NOAH fetch + consolidation | Done |
| NCEP/NCAR Reanalysis fetch + consolidation | Done |
| xarray-based NetCDF consolidation | Done |
| Progress bars (tqdm + dask ProgressBar) | Done |
| Incremental download with manifest tracking | Done |
| Notebook for inspecting consolidated datasets | Done |
| gdptools spatial aggregation to HRU fabric | Not started |
| Normalization and range-bound methods | Not started |
| Target builders (runoff, AET, recharge, soil moisture, SCA) | Not started |
| MWBM fetch (ScienceBase) | Not started |
| MOD16A2 v061 fetch | Not started |
| MOD10C1 v061 fetch | Not started |
| SSEBop fetch | Not started |
| Reitz 2017 recharge fetch (ScienceBase) | Not started |
| WaterGAP 2.2d fetch (PANGAEA) | Not started |

## Known Gaps

See `catalog/sources.yml` `status:` and `notes:` fields for per-source details.

**Resolved:**
- MWBM ScienceBase item ID — confirmed: `55fc3f98e4b05d6c4e5029a1`
- Reitz 2017 ScienceBase item ID — confirmed: `56c49126e4b0946c65219231`
- Recharge normalization window — confirmed 2000–2009 from TM 6-B10 body text
- MOD16A2 / MOD10C1 v006 → v061: both decommissioned; use v061 in new runs
- MERRA-2 variable — use `GWETTOP` (0–0.05m, dimensionless); product M2TMNXLND
- NLDAS NOAH variable names — confirmed from file inspection: `SoilM_0_10cm`, `SoilM_10_40cm`, `SoilM_40_100cm`, `SoilM_100_200cm`

**Still open:**
- WaterGAP 2.2a — registration-gated; substitute candidate is WaterGAP 2.2d on PANGAEA
- SCA CI-bounds formula — `PRMSobjfun.f` not publicly available; formula unconfirmed
- SSEBop — version and access URL used in original TM 6-B10 unconfirmed

## References

- Hay, L.E., and others, 2022, USGS TM 6-B10
- Bock, A.R., and others, 2016/2018 — NHM-MWBM
- Reitz, M., and others, 2017 — Recharge estimates
- Xia, Y., and others, 2012 — NLDAS-2
