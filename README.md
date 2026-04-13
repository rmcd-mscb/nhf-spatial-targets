# nhf-spatial-targets

Curated calibration target datasets for the [National Hydrologic Model (NHM)](https://www.usgs.gov/mission-areas/water-resources/science/national-hydrologic-model-infrastructure).

Builds the five baseline calibration targets documented in [Hay and others (2022), USGS TM 6-B10](https://doi.org/10.3133/tm6B10) by spatially aggregating gridded source datasets to an NHM HRU fabric using [gdptools](https://github.com/rmcd-mscb/gdptools).

## Calibration Targets

| Target | PRMS Variable | Sources | Method | Time Step |
|---|---|---|---|---|
| Runoff | `basin_cfs` | ERA5-Land (`ro`) + GLDAS-2.1 NOAH (`Qs_acc + Qsb_acc`) | Multi-source min/max | Monthly |
| AET | `hru_actet` | MWBM + MOD16A2 v061 + SSEBop | Multi-source min/max | Monthly |
| Recharge | `recharge` | Reitz 2017 + WaterGAP 2.2d | Normalized min/max | Annual |
| Soil Moisture | `soil_rechr` | MERRA-2 + NCEP/NCAR + NLDAS-MOSAIC + NLDAS-NOAH | Normalized min/max | Monthly + Annual |
| Snow Cover | `snowcov_area` | MOD10C1 v061 | MODIS CI bounds | Daily |

## Quick Start

```bash
# Install with pixi
pixi install

# 1. Create a project
pixi run init -- --project-dir /data/gfv11-targets

# 2. Edit the generated config.yml to set:
#    - fabric.path     (path to HRU GeoPackage)
#    - fabric.id_col   (HRU identifier column)
#    - datastore        (shared raw-data directory, e.g. /data/nhf-datastore)
#    Then fill in .credentials.yml with NASA Earthdata credentials.

# 3. Validate the project (writes fabric.json and manifest.json)
pixi run validate -- --project-dir /data/gfv11-targets

# 4. Fetch source datasets into the shared datastore
pixi run fetch-all -- --project-dir /data/gfv11-targets --period 2000/2020

# Or fetch individual sources:
pixi run fetch-merra2       -- --project-dir /data/gfv11-targets --period 1980/2020
pixi run fetch-nldas-mosaic -- --project-dir /data/gfv11-targets --period 1980/2020
pixi run fetch-nldas-noah   -- --project-dir /data/gfv11-targets --period 1980/2020
pixi run fetch-ncep-ncar    -- --project-dir /data/gfv11-targets --period 1980/2020
pixi run fetch-mod16a2      -- --project-dir /data/gfv11-targets --period 2000/2020
pixi run fetch-mod10c1      -- --project-dir /data/gfv11-targets --period 2000/2020
pixi run fetch-watergap22d  -- --project-dir /data/gfv11-targets --period 1901/2016
pixi run fetch-reitz2017    -- --project-dir /data/gfv11-targets --period 2000/2013

# 5. Aggregate (sources that use remote/STAC data)
pixi run agg-ssebop -- --project-dir /data/gfv11-targets --period 2000/2020  # --batch-size N (default 500)

# 6. Run all targets
pixi run run -- --project-dir /data/gfv11-targets

# Inspect the catalog
pixi run catalog-sources
pixi run catalog-variables
```

## Projects & Datastore

The pipeline separates **projects** (fabric-specific) from the **datastore** (shared raw data, fabric-independent). Multiple projects can share one datastore — if data has already been fetched, another project pointing to the same datastore will reuse it.

```
/data/nhf-datastore/               # DATASTORE (shared, fabric-independent)
  ├── merra2/                      #   consolidated NCs from fetch
  ├── nldas_mosaic/
  ├── reitz2017/
  └── ...

/data/gfv11-targets/               # PROJECT (fabric-specific)
  ├── config.yml                   #   points to datastore + fabric
  ├── .credentials.yml
  ├── fabric.json
  ├── manifest.json
  ├── data/aggregated/             #   aggregated outputs
  ├── targets/                     #   final calibration targets
  └── weights/                     #   weight caches (fabric × source grid)
```

**Workflow:** `init` &rarr; edit config &rarr; `validate` &rarr; `fetch` &rarr; `run`

| Step | Command | What it does |
|---|---|---|
| 1 | `nhf-targets init --project-dir <dir>` | Creates project skeleton with `config.yml` and `.credentials.yml` templates |
| 2 | *(manual)* | Edit `config.yml` to set fabric path, datastore, and target settings |
| 3 | `nhf-targets validate --project-dir <dir>` | Verifies config, fabric, credentials; writes `fabric.json` and `manifest.json` |
| 4 | `nhf-targets fetch <source> --project-dir <dir> --period YYYY/YYYY` | Downloads source granules into `<datastore>/<source_key>/` |
| 5 | `nhf-targets agg <source> --project-dir <dir> --period YYYY/YYYY` | Aggregates remote data to HRU fabric |
| 6 | `nhf-targets run --project-dir <dir>` | Builds calibration targets from fetched/aggregated data |

**Key paths:**

- `<project>/config.yml` — project configuration (fabric, datastore, targets, dir_mode)
- `<project>/fabric.json` — computed fabric metadata (bbox, HRU count, CRS, sha256)
- `<project>/manifest.json` — provenance record (download timestamps, checksums, periods)
- `<project>/.credentials.yml` — NASA Earthdata credentials (gitignored)
- `<datastore>/<source_key>/` — shared raw downloads (can be on a separate drive)
- `<project>/data/aggregated/` — spatially aggregated outputs
- `<project>/targets/` — final calibration target datasets
- `<project>/weights/` — gdptools weight caches (reusable across runs)

## Fetch & Consolidation Pipeline

Each `fetch` command downloads source granules and consolidates them into a single CF-1.6 compliant NetCDF file per source, stored in `<datastore>/<source_key>/`. Consolidation uses xarray to merge per-granule files, sort by time, and write a single consolidated output. Progress bars (tqdm + dask) are shown during download and consolidation.

| Source | Command | Access Method | Consolidated Output |
|---|---|---|---|
| MERRA-2 | `fetch merra2` | earthaccess (monthly `.nc4`) | `merra2_consolidated.nc` |
| NLDAS-2 MOSAIC | `fetch nldas-mosaic` | earthaccess (monthly NetCDF) | `nldas_mosaic_consolidated.nc` |
| NLDAS-2 NOAH | `fetch nldas-noah` | earthaccess (monthly NetCDF) | `nldas_noah_consolidated.nc` |
| NCEP/NCAR | `fetch ncep-ncar` | HTTP (annual daily `.nc`) | `ncep_ncar_consolidated.nc` |
| MOD16A2 v061 | `fetch mod16a2` | earthaccess (8-day HDF4, 500m) | `mod16a2_v061_{year}_consolidated.nc` |
| MOD10C1 v061 | `fetch mod10c1` | earthaccess (daily HDF4, 0.05&deg;) | `mod10c1_v061_{year}_consolidated.nc` |
| WaterGAP 2.2d | `fetch watergap22d` | pangaeapy (single NC4) | `watergap22d_qrdif_cf.nc` |
| Reitz 2017 | `fetch reitz2017` | sciencebasepy (annual GeoTIFF) | `reitz2017_consolidated.nc` |

All fetch commands support **incremental download** — periods already recorded in `manifest.json` are skipped.

## Aggregation

Sources that are accessed remotely (e.g. via STAC) are aggregated directly to the HRU fabric without local download:

| Source | Command | Method |
|---|---|---|
| SSEBop AET | `agg ssebop` | Area-weighted mean via gdptools + USGS NHGF STAC (Zarr) |

Spatial batching (KD-tree) is used to handle large fabrics efficiently. The `--batch-size` flag controls HRUs per batch (default 500).

## Repository Structure

```
nhf-spatial-targets/
├── catalog/
│   ├── sources.yml          # data source registry with access info
│   └── variables.yml        # calibration target definitions and range methods
├── config/
│   └── pipeline.yml         # reference run configuration
├── notebooks/               # visualization & inspection notebooks
│   ├── inspect_consolidated.ipynb
│   ├── extract_geofabric.ipynb
│   ├── visualize_merra2.ipynb
│   ├── visualize_nldas_mosaic.ipynb
│   ├── visualize_nldas_noah.ipynb
│   ├── visualize_ncep_ncar.ipynb
│   ├── visualize_mod16a2.ipynb
│   ├── visualize_mod10c1.ipynb
│   ├── visualize_watergap22d.ipynb
│   ├── visualize_reitz2017.ipynb
│   └── visualize_ssebop_aet.ipynb
├── src/nhf_spatial_targets/
│   ├── cli.py               # nhf-targets CLI (cyclopts)
│   ├── _logging.py          # structured logging setup
│   ├── catalog.py           # catalog interface to YAML files
│   ├── project.py         # project path resolution
│   ├── validate.py          # preflight checks
│   ├── init_run.py          # project skeleton creation
│   ├── fetch/               # per-source download & consolidation
│   │   ├── consolidate.py   # xarray merge + CF-1.6 compliance
│   │   ├── merra2.py        # MERRA-2 via earthaccess
│   │   ├── nldas.py         # NLDAS-2 MOSAIC & NOAH via earthaccess
│   │   ├── ncep_ncar.py     # NCEP/NCAR Reanalysis via HTTP
│   │   ├── modis.py         # MOD16A2 & MOD10C1 via earthaccess
│   │   ├── pangaea.py       # WaterGAP 2.2d via pangaeapy
│   │   ├── reitz2017.py     # Reitz 2017 via sciencebasepy
│   │   ├── sciencebase.py   # MWBM fetch (stub)
│   │   ├── _auth.py         # Earthdata login helper
│   │   └── _period.py       # period parsing utilities
│   ├── aggregate/           # spatial aggregation to HRU fabric
│   │   ├── ssebop.py        # SSEBop AET via STAC + gdptools
│   │   ├── batching.py      # KD-tree spatial batching
│   │   └── gdptools_agg.py  # generic gdptools aggregation (stub)
│   ├── normalize/           # normalization and CI bound methods (stub)
│   └── targets/             # per-variable target builders (stubs)
├── tests/
├── pixi.toml
└── pyproject.toml
```

## Development

### Prerequisites

Install [pixi](https://pixi.sh) — the only prerequisite. Pixi manages Python, all
dependencies, and task runners. Do **not** use pip, conda, or system Python directly.

**Linux / macOS:**

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://pixi.sh/install.ps1 | iex
```

### Setup

```bash
# Install the default environment
pixi install

# Install the dev environment (adds pytest, ruff, mypy, pre-commit, ipykernel)
pixi install -e dev

# Set up pre-commit hooks (ruff, pytest, nbstripout)
pixi run -e dev pre-commit install
```

### Day-to-day commands

```bash
pixi run -e dev test          # unit tests (excludes integration)
pixi run -e dev test-integration  # integration tests (requires credentials/network)
pixi run -e dev lint          # ruff check src/ tests/
pixi run -e dev fmt           # ruff format src/ tests/
pixi run -e dev fmt-check     # ruff format --check (used by pre-commit)
```

### Platform notes

Pixi supports **linux-64**, **osx-arm64**, and **win-64** (see `pixi.toml`).

| Feature | Linux / macOS | Windows |
|---|---|---|
| `dir_mode` in config.yml | Applied (e.g. `"2775"` for setgid + group-writable) | Ignored — Windows does not support Unix permissions |
| `libgdal-hdf4` (HDF4 support for MODIS) | Installed via conda-forge | Installed via conda-forge |
| NASA Earthdata credentials | `.credentials.yml` or `~/.netrc` | `.credentials.yml` (netrc path may differ) |
| Peak memory logging | Uses `/proc` (Linux) or `resource` (macOS) | Gracefully skipped |
| Pre-commit hooks | Fully supported | Fully supported |

## Implementation Status

| Component | Status |
|---|---|
| Project infrastructure (pixi, ruff, pytest, pre-commit) | Done |
| CLI with cyclopts + structured logging | Done |
| Project init / validate / config (project + datastore model) | Done |
| Catalog YAML registry (`sources.yml`, `variables.yml`) | Done |
| MERRA-2 fetch + consolidation | Done |
| NLDAS-2 MOSAIC fetch + consolidation | Done |
| NLDAS-2 NOAH fetch + consolidation | Done |
| NCEP/NCAR Reanalysis fetch + consolidation | Done |
| MOD16A2 v061 fetch + consolidation | Done |
| MOD10C1 v061 fetch + consolidation | Done |
| WaterGAP 2.2d fetch + consolidation | Done |
| Reitz 2017 recharge fetch + consolidation | Done |
| CF-1.6 compliant NetCDF outputs | Done |
| Incremental download with manifest tracking | Done |
| SSEBop AET aggregation (STAC + gdptools) | Done |
| Spatial batching for large fabrics | Done |
| Visualization notebooks (all sources + SSEBop agg) | Done |
| `fetch all` command (sequential, all 8 sources) | Done |
| ERA5-Land + GLDAS-2.1 NOAH runoff fetch | Not started |
| Generic gdptools aggregation module | Not started |
| Normalization and range-bound methods | Not started |
| Target builders (runoff, AET, recharge, soil moisture, SCA) | Not started |

## Known Gaps

See `catalog/sources.yml` `status:` and `notes:` fields for per-source details.

**Resolved:**
- Reitz 2017 ScienceBase item ID — confirmed: `56c49126e4b0946c65219231`
- Runoff source replacement — NHM-MWBM removed; replaced by ERA5-Land (CDS) + GLDAS-2.1 NOAH monthly. ERA5-Land ssro also added as third recharge source. Closes issue #41.
- Recharge normalization window — confirmed 2000-2009 from TM 6-B10 body text
- MOD16A2 / MOD10C1 v006 &rarr; v061: both decommissioned; use v061 in new runs
- MERRA-2 variable — use `GWETTOP` (0-0.05m, dimensionless); product M2TMNXLND
- MERRA-2 layer depths — `dzsf`=0.05m (constant), `dzrz`=1.00m, `dzpr`=spatially varying (surface to bedrock). Thicknesses in M2CONXLND.
- NLDAS NOAH variable names — confirmed: `SoilM_0_10cm`, `SoilM_10_40cm`, `SoilM_40_100cm`, `SoilM_100_200cm`
- WaterGAP 2.2d — confirmed on PANGAEA (doi:10.1594/PANGAEA.918447), CC BY-NC 4.0

**Still open:**
- SCA CI-bounds formula — `PRMSobjfun.f` not publicly available; formula unconfirmed

**Resolved (previously open):**
- SSEBop — accessed via USGS NHGF STAC catalog (collection `ssebopeta_monthly`, doi:10.5066/P9L2YMV, 2000–2023 monthly, 1km). Aggregated directly to HRU fabric via gdptools.

## References

- Hay, L.E., and others, 2022, USGS TM 6-B10
- Reitz, M., and others, 2017 — Recharge estimates
- Xia, Y., and others, 2012 — NLDAS-2
