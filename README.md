# nhf-spatial-targets

Curated calibration target datasets for the [National Hydrologic Model (NHM)](https://www.usgs.gov/mission-areas/water-resources/science/national-hydrologic-model-infrastructure).

Builds the five baseline calibration targets documented in [Hay and others (2022), USGS TM 6-B10](https://doi.org/10.3133/tm6B10) by spatially aggregating gridded source datasets to an NHM HRU fabric using [gdptools](https://github.com/rmcd-mscb/gdptools).

## Calibration Targets

| Target | PRMS Variable | Sources | Method | Time Step |
|---|---|---|---|---|
| Runoff | `basin_cfs` | ERA5-Land (`ro`) + GLDAS-2.1 NOAH (`Qs_acc + Qsb_acc`) | Multi-source min/max | Monthly |
| AET | `hru_actet` | MOD16A2 v061 + SSEBop | Multi-source min/max | Monthly |
| Recharge | `recharge` | Reitz 2017 + WaterGAP 2.2d | Normalized min/max | Annual |
| Soil Moisture | `soil_rechr` | MERRA-2 + NCEP/NCAR + NLDAS-MOSAIC + NLDAS-NOAH | Normalized min/max | Monthly + Annual |
| Snow Cover | `snowcov_area` | MOD10C1 v061 | MODIS CI bounds | Daily |

## Quick Start

**Install pixi first** (the only system prerequisite — manages Python and all dependencies):

```bash
# Linux / macOS
curl -fsSL https://pixi.sh/install.sh | sh

# Windows (PowerShell)
irm https://pixi.sh/install.ps1 | iex
```

Then restart your shell (or `source ~/.bashrc`) so `pixi` is on your PATH.

```bash
# Install the project environment
pixi install

# 1. Create a project directory skeleton
pixi run init -- --project-dir /data/gfv11-targets

# 2. Edit the generated files:
#    config.yml  — set:
#      fabric.path     (path to HRU GeoPackage or Parquet)
#      fabric.id_col   (HRU identifier column name)
#      datastore       (shared raw-data directory, SEPARATE from project,
#                       e.g. /data/nhf-datastore — see "Projects & Datastore" below)
#    .credentials.yml  — fill in NASA Earthdata and Copernicus CDS credentials
#
# 3. Accept the ERA5-Land CDS licence (one-time, per CDS account):
#    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences

# 4. Materialize credentials into ~/.cdsapirc and ~/.netrc
pixi run materialize-credentials -- --project-dir /data/gfv11-targets

# 5. Validate the project (checks fabric, credentials, datastore; writes fabric.json)
pixi run validate -- --project-dir /data/gfv11-targets

# 6. Fetch source datasets into the shared datastore (run individually or via SLURM)
pixi run fetch-era5-land    -- --project-dir /data/gfv11-targets --period 1979/2025
pixi run fetch-gldas        -- --project-dir /data/gfv11-targets --period 2000/2025
pixi run fetch-merra2       -- --project-dir /data/gfv11-targets --period 1980/2025
pixi run fetch-nldas-mosaic -- --project-dir /data/gfv11-targets --period 1979/2025
pixi run fetch-nldas-noah   -- --project-dir /data/gfv11-targets --period 1979/2025
pixi run fetch-ncep-ncar    -- --project-dir /data/gfv11-targets --period 1979/2025
pixi run fetch-mod16a2      -- --project-dir /data/gfv11-targets --period 2000/2025
pixi run fetch-mod10c1      -- --project-dir /data/gfv11-targets --period 2000/2025
pixi run fetch-watergap22d  -- --project-dir /data/gfv11-targets --period 1979/2016
pixi run fetch-reitz2017    -- --project-dir /data/gfv11-targets --period 2000/2013

# 7. Aggregate (sources served remotely via STAC — no local download)
pixi run agg-ssebop -- --project-dir /data/gfv11-targets --period 2000/2020  # --batch-size N (default 500)

# 8. Build calibration targets
pixi run run -- --project-dir /data/gfv11-targets

# Inspect the catalog
pixi run catalog-sources
pixi run catalog-variables
```

## Projects & Datastore

The pipeline has two independent concepts:

| Concept | What it holds | Tied to fabric? |
|---|---|---|
| **Datastore** | Raw downloaded source files (ERA5-Land hourly NCs, MERRA-2 granules, MODIS HDF, …) | **No** — one datastore can serve many fabrics |
| **Project** | Everything fabric-specific: config, credentials, aggregated outputs, targets, weight caches | **Yes** — one project = one HRU fabric |

This separation means expensive downloads only happen once. If you switch fabrics (e.g. GFv1.1 → GFv2.0), create a new project directory pointing at the **same datastore** — all raw data is reused.

```
/data/nhf-datastore/               # DATASTORE — shared, fabric-independent
  ├── era5_land/                   #   hourly monthly chunks + per-year NCs
  ├── gldas/                       #   monthly global granules
  ├── merra2/                      #   monthly .nc4 files + consolidated NC
  ├── nldas_mosaic/                #   monthly CONUS files + consolidated NC
  ├── nldas_noah/                  #   monthly CONUS files + consolidated NC
  ├── ncep_ncar/                   #   annual daily files + consolidated NC
  ├── mod16a2/                     #   yearly consolidated NCs (HDF tiles deleted)
  ├── mod10c1/                     #   yearly consolidated NCs (daily files deleted)
  ├── watergap22d/                 #   single global NC4
  └── reitz2017/                   #   annual GeoTIFFs + consolidated NC

/data/gfv11-targets/               # PROJECT — fabric-specific (GFv1.1)
  ├── config.yml                   #   fabric path + datastore path + settings
  ├── .credentials.yml             #   NASA Earthdata + CDS credentials (gitignored)
  ├── fabric.json                  #   computed fabric metadata (written by validate)
  ├── manifest.json                #   provenance: download timestamps, checksums
  ├── data/aggregated/             #   source data aggregated to this fabric's HRUs
  ├── targets/                     #   final calibration target NetCDF files
  └── weights/                     #   gdptools weight caches (fabric × source grid)

/data/gfv20-targets/               # PROJECT — same datastore, different fabric (GFv2.0)
  ├── config.yml                   #   points to same /data/nhf-datastore
  └── ...                          #   all fetch data reused; only agg/targets differ
```

### Using a different fabric

To build targets for a new fabric (e.g. upgrading from GFv1.1 to GFv2.0, or a custom HRU delineation):

1. Create a new project directory — **do not reuse an existing one**:
   ```bash
   pixi run init -- --project-dir /data/gfv20-targets
   ```
2. Edit `/data/gfv20-targets/config.yml`:
   - Set `fabric.path` to the new fabric file (GeoPackage, GeoParquet, or Shapefile)
   - Set `fabric.id_col` to the HRU identifier column in that file
   - Set `datastore` to the **same** datastore path used by other projects (so fetched data is reused)
3. Copy or symlink `.credentials.yml` from an existing project, or fill it in fresh
4. Run `materialize-credentials` and `validate` for the new project
5. Skip `fetch` steps — the datastore already has the raw data
6. Run `agg` and `run` — weight caches are recomputed for the new fabric geometry

The fabric file must contain a polygon geometry column and a unique integer HRU ID column. Supported formats: GeoPackage (`.gpkg`), GeoParquet (`.parquet`), Shapefile (`.shp`).

### Workflow

`init` &rarr; edit config &rarr; `materialize-credentials` &rarr; `validate` &rarr; `fetch` &rarr; `run`

| Step | Command | What it does |
|---|---|---|
| 1 | `nhf-targets init --project-dir <dir>` | Creates project skeleton with `config.yml` and `.credentials.yml` templates |
| 2 | *(manual)* | Edit `config.yml`: set `fabric.path`, `fabric.id_col`, `datastore` path, and target settings. Fill in `.credentials.yml` with NASA Earthdata **and** Copernicus CDS credentials. **Accept the ERA5-Land CDS licence** (one-time) at <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences> |
| 3 | `nhf-targets materialize-credentials --project-dir <dir>` | Writes `~/.cdsapirc` (CDS) and `~/.netrc` (NASA Earthdata) from `.credentials.yml`. Re-run whenever credentials change. |
| 4 | `nhf-targets validate --project-dir <dir>` | Verifies config, fabric, datastore, credentials; writes `fabric.json` and `manifest.json` |
| 5 | `nhf-targets fetch <source> --project-dir <dir> --period YYYY/YYYY` | Downloads source granules into `<datastore>/<source_key>/` (skipped if already fetched) |
| 6 | `nhf-targets agg <source> --project-dir <dir> --period YYYY/YYYY` | Aggregates remote/STAC data to HRU fabric |
| 7 | `nhf-targets run --project-dir <dir>` | Builds calibration targets from fetched/aggregated data |

**Key paths:**

- `<project>/config.yml` — project configuration (fabric, datastore, targets, dir_mode)
- `<project>/fabric.json` — computed fabric metadata (bbox, HRU count, CRS, sha256)
- `<project>/manifest.json` — provenance record (download timestamps, checksums, periods)
- `<project>/.credentials.yml` — NASA Earthdata + CDS credentials (gitignored, never commit)
- `<datastore>/<source_key>/` — shared raw downloads (fabric-independent, can be on a separate drive)
- `<project>/data/aggregated/` — spatially aggregated outputs (fabric-specific)
- `<project>/targets/` — final calibration target datasets
- `<project>/weights/` — gdptools weight caches (fabric × source grid, reusable across runs)

## Fetch & Consolidation Pipeline

Each `fetch` command downloads source granules and consolidates them into a single CF-1.6 compliant NetCDF file per source, stored in `<datastore>/<source_key>/`. Consolidation uses xarray to merge per-granule files, sort by time, and write a single consolidated output. Progress bars (tqdm + dask) are shown during download and consolidation.

| Source | Command | Access Method | Consolidated Output |
|---|---|---|---|
| ERA5-Land | `fetch era5-land` | CDS API (monthly CONUS NetCDF, 0.1°) — **requires CDS licence; see note below** | `era5_land_{var}_{year}.nc` per variable/year |
| GLDAS-2.1 NOAH | `fetch gldas` | earthaccess (monthly global NetCDF) | `gldas_consolidated.nc` |
| MERRA-2 | `fetch merra2` | earthaccess (monthly `.nc4`) | `merra2_consolidated.nc` |
| NLDAS-2 MOSAIC | `fetch nldas-mosaic` | earthaccess (monthly NetCDF) | `nldas_mosaic_consolidated.nc` |
| NLDAS-2 NOAH | `fetch nldas-noah` | earthaccess (monthly NetCDF) | `nldas_noah_consolidated.nc` |
| NCEP/NCAR | `fetch ncep-ncar` | HTTP (annual daily `.nc`) | `ncep_ncar_consolidated.nc` |
| MOD16A2 v061 | `fetch mod16a2` | earthaccess (8-day HDF4, 500m) | `mod16a2_v061_{year}_consolidated.nc` |
| MOD10C1 v061 | `fetch mod10c1` | earthaccess (daily HDF4, 0.05&deg;) | `mod10c1_v061_{year}_consolidated.nc` |
| WaterGAP 2.2d | `fetch watergap22d` | pangaeapy (single NC4) | `watergap22d_qrdif_cf.nc` |
| Reitz 2017 | `fetch reitz2017` | sciencebasepy (annual GeoTIFF) | `reitz2017_consolidated.nc` |

All fetch commands support **incremental download** — periods already recorded in `manifest.json` are skipped.

> **ERA5-Land CDS licence** — Before running `fetch era5-land` for the first time, you must accept the dataset licence in your Copernicus CDS account. Log in at <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences> and accept any pending licences. Without this step the fetch will fail with `403 Forbidden: required licences not accepted`.

### Datastore Storage Estimates

Rough per-source estimates for the reference pipeline period (2000–2010 for most sources; see `config/pipeline.yml`). ERA5-Land dominates because raw per-year hourly files are retained for idempotent re-runs.

| Source | Period fetched | Files kept on disk | Estimated size |
|---|---|---|---|
| ERA5-Land | 2000–2010 (11 yr) | 3 vars × 11 yr hourly CONUS NCs + consolidated daily + monthly | **30–60 GB** |
| MERRA-2 | 1982–2010 (29 yr) | 348 monthly global `.nc4` files + consolidated | **3–5 GB** |
| NLDAS-2 MOSAIC | 1982–2010 (29 yr) | 348 monthly CONUS `.nc4` files + consolidated | **1–3 GB** |
| NLDAS-2 NOAH | 1982–2010 (29 yr) | 348 monthly CONUS `.nc4` files + consolidated | **1–3 GB** |
| NCEP/NCAR | 1982–2010 (29 yr) | Daily annual files deleted after monthly aggregation | **< 1 GB** |
| MOD16A2 v061 | 2000–2010 (11 yr) | HDF tiles deleted; 11 yearly consolidated NCs at 500m CONUS | **10–20 GB** |
| MOD10C1 v061 | 2000–2010 (11 yr) | Daily files deleted; 11 yearly consolidated NCs at 0.05° | **2–5 GB** |
| GLDAS-2.1 NOAH | 2000–2010 (11 yr) | ~132 monthly global granules + consolidated | **1–2 GB** |
| Reitz 2017 | 2000–2013 (14 yr) | 28 zip files + consolidated NC | **2–4 GB** |
| WaterGAP 2.2d | 1901–2016 (full) | Single global NC4 | **< 1 GB** |
| **Total** | | | **~50–100 GB** |

These are order-of-magnitude estimates. Actual sizes depend on CDS compression, exact MODIS tile counts for the CONUS bbox, and how far back the pipeline period extends. Sizes scale roughly linearly with the number of years fetched.

### Running Fetches: PC vs HPC

#### On a PC / workstation

Run each fetch sequentially in a terminal. Each command is independently resumable — partial runs are safe:

```bash
# Once per project setup:
pixi run materialize-credentials -- --project-dir /data/gfv11-targets
pixi run validate               -- --project-dir /data/gfv11-targets

# Then run fetches one at a time (or in separate terminal windows):
pixi run fetch-era5-land    -- --project-dir /data/gfv11-targets --period 1979/2025
pixi run fetch-gldas        -- --project-dir /data/gfv11-targets --period 2000/2025
pixi run fetch-merra2       -- --project-dir /data/gfv11-targets --period 1980/2025
# ... etc.
```

Expect ERA5-Land and MODIS fetches to take many hours for a full 1979–2025 period. All fetches are incremental — interrupting and restarting picks up where it left off.

#### On HPC (SLURM)

The [`fetch_all.slurm`](fetch_all.slurm) script submits all 10 sources as a SLURM array job (one job per array index), so they run concurrently on separate compute nodes.

**Prerequisites — complete these before submitting:**

1. Project is initialised and `config.yml` is filled in
2. Credentials materialised: `pixi run materialize-credentials -- --project-dir <dir>`
3. Project validated: `pixi run validate -- --project-dir <dir>` (writes `fabric.json`)
4. ERA5-Land CDS licence accepted at <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences>
5. `PROJECT_DIR` updated inside `fetch_all.slurm` to your actual project path

```bash
# From the repo root:
mkdir -p logs

# Edit PROJECT_DIR at the top of fetch_all.slurm, then submit all 10 sources:
sbatch fetch_all.slurm

# Or submit a single source by index (e.g. rerun ERA5-Land only):
sbatch --array=0 fetch_all.slurm

# Monitor jobs:
squeue -u $USER

# Check output / errors (format: logs/fetch_<arrayindex>_<jobid>.out/err):
tail -f logs/fetch_0_*.out   # ERA5-Land live log
cat  logs/fetch_2_*.err      # MERRA-2 error output
```

Array index → source mapping:

| Index | Source | Period in script | Notes |
|---|---|---|---|
| 0 | ERA5-Land | 1979/2025 | CDS API; licence required; ~30–60 GB |
| 1 | GLDAS-2.1 NOAH | 2000/2025 | NASA GES DISC via earthaccess |
| 2 | MERRA-2 | 1980/2025 | NASA GES DISC via earthaccess |
| 3 | NLDAS-2 MOSAIC | 1979/2025 | NASA GES DISC via earthaccess |
| 4 | NLDAS-2 NOAH | 1979/2025 | NASA GES DISC via earthaccess |
| 5 | NCEP/NCAR | 1979/2025 | NOAA PSL via HTTP |
| 6 | WaterGAP 2.2d | 1979/2016 | PANGAEA; dataset ends 2016 |
| 7 | Reitz 2017 | 2000/2013 | USGS ScienceBase; dataset ends 2013 |
| 8 | MOD16A2 v061 | 2000/2025 | LP DAAC via earthaccess; ~10–20 GB |
| 9 | MOD10C1 v061 | 2000/2025 | NSIDC via earthaccess |

All fetch routines are network I/O-bound (sequential downloads); the script allocates 1 CPU and 128 GB RAM per task with a 24-hour wall-clock limit. The 128 GB figure is driven by ERA5-Land hourly CONUS concatenation (each year's 12 monthly chunks are combined into a per-year NC via `xr.open_mfdataset`); other tasks use far less. Override `PROJECT_DIR` and `REPO_DIR` via environment before submission (e.g. `export PROJECT_DIR=/path/to/project; sbatch fetch_all.slurm`). SLURM directives (`--account`, `--partition`, `--chdir`) at the top of `fetch_all.slurm` may need to be adjusted for your cluster.

### Running Aggregation: PC vs HPC

#### On a PC / workstation

Run aggregators individually or all at once:

```bash
# One source at a time
pixi run agg-era5-land -- --project-dir /data/gfv11-targets
pixi run agg-mod10c1   -- --project-dir /data/gfv11-targets

# All nine tier-1/2 sources sequentially
pixi run agg-all -- --project-dir /data/gfv11-targets

# SSEBop (remote STAC, takes a period)
pixi run agg-ssebop -- --project-dir /data/gfv11-targets --period 2000/2023
```

Each aggregator writes one NetCDF per source to `$PROJECT_DIR/data/aggregated/` and caches per-batch weights under `$PROJECT_DIR/weights/`.

#### On HPC (SLURM)

Two scripts at the repo root:

- [`agg_all.slurm`](agg_all.slurm) — 9-element array for local-NC aggregators
- [`agg_ssebop.slurm`](agg_ssebop.slurm) — single job for SSEBop (remote STAC)

**Prerequisites:**

1. Datastore hydrated (see fetch section above)
2. `pixi run validate -- --project-dir <dir>` completed (writes `fabric.json`)
3. `PROJECT_DIR` set via environment or edited inside the scripts

```bash
# From the repo root:
mkdir -p logs
export PROJECT_DIR=/path/to/gfv11-targets

# All 9 local-NC sources (may run in parallel on the cluster):
sbatch agg_all.slurm

# Rerun a single source by index (e.g. MOD10C1 at 8):
sbatch --array=8 agg_all.slurm

# Bump memory for a MODIS rerun that OOMed:
sbatch --array=7-8 --mem=256G agg_all.slurm

# SSEBop (remote STAC, separate script):
sbatch agg_ssebop.slurm

# Monitor:
squeue -u $USER

# Inspect logs (format: logs/agg_<arrayindex>_<jobid>.out/err):
tail -f logs/agg_8_*.out   # MOD10C1 live log
cat  logs/agg_7_*.err      # MOD16A2 error output
```

Array index → source mapping for `agg_all.slurm`:

| Index | Source | Notes |
|---|---|---|
| 0 | ERA5-Land | 0.1° monthly, runoff (ro, sro, ssro) |
| 1 | GLDAS-2.1 NOAH | 0.25° monthly, runoff (Qs + Qsb) |
| 2 | MERRA-2 | ~0.5° monthly, soil wetness |
| 3 | NCEP/NCAR | ~1.9° monthly, soil moisture |
| 4 | NLDAS-2 MOSAIC | 0.125° monthly, soil moisture (3 layers) |
| 5 | NLDAS-2 NOAH | 0.125° monthly, soil moisture (4 layers) |
| 6 | WaterGAP 2.2d | 0.5° monthly, diffuse recharge |
| 7 | MOD16A2 v061 | 500m 8-day AET (sinusoidal) — memory-heavy |
| 8 | MOD10C1 v061 | 0.05° daily SCA (CI-masked) — memory-heavy |

All nine jobs are CPU/memory-bound; the script allocates 1 CPU and 128 GB RAM per task with a 24-hour wall-clock limit. The 128 GB figure is sized for MOD10C1's daily 2000-present stack after the aggregator's in-memory `.load()`. Override `BATCH_SIZE` (default 10000 HRUs/batch, tuned for 128 GB) with `BATCH_SIZE=2500 sbatch agg_all.slurm` if a source OOMs. SLURM directives (`--account`, `--partition`) at the top of each script may need adjustment for non-Hovenweep clusters.

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
| `fetch all` command (sequential, all 10 sources) | Done |
| ERA5-Land fetch + monthly CDS splitting (closes #41 follow-up) | Done |
| GLDAS-2.1 NOAH fetch | Done |
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
