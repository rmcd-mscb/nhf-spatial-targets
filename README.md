# nhf-spatial-targets

Curated calibration target datasets for the [National Hydrologic Model (NHM)](https://www.usgs.gov/mission-areas/water-resources/science/national-hydrologic-model-infrastructure).

Builds the baseline calibration targets documented in [Hay and others (2022), USGS TM 6-B10](https://doi.org/10.3133/tm6B10) by spatially aggregating gridded source datasets to an NHM HRU fabric using [gdptools](https://github.com/rmcd-mscb/gdptools). Where original sources have been retired (MOD16A2 v006, MOD10C1 v006, MERRA-Land, NHM-MWBM, WaterGAP 2.2a) the catalog substitutes the modern replacement; see [`docs/references/known-gaps-resolved.md`](docs/references/known-gaps-resolved.md).

## Calibration Targets

| Target | PRMS Variable | Sources | Method | Time Step |
|---|---|---|---|---|
| Runoff | `basin_cfs` | ERA5-Land (`ro`) + GLDAS-2.1 NOAH (`Qs_acc + Qsb_acc`) + MWBM ClimGrid (`runoff`) | NaN-aware multi-source min/max | Monthly |
| AET | `hru_actet` | SSEBop + MWBM ClimGrid + MOD16A2 v061 | Multi-source min/max | Monthly |
| Recharge | `recharge` | Reitz 2017 + WaterGAP 2.2d + ERA5-Land (`ssro`) | Normalized min/max (2000–2009) | Annual |
| Soil Moisture | `soil_rechr` | MERRA-2 + NCEP/NCAR + NLDAS-MOSAIC + NLDAS-NOAH | Normalized min/max (per calendar month) | Monthly + Annual |
| Snow Cover | `snowcov_area` | MOD10C1 v061 | MODIS CI bounds (CI > 70%) | Daily |
| SWE | `pkwater_equiv` | Daymet + SNODAS + ERA5-Land (`sd`) + Margulis WUS-SR¹ | NaN-aware multi-source min/max | Daily |

¹ Margulis Western US Snow Reanalysis (NSIDC-0719) is **fabric-scoped to Oregon**
via `catalog/sources.yml` → `margulis_wus_sr.fabric_scope`. Non-Oregon projects
reduce the SWE bound to the remaining three sources at target-build time; raw
downloads remain reusable across any project sharing the datastore.

The runoff target builder is implemented (`targets/run.py`, NaN-aware across
all three sources). AET / recharge / soil moisture / snow cover / SWE
builders raise `NotImplementedError` until their per-variable modules land;
the `nhf-targets run` driver logs a `WARNING ... skipping` for any target
whose builder is still a stub.

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
#      daymet_root     (optional; path to operator-staged Daymet zarr stores)
#    .credentials.yml  — fill in NASA Earthdata and Copernicus CDS credentials
#
# 3. Accept the ERA5-Land CDS licence (one-time, per CDS account):
#    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences

# 4. Materialize credentials into ~/.cdsapirc and ~/.netrc
pixi run materialize-creds -- --project-dir /data/gfv11-targets

# 5. Validate the project (checks fabric, credentials, datastore; writes fabric.json)
pixi run validate -- --project-dir /data/gfv11-targets

# 6. Fetch source datasets into the shared datastore
pixi run fetch-era5-land       -- --project-dir /data/gfv11-targets --period 1979/2025
pixi run fetch-gldas           -- --project-dir /data/gfv11-targets --period 2000/2025
pixi run fetch-merra2          -- --project-dir /data/gfv11-targets --period 1980/2025
pixi run fetch-nldas-mosaic    -- --project-dir /data/gfv11-targets --period 1979/2025
pixi run fetch-nldas-noah      -- --project-dir /data/gfv11-targets --period 1979/2025
pixi run fetch-ncep-ncar       -- --project-dir /data/gfv11-targets --period 1979/2025
pixi run fetch-watergap22d     -- --project-dir /data/gfv11-targets --period 1979/2016
pixi run fetch-reitz2017       -- --project-dir /data/gfv11-targets --period 2000/2013
pixi run fetch-mod16a2         -- --project-dir /data/gfv11-targets --period 2000/2025
pixi run fetch-mod10c1         -- --project-dir /data/gfv11-targets --period 2000/2025
pixi run fetch-mwbm-climgrid   -- --project-dir /data/gfv11-targets               # manual stage required
pixi run fetch-daymet          -- --project-dir /data/gfv11-targets               # manual stage required
pixi run fetch-snodas          -- --project-dir /data/gfv11-targets --period 2003/2024
pixi run fetch-margulis-wus-sr -- --project-dir /data/gfv11-targets --period 1985/2021  # OR fabric only

# 7. Aggregate sources to the HRU fabric
pixi run agg-all     -- --project-dir /data/gfv11-targets                # 12 local-NC sources
pixi run agg-ssebop  -- --project-dir /data/gfv11-targets --period 2000/2023  # remote STAC
pixi run agg-daymet  -- --project-dir /data/gfv11-targets --period 1980/2024  # local zarr

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
| **Datastore** | Raw downloaded source files (ERA5-Land hourly NCs, MERRA-2 granules, MODIS HDF, SNODAS daily tars, Daymet zarrs, …) | **No** — one datastore can serve many fabrics |
| **Project** | Everything fabric-specific: config, credentials, aggregated outputs, targets, weight caches | **Yes** — one project = one HRU fabric |

This separation means expensive downloads only happen once. If you switch fabrics (e.g. GFv1.1 → GFv2.0), create a new project directory pointing at the **same datastore** — all raw data is reused.

```
/data/nhf-datastore/               # DATASTORE — shared, fabric-independent
  ├── era5_land/                   #   hourly chunks + daily/monthly consolidated NCs
  ├── gldas/                       #   monthly global granules + consolidated
  ├── merra2/                      #   monthly .nc4 files + consolidated
  ├── nldas_mosaic/                #   monthly CONUS files + consolidated
  ├── nldas_noah/                  #   monthly CONUS files + consolidated
  ├── ncep_ncar/                   #   annual daily files + consolidated
  ├── mod16a2/                     #   yearly consolidated NCs (HDF tiles deleted)
  ├── mod10c1/                     #   yearly consolidated NCs (daily files deleted)
  ├── watergap22d/                 #   single global NC4
  ├── reitz2017/                   #   annual GeoTIFFs + consolidated NC
  ├── mwbm_climgrid/               #   single monolithic monthly NC (ScienceBase)
  ├── daymet/                      #   per-region zarr stores (operator-staged)
  ├── snodas/                      #   per-year daily CF NetCDFs
  └── margulis_wus_sr/             #   per-water-year per-tile granules (OR scope)

/data/gfv11-targets/               # PROJECT — fabric-specific (GFv1.1)
  ├── config.yml                   #   fabric path + datastore path + settings
  ├── .credentials.yml             #   NASA Earthdata + CDS credentials (gitignored)
  ├── fabric.json                  #   computed fabric metadata (written by validate)
  ├── manifest.json                #   provenance: download timestamps, checksums
  ├── data/aggregated/             #   source data aggregated to this fabric's HRUs
  ├── targets/                     #   final calibration target NetCDF files
  ├── weights/                     #   gdptools weight caches (fabric × source grid)
  └── logs/                        #   structured pipeline logs

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
4. Run `materialize-creds` and `validate` for the new project
5. Skip `fetch` steps — the datastore already has the raw data
6. Run `agg` and `run` — weight caches are recomputed for the new fabric geometry

The fabric file must contain a polygon geometry column and a unique integer HRU ID column. Supported formats: GeoPackage (`.gpkg`), GeoParquet (`.parquet`), Shapefile (`.shp`).

### Workflow

`init` &rarr; edit config &rarr; `materialize-creds` &rarr; `validate` &rarr; `fetch` &rarr; `agg` &rarr; `run`

| Step | Command | What it does |
|---|---|---|
| 1 | `nhf-targets init --project-dir <dir>` | Creates project skeleton with `config.yml` and `.credentials.yml` templates |
| 2 | *(manual)* | Edit `config.yml`: set `fabric.path`, `fabric.id_col`, `datastore` path, optional `daymet_root`, and target settings. Fill in `.credentials.yml` with NASA Earthdata **and** Copernicus CDS credentials. **Accept the ERA5-Land CDS licence** (one-time) at <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences> |
| 3 | `nhf-targets materialize-credentials --project-dir <dir>` | Writes `~/.cdsapirc` (CDS) and `~/.netrc` (NASA Earthdata) from `.credentials.yml`. Re-run whenever credentials change. |
| 4 | `nhf-targets validate --project-dir <dir>` | Verifies config, fabric, datastore, credentials; writes `fabric.json` and merges into `manifest.json` |
| 5 | `nhf-targets fetch <source> --project-dir <dir> [--period YYYY/YYYY]` | Downloads source granules into `<datastore>/<source_key>/` (skipped if already fetched per the manifest) |
| 6 | `nhf-targets agg <source> --project-dir <dir>` | Aggregates source data to HRU fabric. SSEBop and Daymet take an explicit `--period` (remote STAC / multi-year zarr); others infer the period from available years on disk. |
| 7 | `nhf-targets run --project-dir <dir>` | Builds calibration targets from aggregated data; stub builders are skipped with a warning |

**Key paths:**

- `<project>/config.yml` — project configuration (fabric, datastore, daymet_root, targets, dir_mode)
- `<project>/fabric.json` — computed fabric metadata (bbox, HRU count, CRS, sha256)
- `<project>/manifest.json` — provenance record (sources, periods, completion records, run history)
- `<project>/.credentials.yml` — NASA Earthdata + CDS credentials (gitignored, never commit)
- `<datastore>/<source_key>/` — shared raw downloads (fabric-independent, can be on a separate drive)
- `<project>/data/aggregated/` — spatially aggregated outputs, native units, `id_col`-sorted (fabric-specific)
- `<project>/targets/` — final calibration target datasets
- `<project>/weights/` — gdptools weight caches (fabric × source grid, reusable across runs)

## Fetch & Consolidation Pipeline

Each `fetch` command downloads source granules and consolidates them into CF-1.6 compliant NetCDFs stored in `<datastore>/<source_key>/`. Consolidation uses xarray to merge per-granule files and writes consolidated outputs with CF-1.6 attributes sourced from `catalog/sources.yml` via `fetch/consolidate.py:apply_cf_metadata`. Progress bars (tqdm + dask) are shown during download and consolidation.

| Source | Command | Access Method | Consolidated Output |
|---|---|---|---|
| ERA5-Land | `fetch era5-land` | CDS API (monthly chunks, 0.1°) — **requires CDS licence; see note below** | per-year hourly + `era5_land_daily_<year>.nc` + `era5_land_monthly_<year>.nc` |
| GLDAS-2.1 NOAH | `fetch gldas` | earthaccess (monthly global NetCDF) | `gldas_consolidated.nc` |
| MERRA-2 | `fetch merra2` | earthaccess (monthly `.nc4`) | `merra2_consolidated.nc` |
| NLDAS-2 MOSAIC | `fetch nldas-mosaic` | earthaccess (monthly NetCDF) | `nldas_mosaic_consolidated.nc` |
| NLDAS-2 NOAH | `fetch nldas-noah` | earthaccess (monthly NetCDF) | `nldas_noah_consolidated.nc` |
| NCEP/NCAR | `fetch ncep-ncar` | HTTP (annual daily `.nc`) | `ncep_ncar_consolidated.nc` |
| MOD16A2 v061 | `fetch mod16a2` | earthaccess (8-day HDF4, 500m) | `mod16a2_v061_<year>_consolidated.nc` |
| MOD10C1 v061 | `fetch mod10c1` | earthaccess (daily HDF4, 0.05°) | `mod10c1_v061_<year>_consolidated.nc` |
| WaterGAP 2.2d | `fetch watergap22d` | pangaeapy (single NC4) | `watergap22d_qrdif_cf.nc` |
| Reitz 2017 | `fetch reitz2017` | sciencebasepy (annual GeoTIFF) | `reitz2017_consolidated.nc` |
| MWBM ClimGrid | `fetch mwbm-climgrid` | manual download (ScienceBase, CAPTCHA-gated ~7.5 GB NC) | `ClimGrid_WBM.nc` (verify-only) |
| Daymet V4 R1 | `fetch daymet` | manual staging (ORNL DAAC regional zarrs) | `daymet_{na,hi,pr}.zarr` (verify-only, recorded in manifest) |
| SNODAS | `fetch snodas` | earthaccess + HTTPS (NSIDC G02158 daily `.tar`) | `snodas_daily_<year>.nc` |
| Margulis WUS-SR | `fetch margulis-wus-sr` | earthaccess (NSIDC-0719 per-WY per-tile) | raw granules under `raw/<year>/` (consolidation TBD) |
| **all** | `fetch all` | runs all 14 sources sequentially | — |

All fetch commands are **incremental** — periods (or per-year completion records, for sources that track them) already recorded in `manifest.json` are skipped on re-run. `mwbm-climgrid` and `daymet` are verify-only: the operator stages the file(s) manually (see `docs/sources/mwbm_climgrid.md`, `docs/sources/daymet.md`), then the fetch command registers them in the manifest.

> **ERA5-Land CDS licence** — Before running `fetch era5-land` for the first time, you must accept the dataset licence in your Copernicus CDS account. Log in at <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences> and accept any pending licences. Without this step the fetch will fail with `403 Forbidden: required licences not accepted`.

### Datastore Storage Estimates

Rough per-source estimates for a full NHM modelling period (1979–2025 for sources that allow it). ERA5-Land dominates because raw per-year hourly files are retained for idempotent re-runs; SNODAS adds substantially once the daily archive is consolidated for 2003+.

| Source | Period fetched | Files kept on disk | Estimated size |
|---|---|---|---|
| ERA5-Land | 1979–2025 (47 yr × 4 vars) | per-year hourly + daily + monthly NCs | **150–250 GB** |
| MERRA-2 | 1980–2025 (45 yr) | ~540 monthly global `.nc4` + consolidated | **5–8 GB** |
| NLDAS-2 MOSAIC | 1979–2025 (47 yr) | ~564 monthly CONUS `.nc4` + consolidated | **2–4 GB** |
| NLDAS-2 NOAH | 1979–2025 (47 yr) | ~564 monthly CONUS `.nc4` + consolidated | **2–4 GB** |
| NCEP/NCAR | 1979–2025 (47 yr) | Daily annual files + consolidated | **1–2 GB** |
| MOD16A2 v061 | 2000–2025 (26 yr) | HDF tiles deleted; per-year consolidated NCs at 500 m CONUS | **20–35 GB** |
| MOD10C1 v061 | 2000–2025 (26 yr) | Daily files deleted; per-year consolidated NCs at 0.05° | **5–10 GB** |
| GLDAS-2.1 NOAH | 2000–2025 (26 yr) | ~312 monthly global granules + consolidated | **2–4 GB** |
| Reitz 2017 | 2000–2013 (14 yr) | Annual GeoTIFFs + consolidated NC | **2–4 GB** |
| WaterGAP 2.2d | 1901–2016 (full) | Single global NC4 | **< 1 GB** |
| MWBM ClimGrid | 1895–2020 (full) | Single global NC (~7.5 GB published) | **~8 GB** |
| Daymet V4 R1 | 1980–2024 (NA/HI/PR) | Regional zarr stores, operator-staged | **~1–2 TB** if all three regions kept |
| SNODAS | 2003–2024 (22 yr) | per-year daily NCs (raw `.tar` optional) | **80–150 GB** |
| Margulis WUS-SR | 1985–2021 (WUS) | Per-water-year per-tile granules (OR fabric only) | **~20 GB** |
| **Total (typical CONUS run, excluding Daymet)** | | | **~300–500 GB** |

These are order-of-magnitude estimates. Daymet's regional zarrs are sized for the full dataset; in practice the operator stages only the regions they need (NA covers CONUS).

### Running Fetches: PC vs HPC

#### On a PC / workstation

Run each fetch sequentially in a terminal. Each command is independently resumable — partial runs are safe:

```bash
# Once per project setup:
pixi run materialize-creds -- --project-dir /data/gfv11-targets
pixi run validate          -- --project-dir /data/gfv11-targets

# Then run fetches one at a time (or in separate terminal windows):
pixi run fetch-era5-land -- --project-dir /data/gfv11-targets --period 1979/2025
pixi run fetch-gldas     -- --project-dir /data/gfv11-targets --period 2000/2025
# ... etc.
```

Expect ERA5-Land and MODIS fetches to take many hours for a full 1979–2025 period; SNODAS spans ~37k daily granules. All fetches are incremental — interrupting and restarting picks up where it left off.

#### On HPC (SLURM)

Six SLURM scripts at the repo root cover all sources:

- [`fetch_all.slurm`](fetch_all.slurm) — 14-element array, one task per source (general case)
- [`fetch_era5_land.slurm`](fetch_era5_land.slurm) — sharded array (default 4 workers) for ERA5-Land specifically; respects CDS per-user throttling
- [`fetch_snodas.slurm`](fetch_snodas.slurm) — sharded array (default 4 workers) for SNODAS; per-day idempotent
- [`fetch_margulis_wus_sr.slurm`](fetch_margulis_wus_sr.slurm) — single task; Oregon-scoped, no sharding yet

**Prerequisites — complete these before submitting:**

1. Project is initialised and `config.yml` is filled in
2. Credentials materialised: `pixi run materialize-creds -- --project-dir <dir>`
3. Project validated: `pixi run validate -- --project-dir <dir>` (writes `fabric.json`)
4. ERA5-Land CDS licence accepted at <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences>
5. For `mwbm-climgrid` and `daymet`: the source file(s) are pre-staged in the datastore (see `docs/sources/`)
6. `PROJECT_DIR` exported in the environment (or edit the default at the top of each script)

```bash
# From the repo root:
mkdir -p logs
export PROJECT_DIR=/path/to/gfv11-targets

# General fetch-all array (14 sources):
sbatch fetch_all.slurm

# Single source by index (e.g. rerun MERRA-2 only):
sbatch --array=2 fetch_all.slurm

# Force re-fetch (re-runs per-year work for sources that track per-year completion):
FORCE=1 sbatch --array=8 fetch_all.slurm

# Monitor jobs:
squeue -u $USER

# Check output / errors (format: logs/fetch_<arrayindex>_<jobid>.out/err):
tail -f logs/fetch_0_*.out
```

Array index → source mapping (`fetch_all.slurm`):

| Index | Source | Period in script | Notes |
|---|---|---|---|
| 0 | ERA5-Land | 1979/2025 | CDS API; licence required; very large |
| 1 | GLDAS-2.1 NOAH | 2000/2025 | NASA GES DISC via earthaccess |
| 2 | MERRA-2 | 1980/2025 | NASA GES DISC via earthaccess |
| 3 | NLDAS-2 MOSAIC | 1979/2025 | NASA GES DISC via earthaccess |
| 4 | NLDAS-2 NOAH | 1979/2025 | NASA GES DISC via earthaccess |
| 5 | NCEP/NCAR | 1979/2025 | NOAA PSL via HTTP |
| 6 | WaterGAP 2.2d | 1979/2016 | PANGAEA; dataset ends 2016 |
| 7 | Reitz 2017 | 2000/2013 | USGS ScienceBase; dataset ends 2013 |
| 8 | MOD16A2 v061 | 2000/2025 | LP DAAC via earthaccess |
| 9 | MOD10C1 v061 | 2000/2025 | NSIDC via earthaccess |
| 10 | MWBM ClimGrid | 1979/2020 | ScienceBase, manual download |
| 11 | Daymet V4 R1 | 1980/2024 | ORNL DAAC zarr, manual staging |
| 12 | SNODAS | 2003/2024 | NSIDC G02158 via earthaccess |
| 13 | Margulis WUS-SR | 1985/2021 | NSIDC-0719 via earthaccess; OR-fabric only |

Most fetch routines are network I/O-bound; the general script allocates 1 CPU and 128 GB RAM per task with a 24-hour wall-clock limit. Per-source scripts (`fetch_era5_land.slurm`, `fetch_snodas.slurm`, `fetch_margulis_wus_sr.slurm`) tune memory and concurrency for their workload — notably SNODAS uses only 8 GB because PR #110's dask-streaming consolidator bounds peak RSS. Override `PROJECT_DIR` and `REPO_DIR` via environment before submission. SLURM directives (`--account`, `--partition`) at the top of each script may need adjustment for your cluster.

### Running Aggregation: PC vs HPC

#### On a PC / workstation

```bash
# One source at a time
pixi run agg-era5-land -- --project-dir /data/gfv11-targets
pixi run agg-mod10c1   -- --project-dir /data/gfv11-targets

# All 12 tier-1/2 local-NC sources sequentially
pixi run agg-all -- --project-dir /data/gfv11-targets

# SSEBop (remote STAC) — takes an explicit period
pixi run agg-ssebop -- --project-dir /data/gfv11-targets --period 2000/2023

# Daymet (local zarr) — takes a period and a region (default 'na')
pixi run agg-daymet -- --project-dir /data/gfv11-targets --period 1980/2024 --region na
```

Each aggregator writes one or more NetCDFs to `<project>/data/aggregated/<source_key>/`, in native units, with rows sorted by `id_col` (canonical order enforced at emission; see CLAUDE.md). Per-batch weights are cached under `<project>/weights/` and are reusable across runs sharing the same fabric × source grid.

#### On HPC (SLURM)

Three scripts at the repo root:

- [`agg_all.slurm`](agg_all.slurm) — 12-element array for local-NC aggregators
- [`agg_ssebop.slurm`](agg_ssebop.slurm) — single job for SSEBop (remote STAC)
- [`agg_daymet.slurm`](agg_daymet.slurm) — single job for Daymet (local zarr, per-region)

**Prerequisites:**

1. Datastore hydrated (see fetch section above)
2. `pixi run validate -- --project-dir <dir>` completed (writes `fabric.json`)
3. `PROJECT_DIR` set via environment or edited inside the scripts

```bash
mkdir -p logs
export PROJECT_DIR=/path/to/gfv11-targets

# All 12 local-NC sources (run as a SLURM array):
sbatch agg_all.slurm

# Rerun a single source by index (e.g. MOD10C1 at 9):
sbatch --array=9 agg_all.slurm

# Bump memory for a MODIS rerun that OOMed:
sbatch --array=8-9 --mem=256G agg_all.slurm

# Smaller spatial batch size:
BATCH_SIZE=2500 sbatch agg_all.slurm

# SSEBop and Daymet — separate scripts:
sbatch agg_ssebop.slurm
PERIOD=2010/2020 sbatch agg_daymet.slurm
```

Array index → source mapping for `agg_all.slurm`:

| Index | Source | Notes |
|---|---|---|
| 0 | ERA5-Land | 0.1° monthly, runoff (ro, sro, ssro) + sd |
| 1 | GLDAS-2.1 NOAH | 0.25° monthly, runoff (Qs + Qsb) |
| 2 | MERRA-2 | ~0.5° monthly, soil wetness (GWETTOP) |
| 3 | NCEP/NCAR | ~1.9° monthly, soil moisture |
| 4 | NLDAS-2 MOSAIC | 0.125° monthly, soil moisture (3 layers) |
| 5 | NLDAS-2 NOAH | 0.125° monthly, soil moisture (4 layers) |
| 6 | WaterGAP 2.2d | 0.5° monthly, diffuse recharge |
| 7 | Reitz 2017 | 800 m annual recharge |
| 8 | MOD16A2 v061 | 500 m 8-day AET (sinusoidal; fill-mask + masked_mean) — memory-heavy |
| 9 | MOD10C1 v061 | 0.05° daily SCA (CI-gated + masked_mean) — memory-heavy |
| 10 | SNODAS | ~1 km daily SWE (CONUS) — WGS84-native, weight gen is the hot path |
| 11 | MWBM ClimGrid | 2.5 arcmin monthly, 4 vars (clipped to 1979/2020 in-script) |

All jobs are CPU/memory-bound; `agg_all.slurm` allocates 1 CPU and 128 GB RAM per task with a 24-hour wall-clock limit. Override `BATCH_SIZE` (default 10000 HRUs/batch, tuned for 128 GB) with `BATCH_SIZE=2500 sbatch agg_all.slurm` if a source OOMs. WGS84-native sources (SNODAS, GLDAS) cost ~5–8× more in weight generation than projected sources (Daymet, MODIS sinusoidal) at comparable resolution — see [`docs/architecture/transformation-pipeline.md`](docs/architecture/transformation-pipeline.md).

### Running Targets: PC vs HPC

```bash
# Build all enabled targets for a project
pixi run run -- --project-dir /data/gfv11-targets

# Build one target
pixi run run-runoff -- --project-dir /data/gfv11-targets
pixi run run-aet    -- --project-dir /data/gfv11-targets   # stub: skipped with warning
```

On HPC, [`run_all.slurm`](run_all.slurm) submits a 6-element array (runoff, aet, rch, som, sca, swe). Targets whose builders are still stubs raise `NotImplementedError`, which the CLI catches and logs as a `WARNING ... skipping` so the array doesn't fail noisily.

### Inspection Notebooks

`notebooks/consolidated/` and `notebooks/aggregated/` contain per-target sanity-check notebooks that compare sources at the gridded (pre-aggregation) and HRU (post-aggregation) scales. Run them locally, or headlessly on HPC via:

- [`inspect_consolidated.slurm`](inspect_consolidated.slurm) — 5-element array, `notebooks/consolidated/inspect_consolidated_<target>.ipynb` (192 GB by default; CONUS-wide gridded comparisons)
- [`inspect_aggregated.slurm`](inspect_aggregated.slurm) — 5-element array, `notebooks/aggregated/inspect_aggregated_<target>.ipynb`

These inspection arrays stay 5-element — there is no SWE inspection notebook yet (the SWE target builder is unimplemented; see #101).

Set `SAVE_FIGURES=1` to also write rendered panels under `docs/figures/{consolidated,aggregated}/<project>/`. `notebooks/targets/inspect_target_runoff.ipynb` inspects the final per-HRU runoff bounds.

## Aggregation Transformation Policy

Where a transformation runs depends on the spatial scale at which it is
defined. Aggregation (gdptools area-weighted mean) is a one-way information
bottleneck — pixel-defined operations must run pre-aggregation, HRU-defined
operations must run post-aggregation, linear operations commute and live
downstream by convention. Full architectural reference is in
[`docs/architecture/transformation-pipeline.md`](docs/architecture/transformation-pipeline.md);
quick rules and the `mean` vs `masked_mean` `stat_method` policy are summarised
in `CLAUDE.md`. The aggregated NCs at `<project>/data/aggregated/<source_key>/`
carry **native variable names**, **native units**, and **honest NaN HRUs**
— unit conversions, multi-source combination, and any imputation happen
downstream in `targets/`.

## Repository Structure

```
nhf-spatial-targets/
├── catalog/
│   ├── sources.yml          # data source registry with access info
│   └── variables.yml        # calibration target definitions and range methods
├── config/
│   └── pipeline.yml         # reference run configuration
├── docs/
│   ├── architecture/        # transformation-pipeline policy & rationale
│   ├── references/          # TM 6-B10 crib sheet, lessons-learned, known-gaps-resolved
│   ├── sources/             # per-source operator notes (manual-staging caveats etc.)
│   └── figures/             # rendered inspection panels (gitignored except small samples)
├── notebooks/
│   ├── consolidated/        # gridded source sanity checks (pre-aggregation)
│   ├── aggregated/          # HRU-aggregated source sanity checks
│   ├── targets/             # final calibration target inspections
│   ├── visualize_*.ipynb    # per-source visualization notebooks
│   ├── inspect_margulis_wus_sr.ipynb
│   ├── extract_geofabric.ipynb
│   └── debug_mod16a2_agg.ipynb
├── src/nhf_spatial_targets/
│   ├── cli.py               # nhf-targets CLI (cyclopts)
│   ├── _logging.py          # structured logging setup
│   ├── catalog.py           # catalog interface to YAML files
│   ├── credentials.py       # ~/.cdsapirc and ~/.netrc materialisation
│   ├── defaults.py          # default config schema and merge logic
│   ├── workspace.py         # Project dataclass, path resolution, make_dir()
│   ├── validate.py          # preflight checks; writes fabric.json + manifest merge
│   ├── init_run.py          # project skeleton creation
│   ├── fetch/               # per-source download & consolidation
│   │   ├── consolidate.py   # xarray merge + CF-1.6 (single entry point)
│   │   ├── era5_land.py     # ERA5-Land via CDS API
│   │   ├── gldas.py         # GLDAS-2.1 NOAH via earthaccess
│   │   ├── merra2.py        # MERRA-2 via earthaccess
│   │   ├── nldas.py         # NLDAS-2 MOSAIC & NOAH via earthaccess
│   │   ├── ncep_ncar.py     # NCEP/NCAR Reanalysis via HTTP
│   │   ├── modis.py         # MOD16A2 & MOD10C1 via earthaccess
│   │   ├── pangaea.py       # WaterGAP 2.2d via pangaeapy
│   │   ├── reitz2017.py     # Reitz 2017 via sciencebasepy
│   │   ├── mwbm_climgrid.py # USGS MWBM (ClimGrid) ScienceBase register
│   │   ├── daymet.py        # Daymet V4 zarr register (verify-only)
│   │   ├── snodas.py        # SNODAS daily SWE via NSIDC G02158
│   │   ├── margulis_wus_sr.py  # NSIDC-0719 via earthaccess
│   │   ├── sciencebase.py   # ScienceBase helpers
│   │   ├── _auth.py         # Earthdata login helper
│   │   └── _period.py       # period parsing utilities
│   ├── aggregate/           # spatial aggregation to HRU fabric
│   │   ├── _driver.py       # generic per-source agg driver (period iteration, manifest)
│   │   ├── _adapter.py      # SourceAdapter protocol + stat_method dispatch
│   │   ├── _coords.py       # CRS / coordinate normalisation helpers
│   │   ├── batching.py      # KD-tree spatial batching
│   │   ├── gdptools_agg.py  # gdptools wrapper used by adapters
│   │   ├── ssebop.py        # SSEBop AET (remote STAC; per-year NCs)
│   │   ├── era5_land.py     # ERA5-Land monthly
│   │   ├── gldas.py         # GLDAS-2.1 NOAH monthly
│   │   ├── merra2.py        # MERRA-2 monthly
│   │   ├── ncep_ncar.py     # NCEP/NCAR monthly
│   │   ├── nldas_mosaic.py  # NLDAS-2 MOSAIC monthly
│   │   ├── nldas_noah.py    # NLDAS-2 NOAH monthly
│   │   ├── watergap22d.py   # WaterGAP 2.2d monthly
│   │   ├── reitz2017.py     # Reitz 2017 annual
│   │   ├── mod16a2.py       # MOD16A2 v061 8-day (fill-mask + masked_mean)
│   │   ├── mod10c1.py       # MOD10C1 v061 daily (CI gate + masked_mean)
│   │   ├── mwbm_climgrid.py # MWBM ClimGrid monthly
│   │   ├── daymet.py        # Daymet V4 daily SWE (per-region zarr)
│   │   └── snodas.py        # SNODAS daily SWE (CONUS)
│   ├── normalize/
│   │   └── methods.py       # per-HRU normalisation, NN-fill helpers (used by target builders)
│   └── targets/             # per-variable target builders
│       ├── _common.py       # write_target_nc, project-config defaults
│       ├── run.py           # runoff (3-source NaN-aware) — implemented
│       ├── aet.py           # stub
│       ├── rch.py           # stub
│       ├── som.py           # stub
│       └── sca.py           # stub
├── tests/                   # pytest suite (run via `pixi run -e dev test`)
├── *.slurm                  # HPC scripts: fetch / agg / run / inspect
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

# Install the dev environment (adds pytest, ruff, mypy, pre-commit, ipykernel, gh)
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

Always commit via `pixi run git commit` (not bare `git commit`) — the pre-commit
config drives ruff and pytest through `pixi run`, and a PreToolUse hook in
`.claude/settings.json` enforces this for Claude sessions. See `CLAUDE.md` for
the full pre-commit quality gate and git workflow.

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
| Project init / validate / config (project + datastore + defaults layer) | Done |
| Catalog YAML registry (`sources.yml`, `variables.yml`) | Done |
| CF-1.6 compliant NetCDF outputs across consolidate / aggregate / target | Done |
| Canonical `id_col`-sorted row order at emission (#93) | Done |
| Incremental download with manifest tracking (read-merge-write, #97 fix) | Done |
| Fetch + consolidate: ERA5-Land, GLDAS, MERRA-2, NLDAS MOSAIC/NOAH, NCEP/NCAR, MOD16A2 v061, MOD10C1 v061, WaterGAP 2.2d, Reitz 2017, MWBM ClimGrid, Daymet (verify), SNODAS, Margulis WUS-SR | Done |
| Aggregator driver + `SourceAdapter` protocol (`mean` / `masked_mean`) | Done |
| Aggregation: SSEBop (STAC), ERA5-Land, GLDAS, MERRA-2, NCEP/NCAR, NLDAS MOSAIC/NOAH, WaterGAP 2.2d, Reitz 2017, MOD16A2 v061, MOD10C1 v061, MWBM ClimGrid, Daymet, SNODAS | Done |
| Spatial batching for large fabrics | Done |
| Visualization + inspection notebooks (gridded + HRU) | Done |
| `fetch all` and `agg all` driver commands | Done |
| **Runoff target builder** (3-source NaN-aware; PR #92/#95) | Done |
| AET target builder | Not started (stub) |
| Recharge target builder | Not started (stub) |
| Soil moisture target builder | Not started (stub) |
| Snow-covered-area target builder | Not started (stub) |
| SWE target builder (4-source, OR-scoped Margulis) | Not started (umbrella #101) |
| Margulis WUS-SR per-WY granule consolidation | Not started (umbrella #101) |

## Known Gaps

See `catalog/sources.yml` `status:` and `notes:` fields for per-source details.

**Resolved:** see [`docs/references/known-gaps-resolved.md`](docs/references/known-gaps-resolved.md) — includes the runoff source replacement (NHM-MWBM → ERA5-Land + GLDAS + MWBM ClimGrid), MOD16A2/MOD10C1 v006 → v061 upgrade, MERRA-Land → MERRA-2 variable mapping, NHM-MWBM family substitution, WaterGAP 2.2a → 2.2d, Reitz 2017 ScienceBase ID, SSEBop access via USGS NHGF STAC, and the MOD16A2 fill-mask fix (PR #88) that restored seasonal swing.

**Still open:**

- SCA CI-bounds formula — `PRMSobjfun.f` not publicly available; exact upper/lower bound formula from a daily SCA + CI pair remains unconfirmed.
- SNODAS archive day-gaps and rare partial-bundle days (3–5/yr in 2004–2007) — fill policy TBD; do not add interpolation to the consolidator unilaterally.

## References

- Hay, L.E., and others, 2022, USGS TM 6-B10 — calibration target methodology
- Markstrom, S.L., and others, 2015, USGS TM 6-B7 — PRMS-V model description (defines `soil_rechr`, `pkwater_equiv`, etc.)
- Regan, R.S., and others, 2018, USGS TM 6-B9 — NHM infrastructure
- Muñoz-Sabater, J., and others, 2021 — ERA5-Land
- Rodell, M., and others, 2004 — GLDAS
- Reitz, M., and others, 2017 — Recharge estimates
- Xia, Y., and others, 2012 — NLDAS-2
- Wieczorek, M.E., and others, 2024 — USGS MWBM (ClimGrid-forced)
- Margulis, S.A., and others — Western US Snow Reanalysis (NSIDC-0719)
