# Getting started

Hands-on quick start for first-time operators. For the long-form README with rationale + storage estimates + HPC SLURM cookbook, see the [GitHub README](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/README.md).

## Prerequisites

**Install [pixi](https://pixi.sh)** — the only system prerequisite. Pixi manages Python and every dependency.

```bash
# Linux / macOS
curl -fsSL https://pixi.sh/install.sh | sh

# Windows (PowerShell)
irm https://pixi.sh/install.ps1 | iex
```

Restart your shell (`source ~/.bashrc`) so `pixi` is on your PATH.

## Bootstrap a project

```bash
git clone git@github.com:rmcd-mscb/nhf-spatial-targets.git
cd nhf-spatial-targets
pixi install                # default env

# 1. Create a project skeleton at a path of your choosing
pixi run init -- --project-dir /data/my-targets

# 2. Edit /data/my-targets/config.yml — set:
#      fabric.path     (HRU GeoPackage / Parquet / Shapefile)
#      fabric.id_col   (HRU identifier column name)
#      datastore       (separate from project, e.g. /data/nhf-datastore)
# 3. Fill in /data/my-targets/.credentials.yml with NASA Earthdata + Copernicus CDS credentials
# 4. Accept the ERA5-Land CDS licence (one-time): https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#manage-licences

# 5. Materialize credentials into ~/.cdsapirc + ~/.netrc
pixi run materialize-creds -- --project-dir /data/my-targets

# 6. Validate (writes fabric.json, checks credentials)
pixi run validate -- --project-dir /data/my-targets

# 7. Fetch source data (each step is incremental + resumable)
pixi run fetch-all -- --project-dir /data/my-targets

# 8. Aggregate sources to the HRU fabric
pixi run agg-all -- --project-dir /data/my-targets
pixi run agg-ssebop -- --project-dir /data/my-targets --period 2000/2023
pixi run agg-daymet -- --project-dir /data/my-targets --period 1980/2024

# 9. Build calibration targets
pixi run run -- --project-dir /data/my-targets
```

## What you get

```
/data/my-targets/
├── config.yml                project configuration
├── fabric.json               computed fabric metadata
├── manifest.json             provenance record (sources, periods, run history)
├── data/aggregated/          source data aggregated to your HRU fabric
└── targets/                  final calibration target NetCDF files (6 of them)
```

Each target NC carries `(lower_bound, upper_bound)` per `(HRU, time)` in the variable's PRMS units, plus an `n_sources` int8 diagnostic and an optional `_nn_filled.nc` companion.

## Projects vs datastore

The pipeline cleanly separates two concepts:

- **Datastore** (`<datastore>/`) — raw downloaded source files, **fabric-independent**. One datastore can serve many fabrics.
- **Project** (`<project>/`) — everything fabric-specific (config, credentials, aggregated outputs, targets, weight caches).

This means expensive downloads happen **once**. Building targets for a new fabric just creates a new project pointing at the same datastore — no re-fetching.

Detailed layout + sizing tables: see the [README on GitHub](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/README.md#projects--datastore).

## Day-to-day commands

```bash
pixi run catalog-sources       # inspect the source registry
pixi run catalog-variables     # inspect the calibration-target definitions

# Build a single target
pixi run run-runoff -- --project-dir /data/my-targets
pixi run run-sca    -- --project-dir /data/my-targets
pixi run run-swe    -- --project-dir /data/my-targets
```

## Where to go next

- [Architecture · Transformation pipeline](architecture/transformation-pipeline.md) — pre/post-aggregation policy, `mean` vs `masked_mean`, canonical row order
- [Architecture · Python patterns](architecture/python-patterns.md) — `from __future__ import annotations`, atomic writes, fingerprint caches, why every module looks the way it does
- [Sources](sources/index.md) — per-source operator notes for every gridded dataset
- [References · Known gaps resolved](references/known-gaps-resolved.md) — every source substitution since the original TM 6-B10
- [Contributing](contributing.md) — how to add a new source / target / fabric
- [Conventions](conventions.md) — pre-commit gate, manifest concurrency, test discipline (the `CLAUDE.md` doc)

## HPC use

For SLURM submission scripts (fetch / agg / run / inspect arrays), prerequisites, and per-source memory tuning, see the [README §Running on HPC](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/README.md#running-fetches-pc-vs-hpc) sections.
