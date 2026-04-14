#!/bin/bash
#SBATCH --job-name=nhf-fetch
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fetch_%a_%A.out
#SBATCH --error=logs/fetch_%a_%A.err

# NHF Spatial Targets — parallel fetch array
# Submits one SLURM job per source dataset (10 total).
# Each job downloads independently; they can run concurrently.
#
# Usage (REPO_DIR and PROJECT_DIR are overridable via environment):
#   mkdir -p logs
#   export REPO_DIR=/path/to/nhf-spatial-targets
#   export PROJECT_DIR=/path/to/your/project
#   sbatch fetch_all.slurm
#
# To run a single source only (e.g. index 2 = merra2):
#   sbatch --array=2 fetch_all.slurm

set -euo pipefail

# Repo and project directories — override via environment, or edit these
# defaults for your site. The repo default mirrors the USGS Hovenweep path;
# $PROJECT_DIR must be a project directory created by `nhf-targets init`.
REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"

cd "$REPO_DIR"

# Map array index → pixi task name
FETCH_TASKS=(
    "fetch-era5-land"      # 0  — 1979–present at 0.1°, hourly (CDS)
    "fetch-gldas"          # 1  — 2000–present at 0.25°, monthly (NASA GES DISC)
    "fetch-merra2"         # 2  — 1980–present at 0.5×0.625°, monthly (NASA GES DISC)
    "fetch-nldas-mosaic"   # 3  — 1979–present at 0.125°, monthly (NASA GES DISC)
    "fetch-nldas-noah"     # 4  — 1979–present at 0.125°, monthly (NASA GES DISC)
    "fetch-ncep-ncar"      # 5  — 1948–present at ~1.875°, daily→monthly (NOAA PSL)
    "fetch-watergap22d"    # 6  — 1901–2016 at 0.5°, monthly (PANGAEA) — ends 2016
    "fetch-reitz2017"      # 7  — 2000–2013 at 800m, annual (ScienceBase) — ends 2013
    "fetch-mod16a2"        # 8  — 2000–present at 500m, 8-day (LP DAAC/earthaccess)
    "fetch-mod10c1"        # 9  — 2000–present at 0.05°, daily (NSIDC/earthaccess)
)

# Map array index → fetch period (YYYY/YYYY)
# Periods chosen to cover the NHM model run (1979–2025) as fully as each
# source allows.  Sources with fixed ends (WaterGAP 2.2d: 2016,
# Reitz 2017: 2013, MODIS: 2000) are clamped to their available range.
FETCH_PERIODS=(
    "1979/2025"   # 0  era5-land
    "2000/2025"   # 1  gldas       (v2.1 starts 2000)
    "1980/2025"   # 2  merra2      (starts 1980)
    "1979/2025"   # 3  nldas-mosaic
    "1979/2025"   # 4  nldas-noah
    "1979/2025"   # 5  ncep-ncar   (product starts 1948; clamp to model start)
    "1979/2016"   # 6  watergap22d (dataset ends 2016)
    "2000/2013"   # 7  reitz2017   (dataset ends 2013)
    "2000/2025"   # 8  mod16a2     (MODIS starts 2000)
    "2000/2025"   # 9  mod10c1     (MODIS starts 2000)
)

TASK="${FETCH_TASKS[$SLURM_ARRAY_TASK_ID]}"
PERIOD="${FETCH_PERIODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Array task $SLURM_ARRAY_TASK_ID: $TASK  period=$PERIOD ==="
echo "=== Start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:  $(hostname) ==="

pixi run "$TASK" -- --project-dir "$PROJECT_DIR" --period "$PERIOD"

echo "=== Done:  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
