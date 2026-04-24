#!/bin/bash
#SBATCH --job-name=nhf-era5-land
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-3          # 4 workers — adjust with --n-workers below
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00      # ERA5-Land is slow; 48 h per worker is conservative
#SBATCH --output=logs/era5_land_%a_%A.out
#SBATCH --error=logs/era5_land_%a_%A.err

# NHF Spatial Targets — parallel ERA5-Land fetch
#
# Each array task downloads and consolidates a non-overlapping slice of years.
# Years are round-robin assigned by worker_index so load spreads evenly.
# Each worker reads manifest.json first and skips years already completed by
# a prior run or sibling worker. File-level idempotency (monthly chunk files,
# daily NC mtime checks) handles partially-complete years on restart.
#
# Parallel download limit: 4 workers × 3 variables × 1 active CDS request
# each = at most 12 simultaneous queued CDS requests, well within the
# per-user limit (~20). Increase --array / N_WORKERS cautiously — the CDS
# throttles aggressively above ~4-5 simultaneous users.
#
# Usage:
#   mkdir -p logs
#   export PROJECT_DIR=/path/to/your/project
#   sbatch fetch_era5_land.slurm
#
# To change the number of workers (must edit BOTH --array and N_WORKERS):
#   Edit  --array=0-2  and  N_WORKERS=3  below for 3 workers, etc.
#
# To resume after a partial run — just re-submit; no other flags needed.
# Completed years are detected from manifest.json + on-disk file presence.
#
# To fetch a subset of years only:
#   PERIOD=2000/2010 sbatch fetch_era5_land.slurm

set -euo pipefail

# Must match the --array upper bound + 1 (i.e. --array=0-3 → N_WORKERS=4).
N_WORKERS=4

REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
PERIOD="${PERIOD:-1979/2025}"

echo "=== ERA5-Land fetch: worker ${SLURM_ARRAY_TASK_ID}/${N_WORKERS} ==="
echo "=== Period:  ${PERIOD} ==="
echo "=== Project: ${PROJECT_DIR} ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:    $(hostname) ==="

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} not found" >&2; exit 1; }

pixi run fetch-era5-land -- \
    --project-dir "${PROJECT_DIR}" \
    --period      "${PERIOD}" \
    --worker-index "${SLURM_ARRAY_TASK_ID}" \
    --n-workers    "${N_WORKERS}"

echo "=== Done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
