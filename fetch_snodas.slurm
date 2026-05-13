#!/bin/bash
#SBATCH --job-name=nhf-snodas
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-3          # 4 workers — adjust with --n-workers below
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00      # SNODAS daily archive is ~1.7k granules/year × ~22 years
#SBATCH --output=logs/snodas_%a_%A.out
#SBATCH --error=logs/snodas_%a_%A.err

# NHF Spatial Targets — parallel SNODAS fetch (NSIDC G02158)
#
# Each array task downloads a non-overlapping slice of years to
# <datastore>/snodas/raw/<year>/. Years are round-robin assigned by
# worker_index so load spreads evenly. Each worker reads manifest.json
# at startup and skips years already recorded with n_granules > 0
# (years recorded with n_granules: 0 are retried, in case CMR coverage
# fills in retroactively).
#
# Decoding the raw int16 binary + ENVI .Hdr bundles into CF NetCDFs is
# DEFERRED to the SNODAS aggregate follow-up issue (umbrella #101);
# this script only stages the raw bytes. See docs/sources/snodas.md.
#
# Usage:
#   mkdir -p logs
#   export PROJECT_DIR=/path/to/your/project
#   sbatch fetch_snodas.slurm
#
# To change the number of workers (must edit BOTH --array and N_WORKERS):
#   Edit  --array=0-2  and  N_WORKERS=3  below for 3 workers, etc.
#
# To resume after a partial run — just re-submit; no other flags needed.
# Completed years are detected from manifest.json + on-disk file presence.
#
# To fetch a subset of years only:
#   PERIOD=2003/2010 sbatch fetch_snodas.slurm

set -euo pipefail

# Must match the --array upper bound + 1 (i.e. --array=0-3 → N_WORKERS=4).
N_WORKERS=4

REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
PERIOD="${PERIOD:-2003/2024}"

echo "=== SNODAS fetch: worker ${SLURM_ARRAY_TASK_ID}/${N_WORKERS} ==="
echo "=== Period:  ${PERIOD} ==="
echo "=== Project: ${PROJECT_DIR} ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:    $(hostname) ==="

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} not found" >&2; exit 1; }

pixi run fetch-snodas -- \
    --project-dir "${PROJECT_DIR}" \
    --period       "${PERIOD}" \
    --worker-index "${SLURM_ARRAY_TASK_ID}" \
    --n-workers    "${N_WORKERS}"

echo "=== Done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
