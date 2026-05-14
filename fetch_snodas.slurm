#!/bin/bash
#SBATCH --job-name=nhf-snodas
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-3          # 4 workers — adjust with --n-workers below
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G             # The consolidator streams via dask's synchronous
                             # scheduler so peak RSS is bounded to ~hundreds of
                             # MB regardless of n_days. 8G is generous headroom.
                             # Pre-#110 used 32G and still OOM-killed because the
                             # whole (366, 3351, 6935) int16 staging array (~17 GB)
                             # was resident at once. See consolidate_year_snodas
                             # comments + docs/sources/snodas.md "Prerequisites".
#SBATCH --time=48:00:00      # SNODAS daily archive is ~1.7k granules/year × ~22 years
#SBATCH --output=logs/snodas_%a_%A.out
#SBATCH --error=logs/snodas_%a_%A.err

# NHF Spatial Targets — parallel SNODAS fetch + consolidate (NSIDC G02158)
#
# Each array task downloads a non-overlapping slice of years to
# <datastore>/snodas/raw/<year>/ AND consolidates each year's daily .tar
# bundles into a CF-1.6 NetCDF at
# <datastore>/snodas/daily/snodas_daily_<year>.nc.  Years are round-robin
# assigned by worker_index so load spreads evenly. Each worker reads
# manifest.json at startup and skips years already fully consolidated
# (n_granules > 0 AND daily_path exists on disk); years recorded with
# n_granules == 0 or without daily_path are retried.
#
# To re-consolidate an already-downloaded corpus without re-fetching, just
# delete the daily NCs (or never let them be written in the first place)
# and resubmit — the per-day already_present shortcut keeps HTTP traffic
# to zero, and consolidation runs alone. See docs/sources/snodas.md
# "Backfill" for details.
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

# Flush Python stdout per-line so slurm logs (--output=...) update in
# real time instead of dumping batches when the 8 KB block buffer fills.
# Slow workloads can otherwise look frozen for tens of minutes.
export PYTHONUNBUFFERED=1

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
