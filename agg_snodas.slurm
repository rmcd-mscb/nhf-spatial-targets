#!/bin/bash
#SBATCH --job-name=nhf-agg-snodas
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-3          # 4 workers — adjust with N_WORKERS below
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G            # Per-worker peak ≈ one batch's weight-gen +
                             # one batch's aggregation working set. 64G is
                             # comfortable for the 1 km SNODAS source × any
                             # CONUS fabric at batch_size up to 10000.
#SBATCH --time=12:00:00      # 21 years / 4 workers ≈ 5-6 years per worker;
                             # ~1 hr/year post-#121 (no source-grid reproject)
                             # → ~6 hrs per worker plus weight-gen tax on
                             # year 1. 12 hrs is generous headroom.
#SBATCH --output=logs/agg_snodas_%a_%A.out
#SBATCH --error=logs/agg_snodas_%a_%A.err

# NHF Spatial Targets — parallel SNODAS aggregation (year-sharded).
#
# Each array task processes a non-overlapping ROUND-ROBIN slice of the
# years on disk under <datastore>/snodas/daily/, writes per-year
# aggregated NCs to <project>/data/aggregated/snodas/, and merges its
# slice into manifest.json under an advisory flock so parallel workers
# can't lose each other's provenance entries (issue #156).
#
# Weight CSVs are computed by EACH worker on its first year and
# reused across that worker's remaining years. The atomic temp+rename
# in compute_or_load_weights handles the cross-worker race when two
# workers try to populate the same weight batch concurrently —
# last-writer-wins is correct because the content is deterministic.
# Weight cache invalidation includes source CRS in the SHA-256
# fingerprint (issue #156), so the PR #155 EPSG:4326 -> EPSG:5070
# class of stale-cache bug can't recur silently.
#
# Year sharding is round-robin (same as fetch_snodas.slurm): worker 0
# takes years [0::N], worker 1 takes [1::N], etc. This keeps load
# balanced when per-year cost is uneven (early years sparse, late
# years not).
#
# Usage:
#   mkdir -p logs
#   export PROJECT_DIR=/path/to/your/project
#   sbatch agg_snodas.slurm
#
# To change the number of workers (must edit BOTH --array and N_WORKERS):
#   Edit  --array=0-2  and  N_WORKERS=3  below for 3 workers, etc.
#
# To resume after a partial run — just re-submit; no other flags needed.
# Per-year aggregated NCs already on disk are skipped by the aggregator's
# existence check. Weight CSVs already on disk pass the fingerprint check
# and are reused. The flock-protected manifest merge means a re-run
# safely adds any newly-produced files to the existing entry.
#
# To restrict to a subset of years:
#   PERIOD=2010/2020 sbatch agg_snodas.slurm

set -euo pipefail

# Flush Python stdout per-line so SLURM logs (--output=...) update in
# real time instead of dumping batches when the 8 KB block buffer fills.
export PYTHONUNBUFFERED=1

# Must match the --array upper bound + 1 (i.e. --array=0-3 → N_WORKERS=4).
N_WORKERS=4

REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
PERIOD="${PERIOD:-}"

echo "=== SNODAS agg: worker ${SLURM_ARRAY_TASK_ID}/${N_WORKERS} ==="
echo "=== Project: ${PROJECT_DIR} ==="
echo "=== Period:  ${PERIOD:-<all years on disk>} ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:    $(hostname) ==="

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} not found" >&2; exit 1; }

EXTRA_ARGS=()
if [ -n "${PERIOD}" ]; then
    EXTRA_ARGS+=("--period" "${PERIOD}")
fi

pixi run agg-snodas -- \
    --project-dir   "${PROJECT_DIR}" \
    --worker-index  "${SLURM_ARRAY_TASK_ID}" \
    --n-workers     "${N_WORKERS}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "=== Done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
