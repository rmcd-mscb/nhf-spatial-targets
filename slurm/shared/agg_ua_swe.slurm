#!/bin/bash
#SBATCH --job-name=nhf-agg-ua-swe
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-3          # 4 workers — adjust with N_WORKERS below
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G           # Per-worker peak ≈ one batch's weight-gen +
                             # one batch's aggregation working set. 128G
                             # matches the proven --mem in agg_all_<fabric>.slurm
                             # and agg_snodas.slurm; UA SWE is 4 km (coarser
                             # than SNODAS ~1 km) so the per-batch pixel set is
                             # smaller, but keep 128G for headroom on the CONUS
                             # fabric. Per CLAUDE.md memory: prefer SLURM --mem
                             # bumps over in-code memory refactors.
#SBATCH --time=16:00:00      # 41 calendar years (1982-2022) / 4 workers ≈ 10-11
                             # years per worker. UA SWE is pre-projected to
                             # EPSG:5070 (matches WEIGHT_GEN_CRS), so weight-gen
                             # skips the source-grid reproject and is fast (the
                             # ~5-8x tax that hits WGS84-native sources does not
                             # apply). 16 hrs is generous headroom.
#SBATCH --output=logs/agg_ua_swe_%a_%A.out
#SBATCH --error=logs/agg_ua_swe_%a_%A.err

# NHF Spatial Targets — parallel UA SWE (NSIDC-0719) aggregation (year-sharded).
#
# Each array task processes a non-overlapping ROUND-ROBIN slice of the
# calendar years on disk under <datastore>/ua_swe/daily/ (ua_swe_daily_<YYYY>.nc,
# CY1982-2022, EPSG:5070), writes per-year aggregated NCs to
# <project>/data/aggregated/ua_swe/, and merges its slice into manifest.json
# under an advisory flock so parallel workers can't lose each other's
# provenance entries.
#
# This aggregates THREE variables to the fabric: `swe` (a SWE-target source)
# and the depth-thresholded `snow_covered_fraction` (the second SCA-target
# source, #237), plus `snow_depth`. The snow_covered_fraction binary is derived
# PRE-aggregation in aggregate/ua_swe.py's pre-aggregate hook; the
# `snow_covered_area.depth_threshold_mm` config value (default 1.0 mm) is read
# at agg time and stamped on the output (changing it requires re-aggregating).
#
# Weight CSVs are computed by EACH worker on its first year and reused across
# that worker's remaining years; the atomic temp+rename in
# compute_or_load_weights handles the cross-worker race (last-writer-wins is
# correct because the content is deterministic). Weight cache invalidation
# includes source CRS in the SHA-256 fingerprint, so an EPSG mismatch can't
# silently reuse a stale cache.
#
# Year sharding is round-robin (same as agg_snodas.slurm): worker 0 takes years
# [0::N], worker 1 takes [1::N], etc. — keeps load balanced when per-year cost
# is uneven.
#
# PREREQUISITE: the consolidated NCs must be CALENDAR year
# (ua_swe_daily_<YYYY>.nc), written by PR-A2's consolidator. A datastore that
# still holds the pre-A2 water-year files (ua_swe_daily_WY<YYYY>.nc) must be
# re-consolidated first (rm the WY files, re-run `nhf-targets fetch ua-swe` —
# raw under <datastore>/ua_swe/raw/ is cached, so no re-download).
#
# Usage:
#   mkdir -p logs
#   export PROJECT_DIR=/path/to/your/project
#   sbatch slurm/shared/agg_ua_swe.slurm
#
# To change the number of workers (must edit BOTH --array and N_WORKERS):
#   Edit  --array=0-2  and  N_WORKERS=3  below for 3 workers, etc.
#
# To resume after a partial run — just re-submit; no other flags needed.
# Per-year aggregated NCs already on disk are skipped by the aggregator's
# existence check; weight CSVs already on disk pass the fingerprint check and
# are reused; the flock-protected manifest merge makes a re-run additive.
#
# To restrict to a subset of years:
#   PERIOD=2000/2010 sbatch slurm/shared/agg_ua_swe.slurm

set -euo pipefail

# Flush Python stdout per-line so SLURM logs update in real time.
export PYTHONUNBUFFERED=1

# Must match the --array upper bound + 1 (i.e. --array=0-3 → N_WORKERS=4).
N_WORKERS=4

REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR to the target project dir, e.g. PROJECT_DIR=/caldera/.../gfv2-spatial-targets}"
PERIOD="${PERIOD:-}"

echo "=== UA SWE agg: worker ${SLURM_ARRAY_TASK_ID}/${N_WORKERS} ==="
echo "=== Project: ${PROJECT_DIR} ==="
echo "=== Period:  ${PERIOD:-<all years on disk>} ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:    $(hostname) ==="

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} not found" >&2; exit 1; }

EXTRA_ARGS=()
if [ -n "${PERIOD}" ]; then
    EXTRA_ARGS+=("--period" "${PERIOD}")
fi

pixi run agg-ua-swe -- \
    --project-dir   "${PROJECT_DIR}" \
    --worker-index  "${SLURM_ARRAY_TASK_ID}" \
    --n-workers     "${N_WORKERS}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "=== Done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
