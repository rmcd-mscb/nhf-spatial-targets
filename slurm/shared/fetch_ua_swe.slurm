#!/bin/bash
#SBATCH --job-name=nhf-ua-swe
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-3          # 4 workers — adjust with N_WORKERS below
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G            # Unlike the SNODAS streaming consolidator, the UA
                             # SWE consolidator windows two adjacent water-year
                             # rasters into one calendar year AND pre-projects to
                             # EPSG:5070, so peak RSS is higher. 64G is the proven
                             # value from the original per-project run; per
                             # CLAUDE.md memory, prefer a --mem bump over an
                             # in-code memory refactor.
#SBATCH --time=6:00:00       # Full archive (42 water years, ~90 MB raw each) ≈
                             # ~1.5 h wall-clock across 4 workers per the original
                             # run; 6 h is headroom. A re-consolidate (raw already
                             # under ua_swe/raw/) skips all downloads and is faster.
#SBATCH --output=logs/ua_swe_%a_%A.out
#SBATCH --error=logs/ua_swe_%a_%A.err

# NHF Spatial Targets — parallel UA SWE fetch + consolidate (NSIDC-0719).
#
# Each array task downloads a non-overlapping ROUND-ROBIN slice of the published
# water-year rasters (4km_SWE_Depth_WY<YYYY>_v01.nc) to <datastore>/ua_swe/raw/,
# then consolidates each CALENDAR year — [Jan 1–Sep 30] from WY X plus
# [Oct 1–Dec 31] from WY X+1 — and pre-projects it to EPSG:5070, writing
# <datastore>/ua_swe/daily/ua_swe_daily_<YYYY>.nc (CY1982-2022). Workers merge
# into manifest.json under an advisory flock so concurrent provenance writes
# can't clobber each other.
#
# RE-CONSOLIDATE (no re-download): the raw water-year files under
# <datastore>/ua_swe/raw/ are preserved and cached. To rebuild the consolidated
# calendar-year NCs — e.g. a datastore that still holds the pre-A2 water-year
# outputs (ua_swe_daily_WY<YYYY>.nc) — delete the daily NCs and resubmit; the
# per-file already-present check keeps HTTP traffic to zero and consolidation
# runs alone:
#   rm <datastore>/ua_swe/daily/ua_swe_daily_*.nc
#   PROJECT_DIR=... sbatch slurm/shared/fetch_ua_swe.slurm
#
# Usage:
#   mkdir -p logs
#   export PROJECT_DIR=/path/to/your/project
#   sbatch slurm/shared/fetch_ua_swe.slurm
#
# To change the number of workers (must edit BOTH --array and N_WORKERS):
#   Edit  --array=0-2  and  N_WORKERS=3  below for 3 workers, etc.
#
# To resume after a partial run — just re-submit; no other flags needed.
# Completed years are detected from manifest.json + on-disk file presence.
#
# To fetch a subset of years only (calendar-year range resolves the WY files
# needed):
#   PERIOD=2000/2010 sbatch slurm/shared/fetch_ua_swe.slurm

set -euo pipefail

# Group-writable output on the shared HPC filesystem — SLURM does not
# inherit the login-shell umask. Belt-and-suspenders with os.umask(0o002)
# in the CLI (see io_nc.apply_umask_mode for why umask alone is not enough
# for the mkstemp-based atomic NetCDF writers).
umask 002

# Flush Python stdout per-line so slurm logs update in real time.
export PYTHONUNBUFFERED=1

# Must match the --array upper bound + 1 (i.e. --array=0-3 → N_WORKERS=4).
N_WORKERS=4

REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR to the target project dir, e.g. PROJECT_DIR=/caldera/.../gfv2-spatial-targets}"
PERIOD="${PERIOD:-1981/2023}"

echo "=== UA SWE fetch: worker ${SLURM_ARRAY_TASK_ID}/${N_WORKERS} ==="
echo "=== Period:  ${PERIOD} ==="
echo "=== Project: ${PROJECT_DIR} ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:    $(hostname) ==="

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} not found" >&2; exit 1; }

pixi run fetch-ua-swe -- \
    --project-dir "${PROJECT_DIR}" \
    --period       "${PERIOD}" \
    --worker-index "${SLURM_ARRAY_TASK_ID}" \
    --n-workers    "${N_WORKERS}"

echo "=== Done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
