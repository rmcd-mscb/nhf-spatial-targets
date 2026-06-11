#!/bin/bash
#SBATCH --job-name=nhf-rechunk
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/rechunk_%a_%A.out
#SBATCH --error=logs/rechunk_%a_%A.err

# Backfill gfv2's aggregated NCs to the #165 chunked+compressed layout (ST5).
# One SLURM array task per rechunkable source (13 total); they run concurrently.
# daymet + ssebop are intentionally left as-is (#165 ST3) and are NOT listed.
#
# SAFE TO RE-RUN: rechunk is idempotent (already-chunked files are skipped),
# atomic (tmp + rename), and value-preserving (every variable verified before
# the rename). A task that dies mid-source resumes cleanly on resubmit.
#
# USAGE
#   mkdir -p logs
#   sbatch rechunk_gfv2.slurm
#   # one source only (e.g. snodas = index 11):
#   sbatch --array=11 rechunk_gfv2.slurm
#   # bump memory if a daily source OOMs (loads one file at a time):
#   sbatch --mem=96G rechunk_gfv2.slurm
#
# Check results:
#   grep -h "Rechunked\|failed" logs/rechunk_*_<jobid>.out logs/rechunk_*_<jobid>.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# REPO_DIR defaults to the repo you ran `sbatch` from. sbatch copies the script
# to a spool dir before executing, so ${BASH_SOURCE[0]} does NOT point at the
# repo under SLURM — use $SLURM_SUBMIT_DIR (the submit cwd), falling back to
# $PWD when run directly. The 2>/dev/null||true lets the friendly cd/guard
# message below fire instead of a raw git error. Override REPO_DIR to point at
# a different checkout if you prefer; it must have the #165 rechunk command.
REPO_DIR="${REPO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"

[ -n "$REPO_DIR" ] || { echo "ERROR: could not resolve REPO_DIR (not a git" \
    "checkout?). Set REPO_DIR=... explicitly." >&2; exit 1; }
cd "$REPO_DIR" || { echo "ERROR: REPO_DIR=$REPO_DIR not found" >&2; exit 1; }
[ -f src/nhf_spatial_targets/rechunk.py ] || {
    echo "ERROR: $REPO_DIR lacks the #165 rechunk command. Checkout main or" \
         "rebase your branch onto origin/main first." >&2
    exit 1
}

# Source keys = aggregated/ subdir names. daymet/ssebop excluded by design.
SOURCES=(
    "era5_land"               # 0
    "era5_land_sd"            # 1
    "gldas_noah_v21_monthly"  # 2
    "merra2"                  # 3
    "mod10c1_v061"            # 4  (daily; largest — ~76 GB)
    "mod16a2_v061"            # 5
    "mwbm_climgrid"           # 6
    "ncep_ncar"               # 7
    "nldas_mosaic"            # 8
    "nldas_noah"              # 9
    "reitz2017"               # 10
    "snodas"                  # 11 (daily; ~21 GB)
    "watergap22d"             # 12
)

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#SOURCES[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range" >&2
    exit 2
fi
SRC="${SOURCES[$SLURM_ARRAY_TASK_ID]}"

echo "=== rechunk source: $SRC ==="
echo "=== Repo:    $REPO_DIR ($(git -C "$REPO_DIR" branch --show-current 2>/dev/null)) ==="
echo "=== Project: $PROJECT_DIR ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname) ==="

pixi run nhf-targets maintenance rechunk \
    --project-dir "$PROJECT_DIR" \
    --layer aggregated \
    --source "$SRC"

echo "=== Done:    $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
