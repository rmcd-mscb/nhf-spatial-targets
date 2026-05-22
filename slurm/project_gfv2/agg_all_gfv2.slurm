#!/bin/bash
#SBATCH --job-name=nhf-agg-gfv2
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-13
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/agg_%a_%A.out
#SBATCH --error=logs/agg_%a_%A.err

# gfv2 aggregation array. Thin wrapper: sets the fabric's PROJECT_DIR, then
# sources the shared task array + dispatch. Only PROJECT_DIR differs from
# slurm/project_or/agg_all_or.slurm.
#
# Usage (from the repo root so logs/ resolves there):
#   mkdir -p logs
#   sbatch slurm/project_gfv2/agg_all_gfv2.slurm
#   sbatch --array=8-9 --mem=256G slurm/project_gfv2/agg_all_gfv2.slurm   # MODIS rerun

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
# Resolve the repo from the submit cwd (sbatch spool-copies the script, so
# $BASH_SOURCE is not in the repo — issue #174). Override REPO_DIR to run a
# worktree/branch checkout.
REPO_DIR="${REPO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
[ -n "$REPO_DIR" ] || {
    echo "ERROR: could not resolve REPO_DIR; set REPO_DIR=/path/to/repo and resubmit" >&2
    exit 1
}

source "$REPO_DIR/slurm/shared/agg_all.body.sh"
