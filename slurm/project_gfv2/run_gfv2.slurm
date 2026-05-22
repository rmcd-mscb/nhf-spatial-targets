#!/bin/bash
#SBATCH --job-name=nhf-run-gfv2
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/run_%a_%A.out
#SBATCH --error=logs/run_%a_%A.err

# gfv2 target-build array. Thin wrapper over the shared target dispatch; only
# PROJECT_DIR differs from slurm/project_or/run_or.slurm. Run the agg array
# (slurm/project_gfv2/agg_all_gfv2.slurm) + agg_ssebop + agg_daymet first.
#
# Usage (from the repo root so logs/ resolves there):
#   mkdir -p logs
#   sbatch slurm/project_gfv2/run_gfv2.slurm
#   sbatch --array=0 slurm/project_gfv2/run_gfv2.slurm     # runoff only

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
REPO_DIR="${REPO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
[ -n "$REPO_DIR" ] || {
    echo "ERROR: could not resolve REPO_DIR; set REPO_DIR=/path/to/repo and resubmit" >&2
    exit 1
}

source "$REPO_DIR/slurm/shared/run_all.body.sh"
