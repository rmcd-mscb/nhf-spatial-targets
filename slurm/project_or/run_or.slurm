#!/bin/bash
#SBATCH --job-name=nhf-run-or
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/run_%a_%A.out
#SBATCH --error=logs/run_%a_%A.err

# Oregon (or-spatial-targets) target-build array — the "nhf-targets run for
# Oregon" entry point. Thin wrapper over the shared target dispatch; only
# PROJECT_DIR differs from slurm/project_gfv2/run_gfv2.slurm.
#
# Builds the 5 implemented targets (runoff/aet/rch/som/swe). Index 4 (run-sca)
# is a stub: it raises NotImplementedError, which the CLI catches and logs as
# "skipping", so the array completes without a failure. SWE picks up the OR-only
# Margulis WUS-SR source via fabric_scope.
#
# Prerequisites (see slurm/project_or/README.md for the full order):
#   agg_all_or.slurm + agg_ssebop.slurm + agg_daymet.slurm (--region na) done.
#
# Usage (from the repo root so logs/ resolves there):
#   mkdir -p logs
#   sbatch slurm/project_or/run_or.slurm

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets}"
REPO_DIR="${REPO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
[ -n "$REPO_DIR" ] || {
    echo "ERROR: could not resolve REPO_DIR; set REPO_DIR=/path/to/repo and resubmit" >&2
    exit 1
}

source "$REPO_DIR/slurm/shared/run_all.body.sh"
