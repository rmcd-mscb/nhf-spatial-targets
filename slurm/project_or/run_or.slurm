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
# Builds all six implemented targets (runoff/aet/rch/som/sca/swe). Index 4
# (run-sca) is the two-source MOD10C1 + UA SWE bound (#237). A target whose
# configured sources are not yet aggregated self-skips (logged "skipping") so
# the array completes without a failure. SWE picks up the OR-only Margulis
# WUS-SR source via fabric_scope.
#
# Prerequisites (see slurm/project_or/README.md for the full order):
#   agg_all_or.slurm + slurm/shared/agg_ssebop.slurm + slurm/shared/agg_daymet.slurm (--region na) done.
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

# Guard the wrong-repo case: $SLURM_SUBMIT_DIR may sit inside a *different* git
# checkout than the one holding these scripts. Require the shared body to exist
# and the tree to identify as nhf-spatial-targets before sourcing, so a stale
# clone fails loudly instead of silently running the wrong code.
BODY="$REPO_DIR/slurm/shared/run_all.body.sh"
if [ ! -f "$BODY" ] || ! grep -q 'name = "nhf-spatial-targets"' "$REPO_DIR/pyproject.toml" 2>/dev/null; then
    echo "ERROR: REPO_DIR=$REPO_DIR (resolved from ${SLURM_SUBMIT_DIR:-$PWD}) is not an nhf-spatial-targets checkout with $BODY. Set REPO_DIR=/path/to/nhf-spatial-targets and resubmit." >&2
    exit 1
fi

source "$BODY"
