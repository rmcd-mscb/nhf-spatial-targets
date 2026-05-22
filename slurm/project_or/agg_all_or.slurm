#!/bin/bash
#SBATCH --job-name=nhf-agg-or
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-13
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/agg_%a_%A.out
#SBATCH --error=logs/agg_%a_%A.err

# Oregon (or-spatial-targets) aggregation array. Thin wrapper: sets the fabric's
# PROJECT_DIR, then sources the shared task array + dispatch. Only PROJECT_DIR
# differs from slurm/project_gfv2/agg_all_gfv2.slurm.
#
# Index 13 (agg-margulis-wus-sr) is the OR-only SWE source (fabric_scope=[or]);
# it is a no-op on non-OR fabrics but produces real output here.
#
# This array excludes SSEBop and Daymet (remote/region-arg aggregators); the
# Oregon build also needs:
#   PROJECT_DIR=.../or-spatial-targets sbatch slurm/shared/agg_ssebop.slurm
#   PROJECT_DIR=.../or-spatial-targets sbatch slurm/shared/agg_daymet.slurm   # --region na
# See slurm/project_or/README.md for the full submit order.
#
# Usage (from the repo root so logs/ resolves there):
#   mkdir -p logs
#   sbatch slurm/project_or/agg_all_or.slurm

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets}"
# Resolve the repo from the submit cwd (sbatch spool-copies the script — #174).
REPO_DIR="${REPO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
[ -n "$REPO_DIR" ] || {
    echo "ERROR: could not resolve REPO_DIR; set REPO_DIR=/path/to/repo and resubmit" >&2
    exit 1
}

# Guard the wrong-repo case: $SLURM_SUBMIT_DIR may sit inside a *different* git
# checkout than the one holding these scripts. Require the shared body to exist
# and the tree to identify as nhf-spatial-targets before sourcing, so a stale
# clone fails loudly instead of silently running the wrong code.
BODY="$REPO_DIR/slurm/shared/agg_all.body.sh"
if [ ! -f "$BODY" ] || ! grep -q 'name = "nhf-spatial-targets"' "$REPO_DIR/pyproject.toml" 2>/dev/null; then
    echo "ERROR: REPO_DIR=$REPO_DIR (resolved from ${SLURM_SUBMIT_DIR:-$PWD}) is not an nhf-spatial-targets checkout with $BODY. Set REPO_DIR=/path/to/nhf-spatial-targets and resubmit." >&2
    exit 1
fi

source "$BODY"
