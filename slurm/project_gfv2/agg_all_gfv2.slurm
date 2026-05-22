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
# This array excludes SSEBop and Daymet (remote/region-arg aggregators); run
# those separately:
#   PROJECT_DIR=$GFV2 sbatch slurm/shared/agg_ssebop.slurm
#   PROJECT_DIR=$GFV2 sbatch slurm/shared/agg_daymet.slurm
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
