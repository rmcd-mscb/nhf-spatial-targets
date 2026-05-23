#!/bin/bash
#SBATCH --job-name=nhf-render-or
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=192G
#SBATCH --time=8:00:00
#SBATCH --output=logs/render_%a_%A.out
#SBATCH --error=logs/render_%a_%A.err

# Oregon (or-spatial-targets) figure-render array — one task per
# notebook group (consolidated/aggregated/targets). Thin wrapper over the
# shared render dispatch; only PROJECT_DIR differs from
# slurm/project_gfv2/render_gfv2.slurm.
#
# Re-render one group without re-running the others:
#   sbatch --array=2   slurm/project_or/render_or.slurm   # targets only
#   sbatch --array=1-2 slurm/project_or/render_or.slurm   # aggregated + targets
#
# The --mem ceiling matches the heaviest group (consolidated, CONUS-gridded
# source data); aggregated + targets tasks request the same but use only a
# fraction. Per-task --mem differentiation isn't worth the array-script
# complexity at 3 groups.
#
# Usage (from the repo root so logs/ resolves there):
#   mkdir -p logs
#   sbatch slurm/project_or/render_or.slurm

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets}"
REPO_DIR="${REPO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
[ -n "$REPO_DIR" ] || {
    echo "ERROR: could not resolve REPO_DIR; set REPO_DIR=/path/to/repo and resubmit" >&2
    exit 1
}

# Wrong-repo guard (matches run_or.slurm / agg_all_or.slurm): require the
# shared body to exist and the tree to identify as nhf-spatial-targets before
# sourcing, so a stale clone fails loudly instead of silently running the
# wrong code.
BODY="$REPO_DIR/slurm/shared/render_all.body.sh"
if [ ! -f "$BODY" ] || ! grep -q 'name = "nhf-spatial-targets"' "$REPO_DIR/pyproject.toml" 2>/dev/null; then
    echo "ERROR: REPO_DIR=$REPO_DIR (resolved from ${SLURM_SUBMIT_DIR:-$PWD}) is not an nhf-spatial-targets checkout with $BODY. Set REPO_DIR=/path/to/nhf-spatial-targets and resubmit." >&2
    exit 1
fi

source "$BODY"
