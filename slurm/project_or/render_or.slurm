#!/bin/bash
#SBATCH --job-name=nhf-render-or
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=192G
#SBATCH --time=8:00:00
#SBATCH --output=logs/render_%j.out
#SBATCH --error=logs/render_%j.err

# Render Oregon inspection figures via scripts/render_figures.py (PR #187).
# Walks consolidated/, aggregated/, and targets/ notebooks and writes PNGs to
# docs/figures/{consolidated,aggregated,targets}/or-spatial-targets/ — without
# committing per-project .ipynb (the committed gfv2-pinned notebooks are
# repointed via a temp copy at execute time).
#
# Memory ceiling matches inspect_consolidated.slurm (the heaviest group is
# CONUS gridded pre-aggregation). Targets render uses much less; one job for
# all three groups is simpler than per-group arrays.
#
# Usage (from the repo root so logs/ resolves there):
#   mkdir -p logs
#   sbatch slurm/project_or/render_or.slurm
#   # Just one group (faster iteration):
#   GROUP=targets sbatch slurm/project_or/render_or.slurm

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets}"
GROUP="${GROUP:-all}"
# Resolve the repo from the submit cwd (sbatch spool-copies the script — #174).
REPO_DIR="${REPO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
[ -n "$REPO_DIR" ] || {
    echo "ERROR: could not resolve REPO_DIR; set REPO_DIR=/path/to/repo and resubmit" >&2
    exit 1
}

# Same identity guard as the agg/run wrappers (#184 review): require the repo
# to look like nhf-spatial-targets so a stale clone or wrong-repo submit fails
# loudly instead of running the wrong driver.
DRIVER="$REPO_DIR/scripts/render_figures.py"
if [ ! -f "$DRIVER" ] || ! grep -q 'name = "nhf-spatial-targets"' "$REPO_DIR/pyproject.toml" 2>/dev/null; then
    echo "ERROR: REPO_DIR=$REPO_DIR (resolved from ${SLURM_SUBMIT_DIR:-$PWD}) is not an nhf-spatial-targets checkout with $DRIVER. Set REPO_DIR=/path/to/nhf-spatial-targets and resubmit." >&2
    exit 1
fi

cd "$REPO_DIR"

echo "=== nhf-targets render-figures ==="
echo "=== Project: $PROJECT_DIR ==="
echo "=== Group:   $GROUP ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:    $(hostname) ==="

pixi run -e dev render-figures -- --group "$GROUP" --project-dir "$PROJECT_DIR"

echo "=== Done:    $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
