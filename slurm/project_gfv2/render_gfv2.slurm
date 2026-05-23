#!/bin/bash
#SBATCH --job-name=nhf-render-gfv2
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=192G
#SBATCH --time=8:00:00
#SBATCH --output=logs/render_%j.out
#SBATCH --error=logs/render_%j.err

# Render gfv2 inspection figures via scripts/render_figures.py (PR #187).
# Walks consolidated/, aggregated/, and targets/ notebooks and writes PNGs to
# docs/figures/{consolidated,aggregated,targets}/gfv2-spatial-targets/. Uses
# --project-dir so the temp-notebook code path is exercised symmetrically with
# Oregon — the committed notebooks' embedded outputs are not overwritten on
# each render (unlike the older slurm/shared/inspect_*.slurm scripts, which
# nbconvert --inplace and modify the notebook itself).
#
# Usage (from the repo root so logs/ resolves there):
#   mkdir -p logs
#   sbatch slurm/project_gfv2/render_gfv2.slurm
#   # Just one group:
#   GROUP=targets sbatch slurm/project_gfv2/render_gfv2.slurm

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
GROUP="${GROUP:-all}"
REPO_DIR="${REPO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
[ -n "$REPO_DIR" ] || {
    echo "ERROR: could not resolve REPO_DIR; set REPO_DIR=/path/to/repo and resubmit" >&2
    exit 1
}

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
