#!/bin/bash
#SBATCH --job-name=nhf-rechunk-tgt
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/rechunk_tgt_%j.out
#SBATCH --error=logs/rechunk_tgt_%j.err

# Convert a project's target NCs to the canonical columnar layout (#165 ST3b).
# The 11 GB swe_targets.nc + nn_filled written by the old streaming stitch carry
# time-slab chunks; shape-aware rechunk (PR #172) converts them so a per-HRU
# calibration read is 1 chunk instead of ~39. --mem=64G covers loading an 11 GB
# file whole (+ tmp write + read-back bit-identity verify). Idempotent +
# value-checked per file; safe to re-run (already-canonical targets are skipped).
#
# USAGE
#   mkdir -p logs
#   sbatch rechunk_gfv2_targets.slurm
#   # preview (no writes; fine on the login node):
#   pixi run nhf-targets rechunk --project-dir <gfv2> --layer target --dry-run

set -euo pipefail
export PYTHONUNBUFFERED=1

# REPO_DIR defaults to the repo this script lives in; override if desired.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"

cd "$REPO_DIR" || { echo "ERROR: REPO_DIR=$REPO_DIR not found" >&2; exit 1; }
[ -f src/nhf_spatial_targets/rechunk.py ] || {
    echo "ERROR: $REPO_DIR lacks the #165 rechunk command. Checkout main or" \
         "rebase your branch onto origin/main first." >&2
    exit 1
}

echo "=== rechunk targets: $PROJECT_DIR ==="
echo "=== Repo:    $REPO_DIR ($(git -C "$REPO_DIR" branch --show-current 2>/dev/null)) ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname) ==="

pixi run nhf-targets rechunk --project-dir "$PROJECT_DIR" --layer target

echo "=== Done:    $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
