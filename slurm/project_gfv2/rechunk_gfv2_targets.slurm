#!/bin/bash
#SBATCH --job-name=nhf-rechunk-tgt
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=160G
#SBATCH --time=2:00:00
#SBATCH --output=logs/rechunk_tgt_%j.out
#SBATCH --error=logs/rechunk_tgt_%j.err

# Convert a project's target NCs to the canonical columnar layout (#165 ST3b).
# The 11 GB swe_targets.nc + nn_filled written by the old streaming stitch carry
# time-slab chunks; shape-aware rechunk (PR #172) converts them so a per-HRU
# calibration read is 1 chunk instead of ~39. Idempotent + value-checked per
# file; safe to re-run (already-canonical targets are skipped).
#
# --mem=160G: rechunk_file is EAGER — it `.load()`s the whole dataset, then the
# verify loop reads each var back from the tmp file to compare. The gfv2 SWE
# targets are ~11 GB on disk but ~55 GB DECOMPRESSED (16802 x 361471 float32
# bounds x2 + int8 n_sources), so peak RSS is ~load(55) + largest-var read-back
# (~24) + numpy temporaries ~= 100 GB. 64G OOM-killed it at the SWE step (job
# 21635275); 160G leaves headroom on the 512 GB cn nodes. A project whose
# largest target is smaller can drop this back down.
#
# USAGE
#   mkdir -p logs
#   sbatch rechunk_gfv2_targets.slurm
#   # preview (no writes; fine on the login node):
#   pixi run nhf-targets maintenance rechunk --project-dir <gfv2> --layer target --dry-run

set -euo pipefail
export PYTHONUNBUFFERED=1

# REPO_DIR defaults to the repo you ran `sbatch` from. sbatch copies the script
# to a spool dir before executing, so ${BASH_SOURCE[0]} does NOT point at the
# repo under SLURM — use $SLURM_SUBMIT_DIR (the submit cwd), falling back to
# $PWD when run directly. The 2>/dev/null||true lets the friendly cd/guard
# message below fire instead of a raw git error. Override REPO_DIR if desired.
REPO_DIR="${REPO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"

[ -n "$REPO_DIR" ] || { echo "ERROR: could not resolve REPO_DIR (not a git" \
    "checkout?). Set REPO_DIR=... explicitly." >&2; exit 1; }
cd "$REPO_DIR" || { echo "ERROR: REPO_DIR=$REPO_DIR not found" >&2; exit 1; }
[ -f src/nhf_spatial_targets/rechunk.py ] || {
    echo "ERROR: $REPO_DIR lacks the #165 rechunk command. Checkout main or" \
         "rebase your branch onto origin/main first." >&2
    exit 1
}

echo "=== rechunk targets: $PROJECT_DIR ==="
echo "=== Repo:    $REPO_DIR ($(git -C "$REPO_DIR" branch --show-current 2>/dev/null)) ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname) ==="

pixi run nhf-targets maintenance rechunk --project-dir "$PROJECT_DIR" --layer target

echo "=== Done:    $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
