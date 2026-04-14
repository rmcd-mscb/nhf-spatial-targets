#!/bin/bash
#SBATCH --job-name=nhf-agg-ssebop
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/agg_ssebop_%j.out
#SBATCH --error=logs/agg_ssebop_%j.err

# NHF Spatial Targets — SSEBop aggregation (remote STAC)
# SSEBop reads from the USGS NHGF STAC Zarr store rather than a local
# consolidated NC, so it's kept separate from agg_all.slurm. Single job,
# not an array.
#
# Prerequisites:
#   - Project is initialised and `nhf-targets validate` has produced
#     $PROJECT_DIR/fabric.json
#
# Usage:
#   mkdir -p logs                         # required — SLURM refuses to start
#                                         # with no log output if logs/ is missing
#   export PROJECT_DIR=/path/to/your/project
#   sbatch agg_ssebop.slurm
#
# Override the period (default is SSEBop catalog availability 2000/2023):
#   PERIOD=2010/2020 sbatch agg_ssebop.slurm
#
# Override spatial batch size (default 10000, tuned for 128 GB):
#   BATCH_SIZE=2500 sbatch agg_ssebop.slurm

set -euo pipefail

REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
PERIOD="${PERIOD:-2000/2023}"
BATCH_SIZE="${BATCH_SIZE:-10000}"

cd "$REPO_DIR" || { echo "ERROR: REPO_DIR=$REPO_DIR not found" >&2; exit 1; }

echo "=== agg-ssebop  period=$PERIOD  batch_size=$BATCH_SIZE ==="
echo "=== Project: $PROJECT_DIR ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:    $(hostname) ==="

pixi run agg-ssebop -- \
    --project-dir "$PROJECT_DIR" \
    --period "$PERIOD" \
    --batch-size "$BATCH_SIZE"

echo "=== Done:  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
