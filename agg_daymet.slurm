#!/bin/bash
#SBATCH --job-name=nhf-agg-daymet
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/agg_daymet_%j.out
#SBATCH --error=logs/agg_daymet_%j.err

# NHF Spatial Targets — Daymet aggregation (local zarr, NA region)
# Daymet is distributed as a single multi-year zarr store per region
# at $daymet_root/daymet_{na,hi,pr}.zarr, recorded in manifest.json by
# `nhf-targets fetch daymet`. Like SSEBop, the aggregator requires an
# explicit --period, so it's kept separate from agg_all.slurm. Single
# job, not an array.
#
# Prerequisites:
#   - Project is initialised and `nhf-targets validate` has produced
#     $PROJECT_DIR/fabric.json
#   - `nhf-targets fetch daymet` has been run and the manifest carries
#     a `sources.daymet.regions.<region>.path` entry pointing at the
#     staged zarr.
#
# Usage:
#   mkdir -p logs                         # required — SLURM refuses to start
#                                         # with no log output if logs/ is missing
#   export PROJECT_DIR=/path/to/your/project
#   sbatch agg_daymet.slurm
#
# Override the period (default is Daymet catalog availability 1980/2024):
#   PERIOD=2010/2020 sbatch agg_daymet.slurm
#
# Override the region (default NA; HI and PR are not wired in this
# build and will surface NotImplementedError):
#   REGION=na sbatch agg_daymet.slurm
#
# Override spatial batch size (default 10000, tuned for 128 GB):
#   BATCH_SIZE=2500 sbatch agg_daymet.slurm

set -euo pipefail

REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
PERIOD="${PERIOD:-1980/2024}"
REGION="${REGION:-na}"
BATCH_SIZE="${BATCH_SIZE:-10000}"

cd "$REPO_DIR" || { echo "ERROR: REPO_DIR=$REPO_DIR not found" >&2; exit 1; }

echo "=== agg-daymet  region=$REGION  period=$PERIOD  batch_size=$BATCH_SIZE ==="
echo "=== Project: $PROJECT_DIR ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:    $(hostname) ==="

pixi run agg-daymet -- \
    --project-dir "$PROJECT_DIR" \
    --period "$PERIOD" \
    --region "$REGION" \
    --batch-size "$BATCH_SIZE"

echo "=== Done:  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
