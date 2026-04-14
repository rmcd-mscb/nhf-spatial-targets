#!/bin/bash
#SBATCH --job-name=nhf-agg
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/agg_%a_%A.out
#SBATCH --error=logs/agg_%a_%A.err

# NHF Spatial Targets — parallel aggregation array (tier-1 + tier-2)
# Submits one SLURM job per aggregation source (9 total). SSEBop is the
# remote-STAC aggregator and has its own separate script (agg_ssebop.slurm).
# Each job aggregates independently; they can run concurrently on separate
# compute nodes.
#
# Prerequisites:
#   - Datastore already hydrated (see fetch_all.slurm)
#   - `nhf-targets validate` has produced $PROJECT_DIR/fabric.json
#
# Usage (REPO_DIR and PROJECT_DIR are overridable via environment):
#   mkdir -p logs
#   export PROJECT_DIR=/path/to/your/project
#   sbatch agg_all.slurm
#
# Run a single source by index (e.g. MOD10C1 = 8):
#   sbatch --array=8 agg_all.slurm
#
# Bump memory for a MODIS rerun:
#   sbatch --array=7-8 --mem=256G agg_all.slurm
#
# Override spatial batch size (default 10000 HRUs/batch, tuned for 128 GB):
#   BATCH_SIZE=2500 sbatch agg_all.slurm

set -euo pipefail

# Repo and project directories — override via environment, or edit these
# defaults for your site. The repo default mirrors the USGS Hovenweep path;
# $PROJECT_DIR must be a project directory created by `nhf-targets init`.
REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
BATCH_SIZE="${BATCH_SIZE:-10000}"

cd "$REPO_DIR"

# Map array index -> pixi task name
AGG_TASKS=(
    "agg-era5-land"     # 0 — ERA5-Land      (0.1°  monthly, runoff)
    "agg-gldas"         # 1 — GLDAS-2.1 NOAH (0.25° monthly, runoff)
    "agg-merra2"        # 2 — MERRA-2        (~0.5°  monthly, soil wetness)
    "agg-ncep-ncar"     # 3 — NCEP/NCAR      (~1.9° monthly, soil moisture)
    "agg-nldas-mosaic"  # 4 — NLDAS-2 MOSAIC (0.125° monthly, soil moisture)
    "agg-nldas-noah"    # 5 — NLDAS-2 NOAH   (0.125° monthly, soil moisture)
    "agg-watergap22d"   # 6 — WaterGAP 2.2d  (0.5° monthly, recharge)
    "agg-mod16a2"       # 7 — MOD16A2 v061   (500m 8-day AET)  — memory-heavy
    "agg-mod10c1"       # 8 — MOD10C1 v061   (0.05° daily SCA) — memory-heavy
)

TASK="${AGG_TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Array task $SLURM_ARRAY_TASK_ID: $TASK  batch_size=$BATCH_SIZE ==="
echo "=== Start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:  $(hostname) ==="

pixi run "$TASK" -- --project-dir "$PROJECT_DIR" --batch-size "$BATCH_SIZE"

echo "=== Done:  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
