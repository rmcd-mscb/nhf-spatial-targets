#!/bin/bash
#SBATCH --job-name=nhf-run
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/run_%a_%A.out
#SBATCH --error=logs/run_%a_%A.err

# NHF Spatial Targets — parallel target-build array.
# Submits one SLURM job per calibration target (5 total). Each target depends
# on its source aggregations being complete (run agg_all.slurm first); targets
# are otherwise independent and can run concurrently on separate compute nodes.
#
# Stub targets (AET, RCH, SOM, SCA) raise NotImplementedError under the hood;
# the CLI catches that and logs "skipping", so the array won't fail noisily on
# bulk submits. As each target's builder lands, the skip becomes a real build
# with no slurm-script change.
#
# Prerequisites:
#   - agg_all.slurm has completed for the sources each target needs
#   - $PROJECT_DIR/fabric.json exists (validate has run)
#
# Usage (REPO_DIR and PROJECT_DIR are overridable via environment):
#   mkdir -p logs                         # required — SLURM refuses to start
#                                         # with no log output if logs/ is missing
#   export PROJECT_DIR=/path/to/your/project
#   sbatch run_all.slurm
#
# Run a single target by index (e.g. runoff = 0):
#   sbatch --array=0 run_all.slurm
#
# Bump memory for an unusually large fabric:
#   sbatch --mem=64G run_all.slurm

set -euo pipefail

# Repo and project directories — override via environment, or edit these
# defaults for your site. The repo default mirrors the USGS Hovenweep path;
# $PROJECT_DIR must be a project directory created by `nhf-targets init`.
REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"

cd "$REPO_DIR" || { echo "ERROR: REPO_DIR=$REPO_DIR not found" >&2; exit 1; }

# Map array index -> pixi task name
RUN_TASKS=(
    "run-runoff"   # 0 — Runoff (3 sources: ERA5-Land, GLDAS, MWBM)
    "run-aet"      # 1 — AET (stub until targets/aet.py implemented)
    "run-rch"      # 2 — Recharge (stub)
    "run-som"      # 3 — Soil moisture (stub)
    "run-sca"      # 4 — SCA (stub)
)

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#RUN_TASKS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range [0, ${#RUN_TASKS[@]})" >&2
    exit 2
fi

TASK="${RUN_TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Array task $SLURM_ARRAY_TASK_ID: $TASK ==="
echo "=== Project: $PROJECT_DIR ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:    $(hostname) ==="

pixi run "$TASK" -- --project-dir "$PROJECT_DIR"

echo "=== Done:    $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
