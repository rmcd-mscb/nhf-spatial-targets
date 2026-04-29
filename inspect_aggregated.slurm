#!/bin/bash
#SBATCH --job-name=nhf-inspect
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/inspect_%a_%A.out
#SBATCH --error=logs/inspect_%a_%A.err

# NHF Spatial Targets — execute inspect_aggregated_* notebooks headlessly
# on a compute node so plot generation has the memory it needs (the login
# node is too constrained for CONUS-wide HRU choropleths over 10-year
# stacks).
#
# Each array task executes one notebook in-place via `jupyter nbconvert
# --execute --inplace`, which embeds rendered figures and dataframe
# outputs into the .ipynb file. After the job completes, open the
# notebook in JupyterLab/VSCode to view results — no kernel re-run
# required.
#
# Notebooks have hardcoded PROJECT_DIR pointing at gfv2-spatial-targets.
# To inspect a different project, edit the PROJECT_DIR cell in each .ipynb
# before submitting (the SLURM env var below is informational only — it
# does not override the in-notebook path).
#
# Prerequisites:
#   - Aggregated outputs already exist under $PROJECT_DIR/data/aggregated/
#     (see agg_all.slurm)
#   - Dev environment installed: `pixi install -e dev`
#
# Usage:
#   mkdir -p logs                         # required — SLURM refuses to
#                                         # start with no log output
#   sbatch inspect_aggregated.slurm
#
# Run a single notebook by index (e.g. AET = 0):
#   sbatch --array=0 inspect_aggregated.slurm
#
# Bump memory for a heavier rerun:
#   sbatch --mem=256G inspect_aggregated.slurm
#
# To also save figures to docs/figures/inspect_aggregated/, set:
#   SAVE_FIGURES=1 sbatch inspect_aggregated.slurm

set -euo pipefail

REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
SAVE_FIGURES="${SAVE_FIGURES:-0}"
TIMEOUT="${TIMEOUT:-3600}"  # per-cell execution timeout in seconds

cd "$REPO_DIR" || { echo "ERROR: REPO_DIR=$REPO_DIR not found" >&2; exit 1; }

# Map array index -> notebook filename (relative to notebooks/inspect_aggregated/)
NOTEBOOKS=(
    "inspect_aggregated_aet.ipynb"                # 0 — SSEBop + MOD16A2 v061
    "inspect_aggregated_recharge.ipynb"           # 1 — WaterGAP 2.2d + Reitz 2017 + ERA5-Land ssro
    "inspect_aggregated_runoff.ipynb"             # 2 — ERA5-Land + GLDAS-2.1 NOAH
    "inspect_aggregated_soil_moisture.ipynb"      # 3 — MERRA-2 + NCEP/NCAR + NLDAS Mosaic + NLDAS NOAH
    "inspect_aggregated_snow_covered_area.ipynb"  # 4 — MOD10C1 v061
)

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#NOTEBOOKS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range [0, ${#NOTEBOOKS[@]})" >&2
    exit 2
fi

NB="${NOTEBOOKS[$SLURM_ARRAY_TASK_ID]}"
NB_PATH="notebooks/inspect_aggregated/$NB"

echo "=== Array task $SLURM_ARRAY_TASK_ID: $NB ==="
echo "=== Project:       $PROJECT_DIR ==="
echo "=== Save figures:  $SAVE_FIGURES ==="
echo "=== Cell timeout:  ${TIMEOUT}s ==="
echo "=== Start:         $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:          $(hostname) ==="

# Optionally toggle _helpers.SAVE_FIGURES via a startup hook injected into
# the kernel. nbconvert honours PYTHONSTARTUP for the executing kernel, so
# we drop a one-liner there when SAVE_FIGURES=1.
STARTUP_FILE=""
if [[ "$SAVE_FIGURES" == "1" ]]; then
    STARTUP_FILE="$(mktemp --suffix=.py)"
    trap 'rm -f "$STARTUP_FILE"' EXIT
    cat > "$STARTUP_FILE" <<'EOF'
import sys
sys.path.insert(0, "notebooks/inspect_aggregated")
import _helpers
_helpers.SAVE_FIGURES = True
EOF
    export PYTHONSTARTUP="$STARTUP_FILE"
fi

# Force kernel_name=python3 to override each notebook's metadata.kernelspec
# (which varies — some are pinned to a user-registered "nhf-spatial-targets"
# kernel that won't exist on the compute node). The default `python3`
# kernelspec ships with ipykernel and resolves to the pixi dev env's Python
# because we're running under `pixi run -e dev`.
pixi run -e dev jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout="$TIMEOUT" \
    "$NB_PATH"

echo "=== Done:          $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
