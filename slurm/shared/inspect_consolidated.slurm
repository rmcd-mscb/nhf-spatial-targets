#!/bin/bash
#SBATCH --job-name=nhf-inspect-consolidated
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --output=logs/inspect_consolidated_%a_%A.out
#SBATCH --error=logs/inspect_consolidated_%a_%A.err

# NHF Spatial Targets — execute inspect_consolidated_* notebooks headlessly
# on a compute node so plot generation has the memory it needs (CONUS-wide
# gridded comparisons across 2-4 sources blow past login-node limits).
#
# Each array task executes one notebook in-place via `jupyter nbconvert
# --execute --inplace`, which embeds rendered figures and dataframe
# outputs into the .ipynb file. With SAVE_FIGURES=1 the notebooks ALSO
# write each panel to docs/figures/consolidated/<project>/<name>.png
# (the script always passes a <project> tag derived from $PROJECT_DIR).
#
# Memory note:
#   192G is the default because the NLDAS+MERRA-2+NCEP+NCAR soil-moisture
#   gridded comparison and the ERA5-Land/GLDAS runoff comparison hold
#   multiple full-CONUS rasters in RAM simultaneously. AET and recharge
#   fit comfortably below this; SCA needs the least. Bump with `--mem=...`
#   if a heavier rerun is needed.
#
# Notebooks have hardcoded DATASTORE / PROJECT paths pointing at
# nhf-datastore + gfv2-spatial-targets on caldera. Edit them in-place
# before submitting if pointing at a different project.
#
# Prerequisites:
#   - Consolidated source NCs already present in the datastore (fetch
#     stage complete)
#   - Dev environment installed: `pixi install -e dev`
#
# Usage:
#   mkdir -p logs                         # required — SLURM refuses to
#                                         # start with no log output
#   sbatch inspect_consolidated.slurm
#
# Run a single notebook by index (e.g. AET = 0):
#   sbatch --array=0 inspect_consolidated.slurm
#
# Bump memory:
#   sbatch --mem=256G inspect_consolidated.slurm
#
# To save figures to docs/figures/consolidated/<project>/, set:
#   SAVE_FIGURES=1 sbatch inspect_consolidated.slurm

set -euo pipefail

# Flush Python stdout per-line so slurm logs (--output=...) update in
# real time instead of dumping batches when the 8 KB block buffer fills.
# Slow workloads can otherwise look frozen for tens of minutes.
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
SAVE_FIGURES="${SAVE_FIGURES:-0}"
TIMEOUT="${TIMEOUT:-3600}"  # per-cell execution timeout in seconds

cd "$REPO_DIR" || { echo "ERROR: REPO_DIR=$REPO_DIR not found" >&2; exit 1; }

# Map array index -> notebook filename (relative to notebooks/consolidated/)
NOTEBOOKS=(
    "inspect_consolidated_aet.ipynb"                # 0 — SSEBop + MWBM + MOD16A2 v061
    "inspect_consolidated_recharge.ipynb"           # 1 — Reitz 2017 + WaterGAP 2.2d + ERA5-Land ssro
    "inspect_consolidated_runoff.ipynb"             # 2 — ERA5-Land + GLDAS-2.1 NOAH
    "inspect_consolidated_soil_moisture.ipynb"      # 3 — MERRA-2 + NLDAS NOAH + NLDAS Mosaic + NCEP/NCAR
    "inspect_consolidated_snow_covered_area.ipynb"  # 4 — MOD10C1 v061
)

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#NOTEBOOKS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range [0, ${#NOTEBOOKS[@]})" >&2
    exit 2
fi

NB="${NOTEBOOKS[$SLURM_ARRAY_TASK_ID]}"
NB_PATH="notebooks/consolidated/$NB"

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
    cat > "$STARTUP_FILE" <<EOF
import sys
sys.path.insert(0, "notebooks/consolidated")
import _helpers
_helpers.SAVE_FIGURES = True
_helpers.PROJECT = "$(basename "$PROJECT_DIR")"
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
