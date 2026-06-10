# Shared body for the per-fabric target-build array wrappers
# (slurm/project_*/run_*.slurm). NOT submitted directly — it is `source`d by a
# wrapper that has already declared the #SBATCH headers and set PROJECT_DIR +
# REPO_DIR.
#
# Submits one SLURM array task per calibration target (6 total). Each target
# depends on its source aggregations being complete (run the agg array first);
# targets are otherwise independent and run concurrently on separate nodes.
#
# All six targets are implemented (runoff/aet/rch/som/sca/swe). SCA is the
# two-source NaN-aware bound from MOD10C1 (calcSCA CI-interval) + UA SWE
# (depth-thresholded snow_covered_fraction), #237. Each target requires its
# configured sources to be aggregated FIRST: a target whose sources are not yet
# on disk fails its array task with FileNotFoundError (exit 1) — it does NOT
# self-skip. Run the agg array to completion before this one. (The CLI only
# logs "skipping" for a builder that raises NotImplementedError; no target does
# anymore, so in practice nothing self-skips.)

# Flush Python stdout per-line so slurm logs update in real time.
export PYTHONUNBUFFERED=1

cd "$REPO_DIR" || {
    echo "ERROR: REPO_DIR=$REPO_DIR not found" >&2
    exit 1
}

# Map array index -> pixi task name
RUN_TASKS=(
    "run-runoff"   # 0 — Runoff (3 sources: ERA5-Land, GLDAS, MWBM)
    "run-aet"      # 1 — AET (MOD16A2, SSEBop[, MWBM])
    "run-rch"      # 2 — Recharge (Reitz 2017, ERA5-Land[, WaterGAP])
    "run-som"      # 3 — Soil moisture (MERRA-2, NLDAS-MOSAIC, NLDAS-NOAH)
    "run-sca"      # 4 — SCA (MOD10C1 CI-bounds + UA SWE depth-threshold, #237)
    "run-swe"      # 5 — SWE (Daymet, SNODAS, ERA5-Land[, Margulis on OR])
)

if ((SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#RUN_TASKS[@]})); then
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
