# Shared body for the per-fabric figure-render array wrappers
# (slurm/project_*/render_*.slurm). NOT submitted directly — it is `source`d by
# a wrapper that has already declared the #SBATCH headers and set PROJECT_DIR +
# REPO_DIR.
#
# Submits one SLURM array task per inspection-notebook group (3 total):
#   0 — consolidated  (datastore NCs; CONUS-gridded, memory ceiling)
#   1 — aggregated    (per-source HRU NCs)
#   2 — targets       (per-target HRU NCs)
#
# Groups are independent — re-render one without re-running the others via:
#   sbatch --array=2 slurm/project_<fabric>/render_<fabric>.slurm   # targets only
#   sbatch --array=1-2 slurm/project_<fabric>/render_<fabric>.slurm # aggregated + targets

export PYTHONUNBUFFERED=1

cd "$REPO_DIR" || {
    echo "ERROR: REPO_DIR=$REPO_DIR not found" >&2
    exit 1
}

# Map array index -> render-figures group name. Order chosen so the heaviest
# group (consolidated, CONUS-gridded source data) runs on index 0 — a "submit
# just the first one" smoke test exercises the memory ceiling first.
RENDER_GROUPS=(
    "consolidated"   # 0 — datastore NC inspection (gridded pre-aggregation)
    "aggregated"     # 1 — per-source aggregated HRU NC inspection
    "targets"        # 2 — per-target HRU NC inspection
)

if ((SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#RENDER_GROUPS[@]})); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range [0, ${#RENDER_GROUPS[@]})" >&2
    exit 2
fi

GROUP="${RENDER_GROUPS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Array task $SLURM_ARRAY_TASK_ID: render-figures --group $GROUP ==="
echo "=== Project: $PROJECT_DIR ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:    $(hostname) ==="

pixi run -e dev render-figures -- --group "$GROUP" --project-dir "$PROJECT_DIR"

echo "=== Done:    $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
