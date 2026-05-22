# Shared body for the per-fabric aggregation array wrappers
# (slurm/project_*/agg_all_*.slurm). NOT submitted directly — it is `source`d
# by a wrapper that has already declared the #SBATCH headers and set
# PROJECT_DIR + REPO_DIR. Keeping the task array + dispatch here means the two
# fabrics differ only in their wrapper's PROJECT_DIR default.
#
# Submits one SLURM array task per aggregation source (14 total). SSEBop is the
# remote-STAC aggregator and has its own script (slurm/shared/agg_ssebop.slurm).
# Daymet has its own script too (slurm/shared/agg_daymet.slurm) because it
# requires a --region argument plus an operator-staged zarr root.
#
# Override spatial batch size (default 10000 HRUs/batch, tuned for 128 GB):
#   BATCH_SIZE=2500 sbatch slurm/project_<fabric>/agg_all_<fabric>.slurm

# Flush Python stdout per-line so slurm logs (--output=...) update in real time
# instead of dumping batches when the 8 KB block buffer fills.
export PYTHONUNBUFFERED=1

BATCH_SIZE="${BATCH_SIZE:-10000}"

cd "$REPO_DIR" || {
    echo "ERROR: REPO_DIR=$REPO_DIR not found" >&2
    exit 1
}

# Map array index -> pixi task name
AGG_TASKS=(
    "agg-era5-land"     # 0 — ERA5-Land      (0.1°  monthly, runoff)
    "agg-gldas"         # 1 — GLDAS-2.1 NOAH (0.25° monthly, runoff)
    "agg-merra2"        # 2 — MERRA-2        (~0.5x0.625° monthly, soil wetness)
    "agg-ncep-ncar"     # 3 — NCEP/NCAR      (~1.875° monthly, soil moisture)
    "agg-nldas-mosaic"  # 4 — NLDAS-2 MOSAIC (0.125° monthly, soil moisture)
    "agg-nldas-noah"    # 5 — NLDAS-2 NOAH   (0.125° monthly, soil moisture)
    "agg-watergap22d"   # 6 — WaterGAP 2.2d  (0.5° monthly, recharge)
    "agg-reitz2017"     # 7 — Reitz 2017     (800m annual, recharge)
    "agg-mod16a2"       # 8 — MOD16A2 v061   (500m 8-day AET)  — memory-heavy
    "agg-mod10c1"       # 9 — MOD10C1 v061   (0.05° daily SCA) — memory-heavy
    "agg-snodas"        # 10 — SNODAS         (~1 km daily SWE, CONUS)
    "agg-mwbm-climgrid" # 11 — USGS MWBM (ClimGrid) (2.5 arcmin monthly, 4 vars)
    "agg-era5-land-sd"  # 12 — ERA5-Land sd   (0.1° daily snow depth, SWE source)
    "agg-margulis-wus-sr" # 13 — Margulis WUS-SR (90 m daily SWE, OR-only SWE source)
)

# Optional --period override per task. Empty string means "no clip"; the
# aggregator iterates every year present in the source files. Used by
# monolithic single-file sources (mwbm-climgrid: 1895-2020 in one NC) where the
# fetch period is provenance-only and can't subset bytes — clip at agg time.
AGG_PERIODS=(
    ""              # 0  era5-land
    ""              # 1  gldas
    ""              # 2  merra2
    ""              # 3  ncep-ncar
    ""              # 4  nldas-mosaic
    ""              # 5  nldas-noah
    ""              # 6  watergap22d
    ""              # 7  reitz2017
    ""              # 8  mod16a2
    ""              # 9  mod10c1
    ""              # 10 snodas
    "1979/2020"     # 11 mwbm-climgrid (publisher: 1895-2020; clip to NHM window, drop spinup)
    ""              # 12 era5-land-sd
    ""              # 13 margulis-wus-sr (1985-2020 on disk; clip via --period if desired)
)

if ((SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#AGG_TASKS[@]})); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range [0, ${#AGG_TASKS[@]})" >&2
    exit 2
fi

TASK="${AGG_TASKS[$SLURM_ARRAY_TASK_ID]}"
PERIOD="${AGG_PERIODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Array task $SLURM_ARRAY_TASK_ID: $TASK  batch_size=$BATCH_SIZE  period=${PERIOD:-<none>} ==="
echo "=== Project: $PROJECT_DIR ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:    $(hostname) ==="

if [ -n "$PERIOD" ]; then
    pixi run "$TASK" -- --project-dir "$PROJECT_DIR" --batch-size "$BATCH_SIZE" --period "$PERIOD"
else
    pixi run "$TASK" -- --project-dir "$PROJECT_DIR" --batch-size "$BATCH_SIZE"
fi

echo "=== Done:  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
