# gfv2 (`gfv2-spatial-targets`) SLURM run order

Scripts for building the **gfv2 CONUS** calibration targets — the default
fabric. Submit **from the repo root** (so `logs/` resolves there) after
`nhf-targets validate` has written `fabric.json` and the datastore is hydrated.

On **caldera**, every submission must carry `--account=impd`: the `default`
account has `MaxSubmit=0` and rejects jobs with `AssocMaxSubmitJobLimit`. It is
already baked into each `#SBATCH` header here, so the bare `sbatch` lines below
work as written; if you copy a command elsewhere, keep `--account=impd`.

The aggregation array (`agg_all_gfv2.slurm`) deliberately excludes SSEBop
(remote STAC) and Daymet (needs `--region` + a staged zarr root), so those are
submitted separately.

```bash
cd <repo root>
mkdir -p logs
GFV2=/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets

# 1. Aggregate the 15 array sources (indices 0–14).
sbatch slurm/project_gfv2/agg_all_gfv2.slurm
#   MODIS reruns need more memory (indices 8–9):
#   sbatch --array=8-9 --mem=256G slurm/project_gfv2/agg_all_gfv2.slurm

# 2. Aggregate the two companion sources the array excludes.
PROJECT_DIR=$GFV2 sbatch slurm/shared/agg_ssebop.slurm
PROJECT_DIR=$GFV2 sbatch slurm/shared/agg_daymet.slurm   # edit --region (na) inside if needed

# 3. Build all six implemented targets (runoff/aet/rch/som/sca/swe).
#    Submit after the aggregations above have completed.
sbatch slurm/project_gfv2/run_gfv2.slurm
#   sbatch --array=0 slurm/project_gfv2/run_gfv2.slurm     # runoff only

# 4. Render the per-project inspection figures (consolidated + aggregated +
#    targets) to docs/figures/.../gfv2-spatial-targets/. Submit after the
#    targets above have built. Array indices: 0=consolidated, 1=aggregated,
#    2=targets — submit a subset to re-render one group without redoing
#    the others.
sbatch slurm/project_gfv2/render_gfv2.slurm                # all 3 groups
sbatch --array=2   slurm/project_gfv2/render_gfv2.slurm    # targets only
sbatch --array=1-2 slurm/project_gfv2/render_gfv2.slurm    # aggregated + targets
```

## Rechunk backfill (gfv2-only, optional)

These convert already-written NCs to the #165 chunked+compressed layout. They
are **maintenance**, not part of a fresh build, and exist only for gfv2 because
gfv2 carries the large pre-#165 artifacts. Both are idempotent, atomic, and
value-checked per variable — safe to re-run; already-canonical files are
skipped.

```bash
# Aggregated NCs — one array task per rechunkable source (13; daymet + ssebop
# excluded by design). Bump --mem if a daily source OOMs.
sbatch slurm/project_gfv2/rechunk_gfv2.slurm
sbatch --array=11 slurm/project_gfv2/rechunk_gfv2.slurm    # one source (snodas)

# Target NCs — single job; --mem=160G headroom for the ~11 GB (decompressed
# ~55 GB) SWE targets. Preview first on the login node:
pixi run nhf-targets rechunk --project-dir $GFV2 --layer target --dry-run
sbatch slurm/project_gfv2/rechunk_gfv2_targets.slurm
```

For OR there is no `rechunk_*` wrapper — OR's targets were written after #165,
so they are already canonical. If a future project needs the backfill, point
either gfv2 wrapper at it with `PROJECT_DIR=<that project> sbatch …` (the
rechunk command is fabric-independent).

## Notes

- `PROJECT_DIR` defaults to `gfv2-spatial-targets` in the `project_gfv2/`
  wrappers; the `shared/` scripts also default to gfv2, so they need no
  `PROJECT_DIR` override here (OR does — see `slurm/project_or/README.md`).
- `REPO_DIR` auto-resolves from the submit directory (`$SLURM_SUBMIT_DIR`, not
  `$BASH_SOURCE` — sbatch spool-copies the script, issue #174). Override it to
  run a worktree/branch checkout.
- Override the aggregation spatial batch size (default 10000 HRUs/batch, tuned
  for 128 GB) with `BATCH_SIZE=2500 sbatch slurm/project_gfv2/agg_all_gfv2.slurm`.
