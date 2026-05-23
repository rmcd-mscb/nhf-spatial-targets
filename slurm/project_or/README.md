# Oregon (`or-spatial-targets`) SLURM run order

Scripts for building the Oregon calibration targets. Submit **from the repo
root** (so `logs/` resolves there) after `nhf-targets validate` has written
`fabric.json` and the datastore is hydrated.

The aggregation array (`agg_all_or.slurm`) deliberately excludes SSEBop (remote
STAC) and Daymet (needs `--region`), so those are submitted separately.

```bash
cd <repo root>
mkdir -p logs
OR=/caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets

# 1. Aggregate the 14 array sources (incl. Margulis WUS-SR, OR-only SWE).
sbatch slurm/project_or/agg_all_or.slurm

# 2. Aggregate the two companion sources the array excludes.
PROJECT_DIR=$OR sbatch slurm/shared/agg_ssebop.slurm
PROJECT_DIR=$OR sbatch slurm/shared/agg_daymet.slurm   # edit --region (na) inside if needed

# 3. Build the 5 implemented targets (runoff/aet/rch/som/swe; sca self-skips).
#    Submit after the aggregations above have completed.
sbatch slurm/project_or/run_or.slurm

# 4. Render the per-project inspection figures (consolidated + aggregated +
#    targets) to docs/figures/.../or-spatial-targets/. Submit after the
#    targets above have built. Pass GROUP=targets for a faster subset.
sbatch slurm/project_or/render_or.slurm
```

Notes:

- `PROJECT_DIR` defaults to `or-spatial-targets` in the `project_or/` wrappers;
  the `shared/` scripts default to gfv2, so pass `PROJECT_DIR=$OR` to them.
- Margulis WUS-SR raw downloads live in the shared datastore and are fetched via
  `slurm/project_or/fetch_margulis_wus_sr.slurm` (OR-scoped, but the raw NCs are
  reusable across projects sharing the datastore).
- `REPO_DIR` auto-resolves from the submit directory; override it to run a
  worktree/branch checkout.
