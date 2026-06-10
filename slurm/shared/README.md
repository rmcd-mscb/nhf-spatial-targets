# `slurm/shared/` — fabric-independent SLURM scripts

This directory holds the SLURM pieces that do **not** depend on which fabric you
are building. Two kinds of file live here:

1. **Standalone `.slurm` scripts** — submitted directly with `sbatch`, but they
   default to the gfv2 project, so pass `PROJECT_DIR=$OTHER` to point them at
   another fabric (e.g. `PROJECT_DIR=$OR sbatch slurm/shared/agg_ssebop.slurm`).
   These cover the aggregators and fetchers that the per-fabric `agg_all_*`
   array deliberately excludes (SSEBop's remote STAC, Daymet's `--region` +
   staged zarr) plus the single-source fetchers.

2. **`*.body.sh` shared bodies** — **NOT submitted directly.** Each is `source`d
   by a thin per-fabric wrapper (`slurm/project_<fabric>/<verb>_<fabric>.slurm`)
   that has already declared the `#SBATCH` headers and exported `PROJECT_DIR` +
   `REPO_DIR`. The body holds the array task-index → pixi-task mapping and the
   dispatch. **Source, not submit** — running `sbatch slurm/shared/agg_all.body.sh`
   is a mistake (no headers, no `PROJECT_DIR`).

## Why the split

The two fabrics (gfv2 CONUS, OR) differ only in their wrapper's `PROJECT_DIR`
default. Keeping the task array + dispatch in one `*.body.sh` means a change to
the source list or array layout is made **once**, and both fabrics inherit it —
the wrappers in `slurm/project_gfv2/` and `slurm/project_or/` stay trivial.

```
slurm/
  shared/
    agg_all.body.sh      # 15-source aggregation array (sourced)
    run_all.body.sh      # 6-target build array (sourced)
    render_all.body.sh   # 3-group figure-render array (sourced)
    agg_ssebop.slurm     # standalone: remote-STAC aggregator
    agg_daymet.slurm     # standalone: needs --region + staged zarr
    fetch_*.slurm        # standalone single-source fetchers
  project_gfv2/          # gfv2 wrappers (PROJECT_DIR -> gfv2-spatial-targets)
  project_or/            # OR wrappers (PROJECT_DIR -> or-spatial-targets)
```

## Conventions shared by every script here

- **caldera account:** all submissions need `--account=impd` (the `default`
  account rejects jobs with `AssocMaxSubmitJobLimit`). Wrappers bake it into
  their `#SBATCH` headers; keep it when copying a command.
- **Submit from the repo root** so `logs/` resolves there (`mkdir -p logs`
  first).
- **`REPO_DIR`** auto-resolves from `$SLURM_SUBMIT_DIR` (the submit cwd), not
  `$BASH_SOURCE` — sbatch spool-copies the script before running it (issue
  #174). Override `REPO_DIR` to run a worktree/branch checkout.
- **`PROJECT_DIR`** defaults to gfv2 in `shared/` scripts; pass it explicitly
  for any other fabric.

## Per-fabric submit order

The submit-order walkthroughs live with the wrappers, not here:

- gfv2 (default CONUS): [`slurm/project_gfv2/README.md`](../project_gfv2/README.md)
- Oregon: [`slurm/project_or/README.md`](../project_or/README.md)
