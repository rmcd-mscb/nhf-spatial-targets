# Contributing to nhf-spatial-targets

## Prerequisites

- [pixi](https://pixi.sh) — manages Python environments and dependencies

## Setup

```bash
git clone <repo-url>
cd nhf-spatial-targets
pixi install -e dev
pixi run -e dev pre-commit install
```

## Development Workflow

1. **Create an issue** on GitHub describing the work.
2. **Create a branch** from `main`:
   ```bash
   git checkout -b <type>/<issue#>-short-description
   ```
   Types: `feature`, `fix`, `refactor`, `docs`, `test`, `chore`
3. **Develop** on the branch. Pre-commit hooks will automatically run formatting checks, linting, and unit tests on each commit.
4. **Open a pull request** referencing the issue (e.g., "Closes #12").
5. **CI must pass.** PRs are squash-merged after review.

## Running Checks Manually

```bash
pixi run -e dev fmt           # auto-format code
pixi run -e dev fmt-check     # check formatting without modifying
pixi run -e dev lint          # lint with ruff
pixi run -e dev test          # run full test suite
```

## Code Conventions

- Python >=3.11, `from __future__ import annotations` in all modules
- Type hints on all public functions
- Ruff for lint and format (line length 88)
- New modules in `fetch/`, `aggregate/`, `normalize/`, or `targets/` must have a corresponding `tests/test_<module>.py`

If you have Python experience but limited software-engineering background and want to understand the non-obvious patterns (`if TYPE_CHECKING:`, frozen dataclass adapters, atomic NetCDF writes, `flock`-guarded manifest writes, fingerprint-based cache invalidation), read [`docs/architecture/python-patterns.md`](docs/architecture/python-patterns.md). It explains why every module looks the way it does in ~7 short sections.

## Project conventions

Project-wide conventions — the pre-commit quality gate, transformation policy
(pre-aggregation / post-aggregation / per-HRU / target-stage), `stat_method`
choice between `mean` and `masked_mean`, manifest read-merge-write rule,
`fabric_scope` semantics, CF-1.6 NetCDF policy, canonical `id_col`-ascending
row order at emission, test-coverage rule, and git workflow — live in
[`CLAUDE.md`](CLAUDE.md). Despite the filename it applies equally to human
contributors; the AI assistant just happens to be the most disciplined reader.
Skim it before opening your first PR; refer back to the relevant section when
touching the matching subsystem.

## Data Sources

- All source metadata lives in `catalog/sources.yml` — do not hardcode URLs or product names
- When adding a new source, add it to `catalog/sources.yml` first, then write the fetch module

## Extending the Pipeline

Three common extensions and their file touch-points. Each is a numbered
checklist so a new contributor can self-verify the PR. Cross-references to
`CLAUDE.md` flag where the convention doc is the authoritative source.

### Adding a new source

A "source" is a gridded dataset that gets fetched, consolidated, and
spatially aggregated to the HRU fabric (e.g. ERA5-Land, GLDAS, MOD16A2).
Most new sources slot into the declarative `SourceAdapter` pattern; only
sources needing per-batch logic (STAC streaming, monthly + daily emission
from the same NCs) bypass the adapter and call the driver directly.

1. **Catalog entry** — add the source to [`catalog/sources.yml`](catalog/sources.yml)
   with `access`, `variables` (with `units` and `long_name`), `period`,
   `time_step`, and `status`. If the source is restricted to specific
   fabrics, add a `fabric_scope:` block (see `CLAUDE.md` §Data & Catalog
   Conventions). Catalog units are the single source of truth — never
   hardcode units in code.
2. **Fetch module** — write `src/nhf_spatial_targets/fetch/<source_key>.py`.
   Download to `<datastore>/<source_key>/`, then route consolidation
   through [`fetch/consolidate.py:apply_cf_metadata`](src/nhf_spatial_targets/fetch/consolidate.py)
   for CF-1.6 attrs (`Conventions=CF-1.6`, variable `units` /
   `long_name` / `cell_methods` / `grid_mapping`, coordinate
   `standard_name` / `units` / `axis`, WGS84 `crs` ancillary variable).
   Never call `ds.to_netcdf` directly — use `io_nc.build_encoding` +
   `io_nc.atomic_to_netcdf` (see `CLAUDE.md` §Data & Catalog Conventions).
3. **Aggregate module** — write `src/nhf_spatial_targets/aggregate/<source_key>.py`.
   For the common case, declare a `SourceAdapter` (see
   [`aggregate/_adapter.py`](src/nhf_spatial_targets/aggregate/_adapter.py))
   with `source_key`, `output_name`, `variables`, coord names, `source_crs`,
   `output_cadence`, and a `def aggregate_<source>(project, ...)`
   convenience wrapper. If the source needs pre-aggregation masking or a
   per-pixel quality gate (e.g. MOD16A2 fill-mask, MOD10C1 CI > 70), set
   `pre_aggregate_hook` and override `stat_method="masked_mean"` — see
   `CLAUDE.md` §Aggregation Transformation Policy for the `mean` vs
   `masked_mean` rule.
4. **CLI dispatch** — add imports and `fetch_app` / `agg_app` subcommand
   functions in [`src/nhf_spatial_targets/cli/`](src/nhf_spatial_targets/cli/)
   (`fetch.py` and `agg.py` after the #221 split). Follow the existing
   pattern (one `@fetch_app.command` and one `@agg_app.command` per source);
   the `fetch all` and `agg all` Python drivers pick the new source up
   automatically once the function is registered. **Note:** SLURM array
   indices still need to be added manually (next step).
5. **Pixi tasks** — add `fetch-<source>` and `agg-<source>` rows to the
   `[tasks]` block in [`pixi.toml`](pixi.toml), mirroring the existing
   entries.
6. **SLURM array indexes** — extend the `FETCH_TASKS` array in
   [`slurm/shared/fetch_all.slurm`](slurm/shared/fetch_all.slurm) and the
   `AGG_TASKS` / `AGG_PERIODS` arrays in
   [`slurm/shared/agg_all.body.sh`](slurm/shared/agg_all.body.sh). Add a
   new index at the end so existing indexes stay stable for operators with
   running jobs.
7. **Operator notes** — add `docs/sources/<source_key>.md` with access
   prerequisites, known gaps, and any quirks (see existing files in
   [`docs/sources/`](docs/sources/) for the format).
8. **Tests** — `tests/test_fetch_<source>.py` plus
   `tests/test_aggregate_<source>.py`. At minimum: CF-1.6 attrs on the
   consolidated NC and a synthetic-fabric end-to-end aggregation against
   a fixture grid. Exploratory characterization is fine in a notebook or
   REPL first, but the test files must exist before the PR opens (see
   `CLAUDE.md` §Test Coverage Rule).
9. **README updates** — flip the matching row to **Done** in §Implementation
   Status; add a row to §Fetch & Consolidation Pipeline; add a row to
   §Calibration Targets if the new source feeds a target.

### Adding a new target

A "target" is a calibration variable (runoff, AET, recharge, SOM, SCA, SWE).
As of PR #219, targets are **declarative**: each `targets/<target>.py` module
declares a `TargetAdapter` instance and a thin `build(project)` that
delegates to the generic driver. The pattern mirrors
`aggregate/_adapter.py:SourceAdapter` on the aggregator side.

The six existing adapters live in `targets/{run,aet,rch,som,sca,swe}.py`:
[`run.py`](src/nhf_spatial_targets/targets/run.py) is the canonical
multi-source single-shot example; [`sca.py`](src/nhf_spatial_targets/targets/sca.py)
is the canonical year-chunked example; [`som.py`](src/nhf_spatial_targets/targets/som.py)
shows the multi-variant pattern (monthly + annual emit).

1. **Variable definition** — add the target's variable entry (units,
   `range_method`, normalization parameters) to
   [`catalog/variables.yml`](catalog/variables.yml).
2. **Declare per-source shims** — a tuple of `SourceShim` instances in
   your new module (see [`targets/_shims.py`](src/nhf_spatial_targets/targets/_shims.py)).
   Each shim names the on-disk source key, the aggregated variable to read,
   a human label for the output NC's `source` attr, the unit-shim callable,
   and (optionally) the `expected_cf_units` string so `validate_source_units`
   catches catalog drift.
3. **Write the source loader** — a callable that returns a
   `SourceLoaderResult` (bounds + `n_sources` + time index + offset +
   per-target `extra_attrs`). The loader signature receives the project,
   adapter, period, fabric metadata, and (for year-chunked targets) a
   `year_context` tuple. Unit shims, per-source normalization, and
   multi-source combination live in the loader.
4. **Declare the adapter** as a module-level `ADAPTER = TargetAdapter(...)`
   (see [`targets/_adapter.py`](src/nhf_spatial_targets/targets/_adapter.py)
   for the full field list). Set `cadence`, `bounds_units`,
   `bounds_long_name_kind`, `cell_methods`, `title`, `nn_title`, and the
   loader callable. For year-chunked daily targets (SCA, SWE) set
   `year_chunked=True` plus `intermediates_subdir` / `intermediate_base`.
   For targets that need per-HRU area (mm/month → cfs), set
   `needs_hru_area=True`.
5. **Wire `build(project)`** as a one-line delegate:
   ```python
   def build(project):
       from nhf_spatial_targets.targets._driver import build as run_driver
       run_driver(ADAPTER, project)
   ```
6. **CLI dispatch** — wire the new builder into the `_dispatch` table in
   [`src/nhf_spatial_targets/cli/run.py`](src/nhf_spatial_targets/cli/run.py)
   (import the module, add a `"<target_name>": <module>.build` entry).
7. **Pixi task** — add `run-<target> = { cmd = "nhf-targets run --target <target>" }`
   to [`pixi.toml`](pixi.toml).
8. **SLURM array index** — append the new task to the `RUN_TASKS` array in
   [`slurm/shared/run_all.body.sh`](slurm/shared/run_all.body.sh). Add at
   the end so existing array indexes remain stable.
9. **Default config** — if the target needs new config keys (sources list,
   period, normalization params, `nn_fill` flag), update
   [`src/nhf_spatial_targets/defaults.py`](src/nhf_spatial_targets/defaults.py),
   [`src/nhf_spatial_targets/init_run.py:_CONFIG_TEMPLATE`](src/nhf_spatial_targets/init_run.py)
   (the single source of truth for the config schema),
   and `OPTIONAL_CONFIG_FEATURES` in
   [`src/nhf_spatial_targets/upgrade_config.py`](src/nhf_spatial_targets/upgrade_config.py)
   so existing-project operators find the new keys via
   `nhf-targets maintenance check-config` (see `CLAUDE.md` §Config schema additions).
10. **Inspect notebook** — copy
    [`notebooks/targets/inspect_target_swe.ipynb`](notebooks/targets/inspect_target_swe.ipynb)
    to `notebooks/targets/inspect_target_<target>.ipynb` and retarget the
    load + plotting cells. Visual inspection has caught real unit / variable
    bugs that tests missed.
11. **Tests** — `tests/test_targets_<target>.py` with synthetic-fabric
    fixtures. See `tests/test_targets_run.py` (single-shot) and
    `tests/test_targets_sca.py` (year-chunked) for shape templates.
12. **README updates** — flip the matching row to **Done** in §Implementation
    Status and refresh the §Calibration Targets row.

### Adding a new fabric

A "fabric" is an HRU polygon set (e.g. GFv1.1, GFv2.0, or a custom
delineation). Adding one rarely requires code changes — the same
fabric-independent pipeline runs against a new `config.yml`.

For the operator-facing recipe (create a new project directory pointing at
the same datastore, edit `fabric.path` / `fabric.id_col`, skip `fetch`,
re-run `agg` and `run`), see [README.md §Using a different
fabric](README.md#using-a-different-fabric). It is the authoritative recipe;
this section exists to flag the **code-level touch-points** that occasionally
come up.

1. **Per-fabric SLURM wrappers** — if you want the new fabric to have its
   own `fetch_all` / `agg_all` / `run_all` wrappers analogous to
   [`slurm/project_gfv2/`](slurm/project_gfv2/) and
   [`slurm/project_or/`](slurm/project_or/), copy one of those directories
   to `slurm/project_<fabric>/` and update the `PROJECT_DIR` default at the
   top of each wrapper. The fabric-independent body scripts in
   `slurm/shared/` need no changes.
2. **Fabric-scoped sources** — if the new fabric should be allowed to use a
   fabric-restricted source (currently Margulis WUS-SR is OR-only), extend
   the `FABRIC_SCOPE_TOKENS` set and the matching `validate_fabric_scope`
   check in [`src/nhf_spatial_targets/catalog.py`](src/nhf_spatial_targets/catalog.py),
   then add the new token to the source's `fabric_scope.fabrics` list in
   `catalog/sources.yml`. See `CLAUDE.md` §Data & Catalog Conventions.
3. **No source-code changes for routine fabric swaps** — the aggregator
   recomputes weight caches automatically against the new fabric geometry.
   The fabric file just needs a polygon geometry column and a unique
   integer HRU ID column (GeoPackage, GeoParquet, or Shapefile).

## Keeping documentation current

When implementing a new target builder or source, update these documentation surfaces in the same PR:

- **`README.md` §Implementation Status** — flip the matching row to **Done** with the PR reference
- **`README.md` §Calibration Targets** — refresh the per-target row if sources, period, or method changes
- **`README.md` §Fetch & Consolidation Pipeline** — refresh the per-source row if a new source lands
- **`docs/sources/<source_key>.md`** — operator notes for new sources (see #220)

A stale README is the highest-impact trust signal for new operators. CI does not catch documentation drift; reviewers should.
