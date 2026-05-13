# nhf-spatial-targets — Claude Development Guide

## Project Purpose
Build curated calibration target datasets for the USGS National Hydrologic Model (NHM) by spatially aggregating gridded source datasets to an HRU fabric using gdptools. Targets documented in USGS TM 6-B10 (Hay et al., 2022): runoff, AET, recharge, soil moisture, snow-covered area.

## Environment & Commands

**All commands run via pixi (not pip/conda/python directly):**

```bash
# Create a project
pixi run init -- --project-dir /data/nhf-runs/my-run

# Edit config.yml to set fabric path, datastore, and credentials
# Then validate the project
pixi run validate -- --project-dir /data/nhf-runs/my-run

# Run the full pipeline against a project
pixi run run -- --project-dir /data/nhf-runs/my-run

# Run a single target (runoff is implemented; aet/rch/som/sca are stubs)
pixi run run-runoff -- --project-dir /data/nhf-runs/my-run
pixi run run-aet -- --project-dir /data/nhf-runs/my-run

# Aggregate sources to fabric (full source period; clipping happens in targets)
pixi run nhf-targets agg era5-land    --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg gldas        --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg merra2       --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg ncep-ncar    --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg nldas-mosaic --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg nldas-noah   --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg watergap22d  --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg reitz2017    --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg mod16a2      --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg mod10c1      --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg all          --project-dir /data/nhf-runs/my-run

# Catalog inspection
pixi run catalog-sources
pixi run catalog-variables

# Development (requires dev environment: pixi install -e dev)
pixi run -e dev test          # pytest (excludes integration tests)
pixi run -e dev test-integration  # integration tests (requires credentials/network)
pixi run -e dev lint          # ruff check src/ tests/
pixi run -e dev fmt           # ruff format src/ tests/
pixi run -e dev fmt-check     # ruff format --check src/ tests/
```

**Install environment:**
```bash
pixi install           # default env
pixi install -e dev    # dev env (adds pytest, ruff, mypy, ipykernel)
```

```bash
# Set up pre-commit hooks (once after cloning)
pixi run -e dev pre-commit install
```

**CLI entry point:** `nhf-targets` (defined in pyproject.toml `[project.scripts]`)

## Repository Layout

```
catalog/           # YAML data source registry and variable definitions
config/            # pipeline.yml reference configuration
src/nhf_spatial_targets/
  _logging.py      # Structured logging setup
  catalog.py       # Python API for catalog/ YAML files
  cli.py           # Cyclopts CLI: nhf-targets init | materialize-credentials | validate | run | fetch | catalog
  credentials.py   # materialize_cdsapirc / materialize_netrc_earthdata helpers
  defaults.py      # Default config schema and merge logic
  workspace.py     # Project path resolution, Project dataclass, make_dir()
  validate.py      # Preflight checks (fabric, datastore, credentials, catalog)
  init_run.py      # Project skeleton creation
  fetch/           # per-source download modules (one file per source, ~10 modules)
  aggregate/       # gdptools area-weighted aggregation (one file per source, ~15 modules)
  normalize/       # normalization and range-bound methods
  targets/         # per-variable target builders (run/aet/rch/som/sca)
tests/
```

## Code Conventions

- **Python >=3.11**, `from __future__ import annotations` in all modules
- **src layout** — package root is `src/nhf_spatial_targets/`
- **No implicit returns** from builder functions — raise `NotImplementedError` until implemented
- **Type hints** on all public functions
- **Docstrings** on public functions only (not stubs)
- **Ruff** for lint and format (line length 88); run before committing

## Data & Catalog Conventions

- All data source metadata lives in `catalog/sources.yml` — do not hardcode URLs or product names in Python modules
- All variable/range-method definitions live in `catalog/variables.yml`
- `catalog.py` is the single Python interface to both YAML files — do not read YAML directly elsewhere
- When adding a new source, add it to `catalog/sources.yml` first, then write the fetch module
- Mark superseded sources with `superseded_by:` key and `status: superseded`
- Fabric-restricted sources (e.g. Margulis WUS-SR for Oregon) carry an optional `fabric_scope: {fabrics: [<token>], notes: ...}` block. Allowed fabric tokens are enumerated by `catalog.FABRIC_SCOPE_TOKENS` and validated by `catalog.validate_fabric_scope`; adding a new fabric means extending that set and the matching check in target builders. Raw downloads remain reusable across projects sharing a datastore — `fabric_scope` is enforced at the target-build stage, not at fetch time
- **CF-1.6 compliance is required for every NetCDF the pipeline writes** — consolidated source NCs in `<datastore>/<source>/{daily,monthly}/`, aggregated NCs in `<project>/data/aggregated/`, and final target NCs in `<project>/targets/`. Use `fetch/consolidate.py:apply_cf_metadata` as the single entry point for setting `Conventions=CF-1.6`, variable `units` / `long_name` / `cell_methods` / `grid_mapping` from the catalog, coordinate `standard_name` / `units` / `axis`, and the WGS84 `crs` ancillary variable. Do not set these attrs by hand in source-specific code — read everything from `catalog/sources.yml` so a unit correction in the catalog flows through every NC on the next consolidate. Each fetch/consolidate module should have a test that asserts the output NC carries the required CF-1.6 attribute set.
- **Canonical row order on every fabric-aligned artifact is `id_col` ascending**, enforced at emission (issue #93). Aggregator (`aggregate/_driver.py`, `aggregate/ssebop.py`) sorts `year_ds` by `id_col` immediately before `_atomic_write_netcdf`; target writers call `write_target_nc(..., sort_dim=project.id_col)`. `validate` records `id_col_sorted: bool` on `fabric.json` / `manifest.json` and warns when the source `.gpkg` is not monotonic (it does not fail — the aggregator canonicalizes anyway). `read_aggregated_source` keeps a defensive `.sortby(id_col)` for pre-#93 NCs already on disk. Downstream code may rely on positional alignment without runtime checks; full reasoning is in [docs/architecture/transformation-pipeline.md](docs/architecture/transformation-pipeline.md#canonical-row-order-on-emission).

## Aggregation Transformation Policy

Where to put a transformation depends on the spatial scale at which it is
defined. Aggregation (gdptools area-weighted mean) is a one-way information
bottleneck — pixel-defined operations must run pre-aggregation, HRU-defined
operations must run post-aggregation, linear operations commute and live
downstream by convention.

Full architectural reference: `docs/architecture/transformation-pipeline.md`.

**Quick rules for new source / target / normalize modules:**

- **Pre-aggregation (`aggregate/<src>.py` `pre_aggregate_hook`):** flag-value
  masks, sums-of-accumulations needed to produce a single var to aggregate,
  per-pixel quality gates (e.g. CI > 70 in MOD10C1). These do not commute
  with area-weighted mean.
- **Post-aggregation cosmetic (`aggregate/<src>.py` `post_aggregate_hook`):**
  rename auxiliary diagnostic variables (e.g. `valid_mask` →
  `valid_area_fraction` after the per-pixel 0/1 mask becomes an HRU
  fraction), attach attrs. Do not modify aggregated source values.
- **Per-HRU transforms (`normalize/methods.py`):** 0–1 normalization,
  multi-source min/max, NN-fill of NaN HRUs (when applied at all, see below).
  Defined at HRU scale, must run post-aggregation. NN-fill, when applied,
  is a **target-stage post-processing step** on the multi-source-combined
  bounds (not on aggregated NCs); the target builder writes a parallel
  ``<target>_nn_filled.nc`` alongside the honest-NaN ``<target>.nc``. See
  ``targets/run.py`` for the canonical implementation.
- **Linear unit conversions (`targets/<tgt>.py`):** `× 1000`, `÷ 100`,
  `× 8 × days_in_month`, mm/month → cfs, etc. These commute with
  aggregation; we put them downstream so the aggregated NC stays in native
  units (easier to spot a missed conversion factor).
- **Multi-source combination (`targets/<tgt>.py`):** must be post-aggregation
  by definition (different sources have different grids). For
  `multi_source_minmax` targets use **NaN-aware** reduction
  (`np.fmin`/`np.fmax` or xarray `.min/max(skipna=True)` along a stacked
  source dim) so a bound is well-defined whenever ≥1 source is finite at the
  HRU/time. The bound is NaN only when *every* source is NaN there.

**`stat_method` choice: `mean` vs `masked_mean`.** gdptools' area-weighted
mean comes in two flavours, and the right choice depends on whether the
source has explicit per-pixel masking:

- **`stat_method="mean"` (default).** NaN propagates: any NaN pixel
  contributing to an HRU makes the HRU value NaN. Use this when source
  pixels arrive at the aggregator without per-pixel masking, so NaN means
  "no source data here" (geometric partial coverage, true upstream gaps).
- **`stat_method="masked_mean"`.** NaN pixels are skipped; the HRU value is
  the area-weighted mean of the *survivors*. Use this when the source
  **deliberately** masks pixels in `pre_aggregate_hook` (fill-value mask,
  quality gate, etc.). Without it, the per-pixel mask would poison every
  HRU that touches even one masked pixel — defeating the point of having a
  per-pixel mask in the first place. Currently used by
  `aggregate/mod16a2.py` (PR #88 fill mask) and `aggregate/mod10c1.py`
  (CI > 70 gate). Configurable via `SourceAdapter.stat_method`.

**The aggregated NC at `<project>/data/aggregated/<source_key>/...`
therefore carries the source's NATIVE variable names and NATIVE units**,
with flag-masked and quality-gated values. **HRU NaN values are honest**:
the aggregator never imputes. NN-fill, when desired, is a target-stage
concern — applied to the per-HRU per-time bound in `targets/<tgt>.py`
*after* multi-source combination, never to the aggregated NC itself. This
keeps the aggregated NCs as the canonical "what the source actually
covers" record while letting individual targets choose imputation policy.

Inspect notebooks in `notebooks/aggregated/` apply the same `÷ 100` /
`× 1000` / etc. conversions inline, mirroring `targets/`, so they should
produce order-of-magnitude-matching results when validated against gridded
means.

**Why ordering matters (the gotcha):** post-aggregation gating of a
per-pixel quality field gives a different answer than pre-aggregation
gating, because the area-weighted mean has already mixed high- and low-
confidence pixels. An HRU with 50% high-CI snowy and 50% low-CI cloud
pixels gives different results pre- vs post-gating. See the worked example
in `transformation-pipeline.md`.

## Projects & Datastore

The pipeline separates **projects** (fabric-specific) from the **datastore** (shared raw data, fabric-independent). Multiple projects can share one datastore — if data has already been fetched for one project, another project pointing to the same datastore will find and reuse it.

**Conceptual layout:**
```
/mnt/d/nhf-datastore/              # DATASTORE (shared, fabric-independent)
  ├── merra2/                      #   consolidated NCs from fetch
  ├── nldas_mosaic/
  ├── reitz2017/
  └── ...

/mnt/d/gfv11-spatial-targets/      # PROJECT (fabric-specific)
  ├── config.yml                   #   points to datastore + fabric
  ├── .credentials.yml
  ├── fabric.json
  ├── manifest.json
  ├── data/aggregated/             #   aggregated outputs
  ├── targets/                     #   final calibration targets
  ├── weights/                     #   weight caches (fabric × source grid)
  └── logs/

/mnt/d/gfv20-spatial-targets/      # ANOTHER PROJECT, same datastore
  ├── config.yml                   #   same datastore, different fabric
  └── ...
```

**Workflow:**
1. `nhf-targets init --project-dir <project-dir>` creates a project skeleton with `config.yml` template
2. User edits `config.yml` to set `fabric.path`, `fabric.id_col`, `datastore` path, and credentials; fills in `.credentials.yml` with NASA Earthdata and CDS credentials
3. `nhf-targets materialize-credentials --project-dir <project-dir>` copies credentials from `.credentials.yml` into `~/.cdsapirc` and `~/.netrc` (run after editing or rotating `.credentials.yml`)
4. `nhf-targets validate --project-dir <project-dir>` runs preflight checks and writes `fabric.json`
5. `nhf-targets fetch <source> --project-dir <project-dir>` downloads to the shared datastore
6. `nhf-targets agg ssebop --project-dir <project-dir>` aggregates remote data to fabric
7. `nhf-targets run --project-dir <project-dir>` builds calibration targets

**Key paths:**
- `<project>/config.yml` — project configuration (fabric, datastore, targets, dir_mode)
- `<project>/fabric.json` — computed fabric metadata (written by `validate`, required before `fetch`/`run`)
- `<project>/manifest.json` — provenance record; populated as the pipeline runs
- `<project>/.credentials.yml` — NASA Earthdata and Copernicus CDS credentials (gitignored, never commit)
- `<datastore>/<source_key>/` — shared raw downloads (fabric-independent, can be on a separate drive)
- `<project>/data/aggregated/` — spatially aggregated outputs (fabric-specific)
- `<project>/targets/` — final calibration target datasets
- `<project>/weights/` — gdptools weight caches (fabric × source grid, reusable)

**Notes:**
- Projects and datastore should be outside the repo (recommended for large datasets)
- The datastore path must be explicitly set in `config.yml` — it should not be the same as the project directory
- `dir_mode` in config.yml sets Unix directory permissions (e.g., "2775" for setgid + group-writable); ignored on Windows
- Never delete a project directory — it is the audit trail

## Relationship to TM 6-B10 (Hay et al. 2023)

`docs/references/tm6b10.pdf` (and `tm6b10.md`, a pymupdf4llm conversion) is the
methodological reference for the five calibration targets. A short crib sheet
keyed to this repo lives at `docs/references/tm6b10-summary.md`. This pipeline
intentionally differs from the report in two ways:

- **Time windows are not fixed to the report.** TM 6-B10 uses specific periods
  (e.g. 2000–2010 for AET, 2000–2009 for RCH normalization, 1982–2010 for SOM,
  2000–2010 for SCA). Here, target windows are driven by **available data or
  user preference** via project config, not hardcoded to the report's numbers.
  The specific window used for each target is recorded in the target output
  metadata and the project `manifest.json`. `period` fields in
  `catalog/variables.yml` reflect historical defaults, not hard constraints.
- **Dataset versions are current, not original.** Where the original sources
  have been decommissioned, retired, or superseded (e.g. MOD16A2 v006 → v061,
  MOD10C1 v006 → v061, MERRA-Land → MERRA-2, NHM-MWBM → ERA5-Land +
  GLDAS-2.1 NOAH, WaterGAP 2.2a → 2.2d) we use the modern replacement.
  `catalog/sources.yml` is authoritative for which version is in use; the
  "Known Gaps (Resolved)" block below documents each substitution.

## Known Gaps (do not implement until resolved)

See `catalog/sources.yml` `status:` and `notes:` fields for per-source gaps.
See `docs/references/known-gaps-resolved.md` for resolved items.

**Still open:**
- SCA CI-bounds formula — PRMSobjfun.f not publicly available; formula unconfirmed

## Testing

- Tests live in `tests/`
- `test_catalog.py` covers catalog load/lookup — keep passing at all times
- New fetch/aggregate/target modules get a corresponding `tests/test_<module>.py`
- Use `pytest.mark.integration` for tests that require network access or real data files

## Git Workflow

All work follows issue-branch-PR flow:

1. Create a GitHub issue describing the work
2. Branch from main: `<type>/<issue#>-short-description`
   - Types: `feature`, `fix`, `refactor`, `docs`, `test`, `chore`
   - Example: `feature/12-add-mwbm-fetch`, `fix/13-catalog-validation`
3. Develop on the branch, committing as needed
4. Open PR referencing the issue (e.g., "Closes #12")
5. CI must pass; squash merge after review

**Claude guardrails:**
- NEVER commit specs, plans, or code directly to main. Always create a feature branch first.
- Rebase feature branches against origin/main before opening PRs.
- Before every commit, run: `git branch --show-current` (must not be main/master), `git status` (review untracked), `git diff --cached --stat` (review staged). Stage files explicitly by path — never use `git add -A` or `git add .`.

## Pre-commit Quality Gate

Before suggesting a commit, always run:

```bash
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
```

Pre-commit hooks enforce this automatically, but Claude should run these proactively.

**Always commit via `pixi run git commit`, not bare `git commit`.** The pre-commit config runs ruff and pytest through `pixi run`; invoking `git commit` outside a pixi shell forces every hook to re-resolve the pixi environment, which is slow and prone to stalling on long-running hooks. A PreToolUse hook in `.claude/settings.json` blocks bare `git commit` for Claude sessions — humans should follow the same convention.

## Subagent Dispatch Rules
- Verify subagent prompts list ONLY the cells/files actually being changed - do not include unchanged items.
- NEVER allow subagents to force-push to PR branches; require explicit user approval before any force-push operation.
- After subagent completes, verify their changes match the spec before proceeding.

## Test Coverage Rule

Every new module in `fetch/`, `aggregate/`, `normalize/`, or `targets/` must have a corresponding `tests/test_<module>.py`. Every PR should maintain or improve test coverage.

### When to write the tests

Strict test-first TDD is **not** required for exploratory data work. For a new
or unfamiliar source, it is fine — and usually better — to characterize the
data first in a notebook, REPL, or scratch script: figure out the schema, units,
chunking, edge cases, and failure modes against real files before committing
to a test. Writing tests up front against a guessed shape tends to encode the
guess rather than the truth.

The hard line is the **PR boundary**, not the first line of code:

- Once an implementation lands in `src/`, the matching `tests/test_<module>.py`
  must exist before the PR is opened. "I'll add tests later" is not allowed to
  become "I shipped without tests."
- For bug fixes, invert the order: write the failing test first, then fix.
  This converts every bug into permanent regression coverage.
- Exploratory notebooks and scratch scripts are not a substitute for tests —
  they are scaffolding to be discarded or moved into `notebooks/` once the
  characterization work is done.

## Test Execution Discipline
- Do NOT use `pkill` to terminate pytest runs - let them complete or use Ctrl+C semantics.
- Batch test execution: run the full suite once at the end of a multi-step task rather than after every sub-step.
- For docs-only changes, skip running the full pytest suite when possible.
- When executing a multi-task plan, only run the full test suite at the end of each task (not between sub-steps within a task). Use targeted test selection for incremental verification.

## Dependencies

Managed by **pixi** (`pixi.toml`). Key packages:
- `gdptools` — spatial aggregation (PyPI)
- `earthaccess` — NASA EDL access for MODIS, MERRA-2, NLDAS
- `sciencebasepy` — USGS ScienceBase access
- `xarray`, `rioxarray`, `geopandas` — data handling
- `cyclopts`, `rich` — CLI

Do not add dependencies directly to `pyproject.toml` `[project.dependencies]` — add to `pixi.toml` instead (pyproject.toml is for build metadata only).
