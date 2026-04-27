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

# Run a single target
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
  catalog.py       # Python API for catalog/ YAML files
  cli.py           # Cyclopts CLI: nhf-targets init | materialize-credentials | validate | run | fetch | catalog
  credentials.py   # materialize_cdsapirc / materialize_netrc_earthdata helpers
  workspace.py     # Project path resolution, Project dataclass, make_dir()
  validate.py      # Preflight checks (fabric, datastore, credentials, catalog)
  init_run.py      # Project skeleton creation
  fetch/           # per-source download modules (one file per source)
  aggregate/       # gdptools area-weighted aggregation
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

**Resolved:**
- Reitz 2017 ScienceBase item ID — confirmed: `56c49126e4b0946c65219231`, doi:10.5066/F7PN93P0
- Runoff source replacement — NHM-MWBM removed; replaced by ERA5-Land (CDS) + GLDAS-2.1 NOAH monthly. ERA5-Land ssro also added as third recharge source. Closes issue #41.
- Recharge normalization window — confirmed **2000-2009** from TM 6-B10 body text
- MOD16A2 / MOD10C1 v006 → v061: both decommissioned; use v061 in all new runs
- MERRA-2 variable — use `GWETTOP` (0-0.05m, dimensionless); product M2TMNXLND
- MERRA-2 layer depths — dzsf=0.05m (constant globally), dzrz=1.00m (per GMAO FAQ), dzpr=spatially varying (surface to bedrock, ~1.3-8.5m). Thicknesses in M2CONXLND collection.
- NLDAS NOAH variable names — confirmed from file inspection: SoilM_0_10cm, SoilM_10_40cm, SoilM_40_100cm, SoilM_100_200cm
- WaterGAP 2.2d — confirmed: doi:10.1594/PANGAEA.918447, variable qrdif (diffuse groundwater recharge), 1901-2016 monthly, 0.5° global, CC BY-NC 4.0

**Still open:**
- SCA CI-bounds formula — PRMSobjfun.f not publicly available; formula unconfirmed

**Resolved (previously open):**
- SSEBop — accessed via USGS NHGF STAC catalog (collection `ssebopeta_monthly`, doi:10.5066/P9L2YMV, 2000–2023 monthly, 1km). Aggregated directly to HRU fabric via gdptools — no local download. See PR #34.

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

## Pre-commit Quality Gate

Before suggesting a commit, always run:

```bash
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
```

Pre-commit hooks enforce this automatically, but Claude should run these proactively.

**Always commit via `pixi run git commit`, not bare `git commit`.** The pre-commit config runs ruff and pytest through `pixi run`; invoking `git commit` outside a pixi shell forces every hook to re-resolve the pixi environment, which is slow and prone to stalling on long-running hooks. A PreToolUse hook in `.claude/settings.json` blocks bare `git commit` for Claude sessions — humans should follow the same convention.

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

## Dependencies

Managed by **pixi** (`pixi.toml`). Key packages:
- `gdptools` — spatial aggregation (PyPI)
- `earthaccess` — NASA EDL access for MODIS, MERRA-2, NLDAS
- `sciencebasepy` — USGS ScienceBase access
- `xarray`, `rioxarray`, `geopandas` — data handling
- `cyclopts`, `rich` — CLI

Do not add dependencies directly to `pyproject.toml` `[project.dependencies]` — add to `pixi.toml` instead (pyproject.toml is for build metadata only).
