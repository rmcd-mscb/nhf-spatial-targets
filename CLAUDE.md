# nhf-spatial-targets — Claude Development Guide

## Project Purpose
Build curated calibration target datasets for the USGS National Hydrologic Model (NHM) by spatially aggregating gridded source datasets to an HRU fabric using gdptools. Targets documented in USGS TM 6-B10 (Hay et al., 2022): runoff, AET, recharge, soil moisture, snow-covered area.

## Environment & Commands

**All commands run via pixi (not pip/conda/python directly):**

```bash
# Create a workspace
pixi run init -- --workdir /data/nhf-runs/my-run

# Edit config.yml to set fabric path, datastore, and credentials
# Then validate the workspace
pixi run validate -- --workdir /data/nhf-runs/my-run

# Run the full pipeline against a workspace
pixi run run -- --workdir /data/nhf-runs/my-run

# Run a single target
pixi run run-aet -- --workdir /data/nhf-runs/my-run

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
  cli.py           # Cyclopts CLI: nhf-targets init | validate | run | fetch | catalog
  workspace.py     # Workspace path resolution, Workspace dataclass, make_dir()
  validate.py      # Preflight checks (fabric, datastore, credentials, catalog)
  init_run.py      # Workspace skeleton creation
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

## Workspaces & Datastore

The project separates **workspaces** (fabric-dependent) from the **datastore** (shared raw data, fabric-independent).

**Workflow:**
1. `nhf-targets init --workdir <dir>` creates a workspace skeleton with `config.yml` template
2. User edits `config.yml` to set `fabric.path`, `fabric.id_col`, `datastore` path, and credentials
3. `nhf-targets validate --workdir <dir>` runs preflight checks and writes `fabric.json`
4. `nhf-targets fetch <source> --workdir <dir>` downloads to the shared datastore
5. `nhf-targets run --workdir <dir>` builds calibration targets

**Key paths:**
- `<workdir>/config.yml` — workspace configuration (fabric, datastore, targets, dir_mode)
- `<workdir>/fabric.json` — computed fabric metadata (written by `validate`, required before `fetch`/`run`)
- `<workdir>/manifest.json` — provenance record; populated as the pipeline runs
- `<workdir>/.credentials.yml` — NASA Earthdata credentials (gitignored, never commit)
- `<datastore>/<source_key>/` — shared raw downloads (fabric-independent, can be on a separate drive)
- `<workdir>/data/aggregated/` — spatially aggregated outputs (fabric-dependent)
- `<workdir>/targets/` — final calibration target datasets

**Notes:**
- `--workdir` and datastore can be outside the repo (recommended for large datasets)
- `dir_mode` in config.yml sets Unix directory permissions (e.g., "2775" for setgid + group-writable); ignored on Windows
- Never delete a workspace — it is the audit trail

## Known Gaps (do not implement until resolved)

See `catalog/sources.yml` `status:` and `notes:` fields for per-source gaps.

**Resolved:**
- MWBM ScienceBase item ID — confirmed: `55fc3f98e4b05d6c4e5029a1`, doi:10.5066/F7VD6WJQ
- Reitz 2017 ScienceBase item ID — confirmed: `56c49126e4b0946c65219231`, doi:10.5066/F7PN93P0
- Recharge normalization window — confirmed **2000-2009** from TM 6-B10 body text
- MOD16A2 / MOD10C1 v006 → v061: both decommissioned; use v061 in all new runs
- MERRA-2 variable — use `GWETTOP` (0-0.05m, dimensionless); product M2TMNXLND
- MERRA-2 layer depths — dzsf=0.05m (constant globally), dzrz=1.00m (per GMAO FAQ), dzpr=spatially varying (surface to bedrock, ~1.3-8.5m). Thicknesses in M2CONXLND collection.
- NLDAS NOAH variable names — confirmed from file inspection: SoilM_0_10cm, SoilM_10_40cm, SoilM_40_100cm, SoilM_100_200cm
- WaterGAP 2.2d — confirmed: doi:10.1594/PANGAEA.918447, variable qrdif (diffuse groundwater recharge), 1901-2016 monthly, 0.5° global, CC BY-NC 4.0

**Still open:**
- SCA CI-bounds formula — PRMSobjfun.f not publicly available; formula unconfirmed
- SSEBop — version and access URL used in original TM 6-B10 unconfirmed

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

## Test Coverage Rule

Every new module in `fetch/`, `aggregate/`, `normalize/`, or `targets/` must have a corresponding `tests/test_<module>.py`. Every PR should maintain or improve test coverage.

## Dependencies

Managed by **pixi** (`pixi.toml`). Key packages:
- `gdptools` — spatial aggregation (PyPI)
- `earthaccess` — NASA EDL access for MODIS, MERRA-2, NLDAS
- `sciencebasepy` — USGS ScienceBase access
- `xarray`, `rioxarray`, `geopandas` — data handling
- `cyclopts`, `rich` — CLI

Do not add dependencies directly to `pyproject.toml` `[project.dependencies]` — add to `pixi.toml` instead (pyproject.toml is for build metadata only).
