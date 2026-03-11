# nhf-spatial-targets — Claude Development Guide

## Project Purpose
Build curated calibration target datasets for the USGS National Hydrologic Model (NHM) by spatially aggregating gridded source datasets to an HRU fabric using gdptools. Targets documented in USGS TM 6-B10 (Hay et al., 2022): runoff, AET, recharge, soil moisture, snow-covered area.

## Environment & Commands

**All commands run via pixi (not pip/conda/python directly):**

```bash
# Create a run workspace (required before running the pipeline)
pixi run init -- --fabric /data/gfv1.1.gpkg --id gfv11 --workdir /data/nhf-runs

# Run the full pipeline against a workspace
pixi run run -- --run-dir /data/nhf-runs/2026-03-11T1500_v0.1.0

# Run a single target
pixi run run-aet -- --run-dir /data/nhf-runs/2026-03-11T1500_v0.1.0

# Catalog inspection
pixi run catalog-sources
pixi run catalog-variables

# Development
pixi run test          # pytest tests/ -v
pixi run lint          # ruff check src/ tests/
pixi run fmt           # ruff format src/ tests/
pixi run fmt-check     # ruff format --check src/ tests/
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
config/            # pipeline.yml run configuration
src/nhf_spatial_targets/
  catalog.py       # Python API for catalog/ YAML files
  cli.py           # Click CLI: nhf-targets run | catalog | init
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

## Data Provenance & Run Workspaces

- `nhf-targets init --fabric <gpkg> --workdir <dir>` creates a run workspace
- Each workspace is tied to a specific fabric (identified by sha256 of the GeoPackage)
- Run ID format: `YYYY-MM-DDTHHMM_<id>_v<version>` (e.g. `2026-03-11T1500_gfv11_v0.1.0`); `--id` is optional, omitting it gives `YYYY-MM-DDTHHMM_v<version>`
- Raw downloads live at `<run_dir>/data/raw/<source_key>/` — subsetted to fabric bbox + buffer
- If the same fabric is reused, `init` offers to symlink the prior run's raw data
- `--workdir` can be outside the repo (recommended for large datasets)
- `.credentials.yml` is always gitignored — never commit it
- `manifest.json` is the provenance record; populated as the pipeline runs
- Never delete a run directory — it is the audit trail

## Known Gaps (do not implement until resolved)

See `catalog/sources.yml` `status:` and `notes:` fields for per-source gaps:
- MWBM ScienceBase item ID — needs verification
- MOD16A2 / MOD10C1 v006 → v061 transition
- WaterGAP 2.2a access — registration-gated, may need substitute
- SCA CI-bounds formula — verify against PRMSobjfun.f
- Recharge normalization window — 2000-2009 vs 1990-1999 discrepancy

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
pixi run fmt && pixi run lint && pixi run test
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
- `click`, `rich` — CLI

Do not add dependencies directly to `pyproject.toml` `[project.dependencies]` — add to `pixi.toml` instead (pyproject.toml is for build metadata only).
