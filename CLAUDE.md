# nhf-spatial-targets — Claude Development Guide

## Project Purpose
Build curated calibration target datasets for the USGS National Hydrologic Model (NHM) by spatially aggregating gridded source datasets to an HRU fabric using gdptools. Targets documented in USGS TM 6-B10 (Hay et al., 2022): runoff, AET, recharge, soil moisture, snow-covered area.

## Environment & Commands

**All commands run via pixi (not pip/conda/python directly):**

```bash
pixi run test          # pytest tests/ -v
pixi run lint          # ruff check src/ tests/
pixi run fmt           # ruff format src/ tests/
pixi run run           # full pipeline (requires fabric + data)
pixi run run-aet       # single target
pixi run catalog-sources
pixi run catalog-variables
```

**Install environment:**
```bash
pixi install           # default env
pixi install -e dev    # dev env (adds pytest, ruff, mypy, ipykernel)
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

## Data Provenance

- All downloaded source data is stored locally under `data/raw/<source_key>/`
- Each pipeline run is isolated under `runs/<run_id>/` (created by `nhf-targets init`)
- Never delete raw data; re-download only if explicitly requested

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

## Dependencies

Managed by **pixi** (`pixi.toml`). Key packages:
- `gdptools` — spatial aggregation (PyPI)
- `earthaccess` — NASA EDL access for MODIS, MERRA-2, NLDAS
- `sciencebasepy` — USGS ScienceBase access
- `xarray`, `rioxarray`, `geopandas` — data handling
- `click`, `rich` — CLI

Do not add dependencies directly to `pyproject.toml` `[project.dependencies]` — add to `pixi.toml` instead (pyproject.toml is for build metadata only).
