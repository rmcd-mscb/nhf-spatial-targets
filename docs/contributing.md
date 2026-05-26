# Contributing

Quick orientation for new contributors. For the comprehensive guide (setup, code conventions, "Extending the Pipeline" recipes with full file-by-file checklists), see the [`CONTRIBUTING.md` at the repo root](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/CONTRIBUTING.md).

## Setup

```bash
git clone git@github.com:rmcd-mscb/nhf-spatial-targets.git
cd nhf-spatial-targets
pixi install -e dev
pixi run -e dev pre-commit install
```

## Day-to-day commands

```bash
pixi run -e dev fmt           # auto-format with ruff
pixi run -e dev lint          # lint with ruff
pixi run -e dev test          # unit tests (excludes integration)
pixi run docs-serve           # docs site at http://127.0.0.1:8000
```

## Pre-commit gate

Before committing, the pre-commit hook runs `ruff fmt`, `ruff lint`, and the pytest suite via `pixi run`. **Always commit via `pixi run git commit`** (not bare `git commit`) — a PreToolUse hook enforces this for Claude sessions; the convention applies equally to human contributors.

## Workflow

1. Create a GitHub issue describing the work
2. Branch from `main`: `<type>/<issue#>-short-description` (types: `feature`, `fix`, `refactor`, `docs`, `test`, `chore`)
3. Develop, committing as needed
4. Open a PR referencing the issue
5. CI must pass; squash-merge after review

## Extending the pipeline

Three common extensions have file-by-file checklists in the [full CONTRIBUTING.md](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/CONTRIBUTING.md#extending-the-pipeline):

### Adding a new source

A gridded dataset that gets fetched, consolidated, and spatially aggregated to the HRU fabric (e.g. ERA5-Land, GLDAS, MOD16A2). The declarative [`SourceAdapter`](api/aggregate-adapter.md) pattern handles most new sources; only those needing per-batch logic (STAC streaming, monthly + daily emission from the same NCs) bypass the adapter.

Key touch-points: `catalog/sources.yml`, `fetch/<key>.py`, `aggregate/<key>.py`, pixi tasks, SLURM array indexes, [`docs/sources/<key>.md`](sources/index.md), tests.

### Adding a new target

A calibration variable (runoff, AET, recharge, SOM, SCA, SWE). As of [issue #219](https://github.com/rmcd-mscb/nhf-spatial-targets/issues/219), targets are **declarative**: each `targets/<target>.py` module declares a [`TargetAdapter`](api/targets-adapter.md) instance and a thin `build(project)` that delegates to the generic [target driver](api/targets-driver.md).

The pattern mirrors `SourceAdapter` on the aggregator side. Reference builders: `targets/run.py` (single-shot multi-source), `targets/sca.py` (year-chunked single-source), `targets/som.py` (multi-variant monthly + annual).

### Adding a new fabric

An HRU polygon set (e.g. GFv1.1, GFv2.0, or a custom delineation). Rarely requires code changes — the same fabric-independent pipeline runs against a new `config.yml`. See the [README §Using a different fabric](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/README.md#using-a-different-fabric) for the operator recipe.

## Conventions

The project's full conventions doc lives at [`CLAUDE.md`](conventions.md) (despite the name, it applies equally to human contributors). Key invariants:

- All source metadata in [`catalog/sources.yml`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/catalog/sources.yml) — no hardcoded URLs / product names in Python.
- All pipeline-written NetCDFs are CF-1.6 compliant and routed through [`io_nc.build_encoding`](api/io-nc.md) + `atomic_to_netcdf`.
- Canonical row order on every fabric-aligned artifact is `id_col` ascending, enforced at emission.
- Manifest writes are flock-guarded read-merge-write (concurrency safety for SLURM arrays).
- Year-chunked target intermediates have fingerprint global attrs for cache invalidation.
- See [Architecture · Python patterns](architecture/python-patterns.md) for the patterns that show up across the codebase.

## Documentation rule

When implementing a new target or source, update README.md §Implementation Status (flip the matching row to **Done** with the PR reference), README.md §Calibration Targets or §Fetch & Consolidation Pipeline (refresh sources / period / method as relevant), and `docs/sources/<key>.md` if a new source landed. CI doesn't catch documentation drift; reviewers should.
