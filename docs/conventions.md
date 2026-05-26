# Conventions

The project's full conventions doc is [`CLAUDE.md`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/CLAUDE.md) at the repo root (450+ lines, comprehensive). Despite the name, it applies equally to human contributors — every section describes a project invariant, not an AI-specific instruction. This page surfaces the most important sections; for the full reference, read the source.

## Pre-commit quality gate

Before suggesting a commit, run:

```bash
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
```

Pre-commit hooks enforce this automatically. **Always commit via `pixi run git commit`** (not bare `git commit`) — the pre-commit config drives ruff and pytest through `pixi run`; invoking `git commit` outside a pixi shell forces every hook to re-resolve the pixi environment.

## Catalog is authoritative

All data source metadata lives in [`catalog/sources.yml`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/catalog/sources.yml); all variable definitions in [`catalog/variables.yml`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/catalog/variables.yml). The [`catalog`](api/catalog.md) Python module is the **single Python interface** — do not read YAML directly elsewhere in the codebase. When adding a new source, edit the catalog **first**, then write the fetch module.

When a catalog declaration contradicts an on-disk NetCDF attribute (e.g. a source labels units one way internally and the catalog corrects it), **the catalog wins**. See [`docs/architecture/transformation-pipeline.md`](architecture/transformation-pipeline.md) for the full rule.

## CF-1.6 compliance is required

Every NetCDF the pipeline writes must be CF-1.6 compliant. Use `fetch/consolidate.py:apply_cf_metadata` as the single entry point for setting `Conventions=CF-1.6`, variable `units` / `long_name` / `cell_methods` / `grid_mapping`, coordinate `standard_name` / `units` / `axis`, and the WGS84 `crs` ancillary variable.

**Never write a NetCDF with a bare `ds.to_netcdf(...)`.** Route every pipeline-written NC through [`io_nc.build_encoding`](api/io-nc.md) + `atomic_to_netcdf` so it gets the canonical chunking + zlib + pinned-time encoding. Full policy in [`docs/architecture/nc-encoding-policy.md`](architecture/nc-encoding-policy.md).

## Canonical row order

Every fabric-aligned artifact is `id_col` ascending, enforced at emission (issue #93). Aggregator sorts `year_ds` by `id_col` immediately before atomic write; target writers call `write_target_nc(..., sort_dim=project.id_col)`. `validate` records `id_col_sorted: bool` on `fabric.json`. Downstream code may rely on positional alignment without runtime checks; reasoning in [`docs/architecture/transformation-pipeline.md`](architecture/transformation-pipeline.md).

## Aggregation transformation policy

Where a transformation runs depends on the spatial scale at which it is defined. Aggregation (gdptools area-weighted mean) is a one-way information bottleneck — pixel-defined operations must run pre-aggregation, HRU-defined operations must run post-aggregation, linear operations commute and live downstream by convention.

- **Pre-aggregation** (`aggregate/<src>.py` `pre_aggregate_hook`): flag-value masks, per-pixel quality gates (e.g. MOD10C1 CI > 70).
- **Post-aggregation cosmetic** (`aggregate/<src>.py` `post_aggregate_hook`): rename auxiliary diagnostic variables, attach attrs.
- **Per-HRU transforms** (`normalize/methods.py`): 0–1 normalization, multi-source min/max, NN-fill (target-stage only).
- **Linear unit conversions** (`targets/<tgt>.py`): `× 1000`, `÷ 100`, mm/month → cfs, etc.
- **Multi-source combination** (`targets/<tgt>.py`): post-aggregation by definition; NaN-aware (`np.fmin`/`np.fmax`).

Full reference + the `mean` vs `masked_mean` `stat_method` decision tree in [`docs/architecture/transformation-pipeline.md`](architecture/transformation-pipeline.md).

## Subagent dispatch rules

- Verify subagent prompts list ONLY the cells/files actually being changed.
- NEVER allow subagents to force-push to PR branches; require explicit user approval.
- After subagent completes, verify their changes match the spec before proceeding.

## Test discipline

- Tests live in `tests/`; every new module in `fetch/` / `aggregate/` / `normalize/` / `targets/` gets a corresponding `tests/test_<module>.py`.
- Strict test-first TDD is **not** required for exploratory data work — characterize new sources in a notebook / REPL first, then commit tests before the PR opens.
- For bug fixes, invert: failing test first, then fix.
- **On the caldera HPC**: run newly written tests locally to verify, but **skip the full suite** — push and let CI gate the regression check. CI on a clean Linux runner is materially faster than pixi+pytest on the shared HPC filesystem.

## Python patterns

The repo uses several patterns that look unusual outside production codebases: `from __future__ import annotations` everywhere, `if TYPE_CHECKING:` import guards, frozen-dataclass plugin adapters ([`SourceAdapter`](api/aggregate-adapter.md), [`TargetAdapter`](api/targets-adapter.md)), atomic file writes (tempfile + `os.replace`), manifest read-merge-write under `flock`, and fingerprint-based cache invalidation. Each has a production-correctness reason; full explanation in [Architecture · Python patterns](architecture/python-patterns.md).
