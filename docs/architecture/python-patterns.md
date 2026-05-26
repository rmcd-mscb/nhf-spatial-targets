# Python patterns in this repo

This document is the architectural reference for **the non-obvious Python patterns** the pipeline uses across `src/nhf_spatial_targets/`. Each pattern exists for a production-correctness reason (concurrency, partial-write safety, plugin extensibility, cache invalidation) — not for stylistic preference. A geoscientist reading the code asks "why is this here?" and this doc is the answer.

If you are new to the repo and have Python experience but limited software-engineering background, read this once before touching code. It will save you 30 Slack questions and explain why every module starts with the same three weird lines.

## TL;DR

| Pattern | Where to look | Why it's there |
|---|---|---|
| `from __future__ import annotations` (every module) | top of any `.py` | Cheap, modern type-annotation syntax (`X \| Y`) without runtime cost |
| `if TYPE_CHECKING:` import guards | [`cli.py:12-13`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/cli.py#L12-L13) | Heavy imports only at type-check time, never at runtime |
| `Annotated[Type, Parameter(...)]` cyclopts wiring | every `@app.command` in [`cli.py`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/cli.py) | CLI help text lives next to the parameter; no separate argparse spec |
| `@dataclass(frozen=True)` plugin adapters | [`aggregate/_adapter.py:SourceAdapter`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/aggregate/_adapter.py) | Declarative source plugins; immutable singletons safe to share |
| Atomic file writes (tempfile + `os.replace`) | [`io_nc.py:atomic_to_netcdf`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/io_nc.py) | Crashed jobs never leave partial NetCDFs that look valid |
| Manifest read-merge-write under `flock` | [`reconcile.py:_apply_records`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/reconcile.py) | Concurrent SLURM array jobs can't clobber each other's manifest writes |
| Fingerprint-based cache invalidation | [`targets/_common.py:should_skip_year_build`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/targets/_common.py) | Year-chunked intermediates auto-rebuild when config or code changes |

## 1. `from __future__ import annotations`

**What:** Every module in `src/nhf_spatial_targets/` starts with `from __future__ import annotations` immediately after the module docstring.

**Why:** PEP 563 (Postponed Evaluation of Annotations). It tells Python to keep type annotations as strings at function-definition time instead of resolving them eagerly. Two practical wins:

1. We can use the modern `X | Y` union syntax (e.g. `Path | str`) and forward references (annotating a function with a class defined later in the file) without runtime errors on older Python or paying the import cost of `typing.Union`.
2. Annotations never trigger code paths at module import time — useful when an annotation references a heavy type (a numpy dtype, a geopandas class) that you don't want to import just to declare a signature.

**How to extend:** add it to every new module after the docstring. There is no exception. The pre-commit ruff config doesn't enforce this but the codebase is uniform on it, so reviewers will flag a missing line.

## 2. `if TYPE_CHECKING:` import guards

**What:** Some imports live inside an `if TYPE_CHECKING:` block at the top of a module. Example from [`cli.py:12-13`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/cli.py#L12-L13):

```python
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from nhf_spatial_targets.workspace import Project
```

**Why:** `TYPE_CHECKING` is `False` at runtime and `True` only when a static type checker (mypy, pyright) is reading the file. So the import inside the guard runs at type-check time but never at runtime. Two reasons we use it:

1. **Avoid circular imports.** `cli.py` references `Project` in annotations but `workspace.py` may transitively import something from `cli.py`. The guard keeps the type reference without creating a runtime cycle.
2. **Avoid heavy imports.** When a module only needs a class for annotations, not at runtime, the guard prevents loading geopandas / xarray / etc. just to declare a signature.

**How to extend:** use the guard when an import is only needed for type annotations. Pair it with `from __future__ import annotations` (which makes the annotation a string at runtime, so the guarded import is never resolved at runtime). For runtime-needed imports, never use the guard.

## 3. `Annotated[Type, Parameter(...)]` cyclopts wiring

**What:** Every CLI command in `cli.py` annotates its parameters with `Annotated[Type, Parameter(...)]` instead of using a separate argparse-style spec. Example:

```python
workdir: Annotated[
    Path,
    Parameter(name=["--project-dir", "-d"], help="Project created by 'nhf-targets init'."),
],
```

**Why:** cyclopts (our CLI framework) inspects the `Annotated` metadata to build the parser. The parameter's flag name, alias, and help string live **next to the type** rather than in a parallel declaration. The function signature is the single source of truth — no risk of the help text drifting away from the implementation.

**How to extend:** when adding a new CLI command, use `Annotated[<concrete-type>, Parameter(name=..., help=...)] = <default>` for each parameter. Defaults live on the function signature (so type checkers see them); cyclopts reads `name` and `help` from the `Parameter` metadata. For parameters reused across commands (e.g. `--batch-size` on every `agg` subcommand), pull the `Parameter` into a module-level alias and reference it — see `_AGG_BATCH_SIZE_PARAM` in [`cli.py`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/cli.py).

## 4. `@dataclass(frozen=True)` plugin adapters

**What:** Plugin extension points are declared as frozen dataclasses, not classes you subclass. The canonical example is [`aggregate/_adapter.py:SourceAdapter`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/aggregate/_adapter.py): a 60-field frozen dataclass with hook fields (`pre_aggregate_hook`, `post_aggregate_hook`, `stat_method`, …) that each gridded source instantiates as a module-level constant:

```python
ADAPTER = SourceAdapter(
    source_key="era5_land",
    output_name="era5_land",
    variables=("ro", "sro", "ssro"),
    stat_method="mean",
    output_cadence="monthly",
    # ...
)
```

**Why:** declarative plugin patterns scale better than inheritance hierarchies for data pipelines. There is no `SourceBase` class with abstract methods to override; each source is a small new file declaring an adapter instance, which the shared driver (`aggregate/_driver.py`) consumes generically. Frozen so the instance can be safely shared as a module-level constant — it can't be mutated by accident.

**How to extend:** when adding a new gridded source, write `src/nhf_spatial_targets/aggregate/<source>.py` that declares an `ADAPTER = SourceAdapter(...)` at module level and a thin `aggregate_<source>(project, ...)` function that hands the adapter to the driver. See `aggregate/era5_land.py` for the simplest case and `aggregate/mod10c1.py` for an adapter that uses `pre_aggregate_hook` (per-pixel CI gate) + `stat_method="masked_mean"`. A `TargetAdapter` analog is planned for target builders (#219); until then, target builders follow a less declarative pattern.

## 5. Atomic file writes (tempfile + `os.replace`)

**What:** Every NetCDF and JSON write in the pipeline goes through an atomic write — write to a tempfile in the **same directory** as the final path, then `os.replace(tmp, final)`. The canonical entry is [`io_nc.py:atomic_to_netcdf`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/io_nc.py#L260):

```python
tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".nc.tmp")
# ... write to tmp_path ...
tmp_path.replace(path)
```

**Why:** a partial write left by a crashed or SIGKILL'd job is worse than no write. A truncated NetCDF looks valid to `xr.open_dataset` (the file format doesn't have a "this is incomplete" marker) — it just has missing or zero-padded data at the tail. POSIX guarantees that `rename` on the same filesystem is atomic: readers see either the old file or the new file, never a partial one. The tempfile must be in the same directory as the target so the rename stays atomic (cross-filesystem renames degrade to copy+delete and are not atomic). On any exception, the tempfile is unlinked so no `.tmp` cruft accumulates.

**How to extend:** never call `ds.to_netcdf(...)` or `path.write_text(...)` directly for any persisted artifact (target NCs, aggregated NCs, manifest.json, fabric.json). Route NetCDFs through `io_nc.atomic_to_netcdf`; route JSON through the same tempfile-then-rename pattern (see [`reconcile.py:_atomic_write`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/reconcile.py#L108) for the JSON variant). The exception is logs and scratch files (under `<project>/logs/` or `/tmp`) where a partial file is harmless.

## 6. Manifest read-merge-write under `flock`

**What:** `manifest.json` lives at the project root and accumulates provenance for every fetch, aggregation, and target run. Multiple SLURM array tasks can hit it concurrently (e.g. 14 parallel fetch jobs each appending a source entry). The pipeline serializes those writes with a POSIX advisory lock. Pattern from [`reconcile.py:_apply_records`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/reconcile.py#L119-L180):

```python
with open(lock_path, "a") as lock_f:
    if _HAVE_FLOCK:
        _fcntl.flock(lock_f, _fcntl.LOCK_EX)
    manifest = _read_manifest(manifest_path)   # read current state
    # ... merge new records into manifest ...
    _atomic_write(manifest_path, manifest)     # write back atomically
```

**Why:** without the lock, two concurrent fetches would both `read → mutate → write` and the slower writer would clobber the faster one's changes. This is the classic lost-update race. We use `fcntl.flock` on a sibling `.lock` file (the lock file is appended to so its inode is stable) with `LOCK_EX` (exclusive). On Windows where `fcntl` is unavailable, the code emits a one-time warning that writes are unserialized — acceptable for single-user laptop runs, never for HPC concurrency.

**How to extend:** any code that updates a JSON metadata file under `<project>/` must use the lock + read-merge-write pattern. Never call `manifest.json.write_text(json.dumps(new_records))` from a fetch or aggregate path — that's an overwrite, not a merge, and it will silently delete every other source's provenance the first time two jobs race. See issue #97 for the bug that motivated this pattern, and issue #180 for one outstanding spot in the MODIS fetch that still writes without `flock`.

## 7. Fingerprint-based cache invalidation

**What:** Year-chunked target builders (SCA, SWE) write per-year intermediate NetCDFs to `<project>/targets/.sca_intermediates/<year>.nc` and stitch them on the next run. To know when a per-year intermediate is stale, each intermediate carries two global attrs: `config_fingerprint` (a hash of the active target config) and `code_version` (the package `__version__`). The skip-or-rebuild decision is in [`targets/_common.py:should_skip_year_build`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/targets/_common.py).

```python
# In targets/sca.py: build() iterates years; for each year,
# should_skip_year_build() compares the on-disk intermediate's
# fingerprint attrs against the active config + code version.
# Match -> skip. Mismatch -> log a WARNING, unlink the stale
# file, rebuild the year.
```

**Why:** a 25-year daily target rebuild is expensive (hours of HRU-aggregation reads), and most edits only change one or two years' worth of computation. Fingerprinting lets the operator change `ci_threshold` in `config.yml`, rerun `pixi run run-sca`, and have just the affected years rebuild — without having to manually `rm` the intermediates dir. The same mechanism handles downward period changes (the orphan-pruning helper deletes per-year files outside the new period; see [`prune_orphan_year_intermediates`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/targets/_common.py) and PR #216).

**The geoscientist gotcha:** the `code_version` tag is tied to the package `__version__` string in [`src/nhf_spatial_targets/__init__.py`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/__init__.py). If you edit a target builder's logic **without** bumping `__version__`, the fingerprint won't change and your edit will not take effect on cached intermediates. You must `rm -rf <project>/targets/.<target>_intermediates/` to force a rebuild. The SCA and SWE module docstrings document this; the operator-visible WARNING does not fire in this case because the fingerprint sees no change. Treat mid-version builder edits as a manual-cache-clear operation until a smarter fingerprint scheme lands.

## Where this fits

These seven patterns are **production-correctness scaffolding**. They show up in every module in `src/`, but they're not the *interesting* part of any module — the science is. Read this doc once, then ignore the patterns when reading new code; they should fade into the background.

If you find yourself writing code that looks like one of these patterns but isn't quite right (e.g. a direct `ds.to_netcdf(...)`, a manifest write without `flock`, a new CLI command without `Annotated`), the answer is almost always to route through the existing helper — `io_nc.atomic_to_netcdf`, `reconcile._apply_records` (or its caller), `cli._AGG_*_PARAM`. Reinventing them per-module is how the codebase decays.

For the calibration-target science (what does the bound mean, when does it collapse, how does NaN propagate through multi-source combination), see [`transformation-pipeline.md`](transformation-pipeline.md). For the NetCDF encoding policy (chunk shapes, codecs, time pinning), see [`nc-encoding-policy.md`](nc-encoding-policy.md).
