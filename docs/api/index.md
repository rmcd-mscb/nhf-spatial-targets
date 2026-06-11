# API reference

Auto-generated from numpy-style docstrings in `src/nhf_spatial_targets/` via [mkdocstrings](https://mkdocstrings.github.io/). When running `pixi run docs-serve` locally, these pages live-refresh on any source-tree docstring edit.

## Core modules

| Module | Page | What it does |
|---|---|---|
| `catalog` | [Catalog](catalog.md) | YAML registry interface; single Python entry point to `catalog/sources.yml` + `catalog/variables.yml` |
| `workspace` | [Workspace](workspace.md) | `Project` dataclass, path resolution, `make_dir()` |
| `io_nc` | [NetCDF I/O](io-nc.md) | Canonical chunking + zlib + pinned-time encoding policy; `atomic_to_netcdf` (tempfile + `os.replace`) |
| `aggregate._adapter` | [SourceAdapter](aggregate-adapter.md) | Declarative source plugin for gridded sources aggregated via gdptools |
| `targets._adapter` | [TargetAdapter](targets-adapter.md) | Declarative target plugin for the generic target-build driver |
| `targets._driver` | [Target driver](targets-driver.md) | Generic `build(adapter, project)`; single-shot + year-chunked paths |

## Provenance, validation & release

The invariant-heavy subsystem behind `validate`, the maintenance verbs, and the
ScienceBase release (`CLAUDE.md` §Manifest & config durability).

| Module | Page | What it does |
|---|---|---|
| `validate` | [Validate](validate.md) | Preflight checks; writes `fabric.json` + `config.effective.yml` (the config actuator) |
| `rebuild_manifest` | [Rebuild manifest](rebuild-manifest.md) | `manifest.json` as a deterministic projection of (disk × catalog × `fabric.json`) |
| `release.lineage` | [Lineage](lineage.md) | Shared manifest skeleton, step kinds, `mtime`-derived timestamps |
| `upgrade_config` | [Upgrade config](upgrade-config.md) | Report-only optional-config drift (`maintenance check-config`); the paste-this block |
| `rechunk` | [Rechunk](rechunk.md) | Idempotent backfill of NCs to the canonical chunk/compress layout |
| `fetch.consolidate` | [Consolidate](consolidate.md) | `apply_cf_metadata` — the single CF-1.6 metadata entry point |
| `release` | [Release package](release.md) | ScienceBase data-release subsystem (build → dry-run → publish) |

The operator-facing command sequence for the release subsystem is the
[Data release walkthrough](../data_release/walkthrough.md).

## What's not here

Per-source `fetch/<src>.py`, `aggregate/<src>.py`, and per-target `targets/<tgt>.py` modules are intentionally **not** included in the API reference. Their public surface is one function (`fetch_<src>`, `aggregate_<src>`, `build`) that the CLI dispatch wires up; the interesting per-source / per-target logic is in their module docstrings, which are easier to read in-place than as autodoc pages. See those modules directly under [`src/nhf_spatial_targets/`](https://github.com/rmcd-mscb/nhf-spatial-targets/tree/main/src/nhf_spatial_targets/) if you need them.

The mkdocstrings autodoc filter excludes private members (`_foo`, `__foo`) by default — those show up only when you read the source directly.

## Adding a new module to the API reference

1. Create `docs/api/<module-stub>.md` with a single mkdocstrings directive:

    ```markdown
    # Module title

    ::: nhf_spatial_targets.<dotted.module.path>
        options:
          show_source: true
    ```

2. Add the page to the `nav:` block in [`mkdocs.yml`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/mkdocs.yml) under `API reference:`.
3. Verify with `pixi run docs-build --strict` — broken refs fail the build.
