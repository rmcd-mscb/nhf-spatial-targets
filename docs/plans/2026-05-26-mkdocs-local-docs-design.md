# MkDocs + Material — local documentation site

**Issue:** #233
**Date:** 2026-05-26
**Scope:** local-only browsing of repo docs + auto-generated API reference. Publishing deferred.

## Goal

Replace ad-hoc filesystem / GitHub-web browsing of `docs/` with a unified, searchable, hot-reload-served local documentation site. Contributors run `pixi run docs-serve` and get the full repo documentation in a browser at http://127.0.0.1:8000/ with sidebar nav, full-text search, and live-reload on edits.

Publishing to GH Pages is **deliberately deferred** until we decide whether the audience exists. The config will be publish-ready (no per-host hardcoding); enabling GH Pages later is a one-line workflow file.

## Why MkDocs + Material

Three tools were considered: MkDocs + Material, Sphinx + MyST, Quarto. MkDocs wins for this codebase because:

- All existing docs are already Markdown (`docs/architecture/*.md`, `docs/sources/*.md`, `docs/references/*.md`, top-level `README.md` / `CONTRIBUTING.md` / `CLAUDE.md`). Sphinx defaults to RST; MyST mitigates but adds friction.
- Fast iteration via `mkdocs serve` (sub-second hot-reload) matches the "developer runs locally" goal.
- Material theme is the de facto Python project standard (FastAPI, pydantic, etc.) — operators recognize the look.
- Plugin ecosystem covers our likely future needs (mermaid, autodoc, notebooks) without changing tools.
- `mkdocs gh-deploy` is a single command if we ever decide to publish.

Sphinx remains the geoscience standard (xarray / numpy / scipy) and is better for very heavy autodoc workloads, but we don't have that workload yet.

## Tooling additions

### New pixi feature: `docs`

Parallel to existing `dev` and `marp` features in [`pixi.toml`](../../pixi.toml). Conda-forge dependencies:

```toml
[feature.docs.dependencies]
mkdocs = "*"
mkdocs-material = "*"
mkdocstrings = "*"
mkdocstrings-python = "*"

[feature.docs.target.linux-64.dependencies]
# mirrors the dev feature's platform spec — no platform-specific deps for the docs feature itself

[environments]
docs = { features = ["docs"], solve-group = "default" }
```

Approx install footprint: ~30 MB (mkdocs + material + mkdocstrings, all pure-python wheels). No system-level dependencies.

### Pixi tasks

In the `[tasks]` block of `pixi.toml`:

```toml
docs-serve = { cmd = "mkdocs serve --watch src", description = "Serve docs locally at http://127.0.0.1:8000 with hot reload on docs/ and src/" }
docs-build = { cmd = "mkdocs build --strict", description = "Build the docs site once; --strict fails on broken internal links" }
```

The `--watch src` flag tells mkdocs to also watch the source tree, so mkdocstrings-rendered API pages live-refresh when a docstring is edited.

## File layout

```
mkdocs.yml                                  (new — at repo root)

docs/
├── index.md                                (new — short landing page)
├── api/                                    (new dir — stub pages for mkdocstrings)
│   ├── catalog.md
│   ├── workspace.md
│   ├── io-nc.md
│   ├── aggregate-adapter.md                (SourceAdapter)
│   ├── targets-adapter.md                  (TargetAdapter)
│   └── targets-driver.md
├── architecture/                           (existing — included as-is)
├── sources/                                (existing — included as-is)
├── references/                             (existing — included as-is; large .pdf + .f90 excluded via mkdocs `exclude` patterns)
├── plans/                                  (excluded — internal design scratch)
├── presentations/                          (excluded — Marp .slides.md target a separate PDF pipeline)
└── figures/                                (referenced by existing docs; mkdocs picks up relative-path images automatically)
```

Top-level files pulled into the site via symlinks under `docs/` (e.g. `docs/contributing.md → ../CONTRIBUTING.md`). Symlinks are the simplest approach and avoid taking a plugin dependency for what is essentially a path-rewriting concern. mkdocs follows symlinks transparently. If the symlink approach surfaces cross-platform issues at implementation time, fall back to the `mkdocs-include-markdown-plugin`:

- `README.md` → "Getting Started" section (or `index.md` if we prefer the front page = README)
- `CONTRIBUTING.md` → "Contributing" section
- `CLAUDE.md` → "Conventions" section (despite the name, this is the de-facto convention doc)

## Navigation

Material theme uses tabs at the top + sidebar within each tab. Proposed top-level nav:

```yaml
nav:
  - Home: index.md
  - Getting started:
      - Quick start: README.md
      - Workflow: README.md#workflow
  - Architecture:
      - Transformation pipeline: architecture/transformation-pipeline.md
      - NetCDF encoding policy: architecture/nc-encoding-policy.md
      - Python patterns: architecture/python-patterns.md
      - Reconcile manifest: architecture/reconcile-manifest.md
  - Sources:
      - Overview: sources/index.md  # auto-generated or hand-written intro
      - ERA5-Land: sources/era5_land.md
      - GLDAS: sources/gldas.md
      - ... (15 source docs)
  - References:
      - TM 6-B10 crib sheet: references/tm6b10-summary.md
      - PRMSobjfun summary: references/prmsobjfun-summary.md
      - Calibration target recipes: references/calibration-target-recipes.md
      - Known gaps resolved: references/known-gaps-resolved.md
      - Target period coverage: references/target-period-coverage.md
  - API:
      - Catalog: api/catalog.md
      - Workspace: api/workspace.md
      - NetCDF I/O: api/io-nc.md
      - Source adapter: api/aggregate-adapter.md
      - Target adapter: api/targets-adapter.md
      - Target driver: api/targets-driver.md
  - Contributing:
      - Contributing guide: CONTRIBUTING.md
      - Conventions (CLAUDE.md): CLAUDE.md
```

The Sources sidebar lists every source doc; if that gets long (currently 15) Material's collapsible sections handle it.

## API reference stub format

Each stub in `docs/api/<name>.md` is a one-line mkdocstrings directive:

```markdown
# Catalog

::: nhf_spatial_targets.catalog
    options:
      show_source: true
      heading_level: 2
```

mkdocstrings reads the module's docstrings (we already have numpy-style docstrings on every public function), renders them as a structured page with anchored function signatures, and live-refreshes when the source changes (because of `--watch src` on the serve task).

## Excluded content

Configured via mkdocs `exclude` patterns or `mkdocs-exclude` plugin:

- `docs/plans/*.md` — internal design specs (this file lives here)
- `docs/presentations/*.slides.md`, `docs/presentations/*.pdf` — Marp-rendered separately
- `docs/references/*.pdf` — large source PDFs (tm6b10.pdf etc.); leave them on disk and let docs link out
- `docs/references/*.f90` — Fortran source; not a doc

## Pre-existing markdown that needs review

Some existing docs reference paths assuming filesystem layout (e.g. `[../../src/foo](../../src/foo)` from a doc nested two levels deep). mkdocs serves from a flattened URL tree, so these may need adjustment. The implementation will run `mkdocs build --strict` to surface every broken link; fixes apply inline.

## Acceptance criteria

(Mirrored from the issue.)

1. `pixi install -e docs` succeeds.
2. `pixi run docs-serve` launches local server at http://127.0.0.1:8000/ with hot reload.
3. `pixi run docs-build --strict` exits 0 — no broken internal links.
4. All existing markdown files are reachable via site navigation.
5. mkdocstrings renders API pages for the six minimum modules above.
6. README §Development includes a "Building docs locally" subsection.

## Out of scope (deferred follow-ups)

- Publishing to GitHub Pages — would add a `.github/workflows/docs.yml` calling `mkdocs gh-deploy`. One-line change when we decide.
- Notebook rendering via `mkdocs-jupyter` — useful only if we promote inspect notebooks to deliverables.
- Versioned docs via `mike` — premature until publishing.
- Custom Material theme tweaks — defaults are fine for a first cut.
- Automatic API stub generation (every module gets a stub) — for now we hand-pick the 6 highest-value modules.

## Implementation plan

The change is contained enough that a separate plan file isn't needed; this design spec is the plan. Implementation steps:

1. Add `[feature.docs.dependencies]` block + `[environments] docs = ...` line to `pixi.toml`.
2. Add `docs-serve` and `docs-build` tasks to `pixi.toml` `[tasks]`.
3. Write `mkdocs.yml` with the nav structure above + Material theme config + exclude patterns + mkdocstrings plugin.
4. Write `docs/index.md` (short landing page).
5. Create `docs/api/` directory with 6 stub pages (single mkdocstrings directive each).
6. Run `pixi install -e docs`. Run `pixi run docs-build --strict`. Iterate on the resulting broken-link list until clean.
7. Run `pixi run docs-serve` and click through every navigation entry to verify content renders.
8. Add a "Building docs locally" subsection to README.md §Development.
9. Open PR closing #233.

## Risks

- **Existing doc cross-references may break.** Mitigated by `--strict` build and inline fixes. Expect ~5-10 link fixups.
- **mkdocstrings can fail on import cycles or heavy dependencies.** The repo has `if TYPE_CHECKING:` guards already; mkdocstrings should resolve cleanly. If a specific module fails, we can skip it from the API nav as a follow-up.
- **mkdocs `--watch src` adds CPU pressure during serve.** Acceptable for local dev; would be revisited if we ever serve in production.
