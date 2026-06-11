# CLI Grammar Consistency & Maintenance Sub-App — Design (issue #319)

**Date:** 2026-06-10
**Issue:** #319 — CLI grammar consistency (`-d` alias, `--scope source/sources`,
source naming) + maintenance-verb structure
**Source review:** `docs/reviews/2026-06-10-public-api-evaluation.md` Tier-2
#6/#7/#8 + §3 road-not-taken.

## Decisions (operator-confirmed)

1. **Maintenance sub-app with `check-*` rename.** The four flat maintenance
   verbs move under a new `nhf-targets maintenance` sub-app, and the two
   report-only verbs are renamed so verb mood tracks behavior:

   | Old (root) | New | Behavior |
   |---|---|---|
   | `upgrade-config` | `maintenance check-config` | report-only |
   | `upgrade-manifest` | `maintenance check-manifest` | report-only |
   | `rebuild-manifest` | `maintenance rebuild-manifest` | mutates `manifest.json` |
   | `rechunk` | `maintenance rechunk` | mutates NCs in place |

2. **Hard break — no deprecated aliases.** The old root spellings are removed
   outright. Every committed reference (pixi tasks, SLURM scripts, docs,
   CLAUDE.md, in-code messages) is updated in the same PR. Historical records
   (`docs/superpowers/{specs,plans}`, `docs/reviews`) are point-in-time
   documents and are NOT edited.

3. **`release publish --scope sources` (rename only, same behavior).** The
   publish scope value `source` becomes `sources`, matching
   `build`/`status`/`dry-run`. Publishing a single source child still requires
   the source-key flag; no publish-all-sources behavior is added.

4. **`--source` is the canonical source-key flag.** `rechunk --source` is
   already correct; `release publish --source-key` becomes `--source` (hard
   break). The Python API keyword `source_key=` is internal and unchanged.

## Additional in-scope items (from the issue checklist)

- **`-d` alias on every `fetch` and `agg` command.** A shared
  `_PROJECT_DIR_PARAM` (`name=["--project-dir", "-d"]`) moves into
  `cli/_params.py`; `fetch.py`, `agg.py`, `release.py`, and the new
  `maintenance.py` all use it (release already had its own copy — consolidated).
- **`aggregate` alias for the `agg` sub-app** (additive; `agg` remains primary).
- **`run --target` accepts the short nicknames** used by the pixi task names:
  `rch` → `recharge`, `som` → `soil_moisture`, `sca` → `snow_covered_area`,
  `swe` → `snow_water_equivalent` (long keys keep working; help text lists the
  accepted vocabulary).
- **App tagline** says `nhf-targets` (matching the binary) instead of
  `nhf-spatial-targets`.

## Out of scope

- Renaming the Python implementation modules (`upgrade_config.py`,
  `upgrade_manifest.py`) or their public functions — internal API, not CLI
  grammar. The `docs/api/` pages keep their module focus; only the CLI
  invocations shown in them change.
- Publish-all-sources semantics for `release publish --scope sources`
  (deliberately rejected to keep this PR behavior-neutral).
- The `fetch` vs `agg` `--worker-index`/`-w` alias divergence (review Tier-4,
  not in the issue checklist).

## Architecture

New module `src/nhf_spatial_targets/cli/maintenance.py`:

- `maintenance_app = App(name="maintenance", help=...)` registered on the root
  app in `cli/__init__.py`.
- The four command bodies move verbatim from `cli/run.py` (no behavior
  change beyond names): `check_config_cmd`, `check_manifest_cmd`,
  `rebuild_manifest_cmd`, `rechunk`.
- `cli/__init__.py` re-exports keep working for tests
  (`rechunk`, `rebuild_manifest_cmd`, `check_config_cmd`, `check_manifest_cmd`).

`cli/run.py` keeps `init`, `materialize-credentials`, `validate`, `run` and
gains a `_TARGET_NICKNAMES` mapping applied before dispatch.

`cli/release.py`: `--scope` literal sets become `{fabric, sources, umbrella}`
for publish; `_PUBLISH_PREVIEW_SCOPE` keys updated; `--source-key` parameter
renamed to `--source` (function kwarg `source_key` retained internally);
error/help text updated. Domain-level `PublishResult.scope` literals
(`"source"` etc.) are untouched.

In-code operator messages that print old spellings are updated:
`upgrade_manifest.py` ("run 'nhf-targets maintenance rebuild-manifest …'"),
plus any `rebuild-manifest`/`rechunk` invocation strings in
`release/publish.py`, `release/payload.py`, `rebuild_manifest.py`,
`targets/_intermediates.py`, `fetch/margulis_wus_sr.py` (prose mentions of
chunking stay).

## Surface updates (hard-break sweep)

- **pixi.toml:** tasks `upgrade-config` → `check-config`, `upgrade-manifest` →
  `check-manifest`; all four cmds become `nhf-targets maintenance …`;
  comment block updated.
- **SLURM:** `slurm/project_gfv2/rechunk_gfv2*.slurm` and
  `slurm/project_gfv2/README.md` use `nhf-targets maintenance rechunk`.
- **Docs:** CLAUDE.md (commands, config-schema checklist, manifest section),
  README.md, CONTRIBUTING.md, docs/maintenance.md, docs/api/{index,
  rebuild-manifest, upgrade-config, rechunk, lineage}.md,
  docs/architecture/{reconcile-manifest, nc-encoding-policy,
  transformation-pipeline}.md, docs/data_release/{usgs-process,
  walkthrough}.md, mkdocs.yml nav labels if they name commands.

## Testing

- Move/rename CLI-level tests: `test_upgrade_config.py` /
  `test_upgrade_manifest.py` import from `cli.maintenance`;
  `test_rechunk.py` re-export path unchanged.
- `test_release_cli.py`: `--scope sources`, `--source`.
- New coverage in `test_cli.py` / `test_cli_agg.py`:
  - `maintenance <verb>` dispatches (all four),
  - old root spellings now fail (hard-break guard),
  - `-d` works on a representative `fetch` and `agg` command,
  - `run --target rch` maps to `recharge`; unknown nickname still errors,
  - `aggregate` aliases `agg`.
