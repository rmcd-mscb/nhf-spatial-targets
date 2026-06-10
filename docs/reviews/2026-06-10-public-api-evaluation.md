# Public-API & Operator-Documentation Evaluation — 2026-06-10

**Scope:** the full public-facing surface of `nhf-spatial-targets` as experienced by
an operator/developer who is *not* the original author — the `nhf-targets` CLI, its
`pixi run` wrappers, the prose docs that lead a newcomer through (a) standing up a
**new fabric** and (b) **catching an existing project up** when a new element lands,
the SLURM harness, the config templates, and the mkdocs site.

**Audience for this document:** the earth-science colleague who may continue
development of this repository. Findings are evidence-backed (`file:line`) and tiered
so you can triage.

**Method:** four parallel read-only audits (command surface, new-fabric onboarding,
existing-project catch-up lifecycle, HPC/config/docs-site), followed by direct
verification of every BLOCKER-class claim. One claim was **refuted** on verification
(see [Appendix: refuted claim](#appendix-one-refuted-claim)); it is recorded here so it
is not "rediscovered" later.

---

## 1. The one root cause

Almost every real problem below is a single pattern wearing different hats:

> **Load-bearing operator knowledge lives in `CLAUDE.md`, `MEMORY.md`, and committed
> SLURM scripts — not in the artifact the operator actually edits, the page they
> actually read, or the `--help` they actually scan.**

`CLAUDE.md` is a *developer/agent* guide. A successor scientist running the pipeline
reads `README.md`, `docs/getting-started.md`, the `config.yml` the template generated,
and `nhf-targets --help`. Wherever a fact lives only in `CLAUDE.md` (or only in
project memory, or only baked into a committed `.slurm` file), the operator-facing
surface has a hole. The fixes are correspondingly cheap: **move the tacit fact to the
surface the operator touches.**

The design itself is sound. The command surface is near-complete, the
intent-vs-derived durability split holds in code, the SLURM harness calls only live
commands, and every current catalog source has a doc page. What's missing is
*continuity at the seams* — the places where a command was renamed, a source was
added, or a fact was assumed.

---

## 2. Findings, tiered

### Tier 1 — Fix now (a successor follows a documented recipe and it fails or misleads)

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **`CLAUDE.md` documents `reconcile-manifest` as a live command — it was removed.** Folded into `rebuild-manifest`. A successor (or Claude) typing it gets "unknown command." | [CLAUDE.md:322-324](../../CLAUDE.md#L322); refuted-into [cli/run.py:262](../../src/nhf_spatial_targets/cli/run.py#L262) "Subsumes the former 'reconcile-manifest'." The tombstone doc [docs/architecture/reconcile-manifest.md](../architecture/reconcile-manifest.md) is correct; only `CLAUDE.md` is stale. |
| 2 | **CDS credential template ships the stale `<uid>:<key>` format** against the *new* CDS endpoint, which issues a single Personal Access Token. `materialize_cdsapirc` writes the string verbatim; `validate` only checks file existence, so the failure surfaces as a runtime 401 the operator can't diagnose. | [init_run.py:213-216](../../src/nhf_spatial_targets/init_run.py#L213) — `url: https://cds.climate.copernicus.eu/api` paired with `key: "<uid>:<key>"`. |
| 3 | **`docs/index.md` SCA + SWE source descriptions are pre-#237 stale.** SCA shown as single-source MOD10C1 (now a two-source MOD10C1 ∪ ua_swe bound); SWE shown as 4 sources / "the remaining three" (ua_swe is a 5th, extending to 1982). The site landing page contradicts `README.md` and the config template. | [docs/index.md:15-18](../index.md#L15) vs [README.md:15-16](../../README.md#L15) and [init_run.py:123-153](../../src/nhf_spatial_targets/init_run.py#L123). |
| 4 | **`docs/sources/ua_swe.md` is orphaned from the mkdocs nav.** The Sources nav ends at SSEBop (15 pages, no ua_swe), so the page is unreachable and the documented `--strict` build will warn "not in nav." ua_swe is central to both SCA (#237) and SWE. | [mkdocs.yml:113-129](../../mkdocs.yml#L113); page exists at [docs/sources/ua_swe.md](../sources/ua_swe.md). |
| 5 | **`--account=impd` is mandatory on caldera but absent from operator-facing HPC docs.** The `default` account is rejected with `AssocMaxSubmitJobLimit`. It's baked into the committed `slurm/project_*/` scripts and lives in project memory, but `README.md` tells operators to `sbatch slurm/shared/*.slurm` and only says directives "may need adjustment." A newcomer's every `slurm/shared/*` submission bounces with no explanation. | `README.md` HPC section (~L294, L329); contrast committed `#SBATCH --account=impd` in every `slurm/**/*.slurm`. |

### Tier 2 — Inconsistencies (defensible but will confuse a successor; design-continuity gaps)

| # | Finding | Evidence |
|---|---------|----------|
| 6 | **The maintenance verb family doesn't track behavior.** `upgrade-config` and `upgrade-manifest` are **report-only** (they never mutate; `upgrade-manifest` just tells you to run `rebuild-manifest`), while `rebuild-manifest` and `rechunk` mutate. So two `upgrade-*` verbs don't upgrade, and there is **no `rebuild-config`** — the config actuator is `validate`, which a successor cannot predict from the names. | [cli/run.py:445-452](../../src/nhf_spatial_targets/cli/run.py#L445), [upgrade_manifest.py:32-59](../../src/nhf_spatial_targets/upgrade_manifest.py#L32), [cli/run.py:323](../../src/nhf_spatial_targets/cli/run.py#L323). |
| 7 | **`-d` / `--project-dir` short alias exists on 3 of 5 sub-surfaces.** Root commands and `release` define `-d`; **`fetch` and `agg` do not.** `validate -d X` works; `fetch merra2 -d X` does not. Highest-frequency daily paper-cut. | `-d` at e.g. [cli/run.py:43](../../src/nhf_spatial_targets/cli/run.py#L43); absent in `cli/fetch.py` / `cli/agg.py` (`name=["--project-dir"]` only). |
| 8 | **`release publish --scope source` (singular) vs `--scope sources` (plural) everywhere else.** `build`/`status`/`dry-run` take `sources`; `publish` takes `source`. Guaranteed paper-cut, undocumented. Same theme: one concept ("a catalog source key") is spelled `--source` (`rechunk`), `--source-key` (`release`), and a positional (`fetch`/`agg`). | `cli/release.py` scope literals (build ~L355 `sources` vs publish ~L406 `source`); [cli/run.py:146](../../src/nhf_spatial_targets/cli/run.py#L146) `--source`. |
| 9 | **`config/pipeline.yml` has drifted from `init_run.py:_CONFIG_TEMPLATE`.** The reference config is **missing `datastore`, `dir_mode`, `fabric.buffer_deg`** — including the single most important path an operator must set — and differs on per-target `nn_fill` defaults and `aet.sources`. It self-declares "REFERENCE ONLY," but `CLAUDE.md`'s four-point rule mandates the mirror, and an operator reading it as the schema reference is misled. | [config/pipeline.yml:1-3](../../config/pipeline.yml#L1) (reference-only banner); `grep datastore config/pipeline.yml` → none; template keys at [init_run.py:40-44](../../src/nhf_spatial_targets/init_run.py#L40). |
| 10 | **`validate` unconditionally requires NASA Earthdata creds + `~/.netrc`** (`required` always seeds `["nasa_earthdata"]`), so *every* project — even one whose enabled targets use no NASA source — fails `validate` until Earthdata creds are materialized. Defensible (nearly every target uses a NASA source) but **undocumented**, and the error gives no hint it's a blanket requirement. | [validate.py:388](../../src/nhf_spatial_targets/validate.py#L388) `required: list[str] = ["nasa_earthdata"]`; netrc check fires because `"nasa_earthdata" in required` is always true ([validate.py:420-421](../../src/nhf_spatial_targets/validate.py#L420)). |
| 11 | **`README.md` Quick-Start fetch list omits `fetch-ua-swe`** (a default source in two enabled targets), and is internally inconsistent on the agg array size ("14" vs "15"). A copy-paste operator never fetches ua_swe, then hits missing aggregated data at `run-sca`/`run-swe`. `getting-started.md` sidesteps this by using `fetch-all`. | `README.md:83-96`; agg count drift `README.md:274,354` ("14") vs `:293` ("15"); array bound is `--array=0-14` = 15 tasks ([slurm/shared/agg_all.body.sh:7](../../slurm/shared/agg_all.body.sh#L7) comment says "14 total"). |
| 12 | **`README.md` internal SCA drift:** the Calibration-Targets table (top) says two-source; the Implementation-Status row says "single-source MOD10C1." Same #237 refactor, half-applied. | `README.md:15` vs `README.md:646`. |
| 13 | **SLURM project dirs are not structurally parallel.** `slurm/project_or/` has a `README.md`; `slurm/project_gfv2/` (the default CONUS fabric) does not. `rechunk_*.slurm` exists only for gfv2. A successor running the *default* fabric has no submit-order doc co-located with the scripts. | `ls slurm/project_gfv2/` (no README), vs [slurm/project_or/README.md](../../slurm/project_or/README.md). |

### Tier 3 — Gaps (missing, but no wrong recipe; reduces discoverability for a successor)

| # | Finding | Evidence |
|---|---------|----------|
| 14 | **No operator-facing "catch-up an existing project" guide exists.** `CLAUDE.md`'s checklists tell the *developer adding* an element what to change; nothing tells the *operator* the sequence to run afterward (`upgrade-config` → paste stub → `validate`; or add source → `fetch`/`agg` → `rebuild-manifest`; or schema bump → `upgrade-manifest` → `rebuild-manifest`). The `upgrade→edit→validate` ordering is unenforced and ambushes the operator at the publish staleness gate. | No README/getting-started mention of any maintenance verb; gate at `release/publish.py` `_preflight_effective_config_current`. |
| 15 | **The five maintenance verbs have no `pixi` wrapper and no `--help` grouping.** `upgrade-config`, `upgrade-manifest`, `rebuild-manifest`, `rechunk` (and the removed `reconcile-manifest`) are reachable only as bare `nhf-targets <verb>` — a different invocation pattern than everything else — and are registered flat on the root app with no `group=`, so `--help` intermixes them with build verbs. Effectively hidden. | [cli/run.py:30-36](../../src/nhf_spatial_targets/cli/run.py#L30); `pixi.toml` has no `upgrade-*`/`rebuild-*`/`rechunk` task. |
| 16 | **`release` family is absent from all user docs** (README/getting-started) despite being seven subcommands and the terminal step of the whole pipeline. Only `pixi.toml` comments mention it. | `pixi.toml:216-222`; no README/getting-started hit. |
| 17 | **API reference omits the provenance/release/validation subsystem.** `docs/api/` covers the read/aggregate/target hot path but has **no page** for `validate`, `rebuild_manifest`/`lineage`, `upgrade_config`, the `release/` package, `rechunk`, or `fetch/consolidate` (`apply_cf_metadata`, the single CF-1.6 entry point) — exactly the invariant-heavy code `CLAUDE.md` flags as subtle. | `ls docs/api/` — 6 module pages, none for the above. |
| 18 | **In-file config template doesn't distinguish required from optional keys**, and ships `fabric.path: /path/to/fabric.gpkg` / `datastore: /path/to/datastore` as unmarked placeholders. A newcomer can run `validate` against placeholders and get a bare `FileNotFoundError` with no hint it was an unedited template value. | [init_run.py:17,40](../../src/nhf_spatial_targets/init_run.py#L17); required-key checks in `workspace.py` (`datastore`, `fabric.path`, `fabric.id_col`). |
| 19 | **`docs/data_release/*.md` (4 pages) are orphaned from mkdocs nav** — the config template points operators to `docs/data_release/` for the release workflow, but the pages aren't reachable from the rendered site and trip `--strict`. | [mkdocs.yml:96-102](../../mkdocs.yml#L96) (not in nav, not in `exclude_docs`). |

### Tier 4 — Polish (low impact; record for completeness)

- **`agg` is the only abbreviated sub-app name** (`fetch`/`catalog`/`release` spelled out). Either alias `aggregate`→`agg` or accept the asymmetry deliberately. ([cli/agg.py:30](../../src/nhf_spatial_targets/cli/agg.py#L30))
- **`run --target` accepts only the long keys** (`recharge`, not `rch`), but the `pixi run run-rch`/`-som`/`-sca` task *names* use the short nicknames; a user typing `--target rch` gets "Unknown target." Help text doesn't list the accepted vocabulary. ([cli/run.py:47-53](../../src/nhf_spatial_targets/cli/run.py#L47))
- **App tagline** says `nhf-spatial-targets` while the binary is `nhf-targets`. ([cli/__init__.py:55](../../src/nhf_spatial_targets/cli/__init__.py#L55))
- **`fetch` redefines `--worker-index`/`--n-workers` without the `-w`/`-n` aliases** that `agg` gives them (via `_params.py`), with divergent help text — two parallel definitions of the same SLURM-array concept.
- **`validate` refreshes the `fabric` block; `rebuild-manifest` preserves it verbatim** — a `rebuild-manifest` after an un-revalidated fabric change carries a stale fabric block. No correctness bug today (publish re-validates) but a subtle divergence worth a one-line note.

---

## 3. Continuity-of-design assessment

You asked whether the API is **complete**, **consistent**, and shows **continuity of
design**. Honest scorecard:

- **Complete — yes, with two narrow holes.** Every pipeline stage has a command; every
  current source has a fetch/agg command, a pixi wrapper, and a doc page. The holes
  are (a) the maintenance/catch-up verbs are present but undiscoverable (Tier 3
  #14–15), and (b) the `release` subsystem is undocumented for users (#16) and
  under-documented in the API reference (#17).

- **Consistent — mostly, fraying at the manifest/config seam.** The build verbs
  (`fetch`/`agg`/`run`/`catalog`) are clean and parallel. The friction is concentrated
  in (a) the maintenance-verb naming not tracking mutate-vs-report (#6), (b) the `-d`
  alias and `--scope source/sources` asymmetries (#7, #8), and (c) the
  `--source`/`--source-key`/positional three-spelling of one concept (#8). None are
  fatal; all are the kind of thing that erodes a successor's trust that the surface was
  designed as a whole.

- **Continuity of design — this is the weakest axis, and it's the one your scenario
  cares most about.** The breaks are exactly where the codebase *evolved*:
  `reconcile-manifest` → `rebuild-manifest` (#1), single-source SCA → two-source (#3,
  #12), the ua_swe #237 addition (#3, #4, #11). Each refactor updated the code and the
  developer guide but left a stale fingerprint on the operator-facing surface. **The
  Documentation Sync Gate in `CLAUDE.md` is the right mechanism; it just hasn't been
  applied to `docs/index.md`, the mkdocs nav, and `config/pipeline.yml` on the last two
  feature waves.**

**The road not taken / reviewer's question for you:** the deepest structural choice
here is whether the maintenance verbs (`upgrade-*`, `rebuild-manifest`, `rechunk`)
should become a `manifest`/`maintenance` **sub-app** (mirroring `fetch`/`agg`/`release`),
so `nhf-targets maintenance --help` reads as a coherent group and the rename to
`manifest rebuild` / `manifest upgrade` makes the report-vs-mutate distinction a
namespace property rather than a verb-mood guess. That's a breaking change to the CLI
grammar, so it belongs in a deliberate decision, not a doc fixup — flagged here as the
one architectural lever worth your judgment.

---

## 4. Recommended fix sequence

**Fix-now (same PR, mechanical, no design decisions — closes every Tier-1 wrong
recipe):**
1. `CLAUDE.md:322-324` → replace `reconcile-manifest` with `rebuild-manifest` (#1).
2. `init_run.py` CDS template → single-PAT format + one-line "Personal Access Token"
   note (#2).
3. `docs/index.md` → refresh SCA (two-source) and SWE (5-source) descriptions (#3);
   `README.md:646` SCA row (#12).
4. `mkdocs.yml` → add `sources/ua_swe.md` and the `data_release/*.md` pages to nav (#4,
   #19).
5. `README.md` → add `fetch-ua-swe` to Quick-Start, fix "14"→"15" agg-array prose; same
   in `slurm/shared/agg_all.body.sh:7` (#11).
6. `README.md` HPC section → explicit "on caldera, pass `--account=impd`" callout (#5).

**Roadmap (a planned doc PR — net-new operator material):**
7. Add an **operator catch-up guide** (a `docs/` page or a README section) with the
   three concrete sequences for new-config-key / new-source / new-manifest-field
   (#14). Mirror the existing developer checklists.
8. Add `slurm/project_gfv2/README.md` mirroring `project_or/` (#13).
9. Reconcile `config/pipeline.yml` with `_CONFIG_TEMPLATE` — at minimum add
   `datastore`, `dir_mode`, `buffer_deg` (#9). (Or: retire `pipeline.yml` and point the
   reference at `init_run.py`, if the four-point rule is more cost than value now.)
10. Extend `docs/api/` to the provenance/release/validation modules (#17); add a `release`
    section to user docs (#16).
11. Mark required keys in the in-file config template; flag the `validate`-needs-NASA-creds
    requirement in getting-started (#10, #18).

**File-issue (CLI-grammar decisions — need your call, don't fix in a doc PR):**
12. `-d` alias on `fetch`/`agg` (#7); `--scope source`→`sources` on `release publish`
    (#8); the `maintenance` sub-app / verb-rename question (#6 + §3 road-not-taken).
    These are breaking or near-breaking; batch them behind a deprecation note.

> **Note on numbering:** the `#N` references in this document are this report's
> internal *finding* IDs (§2 tables), not GitHub issues. The GitHub tracking
> numbers are listed below.

**Follow-up tracking.** Tier-1 closed in PR #313 (tracking issue #312). The roadmap
and CLI-grammar tiers are filed as: operator catch-up guide + maintenance-verb
wrappers → issue #314; `config/pipeline.yml` reconciliation → issue #315;
`slurm/project_gfv2/` README + `slurm/shared/` index → issue #316; mkdocs API
reference + release user docs → issue #317; onboarding required-key markers +
`validate` NASA-cred note → issue #318; CLI grammar consistency + maintenance-verb
structure → issue #319.

---

## Appendix: one refuted claim

An earlier pass flagged the `pixi run nhf-targets agg <src> …` form in `CLAUDE.md:29-42`
as a broken recipe (no `nhf-targets` pixi *task* exists). **Refuted:** `nhf-targets` is
an installed console script ([pyproject.toml:28](../../pyproject.toml#L28)) and
`pixi run <cmd>` executes any binary in the environment, not only named tasks — so the
form runs. It is merely a *style* inconsistency with the `pixi run agg-<src>` task
wrappers documented elsewhere (two valid invocation styles coexist), not a failure.
Recorded so it isn't re-raised.
