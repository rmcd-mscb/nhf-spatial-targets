# Durable Manifest & Config Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `manifest.json` and `config.effective.yml` durable, version-stamped, deterministic projections of (on-disk artifacts × catalog × `fabric.json`) — regenerable at any time, gated (verify-don't-mutate) at publish, and validated against the catalog so dangling refs fail loudly.

**Architecture:** One shared manifest skeleton + a `manifest_schema_version`; one authoritative `rebuild-manifest` command that is a generic, catalog-keyed deterministic projection (subsuming `reconcile-manifest`, `rebuild_lineage`, the unbuilt #277 `backfill-aggregate-sources`, and the #278 consolidate disk-walk); a completeness gate at publish that runs the projection in `--dry-run` and refuses on drift; the same durability treatment for `config.effective.yml`; and loud catalog-key validation of `config.yml` `targets.*.sources[]`.

**Tech Stack:** Python 3.11+, cyclopts CLI, `fcntl` flock + atomic tempfile-rename writes, xarray/netCDF4 for NC attr reads, pixi for env/test orchestration, pytest.

**Source spec:** [docs/superpowers/specs/2026-05-29-durable-manifest-and-config-design.md](../specs/2026-05-29-durable-manifest-and-config-design.md). Locked decisions A–G and Pillars 1–7 are normative; this plan implements them.

---

## Hard constraints (apply to every PR)

These are non-negotiable (spec decisions E/F and CLAUDE.md):

1. **No `datetime.now()` anywhere in the rebuild path.** The rebuild projection derives every timestamp from file `mtime` only. A lint/import guard enforces this (PR-2). `validate` and live capture *may* call `datetime.now()` — they are not the rebuild path.
2. **Deterministic JSON ordering.** Sources are emitted with sorted keys; steps are sorted by `(kind-rank, source_key, first-output-path)`. Same disk + catalog + code → byte-identical manifest (modulo the opt-in `--compute-sha256`).
3. **Regenerate is non-clobbering of identity fields.** `created_utc`, the `fabric` authorship block, and any `release` config are read-merged, never overwritten. Adding a new dataset/variable/target must never mutate an existing manifest until that project produces new on-disk artifacts.
4. **Honesty tags.** Every regenerated record carries `provenance: "reconstructed"`; the manifest carries `manifest_schema_version` (starts at `1`).
5. **Publish gate is verify-don't-mutate.** It runs `rebuild-manifest --dry-run`, refuses if on-disk ≠ projection or incomplete, and tells the operator to run `rebuild-manifest`. `--allow-incomplete-sources` is the logged override.
6. **Workflow:** branch `feature/279-durable-artifacts` already exists off main (holds spec + this plan). Commit via `pixi run git commit` (never bare `git commit`). Stage files explicitly by path. Run `pixi run -e dev fmt` + `pixi run -e dev lint` locally; let GitHub Actions run pytest (slow HPC box) — but author and run targeted tests locally where fast.

## Branch & PR strategy (stacked) — resolved 2026-05-29

The seven PRs ship as a **stacked chain**: each PR branches off the previous PR's branch and targets it as its base, so every PR's diff shows only its own changes. `feature/279-durable-artifacts` is the integration base holding the spec + this plan.

| PR | Branch | Base (PR target) |
|---|---|---|
| (base) | `feature/279-durable-artifacts` | `main` (spec+plan; open a PR or fold into PR-1) |
| PR-1 | `feature/279-pr1-skeleton` | `feature/279-durable-artifacts` |
| PR-2 | `feature/279-pr2-rebuild` | `feature/279-pr1-skeleton` |
| PR-3 | `feature/279-pr3-publish-gate` | `feature/279-pr2-rebuild` |
| PR-4 | `feature/279-pr4-effective-config` | `feature/279-pr3-publish-gate` |
| PR-5 | `feature/279-pr5-config-validation` | `feature/279-pr4-effective-config` |
| PR-6 | `feature/279-pr6-claude-md` | `feature/279-pr5-config-validation` |
| PR-7 | `feature/279-pr7-triangle` | `feature/279-pr6-claude-md` |

Create each branch off its base at the start of that PR's work. After a PR merges, re-target the next PR in the chain to `main` (GitHub auto-retargets on base-branch merge). Never force-push a shared branch without explicit maintainer approval (CLAUDE.md). When a base PR absorbs review changes, rebase the downstream branch onto the updated base before continuing.

## Key existing code (verified on disk 2026-05-29)

- `src/nhf_spatial_targets/release/lineage.py` — `_new_manifest_skeleton()` currently returns only `{"sources": {}, "steps": []}` (lineage.py:168-175). `read_manifest` (lineage.py:178-197) raises `ValueError` on corrupt, setdefaults `sources`/`steps`. `atomic_write_manifest` (lineage.py:200-215). `with_flock` (lineage.py:260-274). `build_step_record` (lineage.py:218-257) uses `datetime.now()` when `timestamp_utc is None`. `STEP_KINDS = {"fetch","consolidate","aggregate","target","nn_fill","validate"}` (lineage.py:56-57).
- `src/nhf_spatial_targets/validate.py` — `_write_manifest(workdir, fabric_meta)` (validate.py:513-577) builds a **second, inline** 6-key skeleton (`created_utc`, `last_validated_utc`, `nhf_spatial_targets_version`, `fabric`, `sources`, `steps`). `_write_effective_config(workdir, merged)` (validate.py:77-102) YAML-dumps the merged config to `config.effective.yml` at mode 0444 — **no version/hash stamp**. Catalog consistency check `_check_catalog_consistency` (validate.py:403-413) only validates `variables.yml → sources.yml`, raising plain `ValueError`.
- `src/nhf_spatial_targets/reconcile.py` — `reconcile_manifest(project, *, sources, dry_run, checksum)` with a `_RECONCILERS` registry of only `era5_land` + `mod16a2_v061` → every other source returns `no-hook`.
- `src/nhf_spatial_targets/cli/run.py` — `register(app)` (run.py:27-35) wires root commands via `app.command(fn, name="...")`. `upgrade_config_cmd` (run.py:313-371) is the report-only-command model: exits 0 in sync, 1 on drift, never mutates.
- `src/nhf_spatial_targets/upgrade_config.py` — `OptionalConfigFeature` dataclass + `OPTIONAL_CONFIG_FEATURES` registry + `check_drift(project_dir) -> list[OptionalConfigFeature]`.
- `src/nhf_spatial_targets/release/publish.py` — `PreflightError(ReleaseError)` (publish.py:104-105); `_preflight_common(project)` (publish.py:320-350) checks manifest exists+parses, release defaults populated, authors non-empty. No `_preflight_provenance_complete` and no `allow_incomplete_sources` exist yet.
- `src/nhf_spatial_targets/release/payload.py` — `stage_fabric_child(project, *, copy=False)` (payload.py:173-200) copies the manifest only `if plan.manifest_src.exists()` — a silent skip.
- `src/nhf_spatial_targets/release/rebuild.py` — `rebuild_lineage(project, *, compute_sha256, dry_run)` (rebuild.py:444-499) synthesizes steps from existing `sources[]`; reads `config.effective.yml` only as a step *output entry* (rebuild.py:408-410), never parses it.
- `src/nhf_spatial_targets/catalog.py` — `sources()`, `source(name)`, `variables()`, `publishable_sources()` (filters `superseded_by`), `validate_fabric_scope`, `FABRIC_SCOPE_TOKENS = frozenset({"or"})`.
- `src/nhf_spatial_targets/defaults.py` — `DEFAULTS` (defaults.py:24-150), `apply_defaults(user)` deep-merge (defaults.py:170-181), `_deep_merge` (defaults.py:232+). Per-target keys: `enabled`, `sources`, `time_step`, `period`, `range_method`, `normalize`/`normalize_period` (rch/som), `ci_threshold` (sca), `nn_fill`, etc.
- `src/nhf_spatial_targets/workspace.py` — `load(workdir) -> Project`; `Project.config` is the merged dict; `Project.id_col`, `Project.area_crs`, `Project.target(name)`, `Project.manifest_path`, `Project.aggregated_dir()`.
- Aggregated NC filenames: `<key>_agg.nc` (single), `<key>_<year>_agg.nc` (year-chunked, e.g. `ssebop_2000_agg.nc`, `mod10c1_v061_2020_agg.nc`), `<key>_<region>_<year>_agg.nc` (e.g. `daymet_na_1980_agg.nc`). Aggregated dirs live under `<project>/data/aggregated/<key>/`.

---

## File structure (created / modified across all PRs)

**New files:**
- `src/nhf_spatial_targets/upgrade_manifest.py` (PR-1) — report-only schema-skew detector.
- `src/nhf_spatial_targets/rebuild_manifest.py` (PR-2) — the generic deterministic projection.
- `tests/test_upgrade_manifest.py` (PR-1)
- `tests/test_rebuild_manifest.py` (PR-2)
- `tests/test_effective_config.py` (PR-4)

**Modified files:**
- `src/nhf_spatial_targets/release/lineage.py` (PR-1: skeleton + schema version; PR-2: deterministic-ordering helpers, mtime helper)
- `src/nhf_spatial_targets/validate.py` (PR-1: use shared skeleton; PR-4: stamp effective config; PR-5: catalog-key validation of `targets.*.sources[]`)
- `src/nhf_spatial_targets/cli/run.py` (PR-1: `upgrade-manifest`; PR-2: `rebuild-manifest`)
- `src/nhf_spatial_targets/reconcile.py` (PR-2: thin shim / deprecation)
- `src/nhf_spatial_targets/release/rebuild.py` (PR-2: absorb `rebuild_lineage`; PR-4: parse effective config)
- `src/nhf_spatial_targets/release/publish.py` (PR-3: completeness gate; PR-4: effective-config staleness gate)
- `src/nhf_spatial_targets/release/payload.py` (PR-3: hard-error on missing manifest)
- `src/nhf_spatial_targets/upgrade_config.py` (PR-5: whole-target + new-source hints)
- `src/nhf_spatial_targets/targets/_driver.py` / `_writers.py` (PR-7: resolved-param NC attrs)
- `CLAUDE.md` (PR-6: "Manifest & config durability" section)

---

# PR-1 — Schema version + single skeleton + `upgrade-manifest`

**Branch note (stacked):** Branch `feature/279-pr1-skeleton` off `feature/279-durable-artifacts`; open PR-1 with base = `feature/279-durable-artifacts`. Earlier PRs reference `#279`; `Closes #279` goes only on the final PR in the chain.

```bash
git switch feature/279-durable-artifacts
git switch -c feature/279-pr1-skeleton
```

**Goal:** Unify the two manifest skeletons into one canonical top-level shape in `lineage.py`, stamp `manifest_schema_version` (start `1`), have `read_manifest` surface it (default `0` for pre-version manifests), and add a report-only `nhf-targets upgrade-manifest -d <dir>`.

### Task 1.1: Add schema-version constant + canonical skeleton in lineage.py

**Files:**
- Modify: `src/nhf_spatial_targets/release/lineage.py:161-197`
- Test: `tests/test_release_lineage.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_release_lineage.py`:

```python
def test_new_manifest_skeleton_carries_canonical_top_level():
    from nhf_spatial_targets.release.lineage import (
        CURRENT_MANIFEST_SCHEMA_VERSION,
        _new_manifest_skeleton,
    )

    skel = _new_manifest_skeleton()
    assert set(skel) == {
        "manifest_schema_version",
        "created_utc",
        "last_validated_utc",
        "nhf_spatial_targets_version",
        "fabric",
        "sources",
        "steps",
    }
    assert skel["manifest_schema_version"] == CURRENT_MANIFEST_SCHEMA_VERSION
    assert CURRENT_MANIFEST_SCHEMA_VERSION == 1
    # Identity/timestamp fields are left for the caller (validate) to fill,
    # so the skeleton itself never calls datetime.now().
    assert skel["fabric"] is None
    assert skel["created_utc"] is None
    assert skel["sources"] == {}
    assert skel["steps"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_release_lineage.py::test_new_manifest_skeleton_carries_canonical_top_level -v`
Expected: FAIL — `ImportError: cannot import name 'CURRENT_MANIFEST_SCHEMA_VERSION'` (and skeleton key mismatch).

- [ ] **Step 3: Implement** — in `lineage.py`, add the constant after `STEP_KINDS` (after line 57):

```python
# Bumped ONLY when the manifest top-level *shape* changes (not when
# sources/steps content changes). read_manifest defaults a manifest with no
# version key to 0, so pre-version manifests are detectable and upgradable.
CURRENT_MANIFEST_SCHEMA_VERSION = 1
```

Update the `_Manifest` TypedDict (lineage.py:161-165) to the full shape:

```python
class _Manifest(TypedDict):
    """In-memory shape of ``manifest.json`` (canonical top-level)."""

    manifest_schema_version: int
    created_utc: str | None
    last_validated_utc: str | None
    nhf_spatial_targets_version: str
    fabric: dict | None
    sources: dict[str, dict]
    steps: list[StepRecord]
```

Replace `_new_manifest_skeleton` (lineage.py:168-175):

```python
def _new_manifest_skeleton() -> _Manifest:
    """Return a fresh ``manifest.json`` skeleton — the single source of truth
    for the canonical top-level shape.

    Both ``validate._write_manifest`` and the lineage writers build on this,
    so a lineage-first write can no longer produce a ``fabric``-less manifest
    (the ``fabric`` key is always present, ``None`` until validate fills it).

    Identity/timestamp fields are ``None`` here, never ``datetime.now()`` — the
    skeleton must be importable from the rebuild path without minting a clock
    read. ``validate`` (which may call ``now()``) fills ``created_utc`` /
    ``last_validated_utc`` / ``fabric``.
    """
    return {
        "manifest_schema_version": CURRENT_MANIFEST_SCHEMA_VERSION,
        "created_utc": None,
        "last_validated_utc": None,
        "nhf_spatial_targets_version": _SOFTWARE_VERSION,
        "fabric": None,
        "sources": {},
        "steps": [],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_release_lineage.py::test_new_manifest_skeleton_carries_canonical_top_level -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/release/lineage.py tests/test_release_lineage.py
pixi run git commit -m "feature(manifest): canonical skeleton + manifest_schema_version (#279)"
```

### Task 1.2: `read_manifest` surfaces `manifest_schema_version` (default 0)

**Files:**
- Modify: `src/nhf_spatial_targets/release/lineage.py:178-197`
- Test: `tests/test_release_lineage.py`

- [ ] **Step 1: Write the failing test:**

```python
def test_read_manifest_defaults_preversion_to_zero(tmp_path):
    import json
    from nhf_spatial_targets.release.lineage import read_manifest

    # A pre-version manifest (old validate / old lineage skeleton).
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"sources": {}, "steps": []}))
    m = read_manifest(p)
    assert m["manifest_schema_version"] == 0  # detectable as behind


def test_read_manifest_absent_returns_current_skeleton(tmp_path):
    from nhf_spatial_targets.release.lineage import (
        CURRENT_MANIFEST_SCHEMA_VERSION,
        read_manifest,
    )

    m = read_manifest(tmp_path / "manifest.json")  # absent
    assert m["manifest_schema_version"] == CURRENT_MANIFEST_SCHEMA_VERSION
    assert "fabric" in m  # canonical key always present
```

- [ ] **Step 2: Run — Expected FAIL:** pre-version manifest has no `manifest_schema_version` key → `KeyError`.

Run: `pixi run -e dev pytest tests/test_release_lineage.py -k read_manifest -v`

- [ ] **Step 3: Implement** — in `read_manifest` (lineage.py:195-197), add the version default alongside the existing setdefaults:

```python
    manifest.setdefault("manifest_schema_version", 0)
    manifest.setdefault("sources", {})
    manifest.setdefault("steps", [])
    return manifest
```

(Absent-file path already returns `_new_manifest_skeleton()`, which now carries the current version — no change needed there.)

- [ ] **Step 4: Run — Expected PASS.**

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/release/lineage.py tests/test_release_lineage.py
pixi run git commit -m "feature(manifest): read_manifest surfaces schema version, default 0 (#279)"
```

### Task 1.3: `validate._write_manifest` uses the shared skeleton

**Files:**
- Modify: `src/nhf_spatial_targets/validate.py:513-577`
- Test: `tests/test_validate.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_validate.py` (a focused unit test on `_write_manifest`; mirror existing fabric_meta fixtures in that file):

```python
def test_write_manifest_uses_canonical_skeleton(tmp_path):
    from nhf_spatial_targets import validate as V
    from nhf_spatial_targets.release.lineage import (
        CURRENT_MANIFEST_SCHEMA_VERSION,
        _new_manifest_skeleton,
    )
    import json

    fabric_meta = {
        "path": "/data/fabric.gpkg",
        "sha256": "deadbeef",
        "crs": "EPSG:5070",
        "id_col": "nhru_v11",
        "id_col_sorted": True,
        "hru_count": 10,
    }
    V._write_manifest(tmp_path, fabric_meta)
    m = json.loads((tmp_path / "manifest.json").read_text())

    # Top-level shape matches the single canonical skeleton exactly.
    assert set(m) == set(_new_manifest_skeleton())
    assert m["manifest_schema_version"] == CURRENT_MANIFEST_SCHEMA_VERSION
    assert m["fabric"]["id_col"] == "nhru_v11"
    assert m["created_utc"] is not None  # validate fills the clock


def test_write_manifest_preserves_identity_on_rerun(tmp_path):
    from nhf_spatial_targets import validate as V
    import json

    fabric_meta = {
        "path": "/data/fabric.gpkg", "sha256": "a", "crs": "EPSG:5070",
        "id_col": "nhru_v11", "id_col_sorted": True, "hru_count": 10,
    }
    V._write_manifest(tmp_path, fabric_meta)
    first = json.loads((tmp_path / "manifest.json").read_text())
    # Simulate prior provenance + a source.
    m = first
    m["sources"]["era5_land"] = {"source_key": "era5_land"}
    m["steps"].append({"kind": "validate"})
    (tmp_path / "manifest.json").write_text(json.dumps(m))

    V._write_manifest(tmp_path, fabric_meta)
    second = json.loads((tmp_path / "manifest.json").read_text())
    assert second["created_utc"] == first["created_utc"]  # never re-minted
    assert "era5_land" in second["sources"]                # preserved
    assert len(second["steps"]) == 1                        # preserved
```

- [ ] **Step 2: Run — Expected FAIL:** current `_write_manifest` builds its own 6-key dict (no `manifest_schema_version`), so `set(m) != set(_new_manifest_skeleton())`.

Run: `pixi run -e dev pytest tests/test_validate.py -k write_manifest -v`

- [ ] **Step 3: Implement** — rewrite `_write_manifest` body (validate.py:536-577) to build on the shared skeleton + shared atomic writer. Add the import at the top of validate.py (near the other lineage import at validate.py:237):

```python
from nhf_spatial_targets.release.lineage import (
    _new_manifest_skeleton,
    atomic_write_manifest,
)
```

New body (keep the tolerant corrupt-manifest behaviour — validate warns and rebuilds, unlike `read_manifest` which raises):

```python
    now_utc = datetime.now(timezone.utc).isoformat()
    fabric_block = {
        "path": fabric_meta["path"],
        "sha256": fabric_meta["sha256"],
        "crs": fabric_meta["crs"],
        "id_col": fabric_meta["id_col"],
        "id_col_sorted": fabric_meta["id_col_sorted"],
        "hru_count": fabric_meta["hru_count"],
    }

    path = workdir / "manifest.json"
    preserved: dict = {}
    if path.exists():
        try:
            preserved = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning(
                "manifest.json in %s could not be parsed (%s); writing a "
                "fresh skeleton. Inspect the file manually if you need to "
                "recover any prior provenance.",
                workdir,
                exc,
            )
            preserved = {}

    manifest = _new_manifest_skeleton()  # stamps manifest_schema_version
    manifest["created_utc"] = preserved.get("created_utc") or now_utc
    manifest["last_validated_utc"] = now_utc
    manifest["nhf_spatial_targets_version"] = __version__
    manifest["fabric"] = fabric_block
    manifest["sources"] = preserved.get("sources") or {}
    manifest["steps"] = preserved.get("steps") or []

    atomic_write_manifest(path, manifest)
```

This removes the inline tempfile block (the shared `atomic_write_manifest` does the tempfile+rename). Drop now-unused `tempfile`/`os` imports only if nothing else in validate.py uses them — verify with grep before removing.

- [ ] **Step 4: Run — Expected PASS.** Also run the existing validate suite to confirm no regression:

Run: `pixi run -e dev pytest tests/test_validate.py -v`

- [ ] **Step 5: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/validate.py tests/test_validate.py
pixi run git commit -m "feature(manifest): validate writes via shared canonical skeleton (#279)"
```

### Task 1.4: `upgrade_manifest.py` report-only detector

**Files:**
- Create: `src/nhf_spatial_targets/upgrade_manifest.py`
- Test: `tests/test_upgrade_manifest.py`

- [ ] **Step 1: Write the failing test** — `tests/test_upgrade_manifest.py`:

```python
from __future__ import annotations

import json
import pytest

from nhf_spatial_targets.upgrade_manifest import check_manifest_schema


def _write(tmp_path, payload):
    (tmp_path / "manifest.json").write_text(json.dumps(payload))


def test_reports_behind_for_preversion(tmp_path):
    _write(tmp_path, {"sources": {}, "steps": []})  # no version key -> 0
    assert check_manifest_schema(tmp_path) == 0


def test_in_sync_returns_none(tmp_path):
    from nhf_spatial_targets.release.lineage import CURRENT_MANIFEST_SCHEMA_VERSION

    _write(tmp_path, {"manifest_schema_version": CURRENT_MANIFEST_SCHEMA_VERSION,
                      "sources": {}, "steps": []})
    assert check_manifest_schema(tmp_path) is None


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_manifest_schema(tmp_path)


def test_check_never_mutates(tmp_path):
    _write(tmp_path, {"sources": {}, "steps": []})
    p = tmp_path / "manifest.json"
    before = p.read_text()
    check_manifest_schema(tmp_path)
    assert p.read_text() == before  # report-only
```

- [ ] **Step 2: Run — Expected FAIL:** `ModuleNotFoundError: nhf_spatial_targets.upgrade_manifest`.

Run: `pixi run -e dev pytest tests/test_upgrade_manifest.py -v`

- [ ] **Step 3: Implement** — `src/nhf_spatial_targets/upgrade_manifest.py`:

```python
"""Report whether a project's manifest.json predates the current schema.

Mirrors :mod:`nhf_spatial_targets.upgrade_config`: a report-only operator
discovery path. ``nhf-targets upgrade-manifest -d <dir>`` detects a manifest
whose ``manifest_schema_version`` is behind
:data:`~nhf_spatial_targets.release.lineage.CURRENT_MANIFEST_SCHEMA_VERSION`
and prints what ``rebuild-manifest`` would normalize. **Never mutates.**
"""

from __future__ import annotations

import json
from pathlib import Path

from nhf_spatial_targets.release.lineage import CURRENT_MANIFEST_SCHEMA_VERSION


def check_manifest_schema(project_dir: Path) -> int | None:
    """Return the on-disk ``manifest_schema_version`` if behind current.

    Returns ``None`` when the manifest is at the current schema version (in
    sync). A manifest with no version key reads as ``0`` (pre-version).

    Raises
    ------
    FileNotFoundError
        If ``<project_dir>/manifest.json`` does not exist.
    ValueError
        If the manifest is present but unparseable (loud, never silent).
    """
    manifest_path = Path(project_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {project_dir}. Run "
            f"'nhf-targets validate -d {project_dir}' first."
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manifest.json at {manifest_path} is corrupt: {exc}"
        ) from exc
    version = manifest.get("manifest_schema_version", 0)
    return version if version < CURRENT_MANIFEST_SCHEMA_VERSION else None
```

- [ ] **Step 4: Run — Expected PASS.**

- [ ] **Step 5: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/upgrade_manifest.py tests/test_upgrade_manifest.py
pixi run git commit -m "feature(manifest): report-only upgrade-manifest detector (#279)"
```

### Task 1.5: Wire `upgrade-manifest` CLI command

**Files:**
- Modify: `src/nhf_spatial_targets/cli/run.py` (add `upgrade_manifest_cmd`, register it)
- Modify: `src/nhf_spatial_targets/cli/__init__.py` (re-export for tests)
- Test: `tests/test_upgrade_manifest.py` (CLI-level)

- [ ] **Step 1: Write the failing test** — append a CLI test that invokes the command function directly and asserts exit codes:

```python
def test_cli_exits_1_on_drift(tmp_path, capsys):
    from nhf_spatial_targets.cli.run import upgrade_manifest_cmd

    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    with pytest.raises(SystemExit) as e:
        upgrade_manifest_cmd(tmp_path)
    assert e.value.code == 1


def test_cli_exits_0_in_sync(tmp_path):
    from nhf_spatial_targets.cli.run import upgrade_manifest_cmd
    from nhf_spatial_targets.release.lineage import CURRENT_MANIFEST_SCHEMA_VERSION

    (tmp_path / "manifest.json").write_text(
        json.dumps({"manifest_schema_version": CURRENT_MANIFEST_SCHEMA_VERSION,
                    "sources": {}, "steps": []})
    )
    # No SystemExit on the in-sync path (returns normally / exit 0).
    upgrade_manifest_cmd(tmp_path)
```

- [ ] **Step 2: Run — Expected FAIL:** `ImportError: cannot import name 'upgrade_manifest_cmd'`.

- [ ] **Step 3: Implement** — in `cli/run.py`, add the command (model on `upgrade_config_cmd`, run.py:313-371) and register it. Add to `register()` (after run.py:32):

```python
    app.command(upgrade_manifest_cmd, name="upgrade-manifest")
```

Add the function:

```python
def upgrade_manifest_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir", "-d"],
            help="Project created by 'nhf-targets init'.",
        ),
    ],
):
    """Report whether manifest.json predates the current manifest schema.

    Report-only: never mutates the manifest. Exits 0 if current, 1 if behind
    (so scripted heartbeats detect drift). To normalize a behind manifest, run
    'nhf-targets rebuild-manifest -d <dir>'.
    """
    from rich.console import Console

    from nhf_spatial_targets.release.lineage import CURRENT_MANIFEST_SCHEMA_VERSION
    from nhf_spatial_targets.upgrade_manifest import check_manifest_schema

    console = Console()
    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)
    try:
        behind = check_manifest_schema(workdir)
    except (FileNotFoundError, ValueError) as e:
        print(f"upgrade-manifest failed: {e}", file=sys.stderr)
        sys.exit(1)

    if behind is None:
        console.print(
            "[bold green]manifest.json is at the current schema version "
            f"({CURRENT_MANIFEST_SCHEMA_VERSION}).[/bold green]"
        )
        return

    console.print(
        f"[bold yellow]manifest.json is schema version {behind}; current is "
        f"{CURRENT_MANIFEST_SCHEMA_VERSION}.[/bold yellow]\n\n"
        "Run 'nhf-targets rebuild-manifest -d <dir>' to regenerate it as a "
        "complete, version-stamped projection of the on-disk artifacts. "
        "This command never edits your manifest."
    )
    sys.exit(1)
```

In `cli/__init__.py`, add `upgrade_manifest_cmd` to the `from nhf_spatial_targets.cli.run import (...)` block (after `upgrade_config_cmd`, __init__.py:83) and to `__all__` (after `"upgrade_config_cmd",` __init__.py:110).

- [ ] **Step 4: Run — Expected PASS.** Smoke the CLI wiring:

Run: `pixi run nhf-targets upgrade-manifest --help` (expect the help text, no traceback).

- [ ] **Step 5: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/cli/run.py src/nhf_spatial_targets/cli/__init__.py tests/test_upgrade_manifest.py
pixi run git commit -m "feature(manifest): wire nhf-targets upgrade-manifest command (#279)"
```

### Task 1.6: PR-1 self-review + open PR

- [ ] **Step 1:** Run the targeted suites:

```bash
pixi run -e dev fmt-check && pixi run -e dev lint
pixi run -e dev pytest tests/test_release_lineage.py tests/test_validate.py tests/test_upgrade_manifest.py -v
```

- [ ] **Step 2:** Confirm skeleton parity (the PR-1 acceptance test from the spec): `validate`-written and lineage-skeleton top-levels are identical (covered by Task 1.3 test).
- [ ] **Step 3:** Push, open PR referencing `#279`, **stop for maintainer review** (do not start PR-2 until approved).

```bash
git push -u origin feature/279-pr1-skeleton
gh pr create --base feature/279-durable-artifacts --title "PR-1: manifest schema version + single skeleton + upgrade-manifest (#279)" \
  --body "First slice of #279 (stacked; base = feature/279-durable-artifacts). Unifies the two manifest skeletons, stamps manifest_schema_version (start 1), read_manifest defaults pre-version manifests to 0, adds report-only nhf-targets upgrade-manifest. See docs/superpowers/plans/2026-05-29-durable-manifest-and-config.md."
```

---

# PR-2 — `rebuild-manifest`: the one authoritative projection

**Goal:** A generic, catalog-keyed, deterministic `rebuild-manifest` that projects (datastore consolidated dirs ∩ catalog) ∪ project `data/aggregated/` dirs ∪ `targets/` NCs ∪ `fabric.json` into a complete manifest — byte-identical on re-run, no `datetime.now()` in the path. Subsumes `reconcile`, `rebuild_lineage`, #277 backfill, #278 consolidate-walk.

**Architecture:** `rebuild_manifest(project, *, compute_sha256=False, dry_run=False) -> _Manifest` lives in a new `src/nhf_spatial_targets/rebuild_manifest.py`. It (a) enumerates source dirs from datastore + `data/aggregated/`, (b) builds `sources[]` entries with catalog metadata by key + on-disk file lists/periods/sizes/mtimes, (c) synthesizes `steps[]` deterministically, (d) read-merges identity fields from the existing manifest, (e) writes via `lineage.atomic_write_manifest` under flock — unless `dry_run`, which returns the projection without writing.

### Task 2.1: `mtime`-only timestamp helper + JSON-ordering helpers in lineage.py

**Files:**
- Modify: `src/nhf_spatial_targets/release/lineage.py`
- Test: `tests/test_release_lineage.py`

- [ ] **Step 1: Write the failing test:**

```python
def test_iso_from_mtime_is_deterministic(tmp_path):
    from nhf_spatial_targets.release.lineage import iso_from_mtime

    f = tmp_path / "x.nc"
    f.write_bytes(b"data")
    import os
    os.utime(f, (1_600_000_000, 1_600_000_000))
    a = iso_from_mtime(f)
    b = iso_from_mtime(f)
    assert a == b  # pure function of mtime, no now()
    assert a.startswith("2020-")  # 2020-09-13T... UTC


def test_step_sort_key_orders_by_kind_then_source_then_path():
    from nhf_spatial_targets.release.lineage import step_sort_key

    consolidate = {"kind": "consolidate", "source_key": "merra2",
                   "outputs": [{"path": "/d/merra2/2000.nc"}]}
    aggregate = {"kind": "aggregate", "source_key": "merra2",
                 "outputs": [{"path": "/a/merra2_agg.nc"}]}
    assert step_sort_key(consolidate) < step_sort_key(aggregate)
```

- [ ] **Step 2: Run — Expected FAIL** (`ImportError`).

- [ ] **Step 3: Implement** — add to `lineage.py`:

```python
# Deterministic step ordering: consolidate < aggregate < target < validate
# (fetch/nn_fill slot in their pipeline positions). The rebuild projection
# sorts steps by this rank, then source_key, then first output path, so the
# same disk produces a byte-identical steps[] every run.
_STEP_KIND_RANK: dict[str, int] = {
    "fetch": 0,
    "consolidate": 1,
    "aggregate": 2,
    "nn_fill": 3,
    "target": 4,
    "validate": 5,
}


def iso_from_mtime(path: Path) -> str:
    """Return *path*'s mtime as a UTC ISO-8601 string. No ``datetime.now()``.

    This is the ONLY timestamp source permitted in the rebuild projection
    path (spec decision E); it makes the projection a pure function of disk.
    """
    st = path.stat()
    return datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()


def step_sort_key(step: dict) -> tuple[int, str, str]:
    """Deterministic sort key for a step record."""
    first_out = ""
    outs = step.get("outputs") or []
    if outs:
        first_out = min(str(o.get("path", "")) for o in outs)
    return (
        _STEP_KIND_RANK.get(step.get("kind", ""), 99),
        step.get("source_key") or "",
        first_out,
    )
```

- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit** (`pixi run git commit -m "feature(manifest): mtime + deterministic-ordering helpers for rebuild (#279)"`).

### Task 2.2: Filename → (source_key, year) parse helper

**Files:**
- Create helper in `src/nhf_spatial_targets/rebuild_manifest.py`
- Test: `tests/test_rebuild_manifest.py`

- [ ] **Step 1: Write the failing test** — must cover the spec-named cases:

```python
import pytest
from nhf_spatial_targets.rebuild_manifest import parse_aggregated_filename


@pytest.mark.parametrize("name,year", [
    ("ssebop_2000_agg.nc", 2000),
    ("daymet_na_1980_agg.nc", 1980),
    ("mod10c1_v061_2020_agg.nc", 2020),
    ("merra2_agg.nc", None),       # single-shot, no year
    ("snodas_agg.nc", None),
])
def test_parse_year(name, year):
    assert parse_aggregated_filename(name)[1] == year
```

- [ ] **Step 2: Run — Expected FAIL** (module/function missing).

- [ ] **Step 3: Implement** — in `rebuild_manifest.py`:

```python
import re

# Aggregated NCs end in `_agg.nc`; an optional 4-digit year may precede it.
# `<key>_agg.nc` | `<key>_<year>_agg.nc` | `<key>_<region>_<year>_agg.nc`.
_AGG_YEAR_RE = re.compile(r"_(?P<year>\d{4})_agg\.nc$")


def parse_aggregated_filename(name: str) -> tuple[str, int | None]:
    """Return ``(stem, year)`` for an aggregated NC filename.

    ``stem`` is the filename with the trailing ``_agg.nc`` (and year, if any)
    stripped — diagnostic only; the authoritative source key is the parent
    directory name. ``year`` is ``None`` for single-shot aggregates.
    """
    m = _AGG_YEAR_RE.search(name)
    if m:
        year = int(m.group("year"))
        stem = name[: m.start()]
        return stem, year
    stem = name[:-len("_agg.nc")] if name.endswith("_agg.nc") else name
    return stem, None
```

- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit.**

### Task 2.3: Build one `sources[]` entry from a directory (catalog-keyed)

**Files:**
- Modify: `src/nhf_spatial_targets/rebuild_manifest.py`
- Test: `tests/test_rebuild_manifest.py`

**Design contract:**
- Source key = directory name under `<datastore>/` (consolidated) or `<project>/data/aggregated/` (aggregated).
- If the key is in `catalog.sources()`: pull `access_type`, `doi`/`version`/`url` (whatever the catalog carries), set `derived_variant: False`.
- If not (e.g. `era5_land_sd`): minimal entry, `derived_variant: True` — never orphan a shipped NC.
- Each entry carries `provenance: "reconstructed"`, a sorted `files[]` (path/size/mtime, sha256 only when `compute_sha256`), and a derived `period` from parsed years where present.

- [ ] **Step 1: Write the failing test:**

```python
def test_source_entry_catalog_key(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import build_source_entry

    d = tmp_path / "merra2"
    d.mkdir()
    (d / "merra2_2000.nc").write_bytes(b"x")
    entry = build_source_entry("merra2", d, compute_sha256=False)
    assert entry["source_key"] == "merra2"
    assert entry["provenance"] == "reconstructed"
    assert entry["derived_variant"] is False
    assert len(entry["files"]) == 1
    assert "sha256" not in entry["files"][0]  # opt-in only


def test_source_entry_noncatalog_is_derived_variant(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import build_source_entry

    d = tmp_path / "era5_land_sd"
    d.mkdir()
    (d / "era5_land_sd_agg.nc").write_bytes(b"x")
    entry = build_source_entry("era5_land_sd", d, compute_sha256=False)
    assert entry["derived_variant"] is True
```

- [ ] **Step 2: Run — Expected FAIL.**

- [ ] **Step 3: Implement** `build_source_entry(source_key, source_dir, *, compute_sha256)`:
  - `from nhf_spatial_targets import catalog as _catalog`.
  - `is_catalog = source_key in _catalog.sources()`.
  - Glob `*.nc` recursively, sort paths, build `files[]` via `lineage._file_basics` + optional `lineage.sha256_file`.
  - When `is_catalog`, copy a fixed allowlist of catalog metadata (`access`, `doi`, `version`, `access_type` — guard each with `.get`).
  - Derive `period` from parsed years (`parse_aggregated_filename` for aggregated; for consolidated use a year regex on the filename) → `"<min>/<max>"` when any years found.
  - Return dict with `source_key`, `provenance: "reconstructed"`, `derived_variant`, `files`, optional `period`, catalog metadata.

- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit.**

### Task 2.4: Synthesize `steps[]` deterministically

**Files:**
- Modify: `src/nhf_spatial_targets/rebuild_manifest.py`
- Test: `tests/test_rebuild_manifest.py`

**Design contract** — one step per (kind, source/output group), `timestamp_utc = iso_from_mtime(first output)`, `software_version` from the manifest's existing value or `_SOFTWARE_VERSION`, `provenance: "reconstructed"`, sorted via `lineage.step_sort_key`:
- `consolidate` — one per datastore source dir, outputs = its NCs.
- `aggregate` — one per `data/aggregated/<key>/` dir, outputs = its `_agg.nc` files.
- `target` — one per `targets/*.nc`, `source_key=None`, params from NC attrs (PR-7 enriches; PR-2 reads what exists: `period`, `sources` if present).
- `validate` — one for `fabric.json`, `source_key=None`.

- [ ] **Step 1: Write the failing test:**

```python
def test_synthesize_steps_sorted_and_kinds(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import synthesize_steps

    # minimal synthetic tree
    ds = tmp_path / "datastore" / "merra2"; ds.mkdir(parents=True)
    (ds / "merra2_2000.nc").write_bytes(b"x")
    agg = tmp_path / "proj" / "data" / "aggregated" / "merra2"; agg.mkdir(parents=True)
    (agg / "merra2_agg.nc").write_bytes(b"x")
    tgt = tmp_path / "proj" / "targets"; tgt.mkdir(parents=True)
    (tgt / "aet_targets.nc").write_bytes(b"x")
    (tmp_path / "proj" / "fabric.json").write_text("{}")

    steps = synthesize_steps(
        datastore=tmp_path / "datastore",
        project_dir=tmp_path / "proj",
        compute_sha256=False,
    )
    kinds = [s["kind"] for s in steps]
    assert kinds == sorted(kinds, key=lambda k: ["consolidate","aggregate","target","validate"].index(k))
    assert all(s["provenance"] == "reconstructed" for s in steps)
```

- [ ] **Step 2: Run — Expected FAIL.**
- [ ] **Step 3: Implement** `synthesize_steps(...)` building the four kinds, each step via `lineage.build_step_record(... timestamp_utc=iso_from_mtime(first_output))` then setting `record["provenance"] = "reconstructed"`, finally `sorted(steps, key=lineage.step_sort_key)`. (Note: `build_step_record` raises if `timestamp_utc is None` only via `now()` — always pass `iso_from_mtime`, never `None`, to keep `now()` off the path.)
- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit.**

### Task 2.5: Top-level `rebuild_manifest` (assemble + read-merge + write/dry-run)

**Files:**
- Modify: `src/nhf_spatial_targets/rebuild_manifest.py`
- Test: `tests/test_rebuild_manifest.py`

**Design contract:**
```python
def rebuild_manifest(project, *, compute_sha256=False, dry_run=False) -> _Manifest
```
- `sources` = `{key: build_source_entry(...)}` for sorted union of datastore dirs ∩ catalog (catalog dirs) and all `data/aggregated/` dirs; non-catalog dirs included as `derived_variant`.
- `steps` = `synthesize_steps(...)`.
- Read existing manifest via `lineage.read_manifest`; **read-merge identity fields**: keep its `created_utc`, `fabric`, and any `release`/authorship blocks; set `manifest_schema_version = CURRENT_MANIFEST_SCHEMA_VERSION`; `last_validated_utc` is **not** re-minted (left as the existing value, or `None` — never `now()`).
- If `dry_run`: return the projection, write nothing.
- Else: write via `lineage.with_flock` + `lineage.atomic_write_manifest`.

- [ ] **Step 1: Write the failing tests (the idempotency contract is the headline):**

```python
def _make_tree(tmp_path):
    # datastore + aggregated + targets + fabric.json (reuse Task 2.4 layout)
    ...

def test_rebuild_is_byte_identical_on_rerun(tmp_path, monkeypatch):
    import json
    from nhf_spatial_targets.rebuild_manifest import rebuild_manifest
    project = _make_project(tmp_path)  # workspace.Project pointing at the tree
    m1 = rebuild_manifest(project, dry_run=True)
    m2 = rebuild_manifest(project, dry_run=True)
    assert json.dumps(m1, indent=2) == json.dumps(m2, indent=2)


def test_rebuild_preserves_created_utc_and_fabric(tmp_path):
    # Seed an existing manifest with created_utc + fabric authorship.
    # Assert rebuild keeps them verbatim.
    ...


def test_rebuild_includes_derived_variant(tmp_path):
    # era5_land_sd aggregated dir -> entry with derived_variant True.
    ...


def test_rebuild_records_nonpublishable_source(tmp_path):
    # watergap22d aggregated dir present -> recorded in sources[].
    ...
```

- [ ] **Step 2: Run — Expected FAIL.**
- [ ] **Step 3: Implement** `rebuild_manifest`. Use `project.datastore`, `project.aggregated_dir()`, `project.workdir`, `project.manifest_path`.
- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit.**

### Task 2.6: Guard against `datetime.now()` in the rebuild path

**Files:**
- Test: `tests/test_rebuild_manifest.py`

- [ ] **Step 1: Write the test** — import-source lint guard (cheap, no monkeypatch fragility):

```python
def test_no_datetime_now_in_rebuild_module():
    import inspect
    from nhf_spatial_targets import rebuild_manifest as rm

    src = inspect.getsource(rm)
    assert "datetime.now" not in src, (
        "rebuild_manifest must derive all timestamps from file mtime "
        "(spec decision E). Use lineage.iso_from_mtime."
    )
```

Additionally add a frozen-clock behavioural guard: monkeypatch `nhf_spatial_targets.release.lineage.datetime` to raise if `.now` is called during a `rebuild_manifest(..., dry_run=True)` run, asserting no exception.

- [ ] **Step 2-4:** Run; confirm PASS (rebuild module already mtime-only by construction).
- [ ] **Step 5: Commit.**

### Task 2.7: Wire `rebuild-manifest` CLI + DELETE `reconcile-manifest` outright

**Decision (resolved 2026-05-29):** `reconcile-manifest` is **deleted**, not shimmed. `rebuild-manifest` is generic and catalog-keyed, so the per-source reconcile hook registry (`_RECONCILERS`, the `era5_land`/`mod16a2_v061` fetch-module hooks) is exactly the `no-hook` design being eliminated — there is nothing to preserve.

**Files:**
- Modify: `src/nhf_spatial_targets/cli/run.py` (add `rebuild_manifest_cmd` + register `rebuild-manifest`; **remove** `reconcile_manifest_cmd` and its registration in `register()`, run.py:31)
- Modify: `src/nhf_spatial_targets/cli/__init__.py` (re-export `rebuild_manifest_cmd`; **remove** `reconcile_manifest_cmd` from the import block + `__all__`, __init__.py:79/108)
- Delete: `src/nhf_spatial_targets/reconcile.py`
- Modify: `src/nhf_spatial_targets/fetch/era5_land.py` (remove the now-orphaned `reconcile` hook), `src/nhf_spatial_targets/fetch/modis.py` (remove `reconcile_mod16a2`) — **only after** `grep -rn "reconcile" src/ tests/` confirms `reconcile.py:_RECONCILERS` was the sole caller
- Modify: `src/nhf_spatial_targets/release/rebuild.py` (`rebuild_lineage` → delegate to `synthesize_steps`, or remove and repoint callers)
- Delete: `tests/test_reconcile_manifest.py`, `tests/test_reconcile_era5_land.py`, `tests/test_reconcile_modis.py`
- Modify/Delete: `docs/architecture/reconcile-manifest.md` → replace body with a one-paragraph redirect to `rebuild-manifest` (or delete and add a note in the rebuild docs)

- [ ] **Step 1: Write the failing test** — in `tests/test_rebuild_manifest.py`: `rebuild_manifest_cmd` exists, `--dry-run` writes nothing, default writes a complete manifest. Add a guard test that `nhf_spatial_targets.reconcile` no longer imports and `reconcile-manifest` is not a registered command:

```python
def test_reconcile_manifest_is_removed():
    import importlib
    import pytest

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("nhf_spatial_targets.reconcile")
    from nhf_spatial_targets.cli import app
    assert "reconcile-manifest" not in app  # cyclopts: command name not registered
```

(Confirm the cyclopts membership idiom against the version in use; if `in` is unsupported, assert the command is absent from `app._commands` or invoke `app(["reconcile-manifest", ...])` and assert it errors as unknown.)

- [ ] **Step 2: Run — Expected FAIL** (`reconcile` still imports; command still registered).
- [ ] **Step 3: Implement:**
  - Add `rebuild_manifest_cmd(workdir, *, compute_sha256=False, dry_run=False)` modeled on the old `reconcile_manifest_cmd` shape (Rich summary table: sources count, steps count, derived-variant count); register `app.command(rebuild_manifest_cmd, name="rebuild-manifest")` in `register()`.
  - Remove `reconcile_manifest_cmd` (function + `register()` line) from `cli/run.py`; remove its import + `__all__` entry from `cli/__init__.py`. Add `rebuild_manifest_cmd` to both.
  - `git rm src/nhf_spatial_targets/reconcile.py`.
  - After grep-confirming no other callers, remove the fetch-module reconcile hooks and `git rm` the three reconcile test files.
  - In `release/rebuild.py`: have `rebuild_lineage` delegate to `synthesize_steps` (or remove + repoint). Verify `release/publish.py` / `release/__init__.py` callers still resolve (`grep -rn rebuild_lineage src/`).
- [ ] **Step 4: Run — Expected PASS.** Smoke: `pixi run nhf-targets rebuild-manifest --help`; `pixi run nhf-targets reconcile-manifest --help` should now error (unknown command). Run the broader CLI test module to catch a stale `reconcile` reference.
- [ ] **Step 5: Commit.**

### Task 2.8: Smoke test against real projects (offline, no ScienceBase)

- [ ] **Step 1:** Run against the two real on-disk projects:

```bash
pixi run nhf-targets rebuild-manifest -d /caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets --dry-run
pixi run nhf-targets rebuild-manifest -d /caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets --dry-run
```

- [ ] **Step 2: Assert (manually inspect the dry-run output):** OR gains all **16** aggregated sources incl. `era5_land_sd` (as `derived_variant`); `steps[]` non-empty with `consolidate`/`aggregate`/`target`/`validate` kinds. Re-run `--dry-run` twice and `diff` the JSON — must be byte-identical.

```bash
pixi run nhf-targets rebuild-manifest -d <or> --dry-run > /tmp/or1.json
pixi run nhf-targets rebuild-manifest -d <or> --dry-run > /tmp/or2.json
diff /tmp/or1.json /tmp/or2.json && echo "BYTE-IDENTICAL"
```

- [ ] **Step 3:** Only after dry-run looks correct, run **without** `--dry-run` is the operator's call — do not mutate the real OR/gfv2 manifests as part of CI; note the smoke result in the PR description.
- [ ] **Step 4: PR-2 self-review + open PR; stop for review.**

---

# PR-3 — Completeness gate at publish (verify, don't mutate)

**Goal:** Generalize the (unbuilt #277) preflight into a full per-fabric precondition in `release/publish.py` that runs `rebuild-manifest --dry-run`, compares to the on-disk manifest, and refuses on drift/incompleteness — plus harden `stage_fabric_child` to a hard error when the manifest is missing.

### Task 3.1: `_preflight_provenance_complete` gate

**Files:**
- Modify: `src/nhf_spatial_targets/release/publish.py`
- Test: `tests/test_release_publish.py`

**Design contract** — `_preflight_provenance_complete(project, *, allow_incomplete_sources=False)`:
- Run `rebuild_manifest(project, dry_run=True)` → projection.
- Read on-disk manifest via `lineage.read_manifest`.
- Assert: `manifest_schema_version` current; `fabric` present (non-None); `steps[]` non-empty; `sources[]` ⊇ aggregated dirs **and** ⊇ consolidated datastore sources used; every published target NC has a matching `target` step; on-disk manifest `sources`/`steps` **equal** the projection's (the drift check).
- On any failure → `PreflightError` whose message tells the operator to run `rebuild-manifest`.
- `allow_incomplete_sources=True` downgrades the source-completeness + drift assertions to a logged warning (the deliberate override). Schema/fabric/steps-empty remain fatal.
- Thread `allow_incomplete_sources` from the publish entrypoint(s) (a new CLI flag `--allow-incomplete-sources` on the release/publish command).

- [ ] **Step 1: Write the failing tests** (mirror the spec's PR-3 test list):

```python
def test_under_report_aggregate_raises(...):     # missing aggregated source -> PreflightError
def test_under_report_consolidate_raises(...):    # missing consolidate source -> PreflightError
def test_drift_between_disk_and_projection_refuses(...):
def test_allow_incomplete_sources_bypasses(...):  # warns, does not raise
def test_complete_manifest_passes(...):
```

Compare projection vs on-disk by JSON-normalizing `sources` + `steps`.

- [ ] **Step 2: Run — Expected FAIL.**
- [ ] **Step 3: Implement** the gate; call it from `_preflight_common` (or the scope-specific preflight) so every publish scope is gated. Add the CLI flag in `cli/release.py` (locate the publish subcommand) threaded to `allow_incomplete_sources`.
- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit.**

### Task 3.2: Harden `stage_fabric_child` missing-manifest skip → hard error

**Files:**
- Modify: `src/nhf_spatial_targets/release/payload.py:191-192`
- Test: `tests/test_release_payload.py`

- [ ] **Step 1: Write the failing test:**

```python
def test_stage_fabric_child_missing_manifest_is_hard_error(tmp_path, ...):
    # plan.manifest_src does not exist
    with pytest.raises(<ReleaseError or PreflightError>):
        stage_fabric_child(project, copy=True)
```

- [ ] **Step 2: Run — Expected FAIL** (current code silently skips).
- [ ] **Step 3: Implement** — replace the `if plan.manifest_src.exists():` soft skip with:

```python
    if not plan.manifest_src.exists():
        raise ReleaseError(
            f"Cannot stage fabric child: manifest.json missing at "
            f"{plan.manifest_src}. Run 'nhf-targets validate' then "
            f"'nhf-targets rebuild-manifest' for this project before publishing."
        )
    _copy_file(plan.manifest_src, plan.stage_dir / "manifest.json")
```

(Use whichever exception `payload.py` already imports — `ReleaseError` from `release.publish`, or define a local equivalent; check the import graph to avoid a cycle.)

- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit + PR-3 self-review; stop for review.**

---

# PR-4 — `config.effective.yml` as a regenerable, version-stamped projection

**Goal:** Make `config.effective.yml` a version/hash-stamped projection of (`config.yml` × `defaults.py` × `fabric.json`), regenerated by `validate`, staleness-gated at publish with the same verify-don't-mutate mechanism, and actually parsed (not just listed as an output) by `rebuild.py`.

### Task 4.1: Stamp `config.effective.yml` with schema version + source hash

**Files:**
- Modify: `src/nhf_spatial_targets/validate.py:77-102` (`_write_effective_config`)
- Test: `tests/test_validate.py` / new `tests/test_effective_config.py`

**Design contract:** Add two top-level keys to the emitted YAML mapping (under a dedicated `_meta` block to avoid colliding with config keys):

```yaml
_effective_config_meta:
  effective_config_schema_version: 1
  source_config_sha256: <sha256 of the verbatim config.yml bytes>
```

Compute the hash from the raw `config.yml` text (not the merged dict) so staleness = "config.yml changed since this was generated".

- [ ] **Step 1: Write the failing test:** assert the regenerated file contains `_effective_config_meta.effective_config_schema_version == 1` and a `source_config_sha256` equal to `hashlib.sha256(config_yml_bytes).hexdigest()`.
- [ ] **Step 2: Run — Expected FAIL.**
- [ ] **Step 3: Implement** — `_write_effective_config(workdir, merged)` reads `config.yml` bytes, computes the hash, injects the `_effective_config_meta` block into the dumped mapping (insert into a copy so the live `merged` dict used elsewhere is untouched). Provide a module constant `EFFECTIVE_CONFIG_SCHEMA_VERSION = 1`.
- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit.**

### Task 4.2: Staleness gate at publish

**Files:**
- Modify: `src/nhf_spatial_targets/release/publish.py`
- Test: `tests/test_release_publish.py`

**Design contract** — `_preflight_effective_config_current(project)`:
- Read `config.effective.yml`; extract `_effective_config_meta.source_config_sha256`.
- Recompute the hash of the current `config.yml`.
- If they differ (or the meta block is absent / schema behind) → `PreflightError("config.effective.yml is stale; re-run nhf-targets validate")`.
- Verify-don't-mutate: the gate never regenerates the file.

- [ ] **Step 1: Write the failing tests:** stale effective config (hash mismatch) → `PreflightError`; matching hash → passes; missing meta → `PreflightError`.
- [ ] **Step 2: Run — Expected FAIL.**
- [ ] **Step 3: Implement** and call from the publish preflight.
- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit.**

### Task 4.3: `rebuild.py` parses effective config (not just lists it)

**Files:**
- Modify: `src/nhf_spatial_targets/release/rebuild.py:408-410`
- Test: `tests/test_release_rebuild.py`

- [ ] **Step 1: Write the failing test:** the release-lineage input now reflects the *current* re-merged config values (e.g. the runoff `period` from `config.yml`), not a fossil window.
- [ ] **Step 2: Run — Expected FAIL.**
- [ ] **Step 3: Implement** — parse `config.effective.yml` (strip the `_effective_config_meta` block before consuming) and feed resolved per-target params into the lineage input.
- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit + PR-4 self-review; stop for review.**

### Task 4.4: Config-schema-additions follow-through (CLAUDE.md checklist)

- [ ] Per CLAUDE.md "Config schema additions": `config.effective.yml` is *derived*, not an operator config key, so the four-point checklist (init template / pipeline.yml / test_init_run / upgrade_config) does **not** apply to `_effective_config_meta`. Add a one-line note in the PR description confirming this is intentional. (No code change.)

---

# PR-5 — `config.yml` catalog-key validation + `upgrade-config` extension

**Goal:** `validate` fails loudly on a `targets.*.sources[]` key absent from `sources.yml` (with a superseded-key hint); `upgrade-config` surfaces whole-new-target additions and "new catalog source available for target X" hints.

### Task 5.1: Loud catalog-key validation of `targets.*.sources[]`

**Files:**
- Modify: `src/nhf_spatial_targets/validate.py` (extend `_check_catalog_consistency`, validate.py:403-413)
- Test: `tests/test_validate.py`

- [ ] **Step 1: Write the failing test:**

```python
def test_dangling_target_source_key_fails_loudly(tmp_path, monkeypatch):
    # config.yml with targets.aet.sources: [merra_land]  (superseded -> merra2)
    # validate must raise with a hint naming the superseded_by replacement.
    ...
```

- [ ] **Step 2: Run — Expected FAIL** (current check only validates `variables.yml`).
- [ ] **Step 3: Implement** — after the existing variables check, iterate the merged config `targets`:

```python
    src_keys = set(sources().keys())
    for tgt_name, tgt in (config.get("targets") or {}).items():
        for key in (tgt.get("sources") or []):
            if key not in src_keys:
                hint = _superseded_hint(key)  # look up superseded_by across catalog
                raise ValueError(
                    f"Target '{tgt_name}' config.yml sources[] references "
                    f"'{key}', absent from catalog/sources.yml.{hint}"
                )
```

`_superseded_hint(key)` scans `sources()` for an entry whose `superseded_by == key`'s old name, or checks a known-rename map, returning `" Did you mean '<new>' (superseded_by)?"` or `""`. (Validate already loads the merged `config`; thread it into `_check_catalog_consistency` if not already in scope.)

- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit.**

### Task 5.2: `upgrade-config` surfaces whole-target + new-source hints

**Files:**
- Modify: `src/nhf_spatial_targets/upgrade_config.py`
- Test: `tests/test_upgrade_config.py`

**Design contract:** Extend beyond `OPTIONAL_CONFIG_FEATURES`:
- `check_missing_targets(project_dir) -> list[str]` — targets in `DEFAULTS["targets"]` whose key is absent from the operator's `config.yml` `targets:` mapping.
- `check_available_sources(project_dir) -> dict[str, list[str]]` — per target, catalog sources whose `variables.yml` mapping makes them eligible for that target but which are absent from the operator's `targets.<t>.sources[]`. Report-only hint, not a failure.
- Surface both in `upgrade_config_cmd` output (new Rich tables) without changing its report-only contract.

- [ ] **Step 1: Write the failing tests:** a config missing the `swe` target → reported; a config whose `aet.sources` omits an eligible catalog source → hint listed.
- [ ] **Step 2: Run — Expected FAIL.**
- [ ] **Step 3: Implement** the two functions + CLI rendering.
- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit + PR-5 self-review; stop for review.**

---

# PR-6 — CLAUDE.md "Manifest & config durability" section

**Goal:** Codify the intent-vs-derived principle, the determinism rules, the "when you add a source/variable/target/stage" checklist, and the per-fabric/per-release guarantee — modeled on the existing "Config schema additions" checklist.

### Task 6.1: Add the CLAUDE.md section

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1:** Add a new top-level section **"## Manifest & config durability"** after "Config schema additions", containing:
  1. The intent-vs-derived table + the manifest invariant (`manifest.json = deterministic projection of (disk × catalog × fabric.json)`).
  2. Determinism rules: no `datetime.now()` in the rebuild path (use `lineage.iso_from_mtime`); mtime timestamps; tag `provenance: "reconstructed"`; deterministic JSON ordering (sorted source keys; steps by `lineage.step_sort_key`).
  3. **When you add a source/variable/target/stage:** extend the `rebuild_manifest` projection **and its test** in the same PR; bump `CURRENT_MANIFEST_SCHEMA_VERSION` only if the manifest *shape* changes; never write a manifest or effective config that bypasses the shared skeleton/projection.
  4. Every fabric gets a manifest (`validate`); every release child carries a complete one (`stage_fabric_child`, hard-errors if absent); publish is gated on completeness (`_preflight_provenance_complete`); `config.effective.yml` is staleness-gated.
- [ ] **Step 2:** Docs-only — skip the pytest suite (CLAUDE.md Test Execution Discipline). Run `pixi run -e dev fmt-check` (no-op for md) and a manual read-through.
- [ ] **Step 3: Commit** (`docs(claude): manifest & config durability section (#279)`); PR-6 self-review; stop for review.

---

# PR-7 (optional) — Config ↔ product ↔ manifest triangle

**Goal:** Persist the resolved per-target params (`period`, `sources`, `normalize_period`, `ci_threshold`, `range_method`) as target-NC global attrs so the deterministic projection can read them back into the `target` step; surface config/NC/manifest inconsistency at publish.

### Task 7.1: Persist resolved params as target-NC global attrs

**Files:**
- Modify: `src/nhf_spatial_targets/targets/_driver.py` (`_common_global_attrs`) and/or `_writers.py` (`write_target_nc`)
- Test: existing per-target tests + a focused attr test

**Design contract:** Extend `_common_global_attrs` to include the resolved params. Lists (e.g. `sources`) must be persisted NC-attr-safely — join to a comma-separated string or a JSON string (match whatever the existing `source` attr convention is; the deck-annotation memory notes target NCs already carry a `source` attr + `n_sources`). Keys: `period` (exists), `sources` (resolved list), `normalize_period` (rch/som), `ci_threshold` (sca), `range_method`.

- [ ] **Step 1: Write the failing test:** build a target NC in a tmp project, assert `ds.attrs` contains the resolved `range_method`/`sources`/`period` (+ `normalize_period`/`ci_threshold` where the target defines them).
- [ ] **Step 2: Run — Expected FAIL.**
- [ ] **Step 3: Implement** — add the attrs in `_common_global_attrs`; read each target's resolved config from `project.target(name)`. Use the catalog units-truth rule (read source metadata from catalog, not NC attrs).
- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit.**

### Task 7.2: `rebuild_manifest` reads resolved params into `target` steps

**Files:**
- Modify: `src/nhf_spatial_targets/rebuild_manifest.py` (`synthesize_steps` target branch)
- Test: `tests/test_rebuild_manifest.py`

- [ ] **Step 1: Write the failing test:** a `target` step's `params` reflect the NC's resolved-param attrs (read back, not re-minted).
- [ ] **Step 2: Run — Expected FAIL.**
- [ ] **Step 3: Implement** — in the target-step branch, open each target NC (xarray, attrs only — `xr.open_dataset(..., decode_times=False)` then read `.attrs`) and populate `params`. Keep determinism: attrs are on-disk, mtime drives the timestamp.
- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit.**

### Task 7.3: Consistency check at publish

**Files:**
- Modify: `src/nhf_spatial_targets/release/publish.py`
- Test: `tests/test_release_publish.py`

- [ ] **Step 1: Write the failing test:** a `config.effective.yml` saying `period: 2000-2010` while the target NC/manifest say `1979-2024` → surfaced (warning or `PreflightError`, per maintainer preference — default to `PreflightError` with `--allow-incomplete-sources` NOT bypassing this, since it's a correctness inconsistency, not an incompleteness).
- [ ] **Step 2: Run — Expected FAIL.**
- [ ] **Step 3: Implement** the cross-check comparing effective-config target params to the NC-derived `target` step params.
- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit + PR-7 self-review; stop for review.**

---

## Self-review (run against the spec before handing off)

**Spec coverage** — every Pillar maps to a PR:
- Pillar 1 (schema + one skeleton + upgrade-manifest) → **PR-1** ✓
- Pillar 2 (`rebuild-manifest` generic projection, absorb reconcile/rebuild_lineage/backfill/#278) → **PR-2** ✓
- Pillar 3 (completeness gate, verify-don't-mutate) → **PR-3 Task 3.1** ✓
- Pillar 4 (per-fabric guarantee; `stage_fabric_child` hard error) → **PR-3 Task 3.2** ✓ (discoverability derived, no registry — decision D honored: no registry is built)
- Pillar 5 (effective-config durability + catalog-key validation + upgrade-config extension) → **PR-4 + PR-5** ✓
- Pillar 6 (config↔product↔manifest triangle) → **PR-7** ✓
- Pillar 7 (CLAUDE.md guardrail) → **PR-6** ✓

**Decisions A–G coverage:** A (absorb/supersede) → PR-2 Task 2.7; B (full deterministic regenerate) → PR-2; C (incremental capture kept + non-clobbering) → PR-2 Task 2.5 read-merge; D (no registry) → PR-3 (gate-at-publish only, no registry created); E (determinism: mtime-only, no `now()`, sorted ordering) → PR-2 Tasks 2.1/2.6; F (honesty tags) → PR-2 Tasks 2.3/2.4; G (one spec, separable PRs) → this plan's structure.

**Type/name consistency:** `CURRENT_MANIFEST_SCHEMA_VERSION` (lineage.py) used identically across PR-1/PR-2/PR-3; `iso_from_mtime` / `step_sort_key` defined in PR-2 Task 2.1 and consumed in 2.4/2.5; `rebuild_manifest(project, *, compute_sha256, dry_run)` signature consistent across PR-2 and the PR-3 gate; `build_source_entry` / `synthesize_steps` / `parse_aggregated_filename` names stable.

**Open items — all resolved with the maintainer 2026-05-29:**
1. **Sequencing → stacked PRs.** See "Branch & PR strategy (stacked)" above; each PR branches off and targets the previous PR's branch.
2. **PR-2 → delete `reconcile-manifest` outright** (not a shim). Task 2.7 deletes `reconcile.py`, the fetch-module reconcile hooks, the three reconcile tests, and redirects the reconcile-manifest doc.
3. **PR-4 → as written:** nested `_effective_config_meta` block. (When implementing, still grep for any consumer that reads `config.effective.yml` positionally — none expected.)
4. **PR-7 → as written:** inconsistency at publish is a hard `PreflightError`, not bypassed by `--allow-incomplete-sources`.
