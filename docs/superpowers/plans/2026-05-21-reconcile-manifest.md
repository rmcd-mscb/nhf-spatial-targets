# reconcile-manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `nhf-targets reconcile-manifest` CLI command that scans a project's shared datastore for already-downloaded source NetCDFs and gap-fill-merges `provenance: reconciled` file records into the project's `manifest.json`.

**Architecture:** A new `reconcile.py` module owns the orchestrator, a single atomic gap-fill writer, and a lazy `source_key -> "module:func"` registry. Each participating fetch module exposes a `reconcile(project, *, checksum=False) -> list[dict]` hook that only scans its own on-disk layout and returns file records; the shared writer merges them (append-only by record identity, never mutating existing records, so true `fetch` provenance is preserved). Two hooks ship initially: `era5_land` and `mod16a2_v061`.

**Tech Stack:** Python 3.11+, cyclopts (CLI), rich (output), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-21-reconcile-manifest-design.md`

**Conventions for every commit in this plan:**
- Branch is already `feature/160-reconcile-manifest`.
- Commit via `pixi run git commit` (never bare `git commit`).
- Stage files explicitly by path (never `git add -A`).
- End commit messages with the trailer:
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
- Run targeted tests for the file you touched; the full suite + fmt + lint runs at the end (Task 8). Per project convention, pytest is slow on this HPC — prefer `pixi run -e dev pytest tests/test_reconcile_manifest.py -q`.

---

### Task 1: Pure gap-fill core in `reconcile.py`

The append-only merge is the part that must never regress (it's what keeps a true `fetch` record from being clobbered by a `reconciled` one — see issue #97). Build and test it as pure functions first, no disk.

**Files:**
- Create: `src/nhf_spatial_targets/reconcile.py`
- Test: `tests/test_reconcile_manifest.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reconcile_manifest.py`:

```python
"""Tests for nhf_spatial_targets.reconcile (issue #160)."""

from __future__ import annotations

from nhf_spatial_targets import reconcile


def test_gap_fill_appends_only_new_year_records():
    existing = [{"year": 2020, "provenance": "fetch", "path": "a"}]
    new = [
        {"year": 2020, "provenance": "reconciled", "path": "a"},  # dup year
        {"year": 2021, "provenance": "reconciled", "path": "b"},  # new
    ]
    merged, added = reconcile._gap_fill(existing, new)
    # 2020 fetch record is untouched; only 2021 is appended.
    assert merged == [
        {"year": 2020, "provenance": "fetch", "path": "a"},
        {"year": 2021, "provenance": "reconciled", "path": "b"},
    ]
    assert added == [{"year": 2021, "provenance": "reconciled", "path": "b"}]


def test_gap_fill_keys_on_path_when_no_year():
    existing = [{"path": "x"}]
    new = [{"path": "x"}, {"path": "y"}]
    merged, added = reconcile._gap_fill(existing, new)
    assert added == [{"path": "y"}]
    assert len(merged) == 2


def test_gap_fill_tolerates_malformed_existing_records():
    existing = [{"note": "no id here"}]  # neither year nor path
    new = [{"year": 1999, "path": "z"}]
    merged, added = reconcile._gap_fill(existing, new)
    assert added == [{"year": 1999, "path": "z"}]
    assert merged == [{"note": "no id here"}, {"year": 1999, "path": "z"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_reconcile_manifest.py -q`
Expected: FAIL — `AttributeError: module 'nhf_spatial_targets.reconcile' has no attribute '_gap_fill'` (or ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

Create `src/nhf_spatial_targets/reconcile.py`:

```python
"""Backfill manifest.json provenance from an existing shared datastore.

See docs/architecture/reconcile-manifest.md and issue #160. The merge is
gap-fill only: reconcile appends records for files not already recorded and
never mutates an existing record, so a true ``fetch`` record is never
downgraded to ``reconciled``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def _record_identity(rec: dict) -> tuple[str, object]:
    """Stable dedupe identity for a file record: year if present, else path."""
    if "year" in rec:
        return ("year", int(rec["year"]))
    if "path" in rec:
        return ("path", rec["path"])
    raise ValueError(f"reconcile record lacks both 'year' and 'path': {rec!r}")


def _gap_fill(
    existing: list[dict], new_records: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Append only the new_records whose identity is not already in existing.

    Returns ``(merged, added)``. ``merged`` is ``existing`` (untouched, in
    order) followed by the appended records. Malformed existing records
    (no year/path) are tolerated — they stay in ``merged`` and never block
    an append.
    """
    seen: set[tuple[str, object]] = set()
    for r in existing:
        try:
            seen.add(_record_identity(r))
        except ValueError:
            continue
    added: list[dict] = []
    for r in new_records:
        ident = _record_identity(r)
        if ident not in seen:
            added.append(r)
            seen.add(ident)
    return existing + added, added


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 and return the hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_reconcile_manifest.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/reconcile.py tests/test_reconcile_manifest.py
pixi run git commit -m "feat(reconcile): gap-fill merge core + sha256 helper (#160)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Report type + atomic per-source writer

Add the result dataclass and the flock-guarded read-merge-write that applies one source's records, mirroring the pattern in `aggregate/_driver.update_manifest` and `fetch/era5_land._update_manifest`.

**Files:**
- Modify: `src/nhf_spatial_targets/reconcile.py`
- Test: `tests/test_reconcile_manifest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reconcile_manifest.py`:

```python
import json

from nhf_spatial_targets.workspace import Project


def _make_project(tmp_path) -> Project:
    """A Project pointing workdir/datastore at tmp dirs; config minimal.

    reconcile only touches project.raw_dir(), project.manifest_path, and
    project.datastore, so config/fabric can be empty.
    """
    datastore = tmp_path / "datastore"
    datastore.mkdir()
    return Project(
        workdir=tmp_path,
        datastore=datastore,
        config={},
        fabric={},
        dir_mode=None,
    )


def test_apply_records_creates_minimal_entry_when_absent(tmp_path):
    project = _make_project(tmp_path)
    records = [
        {"year": 2020, "path": "p2020", "provenance": "reconciled"},
        {"year": 2021, "path": "p2021", "provenance": "reconciled"},
    ]
    result = reconcile._apply_records(
        project, "mod16a2_v061", records, dry_run=False
    )
    assert result.added == 2
    assert result.status == "reconciled"

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["sources"]["mod16a2_v061"]
    assert entry["source_key"] == "mod16a2_v061"
    assert entry["period"] == "2020/2021"  # derived from year span
    assert "reconciled_utc" in entry
    assert len(entry["files"]) == 2


def test_apply_records_gap_fills_without_touching_existing(tmp_path):
    project = _make_project(tmp_path)
    # Pre-existing manifest with a true fetch record for 2020.
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "mod16a2_v061": {
                        "source_key": "mod16a2_v061",
                        "period": "2020/2020",
                        "files": [{"year": 2020, "path": "real", "downloaded_utc": "T"}],
                    }
                },
                "steps": [],
            }
        )
    )
    records = [
        {"year": 2020, "path": "ondisk", "provenance": "reconciled"},
        {"year": 2021, "path": "ondisk21", "provenance": "reconciled"},
    ]
    result = reconcile._apply_records(
        project, "mod16a2_v061", records, dry_run=False
    )
    assert result.added == 1
    assert result.already_recorded == 1

    entry = json.loads((tmp_path / "manifest.json").read_text())["sources"]["mod16a2_v061"]
    files_by_year = {f["year"]: f for f in entry["files"]}
    # 2020 fetch record is byte-for-byte preserved (no provenance key added).
    assert files_by_year[2020] == {"year": 2020, "path": "real", "downloaded_utc": "T"}
    assert files_by_year[2021]["provenance"] == "reconciled"
    # Entry-level metadata of the pre-existing source is left alone.
    assert "reconciled_utc" not in entry


def test_apply_records_dry_run_does_not_write(tmp_path):
    project = _make_project(tmp_path)
    records = [{"year": 2020, "path": "p", "provenance": "reconciled"}]
    result = reconcile._apply_records(project, "mod16a2_v061", records, dry_run=True)
    assert result.added == 1
    assert not (tmp_path / "manifest.json").exists()  # nothing written
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_reconcile_manifest.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute '_apply_records'`.

- [ ] **Step 3: Write minimal implementation**

Add to the top imports of `src/nhf_spatial_targets/reconcile.py`:

```python
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone

from nhf_spatial_targets import catalog as _catalog
from nhf_spatial_targets.workspace import Project

try:
    import fcntl as _fcntl

    _HAVE_FLOCK = True
except ImportError:  # pragma: no cover - Windows
    _HAVE_FLOCK = False
```

Append to `src/nhf_spatial_targets/reconcile.py`:

```python
@dataclass
class SourceReconcileResult:
    """Per-source outcome of a reconcile pass (drives the CLI table + tests)."""

    source_key: str
    status: str  # "reconciled" | "no-op" | "no-hook" | "empty"
    on_disk: int = 0
    already_recorded: int = 0
    added: int = 0


def _read_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {"sources": {}, "steps": []}
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manifest.json at {manifest_path} is corrupt and cannot be parsed. "
            f"Inspect or restore it before reconciling. Detail: {exc}"
        ) from exc


def _atomic_write(manifest_path: Path, manifest: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp).replace(manifest_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _apply_records(
    project: Project,
    source_key: str,
    records: list[dict],
    *,
    dry_run: bool,
) -> SourceReconcileResult:
    """Gap-fill-merge ``records`` into ``manifest.json`` for one source.

    flock-guarded read-merge-write. Appends only records whose identity is
    absent; never mutates an existing record. Creates a minimal source entry
    (source_key, access_url, derived period, reconciled_utc) only when the
    source has no existing entry; an existing entry's metadata is left alone.
    """
    manifest_path = project.manifest_path
    lock_path = manifest_path.with_suffix(".lock")
    with open(lock_path, "a") as lock_f:
        if _HAVE_FLOCK:
            _fcntl.flock(lock_f, _fcntl.LOCK_EX)
        manifest = _read_manifest(manifest_path)
        manifest.setdefault("sources", {})
        created = source_key not in manifest["sources"]
        entry = manifest["sources"].get(source_key, {})
        existing_files = entry.get("files", [])
        merged, added = _gap_fill(existing_files, records)

        result = SourceReconcileResult(
            source_key=source_key,
            status="reconciled" if added else "no-op",
            on_disk=len(records),
            already_recorded=len(records) - len(added),
            added=len(added),
        )
        if dry_run or not added:
            return result

        entry["files"] = merged
        if created:
            meta = _catalog.source(source_key)
            entry["source_key"] = source_key
            entry["access_url"] = meta.get("access", {}).get("url", "")
            entry["reconciled_utc"] = datetime.now(timezone.utc).isoformat()
            years = sorted(int(r["year"]) for r in merged if "year" in r)
            if years:
                entry["period"] = f"{years[0]}/{years[-1]}"
        manifest["sources"][source_key] = entry
        _atomic_write(manifest_path, manifest)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_reconcile_manifest.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/reconcile.py tests/test_reconcile_manifest.py
pixi run git commit -m "feat(reconcile): atomic gap-fill writer + result type (#160)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Orchestrator + lazy hook registry

`reconcile_manifest` resolves source keys, dispatches each through the registry (lazy import), and applies the returned records. Test with a monkeypatched registry/hook so it's independent of the real fetch modules.

**Files:**
- Modify: `src/nhf_spatial_targets/reconcile.py`
- Test: `tests/test_reconcile_manifest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reconcile_manifest.py`:

```python
def test_reconcile_manifest_dispatches_registered_hooks(tmp_path, monkeypatch):
    project = _make_project(tmp_path)

    def fake_hook(proj, *, checksum=False):
        return [{"year": 2020, "path": "p", "provenance": "reconciled"}]

    # Register a fake hook under a real catalog key.
    monkeypatch.setattr(
        reconcile,
        "_RECONCILERS",
        {"mod16a2_v061": "tests.test_reconcile_manifest:_HOOK"},
    )
    monkeypatch.setattr(reconcile, "_call_hook", lambda spec, proj, *, checksum: fake_hook(proj))

    results = reconcile.reconcile_manifest(
        project, sources=["mod16a2_v061"], dry_run=False
    )
    assert len(results) == 1
    assert results[0].status == "reconciled"
    assert results[0].added == 1


def test_reconcile_manifest_reports_no_hook_for_unregistered_source(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    monkeypatch.setattr(reconcile, "_RECONCILERS", {})
    results = reconcile.reconcile_manifest(
        project, sources=["era5_land"], dry_run=False
    )
    assert results[0].status == "no-hook"


def test_reconcile_manifest_reports_empty_when_hook_returns_nothing(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    monkeypatch.setattr(reconcile, "_RECONCILERS", {"era5_land": "x:y"})
    monkeypatch.setattr(reconcile, "_call_hook", lambda spec, proj, *, checksum: [])
    results = reconcile.reconcile_manifest(
        project, sources=["era5_land"], dry_run=False
    )
    assert results[0].status == "empty"
    assert not (tmp_path / "manifest.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_reconcile_manifest.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'reconcile_manifest'`.

- [ ] **Step 3: Write minimal implementation**

Add to the imports of `src/nhf_spatial_targets/reconcile.py`:

```python
import importlib
import logging

logger = logging.getLogger(__name__)
```

Append to `src/nhf_spatial_targets/reconcile.py`:

```python
# source_key -> "module:function". Imported lazily so reconciling one source
# doesn't pull in every fetch module's heavy deps (earthaccess, cdsapi).
_RECONCILERS: dict[str, str] = {
    "era5_land": "nhf_spatial_targets.fetch.era5_land:reconcile",
    "mod16a2_v061": "nhf_spatial_targets.fetch.modis:reconcile_mod16a2",
}


def _call_hook(spec: str, project: Project, *, checksum: bool) -> list[dict]:
    module_name, func_name = spec.split(":")
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func(project, checksum=checksum)


def reconcile_manifest(
    project: Project,
    *,
    sources: list[str] | None = None,
    dry_run: bool = False,
    checksum: bool = False,
) -> list[SourceReconcileResult]:
    """Backfill manifest.json from the datastore for the requested sources.

    ``sources=None`` means every catalog source (those without a registered
    reconcile hook are reported with status ``no-hook``). Gap-fill only; see
    module docstring.
    """
    keys = sources if sources else list(_catalog.sources().keys())
    results: list[SourceReconcileResult] = []
    for key in keys:
        spec = _RECONCILERS.get(key)
        if spec is None:
            results.append(SourceReconcileResult(key, "no-hook"))
            continue
        records = _call_hook(spec, project, checksum=checksum)
        if not records:
            results.append(SourceReconcileResult(key, "empty"))
            continue
        results.append(_apply_records(project, key, records, dry_run=dry_run))
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_reconcile_manifest.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/reconcile.py tests/test_reconcile_manifest.py
pixi run git commit -m "feat(reconcile): orchestrator + lazy hook registry (#160)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: ERA5-Land `reconcile` hook

Scan `{datastore}/era5_land/{daily,monthly}/era5_land_{daily,monthly}_{year}.nc` and return year-keyed records.

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/era5_land.py` (add `reconcile`, colocated near `_update_manifest` at the end of the file; add `import re` if absent)
- Test: `tests/test_reconcile_era5_land.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reconcile_era5_land.py`:

```python
"""Tests for fetch.era5_land.reconcile (issue #160)."""

from __future__ import annotations

from datetime import timezone

from nhf_spatial_targets.fetch import era5_land
from nhf_spatial_targets.workspace import Project


def _project(tmp_path) -> Project:
    ds = tmp_path / "datastore"
    ds.mkdir()
    return Project(workdir=tmp_path, datastore=ds, config={}, fabric={}, dir_mode=None)


def _write_year(project, year, *, daily=True, monthly=True):
    root = project.raw_dir("era5_land")
    if daily:
        d = root / "daily"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"era5_land_daily_{year}.nc").write_bytes(b"daily")
    if monthly:
        m = root / "monthly"
        m.mkdir(parents=True, exist_ok=True)
        (m / f"era5_land_monthly_{year}.nc").write_bytes(b"monthly")


def test_reconcile_empty_datastore_returns_empty(tmp_path):
    project = _project(tmp_path)
    assert era5_land.reconcile(project) == []


def test_reconcile_returns_one_record_per_complete_year(tmp_path):
    project = _project(tmp_path)
    _write_year(project, 2019)
    _write_year(project, 2020)
    records = era5_land.reconcile(project)
    years = sorted(r["year"] for r in records)
    assert years == [2019, 2020]
    r = records[0]
    assert r["provenance"] == "reconciled"
    assert r["daily_path"].endswith("era5_land_daily_2019.nc")
    assert r["monthly_path"].endswith("era5_land_monthly_2019.nc")
    # consolidated_utc is an ISO-8601 UTC string derived from mtime.
    assert r["consolidated_utc"].endswith("+00:00")
    assert "sha256_daily" not in r  # checksum off by default


def test_reconcile_skips_year_missing_its_monthly_pair(tmp_path):
    project = _project(tmp_path)
    _write_year(project, 2021, daily=True, monthly=False)
    assert era5_land.reconcile(project) == []


def test_reconcile_checksum_adds_hashes(tmp_path):
    project = _project(tmp_path)
    _write_year(project, 2022)
    (records,) = era5_land.reconcile(project, checksum=True)
    assert "sha256_daily" in records and "sha256_monthly" in records
    assert len(records["sha256_daily"]) == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_reconcile_era5_land.py -q`
Expected: FAIL — `AttributeError: module 'nhf_spatial_targets.fetch.era5_land' has no attribute 'reconcile'`.

- [ ] **Step 3: Write minimal implementation**

Confirm `import re` is at the top of `src/nhf_spatial_targets/fetch/era5_land.py`; add it if missing (alphabetically with the stdlib imports). Then append to the end of the file:

```python
def reconcile(project: "Project", *, checksum: bool = False) -> list[dict]:
    """Scan the datastore for consolidated ERA5-Land NCs (issue #160).

    Returns one record per year that has *both* a daily and a monthly NC on
    disk, tagged ``provenance="reconciled"`` with ``consolidated_utc`` from
    the newer of the pair's mtimes. A year with only one of the pair is
    skipped (incomplete). ``checksum=True`` adds ``sha256_daily`` /
    ``sha256_monthly``.
    """
    from datetime import datetime, timezone

    from nhf_spatial_targets.reconcile import sha256_file

    raw_root = project.raw_dir(_SOURCE_KEY)
    daily_dir = raw_root / "daily"
    monthly_dir = raw_root / "monthly"
    records: list[dict] = []
    for daily_path in sorted(daily_dir.glob("era5_land_daily_*.nc")):
        m = re.search(r"era5_land_daily_(\d{4})\.nc$", daily_path.name)
        if not m:
            continue
        year = int(m.group(1))
        monthly_path = monthly_dir / f"era5_land_monthly_{year}.nc"
        if not monthly_path.exists():
            continue
        mtime = max(daily_path.stat().st_mtime, monthly_path.stat().st_mtime)
        rec = {
            "year": year,
            "daily_path": str(daily_path),
            "monthly_path": str(monthly_path),
            "consolidated_utc": datetime.fromtimestamp(
                mtime, tz=timezone.utc
            ).isoformat(),
            "provenance": "reconciled",
        }
        if checksum:
            rec["sha256_daily"] = sha256_file(daily_path)
            rec["sha256_monthly"] = sha256_file(monthly_path)
        records.append(rec)
    return records
```

Note: the `from nhf_spatial_targets.reconcile import sha256_file` is a function-local import to avoid any import-time coupling between `reconcile.py` and the fetch modules. `"Project"` in the annotation is a forward-ref string — `Project` is not imported in `era5_land.py`, and `from __future__ import annotations` (already at the top of the file) makes the annotation a no-op string at runtime.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_reconcile_era5_land.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/era5_land.py tests/test_reconcile_era5_land.py
pixi run git commit -m "feat(reconcile): era5_land on-disk reconcile hook (#160)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: MOD16A2 `reconcile` hook

Scan `{datastore}/mod16a2_v061/mod16a2_v061_{year}_consolidated.nc` and return year-keyed records. Implemented as a generic `_reconcile_modis(project, source_key, ...)` so `mod10c1` is a trivial follow-on.

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/modis.py` (add `_reconcile_modis` + `reconcile_mod16a2` near `_update_manifest`; confirm `import re` present)
- Test: `tests/test_reconcile_modis.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reconcile_modis.py`:

```python
"""Tests for fetch.modis.reconcile_mod16a2 (issue #160)."""

from __future__ import annotations

from nhf_spatial_targets.fetch import modis
from nhf_spatial_targets.workspace import Project


def _project(tmp_path) -> Project:
    ds = tmp_path / "datastore"
    ds.mkdir()
    return Project(workdir=tmp_path, datastore=ds, config={}, fabric={}, dir_mode=None)


def _write_consolidated(project, year, content=b"nc"):
    root = project.raw_dir("mod16a2_v061")
    root.mkdir(parents=True, exist_ok=True)
    (root / f"mod16a2_v061_{year}_consolidated.nc").write_bytes(content)


def test_reconcile_empty_returns_empty(tmp_path):
    assert modis.reconcile_mod16a2(_project(tmp_path)) == []


def test_reconcile_returns_year_keyed_records(tmp_path):
    project = _project(tmp_path)
    _write_consolidated(project, 2018, content=b"abc")
    _write_consolidated(project, 2019, content=b"defgh")
    records = modis.reconcile_mod16a2(project)
    by_year = {r["year"]: r for r in records}
    assert set(by_year) == {2018, 2019}
    assert by_year[2018]["provenance"] == "reconciled"
    assert by_year[2018]["size_bytes"] == 3
    assert by_year[2018]["path"].endswith("mod16a2_v061_2018_consolidated.nc")
    assert by_year[2018]["downloaded_utc"].endswith("+00:00")
    assert "sha256" not in by_year[2018]


def test_reconcile_checksum_adds_sha256(tmp_path):
    project = _project(tmp_path)
    _write_consolidated(project, 2020)
    (rec,) = modis.reconcile_mod16a2(project, checksum=True)
    assert len(rec["sha256"]) == 64


def test_reconcile_ignores_unrelated_files(tmp_path):
    project = _project(tmp_path)
    root = project.raw_dir("mod16a2_v061")
    root.mkdir(parents=True, exist_ok=True)
    (root / "_tmp_2020_001.nc").write_bytes(b"junk")  # not a consolidated NC
    assert modis.reconcile_mod16a2(project) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_reconcile_modis.py -q`
Expected: FAIL — `AttributeError: module 'nhf_spatial_targets.fetch.modis' has no attribute 'reconcile_mod16a2'`.

- [ ] **Step 3: Write minimal implementation**

Confirm `import re` is at the top of `src/nhf_spatial_targets/fetch/modis.py` (add if missing). Append near `_update_manifest`:

```python
def _reconcile_modis(
    project: "Project", source_key: str, *, checksum: bool = False
) -> list[dict]:
    """Scan the datastore for a MODIS product's per-year consolidated NCs.

    Returns one record per ``{source_key}_{year}_consolidated.nc`` found,
    tagged ``provenance="reconciled"`` with ``downloaded_utc`` from the file
    mtime. ``checksum=True`` adds ``sha256``. Shared by every MODIS product
    (issue #160).
    """
    from datetime import datetime, timezone

    from nhf_spatial_targets.reconcile import sha256_file

    out_dir = project.raw_dir(source_key)
    pat = re.compile(rf"{re.escape(source_key)}_(\d{{4}})_consolidated\.nc$")
    records: list[dict] = []
    for nc in sorted(out_dir.glob(f"{source_key}_*_consolidated.nc")):
        m = pat.search(nc.name)
        if not m:
            continue
        st = nc.stat()
        rec = {
            "year": int(m.group(1)),
            "path": str(nc),
            "size_bytes": st.st_size,
            "downloaded_utc": datetime.fromtimestamp(
                st.st_mtime, tz=timezone.utc
            ).isoformat(),
            "provenance": "reconciled",
        }
        if checksum:
            rec["sha256"] = sha256_file(nc)
        records.append(rec)
    return records


def reconcile_mod16a2(project: "Project", *, checksum: bool = False) -> list[dict]:
    """Reconcile hook for MOD16A2 v061 (issue #160)."""
    return _reconcile_modis(project, _MOD16A2_SOURCE_KEY, checksum=checksum)
```

(`"Project"` is a forward-ref string; `modis.py` already has `from __future__ import annotations`, so no import is needed.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_reconcile_modis.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/modis.py tests/test_reconcile_modis.py
pixi run git commit -m "feat(reconcile): mod16a2 on-disk reconcile hook (#160)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: End-to-end reconcile through the real hooks

Now that both hooks exist and the registry points at them, add integration tests that exercise the full path (orchestrator → real hook → writer) for the acceptance scenarios: full backfill, idempotent gap-fill, and dry-run.

**Files:**
- Test: `tests/test_reconcile_manifest.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reconcile_manifest.py`:

```python
def _seed_era5(project, *years):
    root = project.raw_dir("era5_land")
    (root / "daily").mkdir(parents=True, exist_ok=True)
    (root / "monthly").mkdir(parents=True, exist_ok=True)
    for y in years:
        (root / "daily" / f"era5_land_daily_{y}.nc").write_bytes(b"d")
        (root / "monthly" / f"era5_land_monthly_{y}.nc").write_bytes(b"m")


def test_end_to_end_empty_datastore_is_noop(tmp_path):
    project = _make_project(tmp_path)
    results = reconcile.reconcile_manifest(project, sources=["era5_land"])
    assert results[0].status == "empty"
    assert not (tmp_path / "manifest.json").exists()


def test_end_to_end_full_backfill(tmp_path):
    project = _make_project(tmp_path)
    _seed_era5(project, 2019, 2020, 2021)
    results = reconcile.reconcile_manifest(project, sources=["era5_land"])
    assert results[0].added == 3

    entry = json.loads((tmp_path / "manifest.json").read_text())["sources"]["era5_land"]
    assert {f["year"] for f in entry["files"]} == {2019, 2020, 2021}
    assert all(f["provenance"] == "reconciled" for f in entry["files"])
    assert entry["period"] == "2019/2021"


def test_end_to_end_gap_fill_is_idempotent_and_preserves_fetch(tmp_path):
    project = _make_project(tmp_path)
    _seed_era5(project, 2019, 2020, 2021)
    # Pre-existing manifest with a true fetch record for 2020.
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "era5_land": {
                        "source_key": "era5_land",
                        "period": "2020/2020",
                        "files": [
                            {
                                "year": 2020,
                                "daily_path": "real_d",
                                "monthly_path": "real_m",
                                "consolidated_utc": "T",
                            }
                        ],
                    }
                },
                "steps": [],
            }
        )
    )
    first = reconcile.reconcile_manifest(project, sources=["era5_land"])
    assert first[0].added == 2  # 2019, 2021
    manifest_after_first = (tmp_path / "manifest.json").read_text()

    # Idempotent: a second run adds nothing and leaves the file identical.
    second = reconcile.reconcile_manifest(project, sources=["era5_land"])
    assert second[0].added == 0
    assert second[0].status == "no-op"
    assert (tmp_path / "manifest.json").read_text() == manifest_after_first

    files = json.loads(manifest_after_first)["sources"]["era5_land"]["files"]
    rec_2020 = next(f for f in files if f["year"] == 2020)
    assert rec_2020 == {
        "year": 2020,
        "daily_path": "real_d",
        "monthly_path": "real_m",
        "consolidated_utc": "T",
    }  # untouched fetch record, no provenance key


def test_end_to_end_dry_run_reports_without_writing(tmp_path):
    project = _make_project(tmp_path)
    _seed_era5(project, 2019, 2020)
    results = reconcile.reconcile_manifest(project, sources=["era5_land"], dry_run=True)
    assert results[0].added == 2
    assert not (tmp_path / "manifest.json").exists()
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `pixi run -e dev pytest tests/test_reconcile_manifest.py -q`
Expected: PASS — the implementation already exists (Tasks 1–5); these tests assert the wired-up behavior. If any fail, fix the implementation, not the test. (Most likely failure: registry key mismatch — confirm `_RECONCILERS` keys are exactly `"era5_land"` and `"mod16a2_v061"`, matching `catalog.sources()`.)

- [ ] **Step 3: (only if Step 2 failed) fix implementation**

Adjust `reconcile.py` / hooks to satisfy the tests. Re-run.

- [ ] **Step 4: Commit**

```bash
git add tests/test_reconcile_manifest.py
pixi run git commit -m "test(reconcile): end-to-end backfill / idempotent gap-fill / dry-run (#160)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: CLI command `reconcile-manifest`

Top-level command that loads the project, runs the orchestrator, and prints a rich summary table.

**Files:**
- Modify: `src/nhf_spatial_targets/cli.py` (add command after the `validate` command, ~line 380)
- Test: `tests/test_reconcile_manifest.py` (append a CLI smoke test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reconcile_manifest.py`:

```python
def test_cli_command_is_registered():
    """The reconcile-manifest command is wired into the cyclopts app."""
    from nhf_spatial_targets import cli

    assert "reconcile-manifest" in cli.app
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_reconcile_manifest.py::test_cli_command_is_registered -q`
Expected: FAIL — `assert 'reconcile-manifest' in cli.app` is False.

(If `in cli.app` raises rather than returning False, replace the assertion with `assert "reconcile-manifest" in cli.app._commands` after inspecting `cli.app` in a REPL — cyclopts `App` supports `in` for command-name membership in current versions; adjust to the actual API if needed.)

- [ ] **Step 3: Write minimal implementation**

Insert into `src/nhf_spatial_targets/cli.py` immediately after the `validate` command (after line ~380, before `@fetch_app.command(name="all")`):

```python
@app.command(name="reconcile-manifest")
def reconcile_manifest_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir", "-d"],
            help="Project created by 'nhf-targets init'.",
        ),
    ],
    source: Annotated[
        list[str] | None,
        Parameter(
            name=["--source"],
            help="Catalog source key to reconcile (repeatable). Default: all.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        Parameter(name=["--dry-run"], help="Report what would change; write nothing."),
    ] = False,
    checksum: Annotated[
        bool,
        Parameter(name=["--checksum"], help="Compute sha256 for each record (slow)."),
    ] = False,
):
    """Backfill manifest.json from consolidated NCs already in the datastore.

    Use after creating a new project against a datastore that another project
    already populated. Adds 'provenance: reconciled' file records for sources
    found on disk but missing from this project's manifest; never overwrites
    existing records. See docs/architecture/reconcile-manifest.md.
    """
    from rich.console import Console
    from rich.table import Table

    from nhf_spatial_targets import workspace
    from nhf_spatial_targets.reconcile import reconcile_manifest

    console = Console()
    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    try:
        project = workspace.load(workdir)
        results = reconcile_manifest(
            project, sources=source, dry_run=dry_run, checksum=checksum
        )
    except (FileNotFoundError, ValueError, KeyError, OSError) as e:
        print(f"reconcile-manifest failed: {e}", file=sys.stderr)
        sys.exit(1)

    title = "reconcile-manifest (dry run — no changes written)" if dry_run else "reconcile-manifest"
    table = Table(title=title)
    table.add_column("source", style="bold")
    table.add_column("status")
    table.add_column("on disk", justify="right")
    table.add_column("already recorded", justify="right")
    table.add_column("added", justify="right")
    for r in results:
        table.add_row(
            r.source_key, r.status, str(r.on_disk), str(r.already_recorded), str(r.added)
        )
    console.print(table)

    total_added = sum(r.added for r in results)
    verb = "would add" if dry_run else "added"
    console.print(f"[bold green]{verb} {total_added} reconciled record(s).[/bold green]")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_reconcile_manifest.py::test_cli_command_is_registered -q`
Expected: PASS.

Then manually confirm the command parses (no project needed for `--help`):
Run: `pixi run nhf-targets reconcile-manifest --help`
Expected: usage text listing `--project-dir`, `--source`, `--dry-run`, `--checksum`.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/cli.py tests/test_reconcile_manifest.py
pixi run git commit -m "feat(cli): add reconcile-manifest command (#160)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Docs + final quality gate

**Files:**
- Create: `docs/architecture/reconcile-manifest.md`
- Modify: `CLAUDE.md` (one-line pointer in the *Projects & Datastore* section)

- [ ] **Step 1: Write the architecture doc**

Create `docs/architecture/reconcile-manifest.md`:

```markdown
# reconcile-manifest

`nhf-targets reconcile-manifest --project-dir <dir> [--source <key> ...] [--dry-run] [--checksum]`

## When to use it

The datastore is shared across projects; `manifest.json` is per-project. A
new project created against a datastore another project already populated
starts with an empty manifest even though the NCs are on disk. `reconcile-manifest`
scans the datastore and backfills file records so the project's provenance
matches on-disk reality — without re-running `fetch` for every source.

## What it writes

For each source with a reconcile hook, it adds one `files` record per
consolidated NC found on disk that is not already recorded, tagged:

- `provenance: "reconciled"` — the marker that distinguishes a reconciled
  record from a true `fetch` record (which has no `provenance` key).
- a timestamp from the file's mtime (each source uses its native fetch
  field: `consolidated_utc` for era5_land, `downloaded_utc` for modis).
- `sha256` only when `--checksum` is passed (off by default — reconcile is a
  fast directory scan; hashing a multi-hundred-GB datastore is opt-in).

## Guarantees

- **Gap-fill only.** Existing records are never mutated. A true `fetch`
  record is never downgraded to `reconciled`. Re-running is idempotent.
- **Not auto-run from `validate`.** Reconcile is an explicit operator action
  so the audit trail stays honest — the manifest never silently claims
  provenance the operator didn't ask it to assert.

## Coverage

Reconcile hooks ship for `era5_land` and `mod16a2_v061`. Sources without a
hook are reported as `no-hook` and skipped; adding one is a per-module
`reconcile(project, *, checksum=False) -> list[dict]` function plus a registry
line in `reconcile.py`.
```

- [ ] **Step 2: Add the CLAUDE.md pointer**

In `CLAUDE.md`, under the *Projects & Datastore* → *Workflow* list (after the `validate` / `fetch` steps), add a line:

```markdown
- `nhf-targets reconcile-manifest --project-dir <dir>` backfills `manifest.json`
  from consolidated NCs already in a shared datastore (new project against an
  existing datastore). Gap-fill only; see `docs/architecture/reconcile-manifest.md`.
```

- [ ] **Step 3: Run the full quality gate**

```bash
pixi run -e dev fmt
pixi run -e dev lint
pixi run -e dev pytest tests/test_reconcile_manifest.py tests/test_reconcile_era5_land.py tests/test_reconcile_modis.py -q
```
Expected: fmt reformats nothing new, lint passes, all reconcile tests pass.

- [ ] **Step 4: Commit**

```bash
git add docs/architecture/reconcile-manifest.md CLAUDE.md
pixi run git commit -m "docs(reconcile): architecture note + CLAUDE.md pointer (#160)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Push and open the PR**

```bash
git push -u origin feature/160-reconcile-manifest
gh pr create --base main --title "feat(cli): reconcile-manifest to backfill provenance from datastore (#160)" --body "Closes #160. <summarize: gap-fill-only merge, checksums off by default, era5_land + mod16a2 hooks, docs>"
```

---

## Self-Review

**1. Spec coverage:**
- Gap-fill-only merge → Task 1 (`_gap_fill`), Task 2 (`_apply_records`), Task 6 (idempotency/preserve-fetch e2e). ✓
- Checksums off by default, `--checksum` opt-in → hooks (Tasks 4/5), CLI (Task 7). ✓
- Hook returns records + shared writer → Tasks 1–3 (writer/orchestrator), 4/5 (hooks). ✓
- `reconcile(project, *, checksum=False) -> list[dict]` contract → Tasks 4/5. ✓
- Lazy `module:func` registry → Task 3. ✓
- Minimal entry with derived period when absent; existing entry untouched → Task 2 tests. ✓
- era5_land + mod16a2_v061 hooks → Tasks 4/5. ✓
- CLI `--project-dir`/`--source`/`--dry-run`/`--checksum` + rich summary → Task 7. ✓
- Native timestamp field per module (`consolidated_utc` vs `downloaded_utc`) → Tasks 4/5 + doc. ✓
- Acceptance test scenarios (empty/full/partial/dry-run) → Task 6; `--checksum` → Tasks 4/5. ✓
- Docs note + CLAUDE.md pointer → Task 8. ✓
- Out of scope (agg steps, auto-run from validate) → not implemented, noted in doc. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code. The only conditional is Task 6 Step 3 (fix-if-failing), which is standard TDD wiring verification, not a placeholder. ✓

**3. Type consistency:** `SourceReconcileResult` fields (`source_key`, `status`, `on_disk`, `already_recorded`, `added`) used identically in Tasks 2, 3, 6, 7. Hook names `reconcile` (era5) and `reconcile_mod16a2` (modis) match the `_RECONCILERS` registry specs in Task 3. `_gap_fill`/`_apply_records`/`_call_hook`/`reconcile_manifest`/`sha256_file` signatures consistent across tasks. Registry keys `era5_land`/`mod16a2_v061` match catalog keys and the hook tests. ✓

**One risk to watch (flagged for the implementer):** the `cli.app` membership check in Task 7 Step 1 depends on the cyclopts `App.__contains__` API. If `"reconcile-manifest" in cli.app` raises or misbehaves on the installed cyclopts version, fall back to asserting the command via `cli.app.__help__`/the registered subcommands mapping as noted inline — the assertion is a convenience, not load-bearing for the feature.
```
