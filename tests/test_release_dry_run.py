"""Offline tests for release.dry_run -- the read-only publish preview.

A dry run builds + validates locally and queries ScienceBase read-only; it must
never mutate the remote. These tests assert the structured report (would-create
vs exists vs drift), the mp warn/err degradation, and -- centrally -- that no
mutating client method is ever called.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

import nhf_spatial_targets.release.dry_run as dry_run_mod
from nhf_spatial_targets.release import publish, registry
from nhf_spatial_targets.release.build import build_fabric_child
from nhf_spatial_targets.release.dry_run import dry_run
from nhf_spatial_targets.release.validate_xml import MpResult
from tests.conftest import FakeSbSession, build_release_project, make_sb_client

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
LABEL = "gfv_test"


def _registry(tmp_path):
    return tmp_path / "release_registry.yml"


def _staged_names(project) -> set[str]:
    """ScienceBase basenames of everything publish would upload for the fabric."""
    result = build_fabric_child(project, now=NOW)
    names = {Path(e.path).name for e in result.entries}
    names |= {"checksums.csv", "SHA256SUMS"}
    return names


def _assert_no_writes(session: FakeSbSession) -> None:
    assert session.created == []
    assert session.updated == []
    assert session.uploaded == []
    assert session.deleted == []


# ---------------------------------------------------------------------------
# would-create / exists / drift
# ---------------------------------------------------------------------------


def test_would_create_when_not_in_registry(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    session = FakeSbSession()

    report = dry_run(
        project,
        make_sb_client(session),
        scope="fabric",
        now=NOW,
        registry_path=reg,
        render=False,
    )
    assert len(report.items) == 1
    item = report.items[0]
    assert item.scope == "fabric"
    assert item.sb_id is None
    assert item.diff.remote == "missing"
    assert item.remote_label == "missing"
    assert report.would_create == (item,)
    assert item.file_count > 0 and item.total_bytes > 0
    _assert_no_writes(session)


def test_exists_when_remote_matches_staged(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    session = FakeSbSession()
    # A remote item carrying exactly the staged file set (size omitted -> the
    # parity check compares names only when remote size is unknown).
    files = [{"name": n} for n in _staged_names(project)]
    session.seed_item("FAB", title="t", files=files)
    registry.put_fabric(
        LABEL,
        sb_id="FAB",
        uploaded_utc="2026-01-01T00:00:00+00:00",
        file_count=len(files),
        total_bytes=1,
        manifest_sha256="x",
        path=reg,
    )

    report = dry_run(
        project,
        make_sb_client(session),
        scope="fabric",
        now=NOW,
        registry_path=reg,
        render=False,
    )
    item = report.items[0]
    assert item.sb_id == "FAB"
    assert item.diff.remote == "present"
    assert item.remote_label == "exists"
    _assert_no_writes(session)


def test_drift_when_remote_has_extra_file(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    session = FakeSbSession()
    files = [{"name": n} for n in _staged_names(project)]
    files.append({"name": "orphan_old.nc"})  # not in staged payload -> drift
    session.seed_item("FAB", title="t", files=files)
    registry.put_fabric(
        LABEL,
        sb_id="FAB",
        uploaded_utc="2026-01-01T00:00:00+00:00",
        file_count=len(files),
        total_bytes=1,
        manifest_sha256="x",
        path=reg,
    )

    report = dry_run(
        project,
        make_sb_client(session),
        scope="fabric",
        now=NOW,
        registry_path=reg,
        render=False,
    )
    item = report.items[0]
    assert item.diff.remote == "drift"
    assert item.remote_label == "drift"
    assert "orphan_old.nc" in (item.diff.detail or "")
    _assert_no_writes(session)


# ---------------------------------------------------------------------------
# mp warn / err degradation
# ---------------------------------------------------------------------------


def test_mp_absent_degrades_to_warn(tmp_path, monkeypatch):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    session = FakeSbSession()
    monkeypatch.setattr(
        dry_run_mod,
        "validate_xml_file",
        lambda path: MpResult(
            True, ["mp not found on PATH; well-formedness only."], []
        ),
    )

    report = dry_run(
        project,
        make_sb_client(session),
        scope="fabric",
        now=NOW,
        registry_path=reg,
        render=False,
    )
    assert report.items[0].mp_status == "warn"
    assert report.ok is True  # warnings do not fail a dry run
    _assert_no_writes(session)


def test_mp_errors_make_report_not_ok(tmp_path, monkeypatch):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    session = FakeSbSession()
    monkeypatch.setattr(
        dry_run_mod,
        "validate_xml_file",
        lambda path: MpResult(False, [], ["element <bounding> is missing"]),
    )

    report = dry_run(
        project,
        make_sb_client(session),
        scope="fabric",
        now=NOW,
        registry_path=reg,
        render=False,
    )
    assert report.items[0].mp_status == "err"
    assert report.ok is False


# ---------------------------------------------------------------------------
# rendering + multi-scope
# ---------------------------------------------------------------------------


def test_render_writes_a_table_without_touching_sciencebase(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    session = FakeSbSession()
    console = Console(file=io.StringIO(), width=120)

    report = dry_run(
        project,
        make_sb_client(session),
        scope="all",
        now=NOW,
        registry_path=reg,
        console=console,
        render=True,
    )
    # all scope: fabric + 2 source children (era5_land, daymet) + umbrella.
    kinds = sorted(i.scope for i in report.items)
    assert kinds == ["fabric", "source", "source", "umbrella"]
    output = console.file.getvalue()
    assert "dry run" in output.lower()
    assert "would create" in output  # nothing is in the registry yet
    _assert_no_writes(session)


# ---------------------------------------------------------------------------
# diff_local_vs_remote: stale local + stale remote sb_id (read-only, no build)
# ---------------------------------------------------------------------------


def test_diff_reports_stale_local_when_staged_file_drifts(tmp_path):
    """diff does NOT rebuild, so a staged file edited after build surfaces as
    a 'stale' local payload via the narrowed verify_csv catch."""
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    session = FakeSbSession()
    build_fabric_child(project, now=NOW)  # stage exists with checksums.csv
    # Edit a staged data file's content (via its symlink target) -> verify fails.
    (project.workdir / "targets" / "runoff_targets.nc").write_bytes(b"DRIFTED-CONTENT")

    diffs = publish.diff_local_vs_remote(
        project, make_sb_client(session), scope="fabric", registry_path=reg
    )
    assert diffs[0].local == "stale"
    _assert_no_writes(session)


def test_diff_reports_stale_remote_sb_id_as_missing(tmp_path):
    """A registry sb_id that ScienceBase 404s reads as remote 'missing' with a
    stale-id note (the get_item-404 path in the read-only diff)."""
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    session = FakeSbSession()  # store is empty -> get_item raises not-found
    build_fabric_child(project, now=NOW)  # local payload built
    registry.put_fabric(
        LABEL,
        sb_id="GONE",
        uploaded_utc="2026-01-01T00:00:00+00:00",
        file_count=1,
        total_bytes=1,
        manifest_sha256="x",
        path=reg,
    )

    diffs = publish.diff_local_vs_remote(
        project, make_sb_client(session), scope="fabric", registry_path=reg
    )
    assert diffs[0].local == "built"
    assert diffs[0].remote == "missing"
    assert "stale" in (diffs[0].detail or "")
    _assert_no_writes(session)
