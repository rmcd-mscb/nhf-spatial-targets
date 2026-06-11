"""Offline tests for the ``nhf-targets release`` CLI sub-app.

These exercise the CLI *wiring* only: help + argument parsing for all seven
subcommands, the local artifacts ``init`` / ``build`` / ``manifest`` produce,
and -- for the ScienceBase-touching subcommands -- that the right
``publish_*`` / ``dry_run`` / ``diff_local_vs_remote`` orchestrator is
dispatched with the right scope and flags. ScienceBase is never hit: the
client seam (:func:`cli.release._resolve_client`) and the project loader
(:func:`cli.release._load_project`) are patched, and the orchestrators are
patched with recorders. Live ScienceBase verification is deferred to the
credential/session-wiring phase (not yet implemented).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import nhf_spatial_targets.cli.release as rel
from nhf_spatial_targets.cli import app
from nhf_spatial_targets.release.build import build_all
from nhf_spatial_targets.release.publish import PublishResult
from tests.conftest import FakeSbSession, build_release_project, make_sb_client

SUBCOMMANDS = ("init", "build", "manifest", "dry-run", "publish", "status", "auth-test")
LABEL = "gfv_test"  # build_release_project's default fabric label

# Files every finished stage carries (from build._finish / checksums).
_STAGE_FILES = ("fgdc.xml", "iso.xml", "README.md", "checksums.csv", "SHA256SUMS")


# ---------------------------------------------------------------------------
# Invocation helpers (cyclopts raises SystemExit(0) on success)
# ---------------------------------------------------------------------------


def _run(*tokens: str) -> None:
    """Invoke the app, swallowing the SystemExit(0) cyclopts raises on success."""
    try:
        app(list(tokens), exit_on_error=False)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise


def _expect_exit(code: int, *tokens: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        app(list(tokens), exit_on_error=False)
    assert excinfo.value.code == code


def _fake_result(scope: str, key: str | None) -> PublishResult:
    """A minimal valid PublishResult for a patched publish_* recorder."""
    return PublishResult(
        scope=scope,  # type: ignore[arg-type]
        key=key,
        sb_id="SB1",
        mode="create",
        adopted=False,
        uploaded=("data.nc",),
        skipped=(),
        orphans=(),
        orphans_deleted=(),
        body_patched=(),
        registry_entry={},
    )


def _patch_seams(monkeypatch, project, client):
    """Patch the project loader + client seam to the given offline doubles."""
    monkeypatch.setattr(rel, "_load_project", lambda workdir: project)
    monkeypatch.setattr(rel, "_resolve_client", lambda **kwargs: client)


# ---------------------------------------------------------------------------
# Help + arg parsing for all seven subcommands
# ---------------------------------------------------------------------------


def test_release_help_lists_subcommands(capsys):
    _run("release", "--help")
    out = capsys.readouterr().out
    for name in SUBCOMMANDS:
        assert name in out


@pytest.mark.parametrize("sub", SUBCOMMANDS)
def test_each_subcommand_help_parses(sub):
    # --help parses the subcommand's option spec and exits 0; a parse error
    # would raise a non-zero SystemExit (or a cyclopts error) instead.
    _run("release", sub, "--help")


# ---------------------------------------------------------------------------
# init / build / manifest -- local artifacts (offline)
# ---------------------------------------------------------------------------


def test_init_creates_local_artifacts(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    _run("release", "init", "--project-dir", str(proj))

    assert (proj / "release" / "release.yml").is_file()
    assert (proj / "release" / "build").is_dir()
    assert (proj / "release" / "README.notes.md").is_file()


def test_init_missing_project_dir_exits_2(tmp_path):
    _expect_exit(2, "release", "init", "-d", str(tmp_path / "nope"))


def test_build_creates_stage_artifacts(tmp_path, monkeypatch):
    project = build_release_project(tmp_path)
    monkeypatch.setattr(rel, "_load_project", lambda workdir: project)

    _run("release", "build", "-d", str(project.workdir), "--scope", "fabric")

    stage = project.release_build_dir / "fabric" / LABEL
    for name in _STAGE_FILES:
        assert (stage / name).is_file(), f"missing {name}"


def test_manifest_recomputes_checksums(tmp_path):
    project = build_release_project(tmp_path)
    # Stage fabric + source + umbrella items on disk first.
    build_all(project, scope="all")

    fabric_csv = project.release_build_dir / "fabric" / LABEL / "checksums.csv"
    assert fabric_csv.is_file()
    fabric_csv.unlink()  # simulate a removed/edited integrity file

    _run("release", "manifest", "-d", str(project.workdir), "--scope", "all")

    assert fabric_csv.is_file()  # regenerated
    # Every staged item now carries both integrity files.
    for sub in (project.release_build_dir).rglob("SHA256SUMS"):
        assert sub.is_file()


def test_manifest_no_staged_items_exits_1(tmp_path):
    proj = tmp_path / "proj"
    (proj / "release" / "build").mkdir(parents=True)
    _expect_exit(1, "release", "manifest", "-d", str(proj))


# ---------------------------------------------------------------------------
# dry-run / status -- dispatch with the right scope (read-only)
# ---------------------------------------------------------------------------


def test_dry_run_dispatches_with_scope(tmp_path, monkeypatch):
    project = build_release_project(tmp_path)
    client = make_sb_client(FakeSbSession())
    _patch_seams(monkeypatch, project, client)
    recorder = MagicMock()
    monkeypatch.setattr(rel, "dry_run", recorder)

    _run("release", "dry-run", "-d", str(project.workdir), "--scope", "sources")

    recorder.assert_called_once()
    assert recorder.call_args.args[0] is project
    assert recorder.call_args.args[1] is client
    assert recorder.call_args.kwargs["scope"] == "sources"


def test_dry_run_default_scope_is_fabric(tmp_path, monkeypatch):
    project = build_release_project(tmp_path)
    _patch_seams(monkeypatch, project, make_sb_client(FakeSbSession()))
    recorder = MagicMock()
    monkeypatch.setattr(rel, "dry_run", recorder)

    _run("release", "dry-run", "-d", str(project.workdir))

    assert recorder.call_args.kwargs["scope"] == "fabric"


def test_status_dispatches_diff_with_scope(tmp_path, monkeypatch):
    project = build_release_project(tmp_path)
    client = make_sb_client(FakeSbSession())
    _patch_seams(monkeypatch, project, client)
    recorder = MagicMock(return_value=[])
    monkeypatch.setattr(rel, "diff_local_vs_remote", recorder)

    _run("release", "status", "-d", str(project.workdir), "--scope", "all")

    recorder.assert_called_once()
    assert recorder.call_args.args[0] is project
    assert recorder.call_args.kwargs["scope"] == "all"


# ---------------------------------------------------------------------------
# publish -- without --confirm is a dry run; never mutates ScienceBase
# ---------------------------------------------------------------------------


def test_publish_without_confirm_does_not_publish(tmp_path, monkeypatch):
    project = build_release_project(tmp_path)
    session = FakeSbSession()
    _patch_seams(monkeypatch, project, make_sb_client(session))
    # dry_run runs for real (read-only); the publish orchestrator must NOT.
    pub = MagicMock()
    monkeypatch.setattr(rel, "publish_fabric_child", pub)

    _run("release", "publish", "-d", str(project.workdir), "--scope", "fabric")

    pub.assert_not_called()
    assert session.created == []
    assert session.updated == []
    assert session.uploaded == []
    assert session.deleted == []
    assert not list(project.release_dir.glob("last_publish_*.log"))


# ---------------------------------------------------------------------------
# publish --confirm -- dispatch, log, and registry-commit reminder
# ---------------------------------------------------------------------------


def test_publish_fabric_confirm_dispatches_logs_and_reminds(
    tmp_path, monkeypatch, capsys
):
    project = build_release_project(tmp_path)
    client = make_sb_client(FakeSbSession())
    _patch_seams(monkeypatch, project, client)
    pub = MagicMock(return_value=_fake_result("fabric", LABEL))
    monkeypatch.setattr(rel, "publish_fabric_child", pub)

    _run(
        "release",
        "publish",
        "-d",
        str(project.workdir),
        "--scope",
        "fabric",
        "--confirm",
        "--delete-orphans",
    )

    pub.assert_called_once()
    assert pub.call_args.args[0] is project
    assert pub.call_args.args[1] is client
    assert pub.call_args.kwargs["delete_orphans"] is True
    assert pub.call_args.kwargs["create_new"] is False

    log_path = project.release_dir / f"last_publish_fabric_{LABEL}.log"
    assert log_path.is_file()
    assert "release_registry.yml" in log_path.read_text()
    assert "release_registry.yml" in capsys.readouterr().out


def test_publish_allow_incomplete_sources_threads_to_publisher(tmp_path, monkeypatch):
    """``--allow-incomplete-sources`` reaches the publisher as the deliberate,
    logged override of the completeness gate; it defaults to False."""
    project = build_release_project(tmp_path)
    client = make_sb_client(FakeSbSession())
    _patch_seams(monkeypatch, project, client)
    pub = MagicMock(return_value=_fake_result("fabric", LABEL))
    monkeypatch.setattr(rel, "publish_fabric_child", pub)

    _run(
        "release",
        "publish",
        "-d",
        str(project.workdir),
        "--scope",
        "fabric",
        "--confirm",
        "--allow-incomplete-sources",
    )

    assert pub.call_args.kwargs["allow_incomplete_sources"] is True


def test_publish_sources_confirm_requires_source(tmp_path, monkeypatch):
    project = build_release_project(tmp_path)
    _patch_seams(monkeypatch, project, make_sb_client(FakeSbSession()))
    monkeypatch.setattr(rel, "publish_source_child", MagicMock())

    _expect_exit(
        2,
        "release",
        "publish",
        "-d",
        str(project.workdir),
        "--scope",
        "sources",
        "--confirm",
    )


def test_publish_source_confirm_dispatches_with_key(tmp_path, monkeypatch):
    project = build_release_project(tmp_path)
    _patch_seams(monkeypatch, project, make_sb_client(FakeSbSession()))
    pub = MagicMock(return_value=_fake_result("source", "era5_land"))
    monkeypatch.setattr(rel, "publish_source_child", pub)

    _run(
        "release",
        "publish",
        "-d",
        str(project.workdir),
        "--scope",
        "sources",
        "--source",
        "era5_land",
        "--confirm",
    )

    pub.assert_called_once()
    assert pub.call_args.args[2] == "era5_land"
    assert (project.release_dir / "last_publish_sources_era5_land.log").is_file()


def test_publish_umbrella_confirm_requires_parent_id(tmp_path, monkeypatch):
    project = build_release_project(tmp_path)
    _patch_seams(monkeypatch, project, make_sb_client(FakeSbSession()))
    monkeypatch.setattr(rel, "publish_umbrella", MagicMock())

    _expect_exit(
        2,
        "release",
        "publish",
        "-d",
        str(project.workdir),
        "--scope",
        "umbrella",
        "--confirm",
    )


def test_publish_umbrella_confirm_dispatches_with_parent_id(tmp_path, monkeypatch):
    project = build_release_project(tmp_path)
    _patch_seams(monkeypatch, project, make_sb_client(FakeSbSession()))
    pub = MagicMock(return_value=_fake_result("umbrella", None))
    monkeypatch.setattr(rel, "publish_umbrella", pub)

    _run(
        "release",
        "publish",
        "-d",
        str(project.workdir),
        "--scope",
        "umbrella",
        "--confirm",
        "--parent-id",
        "COMM123",
    )

    pub.assert_called_once()
    assert pub.call_args.kwargs["parent_id"] == "COMM123"
    assert (project.release_dir / "last_publish_umbrella_umbrella.log").is_file()


def test_publish_create_umbrella_composes_umbrella_then_child(tmp_path, monkeypatch):
    project = build_release_project(tmp_path)
    _patch_seams(monkeypatch, project, make_sb_client(FakeSbSession()))
    calls: list[str] = []
    umb = MagicMock(
        side_effect=lambda *a, **k: (
            (calls.append("umbrella")) or _fake_result("umbrella", None)
        )
    )
    fab = MagicMock(
        side_effect=lambda *a, **k: (
            (calls.append("fabric")) or _fake_result("fabric", LABEL)
        )
    )
    monkeypatch.setattr(rel, "publish_umbrella", umb)
    monkeypatch.setattr(rel, "publish_fabric_child", fab)

    _run(
        "release",
        "publish",
        "-d",
        str(project.workdir),
        "--scope",
        "fabric",
        "--confirm",
        "--create-umbrella",
        "--parent-id",
        "COMM123",
    )

    umb.assert_called_once()
    fab.assert_called_once()
    assert calls == ["umbrella", "fabric"]  # umbrella published first
    log_text = (project.release_dir / f"last_publish_fabric_{LABEL}.log").read_text()
    assert "[umbrella]" in log_text
    assert f"[fabric/{LABEL}]" in log_text


@pytest.mark.parametrize(
    "publish_scope,preview_scope",
    [("sources", "sources"), ("umbrella", "all"), ("fabric", "fabric")],
)
def test_publish_without_confirm_previews_mapped_scope(
    tmp_path, monkeypatch, publish_scope, preview_scope
):
    """The publish-scope -> dry-run-scope preview mapping (_PUBLISH_PREVIEW_SCOPE):
    a no-confirm publish runs dry_run with the mapped build scope, never the raw
    publish scope (umbrella is not a valid dry_run scope)."""
    project = build_release_project(tmp_path)
    _patch_seams(monkeypatch, project, make_sb_client(FakeSbSession()))
    recorder = MagicMock()
    monkeypatch.setattr(rel, "dry_run", recorder)

    _run("release", "publish", "-d", str(project.workdir), "--scope", publish_scope)

    recorder.assert_called_once()
    assert recorder.call_args.kwargs["scope"] == preview_scope


def test_publish_create_umbrella_confirm_requires_parent_id(tmp_path, monkeypatch):
    """The --create-umbrella arm of the parent-id guard (not just --scope umbrella)."""
    project = build_release_project(tmp_path)
    _patch_seams(monkeypatch, project, make_sb_client(FakeSbSession()))
    monkeypatch.setattr(rel, "publish_umbrella", MagicMock())
    monkeypatch.setattr(rel, "publish_fabric_child", MagicMock())

    _expect_exit(
        2,
        "release",
        "publish",
        "-d",
        str(project.workdir),
        "--scope",
        "fabric",
        "--confirm",
        "--create-umbrella",
    )


@pytest.mark.parametrize("sub", ["build", "dry-run", "status"])
def test_project_command_missing_dir_exits_2(tmp_path, sub):
    """build / dry-run / status all route through _load_project_or_exit, which
    exits 2 before any client work when --project-dir does not exist."""
    _expect_exit(2, "release", sub, "-d", str(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# auth-test + the client seam
# ---------------------------------------------------------------------------


def test_auth_test_pings_and_prints_identity(monkeypatch, capsys):
    client = make_sb_client(FakeSbSession())
    monkeypatch.setattr(rel, "_resolve_client", lambda **kwargs: client)

    _run("release", "auth-test")

    assert "logged_in=True" in capsys.readouterr().out


def test_auth_test_without_credentials_exits_1(monkeypatch):
    # Exercise the real _resolve_client no-credentials branch + exit wrapper
    # (no SB_TOKEN, no --token). Must not import sciencebasepy.
    monkeypatch.delenv("SB_TOKEN", raising=False)
    _expect_exit(1, "release", "auth-test")


def test_auth_test_ping_failure_exits_1(monkeypatch):
    client = MagicMock()
    client.ping.side_effect = RuntimeError("ScienceBase unreachable")
    monkeypatch.setattr(rel, "_resolve_client", lambda **kwargs: client)
    _expect_exit(1, "release", "auth-test")
