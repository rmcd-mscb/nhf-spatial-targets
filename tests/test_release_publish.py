"""Offline pre-flight + happy-path tests for release.publish.

Pre-flight gates must each fail loudly *before* anything is published; the
happy paths cover create and update for the umbrella, a source child, and a
fabric child against an in-memory ScienceBase (the :class:`FakeSbSession`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import requests

from nhf_spatial_targets.release import publish, registry
from nhf_spatial_targets.release._models import FileEntry, ReleaseError
from nhf_spatial_targets.release.build import BuildResult
from nhf_spatial_targets.release.checksums import ChecksumMismatch
from tests.conftest import FakeSbSession, build_release_project, make_sb_client

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
COMMUNITY = "COMMUNITY"  # the ScienceBase folder that hosts the release


def _registry(tmp_path):
    return tmp_path / "release_registry.yml"


def _seed_umbrella(reg, sb_id="UMB"):
    registry.put_umbrella(
        sb_id=sb_id,
        version="1.0",
        published_utc="2026-01-01T00:00:00+00:00",
        title="Umbrella",
        path=reg,
    )


# ---------------------------------------------------------------------------
# Pre-flight gates (each fatal, before any publish)
# ---------------------------------------------------------------------------


def test_preflight_missing_manifest_is_fatal(tmp_path):
    project = build_release_project(tmp_path, with_manifest=False)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    with pytest.raises(publish.PreflightError, match="manifest.json"):
        publish.publish_fabric_child(
            project, make_sb_client(FakeSbSession()), registry_path=reg, now=NOW
        )


def test_preflight_empty_authors_is_fatal(tmp_path):
    project = build_release_project(tmp_path, authors=[])
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    with pytest.raises(publish.PreflightError, match="authors"):
        publish.publish_fabric_child(
            project, make_sb_client(FakeSbSession()), registry_path=reg, now=NOW
        )


def test_preflight_missing_umbrella_sb_id_is_fatal(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)  # no umbrella recorded
    with pytest.raises(publish.PreflightError, match="umbrella"):
        publish.publish_fabric_child(
            project, make_sb_client(FakeSbSession()), registry_path=reg, now=NOW
        )


def test_checksum_drift_is_fatal(tmp_path, monkeypatch):
    """A staged file edited since build fails the verify_csv gate loudly."""
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)

    def _raise(_stage_dir):
        raise ChecksumMismatch("staged file drifted from checksums.csv")

    monkeypatch.setattr(publish, "verify_csv", _raise)
    with pytest.raises(ChecksumMismatch, match="drifted"):
        publish.publish_fabric_child(
            project, make_sb_client(FakeSbSession()), registry_path=reg, now=NOW
        )


# ---------------------------------------------------------------------------
# Happy-path create + update
# ---------------------------------------------------------------------------


def test_umbrella_create_records_registry(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    session = FakeSbSession()

    result = publish.publish_umbrella(
        project,
        make_sb_client(session),
        parent_id=COMMUNITY,
        registry_path=reg,
        now=NOW,
    )
    assert result.mode == "create"
    assert result.scope == "umbrella"
    assert result.key is None
    # Metadata-only: README/FGDC/ISO + integrity uploaded, no NetCDF data.
    assert "README.md" in result.uploaded
    assert not any(name.endswith(".nc") for name in result.uploaded)

    umbrella = registry.get_umbrella(reg)
    assert umbrella["sb_id"] == result.sb_id
    assert umbrella["version"] == "1.0"
    assert umbrella["title"]
    # The created item lives under the supplied community folder.
    assert session.items[result.sb_id]["parentId"] == COMMUNITY


def test_source_child_create_records_registry(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    session = FakeSbSession()

    result = publish.publish_source_child(
        project, make_sb_client(session), "era5_land", registry_path=reg, now=NOW
    )
    assert result.mode == "create"
    assert result.scope == "source" and result.key == "era5_land"
    # The consolidated NetCDFs are the source child's data payload.
    assert "era5_land_daily_1980.nc" in result.uploaded
    assert "era5_land_monthly_1980.nc" in result.uploaded

    entry = registry.get_source("era5_land", reg)
    assert entry["sb_id"] == result.sb_id
    assert entry["file_count"] == len(result.uploaded) + len(result.skipped)
    assert entry["total_bytes"] > 0


def test_fabric_create_then_update_patches_changed_body(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    session = FakeSbSession()
    client = make_sb_client(session)

    first = publish.publish_fabric_child(project, client, registry_path=reg, now=NOW)
    assert first.mode == "create"
    assert first.body_patched == ()  # create patches nothing

    # No-op update: nothing patched, the unchanged body matches the remote.
    second = publish.publish_fabric_child(project, client, registry_path=reg, now=NOW)
    assert second.mode == "update"
    assert second.body_patched == ()

    # A drifted remote body is patched back to the rendered abstract on update.
    session.items[first.sb_id]["body"] = "STALE DESCRIPTION"
    third = publish.publish_fabric_child(project, client, registry_path=reg, now=NOW)
    assert "body" in third.body_patched
    assert session.items[first.sb_id]["body"] != "STALE DESCRIPTION"


def test_source_child_create_then_update_skips(tmp_path):
    """A source child re-published with no change goes to update + skips its NCs."""
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    session = FakeSbSession()
    client = make_sb_client(session)

    first = publish.publish_source_child(
        project, client, "era5_land", registry_path=reg, now=NOW
    )
    assert first.mode == "create"

    second = publish.publish_source_child(
        project, client, "era5_land", registry_path=reg, now=NOW
    )
    assert second.mode == "update"
    assert "era5_land_daily_1980.nc" in second.skipped
    assert registry.get_source("era5_land", reg)["sb_id"] == first.sb_id


# ---------------------------------------------------------------------------
# _payload_files flat-namespace collision guard
# ---------------------------------------------------------------------------


def test_payload_files_raises_on_basename_collision(tmp_path):
    """Two staged files with the same basename would collide in ScienceBase's
    flat namespace -- _payload_files must refuse rather than silently overwrite."""
    stage = tmp_path / "stage"
    (stage / "a").mkdir(parents=True)
    (stage / "b").mkdir(parents=True)
    (stage / "a" / "data.nc").write_bytes(b"x")
    (stage / "b" / "data.nc").write_bytes(b"y")
    entries = (
        FileEntry(path="a/data.nc", sha256="0" * 64, size=1, mtime="t"),
        FileEntry(path="b/data.nc", sha256="0" * 64, size=1, mtime="t"),
    )
    result = BuildResult(
        kind="fabric", key="x", stage_dir=stage, mcf={}, entries=entries
    )
    with pytest.raises(ReleaseError, match="flat namespace"):
        publish._payload_files(result)


# ---------------------------------------------------------------------------
# _is_not_found classifier (the tightened 404 detection)
# ---------------------------------------------------------------------------


def test_is_not_found_recognizes_real_not_found_shapes():
    assert publish._is_not_found(Exception("Resource not found")) is True
    assert publish._is_not_found(Exception("Other HTTP error: 404: missing")) is True
    exc = requests.exceptions.HTTPError("boom")
    exc.response = SimpleNamespace(status_code=404)
    assert publish._is_not_found(exc) is True


def test_is_not_found_rejects_misleading_messages():
    """A 5xx whose body merely mentions 'not found' or '404' must NOT classify
    as not-found -- otherwise create_new would mint a duplicate item."""
    assert (
        publish._is_not_found(
            Exception("Other HTTP error: 503: upstream service not found")
        )
        is False
    )
    assert (
        publish._is_not_found(Exception("Other HTTP error: 500: route /v/404 failed"))
        is False
    )
    assert publish._is_not_found(Exception("Unauthorized access")) is False


# ---------------------------------------------------------------------------
# typed-record key invariant
# ---------------------------------------------------------------------------


def test_typed_records_enforce_key_invariant():
    """umbrella must be keyless; source/fabric must carry a key (mirrors
    BuildResult)."""
    with pytest.raises(ValueError, match="key invariant"):
        publish.PublishResult(
            scope="umbrella",
            key="oops",
            sb_id="x",
            mode="create",
            adopted=False,
            uploaded=(),
            skipped=(),
            orphans=(),
            orphans_deleted=(),
            body_patched=(),
            registry_entry={},
        )
    with pytest.raises(ValueError, match="key invariant"):
        publish.ScopeDiff(
            scope="fabric", key=None, sb_id=None, local="missing", remote="missing"
        )
