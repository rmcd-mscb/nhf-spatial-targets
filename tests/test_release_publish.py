"""Offline pre-flight + happy-path tests for release.publish.

Pre-flight gates must each fail loudly *before* anything is published; the
happy paths cover create and update for the umbrella, a source child, and a
fabric child against an in-memory ScienceBase (the :class:`FakeSbSession`).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nhf_spatial_targets.release import publish, registry
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
