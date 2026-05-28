"""Offline create-vs-update decision-matrix tests for release.publish.

Every test injects a :class:`tests.conftest.FakeSbSession` (no network) and a
tmp registry path, and drives the publish orchestration through the plan's
idempotency matrix:

    registry sb_id + found              -> update
    registry sb_id + not-found          -> fatal (StaleRegistryError); create_new -> create
    no sb_id + one same-title child     -> adopt (update) + write sb_id to registry
    no sb_id + zero matches             -> create
    no sb_id + multiple same-title      -> SbClientError (ambiguous)
    per-file: unchanged                 -> skipped; changed -> re-uploaded
    orphan remote file                  -> warn+leave; delete_orphans -> deleted

It also asserts registry.put_* is called with the right counts/bytes and that
the flock read-merge-write preserves sibling entries (the umbrella survives a
fabric publish).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nhf_spatial_targets.release import publish, registry
from nhf_spatial_targets.release.build import build_fabric_child
from nhf_spatial_targets.release.sb_client import SbClientError
from tests.conftest import FakeSbSession, build_release_project, make_sb_client

# Clock-injected so each build renders byte-identical metadata.
NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
LABEL = "gfv_test"


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


def _publish(project, session, reg, **kwargs):
    return publish.publish_fabric_child(
        project, make_sb_client(session), registry_path=reg, now=NOW, **kwargs
    )


def _fabric_title(project) -> str:
    """The title the fabric child would publish under (for adopt seeding)."""
    return build_fabric_child(project, now=NOW).mcf["identification"]["title"]


# ---------------------------------------------------------------------------
# create / update / adopt
# ---------------------------------------------------------------------------


def test_first_publish_creates_then_second_updates(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    session = FakeSbSession()

    first = _publish(project, session, reg)
    assert first.mode == "create"
    assert first.adopted is False
    assert first.sb_id  # a real id was assigned
    # Every payload file was sent on create (nothing skipped).
    assert first.skipped == ()
    assert "runoff_targets.nc" in first.uploaded
    assert len(session.created) == 1

    # The registry now records the child under the freshly-minted sb_id.
    entry = registry.get_fabric(LABEL, reg)
    assert entry["sb_id"] == first.sb_id

    # Second publish sees the registry sb_id, fetches the item -> update.
    second = _publish(project, session, reg)
    assert second.mode == "update"
    assert second.adopted is False
    assert second.sb_id == first.sb_id
    assert len(session.created) == 1  # no second create


def test_registry_sb_id_not_found_is_fatal(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    # Registry points at an item ScienceBase does not have.
    registry.put_fabric(
        LABEL,
        sb_id="GONE",
        uploaded_utc="2026-01-01T00:00:00+00:00",
        file_count=1,
        total_bytes=1,
        manifest_sha256="deadbeef",
        path=reg,
    )
    session = FakeSbSession()  # store has no "GONE"

    with pytest.raises(publish.StaleRegistryError, match="stale"):
        _publish(project, session, reg)


def test_create_new_overrides_stale_registry(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    registry.put_fabric(
        LABEL,
        sb_id="GONE",
        uploaded_utc="2026-01-01T00:00:00+00:00",
        file_count=1,
        total_bytes=1,
        manifest_sha256="deadbeef",
        path=reg,
    )
    session = FakeSbSession()

    result = _publish(project, session, reg, create_new=True)
    assert result.mode == "create"
    assert result.sb_id != "GONE"
    # The registry id was overwritten with the fresh one.
    assert registry.get_fabric(LABEL, reg)["sb_id"] == result.sb_id


def test_adopts_single_same_title_child_and_writes_registry(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    session = FakeSbSession()
    # An existing child under the umbrella with the exact title, but no
    # registry entry yet (e.g. published out-of-band): publish must adopt it.
    session.seed_item("ADOPT", title=_fabric_title(project), parent_id="UMB")

    result = _publish(project, session, reg)
    assert result.mode == "update"
    assert result.adopted is True
    assert result.sb_id == "ADOPT"
    assert len(session.created) == 0  # adopted, not created
    # The registry now records the adopted id.
    assert registry.get_fabric(LABEL, reg)["sb_id"] == "ADOPT"


def test_zero_title_matches_creates(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    session = FakeSbSession()
    # A child with a *different* title is not a match.
    session.seed_item("OTHER", title="Some other dataset", parent_id="UMB")

    result = _publish(project, session, reg)
    assert result.mode == "create"
    assert result.adopted is False


def test_multiple_same_title_children_is_ambiguous(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    session = FakeSbSession()
    title = _fabric_title(project)
    session.seed_item("DUP1", title=title, parent_id="UMB")
    session.seed_item("DUP2", title=title, parent_id="UMB")

    with pytest.raises(SbClientError, match="disambiguate"):
        _publish(project, session, reg)


# ---------------------------------------------------------------------------
# per-file skip vs replace
# ---------------------------------------------------------------------------


def test_per_file_skip_then_replace_on_content_change(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    session = FakeSbSession()

    _publish(project, session, reg)  # create: uploads everything

    # No change -> the unchanged data NC is skipped on re-publish.
    second = _publish(project, session, reg)
    assert second.mode == "update"
    assert "runoff_targets.nc" in second.skipped
    assert "aet_targets.nc" in second.skipped
    sha_before = registry.get_fabric(LABEL, reg)["manifest_sha256"]

    # Change one source file's content -> only that file re-uploads.
    (project.workdir / "targets" / "runoff_targets.nc").write_bytes(b"runoff-v2")
    third = _publish(project, session, reg)
    assert "runoff_targets.nc" in third.uploaded
    assert "aet_targets.nc" in third.skipped
    # The recorded payload fingerprint is content-sensitive: it must change.
    assert registry.get_fabric(LABEL, reg)["manifest_sha256"] != sha_before


def test_empty_item_response_is_treated_as_stale(tmp_path):
    """A registry sb_id whose get_item returns an empty/falsy item (vs raising)
    is the same stale condition -- fatal unless create_new."""
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    registry.put_fabric(
        LABEL,
        sb_id="EMPTY",
        uploaded_utc="2026-01-01T00:00:00+00:00",
        file_count=1,
        total_bytes=1,
        manifest_sha256="deadbeef",
        path=reg,
    )
    session = FakeSbSession()
    session.items["EMPTY"] = {}  # get_item returns a falsy item

    with pytest.raises(publish.StaleRegistryError):
        _publish(project, session, reg)

    result = _publish(project, session, reg, create_new=True)
    assert result.mode == "create"
    assert result.sb_id != "EMPTY"


# ---------------------------------------------------------------------------
# orphan handling
# ---------------------------------------------------------------------------


def test_orphan_is_warned_and_left_by_default(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    session = FakeSbSession()

    first = _publish(project, session, reg)
    # A file present on the remote item but never in the staged payload.
    session.items[first.sb_id]["files"].append(
        {
            "name": "legacy_extra.nc",
            "size": 3,
            "checksum": {"type": "MD5", "value": "x"},
        }
    )

    result = _publish(project, session, reg)
    assert "legacy_extra.nc" in result.orphans
    assert result.orphans_deleted == ()
    assert session.deleted == []  # nothing deleted by default


def test_orphan_is_deleted_with_delete_orphans(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg)
    session = FakeSbSession()

    first = _publish(project, session, reg)
    session.items[first.sb_id]["files"].append(
        {
            "name": "legacy_extra.nc",
            "size": 3,
            "checksum": {"type": "MD5", "value": "x"},
        }
    )

    result = _publish(project, session, reg, delete_orphans=True)
    assert "legacy_extra.nc" in result.orphans_deleted
    assert ("legacy_extra.nc", first.sb_id) in session.deleted


# ---------------------------------------------------------------------------
# registry put: counts/bytes + flock merge preserves siblings
# ---------------------------------------------------------------------------


def test_registry_records_counts_bytes_and_preserves_umbrella(tmp_path):
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)
    _seed_umbrella(reg, sb_id="UMB")
    session = FakeSbSession()

    result = _publish(project, session, reg)

    entry = registry.get_fabric(LABEL, reg)
    # file_count / total_bytes match the staged payload publish uploaded.
    assert entry["file_count"] == len(result.uploaded) + len(result.skipped)
    assert entry["total_bytes"] > 0
    assert len(entry["manifest_sha256"]) == 64  # a SHA-256 hex digest

    # The flock read-merge-write left the umbrella entry untouched.
    assert registry.get_umbrella(reg)["sb_id"] == "UMB"


def test_child_publish_without_umbrella_is_fatal(tmp_path):
    """A child can't be parented without a recorded umbrella sb_id."""
    project = build_release_project(tmp_path)
    reg = _registry(tmp_path)  # no umbrella seeded
    session = FakeSbSession()

    with pytest.raises(publish.PreflightError, match="umbrella"):
        _publish(project, session, reg)
