"""Offline pre-flight + happy-path tests for release.publish.

Pre-flight gates must each fail loudly *before* anything is published; the
happy paths cover create and update for the umbrella, a source child, and a
fabric child against an in-memory ScienceBase (the :class:`FakeSbSession`).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import requests

from nhf_spatial_targets.rebuild_manifest import rebuild_manifest
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
# Provenance completeness gate (verify-don't-mutate) -- PR-3 / #279
# ---------------------------------------------------------------------------


def test_complete_manifest_passes(tmp_path):
    """A release-ready project (complete, rebuilt manifest) clears the gate."""
    project = build_release_project(tmp_path)
    publish._preflight_provenance_complete(project)  # does not raise


def test_under_report_aggregate_raises(tmp_path):
    """An aggregated source dir on disk but absent from the manifest is a
    fatal under-report (the #277 case)."""
    project = build_release_project(tmp_path)
    new_agg = project.aggregated_dir() / "gldas_noah_v21_monthly"
    new_agg.mkdir(parents=True)
    (new_agg / "gldas_noah_v21_monthly_2000_agg.nc").write_bytes(b"g")
    # Match the source name (not the universal rebuild-manifest hint) so this
    # genuinely pins the aggregate-dir arm rather than any PreflightError.
    with pytest.raises(publish.PreflightError, match="gldas_noah_v21_monthly"):
        publish._preflight_provenance_complete(project)


def test_under_report_consolidate_raises(tmp_path):
    """A consolidated datastore source on disk but absent from the manifest is
    a fatal under-report (the #278 case)."""
    project = build_release_project(tmp_path)
    new_src = project.datastore / "merra2" / "monthly"
    new_src.mkdir(parents=True)
    (new_src / "merra2_2000.nc").write_bytes(b"m")
    # Match the source name so this pins the datastore arm specifically.
    with pytest.raises(publish.PreflightError, match="merra2"):
        publish._preflight_provenance_complete(project)


def test_drift_between_disk_and_projection_refuses(tmp_path):
    """An on-disk manifest whose steps[] differ from the deterministic
    projection (here: a phantom extra step) is refused as drift, even though
    every source is still present."""
    project = build_release_project(tmp_path)
    manifest = json.loads(project.manifest_path.read_text())
    manifest["steps"].append(
        {"kind": "aggregate", "source_key": "ghost", "outputs": []}
    )
    project.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(publish.PreflightError, match="drift"):
        publish._preflight_provenance_complete(project)


def test_published_target_without_step_raises(tmp_path):
    """A target NC on disk with no matching ``target`` step in the manifest is
    a fatal completeness failure (Pillar 3 assertion (e))."""
    project = build_release_project(tmp_path)
    manifest = json.loads(project.manifest_path.read_text())
    manifest["steps"] = [s for s in manifest["steps"] if s.get("kind") != "target"]
    project.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(publish.PreflightError, match="no matching 'target' step"):
        publish._preflight_provenance_complete(project)


def test_allow_incomplete_sources_bypasses(tmp_path, caplog):
    """``allow_incomplete_sources`` downgrades the source/drift failure to a
    logged warning instead of raising; the warning carries the problem detail
    (here: the under-reported source name), not just the override literal."""
    project = build_release_project(tmp_path)
    new_agg = project.aggregated_dir() / "gldas_noah_v21_monthly"
    new_agg.mkdir(parents=True)
    (new_agg / "gldas_noah_v21_monthly_2000_agg.nc").write_bytes(b"g")
    with caplog.at_level(logging.WARNING):
        publish._preflight_provenance_complete(
            project, allow_incomplete_sources=True
        )  # does not raise
    warnings = "\n".join(r.message for r in caplog.records)
    assert "allow-incomplete-sources" in warnings
    assert "gldas_noah_v21_monthly" in warnings  # the downgraded problem is shown


# --- always-fatal structural checks: fatal AND not bypassable by the override ---


def test_schema_behind_is_fatal_even_with_override(tmp_path):
    """A behind/unexpected manifest_schema_version is fatal and cannot be
    waved through by --allow-incomplete-sources."""
    project = build_release_project(tmp_path)
    manifest = json.loads(project.manifest_path.read_text())
    manifest["manifest_schema_version"] = 0
    project.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(publish.PreflightError, match="schema version"):
        publish._preflight_provenance_complete(project, allow_incomplete_sources=True)


def test_missing_fabric_is_fatal_even_with_override(tmp_path):
    """A manifest with no fabric block is fatal and not bypassable."""
    project = build_release_project(tmp_path)
    manifest = json.loads(project.manifest_path.read_text())
    manifest["fabric"] = None
    project.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(publish.PreflightError, match="no fabric block"):
        publish._preflight_provenance_complete(project, allow_incomplete_sources=True)


def test_empty_steps_is_fatal_even_with_override(tmp_path):
    """An empty steps[] (no lineage) is fatal and not bypassable."""
    project = build_release_project(tmp_path)
    manifest = json.loads(project.manifest_path.read_text())
    manifest["steps"] = []
    project.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(publish.PreflightError, match="no lineage"):
        publish._preflight_provenance_complete(project, allow_incomplete_sources=True)


def test_live_captured_manifest_refused_then_passes_after_rebuild(tmp_path):
    """The intended verify-don't-mutate workflow: a live-captured manifest
    carries richer metadata than the deterministic projection (real command /
    timestamp / params / sha256, no ``provenance`` tag), so it is refused as
    drift even with no source/file divergence. Running rebuild-manifest
    regenerates it as the projection, after which the gate passes -- so the
    *published* manifest is always the deterministic projection (Pillar 6)."""
    project = build_release_project(tmp_path)
    manifest = json.loads(project.manifest_path.read_text())
    # A command string the projection would never emit (it uses
    # "rebuild-manifest:<kind>"); pure capture-vs-rebuild drift, no missing
    # source or target.
    manifest["steps"][0]["command"] = "agg era5-land"
    project.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(publish.PreflightError, match="drift"):
        publish._preflight_provenance_complete(project)

    rebuild_manifest(project, dry_run=False)  # operator's explicit action
    publish._preflight_provenance_complete(project)  # now passes


# ---------------------------------------------------------------------------
# config.effective.yml staleness gate (verify-don't-mutate) -- PR-4 / #279
# ---------------------------------------------------------------------------


def test_effective_config_current_passes(tmp_path):
    """A freshly-seeded project (config.effective.yml stamped to match
    config.yml) clears the staleness gate."""
    project = build_release_project(tmp_path)
    publish._preflight_effective_config_current(project)  # does not raise


def test_effective_config_hash_mismatch_is_fatal(tmp_path):
    """Editing config.yml after generating config.effective.yml makes the
    recorded source_config_sha256 stale -> fatal (the wrong-window fossil
    the spec's Pillar 5 is built to catch)."""
    project = build_release_project(tmp_path)
    # Edit config.yml without re-running validate: the hash now diverges.
    (project.workdir / "config.yml").write_text("datastore: /somewhere/else\n")
    with pytest.raises(publish.PreflightError, match="stale"):
        publish._preflight_effective_config_current(project)


def test_effective_config_missing_file_is_fatal(tmp_path):
    """No config.effective.yml at all (never generated / deleted) -> fatal."""
    project = build_release_project(tmp_path)
    eff = project.workdir / "config.effective.yml"
    eff.chmod(0o644)  # written read-only (0o444); make it deletable
    eff.unlink()
    with pytest.raises(publish.PreflightError, match="not found"):
        publish._preflight_effective_config_current(project)


def test_effective_config_missing_meta_block_is_fatal(tmp_path):
    """A pre-PR-4 effective config (no _effective_config_meta stamp) -> fatal."""
    project = build_release_project(tmp_path)
    eff = project.workdir / "config.effective.yml"
    eff.chmod(0o644)
    eff.write_text("# AUTOGENERATED\ndatastore: /d\n")  # no meta block
    with pytest.raises(publish.PreflightError, match="_effective_config_meta"):
        publish._preflight_effective_config_current(project)


def test_effective_config_schema_behind_is_fatal(tmp_path):
    """A stamped-but-behind effective_config_schema_version -> fatal."""
    import yaml

    project = build_release_project(tmp_path)
    eff = project.workdir / "config.effective.yml"
    body = yaml.safe_load(eff.read_text())
    body["_effective_config_meta"]["effective_config_schema_version"] = 0
    eff.chmod(0o644)
    eff.write_text(yaml.safe_dump(body))
    with pytest.raises(publish.PreflightError, match="behind"):
        publish._preflight_effective_config_current(project)


def test_effective_config_gate_never_mutates(tmp_path):
    """Verify-don't-mutate: the gate reads and refuses, never regenerates the
    file (the same discipline as the manifest gate)."""
    project = build_release_project(tmp_path)
    eff = project.workdir / "config.effective.yml"
    (project.workdir / "config.yml").write_text("datastore: /elsewhere\n")
    before = eff.read_text()
    with pytest.raises(publish.PreflightError):
        publish._preflight_effective_config_current(project)
    assert eff.read_text() == before  # untouched


def test_effective_config_gate_has_no_override(tmp_path):
    """Unconditionally fatal: unlike the manifest provenance gate, the
    effective-config gate takes no allow_incomplete_sources override."""
    import inspect

    sig = inspect.signature(publish._preflight_effective_config_current)
    assert list(sig.parameters) == ["project"]


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
