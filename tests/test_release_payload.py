"""Tests for nhf_spatial_targets.release.payload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nhf_spatial_targets import catalog
from nhf_spatial_targets.release import payload
from nhf_spatial_targets.release._models import (
    DistributionKind,
    ReleaseError,
    SourceChildPlan,
)
from nhf_spatial_targets.release.checksums import compute_checksums, verify_csv
from nhf_spatial_targets.workspace import Project


def _touch(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _build_project(tmp_path, *, fabric_label=None) -> Project:
    """Synthesize a project tree exercising the payload staging rules.

    Aggregated subdirs:
      - era5_land        publishable catalog key -> kept
      - watergap22d      non-publishable catalog key -> excluded
      - mod16a2          superseded catalog key -> excluded
      - era5_land_sd     not a catalog key (derived variant) -> kept
      - daymet           publishable (metadata_only) catalog key -> kept
    """
    workdir = tmp_path / "proj"
    datastore = tmp_path / "store"
    fabric_file = _touch(tmp_path / "fab" / "myfabric.gpkg", b"GPKG-bytes")

    agg = workdir / "data" / "aggregated"
    _touch(agg / "era5_land" / "era5_land_1980_agg.nc", b"era5-1980")
    _touch(agg / "era5_land" / "era5_land_1981_agg.nc", b"era5-1981")
    _touch(agg / "watergap22d" / "watergap22d_1980_agg.nc", b"wg-held")
    _touch(agg / "mod16a2" / "mod16a2_1980_agg.nc", b"superseded")
    _touch(agg / "era5_land_sd" / "era5_land_sd_1980_agg.nc", b"sd-1980")
    _touch(agg / "daymet" / "daymet_na_1980_agg.nc", b"daymet-1980")

    _touch(workdir / "targets" / "runoff_targets.nc", b"runoff")
    _touch(workdir / "targets" / "aet_targets.nc", b"aet")

    (workdir / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))

    # Datastore consolidated NCs (source-child payloads).
    _touch(datastore / "era5_land" / "daily" / "era5_land_daily_1980.nc", b"d-1980")
    _touch(datastore / "era5_land" / "monthly" / "era5_land_monthly_1980.nc", b"m-1980")
    # daymet is metadata_only: even if NCs exist they must not stage.
    _touch(datastore / "daymet" / "daymet_consolidated.nc", b"should-not-stage")

    config = {"release": {"fabric_label": fabric_label}}
    return Project(
        workdir=workdir,
        datastore=datastore,
        config=config,
        fabric={"path": str(fabric_file)},
        dir_mode=None,
    )


# ---------------------------------------------------------------------------
# fabric child
# ---------------------------------------------------------------------------


def test_fabric_label_derived_from_stem(tmp_path):
    project = _build_project(tmp_path)
    assert payload.resolve_fabric_label(project) == "myfabric"


def test_fabric_label_override(tmp_path):
    project = _build_project(tmp_path, fabric_label="gfv2")
    assert payload.resolve_fabric_label(project) == "gfv2"


def test_stage_fabric_child_tree(tmp_path):
    project = _build_project(tmp_path)
    plan = payload.stage_fabric_child(project)
    stage = plan.stage_dir
    assert stage == project.release_build_dir / "fabric" / "myfabric"

    # fabric.gpkg is symlinked by default and resolves to source content.
    gpkg = stage / "fabric.gpkg"
    assert gpkg.is_symlink()
    assert gpkg.read_bytes() == b"GPKG-bytes"

    # aggregated NCs land under aggregated/<source>/, symlinked.
    era = stage / "aggregated" / "era5_land" / "era5_land_1980_agg.nc"
    assert era.is_symlink()
    assert era.read_bytes() == b"era5-1980"

    # target NCs land under targets/.
    assert (stage / "targets" / "runoff_targets.nc").is_symlink()
    assert (stage / "targets" / "aet_targets.nc").read_bytes() == b"aet"


def test_manifest_is_copied_not_symlinked(tmp_path):
    project = _build_project(tmp_path)
    plan = payload.stage_fabric_child(project)
    staged = plan.stage_dir / "manifest.json"
    assert staged.exists()
    assert not staged.is_symlink()  # snapshot copy, never a link
    assert staged.read_bytes() == project.manifest_path.read_bytes()


def test_stage_fabric_child_missing_manifest_is_hard_error(tmp_path):
    """A fabric child must never stage without its provenance snapshot.

    The earlier ``if manifest_src.exists()`` soft skip would emit a child with
    no manifest at all; PR-3 makes that a hard error so staging fails loudly
    and the operator runs validate / rebuild-manifest first.
    """
    project = _build_project(tmp_path)
    project.manifest_path.unlink()
    with pytest.raises(ReleaseError, match="manifest.json"):
        payload.stage_fabric_child(project, copy=True)


def test_watergap22d_excluded_from_fabric(tmp_path):
    project = _build_project(tmp_path)
    plan = payload.stage_fabric_child(project)
    assert not (plan.stage_dir / "aggregated" / "watergap22d").exists()
    # The held-back source must not appear in the plan's file list either.
    # (Check the source-dir segment, not the full path string -- pytest's
    # tmp dir is named after this test and contains "watergap22d".)
    assert all(p.parent.name != "watergap22d" for p in plan.aggregated_files)


def test_era5_land_sd_variant_kept_in_fabric(tmp_path):
    project = _build_project(tmp_path)
    plan = payload.stage_fabric_child(project)
    sd = plan.stage_dir / "aggregated" / "era5_land_sd" / "era5_land_sd_1980_agg.nc"
    assert sd.is_symlink()


def test_daymet_aggregated_kept_in_fabric(tmp_path):
    """daymet is metadata_only as a source child, but its aggregated NCs
    still ship in fabric children (they are derived products)."""
    project = _build_project(tmp_path)
    plan = payload.stage_fabric_child(project)
    assert (plan.stage_dir / "aggregated" / "daymet" / "daymet_na_1980_agg.nc").exists()


def test_stage_fabric_child_copy_mode(tmp_path):
    project = _build_project(tmp_path)
    plan = payload.stage_fabric_child(project, copy=True)
    gpkg = plan.stage_dir / "fabric.gpkg"
    assert not gpkg.is_symlink()
    assert gpkg.read_bytes() == b"GPKG-bytes"
    era = plan.stage_dir / "aggregated" / "era5_land" / "era5_land_1980_agg.nc"
    assert not era.is_symlink()
    assert era.read_bytes() == b"era5-1980"


def test_reserved_metadata_not_written(tmp_path):
    project = _build_project(tmp_path)
    plan = payload.stage_fabric_child(project)
    for name in plan.reserved_files:
        assert not (plan.stage_dir / name).exists()


def test_stage_fabric_child_idempotent(tmp_path):
    project = _build_project(tmp_path)
    payload.stage_fabric_child(project)
    # Re-staging the same inputs must not raise on existing symlinks.
    plan = payload.stage_fabric_child(project)
    era = plan.stage_dir / "aggregated" / "era5_land" / "era5_land_1980_agg.nc"
    assert era.read_bytes() == b"era5-1980"


# ---------------------------------------------------------------------------
# source children
# ---------------------------------------------------------------------------


def test_stage_source_child_data_preserves_subpaths(tmp_path):
    project = _build_project(tmp_path)
    plan = payload.stage_source_child(project, "era5_land")
    assert plan.distribution_kind == "data"
    daily = plan.stage_dir / "daily" / "era5_land_daily_1980.nc"
    monthly = plan.stage_dir / "monthly" / "era5_land_monthly_1980.nc"
    assert daily.is_symlink() and daily.read_bytes() == b"d-1980"
    assert monthly.is_symlink() and monthly.read_bytes() == b"m-1980"


def test_stage_source_child_metadata_only_stages_no_data(tmp_path):
    project = _build_project(tmp_path)
    plan = payload.stage_source_child(project, "daymet")
    assert plan.distribution_kind == "metadata_only"
    assert plan.data_files == ()
    # Stage dir exists but carries no NetCDF payload.
    assert plan.stage_dir.is_dir()
    assert list(plan.stage_dir.rglob("*.nc")) == []


# ---------------------------------------------------------------------------
# sources_used + stage_all + umbrella
# ---------------------------------------------------------------------------


def test_sources_used_filters_to_publishable_catalog_keys(tmp_path):
    project = _build_project(tmp_path)
    used = payload.sources_used(project)
    assert "era5_land" in used
    assert "daymet" in used
    assert "watergap22d" not in used  # non-publishable
    assert "mod16a2" not in used  # superseded
    assert "era5_land_sd" not in used  # not a catalog key


def test_stage_umbrella_metadata_only(tmp_path):
    project = _build_project(tmp_path)
    plan = payload.stage_umbrella(project)
    assert plan.stage_dir == project.release_build_dir / "umbrella"
    assert plan.stage_dir.is_dir()
    assert list(plan.stage_dir.iterdir()) == []


def test_stage_all_returns_payload(tmp_path):
    project = _build_project(tmp_path)
    result = payload.stage_all(project)
    assert result.fabric is not None
    assert result.umbrella is not None
    staged_keys = {s.source_key for s in result.sources}
    assert "era5_land" in staged_keys
    assert "daymet" in staged_keys
    assert "watergap22d" not in staged_keys


def test_stage_missing_aggregated_and_targets(tmp_path):
    """A project without aggregated/targets dirs stages just fabric+manifest."""
    workdir = tmp_path / "proj"
    workdir.mkdir()
    fabric_file = _touch(tmp_path / "f.gpkg", b"gp")
    (workdir / "manifest.json").write_text("{}")
    project = Project(
        workdir=workdir,
        datastore=tmp_path / "store",
        config={},
        fabric={"path": str(fabric_file)},
        dir_mode=None,
    )
    plan = payload.stage_fabric_child(project)
    assert plan.aggregated_files == ()
    assert plan.target_files == ()
    assert (plan.stage_dir / "fabric.gpkg").read_bytes() == b"gp"


def test_superseded_source_excluded_from_fabric(tmp_path):
    """A superseded catalog key (publishable flag True but superseded_by
    set) must not ship in a fabric child."""
    project = _build_project(tmp_path)
    plan = payload.stage_fabric_child(project)
    assert not (plan.stage_dir / "aggregated" / "mod16a2").exists()
    assert all(p.parent.name != "mod16a2" for p in plan.aggregated_files)


# ---------------------------------------------------------------------------
# integration with checksums + re-stage semantics
# ---------------------------------------------------------------------------


def test_checksums_over_symlinked_stage_hash_target_content(tmp_path):
    """The default symlink stage + compute_checksums must hash the link
    *target's* content/size, not the link itself."""
    project = _build_project(tmp_path)
    plan = payload.stage_fabric_child(project)  # symlinks
    entries = compute_checksums(plan.stage_dir)
    era = next(e for e in entries if e.path.endswith("era5_land/era5_land_1980_agg.nc"))
    assert era.sha256 == hashlib.sha256(b"era5-1980").hexdigest()
    assert era.size == len(b"era5-1980")
    verify_csv(plan.stage_dir)  # no raise


def test_restage_copy_mode_picks_up_changed_content(tmp_path):
    project = _build_project(tmp_path)
    payload.stage_fabric_child(project, copy=True)
    (project.targets_dir() / "runoff_targets.nc").write_bytes(b"runoff-v2")
    plan = payload.stage_fabric_child(project, copy=True)
    assert (
        plan.stage_dir / "targets" / "runoff_targets.nc"
    ).read_bytes() == b"runoff-v2"


def test_restage_switches_copy_to_symlink(tmp_path):
    project = _build_project(tmp_path)
    payload.stage_fabric_child(project, copy=True)  # leaves real copies
    plan = payload.stage_fabric_child(project, copy=False)  # must replace with links
    assert (plan.stage_dir / "targets" / "runoff_targets.nc").is_symlink()
    assert (plan.stage_dir / "fabric.gpkg").is_symlink()


# ---------------------------------------------------------------------------
# SourceChildPlan invariant + DistributionKind sync
# ---------------------------------------------------------------------------


def test_metadata_only_plan_with_data_files_raises(tmp_path):
    with pytest.raises(ValueError, match="metadata_only.*must carry no"):
        SourceChildPlan(
            source_key="daymet",
            distribution_kind="metadata_only",
            stage_dir=tmp_path / "s",
            data_files=(tmp_path / "x.nc",),
        )


def test_distribution_kind_literal_matches_catalog_tokens():
    from typing import get_args

    assert set(get_args(DistributionKind)) == catalog.DISTRIBUTION_KIND_TOKENS
