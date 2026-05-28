"""Tests for nhf_spatial_targets.release.payload."""

from __future__ import annotations

import json
from pathlib import Path

from nhf_spatial_targets.release import payload
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
