"""Tests for the workspace defaults integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from nhf_spatial_targets.workspace import load


def _write_project(tmp_path: Path, *, config: dict, fabric: dict | None = None) -> Path:
    workdir = tmp_path / "proj"
    workdir.mkdir()
    (workdir / "config.yml").write_text(yaml.safe_dump(config))
    (workdir / "fabric.json").write_text(json.dumps(fabric or {"id_col": "nhm_id"}))
    return workdir


def test_load_applies_defaults(tmp_path: Path):
    workdir = _write_project(
        tmp_path,
        config={
            "datastore": str(tmp_path / "store"),
            "fabric": {"path": "/some/fabric.gpkg", "id_col": "nhm_id"},
        },
    )
    project = load(workdir)
    # area_crs default applied:
    assert project.area_crs == "EPSG:5070"
    # Per-target merge available via Project.target():
    runoff = project.target("runoff")
    assert runoff["nn_fill"] is True
    assert runoff["sources"] == [
        "era5_land",
        "gldas_noah_v21_monthly",
        "mwbm_climgrid",
    ]


def test_load_user_overrides_area_crs(tmp_path: Path):
    workdir = _write_project(
        tmp_path,
        config={
            "datastore": str(tmp_path / "store"),
            "fabric": {
                "path": "/p",
                "id_col": "nhm_id",
                "area_crs": "EPSG:3338",
            },
        },
    )
    project = load(workdir)
    assert project.area_crs == "EPSG:3338"


def test_load_target_user_sources_replaces_default(tmp_path: Path):
    workdir = _write_project(
        tmp_path,
        config={
            "datastore": str(tmp_path / "store"),
            "fabric": {"path": "/p", "id_col": "nhm_id"},
            "targets": {"runoff": {"sources": ["era5_land"]}},
        },
    )
    project = load(workdir)
    assert project.target("runoff")["sources"] == ["era5_land"]


def test_load_unknown_target_raises(tmp_path: Path):
    workdir = _write_project(
        tmp_path,
        config={
            "datastore": str(tmp_path / "store"),
            "fabric": {"path": "/p", "id_col": "nhm_id"},
        },
    )
    project = load(workdir)
    with pytest.raises(KeyError, match="not_a_target"):
        project.target("not_a_target")
