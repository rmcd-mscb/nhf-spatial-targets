"""Tests for nhf-targets init project creation (simplified)."""

from __future__ import annotations

import pytest
import yaml

from nhf_spatial_targets.init_run import init_project


def test_init_creates_skeleton(tmp_path):
    workdir = tmp_path / "my-workspace"
    init_project(workdir)
    assert workdir.is_dir()
    assert (workdir / "config.yml").exists()
    assert (workdir / ".credentials.yml").exists()
    assert (workdir / "data" / "aggregated").is_dir()
    assert (workdir / "targets").is_dir()
    assert (workdir / "logs").is_dir()


def test_init_config_template_has_required_fields(tmp_path):
    workdir = tmp_path / "ws"
    init_project(workdir)
    config = yaml.safe_load((workdir / "config.yml").read_text())
    assert "fabric" in config
    assert "path" in config["fabric"]
    assert "id_col" in config["fabric"]
    assert "buffer_deg" in config["fabric"]
    assert "datastore" in config
    assert "dir_mode" in config
    assert "targets" in config


def test_init_credentials_template(tmp_path):
    workdir = tmp_path / "ws"
    init_project(workdir)
    creds = yaml.safe_load((workdir / ".credentials.yml").read_text())
    assert "nasa_earthdata" in creds
    assert "username" in creds["nasa_earthdata"]


def test_init_existing_workdir_raises(tmp_path):
    workdir = tmp_path / "ws"
    init_project(workdir)
    with pytest.raises(FileExistsError, match="already exists"):
        init_project(workdir)


def test_init_no_fabric_json(tmp_path):
    workdir = tmp_path / "ws"
    init_project(workdir)
    assert not (workdir / "fabric.json").exists()


def test_init_no_manifest_json(tmp_path):
    workdir = tmp_path / "ws"
    init_project(workdir)
    assert not (workdir / "manifest.json").exists()


def test_init_no_data_raw(tmp_path):
    workdir = tmp_path / "ws"
    init_project(workdir)
    assert not (workdir / "data" / "raw").exists()
