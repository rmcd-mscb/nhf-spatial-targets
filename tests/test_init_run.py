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


# --- commented-out stubs in the init template (#182 follow-up) --------------


def test_init_config_template_parses_as_valid_yaml(tmp_path):
    """The shipped template (incl. the new commented stubs) is valid YAML."""
    workdir = tmp_path / "ws"
    init_project(workdir)
    cfg = yaml.safe_load((workdir / "config.yml").read_text())
    assert cfg["fabric"]["id_col"] == "nhm_id"
    # The commented stubs are not active by default:
    assert "token" not in cfg["fabric"]
    assert "representative_points" not in cfg


def test_init_config_template_carries_fabric_token_stub(tmp_path):
    """The fabric.token stub is present (commented) and points operators
    at FABRIC_SCOPE_TOKENS — without this we silently skipped Margulis on
    Oregon (#182)."""
    workdir = tmp_path / "ws"
    init_project(workdir)
    text = (workdir / "config.yml").read_text()
    assert "# token: or" in text
    assert "FABRIC_SCOPE_TOKENS" in text


def test_init_config_template_carries_representative_points_stub(tmp_path):
    """The representative_points stub teaches operators about the per-target
    notebook override before lookup_hrus_by_points raises mid-render (#192)."""
    workdir = tmp_path / "ws"
    init_project(workdir)
    text = (workdir / "config.yml").read_text()
    assert "# representative_points:" in text
    for tgt in (
        "aet",
        "runoff",
        "recharge",
        "soil_moisture",
        "snow_covered_area",
        "swe",
    ):
        assert tgt in text


def test_init_config_template_carries_release_stub(tmp_path):
    """The release: stub teaches operators about the ScienceBase data-release
    workflow before they need it (#241/#243)."""
    workdir = tmp_path / "ws"
    init_project(workdir)
    text = (workdir / "config.yml").read_text()
    assert "# release:" in text
    # Nested keys live under `#   ipds_number:` / `#   doi:` with extra
    # indentation. Check by-key with re-style anchoring.
    assert "ipds_number" in text
    assert "docs/data_release/" in text
    # Commented-out by default (operator opts in by uncommenting).
    cfg = yaml.safe_load(text)
    assert "release" not in cfg


def test_init_config_template_carries_sca_multi_source_knobs(tmp_path):
    """The SCA block ships ua_swe as a second source plus the three multi-source
    knobs (depth_threshold_mm / forced_zero_combined / min_sources_for_bound) so
    new projects build the two-source bound and discover the tuning keys (#237)."""
    workdir = tmp_path / "ws"
    init_project(workdir)
    cfg = yaml.safe_load((workdir / "config.yml").read_text())
    sca = cfg["targets"]["snow_covered_area"]
    assert sca["sources"] == ["mod10c1_v061", "ua_swe"]
    assert sca["depth_threshold_mm"] == 1.0
    assert sca["forced_zero_combined"] is True
    assert sca["min_sources_for_bound"] == 1
