"""Tests for nhf-targets init workspace creation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nhf_spatial_targets.init_run import _make_run_id, init_run


# ---------------------------------------------------------------------------
# _make_run_id
# ---------------------------------------------------------------------------

def test_run_id_format():
    run_id = _make_run_id()
    # e.g. "2026-03-11T1500_v0.1.0"
    parts = run_id.split("_v")
    assert len(parts) == 2
    ts, version = parts
    assert len(ts) == 15          # "2026-03-11T1500"
    assert ts[4] == "-"
    assert ts[7] == "-"
    assert ts[10] == "T"
    assert version                # version string non-empty


# ---------------------------------------------------------------------------
# init_run (mocked fabric read)
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_fabric(tmp_path: Path) -> Path:
    """Create a minimal fake GeoPackage file (just needs to exist and be hashable)."""
    p = tmp_path / "fabric.gpkg"
    p.write_bytes(b"fake gpkg content for testing")
    return p


@pytest.fixture()
def fake_config(tmp_path: Path) -> Path:
    p = tmp_path / "pipeline.yml"
    p.write_text("fabric:\n  path: fabric.gpkg\n  id_col: nhm_id\n")
    return p


def _mock_fabric_read(fabric_path, id_col, buffer_deg):
    """Bypass geopandas in unit tests."""
    import hashlib
    h = hashlib.sha256()
    with fabric_path.open("rb") as f:
        h.update(f.read())
    return {
        "path": str(fabric_path),
        "sha256": h.hexdigest(),
        "crs": "EPSG:4326",
        "id_col": id_col,
        "hru_count": 42,
        "bbox": {"minx": -125.0, "miny": 24.0, "maxx": -66.0, "maxy": 50.0},
        "bbox_buffered": {
            "minx": -125.1, "miny": 23.9, "maxx": -65.9, "maxy": 50.1,
        },
        "buffer_deg": buffer_deg,
    }


@patch("nhf_spatial_targets.init_run._fabric_metadata", side_effect=_mock_fabric_read)
@patch("nhf_spatial_targets.init_run._find_reusable_raw", return_value=None)
def test_init_creates_skeleton(mock_reuse, mock_meta, fake_fabric, fake_config, tmp_path):
    workdir = tmp_path / "runs"
    run_dir = init_run(
        fabric_path=fake_fabric,
        id_col="nhm_id",
        config_path=fake_config,
        workdir=workdir,
    )

    assert run_dir.exists()
    assert (run_dir / "fabric.json").exists()
    assert (run_dir / "config.yml").exists()
    assert (run_dir / ".credentials.yml").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "data" / "raw").exists()
    assert (run_dir / "data" / "aggregated").exists()
    assert (run_dir / "targets").exists()
    assert (run_dir / "logs").exists()


@patch("nhf_spatial_targets.init_run._fabric_metadata", side_effect=_mock_fabric_read)
@patch("nhf_spatial_targets.init_run._find_reusable_raw", return_value=None)
def test_fabric_json_contents(mock_reuse, mock_meta, fake_fabric, fake_config, tmp_path):
    run_dir = init_run(
        fabric_path=fake_fabric,
        id_col="nhm_id",
        config_path=fake_config,
        workdir=tmp_path / "runs",
    )
    meta = json.loads((run_dir / "fabric.json").read_text())
    assert "sha256" in meta
    assert meta["id_col"] == "nhm_id"
    assert "bbox" in meta
    assert "bbox_buffered" in meta


@patch("nhf_spatial_targets.init_run._fabric_metadata", side_effect=_mock_fabric_read)
@patch("nhf_spatial_targets.init_run._find_reusable_raw", return_value=None)
def test_manifest_skeleton(mock_reuse, mock_meta, fake_fabric, fake_config, tmp_path):
    run_dir = init_run(
        fabric_path=fake_fabric,
        id_col="nhm_id",
        config_path=fake_config,
        workdir=tmp_path / "runs",
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "run_id" in manifest
    assert manifest["sources"] == {}
    assert manifest["steps"] == []
    assert "fabric" in manifest


@patch("nhf_spatial_targets.init_run._fabric_metadata", side_effect=_mock_fabric_read)
@patch("nhf_spatial_targets.init_run._find_reusable_raw", return_value=None)
def test_raw_subdirs_created_per_source(
    mock_reuse, mock_meta, fake_fabric, fake_config, tmp_path
):
    from nhf_spatial_targets.catalog import sources
    run_dir = init_run(
        fabric_path=fake_fabric,
        id_col="nhm_id",
        config_path=fake_config,
        workdir=tmp_path / "runs",
    )
    raw = run_dir / "data" / "raw"
    for key in sources():
        assert (raw / key).is_dir(), f"Missing raw subdir for source: {key}"
