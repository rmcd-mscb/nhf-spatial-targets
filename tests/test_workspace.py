"""Tests for nhf_spatial_targets.workspace."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
import yaml

from nhf_spatial_targets import workspace
from nhf_spatial_targets.workspace import Workspace, load, make_dir


def _write_config(
    workdir: Path,
    datastore: Path,
    *,
    dir_mode: str | None = None,
) -> Path:
    """Write a minimal config.yml and return its path."""
    cfg: dict = {"datastore": str(datastore)}
    if dir_mode is not None:
        cfg["dir_mode"] = dir_mode
    config_path = workdir / "config.yml"
    config_path.write_text(yaml.dump(cfg))
    return config_path


def _write_fabric_json(workdir: Path) -> Path:
    """Write a minimal fabric.json and return its path."""
    fabric_path = workdir / "fabric.json"
    fabric_path.write_text(json.dumps({"id": "gfv11", "sha256": "abc123"}))
    return fabric_path


# --- make_dir tests ---


def test_make_dir_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "newdir"
    result = make_dir(target)
    assert result == target
    assert target.is_dir()


def test_make_dir_parents(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    result = make_dir(target)
    assert result == target
    assert target.is_dir()


def test_make_dir_existing_ok(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    result = make_dir(target)
    assert result == target
    assert target.is_dir()


@pytest.mark.skipif(os.name == "nt", reason="Unix-only permission test")
def test_make_dir_applies_mode(tmp_path: Path) -> None:
    target = tmp_path / "moded"
    mode = 0o2775
    make_dir(target, dir_mode=mode)
    actual = stat.S_IMODE(target.stat().st_mode)
    assert actual == mode


def test_make_dir_no_mode_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace, "_IS_UNIX", False)
    target = tmp_path / "windir"
    result = make_dir(target, dir_mode=0o2775)
    assert result == target
    assert target.is_dir()


# --- load tests ---


def test_load_returns_workspace(tmp_path: Path) -> None:
    datastore = tmp_path / "store"
    datastore.mkdir()
    _write_config(tmp_path, datastore)
    _write_fabric_json(tmp_path)

    ws = load(tmp_path)
    assert isinstance(ws, Workspace)
    assert ws.workdir == tmp_path
    assert ws.datastore == datastore
    assert ws.fabric["id"] == "gfv11"
    assert ws.dir_mode is None


def test_load_fails_without_config(tmp_path: Path) -> None:
    _write_fabric_json(tmp_path)
    with pytest.raises(FileNotFoundError, match="config.yml"):
        load(tmp_path)


def test_load_fails_without_fabric_json(tmp_path: Path) -> None:
    _write_config(tmp_path, tmp_path / "store")
    with pytest.raises(FileNotFoundError, match="validate"):
        load(tmp_path)


# --- Workspace accessor tests ---


def test_workspace_raw_dir(tmp_path: Path) -> None:
    datastore = tmp_path / "store"
    datastore.mkdir()
    _write_config(tmp_path, datastore)
    _write_fabric_json(tmp_path)
    ws = load(tmp_path)
    assert ws.raw_dir("ssebop") == datastore / "ssebop"


def test_workspace_aggregated_dir(tmp_path: Path) -> None:
    datastore = tmp_path / "store"
    datastore.mkdir()
    _write_config(tmp_path, datastore)
    _write_fabric_json(tmp_path)
    ws = load(tmp_path)
    assert ws.aggregated_dir() == tmp_path / "data" / "aggregated"


def test_workspace_targets_dir(tmp_path: Path) -> None:
    datastore = tmp_path / "store"
    datastore.mkdir()
    _write_config(tmp_path, datastore)
    _write_fabric_json(tmp_path)
    ws = load(tmp_path)
    assert ws.targets_dir() == tmp_path / "targets"


def test_workspace_manifest_path(tmp_path: Path) -> None:
    datastore = tmp_path / "store"
    datastore.mkdir()
    _write_config(tmp_path, datastore)
    _write_fabric_json(tmp_path)
    ws = load(tmp_path)
    assert ws.manifest_path == tmp_path / "manifest.json"


def test_workspace_credentials_path(tmp_path: Path) -> None:
    datastore = tmp_path / "store"
    datastore.mkdir()
    _write_config(tmp_path, datastore)
    _write_fabric_json(tmp_path)
    ws = load(tmp_path)
    assert ws.credentials_path == tmp_path / ".credentials.yml"


def test_workspace_dir_mode_parsed_as_octal(tmp_path: Path) -> None:
    datastore = tmp_path / "store"
    datastore.mkdir()
    _write_config(tmp_path, datastore, dir_mode="2775")
    _write_fabric_json(tmp_path)
    ws = load(tmp_path)
    assert ws.dir_mode == 0o2775
