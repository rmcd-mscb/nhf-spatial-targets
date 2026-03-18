"""Shared test fixtures and helpers."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def setup_workspace(workdir: Path, datastore: Path | None = None) -> None:
    """Create minimal workspace files so ``workspace.load()`` succeeds.

    Parameters
    ----------
    workdir : Path
        Workspace directory (must already exist).
    datastore : Path | None
        Datastore directory for raw data. Defaults to ``workdir / "data" / "raw"``.
    """
    if datastore is None:
        datastore = workdir / "data" / "raw"
    datastore.mkdir(parents=True, exist_ok=True)

    config = {
        "fabric": {"path": "", "id_col": "nhm_id"},
        "datastore": str(datastore),
        "dir_mode": "2775",
    }
    config_path = workdir / "config.yml"
    if not config_path.exists():
        config_path.write_text(yaml.dump(config))

    fabric_path = workdir / "fabric.json"
    if not fabric_path.exists():
        fabric = {
            "bbox_buffered": {
                "minx": -125.1,
                "miny": 23.9,
                "maxx": -65.9,
                "maxy": 50.1,
            }
        }
        fabric_path.write_text(json.dumps(fabric))

    manifest_path = workdir / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps({"sources": {}, "steps": []}))
