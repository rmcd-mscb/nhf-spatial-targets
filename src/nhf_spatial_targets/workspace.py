"""Project path resolution and directory helpers."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import yaml

_IS_UNIX = platform.system() != "Windows"


def make_dir(path: Path, *, dir_mode: int | None = None) -> Path:
    """Create a directory (with parents), optionally applying Unix permissions."""
    path.mkdir(parents=True, exist_ok=True)
    if dir_mode is not None and _IS_UNIX:
        os.chmod(path, dir_mode)
    return path


@dataclass(frozen=True)
class Project:
    """Resolved project with validated paths."""

    workdir: Path
    datastore: Path
    config: dict
    fabric: dict
    dir_mode: int | None

    def raw_dir(self, source_key: str) -> Path:
        """Return the raw data directory for a source."""
        return self.datastore / source_key

    def aggregated_dir(self) -> Path:
        """Return the aggregated data directory."""
        return self.workdir / "data" / "aggregated"

    def targets_dir(self) -> Path:
        """Return the targets output directory."""
        return self.workdir / "targets"

    @property
    def manifest_path(self) -> Path:
        """Return path to manifest.json."""
        return self.workdir / "manifest.json"

    @property
    def credentials_path(self) -> Path:
        """Return path to .credentials.yml."""
        return self.workdir / ".credentials.yml"


def load(workdir: Path) -> Project:
    """Load a validated project from config.yml + fabric.json."""
    config_path = workdir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yml not found in {workdir}. "
            f"Run 'nhf-targets init --project-dir {workdir}' first."
        )
    try:
        config = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Cannot parse config.yml in {workdir}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(
            f"config.yml in {workdir} is empty or malformed. "
            f"It must contain YAML key-value pairs."
        )

    if "datastore" not in config:
        raise ValueError(
            f"'datastore' key missing from config.yml in {workdir}. "
            f"This field is required. Edit config.yml and add the datastore path."
        )

    fabric_path = workdir / "fabric.json"
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"fabric.json not found in {workdir}. "
            f"Run 'nhf-targets validate --project-dir {workdir}' first."
        )
    fabric = json.loads(fabric_path.read_text())

    dir_mode_str = config.get("dir_mode")
    if dir_mode_str:
        try:
            dir_mode = int(dir_mode_str, 8)
        except ValueError:
            raise ValueError(
                f"Invalid dir_mode '{dir_mode_str}' in config.yml. "
                f"Expected an octal string like '2775'."
            ) from None
    else:
        dir_mode = None

    return Project(
        workdir=workdir,
        datastore=Path(config["datastore"]),
        config=config,
        fabric=fabric,
        dir_mode=dir_mode,
    )
