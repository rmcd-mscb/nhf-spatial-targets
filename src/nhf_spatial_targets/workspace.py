"""Project path resolution and directory helpers."""

from __future__ import annotations

import copy
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import yaml

from nhf_spatial_targets.defaults import apply_defaults

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

    @property
    def area_crs(self) -> str:
        """Equal-area CRS used for HRU area + NN-fill distances."""
        return self.config["fabric"]["area_crs"]

    @property
    def id_col(self) -> str:
        """Fabric ID column name (also the HRU dim in aggregated NCs)."""
        return self.config["fabric"]["id_col"]

    def target(self, name: str) -> dict:
        """Return the merged config sub-dict for a calibration target.

        Returns a deep copy so callers cannot mutate the underlying
        ``Project.config`` by editing the returned dict. Raises
        ``KeyError`` if ``name`` is not a recognized target.
        """
        targets = self.config.get("targets", {})
        if name not in targets:
            raise KeyError(f"Unknown target '{name}'. Known: {sorted(targets.keys())}")
        return copy.deepcopy(targets[name])


def load(workdir: Path) -> Project:
    """Load a project, merging user config over defaults from ``defaults.py``.

    Reads ``config.yml`` and ``fabric.json``, deep-merges the user config over
    :data:`nhf_spatial_targets.defaults.DEFAULTS` (user wins at every leaf;
    lists replace wholesale; ``None`` user values fall through to defaults),
    and returns a :class:`Project` carrying the merged config.

    Required keys (``datastore``, ``fabric.path``) are checked here so that
    every consumer of ``Project`` can assume they exist.
    """
    config_path = workdir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yml not found in {workdir}. "
            f"Run 'nhf-targets init --project-dir {workdir}' first."
        )
    try:
        user_config = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Cannot parse config.yml in {workdir}: {exc}") from exc
    if user_config is not None and not isinstance(user_config, dict):
        raise ValueError(
            f"config.yml in {workdir} is empty or malformed. "
            f"It must contain YAML key-value pairs."
        )

    config = apply_defaults(user_config)

    if not config.get("datastore"):
        raise ValueError(
            f"'datastore' key missing from config.yml in {workdir}. "
            f"This field is required. Edit config.yml and add the datastore path."
        )

    fabric_cfg = config.get("fabric") or {}
    if not fabric_cfg.get("id_col"):
        raise ValueError(
            f"'fabric.id_col' is empty in config.yml in {workdir}. "
            f"This field is required and must be a non-empty string. "
            f"The default is 'nhm_id'; remove the explicit empty value to "
            f"use the default, or set it to your fabric's HRU column name."
        )
    if not fabric_cfg.get("path"):
        raise ValueError(
            f"'fabric.path' missing from config.yml in {workdir}. "
            f"This field is required. Edit config.yml and add the absolute "
            f"path to your HRU fabric (GeoPackage / shapefile / parquet)."
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
