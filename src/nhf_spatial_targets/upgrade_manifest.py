"""Report whether a project's manifest.json predates the current schema.

Mirrors :mod:`nhf_spatial_targets.upgrade_config`: a report-only operator
discovery path. ``nhf-targets upgrade-manifest -d <dir>`` detects a manifest
whose ``manifest_schema_version`` is behind
:data:`~nhf_spatial_targets.release.lineage.CURRENT_MANIFEST_SCHEMA_VERSION`
and prints what ``rebuild-manifest`` would normalize. **Never mutates.**
"""

from __future__ import annotations

import json
from pathlib import Path

from nhf_spatial_targets.release.lineage import CURRENT_MANIFEST_SCHEMA_VERSION


def check_manifest_schema(project_dir: Path) -> int | None:
    """Return the on-disk ``manifest_schema_version`` if behind current.

    Returns ``None`` when the manifest is at the current schema version (in
    sync). A manifest with no version key reads as ``0`` (pre-version).

    Raises
    ------
    FileNotFoundError
        If ``<project_dir>/manifest.json`` does not exist.
    ValueError
        If the manifest is present but unparseable (loud, never silent).
    """
    manifest_path = Path(project_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {project_dir}. Run "
            f"'nhf-targets validate -d {project_dir}' first."
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest.json at {manifest_path} is corrupt: {exc}") from exc
    version = manifest.get("manifest_schema_version", 0)
    return version if version < CURRENT_MANIFEST_SCHEMA_VERSION else None
