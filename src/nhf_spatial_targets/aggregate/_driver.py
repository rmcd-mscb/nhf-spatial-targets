"""Shared aggregation driver: manifest helper, weight cache, and tier-1 engine."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


def update_manifest(
    project: Project,
    source_key: str,
    access: dict,
    period: str,
    output_file: str,
    weight_files: list[str],
) -> None:
    """Merge an aggregation provenance entry into ``manifest.json`` atomically.

    The manifest is keyed as ``sources[source_key]``; existing entries for
    other sources are preserved. ``period`` is stored as-is for provenance;
    ``fabric_sha256`` is read from ``fabric.json``.
    """
    manifest_path = project.manifest_path
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {project.workdir} is corrupt: {exc}"
            ) from exc
    else:
        manifest = {"sources": {}, "steps": []}

    manifest.setdefault("sources", {})

    fabric_json = project.workdir / "fabric.json"
    fabric_sha = ""
    if fabric_json.exists():
        fabric_meta = json.loads(fabric_json.read_text())
        fabric_sha = fabric_meta.get("sha256", "")

    entry: dict = {
        "source_key": source_key,
        "access_type": access.get("type", ""),
        "period": period,
        "fabric_sha256": fabric_sha,
        "output_file": output_file,
        "weight_files": list(weight_files),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Carry a few optional access identifiers through for provenance parity
    # with ssebop.py's existing behaviour.
    for extra_key in ("collection_id", "short_name", "version", "doi"):
        if extra_key in access:
            entry[extra_key] = access[extra_key]

    manifest["sources"][source_key] = entry

    tmp_fd, tmp_path = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp_path).replace(manifest_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    logger.info("Updated manifest.json with '%s' aggregation provenance", source_key)
