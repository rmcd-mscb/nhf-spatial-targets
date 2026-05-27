"""Synthesize lineage steps from on-disk evidence (release PR-B2).

PR-B (#246) made every new validate / fetch / aggregate / target run
append a lineage step to ``manifest.json.steps[]``. Projects that
predate PR-B have populated ``sources[]`` entries but empty
``steps[]`` arrays -- their FGDC ``dataquality.lineage.processstep[]``
section (PR-D) would render empty until every source re-runs (days of
compute on caldera).

:func:`rebuild_lineage` walks the on-disk evidence and synthesizes
the missing steps:

- Each ``sources[key]`` entry with ``consolidated_nc`` /
  ``consolidated_ncs`` / ``regions`` / ``years`` / ``water_years`` /
  ``files`` / ``file`` becomes one ``kind=consolidate`` step.
- Each ``sources[key]`` entry with ``output_files`` (aggregator
  output list) becomes one ``kind=aggregate`` step.
- Each ``<project>/targets/<*>.nc`` becomes one ``kind=target`` or
  ``kind=nn_fill`` step.
- A single ``kind=validate`` step records the current fabric.json /
  manifest.json / config.effective.yml triple.

Idempotent: synthesized steps carry ``params.synthesized=True`` and
are deduped against existing ``steps[]`` by ``(kind, source_key,
output_paths)``. Re-running ``rebuild_lineage`` against a project
that already has live steps from the post-PR-B pipeline is safe --
the live steps win the dedupe and the synthesized variants are
skipped.

SHA256 is opt-in: ``compute_sha256=False`` (default) skips full-file
hashing because the outputs already exist on disk and may be multi-GB
each. Operators who want full integrity for the published release
can pass ``compute_sha256=True``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from nhf_spatial_targets.release.lineage import (
    build_step_record,
    output_file_entry,
)
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step-signature dedup
# ---------------------------------------------------------------------------


def _step_signature(step: dict) -> tuple:
    """Stable dedup key for a step.

    Two steps match when they produce the same outputs for the same
    ``(kind, source_key)`` pair. The source_key narrows the match so a
    rebuild of one project doesn't dedupe against a completely
    unrelated step that happens to share an output filename.
    """
    output_paths = tuple(
        sorted(o["path"] for o in step.get("outputs", []) if "path" in o)
    )
    return (step.get("kind"), step.get("source_key"), output_paths)


# ---------------------------------------------------------------------------
# Per-source synthesis
# ---------------------------------------------------------------------------


def _consolidate_outputs(source_entry: dict) -> list[Path]:
    """Extract consolidated-NC paths from a manifest source entry.

    Handles the half-dozen shapes fetch modules emit:

    - ``consolidated_nc`` -- single-NC consolidators (merra2, gldas,
      nldas, ncep_ncar, reitz2017, pangaea, mwbm_climgrid).
    - ``consolidated_ncs`` -- MODIS (year-keyed dict of NCs).
    - ``years[*].consolidated_nc`` -- snodas, margulis_wus_sr.
    - ``water_years[*].consolidated_nc`` -- ua_swe.
    - ``regions[*].consolidated_nc`` -- daymet (metadata-only, may be
      absent if the zarr-store fingerprint is the only artifact).
    - ``file.path`` -- mwbm_climgrid, reitz2017, pangaea (single-NC
      registrations that don't use the ``consolidated_nc`` key).

    Returns an empty list when no NC paths are findable -- daymet is
    typically in this bucket because its zarr stores remain at ORNL
    DAAC rather than being republished.
    """
    paths: list[Path] = []
    if isinstance(source_entry.get("consolidated_nc"), str):
        paths.append(Path(source_entry["consolidated_nc"]))
    if isinstance(source_entry.get("consolidated_ncs"), dict):
        for v in source_entry["consolidated_ncs"].values():
            if isinstance(v, str):
                paths.append(Path(v))
    for record_key in ("years", "water_years"):
        for rec in source_entry.get(record_key) or []:
            if not isinstance(rec, dict):
                continue
            nc = rec.get("consolidated_nc") or rec.get("path")
            if isinstance(nc, str):
                paths.append(Path(nc))
    for rec in (source_entry.get("regions") or {}).values():
        if isinstance(rec, dict):
            nc = rec.get("consolidated_nc") or rec.get("path")
            if isinstance(nc, str):
                paths.append(Path(nc))
    file_rec = source_entry.get("file")
    if isinstance(file_rec, dict) and isinstance(file_rec.get("path"), str):
        path = Path(file_rec["path"])
        if path not in paths:
            paths.append(path)
    return paths


def _build_outputs(paths: list[Path], *, compute_sha256: bool) -> list[dict]:
    """Map on-disk paths to output-entry dicts, skipping missing files."""
    entries: list[dict] = []
    for path in paths:
        if not path.exists():
            logger.debug("rebuild_lineage: skipping missing output %s", path)
            continue
        entries.append(output_file_entry(path, compute_sha256=compute_sha256))
    return entries


def _synthesize_consolidate_step(
    source_key: str,
    source_entry: dict,
    *,
    compute_sha256: bool,
) -> dict | None:
    """Build a ``kind=consolidate`` step from an existing source entry.

    Returns None when there is no consolidation evidence to synthesize
    from (e.g. an aggregate-only entry that was written by
    aggregate/_driver.py before the fetcher ran -- shouldn't happen in
    practice but defensive null check).
    """
    outputs = _build_outputs(
        _consolidate_outputs(source_entry), compute_sha256=compute_sha256
    )
    has_evidence = bool(outputs) or any(
        source_entry.get(k)
        for k in (
            "consolidated_nc",
            "consolidated_ncs",
            "years",
            "water_years",
            "regions",
            "files",
            "file",
        )
    )
    if not has_evidence:
        return None
    params: dict = {"synthesized": True}
    for key in ("period", "bbox", "license", "doi", "access_url", "fabric_scope"):
        if key in source_entry:
            params[key] = source_entry[key]
    if isinstance(source_entry.get("files"), list):
        params["n_files"] = len(source_entry["files"])
    if isinstance(source_entry.get("years"), list):
        params["years"] = [int(y["year"]) for y in source_entry["years"] if "year" in y]
    if isinstance(source_entry.get("water_years"), list):
        params["water_years"] = [
            int(y["water_year"])
            for y in source_entry["water_years"]
            if "water_year" in y
        ]
    if isinstance(source_entry.get("regions"), dict):
        params["regions"] = sorted(source_entry["regions"].keys())
    timestamp = source_entry.get("last_consolidated_utc") or source_entry.get(
        "downloaded_utc"
    )
    return build_step_record(
        kind="consolidate",
        source_key=source_key,
        outputs=outputs,
        params=params,
        command=f"rebuild-lineage:consolidate {source_key}",
        timestamp_utc=timestamp,
    )


def _synthesize_aggregate_step(
    project: Project,
    source_key: str,
    source_entry: dict,
    *,
    compute_sha256: bool,
) -> dict | None:
    """Build a ``kind=aggregate`` step from output_files in a source entry."""
    output_files = source_entry.get("output_files")
    if not isinstance(output_files, list) or not output_files:
        return None
    # output_files paths are project-workdir-relative in aggregate/_driver.py.
    abs_paths = [(project.workdir / rel).resolve() for rel in output_files]
    outputs = _build_outputs(abs_paths, compute_sha256=compute_sha256)
    params: dict = {"synthesized": True}
    for key in (
        "period",
        "fabric_sha256",
        "batch_size",
        "n_workers",
        "access_type",
    ):
        if key in source_entry:
            params[key] = source_entry[key]
    return build_step_record(
        kind="aggregate",
        source_key=source_key,
        outputs=outputs,
        params=params,
        command=f"rebuild-lineage:aggregate {source_key}",
        timestamp_utc=source_entry.get("timestamp"),
    )


# ---------------------------------------------------------------------------
# Target NC walking
# ---------------------------------------------------------------------------


def _synthesize_target_steps(project: Project, *, compute_sha256: bool) -> list[dict]:
    """Build one step per ``*_targets.nc`` in ``<project>/targets/``.

    Each NN-filled companion (``*_targets_nn_filled.nc``) becomes a
    ``kind=nn_fill`` step; the unfilled siblings are ``kind=target``.
    Per-year intermediate NCs under ``targets/.<target>_intermediates/``
    are intentionally skipped -- they're not the canonical published
    artifact, and PR-D's FGDC processstep should not enumerate every
    per-year forensic chunk.
    """
    targets_dir = project.targets_dir()
    if not targets_dir.exists():
        return []
    steps: list[dict] = []
    for nc in sorted(targets_dir.glob("*.nc")):
        if not nc.is_file():
            continue
        kind = "nn_fill" if nc.name.endswith("_nn_filled.nc") else "target"
        mtime = datetime.fromtimestamp(nc.stat().st_mtime, tz=timezone.utc).isoformat()
        steps.append(
            build_step_record(
                kind=kind,
                source_key=None,
                outputs=[output_file_entry(nc, compute_sha256=compute_sha256)],
                params={"synthesized": True},
                command=f"rebuild-lineage:{kind} {nc.name}",
                timestamp_utc=mtime,
            )
        )
    return steps


# ---------------------------------------------------------------------------
# Validate step
# ---------------------------------------------------------------------------


def _synthesize_validate_step(project: Project, *, compute_sha256: bool) -> dict | None:
    """Build a single ``kind=validate`` step from on-disk fabric.json.

    Only one validate step is synthesized regardless of how many times
    the operator originally ran ``validate`` -- we have no on-disk
    trace of historical runs, only the current fabric.json. The step
    records the fabric's current sha256, hru_count, and id_col_sorted
    flag so the FGDC fabric metadata is anchored to a real fabric
    state when PR-D renders it.
    """
    fabric_json = project.workdir / "fabric.json"
    manifest_json = project.manifest_path
    effective_config = project.workdir / "config.effective.yml"
    outputs = _build_outputs(
        [p for p in (fabric_json, manifest_json, effective_config)],
        compute_sha256=compute_sha256,
    )
    if not outputs:
        return None
    params: dict = {"synthesized": True}
    fabric_meta = project.fabric or {}
    for key in ("sha256", "hru_count", "id_col", "id_col_sorted"):
        if key in fabric_meta:
            params[key] = fabric_meta[key]
    if "sha256" in params:
        params["fabric_sha256"] = params.pop("sha256")
    timestamp = (
        datetime.fromtimestamp(fabric_json.stat().st_mtime, tz=timezone.utc).isoformat()
        if fabric_json.exists()
        else None
    )
    return build_step_record(
        kind="validate",
        source_key=None,
        outputs=outputs,
        params=params,
        command="rebuild-lineage:validate",
        timestamp_utc=timestamp,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rebuild_lineage(
    project: Project,
    *,
    compute_sha256: bool = False,
    dry_run: bool = False,
) -> dict:
    """Synthesize missing lineage steps for an existing project.

    Walks ``manifest.json.sources[]``, ``<project>/targets/``, and
    ``<project>/fabric.json``; appends synthesized
    ``kind=consolidate`` / ``kind=aggregate`` / ``kind=target`` /
    ``kind=nn_fill`` / ``kind=validate`` records to
    ``manifest.json.steps[]``.

    Idempotent: synthesized steps are deduped against existing
    ``steps[]`` by ``(kind, source_key, output_paths)``. Existing
    live steps (from the post-PR-B pipeline) always win -- the
    synthesized variant is skipped.

    Parameters
    ----------
    project
        Loaded project. ``project.manifest_path`` must exist (run
        ``validate`` first).
    compute_sha256
        Hash on-disk output files when building entries. Default
        ``False`` -- the outputs may be multi-GB and the operator
        may not want to pay the rehash cost just to bootstrap. Pass
        ``True`` before any release-payload staging that depends on
        sha256 integrity.
    dry_run
        Build the synthesis without writing to ``manifest.json``.
        Returns the same summary so an operator can inspect what
        would change.

    Returns
    -------
    Summary dict:
        ``steps_added``: int -- new steps that landed in steps[]
        ``skipped_existing``: int -- synthesized steps elided because
        a matching step already existed
        ``by_kind``: dict[str, int] -- per-kind count of additions
        ``dry_run``: bool
    """
    manifest_path = project.manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run 'nhf-targets validate "
            f"--project-dir {project.workdir}' first."
        )
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("sources", {})
    manifest.setdefault("steps", [])

    existing_signatures = {_step_signature(s) for s in manifest["steps"]}

    new_steps: list[dict] = []
    by_kind: dict[str, int] = {}
    skipped = 0

    def _maybe_add(step: dict | None) -> None:
        nonlocal skipped
        if step is None:
            return
        sig = _step_signature(step)
        if sig in existing_signatures:
            skipped += 1
            return
        existing_signatures.add(sig)
        new_steps.append(step)
        by_kind[step["kind"]] = by_kind.get(step["kind"], 0) + 1

    for source_key, source_entry in sorted(manifest["sources"].items()):
        if not isinstance(source_entry, dict):
            continue
        _maybe_add(
            _synthesize_consolidate_step(
                source_key, source_entry, compute_sha256=compute_sha256
            )
        )
        _maybe_add(
            _synthesize_aggregate_step(
                project, source_key, source_entry, compute_sha256=compute_sha256
            )
        )

    for target_step in _synthesize_target_steps(project, compute_sha256=compute_sha256):
        _maybe_add(target_step)

    _maybe_add(_synthesize_validate_step(project, compute_sha256=compute_sha256))

    summary = {
        "steps_added": len(new_steps),
        "skipped_existing": skipped,
        "by_kind": by_kind,
        "dry_run": dry_run,
    }

    if dry_run:
        logger.info("rebuild_lineage (dry-run): %s", summary)
        return summary

    if not new_steps:
        logger.info("rebuild_lineage: no missing steps to synthesize.")
        return summary

    manifest["steps"].extend(new_steps)
    # Atomic write via tempfile + rename matches the rest of the
    # pipeline's manifest update path.
    import os
    import tempfile

    tmp_fd, tmp_path = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp_path).replace(manifest_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    logger.info(
        "rebuild_lineage: synthesized %d step(s); skipped %d already-present "
        "step(s). by_kind=%s",
        len(new_steps),
        skipped,
        by_kind,
    )
    return summary
