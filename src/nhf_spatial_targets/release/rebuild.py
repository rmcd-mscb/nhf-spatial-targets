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
  config.effective.yml pair (manifest.json is deliberately NOT a
  validate output -- the step lives inside it).

Idempotent: synthesized steps carry ``params.synthesized=True`` and
are deduped against existing ``steps[]`` by ``(kind, source_key,
output_paths)``. Re-running ``rebuild_lineage`` against a project
that already has live steps from the post-PR-B pipeline is safe --
the live steps win the dedupe and the synthesized variants are
skipped.

Concurrency: the read-modify-write delegates to
:func:`lineage.with_flock` + :func:`lineage.read_manifest` +
:func:`lineage.atomic_write_manifest`, so the rebuild can run safely
alongside in-flight aggregate / fetch workers. The manifest is
re-read inside the flock to dedup against steps a concurrent worker
appended while synthesis was running.

SHA256 is opt-in: ``compute_sha256=False`` (default) skips full-file
hashing because the outputs already exist on disk and may be multi-GB
each. Operators who want full integrity for the published release
must pass ``compute_sha256=True`` -- PR-F's release-publish stage
will refuse to stage steps stamped with ``params.sha256_skipped=True``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from nhf_spatial_targets.release.lineage import (
    atomic_write_manifest,
    build_step_record,
    output_file_entry,
    read_manifest,
    with_flock,
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


def _extract_path_from_record(rec: object, keys: tuple[str, ...]) -> Path | None:
    """Return the first existing-string path under any of *keys* in *rec*.

    Used to flatten year / water_year / region / file records that each
    use a different key (``consolidated_nc`` for merra2-shape, ``daily_path``
    for era5/snodas/margulis-shape, ``path`` for mwbm/reitz-shape).
    """
    if not isinstance(rec, dict):
        return None
    for key in keys:
        value = rec.get(key)
        if isinstance(value, str):
            return Path(value)
    return None


def _consolidate_outputs(source_entry: dict) -> list[Path]:
    """Extract consolidated-NC paths from a manifest source entry.

    Handles the source-entry shapes the post-PR-B fetch modules emit:

    - ``consolidated_nc`` (string) -- single-NC consolidators (merra2,
      gldas, nldas, ncep_ncar, reitz2017, pangaea, mwbm_climgrid).
    - ``consolidated_ncs`` (dict) -- MODIS year-keyed dict.
    - ``years[*]`` / ``water_years[*]`` -- snodas, margulis_wus_sr,
      ua_swe; each record carries ``daily_path``, ``consolidated_nc``,
      ``consolidated_path``, or ``path``.
    - ``regions[*]`` (dict) -- daymet; values are zarr-store directories
      and are intentionally skipped (zarr stores are not republished).
    - ``files[*]`` -- era5_land top-level; each record carries
      ``daily_path``, ``monthly_path``, or ``path``.
    - ``file{path}`` -- mwbm_climgrid / reitz2017 / pangaea single-NC
      registrations that don't use the ``consolidated_nc`` key.

    Directories (e.g. daymet zarr stores) are skipped so the synthesized
    ``outputs`` only contains file artifacts that PR-D's FGDC consumer
    can fingerprint. Each path is appended at most once even when a
    source entry duplicates it across multiple keys.
    """
    paths: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        if path in seen:
            return
        # Zarr stores and any other dir-valued artifact: rebuild fingerprints
        # files, not directories. Match the live daymet step's outputs=[].
        if path.exists() and path.is_dir():
            return
        seen.add(path)
        paths.append(path)

    raw_nc = source_entry.get("consolidated_nc")
    if isinstance(raw_nc, str):
        _add(Path(raw_nc))
    if isinstance(source_entry.get("consolidated_ncs"), dict):
        for v in source_entry["consolidated_ncs"].values():
            if isinstance(v, str):
                _add(Path(v))
    record_path_keys = ("daily_path", "consolidated_nc", "consolidated_path", "path")
    for record_key in ("years", "water_years", "files"):
        for rec in source_entry.get(record_key) or []:
            nc = _extract_path_from_record(rec, record_path_keys)
            if nc is not None:
                _add(nc)
    for rec in (source_entry.get("regions") or {}).values():
        nc = _extract_path_from_record(rec, record_path_keys)
        if nc is not None:
            _add(nc)
    file_rec = source_entry.get("file")
    nc = _extract_path_from_record(file_rec, ("path",))
    if nc is not None:
        _add(nc)
    return paths


def _build_outputs(
    paths: list[Path], *, compute_sha256: bool
) -> tuple[list[dict], list[str]]:
    """Map on-disk paths to output-entry dicts; return missing-paths separately.

    A path that doesn't exist is logged at WARNING (matching the
    aggregator's instrumentation in PR-B's fixup) and recorded in the
    returned ``missing`` list. A path whose ``stat()`` or ``open()``
    raises ``OSError`` (TOCTOU race with operator cleanup, NFS stale
    handle, Lustre I/O hiccup) is treated identically: log + record,
    don't abort the whole rebuild.
    """
    entries: list[dict] = []
    missing: list[str] = []
    for path in paths:
        if not path.exists():
            logger.warning(
                "rebuild_lineage: output %s listed in evidence but not on "
                "disk; step will omit it.",
                path,
            )
            missing.append(str(path))
            continue
        try:
            entries.append(output_file_entry(path, compute_sha256=compute_sha256))
        except OSError as exc:
            # File vanished between exists() and stat(), or stat()/open()
            # failed for an environmental reason (stale NFS handle,
            # permission flap, scratch-purge cron). Promote to a logged
            # skip so one bad file doesn't take down the rebuild.
            logger.warning(
                "rebuild_lineage: failed to fingerprint %s (%s); step will omit it.",
                path,
                exc,
            )
            missing.append(str(path))
    return entries, missing


def _maybe_stamp_sha256_skipped(params: dict, compute_sha256: bool) -> None:
    """Add ``params['sha256_skipped'] = True`` when hashing was opted out.

    Lets PR-F's ``release publish`` precondition check refuse to stage
    a release whose lineage has unfingerprinted outputs, without
    re-walking the on-disk evidence.
    """
    if not compute_sha256:
        params["sha256_skipped"] = True


def _synthesize_consolidate_step(
    source_key: str,
    source_entry: dict,
    *,
    compute_sha256: bool,
) -> dict | None:
    """Build a ``kind=consolidate`` step from an existing source entry."""
    candidate_paths = _consolidate_outputs(source_entry)
    outputs, missing = _build_outputs(candidate_paths, compute_sha256=compute_sha256)
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
    if missing:
        params["missing_outputs"] = missing
    _maybe_stamp_sha256_skipped(params, compute_sha256)
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
    if "output_files" not in source_entry:
        return None
    output_files = source_entry["output_files"]
    if not isinstance(output_files, list) or not output_files:
        # Empty list = aborted aggregate. Surface it as a step with an
        # explicit marker rather than silently eliding the evidence,
        # since "never aggregated" and "aggregate aborted" are different
        # states with different remediation.
        params: dict = {"synthesized": True, "status": "aborted"}
        for key in ("period", "fabric_sha256", "access_type"):
            if key in source_entry:
                params[key] = source_entry[key]
        _maybe_stamp_sha256_skipped(params, compute_sha256)
        return build_step_record(
            kind="aggregate",
            source_key=source_key,
            outputs=[],
            params=params,
            command=f"rebuild-lineage:aggregate {source_key}",
            timestamp_utc=source_entry.get("timestamp"),
        )
    # output_files paths are project-workdir-relative in aggregate/_driver.py.
    abs_paths = [(project.workdir / rel).resolve() for rel in output_files]
    outputs, missing = _build_outputs(abs_paths, compute_sha256=compute_sha256)
    params = {"synthesized": True}
    for key in (
        "period",
        "fabric_sha256",
        "batch_size",
        "n_workers",
        "access_type",
    ):
        if key in source_entry:
            params[key] = source_entry[key]
    if missing:
        params["missing_outputs"] = missing
    _maybe_stamp_sha256_skipped(params, compute_sha256)
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
    per-year forensic chunk. (The one-level glob in ``targets_dir.glob``
    does not descend into the intermediates dir, so this exclusion is
    structural -- not dependent on the dir being hidden.)
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
        outputs, missing = _build_outputs([nc], compute_sha256=compute_sha256)
        params: dict = {
            "synthesized": True,
            "timestamp_source": "file_mtime",
        }
        if missing:
            params["missing_outputs"] = missing
        _maybe_stamp_sha256_skipped(params, compute_sha256)
        steps.append(
            build_step_record(
                kind=kind,
                source_key=None,
                outputs=outputs,
                params=params,
                command=f"rebuild-lineage:{kind} {nc.name}",
                timestamp_utc=mtime,
            )
        )
    return steps


# ---------------------------------------------------------------------------
# Validate step
# ---------------------------------------------------------------------------


def _synthesize_validate_step(project: Project, *, compute_sha256: bool) -> dict:
    """Build a single ``kind=validate`` step from on-disk fabric.json.

    Only one validate step is synthesized regardless of how many times
    the operator originally ran ``validate`` -- we have no on-disk
    trace of historical runs, only the current fabric.json. The step
    records the fabric's current sha256, hru_count, and id_col_sorted
    flag so the FGDC fabric metadata is anchored to a real fabric
    state when PR-D renders it.

    ``manifest.json`` is deliberately NOT a validate output -- the
    step lives inside it, so its recorded sha256 would describe the
    pre-append state. This matches the live ``validate`` writer in
    :func:`nhf_spatial_targets.validate.validate_workspace`.

    Raises
    ------
    FileNotFoundError
        If ``fabric.json`` is missing. A validate step that doesn't
        anchor to a real fabric carries no information PR-D's FGDC
        consumer can use; surface the missing file loudly so the
        operator runs ``nhf-targets validate`` first.
    """
    fabric_json = project.workdir / "fabric.json"
    if not fabric_json.exists():
        raise FileNotFoundError(
            f"{fabric_json} not found. The validate step cannot be "
            f"synthesized without the fabric anchor. Run 'nhf-targets "
            f"validate --project-dir {project.workdir}' first."
        )
    effective_config = project.workdir / "config.effective.yml"
    outputs, missing = _build_outputs(
        [p for p in (fabric_json, effective_config)],
        compute_sha256=compute_sha256,
    )
    params: dict = {
        "synthesized": True,
        "timestamp_source": "file_mtime",
    }
    fabric_meta = project.fabric or {}
    for key in ("sha256", "hru_count", "id_col", "id_col_sorted"):
        if key in fabric_meta:
            params[key] = fabric_meta[key]
    if "sha256" in params:
        params["fabric_sha256"] = params.pop("sha256")
    if missing:
        params["missing_outputs"] = missing
    _maybe_stamp_sha256_skipped(params, compute_sha256)
    timestamp = datetime.fromtimestamp(
        fabric_json.stat().st_mtime, tz=timezone.utc
    ).isoformat()
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

    The read-modify-write is flock-protected and re-reads the manifest
    inside the lock, so a concurrent in-flight aggregate or fetch
    worker that appends a step while synthesis is running will not
    have its step clobbered.

    Idempotent: synthesized steps are deduped against existing
    ``steps[]`` by ``(kind, source_key, output_paths)``. Existing
    live steps (from the post-PR-B pipeline) always win -- the
    synthesized variant is skipped.

    Parameters
    ----------
    project
        Loaded project. ``project.manifest_path`` AND
        ``project.workdir / "fabric.json"`` must exist (run
        ``nhf-targets validate`` first).
    compute_sha256
        Hash on-disk output files when building entries. Default
        ``False`` -- the outputs may be multi-GB and the operator
        may not want to pay the rehash cost just to bootstrap.
        Synthesized steps stamped with ``params.sha256_skipped=True``
        when this is False; PR-F's ``release publish`` rejects
        sha256-skipped steps so the operator must explicitly opt in
        before publishing.
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
        ``compute_sha256``: bool -- echo of the input flag
        ``dry_run``: bool

    Raises
    ------
    FileNotFoundError
        If ``manifest.json`` or ``fabric.json`` is missing.
    """
    manifest_path = project.manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run 'nhf-targets validate "
            f"--project-dir {project.workdir}' first."
        )

    # Build the synthesized steps OUTSIDE the flock so the expensive
    # walk (potentially minutes if compute_sha256=True) doesn't block
    # in-flight aggregate / fetch workers. The flock only covers the
    # read-merge-write below.
    new_steps_synth: list[dict] = []
    for source_key, source_entry in sorted(
        read_manifest(manifest_path)["sources"].items()
    ):
        if not isinstance(source_entry, dict):
            continue
        cs = _synthesize_consolidate_step(
            source_key, source_entry, compute_sha256=compute_sha256
        )
        if cs is not None:
            new_steps_synth.append(cs)
        ag = _synthesize_aggregate_step(
            project, source_key, source_entry, compute_sha256=compute_sha256
        )
        if ag is not None:
            new_steps_synth.append(ag)
    new_steps_synth.extend(
        _synthesize_target_steps(project, compute_sha256=compute_sha256)
    )
    new_steps_synth.append(
        _synthesize_validate_step(project, compute_sha256=compute_sha256)
    )

    summary: dict = {
        "steps_added": 0,
        "skipped_existing": 0,
        "by_kind": {},
        "compute_sha256": compute_sha256,
        "dry_run": dry_run,
    }

    if dry_run:
        # Compute the would-be dedup against the current on-disk manifest
        # for the summary report; don't touch disk.
        existing_sigs = {
            _step_signature(s) for s in read_manifest(manifest_path).get("steps", [])
        }
        for step in new_steps_synth:
            sig = _step_signature(step)
            if sig in existing_sigs:
                summary["skipped_existing"] += 1
                continue
            existing_sigs.add(sig)
            summary["steps_added"] += 1
            summary["by_kind"][step["kind"]] = (
                summary["by_kind"].get(step["kind"], 0) + 1
            )
        logger.info("rebuild_lineage (dry-run): %s", summary)
        return summary

    # Flock-protected merge: re-read inside the lock so we dedup against
    # any steps that landed between our synthesis pass and this write.
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")

    def _do_merge() -> None:
        manifest = read_manifest(manifest_path)
        existing_sigs = {_step_signature(s) for s in manifest["steps"]}
        landed: list[dict] = []
        for step in new_steps_synth:
            sig = _step_signature(step)
            if sig in existing_sigs:
                summary["skipped_existing"] += 1
                continue
            existing_sigs.add(sig)
            landed.append(step)
            summary["by_kind"][step["kind"]] = (
                summary["by_kind"].get(step["kind"], 0) + 1
            )
        if landed:
            manifest["steps"].extend(landed)
            atomic_write_manifest(manifest_path, manifest)
        summary["steps_added"] = len(landed)

    with_flock(lock_path, _do_merge)

    if summary["steps_added"] == 0:
        logger.info("rebuild_lineage: no missing steps to synthesize.")
    else:
        logger.info(
            "rebuild_lineage: synthesized %d step(s); skipped %d "
            "already-present step(s). by_kind=%s compute_sha256=%s",
            summary["steps_added"],
            summary["skipped_existing"],
            summary["by_kind"],
            compute_sha256,
        )
    if not compute_sha256 and summary["steps_added"] > 0:
        logger.warning(
            "rebuild_lineage: compute_sha256=False (default). Synthesized "
            "step outputs lack 'sha256' integrity fields and are stamped "
            "with params.sha256_skipped=True. Re-run with "
            "compute_sha256=True before 'release publish', or PR-F's "
            "stage gate will refuse to publish them."
        )
    return summary
