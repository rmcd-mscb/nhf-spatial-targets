"""Lineage capture for ``manifest.json.steps[]``.

Every pipeline write-site (validate, consolidate, aggregate, target,
nn_fill) appends one step record describing what ran, with which inputs
and outputs. ``release.fgdc`` (PR-D) consumes
``manifest.json.steps[]`` to populate FGDC CSDGM
``dataquality.lineage.processstep[]`` -- so this module is the single
seam by which provenance flows from the live pipeline to the published
data release.

The flock pattern mirrors
:func:`nhf_spatial_targets.aggregate._driver.update_manifest`: an
advisory ``LOCK_EX`` on ``<manifest_path>.lock`` guards the
read-modify-write so concurrent SLURM array workers don't lose each
other's step appends.

Public surface:

- :func:`append_step` -- append a single step to ``manifest.json.steps``.
- :func:`merge_source_and_append_step` -- atomic read-merge-write that
  both updates ``manifest["sources"][source_key]`` and appends a step.
  Replaces the bespoke ``_update_manifest`` patterns scattered across
  fetch modules (see #180) with a single canonical implementation.
- :func:`output_file_entry` / :func:`input_file_entry` -- shape helpers.
  Outputs carry ``sha256`` (cheap to compute on a file we just wrote);
  inputs carry only path + size + mtime, by design, so ``aggregate``
  runs that read dozens of consolidated NCs don't pay an extra
  full-file hash pass.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from nhf_spatial_targets import __version__ as _SOFTWARE_VERSION

logger = logging.getLogger(__name__)


try:
    import fcntl as _fcntl

    _HAVE_FLOCK = True
except ImportError:  # Windows fallback.
    _HAVE_FLOCK = False
    logger.warning(
        "lineage: fcntl unavailable on this platform; concurrent manifest "
        "writes WILL lose lineage records under contention. Use a POSIX "
        "system for production runs."
    )


# Kinds of pipeline steps we capture. Mirrors the table in
# ``~/.claude/plans/a-requirement-is-pushlishing-vast-crayon.md``
# (``Lineage capture`` section). The set is closed: PR-D's FGDC
# generator pattern-matches on these strings to choose the right
# CSDGM ``procdesc`` phrasing.
STEP_KINDS = frozenset(
    {"fetch", "consolidate", "aggregate", "target", "nn_fill", "validate"}
)


_SHA256_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*, streamed in 1 MiB blocks.

    Used by :func:`output_file_entry` to fingerprint just-written files;
    streaming keeps memory flat regardless of file size (consolidated
    NCs are routinely multi-GB).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(_SHA256_CHUNK_BYTES)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _file_basics(path: Path) -> dict:
    """Return ``{"path", "size_bytes", "mtime_utc"}`` for *path*."""
    st = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


def output_file_entry(path: Path, *, compute_sha256: bool = True) -> dict:
    """File-entry shape for ``step["outputs"]``.

    Includes ``sha256`` by default -- the file was just written, so
    hashing is cheap relative to the write itself. Pass
    ``compute_sha256=False`` only for tests or for the rare case where
    a step records an output that other consumers will hash later.

    Parameters
    ----------
    path
        Path to the just-written output file. Must exist.
    compute_sha256
        When ``True`` (default), include the file's SHA-256 hex digest.

    Returns
    -------
    dict with keys ``path``, ``size_bytes``, ``mtime_utc`` and, when
    ``compute_sha256`` is True, ``sha256``.
    """
    entry = _file_basics(path)
    if compute_sha256:
        entry["sha256"] = sha256_file(path)
    return entry


def input_file_entry(path: Path) -> dict:
    """File-entry shape for ``step["inputs"]``.

    No SHA-256 by design: an ``aggregate`` step over a 30-source fabric
    reads dozens of multi-GB consolidated NCs, and hashing each input
    would dominate runtime. Outputs do carry sha256 (see
    :func:`output_file_entry`), which is what downstream consumers
    really need for integrity checks of the published release.
    """
    return _file_basics(path)


def _new_manifest_skeleton() -> dict:
    """Return a fresh ``manifest.json`` skeleton.

    Single source of truth: ``{"sources": {}, "steps": []}``. Used both
    when ``manifest.json`` is absent and when this module is invoked
    against a fresh project (validate stage).
    """
    return {"sources": {}, "steps": []}


def read_manifest(manifest_path: Path) -> dict:
    """Read ``manifest.json`` or return a fresh skeleton.

    Raises ``ValueError`` on parse failure -- a corrupt manifest is a
    loud failure mode, not something to silently overwrite.
    """
    if not manifest_path.exists():
        return _new_manifest_skeleton()
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manifest.json at {manifest_path} is corrupt: {exc}. "
            f"Inspect the file manually, restore from backup, or delete "
            f"it to force a fresh skeleton (you will lose prior "
            f"provenance)."
        ) from exc
    manifest.setdefault("sources", {})
    manifest.setdefault("steps", [])
    return manifest


def atomic_write_manifest(manifest_path: Path, manifest: dict) -> None:
    """Write *manifest* to *manifest_path* via tempfile + rename.

    Indented JSON keeps diffs auditable; ``Path.replace`` is atomic on
    the same filesystem, so a partial manifest never lands at the final
    path.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp_path).replace(manifest_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def build_step_record(
    *,
    kind: str,
    source_key: str | None,
    inputs: list[dict] | None = None,
    outputs: list[dict] | None = None,
    params: dict | None = None,
    tool: str = "nhf-targets",
    command: str | None = None,
    software_version: str | None = None,
    timestamp_utc: str | None = None,
) -> dict:
    """Assemble a normalized step record. Pure function; no I/O.

    Exposed for the rare in-process call site (the aggregator) that
    already holds its own flock on ``manifest.json`` and just needs the
    record shape -- calling :func:`append_step` from inside that lock
    would deadlock. Most call sites should use :func:`append_step` or
    :func:`merge_source_and_append_step` instead.
    """
    if kind not in STEP_KINDS:
        raise ValueError(
            f"Unknown step kind {kind!r}; expected one of {sorted(STEP_KINDS)}"
        )
    if timestamp_utc is None:
        timestamp_utc = datetime.now(timezone.utc).isoformat()
    if software_version is None:
        software_version = _SOFTWARE_VERSION
    record: dict = {
        "kind": kind,
        "source_key": source_key,
        "timestamp_utc": timestamp_utc,
        "software_version": software_version,
        "tool": tool,
        "command": command,
        "inputs": list(inputs) if inputs else [],
        "outputs": list(outputs) if outputs else [],
        "params": dict(params) if params else {},
    }
    return record


def with_flock(lock_path: Path, body):
    """Run *body* under an advisory ``LOCK_EX`` on *lock_path*.

    Falls back to a bare invocation when ``fcntl`` is unavailable
    (Windows; a warning is emitted at module import).
    """
    if not _HAVE_FLOCK:
        return body()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lock_f:
        _fcntl.flock(lock_f, _fcntl.LOCK_EX)
        try:
            return body()
        finally:
            _fcntl.flock(lock_f, _fcntl.LOCK_UN)


def append_step(
    manifest_path: Path,
    *,
    kind: str,
    source_key: str | None,
    inputs: list[dict] | None = None,
    outputs: list[dict] | None = None,
    params: dict | None = None,
    tool: str = "nhf-targets",
    command: str | None = None,
    software_version: str | None = None,
    timestamp_utc: str | None = None,
) -> dict:
    """Append a single step to ``manifest.json.steps[]`` under flock.

    Reads (or initializes) the manifest, appends one normalized step
    record, and writes atomically. Concurrent callers see a sorted
    sequence of appends (no torn writes, no lost step).

    Parameters
    ----------
    manifest_path
        Path to ``<project>/manifest.json``. Parent dir must already
        exist; the file may or may not.
    kind
        One of :data:`STEP_KINDS`.
    source_key
        Catalog source key (e.g. ``"era5_land"``); ``None`` for
        fabric-level steps (``validate``, the stitched ``target``).
    inputs, outputs
        File-entry lists. Use :func:`input_file_entry` /
        :func:`output_file_entry` to build them. Default to ``[]``.
    params
        Step-specific parameters (e.g. ``{"batch_size": 10000}``).
        Default ``{}``.
    tool, command
        Provenance for the CLI invocation that ran this step.
    software_version
        Defaults to ``nhf_spatial_targets.__version__``. Tests inject.
    timestamp_utc
        Defaults to ``datetime.now(timezone.utc).isoformat()``. Tests
        inject for byte-stable golden files.

    Returns
    -------
    The step record that was appended (useful for test assertions and
    log lines).
    """
    record = build_step_record(
        kind=kind,
        source_key=source_key,
        inputs=inputs or [],
        outputs=outputs or [],
        params=params,
        tool=tool,
        command=command,
        software_version=software_version,
        timestamp_utc=timestamp_utc,
    )
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")

    def _do() -> None:
        manifest = read_manifest(manifest_path)
        manifest["steps"].append(record)
        atomic_write_manifest(manifest_path, manifest)

    with_flock(lock_path, _do)
    logger.info(
        "lineage: appended step kind=%s source_key=%s outputs=%d",
        record["kind"],
        record["source_key"],
        len(record["outputs"]),
    )
    return record


def merge_source_and_append_step(
    manifest_path: Path,
    source_key: str,
    *,
    source_updates: dict,
    kind: str,
    inputs: list[dict] | None = None,
    outputs: list[dict] | None = None,
    params: dict | None = None,
    tool: str = "nhf-targets",
    command: str | None = None,
    software_version: str | None = None,
    timestamp_utc: str | None = None,
) -> dict:
    """Atomic source-merge + step-append.

    Replaces the bespoke ``_update_manifest`` helpers each fetch module
    used to carry. The merge semantics match the pre-existing pattern:
    ``manifest["sources"][source_key]`` is updated with *source_updates*
    via ``dict.update`` (existing keys preserved unless overridden), so
    e.g. era5_land's ``years`` block survives a re-fetch that only
    rewrites ``last_consolidated_utc``.

    Both operations happen inside the same flock, so a SLURM array
    worker that crashes between source merge and step append cannot
    leave the manifest with one but not the other.

    Parameters
    ----------
    manifest_path
        Path to ``<project>/manifest.json``.
    source_key
        Catalog source key (e.g. ``"merra2"``).
    source_updates
        Keys to merge into ``manifest["sources"][source_key]``. Must
        not include ``"steps"``.
    kind, inputs, outputs, params, tool, command, software_version,
    timestamp_utc
        Forwarded to the step record. See :func:`append_step`.

    Returns
    -------
    The appended step record.
    """
    if "steps" in source_updates:
        raise ValueError(
            "source_updates must not include 'steps' -- the steps array "
            "lives at the top level of manifest.json, not nested under a "
            "source."
        )
    record = build_step_record(
        kind=kind,
        source_key=source_key,
        inputs=inputs or [],
        outputs=outputs or [],
        params=params,
        tool=tool,
        command=command,
        software_version=software_version,
        timestamp_utc=timestamp_utc,
    )
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")

    def _do() -> None:
        manifest = read_manifest(manifest_path)
        existing = manifest["sources"].get(source_key, {})
        existing.update(source_updates)
        manifest["sources"][source_key] = existing
        manifest["steps"].append(record)
        atomic_write_manifest(manifest_path, manifest)

    with_flock(lock_path, _do)
    logger.info(
        "lineage: merged source=%s + appended step kind=%s outputs=%d",
        source_key,
        record["kind"],
        len(record["outputs"]),
    )
    return record
