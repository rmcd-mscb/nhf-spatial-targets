"""Backfill manifest.json provenance from an existing shared datastore.

See docs/architecture/reconcile-manifest.md and issue #160. The merge is
gap-fill only: reconcile appends records for files not already recorded and
never mutates an existing record, so a true ``fetch`` record is never
downgraded to ``reconciled``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nhf_spatial_targets import catalog as _catalog
from nhf_spatial_targets.workspace import Project

try:
    import fcntl as _fcntl

    _HAVE_FLOCK = True
except ImportError:  # pragma: no cover - Windows
    _HAVE_FLOCK = False

logger = logging.getLogger(__name__)


def _record_identity(rec: dict) -> tuple[str, str | int]:
    """Stable dedupe identity for a file record: year if present, else path."""
    if "year" in rec:
        return ("year", int(rec["year"]))
    if "path" in rec:
        return ("path", str(rec["path"]))
    raise ValueError(f"reconcile record lacks both 'year' and 'path': {rec!r}")


def _gap_fill(
    existing: list[dict], new_records: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Append only the new_records whose identity is not already in existing.

    Returns ``(merged, added)``. ``merged`` is ``existing`` (untouched, in
    order) followed by the appended records. Malformed existing records
    (no year/path) are tolerated — they stay in ``merged`` and never block
    an append.
    """
    seen: set[tuple[str, str | int]] = set()
    for r in existing:
        try:
            seen.add(_record_identity(r))
        except ValueError:
            continue
    added: list[dict] = []
    for r in new_records:
        ident = _record_identity(r)
        if ident not in seen:
            added.append(r)
            seen.add(ident)
    return existing + added, added


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 and return the hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class SourceReconcileResult:
    """Per-source outcome of a reconcile pass (drives the CLI table + tests)."""

    source_key: str
    status: str  # "reconciled" | "no-op" | "no-hook" | "empty"
    on_disk: int = 0
    already_recorded: int = 0
    added: int = 0


def _read_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {"sources": {}, "steps": []}
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manifest.json at {manifest_path} is corrupt and cannot be parsed. "
            f"Inspect or restore it before reconciling. Detail: {exc}"
        ) from exc


def _atomic_write(manifest_path: Path, manifest: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp).replace(manifest_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _apply_records(
    project: Project,
    source_key: str,
    records: list[dict],
    *,
    dry_run: bool,
) -> SourceReconcileResult:
    """Gap-fill-merge ``records`` into ``manifest.json`` for one source.

    flock-guarded read-merge-write. Appends only records whose identity is
    absent; never mutates an existing record. Creates a minimal source entry
    (source_key, access_url, derived period, reconciled_utc) only when the
    source has no existing entry; an existing entry's metadata is left alone.
    """
    manifest_path = project.manifest_path
    lock_path = manifest_path.with_suffix(".lock")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lock_f:
        if _HAVE_FLOCK:
            _fcntl.flock(lock_f, _fcntl.LOCK_EX)
        manifest = _read_manifest(manifest_path)
        manifest.setdefault("sources", {})
        created = source_key not in manifest["sources"]
        entry = manifest["sources"].get(source_key, {})
        existing_files = entry.get("files", [])
        merged, new_records = _gap_fill(existing_files, records)

        result = SourceReconcileResult(
            source_key=source_key,
            status="reconciled" if new_records else "no-op",
            on_disk=len(records),
            already_recorded=len(records) - len(new_records),
            added=len(new_records),
        )
        if dry_run or not new_records:
            return result

        entry["files"] = merged
        if created:
            meta = _catalog.source(source_key)
            entry["source_key"] = source_key
            entry["access_url"] = meta.get("access", {}).get("url", "")
            entry["reconciled_utc"] = datetime.now(timezone.utc).isoformat()
            years = sorted(int(r["year"]) for r in merged if "year" in r)
            if years:
                entry["period"] = f"{years[0]}/{years[-1]}"
        manifest["sources"][source_key] = entry
        _atomic_write(manifest_path, manifest)
    return result


# source_key -> "module:function". Imported lazily so reconciling one source
# doesn't pull in every fetch module's heavy deps (earthaccess, cdsapi).
_RECONCILERS: dict[str, str] = {
    "era5_land": "nhf_spatial_targets.fetch.era5_land:reconcile",
    "mod16a2_v061": "nhf_spatial_targets.fetch.modis:reconcile_mod16a2",
}


def _call_hook(spec: str, project: Project, *, checksum: bool) -> list[dict]:
    module_name, func_name = spec.split(":")
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func(project, checksum=checksum)


def reconcile_manifest(
    project: Project,
    *,
    sources: list[str] | None = None,
    dry_run: bool = False,
    checksum: bool = False,
) -> list[SourceReconcileResult]:
    """Backfill manifest.json from the datastore for the requested sources.

    ``sources=None`` means every catalog source (those without a registered
    reconcile hook are reported with status ``no-hook``). Gap-fill only; see
    module docstring.
    """
    keys = sources if sources else list(_catalog.sources().keys())
    results: list[SourceReconcileResult] = []
    for key in keys:
        spec = _RECONCILERS.get(key)
        if spec is None:
            results.append(SourceReconcileResult(key, "no-hook"))
            continue
        records = _call_hook(spec, project, checksum=checksum)
        if not records:
            results.append(SourceReconcileResult(key, "empty"))
            continue
        results.append(_apply_records(project, key, records, dry_run=dry_run))
    return results
