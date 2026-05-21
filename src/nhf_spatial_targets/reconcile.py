"""Backfill manifest.json provenance from an existing shared datastore.

See docs/architecture/reconcile-manifest.md and issue #160. The merge is
gap-fill only: reconcile appends records for files not already recorded and
never mutates an existing record, so a true ``fetch`` record is never
downgraded to ``reconciled``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def _record_identity(rec: dict) -> tuple[str, object]:
    """Stable dedupe identity for a file record: year if present, else path."""
    if "year" in rec:
        return ("year", int(rec["year"]))
    if "path" in rec:
        return ("path", rec["path"])
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
    seen: set[tuple[str, object]] = set()
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
