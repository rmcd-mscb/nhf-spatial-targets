"""Read/merge/write ``catalog/release_registry.yml`` -- publish intent.

The registry is the **source of truth for what we intend to have published**
to ScienceBase: the umbrella DOI parent, one entry per consolidated-source
child, and one entry per fabric child. ScienceBase itself remains the source
of truth for *what actually exists*; reconciling the two (``diff_local_vs_remote``)
needs live SB queries and lives in the publish/status layer, not here.
This module stays pure-local.

Two invariants drive the implementation:

- **Read-merge-write under flock, never overwrite.** Two operators publishing
  different fabrics could otherwise race and clobber each other's entries --
  the same class of bug that wiped 13 sources of manifest provenance in #97.
  Every ``put_*`` reloads the file inside an advisory ``LOCK_EX`` critical
  section (reusing :func:`nhf_spatial_targets.release.lineage.with_flock`),
  merges its one entry, and atomically renames the result into place. Sibling
  entries (other sources, other fabrics, the umbrella) are preserved.

- **Comment header + key order survive every write.** The committed scaffold
  carries an explanatory comment block and the load-bearing ``umbrella: null``
  sentinel; ``yaml.safe_dump`` would strip both. We use ``ruamel.yaml`` in
  round-trip mode so the header and key order are preserved across writes.

A corrupt registry is a **loud failure** (mirroring
:func:`nhf_spatial_targets.release.lineage.read_manifest`), never a silent
reset -- the file records real publish state and silently re-scaffolding it
would orphan live ScienceBase items.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from nhf_spatial_targets.release.lineage import with_flock

logger = logging.getLogger(__name__)

# Repo root is three parents up: release/ -> nhf_spatial_targets/ -> src/ -> root.
_CATALOG_DIR = Path(__file__).resolve().parents[3] / "catalog"
DEFAULT_REGISTRY_PATH = _CATALOG_DIR / "release_registry.yml"

# Field schema enforced by the put_* writers. Kept here so the scaffold
# comment, the plan cheat-sheet, and the code can't drift apart silently.
UMBRELLA_FIELDS: tuple[str, ...] = ("sb_id", "doi", "version", "published_utc", "title")
CHILD_FIELDS: tuple[str, ...] = (
    "sb_id",
    "uploaded_utc",
    "file_count",
    "total_bytes",
    "manifest_sha256",
)

# Seed document used only when the registry file is absent (it is normally
# committed). Carries the header comment so a freshly-seeded file is still
# self-documenting, matching the committed scaffold's intent.
_SCAFFOLD_TEXT = """\
# Running publish-state for the ScienceBase data release. Updated by
# `nhf-targets release publish` as items are created or refreshed;
# flock-protected on write so concurrent publishes don't clobber each other.
#
# `umbrella: null` signals "no first-publish yet"; loaders handle that case
# explicitly rather than indexing into a missing dict.
#
# Per-item schema, written by registry.put_*:
#   umbrella:                   {sb_id, doi, version, published_utc, title}
#   consolidated_sources.<key>: {sb_id, uploaded_utc, file_count,
#                                total_bytes, manifest_sha256}
#   fabrics.<label>:            {sb_id, uploaded_utc, file_count,
#                                total_bytes, manifest_sha256}

umbrella: null
consolidated_sources: {}
fabrics: {}
"""

# Sentinel distinguishing "argument not supplied" (keep the current value)
# from an explicit ``None`` (e.g. clearing a DOI). Used by put_umbrella so a
# refresh that only sets the DOI doesn't blank out sb_id/version/etc.
_KEEP = object()


def _yaml() -> YAML:
    """Return a round-trip YAML handler tuned for the registry.

    Round-trip mode preserves the comment header and key order. A wide line
    width stops ruamel from wrapping long values (e.g. titles) mid-write,
    which would otherwise churn diffs.
    """
    y = YAML()  # default typ="rt" (round-trip)
    y.preserve_quotes = True
    y.width = 4096
    return y


def _load_doc(path: Path):
    """Load the registry document, seeding from the scaffold if absent.

    Returns the parsed mapping (a ruamel ``CommentedMap``) with all three
    top-level keys guaranteed present. ``umbrella`` may be ``None``.

    A genuinely *absent* file is seeded in memory from the scaffold (the
    first-run case). A file that exists but is empty is treated as suspicious
    -- the same loud failure as a corrupt file -- because a registry truncated
    to zero bytes (e.g. a crashed writer) must never be silently re-scaffolded
    over live publish state.

    Raises
    ------
    ValueError
        The file exists but is empty, does not parse, or its top level is not
        a mapping. A populated registry is never silently reset.
    """
    yaml = _yaml()
    if not path.exists():
        return _normalize(yaml.load(_SCAFFOLD_TEXT))
    try:
        data = yaml.load(path.read_text())
    except YAMLError as exc:
        raise ValueError(
            f"release_registry.yml at {path} is corrupt: {exc}. "
            f"Inspect the file manually or restore from git; do NOT "
            f"delete it -- it records live ScienceBase publish state."
        ) from exc
    if data is None:
        raise ValueError(
            f"release_registry.yml at {path} exists but is empty. This is "
            f"treated as corruption, not a fresh start: restore it from git "
            f"(it records live ScienceBase publish state). Delete the file "
            f"only if you intend to re-seed the scaffold from scratch."
        )
    if not hasattr(data, "get"):
        raise ValueError(
            f"release_registry.yml at {path} must be a YAML mapping at the "
            f"top level; got {type(data).__name__}."
        )
    return _normalize(data)


def _normalize(data):
    """Guarantee the two container sections exist so callers never index into
    a missing dict. ``umbrella`` is intentionally left as-is (its ``None``
    sentinel is load-bearing).
    """
    if data.get("consolidated_sources") is None:
        data["consolidated_sources"] = {}
    if data.get("fabrics") is None:
        data["fabrics"] = {}
    return data


def _atomic_dump(path: Path, data) -> None:
    """Serialize *data* to *path* via tempfile + rename.

    ``Path.replace`` is atomic on the same filesystem, so a partial registry
    never lands at the final path even if the process dies mid-write.
    """
    yaml = _yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    yaml.dump(data, buf)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".yml.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(buf.getvalue())
        from nhf_spatial_targets.io_nc import apply_umask_mode

        apply_umask_mode(tmp_path)
        Path(tmp_path).replace(path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _lock_path(path: Path) -> Path:
    """Advisory-lock path for *path* (mirrors lineage's ``<file>.lock``)."""
    return path.with_suffix(path.suffix + ".lock")


def _resolve(path: Path | None) -> Path:
    return DEFAULT_REGISTRY_PATH if path is None else path


# ---------------------------------------------------------------------------
# Pure-local readers
# ---------------------------------------------------------------------------


def load_registry(path: Path | None = None) -> dict:
    """Load + normalize the registry.

    Returns a mapping with ``umbrella`` (a dict or ``None``),
    ``consolidated_sources`` (dict), and ``fabrics`` (dict). The returned
    object is a live ruamel mapping; mutating it does **not** persist unless
    written back via a ``put_*`` helper.

    Parameters
    ----------
    path
        Registry file path. Defaults to the committed
        ``catalog/release_registry.yml``. Tests pass a tmp path.

    Raises
    ------
    ValueError
        The file is present but corrupt or not a top-level mapping.
    """
    return _load_doc(_resolve(path))


def get_umbrella(path: Path | None = None) -> dict | None:
    """Return the umbrella entry, or ``None`` if not yet published.

    The ``umbrella: null`` sentinel is returned as ``None`` -- callers must
    branch on it rather than assume a dict.
    """
    umbrella = load_registry(path).get("umbrella")
    return dict(umbrella) if umbrella is not None else None


def get_source(key: str, path: Path | None = None) -> dict | None:
    """Return the consolidated-source entry for *key*, or ``None``."""
    entry = load_registry(path)["consolidated_sources"].get(key)
    return dict(entry) if entry is not None else None


def get_fabric(label: str, path: Path | None = None) -> dict | None:
    """Return the fabric entry for *label*, or ``None``."""
    entry = load_registry(path)["fabrics"].get(label)
    return dict(entry) if entry is not None else None


# ---------------------------------------------------------------------------
# Flock-protected writers (read-merge-write, never overwrite)
# ---------------------------------------------------------------------------


def put_umbrella(
    *,
    sb_id: str,
    version: str,
    published_utc: str,
    title: str,
    doi: str | None = _KEEP,  # type: ignore[assignment]
    path: Path | None = None,
) -> dict:
    """Write (merge) the umbrella entry under flock; return the merged dict.

    The umbrella's own fields are set from the arguments; the two container
    sections (``consolidated_sources``, ``fabrics``) are preserved untouched.
    ``doi`` defaults to the sentinel ``_KEEP``: omitting it keeps any existing
    DOI (so the post-IPDS ``--refresh-doi`` flow can set just the DOI without
    re-supplying the rest, and a first publish leaves ``doi: null``).

    Parameters
    ----------
    sb_id, version, published_utc, title
        Umbrella item fields. See :data:`UMBRELLA_FIELDS`.
    doi
        DOI string, or ``None`` to record "no DOI yet". Omit to keep the
        current value.
    path
        Registry path. Defaults to the committed file.
    """
    target = _resolve(path)

    def _do() -> dict:
        data = _load_doc(target)
        current = data.get("umbrella") or {}
        merged = dict(current)
        merged["sb_id"] = sb_id
        merged["version"] = version
        merged["published_utc"] = published_utc
        merged["title"] = title
        if doi is not _KEEP:
            merged["doi"] = doi
        merged.setdefault("doi", None)
        # Canonical field order so the serialized block reads predictably.
        ordered = {k: merged[k] for k in UMBRELLA_FIELDS if k in merged}
        data["umbrella"] = ordered
        _atomic_dump(target, data)
        return ordered

    result = with_flock(_lock_path(target), _do)
    logger.info("registry: put umbrella sb_id=%s version=%s", sb_id, version)
    return result


def _put_child(
    section: str,
    key: str,
    *,
    sb_id: str,
    uploaded_utc: str,
    file_count: int,
    total_bytes: int,
    manifest_sha256: str,
    path: Path | None,
) -> dict:
    """Shared read-merge-write for a single source/fabric child entry."""
    target = _resolve(path)

    def _do() -> dict:
        data = _load_doc(target)
        container = data[section]
        current = dict(container.get(key) or {})
        current.update(
            {
                "sb_id": sb_id,
                "uploaded_utc": uploaded_utc,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "manifest_sha256": manifest_sha256,
            }
        )
        ordered = {k: current[k] for k in CHILD_FIELDS if k in current}
        container[key] = ordered
        _atomic_dump(target, data)
        return ordered

    return with_flock(_lock_path(target), _do)


def put_source(
    key: str,
    *,
    sb_id: str,
    uploaded_utc: str,
    file_count: int,
    total_bytes: int,
    manifest_sha256: str,
    path: Path | None = None,
) -> dict:
    """Write (merge) a consolidated-source entry under flock.

    Other sources, the fabrics section, and the umbrella are preserved.
    Re-publishing the same *key* updates that entry in place. See
    :data:`CHILD_FIELDS` for the recorded schema.
    """
    result = _put_child(
        "consolidated_sources",
        key,
        sb_id=sb_id,
        uploaded_utc=uploaded_utc,
        file_count=file_count,
        total_bytes=total_bytes,
        manifest_sha256=manifest_sha256,
        path=path,
    )
    logger.info("registry: put source %s sb_id=%s files=%d", key, sb_id, file_count)
    return result


def put_fabric(
    label: str,
    *,
    sb_id: str,
    uploaded_utc: str,
    file_count: int,
    total_bytes: int,
    manifest_sha256: str,
    path: Path | None = None,
) -> dict:
    """Write (merge) a fabric entry under flock.

    Other fabrics, the sources section, and the umbrella are preserved.
    Re-publishing the same *label* updates that entry in place. See
    :data:`CHILD_FIELDS` for the recorded schema.
    """
    result = _put_child(
        "fabrics",
        label,
        sb_id=sb_id,
        uploaded_utc=uploaded_utc,
        file_count=file_count,
        total_bytes=total_bytes,
        manifest_sha256=manifest_sha256,
        path=path,
    )
    logger.info("registry: put fabric %s sb_id=%s files=%d", label, sb_id, file_count)
    return result
