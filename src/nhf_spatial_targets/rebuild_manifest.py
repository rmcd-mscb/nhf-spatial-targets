"""The one authoritative ``rebuild-manifest`` deterministic projection.

``manifest.json`` is a deterministic projection of (on-disk artifacts x
current catalog x ``fabric.json``). This module computes that projection:

- ``sources[]`` = (datastore consolidated dirs ∩ catalog) ∪ project
  ``data/aggregated/`` dirs. Per-key catalog metadata (the ``access`` block,
  plus any top-level ``doi`` / ``version`` / ``access_type``) is pulled from
  the catalog by key; file lists + periods are parsed from on-disk filenames;
  size/mtime come from ``stat``; sha256 is opt-in. A dir whose name is not a
  catalog key (e.g. ``era5_land_sd``) gets a minimal entry tagged
  ``derived_variant: True`` so shipped NCs are not orphaned.
- ``steps[]`` are synthesized deterministically: ``consolidate`` (datastore
  NCs), ``aggregate`` (aggregated dirs), ``target`` (``targets/`` NCs),
  ``validate`` (``fabric.json``).

Determinism (spec decision E): every timestamp comes from file ``mtime`` via
:func:`nhf_spatial_targets.release.lineage.iso_from_mtime` -- **never** a
wall-clock read. Sources are emitted with sorted keys; steps are sorted by
:func:`~nhf_spatial_targets.release.lineage.step_sort_key`. Same disk + catalog
+ code -> byte-identical manifest (modulo the opt-in ``--compute-sha256``).

Honesty (spec decision F): every regenerated record is tagged
``provenance: "reconstructed"``.

Non-clobbering: regenerate rewrites only the derived catalog (``sources``,
``steps``); ``created_utc``, the ``fabric`` authorship block, and any
``release`` config are read-merged from the existing manifest.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from nhf_spatial_targets import __version__ as _SOFTWARE_VERSION
from nhf_spatial_targets import catalog as _catalog
from nhf_spatial_targets.release import lineage

if TYPE_CHECKING:
    from nhf_spatial_targets.release.lineage import _Manifest
    from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)

# Aggregated NCs end in ``_agg.nc``; an optional 4-digit year may precede it.
# ``<key>_agg.nc`` | ``<key>_<year>_agg.nc`` | ``<key>_<region>_<year>_agg.nc``.
_AGG_YEAR_RE = re.compile(r"_(?P<year>\d{4})_agg\.nc$")

# A standalone 19xx/20xx token anywhere in a filename. Constrained to plausible
# calendar years so source-version tokens (``v061``) and resolution tokens are
# not mistaken for years when deriving a consolidated source's period.
_YEAR_TOKEN_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

# Catalog metadata copied verbatim onto a source entry when the directory name
# is a known catalog key. Each is guarded with ``.get`` so a sparse catalog
# entry never raises.
_CATALOG_METADATA_KEYS = ("access", "doi", "version", "access_type")


def parse_aggregated_filename(name: str) -> tuple[str, int | None]:
    """Return ``(stem, year)`` for an aggregated NC filename.

    ``stem`` is the filename with the trailing ``_agg.nc`` (and year, if any)
    stripped -- diagnostic only; the authoritative source key is the parent
    directory name. ``year`` is ``None`` for single-shot aggregates.
    """
    m = _AGG_YEAR_RE.search(name)
    if m:
        year = int(m.group("year"))
        stem = name[: m.start()]
        return stem, year
    stem = name[: -len("_agg.nc")] if name.endswith("_agg.nc") else name
    return stem, None


def _year_in_filename(name: str) -> int | None:
    """Return the year parsed from a filename, or ``None``.

    Prefers the authoritative aggregated-NC pattern (``_<year>_agg.nc``);
    falls back to the first standalone 19xx/20xx token for consolidated NCs
    (e.g. ``merra2_2000.nc``).
    """
    _, year = parse_aggregated_filename(name)
    if year is not None:
        return year
    m = _YEAR_TOKEN_RE.search(name)
    return int(m.group()) if m else None


def _guarded_output_entries(
    paths: list[Path], *, compute_sha256: bool, source_key: str | None = None
) -> list[dict]:
    """Build file/output entries, skipping any path that vanishes mid-scan.

    The datastore is a shared Lustre filesystem; a concurrent scratch-purge,
    another project's run, or a stale NFS/Lustre handle can make a file
    disappear between the ``rglob``/``glob`` enumeration and the ``stat()``
    inside :func:`~nhf_spatial_targets.release.lineage.output_file_entry`. A
    single such race must not abort the whole multi-source rebuild (the
    deleted ``reconcile.py`` / ``release/rebuild.py`` carried this same
    log-and-continue posture): the offending path is logged at WARNING and
    skipped, every other artifact is still recorded.
    """
    entries: list[dict] = []
    for path in paths:
        try:
            entries.append(
                lineage.output_file_entry(path, compute_sha256=compute_sha256)
            )
        except OSError as exc:
            logger.warning(
                "rebuild-manifest: skipping %s%s (%s); vanished or unreadable "
                "mid-scan.",
                path,
                f" for source {source_key!r}" if source_key else "",
                exc,
            )
    return entries


def build_source_entry(
    source_key: str,
    source_dir: Path,
    *,
    compute_sha256: bool,
) -> dict:
    """Build one ``sources[]`` entry for a source directory.

    The source key is the directory name (under ``<datastore>/`` for
    consolidated sources or ``<project>/data/aggregated/`` for aggregated
    ones). When the key is a known catalog source, a fixed allowlist of
    catalog metadata is copied onto the entry and ``derived_variant`` is
    ``False``; otherwise (e.g. ``era5_land_sd``) a minimal entry tagged
    ``derived_variant: True`` is returned so a shipped NC is never orphaned.

    Every entry carries ``provenance: "reconstructed"`` (spec decision F), a
    sorted ``files[]`` (path/size/mtime, plus ``sha256`` only when
    ``compute_sha256``), and a derived ``period`` (``"<min>/<max>"``) whenever
    years can be parsed from the filenames.
    """
    is_catalog = source_key in _catalog.sources()

    files = sorted(
        (p for p in source_dir.rglob("*.nc") if p.is_file()),
        key=str,
    )
    file_records = _guarded_output_entries(
        files, compute_sha256=compute_sha256, source_key=source_key
    )

    # Years come from filenames (no stat), so a file skipped above by a
    # vanish-race still contributes its year to the derived period.
    years = sorted({y for p in files if (y := _year_in_filename(p.name)) is not None})

    entry: dict = {
        "source_key": source_key,
        "provenance": "reconstructed",
        "derived_variant": not is_catalog,
        "files": file_records,
    }
    if years:
        entry["period"] = f"{years[0]}/{years[-1]}"
    if is_catalog:
        meta = _catalog.source(source_key)
        for key in _CATALOG_METADATA_KEYS:
            value = meta.get(key)
            if value is not None:
                entry[key] = value
    return entry


def _sorted_source_dirs(parent: Path) -> list[Path]:
    """Return the immediate sub-directories of *parent*, sorted by name."""
    if not parent.is_dir():
        return []
    return sorted((p for p in parent.iterdir() if p.is_dir()), key=lambda p: p.name)


def _ncs_in(source_dir: Path) -> list[Path]:
    """Return the ``*.nc`` files under *source_dir*, recursively, sorted."""
    return sorted((p for p in source_dir.rglob("*.nc") if p.is_file()), key=str)


def _target_nc_params(nc: Path) -> dict:
    """Best-effort read of resolved-target attrs (``period``, ``sources``).

    Reads the ``period`` / ``sources`` global attrs when present. Target
    writers may not yet persist the full resolved param set, so a missing attr
    -- or a non-NetCDF placeholder file -- simply yields ``{}``; the read is
    never fatal to the projection. A present-but-unreadable NC (a truncated or
    corrupt *published* target) is logged at WARNING so the lost resolved-param
    provenance is at least visible, rather than vanishing silently.
    """
    params: dict = {}
    try:
        import xarray as xr

        with xr.open_dataset(nc) as ds:
            for attr in ("period", "sources"):
                if attr in ds.attrs:
                    value = ds.attrs[attr]
                    # numpy scalars / arrays -> plain Python for JSON stability.
                    params[attr] = value.tolist() if hasattr(value, "tolist") else value
    except Exception as exc:
        logger.warning(
            "rebuild-manifest: could not read resolved-target attrs from %s "
            "(%s); the target step's params will be empty. If this is a "
            "published target NC and not a placeholder, it may be truncated "
            "or corrupt.",
            nc,
            exc,
        )
        return {}
    return params


def synthesize_steps(
    *,
    datastore: Path,
    project_dir: Path,
    compute_sha256: bool,
) -> list[dict]:
    """Synthesize the deterministic ``steps[]`` for a project.

    One ``consolidate`` step per datastore source dir that is a catalog key
    (matching the ``sources[]`` union), one ``aggregate`` step per
    ``data/aggregated/<key>/`` dir, one ``target`` step per ``targets/*.nc``,
    and one ``validate`` step for ``fabric.json``. Every timestamp is
    ``iso_from_mtime`` of the step's first output (never a wall-clock read);
    every record is tagged ``provenance: "reconstructed"``; the list is sorted
    by :func:`~nhf_spatial_targets.release.lineage.step_sort_key`.
    """
    catalog_keys = set(_catalog.sources())
    steps: list[dict] = []

    def _record(*, kind, source_key, outputs, params=None):
        # Reuse the mtime already captured when ``outputs`` was built rather
        # than re-stat'ing outputs[0] -- that avoids a second TOCTOU window and
        # keeps the step timestamp byte-consistent with the recorded output.
        # ``outputs`` is always non-empty here: every caller skips a group that
        # produced no entries (empty dir, or all files vanished mid-scan).
        timestamp = outputs[0]["mtime_utc"]
        record = lineage.build_step_record(
            kind=kind,
            source_key=source_key,
            outputs=outputs,
            params=params or {},
            command=f"rebuild-manifest:{kind}",
            timestamp_utc=timestamp,
        )
        record["provenance"] = "reconstructed"
        steps.append(record)

    # consolidate: datastore source dirs that are catalog keys.
    for source_dir in _sorted_source_dirs(datastore):
        if source_dir.name not in catalog_keys:
            continue
        outputs = _guarded_output_entries(
            _ncs_in(source_dir),
            compute_sha256=compute_sha256,
            source_key=source_dir.name,
        )
        if not outputs:
            continue
        _record(kind="consolidate", source_key=source_dir.name, outputs=outputs)

    # aggregate: every data/aggregated/<key>/ dir (incl. derived variants).
    aggregated_root = project_dir / "data" / "aggregated"
    for source_dir in _sorted_source_dirs(aggregated_root):
        outputs = _guarded_output_entries(
            _ncs_in(source_dir),
            compute_sha256=compute_sha256,
            source_key=source_dir.name,
        )
        if not outputs:
            continue
        _record(kind="aggregate", source_key=source_dir.name, outputs=outputs)

    # target: one per targets/*.nc (one-level glob; never per-year intermediates).
    targets_dir = project_dir / "targets"
    if targets_dir.is_dir():
        for nc in sorted(targets_dir.glob("*.nc"), key=str):
            if not nc.is_file():
                continue
            outputs = _guarded_output_entries([nc], compute_sha256=compute_sha256)
            if not outputs:
                continue
            _record(
                kind="target",
                source_key=None,
                outputs=outputs,
                params=_target_nc_params(nc),
            )

    # validate: the fabric.json anchor.
    fabric_json = project_dir / "fabric.json"
    if fabric_json.exists():
        outputs = _guarded_output_entries([fabric_json], compute_sha256=compute_sha256)
        if outputs:
            _record(kind="validate", source_key=None, outputs=outputs)

    return sorted(steps, key=lineage.step_sort_key)


def _project_source_dirs(project: "Project") -> dict[str, Path]:
    """Return ``{source_key: source_dir}`` for the sources[] union.

    The union is (datastore consolidated dirs ∩ catalog) ∪ all
    ``data/aggregated/`` dirs. When a key appears in both, the aggregated
    (project-specific final) dir wins -- that is the artifact the published
    fabric ships, and the consolidate step still records the datastore NCs.
    Keys are returned in sorted order for deterministic emission.
    """
    catalog_keys = set(_catalog.sources())
    dirs: dict[str, Path] = {}
    for d in _sorted_source_dirs(project.datastore):
        if d.name in catalog_keys:
            dirs[d.name] = d
    for d in _sorted_source_dirs(project.aggregated_dir()):
        dirs[d.name] = d
    return {key: dirs[key] for key in sorted(dirs)}


def rebuild_manifest(
    project: "Project",
    *,
    compute_sha256: bool = False,
    dry_run: bool = False,
) -> "_Manifest":
    """Project the on-disk artifacts into a complete, canonical ``manifest.json``.

    ``sources`` is the (datastore ∩ catalog) ∪ ``data/aggregated/`` union;
    ``steps`` is the deterministic :func:`synthesize_steps` output. Identity
    fields are read-merged from the existing manifest -- ``created_utc``,
    ``last_validated_utc`` (never re-minted), the ``fabric`` authorship block,
    and any extra top-level blocks (e.g. ``release``) are preserved verbatim;
    only the derived catalog (``sources`` / ``steps``) and
    ``manifest_schema_version`` are rewritten.

    With ``dry_run=True`` the projection is returned without touching disk;
    otherwise it is written via ``lineage.with_flock`` + atomic rename. The
    result is byte-identical on re-run for the same disk + catalog + code
    (modulo the opt-in ``compute_sha256``); no wall-clock read is on the
    path (all timestamps come from file mtime).
    """
    manifest_path = project.manifest_path
    existing = lineage.read_manifest(manifest_path)

    source_dirs = _project_source_dirs(project)
    sources = {
        key: build_source_entry(key, source_dir, compute_sha256=compute_sha256)
        for key, source_dir in source_dirs.items()
    }
    steps = synthesize_steps(
        datastore=project.datastore,
        project_dir=project.workdir,
        compute_sha256=compute_sha256,
    )

    manifest = lineage._new_manifest_skeleton()
    # Read-merge identity fields (never re-mint a clock; never clobber authorship).
    manifest["created_utc"] = existing.get("created_utc")
    manifest["last_validated_utc"] = existing.get("last_validated_utc")
    manifest["nhf_spatial_targets_version"] = (
        existing.get("nhf_spatial_targets_version") or _SOFTWARE_VERSION
    )
    manifest["fabric"] = existing.get("fabric")
    # Regenerate the derived catalog.
    manifest["sources"] = sources
    manifest["steps"] = steps
    # Preserve any extra top-level blocks (e.g. a release config) verbatim.
    for key, value in existing.items():
        if key not in manifest:
            manifest[key] = value

    if dry_run:
        return manifest

    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    lineage.with_flock(
        lock_path, lambda: lineage.atomic_write_manifest(manifest_path, manifest)
    )
    return manifest
