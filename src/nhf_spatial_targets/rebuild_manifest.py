"""The one authoritative ``rebuild-manifest`` deterministic projection.

``manifest.json`` is a deterministic projection of (on-disk artifacts x
current catalog x ``fabric.json``). This module computes that projection:

- ``sources[]`` = (datastore consolidated dirs n catalog) U project
  ``data/aggregated/`` dirs. Per-key metadata (DOI, version, access type)
  is pulled from the catalog by key; file lists + periods are parsed from
  on-disk filenames; size/mtime come from ``stat``; sha256 is opt-in. A dir
  whose name is not a catalog key (e.g. ``era5_land_sd``) gets a minimal
  entry tagged ``derived_variant: True`` so shipped NCs are not orphaned.
- ``steps[]`` are synthesized deterministically: ``consolidate`` (datastore
  NCs), ``aggregate`` (aggregated dirs), ``target`` (``targets/`` NCs),
  ``validate`` (``fabric.json``).

Determinism (spec decision E): every timestamp comes from file ``mtime`` via
:func:`nhf_spatial_targets.release.lineage.iso_from_mtime` -- **never**
``datetime.now()``. Sources are emitted with sorted keys; steps are sorted by
:func:`~nhf_spatial_targets.release.lineage.step_sort_key`. Same disk + catalog
+ code -> byte-identical manifest (modulo the opt-in ``--compute-sha256``).

Honesty (spec decision F): every regenerated record is tagged
``provenance: "reconstructed"``.

Non-clobbering: regenerate rewrites only the derived catalog (``sources``,
``steps``); ``created_utc``, the ``fabric`` authorship block, and any
``release`` config are read-merged from the existing manifest.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from nhf_spatial_targets import __version__ as _SOFTWARE_VERSION
from nhf_spatial_targets import catalog as _catalog
from nhf_spatial_targets.release import lineage

if TYPE_CHECKING:
    from nhf_spatial_targets.release.lineage import _Manifest
    from nhf_spatial_targets.workspace import Project

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


def _file_entry(path: Path, *, compute_sha256: bool) -> dict:
    """Return a sorted-stable file record: path/size/mtime (+ opt-in sha256)."""
    entry = dict(lineage._file_basics(path))
    if compute_sha256:
        entry["sha256"] = lineage.sha256_file(path)
    return entry


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
    file_records = [_file_entry(p, compute_sha256=compute_sha256) for p in files]

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

    PR-7 persists the full resolved per-target params as NC global attrs; PR-2
    reads back whatever already exists. A placeholder / non-NetCDF file (tests,
    a truncated write) simply yields ``{}`` -- the read is never fatal.
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
    except Exception:
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
    ``iso_from_mtime`` of the step's first output (never ``datetime.now()``);
    every record is tagged ``provenance: "reconstructed"``; the list is sorted
    by :func:`~nhf_spatial_targets.release.lineage.step_sort_key`.
    """
    catalog_keys = set(_catalog.sources())
    steps: list[dict] = []

    def _record(*, kind, source_key, outputs, params=None):
        # Anchor the timestamp to the first output's mtime. Steps always carry
        # at least one output here except an empty aggregated/consolidated dir,
        # which is skipped by the callers below, so first-output always exists.
        timestamp = lineage.iso_from_mtime(Path(outputs[0]["path"]))
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
        outputs = [
            lineage.output_file_entry(p, compute_sha256=compute_sha256)
            for p in _ncs_in(source_dir)
        ]
        if not outputs:
            continue
        _record(kind="consolidate", source_key=source_dir.name, outputs=outputs)

    # aggregate: every data/aggregated/<key>/ dir (incl. derived variants).
    aggregated_root = project_dir / "data" / "aggregated"
    for source_dir in _sorted_source_dirs(aggregated_root):
        outputs = [
            lineage.output_file_entry(p, compute_sha256=compute_sha256)
            for p in _ncs_in(source_dir)
        ]
        if not outputs:
            continue
        _record(kind="aggregate", source_key=source_dir.name, outputs=outputs)

    # target: one per targets/*.nc (one-level glob; never per-year intermediates).
    targets_dir = project_dir / "targets"
    if targets_dir.is_dir():
        for nc in sorted(targets_dir.glob("*.nc"), key=str):
            if not nc.is_file():
                continue
            outputs = [lineage.output_file_entry(nc, compute_sha256=compute_sha256)]
            _record(
                kind="target",
                source_key=None,
                outputs=outputs,
                params=_target_nc_params(nc),
            )

    # validate: the fabric.json anchor.
    fabric_json = project_dir / "fabric.json"
    if fabric_json.exists():
        outputs = [
            lineage.output_file_entry(fabric_json, compute_sha256=compute_sha256)
        ]
        _record(kind="validate", source_key=None, outputs=outputs)

    return sorted(steps, key=lineage.step_sort_key)


def _project_source_dirs(project: "Project") -> dict[str, Path]:
    """Return ``{source_key: source_dir}`` for the sources[] union.

    The union is (datastore consolidated dirs n catalog) U all
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

    ``sources`` is the (datastore n catalog) U ``data/aggregated/`` union;
    ``steps`` is the deterministic :func:`synthesize_steps` output. Identity
    fields are read-merged from the existing manifest -- ``created_utc``,
    ``last_validated_utc`` (never re-minted), the ``fabric`` authorship block,
    and any extra top-level blocks (e.g. ``release``) are preserved verbatim;
    only the derived catalog (``sources`` / ``steps``) and
    ``manifest_schema_version`` are rewritten.

    With ``dry_run=True`` the projection is returned without touching disk;
    otherwise it is written via ``lineage.with_flock`` + atomic rename. The
    result is byte-identical on re-run for the same disk + catalog + code
    (modulo the opt-in ``compute_sha256``); no ``datetime.now()`` is on the
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
