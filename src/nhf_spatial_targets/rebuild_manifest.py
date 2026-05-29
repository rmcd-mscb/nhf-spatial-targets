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

from nhf_spatial_targets import catalog as _catalog
from nhf_spatial_targets.release import lineage

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
