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

# Aggregated NCs end in ``_agg.nc``; an optional 4-digit year may precede it.
# ``<key>_agg.nc`` | ``<key>_<year>_agg.nc`` | ``<key>_<region>_<year>_agg.nc``.
_AGG_YEAR_RE = re.compile(r"_(?P<year>\d{4})_agg\.nc$")


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
