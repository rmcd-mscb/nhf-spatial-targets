"""Shared helpers for the inspect_aggregated_*.ipynb notebooks.

This module is a sibling of the notebooks (not packaged into
nhf_spatial_targets). It holds path discovery, fabric I/O, HRU
choropleth plotting, area-weighted means, time-window slicing,
representative-point lookup, catalog-units lookup, and a save-figure
helper used to populate docs/figures/inspect_aggregated/ for downstream
slide / documentation work.

Notebooks import via:

    import sys
    sys.path.insert(0, str(Path.cwd()))   # if needed
    from _helpers import (
        load_project_paths, load_fabric, discover_aggregated, ...
    )

Or, since Jupyter puts the notebook's directory on sys.path, simply:

    from _helpers import load_project_paths, ...
"""

from __future__ import annotations

from pathlib import Path

from nhf_spatial_targets import catalog as cat

SAVE_FIGURES: bool = False
FIGURES_DIR: Path = Path("docs/figures/inspect_aggregated/")


def unit_from_catalog(source_key: str, var: str) -> str:
    """Return units for ``var`` from ``catalog/sources.yml``.

    The catalog has two shapes for ``variables``: a list of dicts (each
    with ``name`` and ``cf_units`` / ``units``), or a flat list of names
    with the unit at the source level. Both are handled. Reading units
    from the catalog (rather than from on-disk attrs) is the convention
    enforced by ``docs/references/calibration-target-recipes.md`` lesson 9.
    """
    src = cat.source(source_key)
    variables = src.get("variables", [])
    for entry in variables:
        if isinstance(entry, dict):
            if entry.get("name") == var or entry.get("file_variable") == var:
                return entry.get("cf_units") or entry.get("units") or src["units"]
        elif entry == var:
            return src["units"]
    raise KeyError(
        f"Variable {var!r} not found in catalog entry for {source_key!r} "
        f"(available: {variables})"
    )
