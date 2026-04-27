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

import pandas as pd
import xarray as xr

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


def select_month(da: xr.DataArray, year: int, month: int) -> xr.DataArray:
    """Select the first timestep in the given calendar month.

    Slices ``da`` to the calendar-month window ``[YYYY-MM-01, YYYY-MM-end]``
    and returns the first hit. Robust to start-of-month / end-of-month /
    mid-month timestamping conventions; the canonical SOM pattern from
    ``docs/references/calibration-target-recipes.md`` lesson 2.

    Raises ``IndexError`` if the window contains no timesteps.
    """
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    sliced = da.sel(time=slice(start, end))
    if sliced.sizes.get("time", 0) == 0:
        raise IndexError(
            f"No timesteps in {da.name or 'array'} between {start.date()} "
            f"and {end.date()}"
        )
    return sliced.isel(time=0)


def discover_aggregated(project_dir: Path, source_key: str) -> list[Path] | None:
    """Return sorted per-year aggregated NC paths, or ``None`` if absent.

    Globs ``<project>/data/aggregated/<source_key>/<source_key>_*_agg.nc``.
    Returns ``None`` when the directory is missing or empty so callers
    can print a single "skip with reason" line and continue.
    """
    agg_dir = Path(project_dir) / "data" / "aggregated" / source_key
    if not agg_dir.is_dir():
        return None
    paths = sorted(agg_dir.glob(f"{source_key}_*_agg.nc"))
    return paths if paths else None
