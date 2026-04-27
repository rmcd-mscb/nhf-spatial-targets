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

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from shapely.geometry import Point

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


DEFAULT_CALDERA_PROJECT = Path(
    "/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets"
)


def load_project_paths(
    project_dir: Path | None = None,
) -> tuple[Path, Path, dict]:
    """Read ``<project>/config.yml`` and return ``(project_dir, datastore_dir, fabric_cfg)``.

    ``fabric_cfg`` is the ``fabric`` sub-block from ``config.yml`` (keys
    typically include ``path``, ``id_col``, ``crs``, ``buffer_deg``).
    Defaults ``project_dir`` to the caldera ``gfv2-spatial-targets`` project
    when called with ``None``.
    """
    project_dir = (
        Path(project_dir) if project_dir is not None else DEFAULT_CALDERA_PROJECT
    )
    cfg_path = project_dir / "config.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"config.yml not found at {cfg_path}. "
            f"Edit PROJECT_DIR at the top of the notebook to point at "
            f"a real project directory."
        )
    cfg = yaml.safe_load(cfg_path.read_text())
    datastore_dir = Path(cfg["datastore"])
    fabric_cfg = dict(cfg["fabric"])
    return project_dir, datastore_dir, fabric_cfg


def load_fabric(fabric_cfg: dict) -> gpd.GeoDataFrame:
    """Read the HRU fabric file and index by ``fabric_cfg['id_col']``.

    Kept in EPSG:4326 for plotting; downstream area calculations
    re-project to EPSG:5070 (CONUS Albers) so we don't pay that cost
    on every map.
    """
    gdf = gpd.read_file(fabric_cfg["path"])
    gdf = gdf.set_index(fabric_cfg["id_col"])
    return gdf


ALBERS_CRS = "EPSG:5070"  # matches the aggregator's WEIGHT_GEN_CRS


def area_weighted_mean(values: pd.Series, fabric_gdf: gpd.GeoDataFrame) -> float:
    """Compute Σ(v · A) / Σ(A) using fabric area in EPSG:5070.

    Skips NaN values (and their corresponding areas). Aligns on the
    fabric's index — ``values`` must be indexed by HRU id.
    """
    aligned = values.reindex(fabric_gdf.index)
    areas = fabric_gdf.to_crs(ALBERS_CRS).area
    mask = ~aligned.isna()
    if not mask.any():
        return float("nan")
    return float((aligned[mask] * areas[mask]).sum() / areas[mask].sum())


def nan_hru_count(values: pd.Series) -> int:
    """Number of NaN HRUs in ``values``."""
    return int(values.isna().sum())


def lookup_hrus_by_points(
    fabric_gdf: gpd.GeoDataFrame,
    points: dict[str, tuple[float, float]],
) -> dict[str, object]:
    """Resolve ``{label: (lon, lat)}`` to ``{label: hru_id}`` via sjoin.

    Raises ``ValueError`` if any point falls outside the fabric — better
    to fail early than silently drop a regime from the time-series cell.
    """
    pts = gpd.GeoDataFrame(
        {"label": list(points.keys())},
        geometry=[Point(lon, lat) for lon, lat in points.values()],
        crs="EPSG:4326",
    )
    fabric_for_join = fabric_gdf.reset_index()
    id_col = fabric_gdf.index.name
    joined = gpd.sjoin(pts, fabric_for_join, predicate="within", how="left")
    missing = joined[joined[id_col].isna()]["label"].tolist()
    if missing:
        raise ValueError(
            f"REPRESENTATIVE_POINTS lie outside the fabric: {missing}. "
            f"Pick coordinates inside the fabric's CONUS extent."
        )
    return dict(zip(joined["label"], joined[id_col].tolist()))
