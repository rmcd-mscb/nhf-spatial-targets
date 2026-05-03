"""Shared helpers for the inspect_aggregated_*.ipynb notebooks.

This module is a sibling of the notebooks (not packaged into
nhf_spatial_targets). It holds path discovery, fabric I/O, HRU
choropleth plotting, area-weighted means, time-window slicing,
representative-point lookup, catalog-units lookup, and a save-figure
helper used to populate docs/figures/aggregated/ for downstream
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
FIGURES_DIR: Path = Path("docs/figures/aggregated/")
PROJECT: str | None = None


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
    ).to_crs(fabric_gdf.crs)
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


def open_year(project_dir: Path, source_key: str, year: int) -> xr.Dataset:
    """Open a single per-year aggregated NC and detach from the file.

    The Dataset is loaded into memory and the underlying file handle is
    closed before return — the SOM/MERRA-2 family of xarray bugs around
    file handles staying open is covered by ``feedback_rioxarray_close.md``.
    """
    paths = discover_aggregated(project_dir, source_key)
    if paths is None:
        raise FileNotFoundError(
            f"No aggregated NCs for {source_key} in {project_dir}"
        )
    target = next((p for p in paths if f"_{year}_agg.nc" in p.name), None)
    if target is None:
        raise FileNotFoundError(
            f"No {source_key}_{year}_agg.nc in {paths[0].parent}"
        )
    with xr.open_dataset(target) as ds:
        loaded = ds.load()
    return loaded


def open_year_range(
    project_dir: Path, source_key: str, years: range
) -> xr.Dataset:
    """Open a contiguous range of per-year aggregated NCs (lazy).

    Caller is responsible for ``.sel(hru=[...])`` and ``.load()`` to bound
    memory, then ``.close()`` afterwards.
    """
    paths = discover_aggregated(project_dir, source_key)
    if paths is None:
        raise FileNotFoundError(
            f"No aggregated NCs for {source_key} in {project_dir}"
        )
    wanted = [
        p for p in paths
        if any(f"_{y}_agg.nc" in p.name for y in years)
    ]
    if not wanted:
        raise FileNotFoundError(
            f"None of years {list(years)} present in {paths[0].parent}"
        )
    return xr.open_mfdataset(wanted, combine="by_coords")


def plot_hru_choropleth(
    ax,
    fabric_gdf: gpd.GeoDataFrame,
    values: pd.Series,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "YlGnBu",
    title: str = "",
    units: str = "",
    nan_color: str = "lightgrey",
) -> None:
    """Render an HRU-level choropleth with NaN HRUs in ``nan_color``.

    Joins ``values`` (indexed by HRU id) onto ``fabric_gdf``. NaN HRUs
    are plotted first in ``nan_color`` so coverage gaps are visually
    obvious; finite-value HRUs are plotted on top.
    """
    plot_gdf = fabric_gdf.copy()
    plot_gdf["value"] = values.reindex(plot_gdf.index)

    nan_mask = plot_gdf["value"].isna()
    if nan_mask.any():
        plot_gdf[nan_mask].plot(ax=ax, color=nan_color, edgecolor="none")

    plot_gdf[~nan_mask].plot(
        ax=ax,
        column="value",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        legend=True,
        legend_kwds={"label": units, "shrink": 0.6},
        edgecolor="none",
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def plot_nan_hrus(
    ax,
    fabric_gdf: gpd.GeoDataFrame,
    values: pd.Series,
    *,
    title: str = "",
) -> None:
    """Boolean coverage map: NaN HRUs in red, finite HRUs in light grey."""
    plot_gdf = fabric_gdf.copy()
    plot_gdf["is_nan"] = values.reindex(plot_gdf.index).isna()
    plot_gdf[~plot_gdf["is_nan"]].plot(ax=ax, color="lightgrey", edgecolor="none")
    plot_gdf[plot_gdf["is_nan"]].plot(ax=ax, color="crimson", edgecolor="none")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def daily_coverage_summary(
    ds: xr.Dataset,
    var: str,
    *,
    threshold: float = 0.0,
    coverage_var: str | None = None,
    coverage_threshold: float = 0.7,
) -> pd.DataFrame:
    """Per-timestep HRU counts for picking inspection-worthy days.

    Computes, per timestep, the number of HRUs where:
      - ``var`` is finite (column ``n_finite``)
      - ``var`` strictly exceeds ``threshold`` (``n_above``)
      - ``coverage_var`` strictly exceeds ``coverage_threshold``
        (``n_covered``, included only when ``coverage_var`` is set)
      - both gates pass (``n_overlap``, included only when
        ``coverage_var`` is set)

    For MOD10C1: a low ``n_overlap`` count means the timestep is
    cloud-dominated or pre-Terra-systematic-data (early 2000), so the
    snow panel will look spatially sparse even though the underlying
    data is correct.

    The dataset's first dim is treated as the time dim; the second is
    treated as the HRU dim.
    """
    arr = ds[var].values
    time_dim = ds[var].dims[0]
    times = pd.DatetimeIndex(ds[time_dim].values)
    finite = np.isfinite(arr)
    above = finite & (arr > threshold)
    cols: dict[str, np.ndarray] = {
        "n_finite": finite.sum(axis=1),
        "n_above": above.sum(axis=1),
    }
    if coverage_var is not None:
        cov_arr = ds[coverage_var].values
        cov_pass = np.isfinite(cov_arr) & (cov_arr > coverage_threshold)
        cols["n_covered"] = cov_pass.sum(axis=1)
        cols["n_overlap"] = (above & cov_pass).sum(axis=1)
    return pd.DataFrame(cols, index=times).rename_axis("date")


def find_best_day(
    ds: xr.Dataset,
    var: str,
    *,
    threshold: float = 0.0,
    coverage_var: str | None = None,
    coverage_threshold: float = 0.7,
    month: int | None = None,
) -> pd.Timestamp:
    """Return the date with the largest count of HRUs that pass the
    primary threshold and (when set) the coverage gate.

    Companion to :func:`daily_coverage_summary`. When picking
    ``TARGET_DAY`` for a notebook, the timestep that maximizes
    ``n_overlap`` produces the spatially richest inspection panel —
    avoids the early-Terra/cloud-dominated days where most HRUs are NaN.

    ``month`` (1-12), if set, restricts selection to that calendar
    month. If no overlap gate is set, the metric falls back to
    ``n_above`` (count of HRUs above the primary threshold).
    """
    df = daily_coverage_summary(
        ds,
        var,
        threshold=threshold,
        coverage_var=coverage_var,
        coverage_threshold=coverage_threshold,
    )
    if month is not None:
        df = df[df.index.month == month]
    if df.empty:
        scope = f" in month {month}" if month is not None else ""
        raise ValueError(f"No timesteps found{scope}")
    metric = "n_overlap" if "n_overlap" in df.columns else "n_above"
    return df[metric].idxmax()


def save_figure(fig, name: str) -> None:
    """Write ``fig`` to ``FIGURES_DIR[/PROJECT]/<name>.png`` iff ``SAVE_FIGURES``.

    No-op when ``SAVE_FIGURES`` is ``False`` (the default). Notebooks
    enable saving by setting ``_helpers.SAVE_FIGURES = True`` near the
    top before any plotting cell runs.

    When ``PROJECT`` is set (notebooks should set
    ``_helpers.PROJECT = PROJECT_DIR.name`` so figures from different
    fabrics stay separate), figures land under
    ``FIGURES_DIR / PROJECT / <name>.png``. With ``PROJECT = None``
    figures land directly in ``FIGURES_DIR`` — fine for ad-hoc local
    work, but commits should always set ``PROJECT`` so the deck's
    figure paths resolve unambiguously.

    Relative paths in ``FIGURES_DIR`` are resolved against the repo
    root (this module's great-grandparent directory). Absolute paths
    (user overrides, pytest tmp_path) are honored as-is.
    """
    if not SAVE_FIGURES:
        return
    target_dir = FIGURES_DIR
    if not target_dir.is_absolute():
        # _helpers.py lives at <repo>/notebooks/aggregated/_helpers.py;
        # repo root is two parents up from that.
        target_dir = Path(__file__).resolve().parent.parent.parent / target_dir
    if PROJECT:
        target_dir = target_dir / PROJECT
    target_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_dir / f"{name}.png", dpi=150, bbox_inches="tight")
