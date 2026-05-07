"""Shared helpers for the inspect_target_*.ipynb notebooks.

Sibling of the notebooks (not packaged into ``nhf_spatial_targets``).
Mirrors ``notebooks/aggregated/_helpers.py``: path discovery, fabric I/O,
HRU choropleth plotting, area-weighted means, representative-point
lookup, and a ``save_figure`` helper that populates
``docs/figures/targets/[<project>/]`` for downstream slide /
documentation work.

Targets are post-combination artefacts (``<project>/targets/<target>_targets.nc``
and the optional ``<target>_targets_nn_filled.nc`` companion). The schema
is established by ``targets/_common.write_target_nc``: ``lower_bound``,
``upper_bound``, ``n_sources``, optional ``nn_filled``, ``time_bnds``,
``centroid_lat`` / ``centroid_lon``, an HRU dim named after
``fabric.id_col``.

Notebooks import via:

    from _helpers import load_project_paths, open_target_nc, ...
"""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from matplotlib.patches import Patch
from shapely.geometry import Point

SAVE_FIGURES: bool = False
FIGURES_DIR: Path = Path("docs/figures/targets/")
PROJECT: str | None = None
ALBERS_CRS: str = "EPSG:5070"  # matches the aggregator's WEIGHT_GEN_CRS


DEFAULT_CALDERA_PROJECT = Path(
    "/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets"
)


_AREA_CACHE: dict[int, pd.Series] = {}


def _fabric_area(fabric_gdf: gpd.GeoDataFrame) -> pd.Series:
    """Return the per-HRU EPSG:5070 area for ``fabric_gdf``, cached.

    Keyed on ``id(fabric_gdf)`` — the cache assumes the GeoDataFrame is
    not mutated in place after first use, which matches the typical
    notebook pattern (load once, treat as read-only). On a 360k-polygon
    CONUS fabric the EPSG:5070 reprojection costs ~5–10 s; caching pays
    for itself the second time the notebook reduces a per-time bound to
    a CONUS-mean series.
    """
    key = id(fabric_gdf)
    if key not in _AREA_CACHE:
        _AREA_CACHE[key] = fabric_gdf.to_crs(ALBERS_CRS).area
    return _AREA_CACHE[key]


def load_project_paths(
    project_dir: Path | None = None,
) -> tuple[Path, Path, dict]:
    """Read ``<project>/config.yml`` and return ``(project_dir, datastore_dir, fabric_cfg)``.

    ``fabric_cfg`` is the ``fabric`` sub-block from ``config.yml`` (keys
    typically include ``path``, ``id_col``, ``crs``, ``buffer_deg``).
    Defaults to the caldera ``gfv2-spatial-targets`` project when called
    with ``None``.
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


def load_fabric(
    fabric_cfg: dict,
    *,
    simplify_tolerance_deg: float | None = 0.005,
) -> gpd.GeoDataFrame:
    """Read the HRU fabric file and index by ``fabric_cfg['id_col']``.

    Kept in EPSG:4326 for plotting; downstream area calculations
    re-project to EPSG:5070 (CONUS Albers) on demand. Area is always
    computed from the *original* (un-simplified) geometry — see
    ``area_weighted_mean`` and ``area_weighted_series`` which read
    geometry off the input GeoDataFrame.

    ``simplify_tolerance_deg`` controls a Douglas-Peucker simplification
    on the EPSG:4326 geometry before plotting. The default ~0.005° (≈
    500 m at mid-latitudes) keeps CONUS-scale choropleths visually
    indistinguishable from the unsimplified version while cutting
    matplotlib render time on a ~360k-polygon fabric by 5-10×. Set to
    ``None`` to disable. **Important**: simplification changes polygon
    *vertices*, not areas, by a fraction of a percent — fine for
    plotting, but if you reuse this GeoDataFrame for area-weighted
    aggregation, prefer reloading without simplification or accept the
    rounding error (sub-1% per polygon).
    """
    gdf = gpd.read_file(fabric_cfg["path"])
    gdf = gdf.set_index(fabric_cfg["id_col"])
    if simplify_tolerance_deg is not None and simplify_tolerance_deg > 0:
        gdf = gdf.copy()
        gdf.geometry = gdf.geometry.simplify(
            simplify_tolerance_deg, preserve_topology=True
        )
    return gdf


def discover_target_nc(
    project_dir: Path, target: str
) -> tuple[Path | None, Path | None]:
    """Find ``<target>_targets.nc`` and the NN-filled companion if present.

    Returns ``(unfilled_path_or_None, filled_path_or_None)``. Either
    can be ``None`` — callers print a clear "skip" line and continue.
    Mirrors the writer convention in
    ``targets/_common.write_target_nc`` plus ``targets/run.py`` which
    emits both files when ``nn_fill: true`` is set in config.
    """
    targets_dir = Path(project_dir) / "targets"
    raw = targets_dir / f"{target}_targets.nc"
    filled = targets_dir / f"{target}_targets_nn_filled.nc"
    return (raw if raw.exists() else None, filled if filled.exists() else None)


def open_target_nc(path: Path) -> xr.Dataset:
    """Open a target NC and detach from the file handle.

    Loads into memory and closes the underlying handle before return.
    Targets are typically a few hundred MB (132 months × ~360k HRUs ×
    3-4 vars × float32/int8) — fits comfortably for an interactive
    notebook session.
    """
    with xr.open_dataset(path) as ds:
        return ds.load()


def area_weighted_mean(values: pd.Series, fabric_gdf: gpd.GeoDataFrame) -> float:
    """Compute Σ(v · A) / Σ(A) using fabric area in EPSG:5070.

    Skips NaN values (and their corresponding areas). Aligns on the
    fabric's index — ``values`` must be indexed by HRU id. Area is
    computed via :func:`_fabric_area` (cached per GeoDataFrame).

    Note: when ``fabric_gdf`` was loaded with simplification (see
    :func:`load_fabric`), polygon areas carry sub-1% bias relative to
    the original geometry. Acceptable for the order-of-magnitude
    inspection use here; reload with ``simplify_tolerance_deg=None`` if
    exact area conservation matters.
    """
    aligned = values.reindex(fabric_gdf.index)
    areas = _fabric_area(fabric_gdf)
    mask = ~aligned.isna()
    if not mask.any():
        return float("nan")
    return float((aligned[mask] * areas[mask]).sum() / areas[mask].sum())


def area_weighted_series(
    da: xr.DataArray, fabric_gdf: gpd.GeoDataFrame, id_dim: str
) -> pd.Series:
    """Per-timestep area-weighted CONUS mean of a (time, hru) DataArray.

    Returns a ``pd.Series`` indexed by time. Aligns the HRU dim against
    the fabric's index (so HRUs in the source not present in the fabric
    are dropped, and vice versa). Area is computed via
    :func:`_fabric_area` (cached per GeoDataFrame), so successive calls
    against the same fabric do not re-pay the EPSG:5070 reprojection.

    Note: when ``fabric_gdf`` was loaded with simplification (see
    :func:`load_fabric`), polygon areas carry sub-1% bias relative to
    the original geometry. Acceptable for the order-of-magnitude CONUS
    series used in the inspection notebooks.
    """
    times = pd.DatetimeIndex(da.time.values)
    arr = da.transpose("time", id_dim).values
    areas = _fabric_area(fabric_gdf)
    # Align array columns to fabric ID order.
    src_ids = pd.Index(da[id_dim].values)
    fab_ids = fabric_gdf.index
    common = fab_ids.intersection(src_ids)
    src_pos = src_ids.get_indexer(common)
    fab_pos = fab_ids.get_indexer(common)
    aligned_areas = areas.iloc[fab_pos].values  # shape (len(common),)
    aligned_arr = arr[:, src_pos]  # shape (T, len(common))
    finite = np.isfinite(aligned_arr)
    weighted = np.where(finite, aligned_arr * aligned_areas, 0.0)
    weight_sum = np.where(finite, aligned_areas, 0.0).sum(axis=1)
    value_sum = weighted.sum(axis=1)
    # ``np.where`` evaluates both branches, so divide-by-zero warnings
    # surface for all-NaN timesteps even though they end up as NaN. Silence
    # just the divide; the mask logic still produces the right answer.
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(weight_sum > 0, value_sum / weight_sum, np.nan)
    return pd.Series(out, index=times)


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


def select_month(da: xr.DataArray, year: int, month: int) -> xr.DataArray:
    """Select the first timestep in the given calendar month.

    Slices ``da`` to the calendar-month window ``[YYYY-MM-01, YYYY-MM-end]``
    and returns the first hit. Robust to start-of-month / end-of-month /
    mid-month timestamping conventions.

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
    ax.set_aspect("equal")  # 1° lon = 1° lat — prevents east-west stretching


def plot_categorical_choropleth(
    ax,
    fabric_gdf: gpd.GeoDataFrame,
    values: pd.Series,
    *,
    categories: dict[int, tuple[str, str]],
    title: str = "",
    nan_color: str = "lightgrey",
) -> None:
    """Render a per-category coloured map (e.g. n_sources flag values).

    ``categories`` maps integer flag value -> (label, color). Useful for
    discrete fields like ``n_sources`` (0 / 1 / 2 / 3) or ``nn_filled``
    (0 / 1) where a continuous colormap would be misleading.
    """
    plot_gdf = fabric_gdf.copy()
    plot_gdf["value"] = values.reindex(plot_gdf.index)

    nan_mask = plot_gdf["value"].isna()
    handles: list[Patch] = []
    if nan_mask.any():
        plot_gdf[nan_mask].plot(ax=ax, color=nan_color, edgecolor="none")
        handles.append(
            Patch(facecolor=nan_color, label=f"no data (n={int(nan_mask.sum())})")
        )

    for flag, (label, color) in categories.items():
        sub = plot_gdf[(~nan_mask) & (plot_gdf["value"] == flag)]
        if len(sub) == 0:
            continue
        sub.plot(ax=ax, color=color, edgecolor="none")
        handles.append(Patch(facecolor=color, label=f"{label} (n={len(sub)})"))
    if handles:
        ax.legend(handles=handles, loc="lower left", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")


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
    ax.set_aspect("equal")


def n_sources_per_time(ds: xr.Dataset) -> pd.DataFrame:
    """Per-timestep histogram of ``n_sources`` flag values.

    Returns a DataFrame indexed by time with one column per flag value
    (0 / 1 / 2 / 3 for a 3-source target). Diagnostic for spotting
    months where coverage drops (e.g. when one source's period ends and
    only the survivors contribute).
    """
    arr = ds["n_sources"].values  # (time, hru), int8
    time = pd.DatetimeIndex(ds["time"].values)
    max_n = int(arr.max())
    cols = {}
    for k in range(max_n + 1):
        cols[f"n={k}"] = (arr == k).sum(axis=1)
    return pd.DataFrame(cols, index=time).rename_axis("date")


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
    root, three parents up from this file
    (``targets/_helpers.py`` -> ``notebooks/`` -> ``<repo>``).
    Absolute paths (user overrides, pytest tmp_path) are honored as-is.
    """
    if not SAVE_FIGURES:
        return
    if not PROJECT:
        warnings.warn(
            "save_figure: SAVE_FIGURES is True but PROJECT is unset. "
            "Figures will land directly in FIGURES_DIR with no project subdir, "
            "risking collision with other fabrics' figures. "
            "Set _helpers.PROJECT = PROJECT_DIR.name to namespace by project.",
            stacklevel=2,
        )
    target_dir = FIGURES_DIR
    if not target_dir.is_absolute():
        # Resolve relative paths against the repo root, three parents up:
        # <repo>/notebooks/targets/_helpers.py -> <repo>.
        target_dir = Path(__file__).resolve().parent.parent.parent / target_dir
    if PROJECT:
        target_dir = target_dir / PROJECT
    target_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_dir / f"{name}.png", dpi=150, bbox_inches="tight")
