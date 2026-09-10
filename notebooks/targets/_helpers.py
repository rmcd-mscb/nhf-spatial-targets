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

import math
import warnings
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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
    # Dispatch on suffix (mirrors validate._gather_fabric_meta): parquet
    # fabrics use geopandas-native read_parquet so this works even when GDAL
    # lacks the ogr_Parquet plugin; everything else goes through pyogrio.
    path = Path(fabric_cfg["path"])
    if path.suffix.lower() in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    gdf = gdf.set_index(fabric_cfg["id_col"])
    if simplify_tolerance_deg is not None and simplify_tolerance_deg > 0:
        gdf = gdf.copy()
        gdf.geometry = gdf.geometry.simplify(
            simplify_tolerance_deg, preserve_topology=True
        )
    return gdf


def load_representative_points(
    project_dir: Path, target: str
) -> dict[str, tuple[float, float]] | None:
    """Return per-project REPRESENTATIVE_POINTS for *target*, or ``None``.

    Reads ``<project_dir>/config.yml`` for a top-level
    ``representative_points:`` block keyed by the notebook's ``TARGET``
    (``aet``, ``recharge``, ``runoff``, ``soil_moisture``,
    ``snow_covered_area``, ``swe``). When absent (gfv2's case), returns
    ``None`` so the notebook can fall back to its hardcoded CONUS defaults.

    Schema::

        representative_points:
          aet:
            "Cascades (Mt. Hood)": [-121.7, 45.4]
            "Willamette Valley": [-123.0, 44.6]
          swe:
            ...

    Mirrors ``notebooks/aggregated/_helpers.py:load_representative_points``.
    """
    cfg_path = Path(project_dir) / "config.yml"
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    block = cfg.get("representative_points") or {}
    raw = block.get(target)
    if not raw:
        return None
    return {
        label: (float(coords[0]), float(coords[1])) for label, coords in raw.items()
    }


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


def open_target_nc(
    path: Path,
    *,
    time: slice | tuple[str | pd.Timestamp, str | pd.Timestamp] | None = None,
) -> xr.Dataset:
    """Open a target NC and detach from the file handle.

    Loads into memory and closes the underlying handle before return.
    Monthly targets are a few hundred MB (132 months × ~360k HRUs ×
    3-4 vars × float32/int8) and fit comfortably for an interactive
    notebook session, so the default ``time=None`` loads everything.

    The **daily** SWE target is the exception: ~11 GB per file on the
    gfv2 fabric (``time=16802 × nat_hru_id=361471`` for three vars), and
    ``.load()`` on the whole thing OOMs a default-mem kernel (issue
    #163). Pass ``time=`` to subset on-disk *before* materialising into
    memory:

        open_target_nc(path, time=("2009-10-01", "2010-09-30"))  # WY2010
        open_target_nc(path, time=slice("2009-10-01", "2010-09-30"))

    A 2-tuple is interpreted as the endpoints of a ``slice`` (inclusive,
    label-based, the same convention as ``ds.sel(time=slice(a, b))``).
    A window outside the file's time range clips (or returns an empty
    ``time`` dim) rather than raising. The netCDF4 build *currently*
    auto-chunks the bound vars at ~431 days/time-chunk (not pinned — see
    issue #163), so a one-water-year window reads roughly one to two
    time-chunks (order hundreds of MB compressed) rather than the full
    11 GB.
    """
    with xr.open_dataset(path) as ds:
        if time is not None:
            if isinstance(time, tuple):
                time = slice(*time)
            ds = ds.sel(time=time)
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


def _axis_labels(fabric_gdf: gpd.GeoDataFrame) -> tuple[str, str]:
    """Axis labels matching the fabric's CRS.

    A geographic CRS really is longitude/latitude; a projected one is
    easting/northing in that CRS's own linear unit (metres for the
    EPSG:5070 Albers the Oregon fabric uses). Hardcoding "Longitude"
    mislabels every projected fabric's figures.
    """
    crs = getattr(fabric_gdf, "crs", None)
    if crs is None or crs.is_geographic:
        return ("Longitude", "Latitude")
    unit = "m"
    try:
        unit_name = crs.axis_info[0].unit_name
        unit = {"metre": "m", "meter": "m", "US survey foot": "ft"}.get(
            unit_name, unit_name
        )
    except (AttributeError, IndexError):  # pragma: no cover - exotic CRS
        pass
    return (f"Easting ({unit})", f"Northing ({unit})")


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
    nan_label: str | None = None,
    legend: bool = True,
) -> None:
    """Render an HRU-level choropleth with NaN HRUs in ``nan_color``.

    Joins ``values`` (indexed by HRU id) onto ``fabric_gdf``. NaN HRUs
    are plotted first in ``nan_color`` so coverage gaps are visually
    obvious; finite-value HRUs are plotted on top.

    ``nan_label`` adds a legend entry naming the grey class and its HRU
    count. Pass it whenever the NaN has a *specific* meaning the reader
    cannot infer -- ``ensemble_std`` is masked wherever ``n_sources <
    2``, which is a deliberate modelling decision (a one-source
    population std is exactly 0 and would read as perfect agreement),
    not a coverage gap. Left ``None`` the map is unchanged.

    ``legend=False`` suppresses this panel's colorbar -- used by
    :func:`plot_member_panels`, where one shared colorbar serves every
    panel and per-panel bars would imply per-panel scales.

    Axis labels follow the fabric's CRS. The Oregon fabric is stored in
    EPSG:5070 Albers, whose coordinates are metres; labelling those
    "Longitude" (as this helper did before issue #351) misreports the
    units on every figure rendered from a projected fabric.
    """
    plot_gdf = fabric_gdf.copy()
    plot_gdf["value"] = values.reindex(plot_gdf.index)

    nan_mask = plot_gdf["value"].isna()
    if nan_mask.any():
        plot_gdf[nan_mask].plot(ax=ax, color=nan_color, edgecolor="none")
        if nan_label:
            ax.legend(
                handles=[
                    Patch(
                        facecolor=nan_color,
                        label=f"{nan_label} (n={int(nan_mask.sum())})",
                    )
                ],
                loc="lower left",
                fontsize=8,
            )

    plot_gdf[~nan_mask].plot(
        ax=ax,
        column="value",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        legend=legend,
        legend_kwds={"label": units, "shrink": 0.6} if legend else None,
        edgecolor="none",
    )
    ax.set_title(title, fontsize=11)
    xlabel, ylabel = _axis_labels(fabric_gdf)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Equal aspect keeps the fabric undistorted: 1 unit east == 1 unit
    # north, whether those units are degrees or projected metres.
    ax.set_aspect("equal")


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
    xlabel, ylabel = _axis_labels(fabric_gdf)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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
    xlabel, ylabel = _axis_labels(fabric_gdf)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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


# --------------------------------------------------------------------------
# Ensemble members (issue #351)
#
# Targets built with ``targets.<t>.emit_members: true`` (issue #338) carry
# one variable per contributing source key alongside the bounds, plus the
# derived ``ensemble_mean`` / ``ensemble_std``. The helpers below read that
# schema and render it. ``snow_covered_area`` deliberately emits no members
# -- its bounds are a MOD10C1 confidence interval, not a member min/max --
# so ``member_keys`` returns ``[]`` there and the notebook skips cleanly.
# --------------------------------------------------------------------------

#: Categorical hues for member identity, in fixed assignment order.
#:
#: These are the light-mode steps of the standard eight-hue categorical
#: theme. The order is deliberate, not arbitrary: the first four validate
#: on the *all-pairs* colorblind test (worst pair CVD dE 9.2, normal-vision
#: 16.3 in OKLab x100), which is the test that applies to a choropleth or a
#: spaghetti plot where every series is on screen simultaneously -- unlike
#: a stacked bar, where only adjacent pairs need to separate.
#:
#: Light-mode only, by design. These helpers render matplotlib PNGs onto a
#: white figure ground for a Marp ``theme: default`` deck; there is no dark
#: surface in play. The dark steps of this same theme do NOT survive the
#: all-pairs test at four slots (violet collides with blue), so do not
#: assume this list is safe to reuse on a dark background.
MEMBER_PALETTE: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#4a3aa7",  # violet
    "#e87ba4",  # magenta
    "#008300",  # green
    "#eda100",  # yellow
    "#e34948",  # red
)

#: Member count beyond which all-pairs colorblind separation is no longer
#: guaranteed by :data:`MEMBER_PALETTE`.
ALL_PAIRS_SAFE_MEMBERS: int = 4

#: :func:`member_argextreme` code for "every finite member agrees".
NO_SPREAD_CODE: int = -1

#: :func:`member_argextreme` code for "exactly one member is finite here".
SINGLE_SOURCE_CODE: int = -2


def member_keys(ds: xr.Dataset) -> list[str]:
    """Return the ensemble member source keys carried by ``ds``.

    Reads the ``member_keys`` global attr stamped by
    ``targets/_writers.write_bounds_target`` and cross-checks each name
    against the variables actually present, so a truncated or
    hand-edited file cannot promise a member it does not hold.

    Returns ``[]`` when the dataset declares ``members_emitted`` other
    than ``"true"`` (the ``snow_covered_area`` case, and every
    ``_nn_filled`` companion) or carries neither attr (a pre-#338
    target). Callers should treat ``[]`` as "skip the member figures",
    not as an error.

    Deliberately attr-driven rather than inferred by subtracting a
    denylist of known non-member variables from ``ds.data_vars``: the
    attr is the machine-readable contract issue #338 introduced, and a
    denylist silently misclassifies the next derived variable added to
    the target schema as a member.
    """
    if ds.attrs.get("members_emitted") != "true":
        return []
    raw = ds.attrs.get("member_keys", "")
    if not raw:
        return []
    declared = [key.strip() for key in raw.split(",") if key.strip()]
    present = [key for key in declared if key in ds.data_vars]
    missing = [key for key in declared if key not in ds.data_vars]
    if missing:
        warnings.warn(
            f"member_keys declares {missing} but those variables are not in "
            "the dataset; dropping them. The file may be truncated or the "
            "attr hand-edited.",
            stacklevel=2,
        )
    return present


def member_colors(keys: Sequence[str]) -> dict[str, str]:
    """Map member source keys to categorical hues, in order.

    Assignment is positional against :data:`MEMBER_PALETTE` and never
    cycles, so within one target every figure paints a given source the
    same color -- the member map, the driver map and the spaghetti all
    agree, which is what lets them be read as one set.

    The mapping is per-target, not global: ``era5_land`` is a runoff
    member and an SWE member and may take a different slot in each,
    because the twelve source keys in the catalog cannot all be given
    an all-pairs-separable hue. Every figure therefore carries its own
    legend, and no figure invites a cross-target color inference.

    Warns past :data:`ALL_PAIRS_SAFE_MEMBERS` (all-pairs separation is
    no longer guaranteed -- facet instead) and raises past the eight
    hues the theme defines rather than generating or recycling one.
    """
    keys = list(keys)
    if len(keys) > len(MEMBER_PALETTE):
        raise ValueError(
            f"member_colors: {len(keys)} member keys exceeds the "
            f"{len(MEMBER_PALETTE)} validated categorical hues. Facet the "
            "figure or fold the tail into an 'other' group rather than "
            "generating a new hue."
        )
    if len(keys) > ALL_PAIRS_SAFE_MEMBERS:
        warnings.warn(
            f"member_colors: {len(keys)} members exceeds "
            f"{ALL_PAIRS_SAFE_MEMBERS}, past which the palette is not "
            "all-pairs colorblind-separable. Every series is on screen at "
            "once in these figures, so prefer faceting.",
            stacklevel=2,
        )
    return {key: MEMBER_PALETTE[i] for i, key in enumerate(keys)}


def member_frame_at_time(
    ds: xr.Dataset,
    keys: Sequence[str],
    time_sel,
    id_dim: str,
) -> pd.DataFrame:
    """Members at one timestep as an ``(hru x member)`` DataFrame.

    ``time_sel`` is anything ``.sel(time=...)`` accepts. A selection
    that still carries a time dimension -- a partial string like
    ``"2005"``, or a label that matches more than one step -- collapses
    to its first step, the same tolerance :func:`select_month` provides.
    That lets the annual, monthly and daily notebooks share one idiom
    instead of each spelling out its own cadence.
    """

    def _one(key: str) -> pd.Series:
        da = ds[key].sel(time=time_sel)
        if "time" in da.dims:
            da = da.isel(time=0)
        return da.to_pandas().reindex(ds[id_dim].values)

    return pd.DataFrame(
        {key: _one(key) for key in keys},
        index=pd.Index(ds[id_dim].values, name=id_dim),
    )


def member_frame_at_hru(
    ds: xr.Dataset,
    keys: Sequence[str],
    hru_id,
    id_dim: str,
) -> pd.DataFrame:
    """Members at one HRU as a ``(time x member)`` DataFrame."""
    columns = {key: ds[key].sel({id_dim: hru_id}).to_pandas() for key in keys}
    frame = pd.DataFrame(columns)
    frame.index = pd.DatetimeIndex(ds["time"].values)
    frame.index.name = "time"
    return frame


def member_argextreme(frame: pd.DataFrame, *, how: str = "max") -> pd.Series:
    """Which member sets the bound at each row of ``frame``.

    Returns a ``pd.Series`` of float codes aligned to ``frame.index``:

    * ``0 .. n-1``   -- positional index into ``frame.columns``
    * :data:`NO_SPREAD_CODE`      -- two or more finite members, all equal
    * :data:`SINGLE_SOURCE_CODE`  -- exactly one finite member
    * ``NaN``        -- no finite member

    The two sentinels exist because a bare ``argmax`` is actively
    misleading on this data. On a summer SWE day every source reads
    0.0 mm; ``np.nanargmax`` breaks that tie by returning column 0, so
    a naive driver map paints the entire state as ``snodas``-driven
    when in truth nothing drives anything and all four sources agree.
    Separating "they agree" from "one of them won" is the whole point
    of the map. The single-source code is split out for the same
    honesty reason: with one finite member there is no comparison to
    win, and that cell's story is coverage (see the ``n_sources`` map),
    not disagreement.
    """
    if how not in ("max", "min"):
        raise ValueError(f"member_argextreme: how must be 'max' or 'min', got {how!r}")

    values = frame.to_numpy(dtype="float64")
    n_finite = np.isfinite(values).sum(axis=1)
    codes = np.full(values.shape[0], np.nan, dtype="float64")

    multi = n_finite >= 2
    if multi.any():
        sub = values[multi]
        spread = np.nanmax(sub, axis=1) - np.nanmin(sub, axis=1)
        picker = np.nanargmax if how == "max" else np.nanargmin
        codes[multi] = np.where(spread == 0, NO_SPREAD_CODE, picker(sub, axis=1))

    codes[n_finite == 1] = SINGLE_SOURCE_CODE
    return pd.Series(codes, index=frame.index, name=f"arg{how}_member")


def member_categories(
    keys: Sequence[str], colors: dict[str, str] | None = None
) -> dict[int, tuple[str, str]]:
    """Category map for :func:`plot_categorical_choropleth` driver maps.

    Pairs each member's positional code with its hue, then appends the
    two neutral sentinel classes. The sentinels are greys on purpose:
    "the sources agree" and "only one source is here" are not members,
    and giving them a categorical hue would read as a fifth source.
    """
    colors = colors or member_colors(keys)
    categories = {i: (key, colors[key]) for i, key in enumerate(keys)}
    categories[NO_SPREAD_CODE] = ("no spread (sources agree)", "#b8b8b3")
    categories[SINGLE_SOURCE_CODE] = ("single source", "#6e6e69")
    return categories


def plot_member_panels(
    fabric_gdf: gpd.GeoDataFrame,
    panels: dict[str, pd.Series],
    *,
    units: str = "",
    cmap: str = "YlGnBu",
    ncols: int = 3,
    vmin: float | None = None,
    vmax: float | None = None,
    colors: dict[str, str] | None = None,
    suptitle: str = "",
    nan_color: str = "lightgrey",
) -> tuple[object, tuple[float, float]]:
    """Small-multiples choropleth, one panel per member, one color scale.

    ``panels`` maps a label (a member source key, or a derived name like
    ``"ensemble_mean"``) to a per-HRU Series. Returns
    ``(fig, (vmin, vmax))`` so the caller can report -- and a test can
    assert on -- the scale that was actually used.

    **The shared scale is the point.** Per-panel autoscaling is the
    default in most small-multiples code and it is wrong here: it
    renormalises each source to its own range, so four sources that
    disagree by a factor of three render as four near-identical maps.
    Unless ``vmin`` / ``vmax`` are given they are pooled across every
    panel (2nd/98th percentile of all finite values together), so panel
    brightness is comparable across members and a systematically wet or
    dry source is visible at a glance.

    ``colors`` (from :func:`member_colors`) tints each panel's frame,
    tying a panel to the same member's line in the spaghetti plot and
    its class in the driver map. The tint is on the spines -- a mark --
    never on the title text, which stays in default ink so identity is
    carried by the label itself and not by color alone.
    """
    import matplotlib.pyplot as plt

    if not panels:
        raise ValueError("plot_member_panels: no panels to draw")

    if vmin is None or vmax is None:
        pooled = np.concatenate(
            [s.to_numpy(dtype="float64").ravel() for s in panels.values()]
        )
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size:
            vmin = float(np.percentile(pooled, 2)) if vmin is None else vmin
            vmax = float(np.percentile(pooled, 98)) if vmax is None else vmax
        else:
            vmin, vmax = (0.0, 1.0)
    if vmin == vmax:  # a constant field would otherwise render blank
        vmax = vmin + 1.0

    ncols = max(1, min(ncols, len(panels)))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5 * ncols, 5.5 * nrows),
        squeeze=False,
        layout="constrained",
    )
    drawn = list(axes.flat)

    for ax, (label, series) in zip(drawn, panels.items()):
        plot_hru_choropleth(
            ax,
            fabric_gdf,
            series,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            title=label,
            units=units,
            nan_color=nan_color,
            legend=False,  # one shared bar below, not one per panel
        )
        if colors and label in colors:
            for spine in ax.spines.values():
                spine.set_edgecolor(colors[label])
                spine.set_linewidth(2.0)

    for ax in drawn[len(panels) :]:
        ax.remove()  # remove, not hide: a hidden axes still reserves grid space

    live = [ax for ax in fig.axes]
    mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    fig.colorbar(mappable, ax=live, shrink=0.7, label=units)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    return fig, (float(vmin), float(vmax))


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
