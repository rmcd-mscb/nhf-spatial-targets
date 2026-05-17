"""Shared multi-source-minmax target machinery.

Used by ``targets/run.py`` (and, in future, ``targets/aet.py``) to:

- read per-year aggregated source NCs as a single lazy xarray DataArray;
- canonicalize cross-source monthly time coordinates onto a master
  month-start index;
- combine sources with NaN-aware reductions and emit a per-cell finite-
  source-count diagnostic;
- compute HRU area and centroids in both equal-area and lat/lon CRSes;
- write CF-1.6 compliant NetCDF output atomically.

Linear unit conversions live in the per-target module (e.g. ``run.py``
keeps ``mm_per_month_to_cfs``); this module is unit-agnostic.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceShim:
    """Per-source contract for a multi-source target builder.

    Co-locates the three facts each target needs about a contributing
    source: the catalog key it loads from, the variable name in the
    aggregated NC, the human-readable label for the output NC's global
    ``source`` attr, and the per-source unit-shim function that brings the
    native variable into the target's common intermediate units (e.g.
    mm/month for runoff and AET).

    Target modules declare a tuple of :class:`SourceShim` instances as
    their ``SHIMS`` constant and call :func:`shims_by_key` to look one up
    by catalog key inside their build loop. Adding a future source is a
    single edit — there is no parallel-dict drift surface.

    Attributes
    ----------
    source_key
        Catalog key (e.g. ``"ssebop"``). Must match ``catalog.source(...)``
        and the directory name under ``<project>/data/aggregated/``.
    aggregated_var
        Variable name to extract from the aggregated NC (e.g. ``"et"``).
    description
        Human-readable label for the output NC's global ``source`` attr.
    to_common_units
        Callable that converts the source's native variable to the
        target's common intermediate units. For multi_source_minmax
        targets the common intermediate is usually mm/month; the per-
        target unit chain then applies a final linear conversion (e.g.
        mm/month → cfs for runoff, mm/month → inches/day for AET).
    """

    source_key: str
    aggregated_var: str
    description: str
    to_common_units: Callable[[xr.DataArray], xr.DataArray]


def shims_by_key(shims: "tuple[SourceShim, ...]") -> "dict[str, SourceShim]":
    """Index a ``SHIMS`` tuple by ``source_key`` for lookup at build time.

    Raises ``ValueError`` if two shims share the same ``source_key`` —
    that would silently shadow one entry, which is exactly the kind of
    drift this refactor prevents.
    """
    out: dict[str, SourceShim] = {}
    for shim in shims:
        if shim.source_key in out:
            raise ValueError(
                f"Duplicate SourceShim.source_key={shim.source_key!r} in "
                f"target SHIMS registry. Each source may appear at most once."
            )
        out[shim.source_key] = shim
    return out


def read_aggregated_source(
    project: Project,
    source_key: str,
    var: str,
    period: tuple[str, str],
    chunks: dict | None = None,
) -> xr.DataArray:
    """Open per-year aggregated NCs for one source and return one variable.

    Reads ``<project.aggregated_dir()>/<source_key>/<source_key>_*_agg.nc``
    via ``xr.open_mfdataset`` (lazy / dask-backed), slices to the requested
    period, and returns the requested variable as a DataArray.

    The returned DataArray is lazy: it holds open file handles via the dask
    graph. The caller is responsible for consuming it (e.g. via
    ``.compute()``, ``.load()``, or a streaming ``to_netcdf``) before the
    underlying dataset goes out of scope. Do **not** call ``.close()`` on
    the returned DataArray directly — it does not own the dataset handle.

    The HRU dim name in the aggregated NCs matches ``project.id_col`` (e.g.
    ``nhm_id``).

    Parameters
    ----------
    project
        Loaded :class:`~nhf_spatial_targets.workspace.Project`.
    source_key
        Catalog key (e.g. ``"era5_land"``).
    var
        Variable name to extract from the aggregated dataset.
    period
        ``(start_iso, end_iso)`` tuple, both inclusive (e.g.
        ``("2000-01-01", "2010-12-31")``).
    chunks
        Forwarded to ``xr.open_mfdataset``. Defaults to
        ``{"time": 12, project.id_col: -1}`` (one calendar year per chunk,
        all HRUs in one chunk).

    Raises
    ------
    FileNotFoundError
        If the source's aggregated directory contains no per-year NCs.
    ValueError
        If the requested period falls entirely outside the source's
        per-year coverage.
    """
    agg_dir = project.aggregated_dir() / source_key
    pattern = f"{source_key}_*_agg.nc"
    paths = sorted(agg_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No aggregated NC files found for source '{source_key}' under "
            f"{agg_dir} (pattern: {pattern}). Run "
            f"'pixi run nhf-targets agg {source_key.replace('_', '-')} "
            f"--project-dir {project.workdir}' first."
        )

    if chunks is None:
        chunks = {"time": 12, project.id_col: -1}

    ds = xr.open_mfdataset(
        [str(p) for p in paths],
        combine="by_coords",
        join="outer",
        chunks=chunks,
        engine="netcdf4",
    )
    if var not in ds:
        available = sorted(ds.data_vars)
        ds.close()
        raise KeyError(
            f"Variable '{var}' not found in aggregated NCs for source "
            f"'{source_key}'. Available variables: {available}."
        )
    # Canonical row order: HRU dim ascending by id_col. Emission-time
    # enforcement landed with issue #93, so per-year NCs aggregated after
    # that change arrive here already sorted (this call is a no-op for
    # them). For pre-#93 NCs already on disk — where gdptools wrote rows
    # in VPU-grouped batch order — this defensive sort keeps positional
    # checks against the fabric correct without forcing a re-aggregate.
    ds = ds.sortby(project.id_col)
    sliced = ds[var].sel(time=slice(period[0], period[1]))
    if sliced.sizes.get("time", 0) == 0:
        # Years parsed from sorted filenames -- avoids triggering dask compute
        # on a possibly large coordinate array, and is robust to time-coord
        # corruption inside the NCs.
        first_year = paths[0].name.rsplit("_", 2)[-2]
        last_year = paths[-1].name.rsplit("_", 2)[-2]
        ds.close()
        raise ValueError(
            f"Requested period {period[0]} .. {period[1]} is entirely "
            f"outside source coverage for '{source_key}' "
            f"(years {first_year} .. {last_year})."
        )
    logger.info(
        "Loaded %s/%s: %d months from %d per-year NCs",
        source_key,
        var,
        sliced.sizes["time"],
        len(paths),
    )
    return sliced


def reindex_to_month_start(
    da: xr.DataArray, master_index: pd.DatetimeIndex
) -> xr.DataArray:
    """Reindex a monthly DataArray onto a master ``freq="MS"`` index.

    Source timestamps may be end-of-month (ERA5-Land), start-of-month (GLDAS,
    MWBM), or mid-month (MERRA-2 etc.). All three convey "which calendar
    month" unambiguously. This helper converts the source's time coordinate
    via ``dt.to_period("M").dt.to_timestamp()`` (yielding the month-start),
    then reindexes onto ``master_index``.

    Months in ``master_index`` that the source does not cover come back as
    NaN — this is what gives the runoff target its period-union semantics:
    a source that ends in 2020 but is asked through 2024 simply contributes
    nothing for the post-2020 cells.

    Parameters
    ----------
    da
        Monthly DataArray to reindex.
    master_index
        Target index. Must be ``DatetimeIndex`` with ``freq="MS"``.
    """
    if not isinstance(master_index, pd.DatetimeIndex):
        raise TypeError(
            f"master_index must be a pandas.DatetimeIndex, got "
            f"{type(master_index).__name__}"
        )
    if master_index.freqstr != "MS":
        raise ValueError(
            f"master_index must have freq='MS' (month-start); got "
            f"freq={master_index.freqstr!r}. Build it with "
            f"pd.date_range(start, end, freq='MS')."
        )
    ms_times = pd.DatetimeIndex(da.time.values).to_period("M").to_timestamp()
    canon = da.assign_coords(time=ms_times)
    return canon.reindex(time=master_index)


def multi_source_nanminmax(
    sources: dict[str, xr.DataArray],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """NaN-aware per-cell min, max, and finite-source count.

    All input DataArrays must share dims and coords (typically
    ``(time, id_col)``). They are stacked on a new ``source`` dim and
    reduced with ``skipna=True``.

    A bound is defined whenever ≥1 source is finite at that cell; the result
    is NaN only when *every* source is NaN there, which is exactly when
    ``n_sources == 0``.

    Parameters
    ----------
    sources
        Mapping from source key to per-source DataArray.

    Returns
    -------
    lower, upper, n_sources
        ``(time, id_col)`` arrays. ``n_sources`` is int8 with values in
        ``[0, len(sources)]``; ``lower`` / ``upper`` preserve the input
        dtype (typically float32).

    Raises
    ------
    ValueError
        If any two sources have different HRU coords (different fabrics).
    """
    keys = list(sources.keys())
    if not keys:
        raise ValueError("multi_source_nanminmax: empty sources dict")
    ref = sources[keys[0]]
    hru_dim = next(d for d in ref.dims if d != "time")
    for k in keys[1:]:
        other = sources[k]
        if not other[hru_dim].equals(ref[hru_dim]):
            raise ValueError(
                f"HRU coords differ between sources '{keys[0]}' and '{k}'. "
                "All sources must be aggregated to the same fabric."
            )

    stacked = xr.concat([sources[k] for k in keys], dim=xr.Variable("source", keys))
    lower = stacked.min(dim="source", skipna=True)
    upper = stacked.max(dim="source", skipna=True)
    n_sources = stacked.notnull().sum(dim="source").astype(np.int8)
    return lower, upper, n_sources


def _load_and_reproject_fabric(project: Project) -> "tuple[object, str]":
    """Load the fabric, validate the id_col, reproject to ``area_crs``.

    Shared expensive setup for :func:`compute_hru_areas`,
    :func:`compute_hru_centroids`, and the combined
    :func:`compute_hru_area_and_centroids`. The fabric IO and the CRS
    reprojection are the two non-trivial costs at fabric scale (~361k
    polygons on gfv2); per-column derivations (area, centroids) are
    near-free relative to those.

    Returns
    -------
    (gdf_eq, id_col)
        Reprojected GeoDataFrame in ``project.area_crs`` and the validated
        HRU id column name.
    """
    import geopandas as gpd

    fabric_path = Path(project.config["fabric"]["path"])
    if fabric_path.suffix.lower() in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(fabric_path)
    else:
        gdf = gpd.read_file(fabric_path)
    id_col = project.id_col
    if id_col not in gdf.columns:
        raise ValueError(
            f"Column '{id_col}' not found in fabric {fabric_path}. "
            f"Available: {list(gdf.columns)}"
        )
    if not gdf[id_col].is_unique:
        n_dupes = (gdf[id_col].value_counts() > 1).sum()
        raise ValueError(
            f"Fabric column '{id_col}' has {n_dupes} duplicate values "
            f"in {fabric_path}. Each HRU must have a unique ID."
        )

    gdf_eq = gdf.to_crs(project.area_crs)
    return gdf_eq, id_col


def compute_hru_centroids(project: Project) -> "pd.DataFrame":
    """Compute per-HRU centroid coords from the fabric.

    Reprojects to ``project.area_crs`` (e.g. EPSG:5070 for CONUS) for
    equal-area centroids, then reprojects centroids to EPSG:4326 for
    ancillary lat/lon. Use this in target builders that do **not** need
    per-HRU area (AET, recharge, soil moisture, SCA, SWE) — it skips the
    `gdf.geometry.area` call that :func:`compute_hru_area_and_centroids`
    performs.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``project.id_col``. Columns: ``centroid_x``,
        ``centroid_y`` (in ``area_crs``), ``centroid_lat``,
        ``centroid_lon`` (EPSG:4326).
    """
    gdf_eq, id_col = _load_and_reproject_fabric(project)
    centroids_eq = gdf_eq.geometry.centroid
    centroids_ll = centroids_eq.to_crs("EPSG:4326")

    df = gdf_eq[[id_col]].copy()
    df["centroid_x"] = centroids_eq.x.astype(float)
    df["centroid_y"] = centroids_eq.y.astype(float)
    df["centroid_lon"] = centroids_ll.x.astype(float)
    df["centroid_lat"] = centroids_ll.y.astype(float)
    df = df.set_index(id_col).sort_index()
    return df


def compute_hru_areas(project: Project) -> "pd.DataFrame":
    """Compute per-HRU area (m²) from the fabric.

    Reprojects to ``project.area_crs`` (e.g. EPSG:5070 for CONUS) so the
    area is computed in an equal-area projection. Always recomputes from
    geometry (no fabric-column fallback) so the area cannot drift from
    the geometry actually being processed.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``project.id_col``. Single column ``area_m2``.
    """
    gdf_eq, id_col = _load_and_reproject_fabric(project)
    df = gdf_eq[[id_col]].copy()
    df["area_m2"] = gdf_eq.geometry.area.astype(float)
    df = df.set_index(id_col).sort_index()
    return df


def compute_hru_area_and_centroids(project: Project) -> "pd.DataFrame":
    """Compute per-HRU area (m²) and centroid coords in a single fabric pass.

    Combined helper for builders that need both (runoff). Lighter
    alternatives are :func:`compute_hru_centroids` (centroids only) and
    :func:`compute_hru_areas` (area only) — prefer them when you don't
    need the other column.

    Always recomputes from geometry (no fabric-column fallback) so the
    area cannot drift from the geometry actually being processed.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``project.id_col``. Columns: ``area_m2``,
        ``centroid_x``, ``centroid_y`` (in ``area_crs``),
        ``centroid_lat``, ``centroid_lon`` (EPSG:4326).
    """
    gdf_eq, id_col = _load_and_reproject_fabric(project)
    centroids_eq = gdf_eq.geometry.centroid
    centroids_ll = centroids_eq.to_crs("EPSG:4326")

    df = gdf_eq[[id_col]].copy()
    df["area_m2"] = gdf_eq.geometry.area.astype(float)
    df["centroid_x"] = centroids_eq.x.astype(float)
    df["centroid_y"] = centroids_eq.y.astype(float)
    df["centroid_lon"] = centroids_ll.x.astype(float)
    df["centroid_lat"] = centroids_ll.y.astype(float)
    df = df.set_index(id_col).sort_index()
    return df


def write_target_nc(
    ds: xr.Dataset,
    output_path: Path,
    title: str,
    extra_global_attrs: dict | None = None,
    sort_dim: str | None = None,
) -> None:
    """Write a target Dataset to NetCDF atomically with CF-1.6 metadata.

    The Dataset is expected to already carry the data variables, ancillary
    coordinates (``time_bnds``, ``centroid_lat``, ``centroid_lon``), and
    per-variable attrs (``units``, ``long_name``, ``cell_methods``, etc.).
    This helper sets the global ``Conventions`` / ``title`` / ``history`` /
    ``software_version`` attrs, applies float32+zlib encoding for the bound
    variables and int8+zlib encoding for the diagnostic variables, and
    writes via tempfile + rename so a partial NetCDF never lands at the
    final path.

    When ``sort_dim`` is given, the Dataset is sorted ascending on that
    dimension before write. Target builders pass ``project.id_col`` here
    to enforce the canonical HRU row order at the emission boundary
    (issue #93). Upstream helpers (``read_aggregated_source``,
    ``compute_hru_area_and_centroids``) already produce sorted data; the
    explicit sort here makes the invariant unmistakable at the file boundary.
    """
    from datetime import datetime, timezone

    from nhf_spatial_targets import __version__

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = ds.copy()
    if sort_dim is not None:
        ds = ds.sortby(sort_dim)
    ds.attrs.setdefault("Conventions", "CF-1.6")
    ds.attrs["title"] = title
    ds.attrs["history"] = (
        f"{datetime.now(timezone.utc).isoformat()} created by "
        f"nhf_spatial_targets v{__version__}"
    )
    ds.attrs.setdefault("institution", "USGS")
    ds.attrs.setdefault("software_version", __version__)
    if extra_global_attrs:
        ds.attrs.update(extra_global_attrs)

    encoding: dict = {}
    for v in ("lower_bound", "upper_bound"):
        if v in ds.data_vars:
            encoding[v] = {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.float32("nan"),
            }
    for v in ("n_sources", "nn_filled"):
        if v in ds.data_vars:
            encoding[v] = {
                "dtype": "int8",
                "zlib": True,
                "complevel": 4,
                "_FillValue": None,
            }
    if "time" in ds.coords or "time" in ds.dims or "time" in ds.variables:
        encoding["time"] = {
            "dtype": "float64",
            "units": "days since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        }
    if "time_bnds" in ds.variables:
        encoding["time_bnds"] = {
            "dtype": "float64",
            "units": "days since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        }

    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        ds.to_netcdf(tmp, format="NETCDF4", encoding=encoding)
        tmp.rename(output_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    logger.info("Wrote %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
