"""Read aggregated NCs, canonicalize coords, and fabric-side derivations.

Pure-function helpers shared by every target builder:

- :func:`read_aggregated_source` opens per-year aggregated NCs as one
  lazy DataArray and slices to a requested period.
- :func:`reindex_to_month_start` / :func:`reindex_to_day_start` normalize
  source timestamps (mid-month, end-of-month, noon, midnight) onto a
  canonical master index.
- :func:`compute_hru_centroids` / :func:`compute_hru_areas` /
  :func:`compute_hru_area_and_centroids` derive per-HRU geometry needed
  for the bound writer.
- :func:`parse_period` splits a ``"YYYY/YYYY"`` config string.
- :func:`check_hru_coords` guards the canonical-row-order invariant
  immediately after a source read.

This module is unit-agnostic; per-source unit shims live in the target
builder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


_AGG_READ_CHUNK_BYTES = 256 * 1024 * 1024


def _read_chunk_hru(
    n_time: int, itemsize: int, target_bytes: int = _AGG_READ_CHUNK_BYTES
) -> int:
    """HRU-dim dask chunk length for reading aggregated NCs (#165 ST3).

    Reads use full-time chunks (``{"time": -1}``) so a file's on-disk time
    chunk is never split (which would re-inflate the same compressed chunk on
    every time slab). The HRU dim is chunked to ~``target_bytes`` per chunk to
    keep per-chunk memory and the dask graph reasonable — deliberately larger
    than the on-disk ~1 MiB HRU chunk, trading chunk granularity for a smaller
    graph on a full-source read.
    """
    return max(1, target_bytes // (max(n_time, 1) * itemsize))


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
        Forwarded to ``xr.open_mfdataset``. When ``None``, defaults to
        ``{"time": -1, project.id_col: <chunk_hru>}`` — full time per chunk,
        HRU dim chunked to ~256 MiB — to align reads with the columnar
        on-disk layout written by the aggregator (#165 ST3). Pass an explicit
        dict to override.

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
        # Align dask chunks to the new columnar on-disk layout (#165 ST3):
        # full time per chunk so a file's on-disk time chunk is never split
        # (splitting would force the compressor to re-inflate the same chunk
        # on every time slab — ~30x amplification on a daily source), with the
        # HRU dim chunked to ~256 MiB. Works for both the new chunked NCs and
        # legacy contiguous NCs (the latter read as bounded strided slabs, not
        # a whole-array load). open_mfdataset chunks per file along the concat
        # dim, so ``{"time": -1}`` yields one time chunk per file — size the
        # HRU chunk from one file's time length (probed cheaply from the first
        # file) so each per-file chunk lands near the byte budget.
        with xr.open_dataset(paths[0], engine="netcdf4") as probe:
            n_time = int(probe.sizes.get("time", 1))
            itemsize = probe[var].dtype.itemsize if var in probe.data_vars else 8
        chunks = {"time": -1, project.id_col: _read_chunk_hru(n_time, itemsize)}

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
    # Substitute netCDF default-fill cells (NC_FILL_DOUBLE etc.) with NaN.
    # Aggregated NCs written before issue #204's fix declare _FillValue=NaN
    # but hold the ~1e36 sentinel in cells gdptools couldn't fill (e.g. OR
    # HRUs outside MWBM CONUS extent). xarray's decode masks via value
    # equality and NaN never matches, so the sentinel survives and poisons
    # downstream np.fmax. Mirrors the write-side fix in
    # aggregate/_driver.py:_atomic_write_netcdf — keeping both means
    # existing on-disk NCs decode correctly without a rewrite while fresh
    # writes are self-consistent on disk.
    from nhf_spatial_targets.io_nc import mask_netcdf_default_fills

    ds = mask_netcdf_default_fills(ds)
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


def reindex_to_day_start(
    da: xr.DataArray, master_index: pd.DatetimeIndex
) -> xr.DataArray:
    """Reindex a daily DataArray onto a master ``freq="D"`` index.

    Source timestamps may be midnight (SNODAS, ERA5-Land daily, Margulis)
    or noon (Daymet — calendar-day mean). Both convey "which calendar
    day" unambiguously; this helper normalises by stripping the time of
    day (``.dt.floor("D")``) before reindexing.

    Days in ``master_index`` that the source does not cover come back as
    NaN — this is what gives the SWE target its period-union semantics:
    a source whose record ends in 2020 but is asked through 2024 simply
    contributes nothing for the post-2020 cells.

    Requires ``da.time`` to be decoded as ``datetime64[ns]`` (the xarray
    default for ``proleptic_gregorian`` calendars, which is what every
    NC the pipeline writes uses — see ``fetch/consolidate.py``). Sources
    decoded as ``cftime`` objects (non-standard calendars like
    ``noleap`` / ``360_day``) raise ``TypeError`` rather than silently
    falling back to a ``DatetimeIndex`` conversion that loses the
    calendar.
    """
    if not isinstance(master_index, pd.DatetimeIndex):
        raise TypeError(
            f"master_index must be a pandas.DatetimeIndex, got "
            f"{type(master_index).__name__}"
        )
    if master_index.freqstr != "D":
        raise ValueError(
            f"master_index must have freq='D' (daily); got "
            f"freq={master_index.freqstr!r}. Build it with "
            f"pd.date_range(start, end, freq='D')."
        )
    if not np.issubdtype(da.time.dtype, np.datetime64):
        raise TypeError(
            f"reindex_to_day_start expects datetime64-decoded time, got "
            f"dtype={da.time.dtype!r}. The pipeline writes every NC with "
            f"calendar='proleptic_gregorian' so xarray should decode to "
            f"datetime64[ns]; a cftime-decoded source indicates either a "
            f"non-standard upstream calendar or decode_cf=False during "
            f"open. Re-open with the default decoder or report as a "
            f"consolidator bug."
        )
    day_times = pd.DatetimeIndex(da.time.values).floor("D")
    canon = da.assign_coords(time=day_times)
    return canon.reindex(time=master_index)


def parse_period(period_str: str) -> tuple[str, str]:
    """Parse 'YYYY-MM-DD/YYYY-MM-DD' (or 'YYYY/YYYY') into ``(start, end)``.

    Used by every target builder to split the project config's
    ``<target>.period`` (and ``<target>.normalize_period`` where present)
    into the two endpoints needed to slice ``read_aggregated_source``'s
    output.
    """
    if "/" not in period_str:
        raise ValueError(
            f"Invalid period {period_str!r}. Expected 'YYYY-MM-DD/YYYY-MM-DD'."
        )
    start, end = period_str.split("/", 1)
    return start.strip(), end.strip()


def check_hru_coords(
    da: xr.DataArray,
    fabric_hru_ids: np.ndarray,
    id_col: str,
    source_key: str,
) -> None:
    """Raise if the source DataArray's HRU dim disagrees with the fabric.

    Both sides are expected to be sorted ascending by ``id_col`` upstream
    (``compute_hru_*`` helpers and ``read_aggregated_source`` enforce
    this). Three outcomes:

    - Coords match exactly: returns ``None``.
    - Coords have the SAME SET but a different order: raises with a
      "canonical-sort invariant regression" message — the upstream
      sort-on-emission contract (#93) has been broken somewhere.
    - Coords have different SETS: raises a "re-aggregate this source
      against the current fabric" message.

    Called by target builders right after ``read_aggregated_source``
    so any HRU misalignment is caught before silent broadcast/intersect
    arithmetic poisons downstream values.
    """
    src_hru_ids = da[id_col].values
    if np.array_equal(src_hru_ids, fabric_hru_ids):
        return
    same_set = len(src_hru_ids) == len(fabric_hru_ids) and np.array_equal(
        np.sort(src_hru_ids), np.sort(fabric_hru_ids)
    )
    if same_set:
        raise ValueError(
            f"HRU coords for source '{source_key}' have the same set as the "
            f"fabric ({len(fabric_hru_ids)} HRUs) but a different order. "
            f"Both sides are expected to be sorted ascending by id_col="
            f"'{id_col}' — this indicates a regression in the canonical-"
            f"sort invariant in targets/_io.py."
        )
    raise ValueError(
        f"HRU coords differ between fabric and source '{source_key}' as "
        f"sets. Fabric has {len(fabric_hru_ids)} HRUs "
        f"(first={fabric_hru_ids[0]}, last={fabric_hru_ids[-1]}); "
        f"source has {len(src_hru_ids)} "
        f"(first={src_hru_ids[0]}, last={src_hru_ids[-1]}). "
        f"Re-aggregate '{source_key}' against the current fabric."
    )


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
