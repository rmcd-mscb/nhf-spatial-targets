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
        On-disk storage key. Used as the subdirectory name under
        ``<project>/data/aggregated/`` and as the prefix of the per-year
        NC filename. Matches the catalog source key in the common case,
        but may be a synthetic key (e.g. ``"era5_land_sd"``) when one
        upstream source produces multiple aggregated cadences in
        separate subdirs — see ``aggregate/era5_land.py``'s
        ``ADAPTER_SD``.
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
    config_label
        Optional user-facing alias used in the project config's
        ``<target>.sources`` list. Defaults to ``source_key``. Set this
        when the storage key is a synthetic disambiguation (e.g.
        ``"era5_land_sd"``) but the config should keep the canonical
        catalog name (``"era5_land"``) — this keeps target builders
        free of any parallel label→storage dict that could drift from
        the SHIMS registry.
    """

    source_key: str
    aggregated_var: str
    description: str
    to_common_units: Callable[[xr.DataArray], xr.DataArray]
    config_label: str | None = None


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


def shims_by_config_label(
    shims: "tuple[SourceShim, ...]",
) -> "dict[str, SourceShim]":
    """Index a ``SHIMS`` tuple by ``config_label`` (defaulted to ``source_key``).

    Used by target builders to resolve a user-facing source name from
    the project config (e.g. ``"era5_land"``) to the SourceShim that
    encodes its storage key, aggregated variable, and unit shim. Raises
    ``ValueError`` on duplicate labels — two shims wanting the same
    config label is the same drift class :func:`shims_by_key` guards.
    """
    out: dict[str, SourceShim] = {}
    for shim in shims:
        label = shim.config_label or shim.source_key
        if label in out:
            raise ValueError(
                f"Duplicate SourceShim config_label={label!r} in target "
                f"SHIMS registry. Each label may appear at most once "
                f"(set distinct config_label values to disambiguate)."
            )
        out[label] = shim
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


# ---------------------------------------------------------------------------
# Shared target-builder helpers
# ---------------------------------------------------------------------------


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
            f"sort invariant in targets/_common.py."
        )
    raise ValueError(
        f"HRU coords differ between fabric and source '{source_key}' as "
        f"sets. Fabric has {len(fabric_hru_ids)} HRUs "
        f"(first={fabric_hru_ids[0]}, last={fabric_hru_ids[-1]}); "
        f"source has {len(src_hru_ids)} "
        f"(first={src_hru_ids[0]}, last={src_hru_ids[-1]}). "
        f"Re-aggregate '{source_key}' against the current fabric."
    )


def build_n_sources_attrs(
    n_sources_count: int,
    ancillary_coords: str = "centroid_lat centroid_lon",
) -> dict:
    """Build the per-variable attrs dict for an ``n_sources`` diagnostic var.

    Parameters
    ----------
    n_sources_count
        Number of source contributors (an integer ≥ 0 and ≤ 5). Determines
        the length of the ``flag_values`` list and the matching
        ``flag_meanings`` labels (``none one two three four five``,
        truncated to ``n_sources_count + 1`` entries).
    ancillary_coords
        Space-separated list of ancillary coordinate variable names to
        record under CF's ``coordinates`` attr. Defaults to the centroid
        pair used by every target builder.
    """
    flag_labels = ["none", "one", "two", "three", "four", "five"]
    if n_sources_count + 1 > len(flag_labels):
        raise ValueError(
            f"build_n_sources_attrs: n_sources_count={n_sources_count} exceeds "
            f"the {len(flag_labels) - 1}-source label vocabulary."
        )
    return {
        "units": "1",
        "long_name": "number of finite source contributions",
        "flag_values": list(range(0, n_sources_count + 1)),
        "flag_meanings": " ".join(flag_labels[: n_sources_count + 1]),
        "coordinates": ancillary_coords,
    }


def write_bounds_target(
    *,
    project: Project,
    lower: xr.DataArray,
    upper: xr.DataArray,
    n_sources: xr.DataArray,
    n_sources_count: int,
    time_index: pd.DatetimeIndex,
    time_offset_unit: object,
    bounds_units: str,
    bounds_long_name_kind: str,
    cell_methods: str,
    output_path: Path,
    title: str,
    nn_title: str,
    extra_global_attrs: dict,
    hru_meta: "pd.DataFrame",
    nn_fill: bool,
    nn_max_candidates: int,
    id_col: str,
) -> None:
    """Assemble + write a bounds-target Dataset, with optional NN-fill companion.

    Consolidates the assemble-and-write pipeline shared by every target
    builder (runoff, AET, recharge, soil moisture): centroid coords,
    ``time_bnds``, per-variable attrs (units / long_name / cell_methods /
    coordinates), global attrs, atomic write via ``write_target_nc``, a
    coverage-summary log line, and the optional ``nn_fill_bounds``
    companion file.

    Parameters
    ----------
    project
        Loaded :class:`~nhf_spatial_targets.workspace.Project`.
    lower, upper, n_sources
        The three combined-source DataArrays from
        :func:`multi_source_nanminmax`.
    n_sources_count
        Total number of source contributors (an int); drives the
        ``n_sources`` diagnostic's ``flag_values`` length.
    time_index
        Master ``DatetimeIndex`` that ``lower`` / ``upper`` are aligned to.
    time_offset_unit
        Offset added to each ``time_index`` entry to form ``time_bnds``'s
        upper edge (e.g. ``pd.offsets.MonthBegin(1)`` for monthly,
        ``pd.offsets.YearBegin(1)`` for annual).
    bounds_units
        Units string for the lower/upper variable attrs (e.g. ``"cfs"``,
        ``"inches/day"``, ``"1"``).
    bounds_long_name_kind
        Substituted into the ``long_name`` template: ``"lower bound of
        {kind} (NaN-aware min across sources)"``. Examples: ``"monthly
        runoff"``, ``"annual recharge"``.
    cell_methods
        CF ``cell_methods`` attr value for both bounds (e.g.
        ``"time: sum"``, ``"time: mean"``).
    output_path
        Final NetCDF path for the unfilled target.
    title
        ``title`` global attr for the unfilled target.
    nn_title
        ``title`` global attr for the NN-filled companion (only used when
        ``nn_fill`` is True).
    extra_global_attrs
        Per-target metadata (``source``, ``period``, ``fabric_sha256``,
        etc.) — passed through to ``write_target_nc``.
    hru_meta
        DataFrame returned by ``compute_hru_centroids`` (or the combined
        helper). Must contain ``centroid_lat``, ``centroid_lon``,
        ``centroid_x``, ``centroid_y`` columns.
    nn_fill
        If True, additionally write ``<output>_nn_filled.nc`` via
        :func:`nn_fill_bounds`.
    nn_max_candidates
        Forwarded to :func:`nn_fill_bounds`.
    id_col
        HRU id column name (e.g. ``"nhm_id"``); the dataset is sorted
        ascending on this dim at emission per the #93 canonical-row-order
        invariant.
    """
    # Avoid a circular-import by deferring this helper-internal import.
    from nhf_spatial_targets.normalize.methods import nn_fill_bounds

    lower.name = "lower_bound"
    upper.name = "upper_bound"
    n_sources.name = "n_sources"

    time_bnds = xr.DataArray(
        list(zip(time_index.values, (time_index + time_offset_unit).values)),
        dims=("time", "nv"),
        coords={"time": time_index.values},
        name="time_bnds",
    )
    centroid_lat = xr.DataArray(
        hru_meta["centroid_lat"].values,
        dims=(id_col,),
        coords={id_col: hru_meta.index.values},
        attrs={
            "units": "degrees_north",
            "standard_name": "latitude",
            "long_name": "HRU centroid latitude",
        },
    )
    centroid_lon = xr.DataArray(
        hru_meta["centroid_lon"].values,
        dims=(id_col,),
        coords={id_col: hru_meta.index.values},
        attrs={
            "units": "degrees_east",
            "standard_name": "longitude",
            "long_name": "HRU centroid longitude",
        },
    )

    lower.attrs.update(
        {
            "units": bounds_units,
            "long_name": (
                f"lower bound of {bounds_long_name_kind} (NaN-aware min across sources)"
            ),
            "cell_methods": cell_methods,
            "coordinates": "centroid_lat centroid_lon",
        }
    )
    upper.attrs.update(
        {
            "units": bounds_units,
            "long_name": (
                f"upper bound of {bounds_long_name_kind} (NaN-aware max across sources)"
            ),
            "cell_methods": cell_methods,
            "coordinates": "centroid_lat centroid_lon",
        }
    )
    n_sources.attrs.update(build_n_sources_attrs(n_sources_count))

    ds = xr.Dataset(
        {
            "lower_bound": lower,
            "upper_bound": upper,
            "n_sources": n_sources,
        },
        coords={
            "time": time_index,
            id_col: lower[id_col],
            "time_bnds": time_bnds,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
        },
    )
    ds["time"].attrs["bounds"] = "time_bnds"
    ds["time"].attrs["axis"] = "T"
    ds["time"].attrs["standard_name"] = "time"
    ds[id_col].attrs["long_name"] = "HRU identifier"
    ds[id_col].attrs["cf_role"] = "timeseries_id"

    ds_loaded = ds.compute()

    write_target_nc(
        ds_loaded,
        output_path,
        title=title,
        extra_global_attrs=extra_global_attrs,
        sort_dim=id_col,
    )

    n = ds_loaded["n_sources"].values
    total = n.size
    none = int((n == 0).sum())
    logger.info(
        "%s coverage: %d/%d cells have >=1 finite source (%.2f%% all-NaN)",
        bounds_long_name_kind,
        total - none,
        total,
        100.0 * none / total if total else 0.0,
    )

    if not nn_fill:
        return

    centroids_xy = hru_meta[["centroid_x", "centroid_y"]].values
    filled_ds, nn_diag = nn_fill_bounds(
        ds_loaded, centroids_xy, max_candidates=nn_max_candidates
    )
    nn_diag.attrs.update(
        {
            "units": "1",
            "long_name": "nearest-neighbor fill flag",
            "flag_values": [0, 1],
            "flag_meanings": "not_filled filled",
            "coordinates": "centroid_lat centroid_lon",
        }
    )
    filled_ds["nn_filled"] = nn_diag
    filled_attrs = dict(extra_global_attrs)
    filled_attrs["nn_fill_max_candidates"] = nn_max_candidates
    filled_attrs["nn_fill_distance_crs"] = project.area_crs
    nn_path = output_path.with_name(
        output_path.stem + "_nn_filled" + output_path.suffix
    )
    write_target_nc(
        filled_ds,
        nn_path,
        title=nn_title,
        extra_global_attrs=filled_attrs,
        sort_dim=id_col,
    )


# ---------------------------------------------------------------------------
# Year-chunked target build (memory-bounded for multi-year daily targets)
# ---------------------------------------------------------------------------


def iter_period_years(
    period_start: str, period_end: str
) -> "list[tuple[int, str, str]]":
    """Iterate ``(year, year_start_iso, year_end_iso)`` over the period.

    The first and last yields are clipped to ``period_start`` / ``period_end``
    when those fall mid-year. Used by year-chunked target builders to drive
    the per-year build loop.

    Examples
    --------
    >>> iter_period_years("1980-06-15", "1982-03-20")
    [(1980, '1980-06-15', '1980-12-31'),
     (1981, '1981-01-01', '1981-12-31'),
     (1982, '1982-01-01', '1982-03-20')]
    """
    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end)
    if end < start:
        raise ValueError(f"period end {period_end!r} precedes start {period_start!r}")
    out: list[tuple[int, str, str]] = []
    for year in range(start.year, end.year + 1):
        ys = max(start, pd.Timestamp(f"{year}-01-01"))
        ye = min(end, pd.Timestamp(f"{year}-12-31"))
        out.append((year, ys.strftime("%Y-%m-%d"), ye.strftime("%Y-%m-%d")))
    return out


def stitch_year_chunks_to_target(
    intermediate_files: "list[Path]",
    output_path: Path,
    *,
    title: str,
    extra_global_attrs: dict | None,
    sort_dim: str,
    time_chunk_days: int = 365,
) -> None:
    """Lazily open per-year target NCs and stream them into one canonical NC.

    Uses ``xr.open_mfdataset`` with explicit ``chunks=`` to keep the
    dataset dask-backed, then writes via ``to_netcdf`` so the per-year
    chunks flow through to disk one at a time instead of being
    materialised in memory. Peak memory therefore stays bounded by one
    year's worth of data regardless of how many years the period spans
    — the whole point of the year-chunked build pattern.

    Per-variable encoding mirrors :func:`write_target_nc` (float32+zlib
    for ``lower_bound`` / ``upper_bound``, int8+zlib for ``n_sources`` /
    ``nn_filled``, ``days since 1970-01-01`` for the time axis); the
    global attrs are taken from the first per-year file and then
    overridden by ``extra_global_attrs`` plus a fresh ``history`` line.

    The stitched output is sorted ascending on ``sort_dim`` (typically
    ``project.id_col``) before write to preserve the canonical row order
    invariant from issue #93. Atomic via tempfile + rename, same as
    ``write_target_nc``.

    Parameters
    ----------
    intermediate_files
        Per-year NC paths to stitch (already sorted by year via
        filename glob).
    output_path
        Final canonical NC path (e.g. ``<project>/targets/swe_targets.nc``).
    title
        Global ``title`` attr for the stitched file.
    extra_global_attrs
        Per-target metadata to overlay on top of the per-year files'
        global attrs (e.g. ``period``, ``source``, ``fabric``).
    sort_dim
        Dimension to sort ascending on before write (typically
        ``project.id_col``).
    time_chunk_days
        Dask chunk size along the time axis when opening intermediates.
        Defaults to 365 — one year per chunk, which is also the natural
        per-file boundary, so to_netcdf streams one file's worth at a
        time.
    """
    from datetime import datetime, timezone

    from nhf_spatial_targets import __version__

    if not intermediate_files:
        raise ValueError(
            "stitch_year_chunks_to_target: intermediate_files is empty; "
            "the year loop produced no per-year NCs."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.open_mfdataset(
        [str(p) for p in intermediate_files],
        combine="by_coords",
        join="outer",
        chunks={"time": time_chunk_days, sort_dim: -1},
        engine="netcdf4",
    )
    try:
        ds = ds.sortby(sort_dim)
        ds.attrs.setdefault("Conventions", "CF-1.6")
        ds.attrs["title"] = title
        ds.attrs["history"] = (
            f"{datetime.now(timezone.utc).isoformat()} stitched from "
            f"{len(intermediate_files)} per-year NCs by "
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
            # Streaming write — dask schedules one chunk at a time, so
            # peak memory stays bounded by ~one year's worth of data.
            ds.to_netcdf(tmp, format="NETCDF4", encoding=encoding)
            tmp.rename(output_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    finally:
        ds.close()

    logger.info(
        "Stitched %d per-year NCs -> %s (%.1f MB)",
        len(intermediate_files),
        output_path,
        output_path.stat().st_size / 1e6,
    )
