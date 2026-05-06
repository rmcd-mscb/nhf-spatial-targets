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

import xarray as xr

from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


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
