"""Merge per-granule NetCDF files into single consolidated datasets."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from nhf_spatial_targets import __version__

logger = logging.getLogger(__name__)


def open_consolidated(nc_path: Path) -> xr.Dataset:
    """Open a consolidated NetCDF file produced by a ``consolidate_*`` function.

    Parameters
    ----------
    nc_path : Path
        Path to a ``*_consolidated.nc`` file.

    Returns
    -------
    xr.Dataset
    """
    return xr.open_dataset(nc_path)


def _fix_time_merra2(ds: xr.Dataset) -> xr.Dataset:
    """Shift MERRA-2 timestamps to mid-month and add time_bnds.

    MERRA-2 monthly files have timestamps at the first of the month plus
    30 minutes (e.g. 2010-01-01T00:30:00).  This normalizes them to the
    15th of each month to provide a conventional mid-month representative
    timestamp for monthly data.
    """
    original_times = pd.DatetimeIndex(ds.time.values)
    epoch = pd.Timestamp("1970-01-01")

    # Build mid-month timestamps (day 15, midnight)
    mid_month = pd.DatetimeIndex(
        [
            t.replace(day=15, hour=0, minute=0, second=0, microsecond=0)
            for t in original_times
        ]
    )

    # Build time_bnds: [first-of-month, first-of-next-month)
    bounds_list = []
    for t in original_times:
        m_start = t.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if t.month == 12:
            m_end = t.replace(
                year=t.year + 1,
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        else:
            m_end = t.replace(
                month=t.month + 1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        bounds_list.append([(m_start - epoch).days, (m_end - epoch).days])

    ds = ds.assign_coords(time=mid_month)
    ds.time.attrs.update(
        {
            "bounds": "time_bnds",
            "cell_methods": "time: mean",
        }
    )

    bnds_arr = np.array(bounds_list, dtype="<i8")
    nv = np.array([0, 1])
    ds["time_bnds"] = xr.DataArray(
        bnds_arr,
        dims=["time", "nv"],
        attrs={"units": "days since 1970-01-01", "calendar": "standard"},
    )
    ds = ds.assign_coords(nv=nv)

    return ds


def consolidate_merra2(
    run_dir: Path,
    variables: list[str],
) -> dict:
    """Merge per-granule MERRA-2 files into a single consolidated NetCDF.

    Opens all ``.nc4`` files in ``data/raw/merra2/``, selects the requested
    variables, concatenates along time, shifts timestamps to mid-month,
    and writes a single ``merra2_consolidated.nc``.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory containing ``data/raw/merra2/*.nc4``.
    variables : list[str]
        Variable names to include (e.g. ["GWETTOP", "GWETROOT", "GWETPROF"]).

    Returns
    -------
    dict
        Provenance record with consolidated file path and timestamp.
    """
    import nhf_spatial_targets.catalog as _catalog

    merra2_dir = run_dir / "data" / "raw" / "merra2"
    nc_files = sorted(merra2_dir.glob("*.nc4"))

    if not nc_files:
        raise FileNotFoundError(
            f"No .nc4 files found in {merra2_dir}. "
            f"Run 'nhf-targets fetch merra2' first."
        )

    logger.info("Merging %d NetCDF files for MERRA-2", len(nc_files))

    datasets = []
    for f in tqdm(nc_files, desc="Reading MERRA-2 files"):
        datasets.append(xr.open_dataset(f, chunks={}))
    ds = xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal")
    ds = ds.sortby("time")
    ds = ds[variables]
    ds = _fix_time_merra2(ds)

    # Add CF and provenance global attributes
    meta = _catalog.source("merra2")
    ds.attrs.update(
        {
            "Conventions": "CF-1.8",
            "history": (f"Consolidated by nhf-spatial-targets v{__version__}"),
            "source": (
                f"NASA MERRA-2 {meta['access']['short_name']}"
                f" v{meta['access'].get('version', 'unknown')}"
            ),
            "time_modification_note": (
                "Original timestamps (YYYY-MM-01T00:30:00) shifted to mid-month "
                "(15th) for consistency. See time_bnds for exact averaging periods."
            ),
            "references": meta["access"]["url"],
        }
    )

    out_path = merra2_dir / "merra2_consolidated.nc"
    logger.info("Writing consolidated file: %s", out_path)
    encoding = {"time": {"units": "days since 1970-01-01", "calendar": "standard"}}
    with ProgressBar():
        ds.to_netcdf(out_path, encoding=encoding)
    for d in datasets:
        d.close()
    ds.close()
    logger.info("Wrote %s", out_path)

    return {
        "consolidated_nc": str(out_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(nc_files),
        "variables": variables,
    }


def consolidate_nldas(
    run_dir: Path,
    source_key: str,
    variables: list[str],
) -> dict:
    """Merge per-granule NLDAS files into a single consolidated NetCDF.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory.
    source_key : str
        Source key (e.g. "nldas_mosaic" or "nldas_noah").
    variables : list[str]
        Variable names to include.

    Returns
    -------
    dict
        Provenance record.
    """
    source_dir = run_dir / "data" / "raw" / source_key
    nc_files = sorted(list(source_dir.glob("*.nc4")) + list(source_dir.glob("*.nc")))

    if not nc_files:
        raise FileNotFoundError(
            f"No NetCDF files found in {source_dir}. "
            f"Run 'nhf-targets fetch {source_key.replace('_', '-')}' first."
        )

    logger.info("Merging %d NetCDF files for %s", len(nc_files), source_key)

    datasets = []
    for f in tqdm(nc_files, desc=f"Reading {source_key} files"):
        datasets.append(xr.open_dataset(f, chunks={}))
    ds = xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal")
    ds = ds.sortby("time")
    ds = ds[variables]

    out_path = source_dir / f"{source_key}_consolidated.nc"
    logger.info("Writing consolidated file: %s", out_path)
    with ProgressBar():
        ds.to_netcdf(out_path)
    for d in datasets:
        d.close()
    ds.close()
    logger.info("Wrote %s", out_path)

    return {
        "consolidated_nc": str(out_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(nc_files),
        "variables": variables,
    }


def consolidate_ncep_ncar(
    run_dir: Path,
    variables: list[str],
) -> dict:
    """Merge per-year NCEP/NCAR monthly files into a single consolidated NetCDF.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory containing ``data/raw/ncep_ncar/*.monthly.nc``.
    variables : list[str]
        Variable names to include.

    Returns
    -------
    dict
        Provenance record.
    """
    ncep_dir = run_dir / "data" / "raw" / "ncep_ncar"
    nc_files = sorted(ncep_dir.glob("*.monthly.nc"))

    if not nc_files:
        raise FileNotFoundError(
            f"No .monthly.nc files found in {ncep_dir}. "
            "Run 'nhf-targets fetch ncep-ncar' first."
        )

    logger.info("Merging %d monthly NetCDF files for NCEP/NCAR", len(nc_files))

    # NCEP/NCAR has separate files per variable per year (e.g. soilw.0-10cm.gauss.2010.monthly.nc).
    # Group by variable, concat each group along time, then merge across variables.
    from collections import defaultdict

    groups: dict[str, list[Path]] = defaultdict(list)
    for f in nc_files:
        ds_peek = xr.open_dataset(f, chunks={})
        data_vars = [v for v in ds_peek.data_vars if v != "time_bnds"]
        key = tuple(sorted(data_vars))
        groups[key].append(f)
        ds_peek.close()

    merged_parts = []
    for var_key, file_group in groups.items():
        datasets = []
        for f in tqdm(file_group, desc=f"Reading NCEP/NCAR {', '.join(var_key)}"):
            datasets.append(xr.open_dataset(f, chunks={}))
        part = xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal")
        part = part.sortby("time")
        merged_parts.append(part)

    ds = xr.merge(merged_parts)
    ds = ds[variables]

    out_path = ncep_dir / "ncep_ncar_consolidated.nc"
    logger.info("Writing consolidated file: %s", out_path)
    with ProgressBar():
        ds.to_netcdf(out_path)
    for part in merged_parts:
        part.close()
    ds.close()
    logger.info("Wrote %s", out_path)

    return {
        "consolidated_nc": str(out_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(nc_files),
        "variables": variables,
    }
