"""Merge per-granule NetCDF files into single consolidated datasets."""

from __future__ import annotations

import gc
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from nhf_spatial_targets import __version__

logger = logging.getLogger(__name__)


def log_memory(label: str) -> None:
    """Log current RSS (Linux /proc) or peak RSS (resource module fallback).

    On macOS the ``resource`` fallback adjusts for bytes-valued ``ru_maxrss``.
    This function never raises — all errors are caught and logged.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    rss_gib = rss_kb / (1024**2)
                    logger.info("[memory] RSS=%.2f GiB — %s", rss_gib, label)
                    return
    except (OSError, ValueError) as exc:
        logger.debug("[memory] /proc/self/status unavailable: %s — %s", exc, label)
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            peak_gib = peak / (1024**3)  # bytes on macOS
        else:
            peak_gib = peak / (1024**2)  # KB on Linux
        logger.info("[memory] peak RSS=%.2f GiB — %s", peak_gib, label)
    except (ImportError, OSError):
        logger.debug("[memory] cannot read RSS on this platform — %s", label)


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
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Consolidated file not found: {nc_path}. "
            f"Run the appropriate fetch command first."
        )
    try:
        return xr.open_dataset(nc_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open consolidated file {nc_path}. "
            f"The file may be corrupt — delete it and re-consolidate. "
            f"Detail: {exc}"
        ) from exc


def _validate_variables(ds: xr.Dataset, variables: list[str]) -> None:
    """Raise ValueError if any requested variables are missing from *ds*."""
    missing = set(variables) - set(ds.data_vars)
    if missing:
        raise ValueError(
            f"Requested variables {sorted(missing)} not found in the "
            f"concatenated dataset. Available variables: {sorted(ds.data_vars)}. "
            f"Check catalog/sources.yml or re-fetch the source data."
        )


def _open_datasets(nc_files: list[Path], desc: str) -> list[xr.Dataset]:
    """Open a list of NetCDF files with tqdm progress, cleaning up on failure."""
    datasets: list[xr.Dataset] = []
    for f in tqdm(nc_files, desc=desc):
        try:
            datasets.append(xr.open_dataset(f, chunks={}))
        except Exception as exc:
            for d in datasets:
                d.close()
            raise RuntimeError(
                f"Failed to open {f.name} during consolidation. "
                f"The file may be corrupt or truncated. "
                f"Delete it and re-run the fetch. Detail: {exc}"
            ) from exc
    return datasets


def _write_netcdf(
    ds: xr.Dataset,
    out_path: Path,
    encoding: dict | None = None,
) -> None:
    """Write dataset to NetCDF, removing partial file on failure."""
    try:
        kwargs: dict = {}
        if encoding is not None:
            kwargs["encoding"] = encoding
        with ProgressBar():
            ds.to_netcdf(out_path, **kwargs)
    except Exception as exc:
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError(
            f"Failed to write consolidated file {out_path}. "
            f"Check available disk space and permissions. Detail: {exc}"
        ) from exc


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

    datasets = _open_datasets(nc_files, "Reading MERRA-2 files")
    try:
        ds = xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal")
        ds = ds.sortby("time")
        _validate_variables(ds, variables)
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
        _write_netcdf(ds, out_path, encoding=encoding)
        logger.info("Wrote %s", out_path)
    finally:
        for d in datasets:
            d.close()

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

    datasets = _open_datasets(nc_files, f"Reading {source_key} files")
    try:
        ds = xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal")
        ds = ds.sortby("time")
        _validate_variables(ds, variables)
        ds = ds[variables]

        out_path = source_dir / f"{source_key}_consolidated.nc"
        logger.info("Writing consolidated file: %s", out_path)
        _write_netcdf(ds, out_path)
        logger.info("Wrote %s", out_path)
    finally:
        for d in datasets:
            d.close()

    return {
        "consolidated_nc": str(out_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(nc_files),
        "variables": variables,
    }


def _time_from_modis_filename(path: Path) -> pd.Timestamp:
    """Extract a timestamp from a MODIS ``AYYYYDDD`` filename pattern."""
    m = re.search(r"\.A(\d{4})(\d{3})\.", path.name)
    if not m:
        raise ValueError(f"Cannot extract date from MODIS filename: {path.name}")
    year, doy = int(m.group(1)), int(m.group(2))
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)


def consolidate_mod10c1(
    run_dir: Path,
    source_key: str,
    variables: list[str],
    year: int,
) -> dict:
    """Merge daily MOD10C1 CONUS subsets for a single year into one NetCDF.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory.
    source_key : str
        Source key (e.g. ``"mod10c1_v061"``).
    variables : list[str]
        Variable names to include.
    year : int
        Year to consolidate.

    Returns
    -------
    dict
        Provenance record.
    """
    source_dir = run_dir / "data" / "raw" / source_key
    year_pattern = re.compile(rf"\.A{year}\d{{3}}\.")
    nc_files = sorted(
        f for f in source_dir.glob("*.conus.nc") if year_pattern.search(f.name)
    )

    if not nc_files:
        raise FileNotFoundError(
            f"No .conus.nc files for year {year} found in {source_dir}. "
            f"Run 'nhf-targets fetch {source_key.replace('_', '-')}' first."
        )

    logger.info(
        "Merging %d MOD10C1 files for %s year %d", len(nc_files), source_key, year
    )

    opened = _open_datasets(nc_files, f"Reading {source_key} {year} files")
    try:
        datasets = [
            ds.expand_dims(time=[_time_from_modis_filename(f)])
            for ds, f in zip(opened, nc_files)
        ]

        ds_merged = xr.concat(
            datasets, dim="time", data_vars="minimal", coords="minimal"
        )
        ds_merged = ds_merged.sortby("time")
        _validate_variables(ds_merged, variables)
        ds_merged = ds_merged[variables]

        out_path = source_dir / f"{source_key}_{year}_consolidated.nc"
        logger.info("Writing consolidated file: %s", out_path)
        _write_netcdf(ds_merged, out_path)
        logger.info("Wrote %s", out_path)
    finally:
        for d in opened:
            d.close()

    return {
        "consolidated_nc": str(out_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(nc_files),
        "variables": variables,
    }


def _mosaic_and_reproject_timestep(
    tile_paths: list[Path],
    variable: str,
    bbox: tuple[float, float, float, float],
    resolution: float = 0.04,
) -> xr.DataArray:
    """Mosaic tiles for one variable/timestep, reproject to EPSG:4326, clip to bbox.

    Parameters
    ----------
    tile_paths : list[Path]
        HDF tile files (sinusoidal CRS) for a single timestep.
    variable : str
        Variable name to extract from each tile.
    bbox : tuple
        ``(minx, miny, maxx, maxy)`` in EPSG:4326 degrees.
    resolution : float
        Output resolution in degrees (default 0.04).

    Returns
    -------
    xr.DataArray
        Loaded, numpy-backed DataArray clipped to *bbox* in EPSG:4326.
    """
    import rioxarray  # noqa: F401
    from rasterio.enums import Resampling
    from rioxarray.merge import merge_arrays

    if "QC" in variable or "qa" in variable.lower():
        resampling = Resampling.nearest
    else:
        resampling = Resampling.average

    arrays = []
    for p in tile_paths:
        try:
            result = rioxarray.open_rasterio(p, variable=variable, masked=True)
        except TypeError as exc:
            if "variable" not in str(exc) and "unexpected keyword" not in str(exc):
                raise  # Re-raise TypeErrors unrelated to the variable= kwarg
            logger.warning(
                "variable= kwarg not supported for %s; "
                "opening without subdataset selection: %s",
                p.name,
                exc,
            )
            result = rioxarray.open_rasterio(p, masked=True)
        # open_rasterio may return Dataset (HDF4), list, or DataArray.
        # Load eagerly to detach from rasterio/GDAL file handles.
        if isinstance(result, xr.Dataset):
            da = result[variable].load()
            result.close()
        elif isinstance(result, list):
            if len(result) > 1:
                logger.warning(
                    "open_rasterio returned %d subdatasets for %s; "
                    "using first subdataset only",
                    len(result),
                    p.name,
                )
            da = result[0].load()
            for item in result:
                item.close()
        else:
            da = result.load()
            result.close()
        arrays.append(da)

    try:
        if len(arrays) == 1:
            mosaic = arrays[0]
        else:
            mosaic = merge_arrays(arrays)

        reprojected = mosaic.rio.reproject(
            "EPSG:4326",
            resolution=resolution,
            resampling=resampling,
        )

        # Clip to bbox in the output CRS (EPSG:4326).
        minx, miny, maxx, maxy = bbox
        clipped = reprojected.rio.clip_box(
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            crs="EPSG:4326",
        )

        # Load into memory and return — caller owns the result.
        result_da = clipped.load()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to mosaic/reproject/clip {len(arrays)} tiles for "
            f"variable '{variable}'. "
            f"Tiles: {[p.name for p in tile_paths]}. Detail: {exc}"
        ) from exc
    finally:
        for a in arrays:
            a.close()
        del arrays

    return result_da


def consolidate_mod16a2_timestep(
    tile_paths: list[Path],
    variables: list[str],
    source_dir: Path,
    ydoy: str,
    bbox: tuple[float, float, float, float],
    resolution: float = 0.04,
) -> Path:
    """Mosaic, reproject, and clip tiles for one timestep to a temp NetCDF.

    Parameters
    ----------
    tile_paths : list[Path]
        HDF tile files for a single MODIS timestep.
    variables : list[str]
        Variable names to extract.
    source_dir : Path
        Directory to write the temp file into.
    ydoy : str
        Seven-digit YYYYDDD token (e.g. "2010001").
    bbox : tuple
        ``(minx, miny, maxx, maxy)`` in EPSG:4326 degrees for clipping.
    resolution : float
        Output resolution in degrees (default 0.04).

    Returns
    -------
    Path
        Path to the written temp NetCDF file.
    """
    timestamp = _time_from_modis_filename(tile_paths[0])

    var_arrays: dict[str, xr.DataArray] = {}
    try:
        for var in variables:
            da = _mosaic_and_reproject_timestep(tile_paths, var, bbox, resolution)
            if "band" in da.dims:
                da = da.squeeze("band", drop=True)
            rename_map = {}
            if "y" in da.dims:
                rename_map["y"] = "lat"
            if "x" in da.dims:
                rename_map["x"] = "lon"
            if rename_map:
                da = da.rename(rename_map)
            var_arrays[var] = da
    except Exception:
        for da in var_arrays.values():
            da.close()
        raise

    ds_step = xr.Dataset(var_arrays)
    ds_step = ds_step.expand_dims(time=[timestamp])

    tmp_path = source_dir / f"_tmp_{os.getpid()}_A{ydoy}.nc"
    try:
        ds_step.to_netcdf(tmp_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Failed to write temp file for timestep A{ydoy}. Detail: {exc}"
        ) from exc
    finally:
        ds_step.close()
        for da in var_arrays.values():
            da.close()
        del ds_step, var_arrays

    logger.info("Wrote temp file: %s", tmp_path.name)
    return tmp_path


def consolidate_mod16a2_finalize(
    tmp_paths: list[Path],
    variables: list[str],
    out_path: Path,
    run_dir: Path,
) -> dict:
    """Lazy-concat per-timestep temp files into the final consolidated NetCDF.

    Parameters
    ----------
    tmp_paths : list[Path]
        Temp NetCDF files produced by ``consolidate_mod16a2_timestep``.
    variables : list[str]
        Variable names to validate.
    out_path : Path
        Path for the final consolidated file.
    run_dir : Path
        Run workspace root (for computing relative paths in provenance).

    Returns
    -------
    dict
        Provenance record.
    """

    def _cleanup_temps() -> None:
        for p in tmp_paths:
            if p.exists():
                p.unlink()
                logger.debug("Removed temp file: %s", p.name)

    if not tmp_paths:
        raise ValueError(
            "No temp files to finalize. This indicates no timesteps were "
            "successfully processed. Check earlier log messages for errors."
        )

    logger.info(
        "Writing final consolidated file from %d timestep files", len(tmp_paths)
    )

    try:
        ds = xr.open_mfdataset(
            [str(p) for p in tmp_paths],
            combine="nested",
            concat_dim="time",
            chunks={},
            data_vars="all",
            join="outer",
        )
        try:
            ds = ds.sortby("time")
            _validate_variables(ds, variables)
            ds = ds[variables]
            _write_netcdf(ds, out_path)
        finally:
            ds.close()
        logger.info("Wrote %s", out_path)
    except RuntimeError:
        _cleanup_temps()
        raise
    except Exception as exc:
        _cleanup_temps()
        raise RuntimeError(
            f"Failed to finalize consolidated file {out_path}. "
            f"Temp files cleaned up. Detail: {exc}"
        ) from exc

    _cleanup_temps()
    log_memory(f"after writing {out_path.name}")

    return {
        "consolidated_nc": str(out_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(tmp_paths),
        "variables": variables,
    }


def consolidate_mod16a2(
    run_dir: Path,
    source_key: str,
    variables: list[str],
    year: int,
    bbox: tuple[float, float, float, float],
    resolution: float = 0.04,
) -> dict:
    """Merge per-granule MOD16A2 HDF files into a consolidated NetCDF.

    Convenience wrapper: groups tiles by timestep, calls
    ``consolidate_mod16a2_timestep`` for each, then
    ``consolidate_mod16a2_finalize`` to produce the final file.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory.
    source_key : str
        Source key (e.g. ``"mod16a2_v061"``).
    variables : list[str]
        Variable names to include.
    year : int
        Year to consolidate.
    bbox : tuple
        ``(minx, miny, maxx, maxy)`` in EPSG:4326 degrees for clipping.
    resolution : float
        Output resolution in degrees (default 0.04 ≈ 4 km).

    Returns
    -------
    dict
        Provenance record.
    """
    from collections import defaultdict

    source_dir = run_dir / "data" / "raw" / source_key
    year_pattern = re.compile(rf"\.A({year}\d{{3}})\.")
    hdf_files = sorted(
        f for f in source_dir.glob("*.hdf") if year_pattern.search(f.name)
    )

    if not hdf_files:
        raise FileNotFoundError(
            f"No .hdf files for year {year} found in {source_dir}. "
            f"Run 'nhf-targets fetch {source_key.replace('_', '-')}' first."
        )

    # Clean up any stale temp files from prior interrupted runs
    for stale in source_dir.glob("_tmp_*_*.nc"):
        try:
            stale.unlink()
            logger.warning("Removed stale temp file: %s", stale.name)
        except OSError as exc:
            logger.warning("Could not remove stale temp file %s: %s", stale.name, exc)

    # Group tiles by timestep (AYYYYDDD token)
    timestep_groups: dict[str, list[Path]] = defaultdict(list)
    for f in hdf_files:
        m = year_pattern.search(f.name)
        if m:
            timestep_groups[m.group(1)].append(f)

    logger.info(
        "Processing %d HDF files across %d time steps for %s year %d",
        len(hdf_files),
        len(timestep_groups),
        source_key,
        year,
    )

    sorted_ydoys = sorted(timestep_groups)
    n_steps = len(sorted_ydoys)
    tmp_paths: list[Path] = []

    for i, ydoy in enumerate(
        tqdm(sorted_ydoys, desc=f"Mosaicking {source_key} {year}"), 1
    ):
        tile_paths = timestep_groups[ydoy]
        logger.info(
            "Consolidating timestep %d/%d (A%s): %d tiles",
            i,
            n_steps,
            ydoy,
            len(tile_paths),
        )
        tmp_path = consolidate_mod16a2_timestep(
            tile_paths=tile_paths,
            variables=variables,
            source_dir=source_dir,
            ydoy=ydoy,
            bbox=bbox,
            resolution=resolution,
        )
        tmp_paths.append(tmp_path)
        gc.collect()
        log_memory(f"after timestep {i}/{n_steps} (A{ydoy})")

    out_path = source_dir / f"{source_key}_{year}_consolidated.nc"
    result = consolidate_mod16a2_finalize(
        tmp_paths=tmp_paths,
        variables=variables,
        out_path=out_path,
        run_dir=run_dir,
    )
    # Override n_files with HDF count (finalize reports timestep count)
    result["n_files"] = len(hdf_files)
    return result


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

    groups: dict[tuple[str, ...], list[Path]] = defaultdict(list)
    for f in nc_files:
        try:
            ds_peek = xr.open_dataset(f, chunks={})
        except Exception as exc:
            raise RuntimeError(
                f"Cannot inspect {f.name} for variable grouping. "
                f"The file may be corrupt. Detail: {exc}"
            ) from exc
        data_vars = [v for v in ds_peek.data_vars if v != "time_bnds"]
        key = tuple(sorted(data_vars))
        groups[key].append(f)
        ds_peek.close()

    # Warn about unexpected variable groups
    expected_vars = set(variables)
    for key in groups:
        if not set(key) & expected_vars:
            logger.warning(
                "Found files with unexpected variables %s in %s; "
                "these will be included in merge but filtered out during "
                "variable selection.",
                key,
                ncep_dir,
            )

    merged_parts: list[xr.Dataset] = []
    try:
        for var_key, file_group in groups.items():
            datasets = _open_datasets(
                file_group, f"Reading NCEP/NCAR {', '.join(var_key)}"
            )
            try:
                part = xr.concat(
                    datasets, dim="time", data_vars="minimal", coords="minimal"
                )
                part = part.sortby("time")
                merged_parts.append(part)
            finally:
                for d in datasets:
                    d.close()

        ds = xr.merge(merged_parts)
        _validate_variables(ds, variables)
        ds = ds[variables]

        out_path = ncep_dir / "ncep_ncar_consolidated.nc"
        logger.info("Writing consolidated file: %s", out_path)
        _write_netcdf(ds, out_path)
        logger.info("Wrote %s", out_path)
    finally:
        for part in merged_parts:
            part.close()

    return {
        "consolidated_nc": str(out_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(nc_files),
        "variables": variables,
    }
