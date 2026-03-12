"""Build Kerchunk virtual Zarr reference stores for fetch modules."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import ujson

from nhf_spatial_targets import __version__

logger = logging.getLogger(__name__)

# Variables to keep in addition to user-requested ones.
_COORD_VARS = {"time", "lat", "lon", "time_bnds", "lat_bnds", "lon_bnds"}


def _filter_refs(refs: dict, keep_vars: set[str]) -> dict:
    """Remove variable keys from a kerchunk reference dict.

    Keeps:
    - Top-level metadata keys (no "/" in key): .zattrs, .zgroup
    - Coordinate variables: time, lat, lon (and any *_bnds)
    - Only the data variables listed in keep_vars

    Drops all other variable keys (e.g., "SFMC/.zarray", "SFMC/0.0.0").
    """
    all_keep = keep_vars | _COORD_VARS
    filtered = {}
    for key in refs:
        if "/" not in key:
            filtered[key] = refs[key]
            continue
        var_name = key.split("/")[0]
        if var_name in all_keep:
            filtered[key] = refs[key]
    return filtered


def _fix_time(combined: dict) -> dict:
    """Shift timestamps to mid-month and add time_bnds.

    Operates directly on the kerchunk reference dict, replacing the time
    coordinate data with mid-month values and adding time_bnds. Uses
    zarr v2 inline base64-encoded chunks compatible with kerchunk references.
    """
    import fsspec
    import xarray as xr

    refs = combined["refs"]

    # Open the combined refs to get the original time values
    fs = fsspec.filesystem(
        "reference",
        fo=combined,
        target_protocol="file",
        remote_protocol="file",
    )
    ds = xr.open_zarr(fs.get_mapper(""), consolidated=False)
    original_times = pd.DatetimeIndex(ds.time.values)
    n_times = len(original_times)
    ds.close()

    epoch = pd.Timestamp("1970-01-01")

    # Build mid-month timestamps (day 15, midnight)
    mid_month = pd.DatetimeIndex(
        [
            t.replace(day=15, hour=0, minute=0, second=0, microsecond=0)
            for t in original_times
        ]
    )
    days_mid = np.array([(t - epoch).days for t in mid_month], dtype="<i8")

    # Update time chunk data (inline base64)
    refs["time/0"] = "base64:" + base64.b64encode(days_mid.tobytes()).decode()

    # Update time .zarray shape to match actual number of time steps
    time_zarr = json.loads(refs["time/.zarray"])
    time_zarr["shape"] = [n_times]
    time_zarr["chunks"] = [n_times]
    refs["time/.zarray"] = json.dumps(time_zarr)

    # Update time .zattrs: use epoch-based units, add bounds and CF attrs
    time_attrs = json.loads(refs["time/.zattrs"])
    time_attrs["units"] = "days since 1970-01-01"
    time_attrs["calendar"] = "standard"
    time_attrs["bounds"] = "time_bnds"
    time_attrs["cell_methods"] = "time: mean"
    refs["time/.zattrs"] = json.dumps(time_attrs)

    # Build time_bnds: [first-of-month, first-of-next-month]
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

    bnds_arr = np.array(bounds_list, dtype="<i8")

    refs["time_bnds/.zarray"] = json.dumps(
        {
            "shape": [n_times, 2],
            "chunks": [n_times, 2],
            "dtype": "<i8",
            "fill_value": 0,
            "order": "C",
            "filters": None,
            "compressor": None,
            "zarr_format": 2,
        }
    )
    refs["time_bnds/.zattrs"] = json.dumps(
        {
            "_ARRAY_DIMENSIONS": ["time", "nv"],
            "units": "days since 1970-01-01",
            "calendar": "standard",
        }
    )
    refs["time_bnds/0.0"] = "base64:" + base64.b64encode(bnds_arr.tobytes()).decode()

    # Add nv dimension
    nv_arr = np.array([0, 1], dtype="<i8")
    refs["nv/.zarray"] = json.dumps(
        {
            "shape": [2],
            "chunks": [2],
            "dtype": "<i8",
            "fill_value": 0,
            "order": "C",
            "filters": None,
            "compressor": None,
            "zarr_format": 2,
        }
    )
    refs["nv/.zattrs"] = json.dumps({"_ARRAY_DIMENSIONS": ["nv"]})
    refs["nv/0"] = "base64:" + base64.b64encode(nv_arr.tobytes()).decode()

    return combined


def _make_relative(refs: dict, base_dir: Path) -> dict:
    """Convert absolute file paths in kerchunk refs to relative paths."""
    base_prefix = str(base_dir) + "/"
    out = {}
    for key, val in refs.items():
        if isinstance(val, list) and len(val) >= 1 and isinstance(val[0], str):
            path = val[0]
            if path.startswith(base_prefix):
                rel = "./" + path[len(base_prefix) :]
                val = [rel] + val[1:]
        out[key] = val
    return out


def consolidate_merra2(
    run_dir: Path,
    variables: list[str],
) -> dict:
    """Build a Kerchunk JSON reference store for MERRA-2 files.

    Always performs a full rebuild from all .nc4 files on disk.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory containing ``data/raw/merra2/*.nc4``.
    variables : list[str]
        Variable names to include (e.g. ["GWETTOP", "GWETROOT", "GWETPROF"]).

    Returns
    -------
    dict
        Provenance record with reference file path and timestamp.
    """
    from datetime import datetime, timezone

    import kerchunk.hdf
    from kerchunk.combine import MultiZarrToZarr

    import nhf_spatial_targets.catalog as _catalog

    merra2_dir = run_dir / "data" / "raw" / "merra2"
    nc_files = sorted(merra2_dir.glob("*.nc4"))

    if not nc_files:
        raise FileNotFoundError(
            f"No .nc4 files found in {merra2_dir}. "
            f"Run 'nhf-targets fetch merra2' first."
        )

    logger.info("Scanning %d NetCDF files for Kerchunk references", len(nc_files))

    keep_vars = set(variables)
    singles = []
    for nc in nc_files:
        with open(nc, "rb") as f:
            h5chunks = kerchunk.hdf.SingleHdf5ToZarr(f, str(nc))
            refs = h5chunks.translate()
        refs["refs"] = _filter_refs(refs["refs"], keep_vars)
        singles.append(refs)

    mzz = MultiZarrToZarr(
        singles,
        concat_dims=["time"],
        identical_dims=["lat", "lon"],
        coo_map={"time": "cf:time"},
    )
    combined = mzz.translate()

    combined = _fix_time(combined)

    # Add CF and provenance global attributes
    meta = _catalog.source("merra2")
    root_attrs = ujson.loads(combined["refs"].get(".zattrs", "{}"))
    root_attrs.update(
        {
            "Conventions": "CF-1.8",
            "history": (
                f"Kerchunk virtual Zarr created by nhf-spatial-targets v{__version__}"
            ),
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
    combined["refs"][".zattrs"] = ujson.dumps(root_attrs)

    # Make paths relative to merra2_dir
    combined["refs"] = _make_relative(combined["refs"], merra2_dir)

    ref_path = merra2_dir / "merra2_refs.json"
    ref_path.write_text(ujson.dumps(combined, indent=2))
    logger.info("Wrote Kerchunk reference store: %s", ref_path)

    return {
        "kerchunk_ref": str(ref_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(nc_files),
        "variables": variables,
    }


def consolidate_nldas(
    run_dir: Path,
    source_key: str,
    variables: list[str],
) -> dict:
    """Build a Kerchunk JSON reference store for NLDAS files.

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
    from datetime import datetime, timezone

    import kerchunk.hdf
    from kerchunk.combine import MultiZarrToZarr

    source_dir = run_dir / "data" / "raw" / source_key
    nc_files = sorted(list(source_dir.glob("*.nc4")) + list(source_dir.glob("*.nc")))

    if not nc_files:
        raise FileNotFoundError(
            f"No NetCDF files found in {source_dir}. "
            f"Run 'nhf-targets fetch {source_key.replace('_', '-')}' first."
        )

    logger.info("Scanning %d NetCDF files for %s", len(nc_files), source_key)

    keep_vars = set(variables)
    singles = []
    for nc in nc_files:
        with open(nc, "rb") as f:
            h5chunks = kerchunk.hdf.SingleHdf5ToZarr(f, str(nc))
            refs = h5chunks.translate()
        refs["refs"] = _filter_refs(refs["refs"], keep_vars)
        singles.append(refs)

    mzz = MultiZarrToZarr(
        singles,
        concat_dims=["time"],
        identical_dims=["lat", "lon"],
        coo_map={"time": "cf:time"},
    )
    combined = mzz.translate()

    combined["refs"] = _make_relative(combined["refs"], source_dir)

    ref_path = source_dir / f"{source_key}_refs.json"
    ref_path.write_text(ujson.dumps(combined, indent=2))
    logger.info("Wrote Kerchunk reference store: %s", ref_path)

    return {
        "kerchunk_ref": str(ref_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(nc_files),
        "variables": variables,
    }
