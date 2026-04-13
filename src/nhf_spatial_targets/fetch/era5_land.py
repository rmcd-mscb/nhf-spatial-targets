"""Fetch ERA5-Land hourly runoff from Copernicus CDS.

Downloads hourly accumulated runoff variables (ro, sro, ssro) for the
CONUS+contributing-watersheds bbox, then aggregates hourly→daily and
daily→monthly. Both daily and monthly consolidated NetCDFs are written
to the shared datastore.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xarray as xr

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets import __version__
from nhf_spatial_targets.fetch._period import years_in_period
from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata
from nhf_spatial_targets.workspace import load as _load_project

logger = logging.getLogger(__name__)

# CDS area parameter [N, W, S, E], snapped to ERA5-Land 0.1° grid.
# Encompasses CONUS contributing watersheds (Canada/Mexico) + ~10 km buffer.
BBOX_NWSE = [53.0, -125.0, 24.7, -66.0]

VARIABLES = ("ro", "sro", "ssro")


def hourly_to_daily(da: xr.DataArray) -> xr.DataArray:
    """Aggregate ERA5-Land hourly accumulated runoff to daily totals.

    ERA5-Land accumulated fields (ro, sro, ssro) represent meters of water
    equivalent accumulated since 00 UTC of the current day, and reset to 0
    at the start of each new day. The conversion proceeds in four steps:

    1. Compute hourly increments via ``.diff(time, label="upper")``.
       Each diff value is the accumulation during the preceding hour.
    2. At the 00 UTC reset, the diff is negative (the accumulation jumps from
       the previous day's total back toward 0). Where the diff is negative,
       substitute the raw accumulated value at that timestamp — which equals
       the 23→00 UTC increment because accumulation restarted at 0 at 00 UTC.
    3. Shift all timestamps back 1 hour so that the 00 UTC increment
       (representing the 23→00 accumulation) is credited to the prior day
       rather than the new day.
    4. Resample to daily sums, which correctly accumulates all hourly
       increments within each calendar day.

    Parameters
    ----------
    da : xr.DataArray
        Hourly accumulated runoff with a 'time' dimension. Time stamps
        must be regular hourly.

    Returns
    -------
    xr.DataArray
        Daily-summed runoff. Time coordinate is the date (00:00) of each
        complete day. Original attrs are preserved.
    """
    incr = da.diff("time", label="upper")
    # Negative jumps occur at the 00 UTC reset; where the diff is negative,
    # replace it with the raw value at that timestamp (which equals the
    # accumulation since the midnight reset, i.e. the true hourly increment).
    incr = xr.where(incr >= 0, incr, da.isel(time=slice(1, None)))
    # Shift the timestamp back by 1 hour so that the 00 UTC increment
    # (which is the 23->00 accumulation) lands inside the prior day.
    incr = incr.assign_coords(time=incr.time - pd.Timedelta(hours=1))
    daily = incr.resample(time="1D").sum()
    daily.attrs = dict(da.attrs)
    return daily


def daily_to_monthly(da: xr.DataArray) -> xr.DataArray:
    """Sum daily totals to monthly totals.

    Uses month-end frequency ('1ME') so the time coordinate marks the
    last day of each month — consistent with other monthly products in
    this codebase. Original attrs are preserved.

    Note: called by ``consolidate_year``, which also deletes any stale
    monthly NCs whose filenames fall outside the updated year range.
    """
    monthly = da.resample(time="1ME").sum()
    monthly.attrs = dict(da.attrs)
    return monthly


# Map short variable name to the CDS request name
_VARIABLE_REQUEST_NAME = {
    "ro": "runoff",
    "sro": "surface_runoff",
    "ssro": "sub_surface_runoff",
}


def _cds_client():
    """Construct a cdsapi.Client. Separated for test injection."""
    import cdsapi

    return cdsapi.Client()


def download_year_variable(
    year: int,
    variable: str,
    output_path: Path,
) -> Path:
    """Download one year of one ERA5-Land variable to ``output_path``.

    Idempotent: if ``output_path`` already exists, returns immediately.
    Submits a single CDS request covering all 12 months × all hours of
    the given year, clipped to ``BBOX_NWSE``.

    Parameters
    ----------
    year : int
    variable : {"ro", "sro", "ssro"}
    output_path : Path
        Target NetCDF file. Parent directory is created if missing.

    Returns
    -------
    Path
        ``output_path`` (for caller convenience).
    """
    if variable not in _VARIABLE_REQUEST_NAME:
        raise ValueError(f"Unknown ERA5-Land variable: {variable!r}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.info("Skipping existing ERA5-Land file: %s", output_path)
        return output_path

    request = {
        "variable": _VARIABLE_REQUEST_NAME[variable],
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": BBOX_NWSE,
        "format": "netcdf",
    }
    client = _cds_client()
    logger.info("Submitting CDS request for %s %d → %s", variable, year, output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        client.retrieve("reanalysis-era5-land", request, str(tmp_path))
        tmp_path.rename(output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return output_path


_SOURCE_KEY = "era5_land"


def _atomic_to_netcdf(ds: xr.Dataset, path: Path) -> None:
    """Write *ds* to *path* atomically via a temporary file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        ds.to_netcdf(tmp, format="NETCDF4")
        tmp.rename(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def consolidate_year(
    year: int,
    hourly_dir: Path,
    daily_dir: Path,
    monthly_dir: Path,
) -> tuple[Path, Path]:
    """Build daily and monthly consolidated NCs for one year.

    Reads ``era5_land_{ro,sro,ssro}_{year}.nc`` from ``hourly_dir``,
    aggregates each variable hourly→daily, merges into a single daily
    dataset, applies CF metadata, and writes atomically. Then aggregates
    all available daily files into a rolling monthly NC.

    **Destructive side effect:** any existing monthly NC in ``monthly_dir``
    whose filename year range differs from the updated range is deleted
    before the new monthly NC is written.

    Parameters
    ----------
    year : int
    hourly_dir : Path
        Directory containing per-variable hourly NCs.
    daily_dir : Path
        Directory to write the daily consolidated NC.
    monthly_dir : Path
        Directory to write (or overwrite) the monthly consolidated NC.

    Returns
    -------
    tuple[Path, Path]
        ``(daily_path, monthly_path)``
    """
    daily_dir = Path(daily_dir)
    monthly_dir = Path(monthly_dir)
    daily_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir.mkdir(parents=True, exist_ok=True)

    daily_path = daily_dir / f"era5_land_daily_{year}.nc"

    # Idempotency guard: skip hourly→daily aggregation if the daily NC is
    # up-to-date.  "Up-to-date" means the daily NC exists AND either:
    #   (a) no hourly input files are present (daily NC is authoritative), or
    #   (b) the daily NC mtime >= the maximum mtime of existing hourly files.
    # If any hourly file is newer than the daily NC, we re-aggregate so that
    # upstream changes to the raw downloads are picked up automatically.
    hourly_paths = [
        Path(hourly_dir) / f"era5_land_{var}_{year}.nc" for var in VARIABLES
    ]
    _skip_aggregation = False
    if daily_path.exists():
        daily_mtime = daily_path.stat().st_mtime
        hourly_mtimes = [p.stat().st_mtime for p in hourly_paths if p.exists()]
        if not hourly_mtimes or max(hourly_mtimes) <= daily_mtime:
            logger.info(
                "Daily NC is up-to-date (no newer hourly inputs), "
                "skipping aggregation: %s",
                daily_path,
            )
            _skip_aggregation = True
        else:
            logger.info(
                "Hourly inputs are newer than daily NC; re-aggregating: %s",
                daily_path,
            )

    if _skip_aggregation:
        pass
    else:
        daily_arrays: dict[str, xr.DataArray] = {}
        for var, path in zip(VARIABLES, hourly_paths):
            if not path.exists():
                raise FileNotFoundError(f"Missing hourly file: {path}")
            ds = xr.open_dataset(path)
            da = ds[var].load()
            ds.close()
            daily_arrays[var] = hourly_to_daily(da)

        daily_ds = xr.Dataset(daily_arrays)
        daily_ds = apply_cf_metadata(daily_ds, _SOURCE_KEY, "daily")
        daily_ds.attrs.update(
            {
                "title": f"ERA5-Land daily runoff (CONUS+ buffered) {year}",
                "institution": "ECMWF",
                "source": "reanalysis-era5-land",
                "references": "doi:10.5194/essd-13-4349-2021",
                "frequency": "day",
                "history": f"Consolidated by nhf-spatial-targets v{__version__}",
            }
        )
        _atomic_to_netcdf(daily_ds, daily_path)
        logger.info("Wrote daily NC: %s", daily_path)

    # Monthly: rebuild from the (possibly multi-year) collection of daily files
    daily_files = sorted(daily_dir.glob("era5_land_daily_*.nc"))
    with xr.open_mfdataset(daily_files, combine="by_coords") as ds_all:
        monthly_arrays: dict[str, xr.DataArray] = {}
        for var in VARIABLES:
            monthly_arrays[var] = daily_to_monthly(ds_all[var].load())
    monthly_ds = xr.Dataset(monthly_arrays)
    monthly_ds = apply_cf_metadata(monthly_ds, _SOURCE_KEY, "monthly")
    start_year = pd.Timestamp(monthly_ds.time.min().values).year
    end_year = pd.Timestamp(monthly_ds.time.max().values).year
    monthly_ds.attrs.update(
        {
            "title": (
                f"ERA5-Land monthly runoff (CONUS+ buffered) {start_year}–{end_year}"
            ),
            "institution": "ECMWF",
            "source": "reanalysis-era5-land",
            "references": "doi:10.5194/essd-13-4349-2021",
            "frequency": "month",
            "history": f"Consolidated by nhf-spatial-targets v{__version__}",
        }
    )
    monthly_path = monthly_dir / f"era5_land_monthly_{start_year}_{end_year}.nc"
    # Remove any stale monthly file with a different year range
    for stale in monthly_dir.glob("era5_land_monthly_*.nc"):
        if stale != monthly_path:
            stale.unlink()
    _atomic_to_netcdf(monthly_ds, monthly_path)
    logger.info("Wrote monthly NC: %s", monthly_path)

    return daily_path, monthly_path


def fetch_era5_land(workdir: Path, period: str) -> dict:
    """Download ERA5-Land hourly runoff and produce daily/monthly NCs.

    Loops over years in ``period``, downloading per-year per-variable
    hourly NCs from CDS into the project's datastore, then consolidating
    each year into the daily file and rebuilding the rolling monthly
    file. Idempotent on already-downloaded years.

    Parameters
    ----------
    workdir : Path
        Project directory containing ``config.yml`` and ``fabric.json``.
    period : str
        Temporal range as ``"YYYY/YYYY"``.

    Returns
    -------
    dict
        Provenance record for the caller.
    """
    ws = _load_project(workdir)
    meta = _catalog.source(_SOURCE_KEY)

    raw_root = ws.raw_dir(_SOURCE_KEY)
    hourly_dir = raw_root / "hourly"
    daily_dir = raw_root / "daily"
    monthly_dir = raw_root / "monthly"

    now_utc = datetime.now(timezone.utc).isoformat()
    files: list[dict] = []

    for year in years_in_period(period):
        for var in VARIABLES:
            out = hourly_dir / f"era5_land_{var}_{year}.nc"
            download_year_variable(year, var, out)
        daily_path, monthly_path = consolidate_year(
            year, hourly_dir, daily_dir, monthly_dir
        )
        files.append(
            {
                "year": year,
                "daily_path": str(daily_path),
                "monthly_path": str(monthly_path),
                "consolidated_utc": now_utc,
            }
        )

    bbox = ws.fabric["bbox_buffered"]
    license_str = meta.get("license", "Copernicus license")
    _update_manifest(workdir, period, bbox, meta, license_str, files)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "license": license_str,
        "variables": [v["name"] for v in meta["variables"]],
        "period": period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "files": files,
    }


def _update_manifest(
    workdir: Path,
    period: str,
    bbox: dict,
    meta: dict,
    license_str: str,
    files: list[dict],
) -> None:
    """Merge ERA5-Land provenance into manifest.json (atomic write)."""
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {workdir} is corrupt and cannot be "
                f"parsed. You may need to delete it and re-run the fetch "
                f"step. Original error: {exc}"
            ) from exc
    else:
        manifest = {"sources": {}, "steps": []}

    manifest.setdefault("sources", {})[_SOURCE_KEY] = {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "license": license_str,
        "period": period,
        "bbox": bbox,
        "variables": [v["name"] for v in meta["variables"]],
        "files": files,
    }

    fd, tmp = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp).replace(manifest_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    logger.info("Updated manifest.json with ERA5-Land provenance")
