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
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl as _fcntl

    _HAVE_FLOCK = True
except ImportError:  # Windows (not used on HPC, but keeps unit tests portable)
    _HAVE_FLOCK = False

import pandas as pd
import xarray as xr

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets import __version__
from nhf_spatial_targets.fetch._period import years_in_period
from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata, resolve_license
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


def download_month_variable(
    year: int,
    month: int,
    variable: str,
    output_path: Path,
) -> Path:
    """Download one month of one ERA5-Land variable to ``output_path``.

    Idempotent: if ``output_path`` already exists, returns immediately.
    Submits a single CDS request covering all days × all hours of the
    given year-month, clipped to ``BBOX_NWSE``.

    Monthly requests (~100 MB per variable-month for the CONUS+buffered
    bbox) stay comfortably within the CDS per-request cost limit. Called
    by ``download_year_variable``. See README "Datastore Storage
    Estimates" for the authoritative per-period totals.

    Parameters
    ----------
    year : int
    month : int
        Calendar month (1–12).
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
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": BBOX_NWSE,
        "format": "netcdf",
    }
    client = _cds_client()
    logger.info(
        "Submitting CDS request for %s %d-%02d → %s",
        variable,
        year,
        month,
        output_path,
    )
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        client.retrieve("reanalysis-era5-land", request, str(tmp_path))
        # CDS API (cdsapi ≥0.7.7) wraps the NetCDF in a zip archive.
        # Detect and extract the .nc member before renaming to output_path.
        if zipfile.is_zipfile(tmp_path):
            with zipfile.ZipFile(tmp_path) as zf:
                nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                if not nc_names:
                    raise ValueError(
                        f"No .nc file found in CDS zip: {list(zf.namelist())}"
                    )
                zf.extract(nc_names[0], path=tmp_path.parent)
                extracted = tmp_path.parent / nc_names[0]
                tmp_path.unlink()
                extracted.rename(tmp_path)
        tmp_path.rename(output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return output_path


def download_year_variable(
    year: int,
    variable: str,
    output_path: Path,
) -> Path:
    """Download one year of one ERA5-Land variable to ``output_path``.

    Splits the annual download into 12 monthly CDS requests to stay within
    the CDS per-request cost/size limit (~100 MB per variable-month, i.e.
    ~1.2 GB per variable-year, for the CONUS+buffered bbox at hourly
    resolution). Monthly chunk files are written alongside ``output_path``
    as::

        era5_land_{variable}_{year}_{month:02d}.nc

    and kept on disk for re-run idempotency. Once all 12 chunks are present
    they are concatenated along the time axis into ``output_path``.

    Idempotent at two levels:
    - If ``output_path`` (year file) exists and is not older than any monthly
      chunk, the function returns immediately without any CDS calls.
    - If an individual monthly chunk already exists, that month's CDS request
      is skipped.

    Parameters
    ----------
    year : int
    variable : {"ro", "sro", "ssro"}
    output_path : Path
        Target per-year NetCDF file. Parent directory is created if missing.

    Returns
    -------
    Path
        ``output_path`` (for caller convenience).
    """
    if variable not in _VARIABLE_REQUEST_NAME:
        raise ValueError(f"Unknown ERA5-Land variable: {variable!r}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Monthly chunk paths live in the same directory as the year file.
    stem = output_path.stem  # e.g. "era5_land_ro_2020"
    chunk_paths = [
        output_path.parent / f"{stem}_{month:02d}.nc" for month in range(1, 13)
    ]

    # Fast path: year file exists, all 12 chunks present, and no chunk newer.
    # Requiring all 12 chunks on disk guards against the case where the year
    # file was written from a partial set (e.g. 6/12) and later chunks arrive
    # with an older mtime (rsync -a, restore-from-backup) that would otherwise
    # bypass the mtime check.
    if output_path.exists():
        existing_chunks = [p for p in chunk_paths if p.exists()]
        if len(existing_chunks) == 12:
            year_mtime = output_path.stat().st_mtime
            chunk_mtimes = [p.stat().st_mtime for p in existing_chunks]
            if max(chunk_mtimes) <= year_mtime:
                logger.info("Skipping up-to-date ERA5-Land year file: %s", output_path)
                return output_path
        else:
            logger.info(
                "Rebuilding %s: year file exists but only %d/12 chunks on disk",
                output_path,
                len(existing_chunks),
            )

    # Download any missing monthly chunks.
    for month, chunk_path in enumerate(chunk_paths, start=1):
        download_month_variable(year, month, variable, chunk_path)

    # Concatenate all 12 monthly chunks into the year file.
    logger.info("Concatenating monthly chunks into year file: %s", output_path)
    with xr.open_mfdataset(chunk_paths, combine="by_coords") as ds:
        year_ds = ds.load()
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        year_ds.to_netcdf(tmp_path, format="NETCDF4")
        tmp_path.rename(output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    logger.info("Wrote year file: %s", output_path)
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
    the year's daily file into a per-year monthly NC
    (``era5_land_monthly_{year}.nc``).

    Both the daily and monthly outputs are idempotent: if they already exist
    and are not older than their inputs, the aggregation step is skipped.

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
            # CDS API ≥0.7 changed the time dimension from "time" to
            # "valid_time"; normalize here so downstream code always sees
            # a "time" dimension. apply_cf_metadata also renames
            # valid_time→time, but hourly_to_daily (below) indexes the
            # "time" dim directly, so we must rename before it runs.
            if "valid_time" in ds.dims:
                ds = ds.rename({"valid_time": "time"})
            if "time" not in ds.dims:
                raise ValueError(
                    f"ERA5-Land hourly file {path} has no 'time' or "
                    f"'valid_time' dim; got dims {list(ds.dims)}. "
                    f"CDS schema may have changed."
                )
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

    # Monthly: per-year NC derived from this year's daily file only.
    # Writing one file per year makes parallel fetching safe (no shared
    # write target) and is directly consumable by the aggregator via its
    # files_glob="era5_land_monthly_*.nc" pattern.
    monthly_path = monthly_dir / f"era5_land_monthly_{year}.nc"

    # Idempotency guard: skip if monthly NC exists and daily is not newer.
    _skip_monthly = False
    if monthly_path.exists():
        monthly_mtime = monthly_path.stat().st_mtime
        daily_mtime_now = daily_path.stat().st_mtime if daily_path.exists() else 0.0
        if daily_mtime_now <= monthly_mtime:
            logger.info(
                "Monthly NC is up-to-date, skipping: %s",
                monthly_path,
            )
            _skip_monthly = True
        else:
            logger.info(
                "Daily NC is newer than monthly NC; re-aggregating: %s",
                monthly_path,
            )

    if not _skip_monthly:
        with xr.open_dataset(daily_path) as ds_year:
            monthly_arrays: dict[str, xr.DataArray] = {}
            for var in VARIABLES:
                monthly_arrays[var] = daily_to_monthly(ds_year[var].load())
        monthly_ds = xr.Dataset(monthly_arrays)
        monthly_ds = apply_cf_metadata(monthly_ds, _SOURCE_KEY, "monthly")
        monthly_ds.attrs.update(
            {
                "title": f"ERA5-Land monthly runoff (CONUS+ buffered) {year}",
                "institution": "ECMWF",
                "source": "reanalysis-era5-land",
                "references": "doi:10.5194/essd-13-4349-2021",
                "frequency": "month",
                "history": f"Consolidated by nhf-spatial-targets v{__version__}",
            }
        )
        _atomic_to_netcdf(monthly_ds, monthly_path)
        logger.info("Wrote monthly NC: %s", monthly_path)

    return daily_path, monthly_path


def _completed_years_from_manifest(workdir: Path) -> set[int]:
    """Return the set of years fully recorded in ``manifest.json``.

    A year is considered complete when its manifest entry carries both
    ``daily_path`` and ``monthly_path`` that exist on disk.  Used by
    ``fetch_era5_land`` to skip years already processed by a prior run or
    a sibling parallel worker.

    Returns an empty set (with a warning) if the manifest is absent or
    unparseable — letting the caller fall through to file-level idempotency.
    """
    manifest_path = workdir / "manifest.json"
    if not manifest_path.exists():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        logger.warning(
            "manifest.json in %s could not be parsed; "
            "assuming no ERA5-Land years completed",
            workdir,
        )
        return set()
    files = manifest.get("sources", {}).get(_SOURCE_KEY, {}).get("files", [])
    completed: set[int] = set()
    for f in files:
        try:
            year = int(f["year"])
        except (KeyError, ValueError, TypeError):
            continue
        daily = f.get("daily_path", "")
        monthly = f.get("monthly_path", "")
        if daily and monthly and Path(daily).exists() and Path(monthly).exists():
            completed.add(year)
    return completed


def fetch_era5_land(
    workdir: Path,
    period: str,
    *,
    worker_index: int = 0,
    n_workers: int = 1,
) -> dict:
    """Download ERA5-Land hourly runoff and produce daily/monthly NCs.

    Reads ``manifest.json`` first to identify years already completed, then
    divides the remaining years across ``n_workers`` parallel processes
    (round-robin by ``worker_index``). Each worker downloads per-year
    per-variable hourly NCs from CDS and consolidates them into a per-year
    daily NC and a per-year monthly NC. Fully idempotent: already-downloaded
    months and completed years are skipped at every level.

    Parameters
    ----------
    workdir : Path
        Project directory containing ``config.yml`` and ``fabric.json``.
    period : str
        Temporal range as ``"YYYY/YYYY"``.
    worker_index : int
        0-based index of this worker within the pool (default ``0``).
    n_workers : int
        Total number of parallel workers (default ``1`` = serial).
        All workers must be given the same ``period`` and ``n_workers``.

    Returns
    -------
    dict
        Provenance record for the caller (includes only this worker's years).
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers!r}")
    if not (0 <= worker_index < n_workers):
        raise ValueError(
            f"worker_index must be in [0, n_workers), "
            f"got worker_index={worker_index!r}, n_workers={n_workers!r}"
        )

    ws = _load_project(workdir)
    meta = _catalog.source(_SOURCE_KEY)

    raw_root = ws.raw_dir(_SOURCE_KEY)
    hourly_dir = raw_root / "hourly"
    daily_dir = raw_root / "daily"
    monthly_dir = raw_root / "monthly"

    now_utc = datetime.now(timezone.utc).isoformat()
    bbox = ws.fabric["bbox_buffered"]
    license_str = resolve_license(meta, _SOURCE_KEY)

    all_years = years_in_period(period)

    # Skip years already fully recorded in the manifest (both daily and monthly
    # output files confirmed present on disk).  This is an optimistic fast-path:
    # file-level idempotency inside download_year_variable / consolidate_year
    # handles partially-complete years correctly regardless.
    completed = _completed_years_from_manifest(workdir)
    remaining = [y for y in all_years if y not in completed]
    if completed:
        skipped = sorted(y for y in all_years if y in completed)
        logger.info(
            "Worker %d/%d: skipping %d manifest-completed year(s): %s",
            worker_index,
            n_workers,
            len(skipped),
            skipped,
        )

    # Assign this worker's slice via round-robin so years spread evenly even
    # when the list length is not divisible by n_workers.
    my_years = remaining[worker_index::n_workers]
    logger.info(
        "Worker %d/%d: assigned %d year(s) to process: %s",
        worker_index,
        n_workers,
        len(my_years),
        my_years,
    )

    files: list[dict] = []

    if not my_years:
        return {
            "source_key": _SOURCE_KEY,
            "access_url": meta["access"]["url"],
            "license": license_str,
            "variables": [v["name"] for v in meta["variables"]],
            "period": period,
            "bbox": bbox,
            "download_timestamp": now_utc,
            "worker_index": worker_index,
            "n_workers": n_workers,
            "files": [],
        }

    try:
        for year in my_years:
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
    except Exception:
        logger.error(
            "ERA5-Land fetch (worker %d/%d) failed after completing %d year(s); "
            "completed chunks preserved on disk, re-run to resume.",
            worker_index,
            n_workers,
            len(files),
            exc_info=True,
        )
        raise
    finally:
        # Persist partial-run provenance even if the loop raised above.
        # Swallow any manifest-write exception so it does not mask the
        # original fetch failure (which is the important signal).
        if files:
            try:
                _update_manifest(workdir, period, bbox, meta, license_str, files)
            except Exception:
                logger.exception(
                    "Failed to persist partial ERA5-Land manifest for "
                    "%d year(s) (worker %d/%d); manifest.json may be stale.",
                    len(files),
                    worker_index,
                    n_workers,
                )

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "license": license_str,
        "variables": [v["name"] for v in meta["variables"]],
        "period": period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "worker_index": worker_index,
        "n_workers": n_workers,
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
    """Merge ERA5-Land provenance into manifest.json (atomic write, file-locked).

    Uses an advisory ``flock(LOCK_EX)`` on a sibling ``.lock`` file so that
    concurrent parallel workers do not race on the shared manifest.  The lock
    is held across the full read-modify-write cycle; it is released
    automatically when the context manager exits (even on exception or kill).
    On platforms without ``fcntl`` (Windows) locking is silently skipped —
    correctness then relies on the atomic rename and incremental merge logic.
    """
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    lock_path = manifest_path.with_suffix(".lock")

    # Open the lock file in append mode (creates it if absent, never truncates).
    # Hold the exclusive lock for the entire read-modify-write so sibling
    # workers see a consistent snapshot.
    with open(lock_path, "a") as _lock_f:
        if _HAVE_FLOCK:
            _fcntl.flock(_lock_f, _fcntl.LOCK_EX)
        # Re-read inside the lock: another worker may have written new years
        # between our last read and now.
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

        manifest.setdefault("sources", {})
        entry = manifest["sources"].get(_SOURCE_KEY, {})

        # Merge files by year so incremental runs accumulate records.
        # Parse defensively: prior manifests may have been hand-edited or
        # written by an older version with a different schema.
        existing_by_year: dict[int, dict] = {}
        for f in entry.get("files", []):
            if "year" not in f:
                logger.warning(
                    "Skipping malformed manifest entry in %s (missing 'year' key): %s",
                    manifest_path,
                    f,
                )
                continue
            try:
                existing_by_year[int(f["year"])] = f
            except (TypeError, ValueError) as exc:
                # Match the sibling "missing year" handler above: skip-and-warn
                # rather than raise. A corrupt prior entry shouldn't block
                # recording of newly fetched years (which is arguably more
                # valuable than failing hard on old cruft).
                logger.warning(
                    "Skipping manifest entry with invalid year %r in %s: %s",
                    f.get("year"),
                    manifest_path,
                    exc,
                )
                continue
        for f in files:
            existing_by_year[int(f["year"])] = f

        merged_files = [existing_by_year[y] for y in sorted(existing_by_year)]

        all_years = sorted(existing_by_year)
        effective_period = f"{all_years[0]}/{all_years[-1]}" if all_years else period

        entry.update(
            {
                "source_key": _SOURCE_KEY,
                "access_url": meta["access"]["url"],
                "license": license_str,
                "period": effective_period,
                "bbox": bbox,
                "variables": [v["name"] for v in meta["variables"]],
                "files": merged_files,
            }
        )
        manifest["sources"][_SOURCE_KEY] = entry

        fd, tmp = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(manifest, f, indent=2)
            Path(tmp).replace(manifest_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        # flock released automatically when the `with open(lock_path)` block exits.
    logger.info("Updated manifest.json with ERA5-Land provenance")
