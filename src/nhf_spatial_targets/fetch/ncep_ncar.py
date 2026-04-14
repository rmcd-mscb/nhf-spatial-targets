"""Fetch NCEP/NCAR Reanalysis soil moisture from NOAA PSL."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr
from tqdm import tqdm

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import parse_period, years_in_period
from nhf_spatial_targets.fetch.consolidate import resolve_license
from nhf_spatial_targets.workspace import load as _load_project

logger = logging.getLogger(__name__)
_SOURCE_KEY = "ncep_ncar"


def consolidate_ncep_ncar(source_dir: Path, variables: list[str]) -> dict:
    """Delegate to the shared consolidate module for testability.

    This thin wrapper exists so tests can patch
    ``nhf_spatial_targets.fetch.ncep_ncar.consolidate_ncep_ncar``.
    """
    from nhf_spatial_targets.fetch.consolidate import (
        consolidate_ncep_ncar as _real_consolidate,
    )

    return _real_consolidate(source_dir=source_dir, variables=variables)


def _manifest_ncep_ncar_files(workdir: Path) -> list[dict]:
    """Read manifest.json and return the ncep_ncar file records list."""
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manifest.json in {workdir} is corrupted and cannot be parsed. "
            f"Inspect the file manually or restore from backup. Detail: {exc}"
        ) from exc
    return manifest.get("sources", {}).get(_SOURCE_KEY, {}).get("files", [])


def _existing_years(workdir: Path) -> set[str]:
    """Return set of year values already fetched from manifest."""
    return {f["year"] for f in _manifest_ncep_ncar_files(workdir) if "year" in f}


def _existing_file_timestamps(workdir: Path) -> dict[str, str]:
    """Return {year: downloaded_utc} from existing manifest."""
    return {
        f["year"]: f["downloaded_utc"]
        for f in _manifest_ncep_ncar_files(workdir)
        if "year" in f and "downloaded_utc" in f
    }


def _year_from_monthly_path(path: Path) -> str:
    """Extract year string from a monthly filename.

    e.g. ``soilw.0-10cm.gauss.2010.monthly.nc`` -> ``"2010"``
    """
    # stem e.g. "soilw.0-10cm.gauss.2010.monthly" — find part before "monthly"
    parts = path.stem.split(".")
    for i, part in enumerate(parts):
        if part == "monthly" and i > 0:
            return parts[i - 1]
    raise ValueError(f"Cannot extract year from NCEP/NCAR filename: {path.name}")


def fetch_ncep_ncar(workdir: Path, period: str) -> dict:
    """Download NCEP/NCAR Reanalysis soil moisture and aggregate to monthly means.

    Downloads annual NetCDF files containing daily averages from NOAA PSL
    for each year in the period, resamples to monthly means, and
    consolidates into a single NetCDF. Supports incremental download — years
    already recorded in ``manifest.json`` are skipped.

    Parameters
    ----------
    workdir : Path
        Project directory. Reads ``fabric.json`` for bbox (stored in
        provenance only — no spatial subsetting). Writes files to
        the datastore under ``ncep_ncar/``.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    ws = _load_project(workdir)
    meta = _catalog.source(_SOURCE_KEY)

    if meta.get("status") == "superseded":
        warnings.warn(
            f"Source '{_SOURCE_KEY}' is superseded. "
            f"Consider using '{meta.get('superseded_by', 'unknown')}'.",
            DeprecationWarning,
            stacklevel=2,
        )

    bbox = ws.fabric["bbox_buffered"]

    # Validate and determine needed years
    parse_period(period)
    all_years = years_in_period(period)

    already_have = _existing_years(workdir)
    needed = [y for y in all_years if str(y) not in already_have]

    output_dir = ws.raw_dir(_SOURCE_KEY)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not needed:
        logger.info(
            "All %d years already downloaded, skipping to consolidation",
            len(all_years),
        )
    else:
        logger.info(
            "Downloading %d of %d years for NCEP/NCAR",
            len(needed),
            len(all_years),
        )

        # Build flat list of (year, var_entry) tasks for progress tracking
        tasks = [
            (year, var_entry) for year in needed for var_entry in meta["variables"]
        ]
        for year, var_entry in tqdm(tasks, desc="NCEP/NCAR download"):
            file_var = var_entry["file_variable"]
            url = var_entry["file_pattern"].format(year=year)
            daily_path = output_dir / f"{file_var}.{year}.nc"

            logger.debug("Downloading %s -> %s", url, daily_path)
            try:
                urllib.request.urlretrieve(url, daily_path)
            except urllib.error.HTTPError as exc:
                raise RuntimeError(
                    f"Failed to download {url}: HTTP {exc.code} {exc.reason}"
                ) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    f"Failed to connect to {url}: {exc.reason}. "
                    "Check network connectivity and DNS resolution."
                ) from exc
            except OSError as exc:
                raise RuntimeError(f"Network error downloading {url}: {exc}") from exc

            # Aggregate daily to monthly means
            logger.debug("Resampling %s to monthly means", daily_path)
            with xr.open_dataset(daily_path) as ds:
                monthly = ds.resample(time="1ME").mean()
                # Rename internal variable to catalog name for disambiguation
                internal_var = file_var.split(".")[0]
                catalog_name = var_entry["name"]
                if internal_var in monthly.data_vars and internal_var != catalog_name:
                    monthly = monthly.rename({internal_var: catalog_name})
                monthly_path = output_dir / f"{file_var}.{year}.monthly.nc"
                monthly.to_netcdf(monthly_path, format="NETCDF3_CLASSIC")

            if not monthly_path.exists() or monthly_path.stat().st_size == 0:
                raise RuntimeError(
                    f"Monthly aggregation produced empty or missing file: "
                    f"{monthly_path}. Daily file preserved at {daily_path}."
                )
            # Delete the raw daily file
            daily_path.unlink()
            logger.debug("Deleted daily file %s", daily_path)

    # Build file inventory from all *.monthly.nc files on disk
    all_monthly_files = sorted(output_dir.glob("*.monthly.nc"))

    existing_timestamps = _existing_file_timestamps(workdir)
    now_utc = datetime.now(timezone.utc).isoformat()

    files = []
    for p in all_monthly_files:
        year_str = _year_from_monthly_path(p)
        files.append(
            {
                "path": str(p),
                "year": year_str,
                "size_bytes": p.stat().st_size,
                "downloaded_utc": existing_timestamps.get(year_str, now_utc),
            }
        )

    # Consolidate into single NetCDF
    var_names = [v["name"] for v in meta["variables"]]
    consolidation = consolidate_ncep_ncar(source_dir=output_dir, variables=var_names)

    # Compute effective period from actual files on disk
    if files:
        all_years_on_disk = sorted(f["year"] for f in files)
        effective_period = f"{all_years_on_disk[0]}/{all_years_on_disk[-1]}"
    else:
        effective_period = period

    _update_manifest(workdir, effective_period, bbox, meta, files, consolidation)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "variables": meta["variables"],
        "period": effective_period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "files": files,
        "consolidated_nc": consolidation.get("consolidated_nc"),
    }


def _update_manifest(
    workdir: Path,
    period: str,
    bbox: dict,
    meta: dict,
    files: list[dict],
    consolidation: dict,
) -> None:
    """Merge NCEP/NCAR provenance into manifest.json."""
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {workdir} is corrupted. Detail: {exc}"
            ) from exc
    else:
        manifest = {"sources": {}, "steps": []}

    if "sources" not in manifest:
        manifest["sources"] = {}

    entry = manifest["sources"].get(_SOURCE_KEY, {})
    entry.update(
        {
            "source_key": _SOURCE_KEY,
            "access_url": meta["access"]["url"],
            "license": resolve_license(meta, _SOURCE_KEY),
            "period": period,
            "bbox": bbox,
            "variables": [v["name"] for v in meta["variables"]],
            "files": files,
            "consolidated_nc": consolidation.get("consolidated_nc"),
            "last_consolidated_utc": consolidation.get("last_consolidated_utc"),
        }
    )
    manifest["sources"][_SOURCE_KEY] = entry

    tmp_fd, tmp_path = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp_path).replace(manifest_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    logger.info("Updated manifest.json with NCEP/NCAR provenance")
