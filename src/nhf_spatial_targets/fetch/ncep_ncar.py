"""Fetch NCEP/NCAR Reanalysis soil moisture from NOAA PSL."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import parse_period, years_in_period

logger = logging.getLogger(__name__)
_SOURCE_KEY = "ncep_ncar"


def consolidate_ncep_ncar(run_dir: Path, variables: list[str]) -> dict:
    """Delegate to consolidate module (added in Task 7).

    This thin wrapper exists so tests can patch
    ``nhf_spatial_targets.fetch.ncep_ncar.consolidate_ncep_ncar``.
    """
    from nhf_spatial_targets.fetch.consolidate import (
        consolidate_ncep_ncar as _real_consolidate,
    )

    return _real_consolidate(run_dir=run_dir, variables=variables)


def _manifest_ncep_ncar_files(run_dir: Path) -> list[dict]:
    """Read manifest.json and return the ncep_ncar file records list."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
        return manifest.get("sources", {}).get(_SOURCE_KEY, {}).get("files", [])
    except (json.JSONDecodeError, KeyError):
        logger.warning(
            "Malformed manifest.json in %s, ignoring ncep_ncar entries", run_dir
        )
        return []


def _existing_years(run_dir: Path) -> set[str]:
    """Return set of year values already fetched from manifest."""
    return {f["year"] for f in _manifest_ncep_ncar_files(run_dir) if "year" in f}


def _existing_file_timestamps(run_dir: Path) -> dict[str, str]:
    """Return {year: downloaded_utc} from existing manifest."""
    return {
        f["year"]: f["downloaded_utc"]
        for f in _manifest_ncep_ncar_files(run_dir)
        if "year" in f and "downloaded_utc" in f
    }


def _year_from_monthly_path(path: Path) -> str:
    """Extract year string from a monthly filename.

    e.g. ``soilw.0-10cm.gauss.2010.monthly.nc`` -> ``"2010"``
    """
    # Filename pattern: {file_variable}.{year}.monthly.nc
    # Split on '.' and find the part before 'monthly'
    parts = path.stem.split(".")  # stem drops the final .nc extension
    # stem of soilw.0-10cm.gauss.2010.monthly.nc is soilw.0-10cm.gauss.2010.monthly
    for i, part in enumerate(parts):
        if part == "monthly" and i > 0:
            return parts[i - 1]
    raise ValueError(f"Cannot extract year from NCEP/NCAR filename: {path.name}")


def fetch_ncep_ncar(run_dir: Path, period: str) -> dict:
    """Download NCEP/NCAR Reanalysis daily files, aggregate to monthly means.

    Downloads daily Gaussian grid NetCDF files from NOAA PSL for each year
    in the period, resamples them to monthly means, and consolidates via
    Kerchunk. Supports incremental download — years already recorded in
    ``manifest.json`` are skipped.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory. Reads ``fabric.json`` for bbox (stored in
        provenance only — no spatial subsetting). Writes files to
        ``data/raw/ncep_ncar/``.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    meta = _catalog.source(_SOURCE_KEY)

    if meta.get("status") == "superseded":
        warnings.warn(
            f"Source '{_SOURCE_KEY}' is superseded. "
            f"Consider using '{meta.get('superseded_by', 'unknown')}'.",
            DeprecationWarning,
            stacklevel=2,
        )

    fabric_path = run_dir / "fabric.json"
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"fabric.json not found in {run_dir}. "
            f"Run 'nhf-targets init' to create a run workspace first."
        )
    try:
        fabric = json.loads(fabric_path.read_text())
        bbox = fabric["bbox_buffered"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(
            f"fabric.json in {run_dir} is malformed or missing required "
            f"fields (bbox_buffered.{{minx,miny,maxx,maxy}}). "
            f"Re-run 'nhf-targets init' to regenerate it."
        ) from exc

    # Validate and determine needed years
    parse_period(period)
    all_years = years_in_period(period)

    already_have = _existing_years(run_dir)
    needed = [y for y in all_years if str(y) not in already_have]

    output_dir = run_dir / "data" / "raw" / _SOURCE_KEY
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

        for year in needed:
            for var_entry in meta["variables"]:
                file_var = var_entry["file_variable"]
                url = var_entry["file_pattern"].format(year=year)
                daily_path = output_dir / f"{file_var}.{year}.nc"

                logger.info("Downloading %s -> %s", url, daily_path)
                try:
                    urllib.request.urlretrieve(url, daily_path)
                except urllib.error.HTTPError as exc:
                    raise RuntimeError(
                        f"Failed to download {url}: HTTP {exc.code} {exc.reason}"
                    ) from exc

                # Aggregate daily to monthly means
                logger.debug("Resampling %s to monthly means", daily_path)
                with xr.open_dataset(daily_path) as ds:
                    monthly = ds.resample(time="1ME").mean()
                    monthly_path = output_dir / f"{file_var}.{year}.monthly.nc"
                    monthly.to_netcdf(monthly_path)

                # Delete the raw daily file
                daily_path.unlink()
                logger.debug("Deleted daily file %s", daily_path)

    # Build file inventory from all *.monthly.nc files on disk
    all_monthly_files = sorted(output_dir.glob("*.monthly.nc"))

    existing_timestamps = _existing_file_timestamps(run_dir)
    now_utc = datetime.now(timezone.utc).isoformat()

    files = []
    for p in all_monthly_files:
        rel = str(p.relative_to(run_dir))
        year_str = _year_from_monthly_path(p)
        files.append(
            {
                "path": rel,
                "year": year_str,
                "size_bytes": p.stat().st_size,
                "downloaded_utc": existing_timestamps.get(year_str, now_utc),
            }
        )

    # Consolidate into Kerchunk reference store (Task 7 adds the real implementation)
    var_names = list({v["name"] for v in meta["variables"]})
    try:
        consolidation = consolidate_ncep_ncar(run_dir=run_dir, variables=var_names)
    except (ImportError, AttributeError, NotImplementedError):
        logger.warning(
            "consolidate_ncep_ncar not yet available; skipping Kerchunk consolidation"
        )
        consolidation = {
            "kerchunk_ref": None,
            "last_consolidated_utc": now_utc,
            "n_files": len(all_monthly_files),
            "variables": var_names,
        }

    # Compute effective period from actual files on disk
    if files:
        all_years_on_disk = sorted(f["year"] for f in files)
        effective_period = f"{all_years_on_disk[0]}/{all_years_on_disk[-1]}"
    else:
        effective_period = period

    _update_manifest(run_dir, effective_period, bbox, meta, files, consolidation)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "variables": meta["variables"],
        "period": effective_period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "files": files,
        "kerchunk_ref": consolidation.get("kerchunk_ref"),
    }


def _update_manifest(
    run_dir: Path,
    period: str,
    bbox: dict,
    meta: dict,
    files: list[dict],
    consolidation: dict,
) -> None:
    """Merge NCEP/NCAR provenance into manifest.json."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"sources": {}, "steps": []}

    if "sources" not in manifest:
        manifest["sources"] = {}

    entry = manifest["sources"].get(_SOURCE_KEY, {})
    entry.update(
        {
            "source_key": _SOURCE_KEY,
            "access_url": meta["access"]["url"],
            "period": period,
            "bbox": bbox,
            "variables": list({v["name"] for v in meta["variables"]}),
            "files": files,
            "kerchunk_ref": consolidation.get("kerchunk_ref"),
            "last_consolidated_utc": consolidation.get("last_consolidated_utc"),
        }
    )
    manifest["sources"][_SOURCE_KEY] = entry

    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Updated manifest.json with NCEP/NCAR provenance")
