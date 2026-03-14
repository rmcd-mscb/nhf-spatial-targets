"""Fetch MODIS products: MOD16A2 (AET) and MOD10C1 (SCA) via earthaccess."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import earthaccess
import xarray as xr

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._auth import earthdata_login
from nhf_spatial_targets.fetch._period import years_in_period
from nhf_spatial_targets.fetch.consolidate import (
    consolidate_mod10c1,
    consolidate_mod16a2,
)

logger = logging.getLogger(__name__)

# MODIS filenames embed the acquisition date as AYYYYDDD (A=Acquisition,
# YYYY=year, DDD=day-of-year).  Examples:
#   MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf
#   MOD10C1.A2010032.061.2020345123456.hdf
#   MOD16A2GF.A2010001.h08v04.061.conus.nc
_MODIS_YEAR_RE = re.compile(r"\.A(\d{4})\d{3}\.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _year_from_path(path: Path) -> int:
    """Extract the 4-digit year from a MODIS filename with ``AYYYYDDD`` pattern.

    Parameters
    ----------
    path : Path
        File path whose *name* contains the MODIS date token.

    Returns
    -------
    int
        The year embedded in the filename.

    Raises
    ------
    ValueError
        If the filename does not contain an ``AYYYYDDD`` pattern.
    """
    m = _MODIS_YEAR_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot extract year from MODIS filename: {path.name}")
    return int(m.group(1))


def _read_fabric_bbox(run_dir: Path) -> dict:
    """Read ``bbox_buffered`` from ``fabric.json``.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory containing ``fabric.json``.

    Returns
    -------
    dict
        The ``bbox_buffered`` mapping with minx/miny/maxx/maxy keys.

    Raises
    ------
    FileNotFoundError
        If ``fabric.json`` does not exist.
    ValueError
        If ``fabric.json`` is malformed or missing required fields.
    """
    fabric_path = run_dir / "fabric.json"
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"fabric.json not found in {run_dir}. "
            f"Run 'nhf-targets init' to create a run workspace first."
        )
    try:
        fabric = json.loads(fabric_path.read_text())
        bbox = fabric["bbox_buffered"]
        # Validate required keys are present
        _ = bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(
            f"fabric.json in {run_dir} is malformed or missing required "
            f"fields (bbox_buffered.{{minx,miny,maxx,maxy}}). "
            f"Re-run 'nhf-targets init' to regenerate it."
        ) from exc
    return bbox


def _bbox_tuple(bbox: dict) -> tuple[float, float, float, float]:
    """Convert a bbox dict to ``(minx, miny, maxx, maxy)`` tuple."""
    return (bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"])


def _manifest_source_files(run_dir: Path, source_key: str) -> list[dict]:
    """Read file records from ``manifest.json`` for a given source.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory.
    source_key : str
        The source key to look up (e.g. ``"mod16a2_v061"``).

    Returns
    -------
    list[dict]
        List of file record dicts, or empty list if not present.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manifest.json in {run_dir} is corrupted and cannot be parsed. "
            f"Inspect the file manually or restore from backup. Detail: {exc}"
        ) from exc
    return manifest.get("sources", {}).get(source_key, {}).get("files", [])


def _existing_years(run_dir: Path, source_key: str) -> set[int]:
    """Return years already fetched for *source_key* from manifest."""
    records = _manifest_source_files(run_dir, source_key)
    skipped = sum(1 for f in records if "year" not in f)
    if skipped:
        logger.warning(
            "%d file record(s) in manifest for %s lack 'year' key; skipping them",
            skipped,
            source_key,
        )
    return {f["year"] for f in records if "year" in f}


def _existing_file_timestamps(run_dir: Path, source_key: str) -> dict[int, str]:
    """Return ``{year: downloaded_utc}`` from existing manifest."""
    return {
        f["year"]: f["downloaded_utc"]
        for f in _manifest_source_files(run_dir, source_key)
        if "year" in f and "downloaded_utc" in f
    }


def _check_superseded(meta: dict, source_key: str) -> None:
    """Emit :class:`DeprecationWarning` if *meta* indicates superseded."""
    if meta.get("status") == "superseded":
        warnings.warn(
            f"Source '{source_key}' is superseded. "
            f"Consider using '{meta.get('superseded_by', 'unknown')}'.",
            DeprecationWarning,
            stacklevel=3,
        )


def _update_manifest(
    run_dir: Path,
    source_key: str,
    period: str,
    bbox: dict,
    meta: dict,
    files: list[dict],
    consolidated_ncs: dict[str, str],
) -> None:
    """Merge MODIS provenance into ``manifest.json`` with atomic write."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {run_dir} is corrupted and cannot be parsed. "
                f"Inspect the file manually or restore from backup. Detail: {exc}"
            ) from exc
    else:
        manifest = {"sources": {}, "steps": []}

    if "sources" not in manifest:
        manifest["sources"] = {}

    now_utc = datetime.now(timezone.utc).isoformat()
    entry = manifest["sources"].get(source_key, {})
    entry.update(
        {
            "source_key": source_key,
            "access_url": meta["access"]["url"],
            "period": period,
            "bbox": bbox,
            "variables": meta["variables"],
            "files": files,
            "consolidated_ncs": consolidated_ncs,
            "last_consolidated_utc": now_utc if consolidated_ncs else None,
        }
    )
    manifest["sources"][source_key] = entry

    tmp_fd, tmp_path = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp_path).replace(manifest_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    logger.info("Updated manifest.json with %s provenance", source_key)


# ---------------------------------------------------------------------------
# fetch_mod16a2
# ---------------------------------------------------------------------------

_MOD16A2_SOURCE_KEY = "mod16a2_v061"


def fetch_mod16a2(run_dir: Path, period: str) -> dict:
    """Download MOD16A2 AET granules for the given period.

    Supports incremental download — years already recorded in
    ``manifest.json`` are skipped. After downloading, runs per-year
    consolidation and updates the manifest.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory. Reads ``fabric.json`` for bbox,
        writes files to ``data/raw/mod16a2_v061/``.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    source_key = _MOD16A2_SOURCE_KEY
    meta = _catalog.source(source_key)
    short_name = meta["access"]["short_name"]

    _check_superseded(meta, source_key)
    earthdata_login(run_dir)
    logger.info("Authenticated with NASA Earthdata")

    bbox = _read_fabric_bbox(run_dir)
    bbox_t = _bbox_tuple(bbox)

    # Determine which years need downloading
    all_years = years_in_period(period)
    already_have = _existing_years(run_dir, source_key)
    needed = [y for y in all_years if y not in already_have]

    output_dir = run_dir / "data" / "raw" / source_key
    output_dir.mkdir(parents=True, exist_ok=True)

    if not needed:
        logger.info(
            "All %d years already downloaded, skipping to consolidation",
            len(all_years),
        )
    else:
        for year in needed:
            temporal = (f"{year}-01-01", f"{year}-12-31")
            logger.debug("bbox=%s, temporal=%s, year=%d", bbox_t, temporal, year)

            granules = earthaccess.search_data(
                short_name=short_name,
                bounding_box=bbox_t,
                temporal=temporal,
            )
            logger.info(
                "Found %d granules for %s year %d",
                len(granules),
                short_name,
                year,
            )

            if not granules:
                raise ValueError(
                    f"No granules found for {short_name} with "
                    f"bbox={bbox_t}, temporal={temporal}"
                )

            downloaded = earthaccess.download(
                granules,
                local_path=str(output_dir),
            )

            if not downloaded:
                raise RuntimeError(
                    f"earthaccess.download() returned no files for "
                    f"{len(granules)} granules. Check network connectivity "
                    f"and Earthdata credentials."
                )
            if len(downloaded) < len(granules):
                logger.warning(
                    "Partial download: got %d of %d granules for year %d. "
                    "Consolidation will proceed with available files only.",
                    len(downloaded),
                    len(granules),
                    year,
                )
            logger.info(
                "Downloaded %d files for year %d to %s",
                len(downloaded),
                year,
                output_dir,
            )

    # Build file inventory from all .hdf files on disk
    all_hdf_files = sorted(output_dir.glob("*.hdf"))

    # Preserve original downloaded_utc for files already in manifest
    existing_timestamps = _existing_file_timestamps(run_dir, source_key)
    now_utc = datetime.now(timezone.utc).isoformat()

    files = []
    for p in all_hdf_files:
        rel = str(p.relative_to(run_dir))
        yr = _year_from_path(p)
        files.append(
            {
                "path": rel,
                "year": yr,
                "size_bytes": p.stat().st_size,
                "downloaded_utc": existing_timestamps.get(yr, now_utc),
            }
        )

    # Per-year consolidation
    variables = meta["variables"]
    consolidated_ncs: dict[str, str] = {}
    years_on_disk = sorted({f["year"] for f in files})
    for year in years_on_disk:
        logger.info("Consolidating %s year %d", source_key, year)
        result = consolidate_mod16a2(run_dir, source_key, variables, year)
        consolidated_ncs[str(year)] = result["consolidated_nc"]

    # Compute effective period from actual files on disk
    if years_on_disk:
        effective_period = f"{years_on_disk[0]}/{years_on_disk[-1]}"
    else:
        effective_period = period

    # Update manifest.json (merge, don't overwrite)
    _update_manifest(
        run_dir,
        source_key,
        effective_period,
        bbox,
        meta,
        files,
        consolidated_ncs,
    )

    return {
        "source_key": source_key,
        "access_url": meta["access"]["url"],
        "variables": variables,
        "period": effective_period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "files": files,
        "consolidated_ncs": consolidated_ncs,
    }


_CONUS_BBOX = {
    "minx": -129.4134,
    "miny": 22.3380,
    "maxx": -63.2790,
    "maxy": 54.6729,
}

_MOD10C1_SOURCE_KEY = "mod10c1_v061"


def _subset_to_conus(hdf_path: Path, bbox: dict | None = None) -> Path:
    """Open a global MOD10C1 HDF file, subset to CONUS, save as NetCDF.

    Parameters
    ----------
    hdf_path : Path
        Path to the global HDF file.
    bbox : dict | None
        Bounding box with minx/miny/maxx/maxy keys. Defaults to
        :data:`_CONUS_BBOX`.

    Returns
    -------
    Path
        Path to the ``.conus.nc`` file that replaced the original HDF.
    """
    if bbox is None:
        bbox = _CONUS_BBOX

    ds = xr.open_dataset(hdf_path)
    try:
        subset = ds.sel(
            lat=slice(bbox["maxy"], bbox["miny"]),
            lon=slice(bbox["minx"], bbox["maxx"]),
        )
        out_path = hdf_path.with_suffix("").with_suffix(".conus.nc")
        subset.to_netcdf(out_path)
    finally:
        ds.close()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"Subset file {out_path} was not written correctly; "
            f"keeping original HDF at {hdf_path}"
        )

    hdf_path.unlink()
    logger.info("Subsetted %s → %s", hdf_path.name, out_path.name)
    return out_path


def fetch_mod10c1(run_dir: Path, period: str) -> dict:
    """Download MOD10C1 daily snow cover CMG files for the given period.

    Supports incremental download — years already recorded in
    ``manifest.json`` are skipped. After downloading, each HDF file is
    subsetted to CONUS and saved as NetCDF. Runs per-year consolidation
    and updates the manifest.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory. Reads ``fabric.json`` for bbox,
        writes files to ``data/raw/mod10c1_v061/``.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    source_key = _MOD10C1_SOURCE_KEY
    meta = _catalog.source(source_key)
    short_name = meta["access"]["short_name"]

    _check_superseded(meta, source_key)
    earthdata_login(run_dir)
    logger.info("Authenticated with NASA Earthdata")

    bbox = _read_fabric_bbox(run_dir)
    bbox_t = _bbox_tuple(bbox)

    # Determine which years need downloading
    all_years = years_in_period(period)
    already_have = _existing_years(run_dir, source_key)
    needed = [y for y in all_years if y not in already_have]

    output_dir = run_dir / "data" / "raw" / source_key
    output_dir.mkdir(parents=True, exist_ok=True)

    if not needed:
        logger.info(
            "All %d years already downloaded, skipping to consolidation",
            len(all_years),
        )
    else:
        for year in needed:
            temporal = (f"{year}-01-01", f"{year}-12-31")
            logger.debug("bbox=%s, temporal=%s, year=%d", bbox_t, temporal, year)

            granules = earthaccess.search_data(
                short_name=short_name,
                bounding_box=bbox_t,
                temporal=temporal,
            )
            logger.info(
                "Found %d granules for %s year %d",
                len(granules),
                short_name,
                year,
            )

            if not granules:
                raise ValueError(
                    f"No granules found for {short_name} with "
                    f"bbox={bbox_t}, temporal={temporal}"
                )

            downloaded = earthaccess.download(
                granules,
                local_path=str(output_dir),
            )

            if not downloaded:
                raise RuntimeError(
                    f"earthaccess.download() returned no files for "
                    f"{len(granules)} granules. Check network connectivity "
                    f"and Earthdata credentials."
                )
            if len(downloaded) < len(granules):
                logger.warning(
                    "Partial download: got %d of %d granules for year %d. "
                    "Consolidation will proceed with available files only.",
                    len(downloaded),
                    len(granules),
                    year,
                )
            logger.info(
                "Downloaded %d files for year %d to %s",
                len(downloaded),
                year,
                output_dir,
            )

            # Subset each HDF to CONUS
            for p in [Path(f) for f in downloaded]:
                if p.suffix == ".hdf":
                    _subset_to_conus(p)

    # Build file inventory from all .conus.nc files on disk
    all_nc_files = sorted(output_dir.glob("*.conus.nc"))

    # Preserve original downloaded_utc for files already in manifest
    existing_timestamps = _existing_file_timestamps(run_dir, source_key)
    now_utc = datetime.now(timezone.utc).isoformat()

    files = []
    for p in all_nc_files:
        rel = str(p.relative_to(run_dir))
        yr = _year_from_path(p)
        files.append(
            {
                "path": rel,
                "year": yr,
                "size_bytes": p.stat().st_size,
                "downloaded_utc": existing_timestamps.get(yr, now_utc),
            }
        )

    # Per-year consolidation
    variables = meta["variables"]
    consolidated_ncs: dict[str, str] = {}
    years_on_disk = sorted({f["year"] for f in files})
    for year in years_on_disk:
        logger.info("Consolidating %s year %d", source_key, year)
        result = consolidate_mod10c1(run_dir, source_key, variables, year)
        consolidated_ncs[str(year)] = result["consolidated_nc"]

    # Compute effective period from actual files on disk
    if years_on_disk:
        effective_period = f"{years_on_disk[0]}/{years_on_disk[-1]}"
    else:
        effective_period = period

    # Update manifest.json (merge, don't overwrite)
    _update_manifest(
        run_dir,
        source_key,
        effective_period,
        bbox,
        meta,
        files,
        consolidated_ncs,
    )

    return {
        "source_key": source_key,
        "access_url": meta["access"]["url"],
        "variables": variables,
        "period": effective_period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "files": files,
        "consolidated_ncs": consolidated_ncs,
    }
