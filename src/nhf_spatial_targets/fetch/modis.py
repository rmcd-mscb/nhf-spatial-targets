"""Fetch MODIS products: MOD16A2 (AET) and MOD10C1 (SCA) via earthaccess."""

from __future__ import annotations

import gc
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
    consolidate_mod16a2_finalize,
    consolidate_mod16a2_timestep,
    log_memory,
)
from nhf_spatial_targets.workspace import load as _load_workspace

logger = logging.getLogger(__name__)

# MODIS filenames embed the acquisition date as AYYYYDDD (A=Acquisition,
# YYYY=year, DDD=day-of-year).  Examples:
#   MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf
#   MOD10C1.A2010032.061.2020345123456.hdf
#   MOD16A2GF.A2010001.h08v04.061.conus.nc
_MODIS_YEAR_RE = re.compile(r"\.A(\d{4})\d{3}\.")
_MODIS_YDOY_RE = re.compile(r"\.A(\d{7})\.")


def _granule_overlaps_bbox(
    granule: object,
    bbox: tuple[float, float, float, float],
) -> bool:
    """Check whether a granule's spatial extent overlaps *bbox*.

    Parameters
    ----------
    granule : earthaccess granule
        Must have UMM metadata with SpatialExtent.
    bbox : tuple
        ``(minx, miny, maxx, maxy)`` in EPSG:4326 degrees.

    Returns
    -------
    bool
        True if the granule overlaps the bbox or if spatial metadata
        is missing (fail-open to avoid dropping valid granules).
    """
    try:
        rects = granule["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"][
            "BoundingRectangles"
        ]
        if not isinstance(rects, list):
            logger.debug("Granule has non-list BoundingRectangles, keeping (fail-open)")
            return True
    except (KeyError, TypeError):
        logger.debug("Granule missing spatial metadata, keeping (fail-open)")
        return True

    minx, miny, maxx, maxy = bbox
    for r in rects:
        try:
            gw = r["WestBoundingCoordinate"]
            ge = r["EastBoundingCoordinate"]
            gs = r["SouthBoundingCoordinate"]
            gn = r["NorthBoundingCoordinate"]
        except (KeyError, TypeError):
            logger.debug("Granule has incomplete bounding rect, keeping (fail-open)")
            return True
        if gw <= maxx and ge >= minx and gs <= maxy and gn >= miny:
            return True
    return False


def _filter_granules_by_bbox(
    granules: list,
    bbox: tuple[float, float, float, float],
) -> list:
    """Filter granules to those overlapping *bbox*.

    Parameters
    ----------
    granules : list
        earthaccess granule objects.
    bbox : tuple
        ``(minx, miny, maxx, maxy)`` in EPSG:4326 degrees.

    Returns
    -------
    list
        Granules whose spatial extent overlaps *bbox*.
    """
    kept = [g for g in granules if _granule_overlaps_bbox(g, bbox)]
    dropped = len(granules) - len(kept)
    if dropped:
        logger.info(
            "Filtered %d of %d granules outside fabric bbox",
            dropped,
            len(granules),
        )
    return kept


def _group_granules_by_timestep(
    granules: list,
) -> dict[str, list]:
    """Group earthaccess granules by AYYYYDDD token.

    Parameters
    ----------
    granules : list
        earthaccess granule objects. Each must have a ``data_links()``
        method returning URLs that contain the MODIS filename.

    Returns
    -------
    dict[str, list]
        Mapping from YYYYDDD token to list of granules.
    """
    from collections import defaultdict

    groups: dict[str, list] = defaultdict(list)
    for g in granules:
        links = g.data_links()
        if not links:
            logger.warning("Granule %s has no data links, skipping", g)
            continue
        filename = links[0].split("/")[-1]
        m = _MODIS_YDOY_RE.search(filename)
        if not m:
            logger.warning("Cannot extract AYYYYDDD from granule URL: %s", links[0])
            continue
        groups[m.group(1)].append(g)

    total_grouped = sum(len(v) for v in groups.values())
    dropped = len(granules) - total_grouped
    if dropped > 0:
        logger.warning(
            "Dropped %d of %d granules during timestep grouping "
            "(no data links or unparseable filenames)",
            dropped,
            len(granules),
        )
    return dict(groups)


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


def _bbox_tuple(bbox: dict) -> tuple[float, float, float, float]:
    """Convert a bbox dict to ``(minx, miny, maxx, maxy)`` tuple."""
    return (bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"])


def _manifest_source_files(workdir: Path, source_key: str) -> list[dict]:
    """Read file records from ``manifest.json`` for a given source.

    Parameters
    ----------
    workdir : Path
        Workspace directory.
    source_key : str
        The source key to look up (e.g. ``"mod16a2_v061"``).

    Returns
    -------
    list[dict]
        List of file record dicts, or empty list if not present.
    """
    ws = _load_workspace(workdir)
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
    return manifest.get("sources", {}).get(source_key, {}).get("files", [])


def _existing_years(workdir: Path, source_key: str) -> set[int]:
    """Return years already fetched for *source_key* from manifest."""
    records = _manifest_source_files(workdir, source_key)
    skipped = sum(1 for f in records if "year" not in f)
    if skipped:
        logger.warning(
            "%d file record(s) in manifest for %s lack 'year' key; skipping them",
            skipped,
            source_key,
        )
    return {f["year"] for f in records if "year" in f}


def _existing_file_timestamps(workdir: Path, source_key: str) -> dict[int, str]:
    """Return ``{year: downloaded_utc}`` from existing manifest."""
    return {
        f["year"]: f["downloaded_utc"]
        for f in _manifest_source_files(workdir, source_key)
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
    workdir: Path,
    source_key: str,
    period: str,
    bbox: dict,
    meta: dict,
    files: list[dict],
    consolidated_ncs: dict[str, str],
) -> None:
    """Merge MODIS provenance into ``manifest.json`` with atomic write."""
    ws = _load_workspace(workdir)
    manifest_path = ws.manifest_path
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {workdir} is corrupted and cannot be parsed. "
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


def fetch_mod16a2(workdir: Path, period: str) -> dict:
    """Download MOD16A2 AET granules for the given period.

    Downloads and consolidates per-timestep to limit peak memory.
    Supports incremental download — years already recorded in
    ``manifest.json`` are skipped.

    Parameters
    ----------
    workdir : Path
        Workspace directory. Reads ``fabric.json`` for bbox,
        writes files to the datastore under ``mod16a2_v061/``.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    ws = _load_workspace(workdir)
    source_key = _MOD16A2_SOURCE_KEY
    meta = _catalog.source(source_key)
    short_name = meta["access"]["short_name"]
    variables = [v["name"] if isinstance(v, dict) else v for v in meta["variables"]]

    _check_superseded(meta, source_key)
    earthdata_login(workdir)
    logger.info("Authenticated with NASA Earthdata")
    log_memory("after authentication")

    bbox = ws.fabric["bbox_buffered"]
    bbox_t = _bbox_tuple(bbox)

    # Determine which years need downloading
    all_years = years_in_period(period)
    already_have = _existing_years(workdir, source_key)
    needed = [y for y in all_years if y not in already_have]

    output_dir = ws.raw_dir(source_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up stale temp files from prior interrupted runs
    for stale in output_dir.glob("_tmp_*_*.nc"):
        try:
            stale.unlink()
            logger.warning("Removed stale temp file: %s", stale.name)
        except OSError as exc:
            logger.warning("Could not remove stale temp file %s: %s", stale.name, exc)

    consolidated_ncs: dict[str, str] = {}

    if not needed:
        logger.info(
            "All %d years already downloaded; will check for missing consolidated files",
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

            # Filter out granules whose spatial extent doesn't overlap
            # the fabric bbox (earthaccess search can return extras)
            granules = _filter_granules_by_bbox(granules, bbox_t)
            log_memory(f"after search for year {year} ({len(granules)} granules)")

            if not granules:
                raise ValueError(
                    f"No granules found for {short_name} with "
                    f"bbox={bbox_t}, temporal={temporal}"
                )

            # Group granules by timestep for batched download
            ts_groups = _group_granules_by_timestep(granules)
            if not ts_groups:
                raise RuntimeError(
                    f"No granules could be grouped by timestep for "
                    f"{short_name} year {year}. This usually means granule "
                    f"URLs have changed format. Check earthaccess granule "
                    f"metadata for the {len(granules)} granules returned."
                )
            sorted_ydoys = sorted(ts_groups)
            n_steps = len(sorted_ydoys)
            logger.info("Grouped into %d timesteps for year %d", n_steps, year)

            tmp_paths: list[Path] = []

            try:
                for i, ydoy in enumerate(sorted_ydoys, 1):
                    batch = ts_groups[ydoy]
                    logger.info(
                        "Downloading timestep %d/%d (A%s): %d granules",
                        i,
                        n_steps,
                        ydoy,
                        len(batch),
                    )

                    downloaded = earthaccess.download(
                        batch,
                        local_path=str(output_dir),
                    )

                    if not downloaded:
                        raise RuntimeError(
                            f"earthaccess.download() returned no files for "
                            f"timestep A{ydoy} ({len(batch)} granules). "
                            f"Check network connectivity and Earthdata credentials."
                        )
                    if len(downloaded) < len(batch):
                        logger.warning(
                            "Partial download for timestep A%s: got %d of %d granules.",
                            ydoy,
                            len(downloaded),
                            len(batch),
                        )

                    log_memory(f"after downloading timestep {i}/{n_steps} (A{ydoy})")

                    # Consolidate this timestep immediately
                    tile_paths = [Path(f) for f in downloaded]
                    tmp_path = consolidate_mod16a2_timestep(
                        tile_paths=tile_paths,
                        variables=variables,
                        source_dir=output_dir,
                        ydoy=ydoy,
                        bbox=bbox_t,
                    )
                    tmp_paths.append(tmp_path)
                    gc.collect()
                    log_memory(f"after consolidating timestep {i}/{n_steps} (A{ydoy})")

                # Finalize: lazy-concat temp files into consolidated NetCDF
                out_path = output_dir / f"{source_key}_{year}_consolidated.nc"
                result = consolidate_mod16a2_finalize(
                    tmp_paths=tmp_paths,
                    variables=variables,
                    out_path=out_path,
                    source_key=source_key,
                    keep_tmp=False,
                )
                consolidated_ncs[str(year)] = result["consolidated_nc"]
                logger.info("Downloaded and consolidated year %d", year)
            except Exception:
                for p in tmp_paths:
                    if p.exists():
                        p.unlink()
                        logger.debug("Cleaned up temp file after failure: %s", p.name)
                raise

    # Re-consolidate any years that were already downloaded but not yet
    # consolidated in this run (e.g. prior download, no consolidated file)
    all_hdf_files = sorted(output_dir.glob("*.hdf"))
    years_on_disk = sorted({_year_from_path(p) for p in all_hdf_files})
    for year in years_on_disk:
        if str(year) not in consolidated_ncs:
            logger.info("Re-consolidating %s year %d", source_key, year)
            result = consolidate_mod16a2(
                output_dir, source_key, variables, year, bbox_t
            )
            consolidated_ncs[str(year)] = result["consolidated_nc"]

    # Build file inventory from all .hdf files on disk
    existing_timestamps = _existing_file_timestamps(workdir, source_key)
    now_utc = datetime.now(timezone.utc).isoformat()

    files = []
    for p in all_hdf_files:
        yr = _year_from_path(p)
        files.append(
            {
                "path": str(p),
                "year": yr,
                "size_bytes": p.stat().st_size,
                "downloaded_utc": existing_timestamps.get(yr, now_utc),
            }
        )

    # Compute effective period from actual files on disk
    if years_on_disk:
        effective_period = f"{years_on_disk[0]}/{years_on_disk[-1]}"
    else:
        effective_period = period

    # Update manifest.json (merge, don't overwrite)
    _update_manifest(
        workdir,
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

    out_path = hdf_path.with_suffix("").with_suffix(".conus.nc")

    try:
        ds = xr.open_dataset(hdf_path, engine="rasterio")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open {hdf_path} with rasterio engine: {exc}"
        ) from exc

    try:
        # rasterio reads MOD10C1 HDF4-EOS CMG files with x/y coords
        # and a singleton band dim
        if "band" in ds.dims:
            if ds.sizes["band"] != 1:
                raise ValueError(
                    f"Expected singleton 'band' dim in {hdf_path}, "
                    f"got size {ds.sizes['band']}"
                )
            ds = ds.squeeze("band", drop=True)

        # Rename x/y -> lon/lat for the subset below
        rename = {}
        if "x" in ds.dims:
            rename["x"] = "lon"
        if "y" in ds.dims:
            rename["y"] = "lat"
        if rename:
            ds = ds.rename(rename)

        missing = {"lat", "lon"} - set(ds.dims)
        if missing:
            raise ValueError(
                f"Expected 'lat' and 'lon' dimensions in {hdf_path} after "
                f"rasterio open + rename, but found dims={list(ds.dims)}. "
                f"Missing: {missing}"
            )

        subset = ds.sel(
            lat=slice(bbox["maxy"], bbox["miny"]),
            lon=slice(bbox["minx"], bbox["maxx"]),
        )
        subset.to_netcdf(out_path)
    finally:
        ds.close()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"Subset file {out_path} was not written correctly; "
            f"keeping original HDF at {hdf_path}"
        )

    hdf_path.unlink()
    logger.info("Subsetted %s -> %s", hdf_path.name, out_path.name)
    return out_path


def fetch_mod10c1(workdir: Path, period: str) -> dict:
    """Download MOD10C1 daily snow cover CMG files for the given period.

    Supports incremental download — years already recorded in
    ``manifest.json`` are skipped. After downloading, each HDF file is
    subsetted to CONUS and saved as NetCDF. Runs per-year consolidation
    and updates the manifest.

    Parameters
    ----------
    workdir : Path
        Workspace directory. Reads ``fabric.json`` for bbox,
        writes files to the datastore under ``mod10c1_v061/``.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    ws = _load_workspace(workdir)
    source_key = _MOD10C1_SOURCE_KEY
    meta = _catalog.source(source_key)
    short_name = meta["access"]["short_name"]

    _check_superseded(meta, source_key)
    earthdata_login(workdir)
    logger.info("Authenticated with NASA Earthdata")

    bbox = ws.fabric["bbox_buffered"]
    bbox_t = _bbox_tuple(bbox)

    # Determine which years need downloading
    all_years = years_in_period(period)
    already_have = _existing_years(workdir, source_key)
    needed = [y for y in all_years if y not in already_have]

    output_dir = ws.raw_dir(source_key)
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
    existing_timestamps = _existing_file_timestamps(workdir, source_key)
    now_utc = datetime.now(timezone.utc).isoformat()

    files = []
    for p in all_nc_files:
        yr = _year_from_path(p)
        files.append(
            {
                "path": str(p),
                "year": yr,
                "size_bytes": p.stat().st_size,
                "downloaded_utc": existing_timestamps.get(yr, now_utc),
            }
        )

    # Per-year consolidation
    variables = [v["name"] if isinstance(v, dict) else v for v in meta["variables"]]
    consolidated_ncs: dict[str, str] = {}
    years_on_disk = sorted({f["year"] for f in files})
    for year in years_on_disk:
        logger.info("Consolidating %s year %d", source_key, year)
        result = consolidate_mod10c1(output_dir, source_key, variables, year)
        consolidated_ncs[str(year)] = result["consolidated_nc"]

    # Compute effective period from actual files on disk
    if years_on_disk:
        effective_period = f"{years_on_disk[0]}/{years_on_disk[-1]}"
    else:
        effective_period = period

    # Update manifest.json (merge, don't overwrite)
    _update_manifest(
        workdir,
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
