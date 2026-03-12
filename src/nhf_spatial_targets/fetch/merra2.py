"""Fetch MERRA-2 monthly land surface diagnostics (M2TMNXLND) via earthaccess."""

from __future__ import annotations

import json
import logging
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import earthaccess

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

_SOURCE_KEY = "merra2"
logger = logging.getLogger(__name__)

_MERRA2_DATE_RE = re.compile(r"\.(\d{4})(\d{2})\.nc4$")
_MERRA2_DATE_URL_RE = re.compile(r"\.(\d{4})(\d{2})\.nc4")


def _granule_year_month(granule: object) -> str | None:
    """Extract 'YYYY-MM' from an earthaccess granule's data links."""
    for link in granule.data_links():
        m = _MERRA2_DATE_URL_RE.search(link)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    return None


def _parse_period(period: str) -> tuple[str, str]:
    """Parse ``"YYYY/YYYY"`` into ``("YYYY-01-01", "YYYY-12-31")``."""
    parts = period.split("/")
    if len(parts) != 2:
        raise ValueError(f"period must be 'YYYY/YYYY', got: {period!r}")
    start_year, end_year = parts
    try:
        start_int, end_int = int(start_year), int(end_year)
    except ValueError:
        raise ValueError(f"period years must be integers, got: {period!r}") from None
    if end_int < start_int:
        raise ValueError(
            f"period end year ({end_year}) is before start year "
            f"({start_year}). Use 'YYYY/YYYY' with start <= end."
        )
    return (f"{start_year}-01-01", f"{end_year}-12-31")


def _months_in_period(period: str) -> list[str]:
    """Return list of 'YYYY-MM' strings for every month in the period."""
    _parse_period(period)  # validate format
    parts = period.split("/")
    start_year, end_year = int(parts[0]), int(parts[1])
    months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            months.append(f"{year}-{month:02d}")
    return months


def _manifest_merra2_files(run_dir: Path) -> list[dict]:
    """Read manifest.json and return the merra2 file records list."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
        return manifest.get("sources", {}).get("merra2", {}).get("files", [])
    except (json.JSONDecodeError, KeyError):
        logger.warning(
            "Malformed manifest.json in %s, ignoring merra2 entries", run_dir
        )
        return []


def _existing_months(run_dir: Path) -> set[str]:
    """Return set of year_month values already fetched from manifest."""
    return {
        f["year_month"] for f in _manifest_merra2_files(run_dir) if "year_month" in f
    }


def _year_month_from_path(path: Path) -> str:
    """Extract 'YYYY-MM' from a MERRA-2 filename like '...201007.nc4'."""
    m = _MERRA2_DATE_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot extract date from MERRA-2 filename: {path.name}")
    return f"{m.group(1)}-{m.group(2)}"


def _existing_file_timestamps(run_dir: Path) -> dict[str, str]:
    """Return {year_month: downloaded_utc} from existing manifest."""
    return {
        f["year_month"]: f["downloaded_utc"]
        for f in _manifest_merra2_files(run_dir)
        if "year_month" in f and "downloaded_utc" in f
    }


def fetch_merra2(run_dir: Path, period: str) -> dict:
    """Download MERRA-2 M2TMNXLND granules for the given period.

    Supports incremental download — months already recorded in
    ``manifest.json`` are skipped. After downloading, builds a
    Kerchunk virtual Zarr reference store and updates the manifest.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory. Reads ``fabric.json`` for bbox,
        writes files to ``data/raw/merra2/``.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    meta = _catalog.source(_SOURCE_KEY)
    short_name = meta["access"]["short_name"]

    if meta.get("status") == "superseded":
        warnings.warn(
            f"Source '{_SOURCE_KEY}' is superseded. "
            f"Consider using '{meta.get('superseded_by', 'unknown')}'.",
            DeprecationWarning,
            stacklevel=2,
        )

    auth = earthaccess.login()
    if auth is None or not auth.authenticated:
        raise RuntimeError(
            "NASA Earthdata login failed. Register at "
            "https://urs.earthdata.nasa.gov/users/new"
        )
    logger.info("Authenticated with NASA Earthdata")

    fabric_path = run_dir / "fabric.json"
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"fabric.json not found in {run_dir}. "
            f"Run 'nhf-targets init' to create a run workspace first."
        )
    try:
        fabric = json.loads(fabric_path.read_text())
        bbox = fabric["bbox_buffered"]
        bbox_tuple = (bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"])
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(
            f"fabric.json in {run_dir} is malformed or missing required "
            f"fields (bbox_buffered.{{minx,miny,maxx,maxy}}). "
            f"Re-run 'nhf-targets init' to regenerate it."
        ) from exc

    # Determine which months need downloading
    already_have = _existing_months(run_dir)
    all_months = _months_in_period(period)
    needed = [m for m in all_months if m not in already_have]

    if not needed:
        logger.info(
            "All %d months already downloaded, skipping to consolidation",
            len(all_months),
        )
    else:
        temporal = _parse_period(period)
        logger.debug("bbox=%s, temporal=%s", bbox_tuple, temporal)

        granules = earthaccess.search_data(
            short_name=short_name,
            bounding_box=bbox_tuple,
            temporal=temporal,
        )
        logger.info("Found %d granules for %s", len(granules), short_name)

        if not granules:
            raise ValueError(
                f"No granules found for {short_name} with "
                f"bbox={bbox_tuple}, temporal={temporal}"
            )

        # Filter granules to only months not already downloaded
        needed_set = set(needed)
        granules = [g for g in granules if _granule_year_month(g) in needed_set]
        if not granules:
            logger.info(
                "All granules matched already-downloaded months, "
                "skipping to consolidation"
            )
        else:
            logger.info(
                "Downloading %d of %d needed months",
                len(granules),
                len(needed),
            )

            output_dir = run_dir / "data" / "raw" / _SOURCE_KEY
            output_dir.mkdir(parents=True, exist_ok=True)

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
                    "Partial download: got %d of %d granules. "
                    "Consolidation will proceed with available files only.",
                    len(downloaded),
                    len(granules),
                )
            logger.info("Downloaded %d files to %s", len(downloaded), output_dir)

    # Build file inventory from all .nc4 files on disk
    output_dir = run_dir / "data" / "raw" / _SOURCE_KEY
    all_nc_files = sorted(output_dir.glob("*.nc4"))

    # Preserve original downloaded_utc for files already in manifest
    existing_timestamps = _existing_file_timestamps(run_dir)
    now_utc = datetime.now(timezone.utc).isoformat()

    files = []
    for p in all_nc_files:
        rel = str(p.relative_to(run_dir))
        ym = _year_month_from_path(p)
        files.append(
            {
                "path": rel,
                "year_month": ym,
                "size_bytes": p.stat().st_size,
                "downloaded_utc": existing_timestamps.get(ym, now_utc),
            }
        )

    # Consolidate into Kerchunk reference store
    var_names = [v["name"] for v in meta["variables"]]
    consolidation = consolidate_merra2(run_dir=run_dir, variables=var_names)

    # Compute effective period from actual files on disk
    if files:
        all_ym = sorted(f["year_month"] for f in files)
        effective_start = all_ym[0][:4]
        effective_end = all_ym[-1][:4]
        effective_period = f"{effective_start}/{effective_end}"
    else:
        effective_period = period

    # Update manifest.json (merge, don't overwrite)
    _update_manifest(run_dir, effective_period, bbox, meta, files, consolidation)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "variables": meta["variables"],
        "period": effective_period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "files": files,
        "kerchunk_ref": consolidation["kerchunk_ref"],
    }


def _update_manifest(
    run_dir: Path,
    period: str,
    bbox: dict,
    meta: dict,
    files: list[dict],
    consolidation: dict,
) -> None:
    """Merge MERRA-2 provenance into manifest.json."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"sources": {}, "steps": []}

    if "sources" not in manifest:
        manifest["sources"] = {}

    merra2 = manifest["sources"].get("merra2", {})
    merra2.update(
        {
            "source_key": _SOURCE_KEY,
            "access_url": meta["access"]["url"],
            "period": period,
            "bbox": bbox,
            "variables": [v["name"] for v in meta["variables"]],
            "files": files,
            "kerchunk_ref": consolidation["kerchunk_ref"],
            "last_consolidated_utc": consolidation["last_consolidated_utc"],
        }
    )
    manifest["sources"]["merra2"] = merra2

    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Updated manifest.json with MERRA-2 provenance")
