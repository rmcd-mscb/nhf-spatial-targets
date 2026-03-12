"""Fetch NLDAS-2 land surface model soil moisture via earthaccess."""

from __future__ import annotations

import json
import logging
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import earthaccess

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._auth import earthdata_login
from nhf_spatial_targets.fetch._period import months_in_period, parse_period

logger = logging.getLogger(__name__)

# Regex to extract YYYY-MM from NLDAS granule URLs/filenames.
# Example: NLDAS_MOS0125_M.A200101.002.grb.SUB.nc4
_NLDAS_DATE_RE = re.compile(r"\.A(\d{4})(\d{2})\.")


def _granule_year_month(granule: object) -> str | None:
    """Extract 'YYYY-MM' from an earthaccess granule's data links."""
    for link in granule.data_links():
        m = _NLDAS_DATE_RE.search(link)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    return None


def _manifest_nldas_files(run_dir: Path, source_key: str) -> list[dict]:
    """Read manifest.json and return the NLDAS file records list."""
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


def _existing_months(run_dir: Path, source_key: str) -> set[str]:
    """Return set of year_month values already fetched from manifest."""
    return {
        f["year_month"]
        for f in _manifest_nldas_files(run_dir, source_key)
        if "year_month" in f
    }


def _year_month_from_path(path: Path) -> str:
    """Extract 'YYYY-MM' from an NLDAS filename like '...A200101...'."""
    m = _NLDAS_DATE_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot extract date from NLDAS filename: {path.name}")
    return f"{m.group(1)}-{m.group(2)}"


def _existing_file_timestamps(run_dir: Path, source_key: str) -> dict[str, str]:
    """Return {year_month: downloaded_utc} from existing manifest."""
    return {
        f["year_month"]: f["downloaded_utc"]
        for f in _manifest_nldas_files(run_dir, source_key)
        if "year_month" in f and "downloaded_utc" in f
    }


def fetch_nldas_mosaic(run_dir: Path, period: str) -> dict:
    """Download NLDAS-2 MOSAIC monthly soil moisture.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory. Reads ``fabric.json`` for bbox,
        writes files to ``data/raw/nldas_mosaic/``.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    return _fetch_nldas("nldas_mosaic", run_dir, period)


def fetch_nldas_noah(run_dir: Path, period: str) -> dict:
    """Download NLDAS-2 NOAH monthly soil moisture.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory. Reads ``fabric.json`` for bbox,
        writes files to ``data/raw/nldas_noah/``.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    return _fetch_nldas("nldas_noah", run_dir, period)


def _fetch_nldas(source_key: str, run_dir: Path, period: str) -> dict:
    """Download NLDAS-2 granules for the given source and period.

    Supports incremental download — months already recorded in
    ``manifest.json`` are skipped. After downloading, builds a
    Kerchunk virtual Zarr reference store and updates the manifest.

    Parameters
    ----------
    source_key : str
        Catalog key: ``"nldas_mosaic"`` or ``"nldas_noah"``.
    run_dir : Path
        Run workspace directory. Reads ``fabric.json`` for bbox,
        writes files to ``data/raw/<source_key>/``.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    from nhf_spatial_targets.fetch.consolidate import consolidate_nldas  # noqa: PLC0415

    meta = _catalog.source(source_key)
    short_name = meta["access"]["short_name"]

    if meta.get("status") == "superseded":
        warnings.warn(
            f"Source '{source_key}' is superseded. "
            f"Consider using '{meta.get('superseded_by', 'unknown')}'.",
            DeprecationWarning,
            stacklevel=3,
        )

    earthdata_login(run_dir)
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
    already_have = _existing_months(run_dir, source_key)
    all_months = months_in_period(period)
    needed = [m for m in all_months if m not in already_have]

    output_dir = run_dir / "data" / "raw" / source_key
    output_dir.mkdir(parents=True, exist_ok=True)

    if not needed:
        logger.info(
            "All %d months already downloaded, skipping to consolidation",
            len(all_months),
        )
    else:
        temporal = parse_period(period)
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

    # Build file inventory from all .nc4 and .nc files on disk
    all_nc_files = sorted(
        list(output_dir.glob("*.nc4")) + list(output_dir.glob("*.nc"))
    )

    # Preserve original downloaded_utc for files already in manifest
    existing_timestamps = _existing_file_timestamps(run_dir, source_key)
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
    consolidation = consolidate_nldas(
        run_dir=run_dir, source_key=source_key, variables=var_names
    )

    # Compute effective period from actual files on disk
    if files:
        all_ym = sorted(f["year_month"] for f in files)
        effective_start = all_ym[0][:4]
        effective_end = all_ym[-1][:4]
        effective_period = f"{effective_start}/{effective_end}"
    else:
        effective_period = period

    # Update manifest.json (merge, don't overwrite)
    _update_manifest(
        run_dir, source_key, effective_period, bbox, meta, files, consolidation
    )

    return {
        "source_key": source_key,
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
    source_key: str,
    period: str,
    bbox: dict,
    meta: dict,
    files: list[dict],
    consolidation: dict,
) -> None:
    """Merge NLDAS provenance into manifest.json."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"sources": {}, "steps": []}

    if "sources" not in manifest:
        manifest["sources"] = {}

    entry = manifest["sources"].get(source_key, {})
    entry.update(
        {
            "source_key": source_key,
            "access_url": meta["access"]["url"],
            "period": period,
            "bbox": bbox,
            "variables": [v["name"] for v in meta["variables"]],
            "files": files,
            "kerchunk_ref": consolidation["kerchunk_ref"],
            "last_consolidated_utc": consolidation["last_consolidated_utc"],
        }
    )
    manifest["sources"][source_key] = entry

    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Updated manifest.json with %s provenance", source_key)
