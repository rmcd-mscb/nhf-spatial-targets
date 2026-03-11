"""Fetch MERRA-2 land surface data for soil moisture variables via earthaccess."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import earthaccess

import nhf_spatial_targets.catalog as _catalog

_SOURCE_KEY = "merra2"


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


def fetch_merra2(run_dir: Path, period: str) -> dict:
    """Download MERRA-2 M2TMNXLND granules for the given period.

    Downloads the full monthly land surface diagnostics product;
    relevant soil moisture variables (SFMC, GWETROOT) are extracted
    downstream during aggregation.

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

    fabric_path = run_dir / "fabric.json"
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"fabric.json not found in {run_dir}. "
            f"Run 'nhf-targets init' to create a run workspace first."
        )
    fabric = json.loads(fabric_path.read_text())
    bbox = fabric["bbox_buffered"]
    bbox_tuple = (bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"])

    temporal = _parse_period(period)

    granules = earthaccess.search_data(
        short_name=short_name,
        bounding_box=bbox_tuple,
        temporal=temporal,
    )

    if not granules:
        raise ValueError(
            f"No granules found for {short_name} with "
            f"bbox={bbox_tuple}, temporal={temporal}"
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

    variables = meta["variables"]
    files = []
    missing = []
    for fpath in downloaded:
        p = Path(fpath)
        if p.exists():
            rel = p.relative_to(run_dir)
            files.append(
                {
                    "path": str(rel),
                    "size_bytes": p.stat().st_size,
                }
            )
        else:
            missing.append(str(fpath))

    if missing:
        raise RuntimeError(
            f"Download reported {len(downloaded)} files but "
            f"{len(missing)} do not exist on disk: {missing}"
        )

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "variables": variables,
        "period": period,
        "bbox": bbox,
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
