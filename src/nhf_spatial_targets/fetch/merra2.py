"""Fetch MERRA-2 monthly soil moisture via earthaccess."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import earthaccess

import nhf_spatial_targets.catalog as _catalog

_SOURCE_KEY = "merra2"
_SHORT_NAME = "M2TMNXLND"


def fetch_merra2(run_dir: Path, period: str) -> dict:
    """Download MERRA-2 monthly soil moisture for the given period.

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

    # Warn if source is superseded
    if meta.get("status") == "superseded":
        warnings.warn(
            f"Source '{_SOURCE_KEY}' is superseded. "
            f"Consider using '{meta.get('superseded_by', 'unknown')}'.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Authenticate
    auth = earthaccess.login()
    if auth is None or not auth.authenticated:
        raise RuntimeError(
            "NASA Earthdata login failed. Register at "
            "https://urs.earthdata.nasa.gov/users/new"
        )

    # Read bounding box from fabric.json
    fabric_path = run_dir / "fabric.json"
    fabric = json.loads(fabric_path.read_text())
    bbox = fabric["bbox_buffered"]
    bbox_tuple = (bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"])

    # Parse period
    start_year, end_year = period.split("/")
    temporal = (f"{start_year}-01-01", f"{end_year}-12-31")

    # Search for granules
    granules = earthaccess.search_data(
        short_name=_SHORT_NAME,
        bounding_box=bbox_tuple,
        temporal=temporal,
    )

    if not granules:
        raise ValueError(
            f"No granules found for {_SHORT_NAME} with "
            f"bbox={bbox_tuple}, temporal={temporal}"
        )

    # Download
    output_dir = run_dir / "data" / "raw" / _SOURCE_KEY
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = earthaccess.download(
        granules,
        local_path=str(output_dir),
    )

    # Build provenance record
    variables = meta["variables"]
    files = []
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

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "variables": variables,
        "period": period,
        "bbox": bbox,
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
