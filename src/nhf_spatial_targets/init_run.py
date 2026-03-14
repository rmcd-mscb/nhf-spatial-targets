"""Logic for 'nhf-targets init' — create an isolated run workspace."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from nhf_spatial_targets import __version__


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def init_run(
    fabric_path: Path,
    id_col: str,
    config_path: Path,
    workdir: Path,
    run_label: str | None = None,
    buffer_deg: float = 0.1,
) -> Path:
    """
    Create a new run workspace under workdir.

    Parameters
    ----------
    fabric_path : Path to the HRU fabric GeoPackage.
    id_col      : Name of the HRU ID column.
    config_path : Path to the pipeline.yml to copy as the run config.
    workdir     : Root directory for all runs (e.g. /data/nhf-runs).
    run_label   : Short label embedded in the run ID (e.g. "gfv11").
                  Produces: 2026-03-11_gfv11_v0.1.0
    buffer_deg  : Degrees to buffer the fabric bbox for source downloads.

    Returns
    -------
    Path to the newly created run directory.
    """
    fabric_path = fabric_path.resolve()
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    run_id = _make_run_id(run_label)
    run_dir = workdir / run_id

    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}\n"
            "Use a different --id label to disambiguate same-day runs, "
            "or remove the existing directory if it is no longer needed."
        )

    # --- fabric metadata (no full file read — only bbox + hash) -------------
    fabric_meta = _fabric_metadata(fabric_path, id_col, buffer_deg)

    # --- check for reusable raw data from a prior run -----------------------
    raw_source = _find_reusable_raw(workdir, fabric_meta["sha256"], run_id)

    # --- build directory skeleton -------------------------------------------
    _create_skeleton(run_dir, raw_source)

    # --- write run files ----------------------------------------------------
    _write_fabric_json(run_dir, fabric_meta)
    _write_config(run_dir, config_path)
    _write_credentials_template(run_dir)
    _write_manifest(run_dir, run_id, fabric_meta)

    return run_dir


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------


def _make_run_id(label: str | None = None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if label:
        return f"{ts}_{label}_v{__version__}"
    return f"{ts}_v{__version__}"


# ---------------------------------------------------------------------------
# Fabric metadata
# ---------------------------------------------------------------------------


def _fabric_metadata(fabric_path: Path, id_col: str, buffer_deg: float) -> dict:
    """Compute fabric hash and bbox without loading all geometry into memory."""
    import geopandas as gpd
    from shapely.geometry import box

    # sha256 of the raw file bytes
    sha256 = _sha256(fabric_path)

    # read only enough to get bbox and crs
    gdf = gpd.read_file(fabric_path, rows=None)
    native_crs = gdf.crs.to_string() if gdf.crs else "unknown"
    hru_count = len(gdf)

    # Reproject bounding box to WGS84 for use with remote APIs (CMR, etc.)
    native_bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    if gdf.crs and not gdf.crs.is_geographic:
        bbox_geom = gpd.GeoSeries([box(*native_bounds)], crs=gdf.crs)
        wgs84_bounds = bbox_geom.to_crs("EPSG:4326").total_bounds
    else:
        wgs84_bounds = native_bounds

    buffered = {
        "minx": float(wgs84_bounds[0]) - buffer_deg,
        "miny": float(wgs84_bounds[1]) - buffer_deg,
        "maxx": float(wgs84_bounds[2]) + buffer_deg,
        "maxy": float(wgs84_bounds[3]) + buffer_deg,
    }

    return {
        "path": str(fabric_path),
        "sha256": sha256,
        "crs": native_crs,
        "id_col": id_col,
        "hru_count": hru_count,
        "bbox": {
            "minx": float(wgs84_bounds[0]),
            "miny": float(wgs84_bounds[1]),
            "maxx": float(wgs84_bounds[2]),
            "maxy": float(wgs84_bounds[3]),
        },
        "bbox_buffered": buffered,
        "buffer_deg": buffer_deg,
    }


def _sha256(path: Path) -> str:
    """Compute SHA-256 of a file, or of all files in a directory (e.g. .gdb)."""
    h = hashlib.sha256()
    if path.is_dir():
        for child in sorted(path.rglob("*"), key=lambda p: str(p)):
            if child.is_file():
                h.update(str(child.relative_to(path)).encode())
                with child.open("rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
    else:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Reusable raw data detection
# ---------------------------------------------------------------------------


def _find_reusable_raw(workdir: Path, sha256: str, current_run_id: str) -> Path | None:
    """
    Scan existing runs for one that used the same fabric (by sha256).
    Returns the path to that run's data/raw/ dir, or None.
    """
    candidates = []
    for fabric_json in sorted(workdir.glob("*/fabric.json"), reverse=True):
        run_dir = fabric_json.parent
        if run_dir.name == current_run_id:
            continue
        try:
            meta = json.loads(fabric_json.read_text())
        except Exception:
            continue
        if meta.get("sha256") == sha256:
            raw_dir = run_dir / "data" / "raw"
            if raw_dir.exists():
                candidates.append((run_dir.name, raw_dir))

    if not candidates:
        return None

    prior_name, prior_raw = candidates[0]
    print(
        f"\nFound existing run with identical fabric:\n"
        f"  {prior_name}\n"
        f"  raw data at: {prior_raw}\n"
    )
    answer = input("Reuse raw downloads from that run? [Y/n] ").strip().lower()
    reuse = answer in ("", "y", "yes")
    return prior_raw if reuse else None


# ---------------------------------------------------------------------------
# Directory skeleton
# ---------------------------------------------------------------------------


def _create_skeleton(run_dir: Path, raw_source: Path | None) -> None:
    """Create the run directory tree."""
    from nhf_spatial_targets.catalog import sources as catalog_sources

    source_keys = list(catalog_sources().keys())

    (run_dir / "data" / "aggregated").mkdir(parents=True)
    (run_dir / "targets").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)

    raw_dir = run_dir / "data" / "raw"

    if raw_source is not None:
        # symlink to the prior run's raw directory
        raw_dir.symlink_to(raw_source, target_is_directory=True)
    else:
        raw_dir.mkdir(parents=True)
        for key in source_keys:
            (raw_dir / key).mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Written files
# ---------------------------------------------------------------------------


def _write_fabric_json(run_dir: Path, fabric_meta: dict) -> None:
    path = run_dir / "fabric.json"
    path.write_text(json.dumps(fabric_meta, indent=2))


def _write_config(run_dir: Path, config_path: Path) -> None:
    shutil.copy2(config_path, run_dir / "config.yml")


def _write_credentials_template(run_dir: Path) -> None:
    template = {
        "nasa_earthdata": {
            "_comment": "NASA Earthdata login — https://urs.earthdata.nasa.gov",
            "username": "",
            "password": "",
        },
        "sciencebase": {
            "_comment": "USGS ScienceBase — https://www.sciencebase.gov",
            "username": "",
            "password": "",
        },
        "watergap": {
            "_comment": "WaterGAP portal — https://www.watermodel.net (if used)",
            "username": "",
            "password": "",
        },
    }
    path = run_dir / ".credentials.yml"
    path.write_text(
        "# nhf-spatial-targets run credentials\n"
        "# Fill in before running the pipeline.\n"
        "# This file is gitignored — do not commit it.\n\n"
        + yaml.dump(template, default_flow_style=False, sort_keys=False)
    )


def _write_manifest(run_dir: Path, run_id: str, fabric_meta: dict) -> None:
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "nhf_spatial_targets_version": __version__,
        "fabric": {
            "path": fabric_meta["path"],
            "sha256": fabric_meta["sha256"],
            "crs": fabric_meta["crs"],
            "id_col": fabric_meta["id_col"],
            "hru_count": fabric_meta["hru_count"],
        },
        "sources": {},  # populated by fetch step
        "steps": [],  # populated as pipeline runs: fetch → aggregate → targets
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
