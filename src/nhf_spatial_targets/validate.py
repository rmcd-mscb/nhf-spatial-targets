"""Preflight validation for an nhf-spatial-targets workspace."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from nhf_spatial_targets import __version__
from nhf_spatial_targets.workspace import make_dir

# The eight source keys whose raw-data subdirectories are created.
_SOURCE_KEYS: list[str] = [
    "merra2",
    "nldas_mosaic",
    "nldas_noah",
    "ncep_ncar",
    "mod16a2_v061",
    "mod10c1_v061",
    "watergap22d",
    "reitz2017",
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_workspace(workdir: Path) -> None:
    """Run preflight checks on a workspace directory, failing fast on errors.

    On success, writes ``fabric.json`` and ``manifest.json`` into *workdir*.
    """
    workdir = Path(workdir).resolve()

    # 1. Config completeness
    config = _check_config(workdir)

    # 2. Fabric exists
    fabric_path = Path(config["fabric"]["path"])
    id_col = config["fabric"]["id_col"]
    _check_fabric_exists(fabric_path)

    # 3-4. Fabric metadata (sha256, bbox, hru_count) + id_col check
    buffer_deg = float(config.get("buffer_deg", 0.1))
    fabric_meta = _fabric_metadata(fabric_path, id_col, buffer_deg)

    # 5. Credentials
    _check_credentials(workdir)

    # 6. Datastore — create directory tree
    datastore = Path(config["datastore"])
    dir_mode_str = config.get("dir_mode")
    dir_mode = int(dir_mode_str, 8) if dir_mode_str else None
    _ensure_datastore(datastore, dir_mode)

    # 7. Catalog consistency
    _check_catalog_consistency()

    # Write outputs
    _write_fabric_json(workdir, fabric_meta)
    _write_manifest(workdir, fabric_meta)


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------


def _check_config(workdir: Path) -> dict:
    """Load and validate config.yml."""
    config_path = workdir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yml not found in {workdir}. "
            f"Run 'nhf-targets init --workdir {workdir}' first."
        )
    try:
        config = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Cannot parse config.yml in {workdir}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(
            f"config.yml in {workdir} is empty or malformed. "
            f"It must contain YAML key-value pairs."
        )

    fabric_cfg = config.get("fabric", {}) or {}
    if not fabric_cfg.get("path"):
        raise ValueError("fabric.path must be non-empty in config.yml")
    if not fabric_cfg.get("id_col"):
        raise ValueError("fabric.id_col must be non-empty in config.yml")
    if not config.get("datastore"):
        raise ValueError("datastore must be non-empty in config.yml")

    return config


def _check_fabric_exists(fabric_path: Path) -> None:
    if not fabric_path.exists():
        raise FileNotFoundError(f"Fabric file not found: {fabric_path}")


def _check_id_column(gdf: object, id_col: str) -> None:
    """Check that id_col exists in a loaded GeoDataFrame."""
    if id_col not in gdf.columns:
        raise ValueError(
            f"Column '{id_col}' not found in fabric. "
            f"Available columns: {list(gdf.columns)}"
        )


def _check_credentials(workdir: Path) -> None:
    cred_path = workdir / ".credentials.yml"
    if not cred_path.exists():
        raise FileNotFoundError(
            f".credentials.yml not found in {workdir}. "
            "Create it with NASA Earthdata credentials before validating."
        )
    try:
        creds = yaml.safe_load(cred_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Cannot parse {cred_path}: {exc}. "
            f"Fix the YAML syntax in your credentials file."
        ) from exc
    earthdata = creds.get("nasa_earthdata", {}) or {}
    if not earthdata.get("username") or not earthdata.get("password"):
        raise ValueError(
            "Earthdata credentials incomplete — "
            "nasa_earthdata.username and nasa_earthdata.password "
            "must be non-empty in .credentials.yml"
        )


def _ensure_datastore(datastore: Path, dir_mode: int | None) -> None:
    make_dir(datastore, dir_mode=dir_mode)
    for key in _SOURCE_KEYS:
        make_dir(datastore / key, dir_mode=dir_mode)


def _check_catalog_consistency() -> None:
    from nhf_spatial_targets.catalog import sources, variables

    src_keys = set(sources().keys())
    for var_name, var_def in variables().items():
        for src in var_def.get("sources", []):
            if src not in src_keys:
                raise ValueError(
                    f"Variable '{var_name}' references source '{src}' "
                    f"which does not exist in catalog/sources.yml"
                )


# ---------------------------------------------------------------------------
# Fabric metadata (moved from init_run.py)
# ---------------------------------------------------------------------------


def _fabric_metadata(fabric_path: Path, id_col: str, buffer_deg: float) -> dict:
    """Compute fabric hash, bbox, and HRU count. Also validates id_col."""
    import geopandas as gpd
    from shapely.geometry import box

    sha256 = _sha256(fabric_path)

    if fabric_path.suffix.lower() in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(fabric_path)
    else:
        gdf = gpd.read_file(fabric_path, rows=None)
    # Check id_col here to avoid reading the fabric twice
    _check_id_column(gdf, id_col)

    native_crs = gdf.crs.to_string() if gdf.crs else "unknown"
    hru_count = len(gdf)

    native_bounds = gdf.total_bounds
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
# Output writers
# ---------------------------------------------------------------------------


def _write_fabric_json(workdir: Path, fabric_meta: dict) -> None:
    path = workdir / "fabric.json"
    path.write_text(json.dumps(fabric_meta, indent=2))


def _write_manifest(workdir: Path, fabric_meta: dict) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "nhf_spatial_targets_version": __version__,
        "fabric": {
            "path": fabric_meta["path"],
            "sha256": fabric_meta["sha256"],
            "crs": fabric_meta["crs"],
            "id_col": fabric_meta["id_col"],
            "hru_count": fabric_meta["hru_count"],
        },
        "sources": {},
        "steps": [],
    }
    path = workdir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
