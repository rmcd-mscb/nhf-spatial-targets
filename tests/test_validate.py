"""Tests for nhf_spatial_targets.validate — preflight project checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pytest
import yaml
from shapely.geometry import box

from nhf_spatial_targets.validate import (
    _SOURCE_KEYS,
    _import_cdsapi,
    validate_credentials,
    validate_workspace,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_fabric(tmp_path) -> Path:
    gdf = gpd.GeoDataFrame(
        {"nhm_id": [1, 2, 3]},
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2), box(2, 2, 3, 3)],
        crs="EPSG:4326",
    )
    path = tmp_path / "fabric.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def _write_config(
    workdir: Path,
    overrides: dict | None = None,
) -> Path:
    """Write a minimal config.yml, merging any *overrides* on top."""
    cfg: dict = {
        "fabric": {
            "path": "/nonexistent/fabric.gpkg",
            "id_col": "nhm_id",
        },
        "datastore": str(workdir / "datastore"),
    }
    if overrides:
        for key, val in overrides.items():
            if isinstance(val, dict) and key in cfg and isinstance(cfg[key], dict):
                cfg[key].update(val)
            else:
                cfg[key] = val
    path = workdir / "config.yml"
    path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    return path


def _write_credentials(
    workdir: Path,
    earthdata_user: str = "user",
    earthdata_pass: str = "pass",
    cds_url: str = "https://cds.climate.copernicus.eu/api",
    cds_key: str = "uid:testkey",
) -> Path:
    creds: dict = {
        "nasa_earthdata": {
            "username": earthdata_user,
            "password": earthdata_pass,
        },
        "cds": {
            "url": cds_url,
            "key": cds_key,
        },
    }
    path = workdir / ".credentials.yml"
    path.write_text(yaml.dump(creds, default_flow_style=False, sort_keys=False))
    return path


def _full_setup(workdir: Path, fabric_path: Path) -> None:
    """Write config + credentials pointing at the real fabric."""
    _write_config(
        workdir,
        overrides={
            "fabric": {"path": str(fabric_path), "id_col": "nhm_id"},
            "datastore": str(workdir / "datastore"),
        },
    )
    _write_credentials(workdir)


# ---------------------------------------------------------------------------
# Config completeness
# ---------------------------------------------------------------------------


def test_validate_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.yml"):
        validate_workspace(tmp_path)


def test_validate_empty_fabric_path(tmp_path):
    _write_config(tmp_path, overrides={"fabric": {"path": "", "id_col": "nhm_id"}})
    with pytest.raises(ValueError, match="fabric.path"):
        validate_workspace(tmp_path)


def test_validate_empty_datastore(tmp_path):
    _write_config(tmp_path, overrides={"datastore": ""})
    with pytest.raises(ValueError, match="datastore"):
        validate_workspace(tmp_path)


# ---------------------------------------------------------------------------
# Fabric checks
# ---------------------------------------------------------------------------


def test_validate_fabric_not_found(tmp_path):
    _write_config(
        tmp_path,
        overrides={"fabric": {"path": "/no/such/file.gpkg", "id_col": "nhm_id"}},
    )
    with pytest.raises(FileNotFoundError, match="Fabric file not found"):
        validate_workspace(tmp_path)


def test_validate_missing_id_col(tmp_path, minimal_fabric):
    _write_config(
        tmp_path,
        overrides={
            "fabric": {"path": str(minimal_fabric), "id_col": "bad_column"},
            "datastore": str(tmp_path / "datastore"),
        },
    )
    _write_credentials(tmp_path)
    with pytest.raises(ValueError, match="bad_column"):
        validate_workspace(tmp_path)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_validate_missing_credentials(tmp_path, minimal_fabric):
    _write_config(
        tmp_path,
        overrides={
            "fabric": {"path": str(minimal_fabric), "id_col": "nhm_id"},
            "datastore": str(tmp_path / "datastore"),
        },
    )
    with pytest.raises(FileNotFoundError, match=".credentials.yml"):
        validate_workspace(tmp_path)


def test_validate_empty_earthdata_creds(tmp_path, minimal_fabric):
    _write_config(
        tmp_path,
        overrides={
            "fabric": {"path": str(minimal_fabric), "id_col": "nhm_id"},
            "datastore": str(tmp_path / "datastore"),
        },
    )
    _write_credentials(tmp_path, earthdata_user="", earthdata_pass="")
    with pytest.raises(ValueError, match="Earthdata"):
        validate_workspace(tmp_path)


def test_validate_credentials_missing_cds(tmp_path: Path) -> None:
    creds = tmp_path / ".credentials.yml"
    creds.write_text(
        yaml.safe_dump({"nasa_earthdata": {"username": "u", "password": "p"}})
    )
    with pytest.raises(ValueError, match="cds"):
        validate_credentials(creds, required=["nasa_earthdata", "cds"])


def test_validate_credentials_with_cds(tmp_path: Path) -> None:
    creds = tmp_path / ".credentials.yml"
    creds.write_text(
        yaml.safe_dump(
            {
                "nasa_earthdata": {"username": "u", "password": "p"},
                "cds": {
                    "url": "https://cds.climate.copernicus.eu/api",
                    "key": "uid:abc",
                },
            }
        )
    )
    validate_credentials(creds, required=["nasa_earthdata", "cds"])  # no raise


def test_validate_credentials_with_cds_requires_cdsapi(monkeypatch) -> None:
    """If cds creds are required but cdsapi is not installed, _import_cdsapi raises."""
    monkeypatch.setitem(sys.modules, "cdsapi", None)
    with pytest.raises(ValueError, match="cdsapi"):
        _import_cdsapi()


# ---------------------------------------------------------------------------
# Output files: fabric.json, manifest.json
# ---------------------------------------------------------------------------


def test_validate_writes_fabric_json(tmp_path, minimal_fabric):
    _full_setup(tmp_path, minimal_fabric)
    validate_workspace(tmp_path)

    fabric = json.loads((tmp_path / "fabric.json").read_text())
    assert len(fabric["sha256"]) == 64
    assert fabric["id_col"] == "nhm_id"
    assert fabric["hru_count"] == 3
    assert "bbox" in fabric
    assert "bbox_buffered" in fabric
    assert fabric["buffer_deg"] == 0.1
    # Buffered box should be wider than the raw bbox
    assert fabric["bbox_buffered"]["minx"] < fabric["bbox"]["minx"]
    assert fabric["bbox_buffered"]["maxy"] > fabric["bbox"]["maxy"]


def test_validate_writes_manifest_json(tmp_path, minimal_fabric):
    _full_setup(tmp_path, minimal_fabric)
    validate_workspace(tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["sources"] == {}
    assert manifest["steps"] == []
    assert "fabric" in manifest
    assert manifest["fabric"]["id_col"] == "nhm_id"
    assert manifest["fabric"]["hru_count"] == 3
    assert "nhf_spatial_targets_version" in manifest
    assert "created_utc" in manifest


# ---------------------------------------------------------------------------
# Datastore creation
# ---------------------------------------------------------------------------


def test_validate_creates_datastore(tmp_path, minimal_fabric):
    ds = tmp_path / "new_datastore"
    _write_config(
        tmp_path,
        overrides={
            "fabric": {"path": str(minimal_fabric), "id_col": "nhm_id"},
            "datastore": str(ds),
        },
    )
    _write_credentials(tmp_path)
    assert not ds.exists()
    validate_workspace(tmp_path)
    assert ds.is_dir()


def test_validate_creates_source_subdirs(tmp_path, minimal_fabric):
    ds = tmp_path / "datastore"
    _write_config(
        tmp_path,
        overrides={
            "fabric": {"path": str(minimal_fabric), "id_col": "nhm_id"},
            "datastore": str(ds),
        },
    )
    _write_credentials(tmp_path)
    validate_workspace(tmp_path)

    created = sorted(p.name for p in ds.iterdir() if p.is_dir())
    assert created == sorted(_SOURCE_KEYS)


# ---------------------------------------------------------------------------
# _sha256 and _fabric_metadata (moved from test_init_run / test_cli)
# ---------------------------------------------------------------------------


def test_sha256_directory_consistent(tmp_path):
    """_sha256 on a directory returns a consistent digest."""
    from nhf_spatial_targets.validate import _sha256

    gdb = tmp_path / "test.gdb"
    gdb.mkdir()
    (gdb / "a.gdbtable").write_bytes(b"table data")
    (gdb / "b.gdbtablx").write_bytes(b"index data")

    h1 = _sha256(gdb)
    h2 = _sha256(gdb)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex digest


def test_sha256_directory_includes_filenames(tmp_path):
    """Two dirs with same bytes but different filenames produce different hashes."""
    from nhf_spatial_targets.validate import _sha256

    dir_a = tmp_path / "a.gdb"
    dir_a.mkdir()
    (dir_a / "file_one.dat").write_bytes(b"same content")

    dir_b = tmp_path / "b.gdb"
    dir_b.mkdir()
    (dir_b / "file_two.dat").write_bytes(b"same content")

    assert _sha256(dir_a) != _sha256(dir_b)


def test_fabric_metadata_reads_parquet(tmp_path):
    """_fabric_metadata can read a GeoParquet fabric file."""
    from nhf_spatial_targets.validate import _fabric_metadata

    gdf = gpd.GeoDataFrame(
        {"nhm_id": [1, 2]},
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
        crs="EPSG:4326",
    )
    parquet_path = tmp_path / "fabric.parquet"
    gdf.to_parquet(parquet_path)

    meta = _fabric_metadata(parquet_path, "nhm_id", buffer_deg=0.1)
    assert meta["hru_count"] == 2
    assert meta["id_col"] == "nhm_id"
    assert len(meta["sha256"]) == 64
    assert meta["bbox"]["minx"] == 0.0
    assert meta["bbox"]["maxy"] == 2.0


def test_fabric_metadata_reads_geoparquet(tmp_path):
    """_fabric_metadata handles the .geoparquet extension."""
    from nhf_spatial_targets.validate import _fabric_metadata

    gdf = gpd.GeoDataFrame(
        {"nhm_id": [1, 2]},
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
        crs="EPSG:4326",
    )
    geoparquet_path = tmp_path / "fabric.geoparquet"
    gdf.to_parquet(geoparquet_path)

    meta = _fabric_metadata(geoparquet_path, "nhm_id", buffer_deg=0.1)
    assert meta["hru_count"] == 2
