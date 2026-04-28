"""Tests for MWBM ClimGrid-forced fetch module."""

from __future__ import annotations

import hashlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from nhf_spatial_targets.fetch.mwbm_climgrid import fetch_mwbm_climgrid


def _make_project(tmp_path: Path) -> Path:
    """Materialize a minimal valid project directory."""
    import json
    import yaml

    datastore = tmp_path / "datastore"
    datastore.mkdir()
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {
                    "path": str(tmp_path / "fabric.gpkg"),
                    "id_col": "nhm_id",
                },
                "datastore": str(datastore),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))
    return tmp_path


def _write_dummy_nc(path: Path) -> None:
    """Write a tiny CF-conformant NC mimicking ClimGrid_WBM.nc structure."""
    import pandas as pd

    times = pd.date_range("1900-01-01", periods=3, freq="MS")
    lats = np.array([40.0, 40.5, 41.0], dtype=np.float64)
    lons = np.array([-105.0, -104.5, -104.0], dtype=np.float64)
    rng = np.random.default_rng(0)
    data_vars = {
        "runoff": (
            ("time", "latitude", "longitude"),
            rng.random((3, 3, 3), dtype=np.float64),
            {"units": "mm", "cell_methods": "time: sum"},
        ),
        "aet": (
            ("time", "latitude", "longitude"),
            rng.random((3, 3, 3), dtype=np.float64),
            {"units": "mm", "cell_methods": "time: sum"},
        ),
        "soilstorage": (
            ("time", "latitude", "longitude"),
            rng.random((3, 3, 3), dtype=np.float64),
            {"units": "mm", "cell_methods": "time: point"},
        ),
        "swe": (
            ("time", "latitude", "longitude"),
            rng.random((3, 3, 3), dtype=np.float64),
            {"units": "mm", "cell_methods": "time: point"},
        ),
    }
    coords = {
        "time": ("time", times),
        "latitude": ("latitude", lats, {"units": "degrees_north", "axis": "Y"}),
        "longitude": ("longitude", lons, {"units": "degrees_east", "axis": "X"}),
    }
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds["time"].attrs.update({"axis": "T", "standard_name": "time"})
    ds.attrs["Conventions"] = "CF-1.6"
    ds.to_netcdf(path)
    ds.close()


@contextmanager
def _patch_sbsession_to_emit(target_dir: Path, filename: str = "ClimGrid_WBM.nc"):
    """Context manager that injects a fake sciencebasepy into sys.modules.

    Uses sys.modules injection (same pattern as test_reitz2017.py) so the
    lazy ``from sciencebasepy import SbSession`` inside fetch_mwbm_climgrid
    resolves to the mock rather than the real network client.
    """
    fake_session = MagicMock()
    fake_session.get_item.return_value = {"id": "64c948dbd34e70357a34c11e"}
    fake_session.get_item_file_info.return_value = [
        {"name": filename, "url": "https://example/fake", "size": 0}
    ]

    def _fake_download(url, name, dest_dir):
        out = Path(dest_dir) / name
        _write_dummy_nc(out)
        # Backfill the publisher size on the file_info record so the
        # post-download size check has the real number to compare.
        fake_session.get_item_file_info.return_value[0]["size"] = out.stat().st_size

    fake_session.download_file.side_effect = _fake_download

    fake_module = types.ModuleType("sciencebasepy")
    fake_module.SbSession = MagicMock(return_value=fake_session)
    original = sys.modules.get("sciencebasepy")
    sys.modules["sciencebasepy"] = fake_module
    try:
        yield fake_session
    finally:
        if original is not None:
            sys.modules["sciencebasepy"] = original
        else:
            sys.modules.pop("sciencebasepy", None)


def test_period_outside_data_range_rejected(tmp_path):
    """Periods outside 1900/2020 raise ValueError before any download."""
    workdir = _make_project(tmp_path)
    with pytest.raises(ValueError, match="outside the MWBM-ClimGrid data range"):
        fetch_mwbm_climgrid(workdir=workdir, period="1850/1900")
    with pytest.raises(ValueError, match="outside the MWBM-ClimGrid data range"):
        fetch_mwbm_climgrid(workdir=workdir, period="2000/2025")


def test_fetch_downloads_and_writes_manifest(tmp_path):
    import json

    workdir = _make_project(tmp_path)
    datastore = workdir / "datastore"

    with _patch_sbsession_to_emit(datastore):
        result = fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")

    nc_path = datastore / "mwbm_climgrid" / "ClimGrid_WBM.nc"
    assert nc_path.exists(), "downloaded file should land in datastore"

    expected_sha = hashlib.sha256(nc_path.read_bytes()).hexdigest()
    assert result["file"]["sha256"] == expected_sha
    assert result["file"]["size_bytes"] == nc_path.stat().st_size
    assert result["doi"] == "10.5066/P9QCLGKM"
    assert result["source_key"] == "mwbm_climgrid"

    manifest = json.loads((workdir / "manifest.json").read_text())
    entry = manifest["sources"]["mwbm_climgrid"]
    assert entry["file"]["sha256"] == expected_sha
    assert entry["file"]["path"] == str(nc_path)


def test_fetch_idempotent_when_manifest_matches(tmp_path):
    """Pre-seed file + manifest with matching sha256/size; fetch is a no-op."""
    import json

    workdir = _make_project(tmp_path)
    datastore = workdir / "datastore"
    nc_dir = datastore / "mwbm_climgrid"
    nc_dir.mkdir(parents=True)
    nc_path = nc_dir / "ClimGrid_WBM.nc"
    _write_dummy_nc(nc_path)

    sha = hashlib.sha256(nc_path.read_bytes()).hexdigest()
    size = nc_path.stat().st_size
    manifest = {
        "sources": {
            "mwbm_climgrid": {
                "source_key": "mwbm_climgrid",
                "file": {
                    "path": str(nc_path),
                    "size_bytes": size,
                    "sha256": sha,
                    "downloaded_utc": "2026-04-28T00:00:00+00:00",
                },
            }
        }
    }
    (workdir / "manifest.json").write_text(json.dumps(manifest))

    with _patch_sbsession_to_emit(datastore) as fake_session:
        result = fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
        # No download/network call should have happened
        fake_session.download_file.assert_not_called()
        fake_session.get_item.assert_not_called()

    assert result["file"]["sha256"] == sha
    assert result["file"]["size_bytes"] == size


def test_fetch_repairs_missing_manifest(tmp_path):
    """File present, no manifest entry → hash + write manifest, skip download."""
    import json

    workdir = _make_project(tmp_path)
    datastore = workdir / "datastore"
    nc_dir = datastore / "mwbm_climgrid"
    nc_dir.mkdir(parents=True)
    nc_path = nc_dir / "ClimGrid_WBM.nc"
    _write_dummy_nc(nc_path)
    expected_sha = hashlib.sha256(nc_path.read_bytes()).hexdigest()

    # No manifest.json yet
    assert not (workdir / "manifest.json").exists()

    with _patch_sbsession_to_emit(datastore) as fake_session:
        result = fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
        fake_session.download_file.assert_not_called()
        fake_session.get_item.assert_not_called()

    assert result["file"]["sha256"] == expected_sha
    manifest = json.loads((workdir / "manifest.json").read_text())
    assert manifest["sources"]["mwbm_climgrid"]["file"]["sha256"] == expected_sha


def _write_dummy_nc_with_bad_cell_methods(path: Path) -> None:
    """Like _write_dummy_nc but flips runoff cell_methods to 'time: point'."""
    import pandas as pd

    times = pd.date_range("1900-01-01", periods=2, freq="MS")
    lats = np.array([40.0, 40.5], dtype=np.float64)
    lons = np.array([-105.0, -104.5], dtype=np.float64)
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        data_vars={
            "runoff": (
                ("time", "latitude", "longitude"),
                rng.random((2, 2, 2)),
                {"units": "mm", "cell_methods": "time: point"},  # WRONG
            ),
            "aet": (
                ("time", "latitude", "longitude"),
                rng.random((2, 2, 2)),
                {"units": "mm", "cell_methods": "time: sum"},
            ),
            "soilstorage": (
                ("time", "latitude", "longitude"),
                rng.random((2, 2, 2)),
                {"units": "mm", "cell_methods": "time: point"},
            ),
            "swe": (
                ("time", "latitude", "longitude"),
                rng.random((2, 2, 2)),
                {"units": "mm", "cell_methods": "time: point"},
            ),
        },
        coords={
            "time": ("time", times),
            "latitude": ("latitude", lats, {"units": "degrees_north", "axis": "Y"}),
            "longitude": ("longitude", lons, {"units": "degrees_east", "axis": "X"}),
        },
    )
    ds["time"].attrs.update({"axis": "T", "standard_name": "time"})
    ds.to_netcdf(path)
    ds.close()


def test_fetch_rejects_mismatched_cell_methods(tmp_path):
    """Publisher metadata divergence from catalog raises a clear error."""
    workdir = _make_project(tmp_path)

    # Adapt the existing helper inline: write a BAD dummy NC instead of good
    fake_session = MagicMock()
    fake_session.get_item.return_value = {"id": "x"}
    fake_session.get_item_file_info.return_value = [
        {"name": "ClimGrid_WBM.nc", "url": "x", "size": 0}
    ]

    def _bad_download(url, name, dest_dir):
        out = Path(dest_dir) / name
        _write_dummy_nc_with_bad_cell_methods(out)
        fake_session.get_item_file_info.return_value[0]["size"] = out.stat().st_size

    fake_session.download_file.side_effect = _bad_download

    fake_module = types.ModuleType("sciencebasepy")
    fake_module.SbSession = MagicMock(return_value=fake_session)
    original = sys.modules.get("sciencebasepy")
    sys.modules["sciencebasepy"] = fake_module
    try:
        with pytest.raises(RuntimeError, match="cell_methods"):
            fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
    finally:
        if original is not None:
            sys.modules["sciencebasepy"] = original
        else:
            sys.modules.pop("sciencebasepy", None)


def test_fetch_rejects_missing_variable(tmp_path):
    """Publisher dropping a variable raises before manifest is written."""
    import pandas as pd

    workdir = _make_project(tmp_path)

    fake_session = MagicMock()
    fake_session.get_item.return_value = {"id": "x"}
    fake_session.get_item_file_info.return_value = [
        {"name": "ClimGrid_WBM.nc", "url": "x", "size": 0}
    ]

    def _missing_var_download(url, name, dest_dir):
        out = Path(dest_dir) / name
        ds = xr.Dataset(
            data_vars={
                "runoff": (
                    ("time", "latitude", "longitude"),
                    np.zeros((1, 1, 1)),
                    {"units": "mm", "cell_methods": "time: sum"},
                ),
                # aet, soilstorage, swe missing
            },
            coords={
                "time": ("time", pd.date_range("1900-01-01", periods=1, freq="MS")),
                "latitude": ("latitude", [40.0], {"axis": "Y"}),
                "longitude": ("longitude", [-105.0], {"axis": "X"}),
            },
        )
        ds["time"].attrs.update({"axis": "T", "standard_name": "time"})
        ds.to_netcdf(out)
        ds.close()
        fake_session.get_item_file_info.return_value[0]["size"] = out.stat().st_size

    fake_session.download_file.side_effect = _missing_var_download

    fake_module = types.ModuleType("sciencebasepy")
    fake_module.SbSession = MagicMock(return_value=fake_session)
    original = sys.modules.get("sciencebasepy")
    sys.modules["sciencebasepy"] = fake_module
    try:
        with pytest.raises(RuntimeError, match="missing variables"):
            fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
    finally:
        if original is not None:
            sys.modules["sciencebasepy"] = original
        else:
            sys.modules.pop("sciencebasepy", None)


# ---------------------------------------------------------------------------
# Polished-tier coverage: gaps surfaced by PR #76 review (test analyzer +
# silent-failure-hunter). Each test pins a specific failure or boundary.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("period", ["1900/1900", "2020/2020", "1900/2020"])
def test_period_boundary_inclusive(tmp_path, period):
    """The publisher-usable window is inclusive at both ends."""
    workdir = _make_project(tmp_path)
    datastore = workdir / "datastore"
    with _patch_sbsession_to_emit(datastore):
        result = fetch_mwbm_climgrid(workdir=workdir, period=period)
    assert result["period"] == period


@pytest.mark.parametrize("period", ["1899/1900", "2020/2021"])
def test_period_boundary_off_by_one_rejected(tmp_path, period):
    """A single year outside the inclusive window still raises ValueError."""
    workdir = _make_project(tmp_path)
    with pytest.raises(ValueError, match="outside the MWBM-ClimGrid data range"):
        fetch_mwbm_climgrid(workdir=workdir, period=period)


def test_fetch_redownloads_when_sha256_mismatch(tmp_path):
    """File present + manifest size matches but sha256 does NOT → re-download."""
    import json

    workdir = _make_project(tmp_path)
    datastore = workdir / "datastore"
    nc_dir = datastore / "mwbm_climgrid"
    nc_dir.mkdir(parents=True)
    nc_path = nc_dir / "ClimGrid_WBM.nc"
    _write_dummy_nc(nc_path)

    # Manifest claims same size but a deliberately wrong sha256.
    size = nc_path.stat().st_size
    bogus_sha = "0" * 64
    actual_sha = hashlib.sha256(nc_path.read_bytes()).hexdigest()
    assert bogus_sha != actual_sha
    (workdir / "manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "mwbm_climgrid": {
                        "source_key": "mwbm_climgrid",
                        "file": {
                            "path": str(nc_path),
                            "size_bytes": size,
                            "sha256": bogus_sha,
                            "downloaded_utc": "2026-04-28T00:00:00+00:00",
                        },
                    }
                }
            }
        )
    )

    with _patch_sbsession_to_emit(datastore) as fake_session:
        result = fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
        # Re-download path was taken.
        fake_session.download_file.assert_called_once()

    # New manifest carries the real sha (post-redownload), not the bogus one.
    manifest = json.loads((workdir / "manifest.json").read_text())
    written_sha = manifest["sources"]["mwbm_climgrid"]["file"]["sha256"]
    assert written_sha != bogus_sha
    assert result["file"]["sha256"] == written_sha


def test_fetch_rejects_corrupt_netcdf_file(tmp_path):
    """A non-NetCDF file (truncated/corrupt download) raises RuntimeError."""
    workdir = _make_project(tmp_path)

    fake_session = MagicMock()
    fake_session.get_item.return_value = {"id": "x"}
    fake_session.get_item_file_info.return_value = [
        {"name": "ClimGrid_WBM.nc", "url": "x", "size": 0}
    ]

    def _corrupt_download(url, name, dest_dir):
        out = Path(dest_dir) / name
        out.write_bytes(b"not-a-netcdf-file-just-some-bytes")
        fake_session.get_item_file_info.return_value[0]["size"] = out.stat().st_size

    fake_session.download_file.side_effect = _corrupt_download

    fake_module = types.ModuleType("sciencebasepy")
    fake_module.SbSession = MagicMock(return_value=fake_session)
    original = sys.modules.get("sciencebasepy")
    sys.modules["sciencebasepy"] = fake_module
    try:
        with pytest.raises(RuntimeError, match="Cannot open downloaded NetCDF"):
            fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
    finally:
        if original is not None:
            sys.modules["sciencebasepy"] = original
        else:
            sys.modules.pop("sciencebasepy", None)


def test_fetch_repair_rejects_zero_byte_file(tmp_path):
    """File present + zero bytes + no manifest → repair branch raises."""
    workdir = _make_project(tmp_path)
    datastore = workdir / "datastore"
    nc_dir = datastore / "mwbm_climgrid"
    nc_dir.mkdir(parents=True)
    nc_path = nc_dir / "ClimGrid_WBM.nc"
    nc_path.touch()  # zero bytes
    assert nc_path.stat().st_size == 0
    assert not (workdir / "manifest.json").exists()

    with pytest.raises(RuntimeError, match="zero bytes"):
        fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")


def test_fetch_raises_on_corrupt_manifest(tmp_path):
    """A manifest.json that doesn't parse as JSON fails fast (no wasted rehash)."""
    workdir = _make_project(tmp_path)
    datastore = workdir / "datastore"
    nc_dir = datastore / "mwbm_climgrid"
    nc_dir.mkdir(parents=True)
    _write_dummy_nc(nc_dir / "ClimGrid_WBM.nc")
    (workdir / "manifest.json").write_text("{not valid json")

    with pytest.raises(ValueError, match="manifest.json.*corrupt"):
        fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")


def test_fetch_raises_when_file_not_in_sciencebase_item(tmp_path):
    """If the publisher renames the file, the lookup miss is reported clearly."""
    workdir = _make_project(tmp_path)

    fake_session = MagicMock()
    fake_session.get_item.return_value = {"id": "64c948dbd34e70357a34c11e"}
    # Item exists but has a different filename than the catalog declares.
    fake_session.get_item_file_info.return_value = [
        {"name": "different_name.nc", "url": "x", "size": 1234}
    ]

    fake_module = types.ModuleType("sciencebasepy")
    fake_module.SbSession = MagicMock(return_value=fake_session)
    original = sys.modules.get("sciencebasepy")
    sys.modules["sciencebasepy"] = fake_module
    try:
        with pytest.raises(RuntimeError, match="not found in ScienceBase item"):
            fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
    finally:
        if original is not None:
            sys.modules["sciencebasepy"] = original
        else:
            sys.modules.pop("sciencebasepy", None)
