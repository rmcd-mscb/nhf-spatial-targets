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
