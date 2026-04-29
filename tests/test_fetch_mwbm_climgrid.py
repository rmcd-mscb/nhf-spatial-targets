"""Tests for MWBM ClimGrid manual-placement registration.

The ScienceBase publisher gates ClimGrid_WBM.nc behind a CAPTCHA, so
``fetch_mwbm_climgrid`` does not download — it fingerprints and
validates whatever the operator has placed at the expected path. These
tests cover the registration paths (initial fingerprint, idempotent
no-op, re-fingerprint after replacement), the validation paths
(missing variable, wrong cell_methods, corrupt NetCDF, zero-byte
file), the period-validation gate, and the concurrent-writer guard.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from nhf_spatial_targets.fetch import mwbm_climgrid as _mwbm_module
from nhf_spatial_targets.fetch.mwbm_climgrid import fetch_mwbm_climgrid


@pytest.fixture(autouse=True)
def _short_stability_window(monkeypatch):
    """Shrink the stability window to keep the suite fast.

    Production default is 0.5s per invocation. Tests don't need to
    wait. The dedicated concurrent-writer test patches `time.sleep`
    directly to a value its mtime-bump trigger can race within.
    """
    monkeypatch.setattr(_mwbm_module, "_STABILITY_SECONDS", 0.0)


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


def _place_file(workdir: Path) -> Path:
    """Materialize a valid dummy NetCDF at the canonical datastore path."""
    nc_dir = workdir / "datastore" / "mwbm_climgrid"
    nc_dir.mkdir(parents=True)
    nc_path = nc_dir / "ClimGrid_WBM.nc"
    _write_dummy_nc(nc_path)
    return nc_path


# ---------------------------------------------------------------------------
# Period gate
# ---------------------------------------------------------------------------


def test_period_outside_data_range_rejected(tmp_path):
    """Periods outside 1900/2020 raise ValueError before touching the file."""
    workdir = _make_project(tmp_path)
    with pytest.raises(ValueError, match="outside the MWBM-ClimGrid data range"):
        fetch_mwbm_climgrid(workdir=workdir, period="1850/1900")
    with pytest.raises(ValueError, match="outside the MWBM-ClimGrid data range"):
        fetch_mwbm_climgrid(workdir=workdir, period="2000/2025")


@pytest.mark.parametrize("period", ["1900/1900", "2020/2020", "1900/2020"])
def test_period_boundary_inclusive(tmp_path, period):
    """The publisher-usable window is inclusive at both ends."""
    workdir = _make_project(tmp_path)
    _place_file(workdir)
    result = fetch_mwbm_climgrid(workdir=workdir, period=period)
    assert result["period"] == period


@pytest.mark.parametrize("period", ["1899/1900", "2020/2021"])
def test_period_boundary_off_by_one_rejected(tmp_path, period):
    """A single year outside the inclusive window still raises ValueError."""
    workdir = _make_project(tmp_path)
    with pytest.raises(ValueError, match="outside the MWBM-ClimGrid data range"):
        fetch_mwbm_climgrid(workdir=workdir, period=period)


# ---------------------------------------------------------------------------
# Missing-file handling — the headline of the manual-placement design.
# ---------------------------------------------------------------------------


def test_missing_file_raises_with_download_instructions(tmp_path):
    """Operator hasn't placed the file → FileNotFoundError pointing at docs."""
    workdir = _make_project(tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
    msg = str(exc_info.value)
    assert "ClimGrid_WBM.nc" in msg
    assert "sciencebase.gov" in msg.lower() or "ScienceBase" in msg
    assert "docs/sources/mwbm_climgrid.md" in msg


def test_zero_byte_file_rejected(tmp_path):
    """A zero-byte file (interrupted browser download) raises RuntimeError."""
    workdir = _make_project(tmp_path)
    nc_dir = workdir / "datastore" / "mwbm_climgrid"
    nc_dir.mkdir(parents=True)
    (nc_dir / "ClimGrid_WBM.nc").touch()
    with pytest.raises(RuntimeError, match="zero bytes"):
        fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")


# ---------------------------------------------------------------------------
# Registration paths
# ---------------------------------------------------------------------------


def test_initial_registration_writes_manifest(tmp_path):
    """First call after manual placement → fingerprint + manifest write."""
    import json

    workdir = _make_project(tmp_path)
    nc_path = _place_file(workdir)
    expected_sha = hashlib.sha256(nc_path.read_bytes()).hexdigest()

    assert not (workdir / "manifest.json").exists()

    result = fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")

    assert result["file"]["sha256"] == expected_sha
    assert result["file"]["size_bytes"] == nc_path.stat().st_size
    assert result["file"]["manual_download"] is True
    assert result["doi"] == "10.5066/P9QCLGKM"
    assert result["source_key"] == "mwbm_climgrid"

    manifest = json.loads((workdir / "manifest.json").read_text())
    entry = manifest["sources"]["mwbm_climgrid"]
    assert entry["file"]["sha256"] == expected_sha
    assert entry["file"]["path"] == str(nc_path)


def test_idempotent_when_manifest_matches(tmp_path):
    """File + manifest agree on size and sha256 → no rewrite of manifest."""
    workdir = _make_project(tmp_path)
    nc_path = _place_file(workdir)

    # First call seeds the manifest.
    fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
    manifest_path = workdir / "manifest.json"
    first_mtime = manifest_path.stat().st_mtime
    first_contents = manifest_path.read_text()

    # Second call sees a matching fingerprint and skips the manifest
    # rewrite. We can't directly assert "no write happened" without
    # patching, but we can assert the content didn't change and the
    # returned record reuses the recorded download_timestamp.
    result = fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
    assert manifest_path.read_text() == first_contents
    assert manifest_path.stat().st_mtime == first_mtime

    sha = hashlib.sha256(nc_path.read_bytes()).hexdigest()
    assert result["file"]["sha256"] == sha


def test_rerun_after_replacement_rewrites_manifest(tmp_path):
    """Operator drops in a new file → fingerprint changes → manifest updates."""
    import json

    workdir = _make_project(tmp_path)
    nc_path = _place_file(workdir)
    fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
    first_manifest = json.loads((workdir / "manifest.json").read_text())
    first_sha = first_manifest["sources"]["mwbm_climgrid"]["file"]["sha256"]

    # Replace the file with different content (different bytes → different sha).
    nc_path.unlink()
    _write_dummy_nc(nc_path)
    # Touch to simulate a fresh placement; force at least 1 byte difference.
    with nc_path.open("ab") as f:
        f.write(b"\0")

    fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
    second_manifest = json.loads((workdir / "manifest.json").read_text())
    second_sha = second_manifest["sources"]["mwbm_climgrid"]["file"]["sha256"]
    assert second_sha != first_sha


# ---------------------------------------------------------------------------
# CF-metadata validation
# ---------------------------------------------------------------------------


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


def test_rejects_mismatched_cell_methods(tmp_path):
    """Non-None publisher cell_methods that disagrees with catalog raises."""
    workdir = _make_project(tmp_path)
    nc_dir = workdir / "datastore" / "mwbm_climgrid"
    nc_dir.mkdir(parents=True)
    _write_dummy_nc_with_bad_cell_methods(nc_dir / "ClimGrid_WBM.nc")

    with pytest.raises(RuntimeError, match="cell_methods"):
        fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
    # Manifest must NOT exist — failed validation must not write provenance.
    assert not (workdir / "manifest.json").exists()


def _write_dummy_nc_without_cell_methods(path: Path) -> None:
    """Like _write_dummy_nc but strips cell_methods on every variable.

    Mirrors the real ClimGrid_WBM.nc, where the publisher ships units /
    standard_name / long_name but omits cell_methods.
    """
    import pandas as pd

    times = pd.date_range("1900-01-01", periods=2, freq="MS")
    lats = np.array([40.0, 40.5], dtype=np.float64)
    lons = np.array([-105.0, -104.5], dtype=np.float64)
    rng = np.random.default_rng(0)
    attrs_no_cm = {"units": "mm", "long_name": "x", "standard_name": "x"}
    ds = xr.Dataset(
        data_vars={
            "runoff": (
                ("time", "latitude", "longitude"),
                rng.random((2, 2, 2)),
                attrs_no_cm,
            ),
            "aet": (
                ("time", "latitude", "longitude"),
                rng.random((2, 2, 2)),
                attrs_no_cm,
            ),
            "soilstorage": (
                ("time", "latitude", "longitude"),
                rng.random((2, 2, 2)),
                attrs_no_cm,
            ),
            "swe": (
                ("time", "latitude", "longitude"),
                rng.random((2, 2, 2)),
                attrs_no_cm,
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


def test_missing_cell_methods_warns_does_not_raise(tmp_path, caplog):
    """Silent cell_methods (publisher gap) → warn + register, do NOT raise.

    Mirrors the real ClimGrid_WBM.nc: the publisher ships units and
    standard_name but no cell_methods. The catalog declaration of
    `time: sum` etc. is documented provenance, not a hard claim about
    file attrs, so a silent file is acceptable as long as variables
    and units check out.
    """
    workdir = _make_project(tmp_path)
    nc_dir = workdir / "datastore" / "mwbm_climgrid"
    nc_dir.mkdir(parents=True)
    nc_path = nc_dir / "ClimGrid_WBM.nc"
    _write_dummy_nc_without_cell_methods(nc_path)

    with caplog.at_level("WARNING", logger="nhf_spatial_targets.fetch.mwbm_climgrid"):
        result = fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")

    # Manifest WAS written — the file is acceptable.
    assert (workdir / "manifest.json").exists()
    assert result["file"]["sha256"]

    # Each declared variable that lost its cell_methods got a warning.
    warning_text = "\n".join(r.getMessage() for r in caplog.records)
    for var in ("runoff", "aet", "soilstorage", "swe"):
        assert var in warning_text, f"missing warning for {var!r}"
    assert "no cell_methods attribute" in warning_text


def test_rejects_missing_variable(tmp_path):
    """Publisher dropping a variable raises before manifest is written."""
    import pandas as pd

    workdir = _make_project(tmp_path)
    nc_dir = workdir / "datastore" / "mwbm_climgrid"
    nc_dir.mkdir(parents=True)
    nc_path = nc_dir / "ClimGrid_WBM.nc"

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
    ds.to_netcdf(nc_path)
    ds.close()

    with pytest.raises(RuntimeError, match="missing variables"):
        fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
    assert not (workdir / "manifest.json").exists()


def test_rejects_corrupt_netcdf_file(tmp_path):
    """A non-NetCDF file (truncated browser download) raises RuntimeError."""
    workdir = _make_project(tmp_path)
    nc_dir = workdir / "datastore" / "mwbm_climgrid"
    nc_dir.mkdir(parents=True)
    (nc_dir / "ClimGrid_WBM.nc").write_bytes(b"not-a-netcdf-file-just-some-bytes")

    with pytest.raises(RuntimeError, match="Cannot open"):
        fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")


# ---------------------------------------------------------------------------
# Manifest hardening
# ---------------------------------------------------------------------------


def test_raises_on_corrupt_manifest(tmp_path):
    """A manifest.json that doesn't parse as JSON fails fast (no wasted rehash)."""
    workdir = _make_project(tmp_path)
    _place_file(workdir)
    (workdir / "manifest.json").write_text("{not valid json")

    with pytest.raises(ValueError, match="manifest.json.*corrupt"):
        fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")


# ---------------------------------------------------------------------------
# Stability / concurrent-writer guard
# ---------------------------------------------------------------------------


def test_verify_file_stable_detects_mtime_change(tmp_path, monkeypatch):
    """The stability helper raises if mtime changes during the window.

    Patches `time.sleep` to bump the file's mtime instead of waiting,
    so the test is deterministic and fast.
    """
    p = tmp_path / "in_progress.nc"
    p.write_bytes(b"some bytes")

    def _bump_mtime_during_sleep(_seconds):
        new_mtime = p.stat().st_mtime + 10
        os.utime(p, (new_mtime, new_mtime))

    monkeypatch.setattr(_mwbm_module.time, "sleep", _bump_mtime_during_sleep)

    with pytest.raises(RuntimeError, match="modified concurrently"):
        _mwbm_module._verify_file_stable(p)


def test_verify_file_stable_detects_size_change(tmp_path, monkeypatch):
    """The stability helper raises if size changes during the window."""
    p = tmp_path / "growing.nc"
    p.write_bytes(b"initial")

    def _grow_during_sleep(_seconds):
        with p.open("ab") as f:
            f.write(b"...more bytes from a concurrent writer...")

    monkeypatch.setattr(_mwbm_module.time, "sleep", _grow_during_sleep)

    with pytest.raises(RuntimeError, match="modified concurrently"):
        _mwbm_module._verify_file_stable(p)


def test_verify_file_stable_passes_when_quiescent(tmp_path):
    """Stable file (no concurrent writer) passes the stability check."""
    p = tmp_path / "quiet.nc"
    p.write_bytes(b"stable content")
    _mwbm_module._verify_file_stable(p)  # no exception


def test_rejects_concurrent_writer(tmp_path, monkeypatch):
    """Initial registration refuses to fingerprint a file being written."""
    workdir = _make_project(tmp_path)
    nc_path = _place_file(workdir)
    assert not (workdir / "manifest.json").exists()

    def _bump_mtime_during_sleep(_seconds):
        new_mtime = nc_path.stat().st_mtime + 10
        os.utime(nc_path, (new_mtime, new_mtime))

    monkeypatch.setattr(_mwbm_module.time, "sleep", _bump_mtime_during_sleep)

    with pytest.raises(RuntimeError, match="modified concurrently"):
        fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")

    # Manifest must NOT have been written — a bad-state fingerprint would
    # corrupt subsequent idempotency checks.
    assert not (workdir / "manifest.json").exists()
