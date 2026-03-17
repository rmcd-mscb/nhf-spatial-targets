"""Tests for PANGAEA fetch module (WaterGAP 2.2d)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture()
def _mock_pangaeapy():
    """Ensure ``pangaeapy`` is importable even when not installed.

    Injects a stub module into ``sys.modules`` so that
    ``from pangaeapy import PanDataSet`` inside *fetch_watergap22d*
    resolves to a ``MagicMock``.  Tests that need to control
    ``PanDataSet`` behaviour should layer their own ``patch`` on top.
    """
    fake = types.ModuleType("pangaeapy")
    fake.PanDataSet = MagicMock()  # default; overridden per-test
    original = sys.modules.get("pangaeapy")
    sys.modules["pangaeapy"] = fake
    yield fake
    if original is not None:
        sys.modules["pangaeapy"] = original
    else:
        sys.modules.pop("pangaeapy", None)


_RAW_FILENAME = "watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4"
_RAW_STEM = Path(_RAW_FILENAME).stem


def _make_watergap_nc(path: Path, n_times: int = 24) -> Path:
    """Create a synthetic WaterGAP-style NC4 with non-CF time encoding.

    Mimics the real file: time as float offsets with
    units='months since 1901-01-01', calendar='proleptic_gregorian'.
    """
    time_vals = np.arange(n_times, dtype=np.float32)
    lat = np.array([89.75, 45.25, 0.25, -44.75], dtype=np.float32)
    lon = np.array([-179.75, -90.25, 0.25, 89.75], dtype=np.float32)
    data = np.random.rand(n_times, len(lat), len(lon)).astype(np.float32)

    ds = xr.Dataset(
        {"qrdif": (["time", "lat", "lon"], data)},
        coords={
            "time": xr.Variable(
                "time",
                time_vals,
                attrs={
                    "units": "months since 1901-01-01",
                    "calendar": "proleptic_gregorian",
                    "standard_name": "time",
                },
            ),
            "lat": xr.Variable(
                "lat",
                lat,
                attrs={
                    "units": "degrees_north",
                    "standard_name": "latitude",
                },
            ),
            "lon": xr.Variable(
                "lon",
                lon,
                attrs={
                    "units": "degrees_east",
                    "standard_name": "longitude",
                },
            ),
        },
        attrs={
            "conventions": "partly ALMA, CF and ISIMIP2b protocol",
            "title": "Test WaterGAP",
        },
    )
    ds["qrdif"].attrs.update(
        {
            "standard_name": "qrdif",
            "long_name": "diffuse groundwater recharge",
            "units": "kg m-2 s-1",
        }
    )
    ds.to_netcdf(path, format="NETCDF4")
    return path


def test_cf_fixup_reconstructs_time(tmp_path: Path):
    """Time coordinate must be proper datetime64 after CF fix-up."""
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    raw = _make_watergap_nc(tmp_path / "raw.nc4", n_times=24)
    fixed = _cf_fixup(raw, tmp_path / "fixed.nc")

    ds = xr.open_dataset(fixed)
    assert ds.time.dtype == np.dtype("datetime64[ns]")
    # 24 months from 1901-01 = 1901-01 through 1902-12
    assert str(ds.time.values[0])[:7] == "1901-01"
    assert str(ds.time.values[12])[:7] == "1902-01"
    assert str(ds.time.values[23])[:7] == "1902-12"
    ds.close()


def test_cf_fixup_adds_grid_mapping(tmp_path: Path):
    """CF fix-up must add crs variable and grid_mapping attr to qrdif."""
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    raw = _make_watergap_nc(tmp_path / "raw.nc4")
    fixed = _cf_fixup(raw, tmp_path / "fixed.nc")

    ds = xr.open_dataset(fixed)
    assert "crs" in ds.data_vars
    assert ds["qrdif"].attrs.get("grid_mapping") == "crs"
    assert ds["crs"].attrs.get("grid_mapping_name") == "latitude_longitude"
    ds.close()


def test_cf_fixup_sets_conventions(tmp_path: Path):
    """CF fix-up must set Conventions to CF-1.6."""
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    raw = _make_watergap_nc(tmp_path / "raw.nc4")
    fixed = _cf_fixup(raw, tmp_path / "fixed.nc")

    ds = xr.open_dataset(fixed)
    assert ds.attrs.get("Conventions") == "CF-1.6"
    ds.close()


def test_cf_fixup_coordinate_metadata(tmp_path: Path):
    """CF fix-up sets coordinate attrs and time_bnds for monthly data."""
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    raw = _make_watergap_nc(tmp_path / "raw.nc4", n_times=3)
    fixed = _cf_fixup(raw, tmp_path / "fixed.nc")

    ds = xr.open_dataset(fixed)
    # Coordinate names
    assert "lat" in ds.dims
    assert "lon" in ds.dims
    # Coordinate attrs
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lat.attrs["units"] == "degrees_north"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert ds.lon.attrs["units"] == "degrees_east"
    assert ds.time.attrs["standard_name"] == "time"
    # time_bnds for monthly data
    assert "time_bnds" in ds.data_vars
    assert ds.time.attrs.get("bounds") == "time_bnds"
    # No spatial_ref
    assert "spatial_ref" not in ds.data_vars
    assert "spatial_ref" not in ds.coords
    ds.close()


def test_cf_fixup_preserves_data(tmp_path: Path):
    """CF fix-up must not alter the qrdif data values."""
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    raw = _make_watergap_nc(tmp_path / "raw.nc4", n_times=12)
    raw_ds = xr.open_dataset(raw, decode_times=False)
    raw_values = raw_ds["qrdif"].values.copy()
    raw_ds.close()

    fixed = _cf_fixup(raw, tmp_path / "fixed.nc")
    fixed_ds = xr.open_dataset(fixed)
    np.testing.assert_array_equal(fixed_ds["qrdif"].values, raw_values)
    fixed_ds.close()


# --- fetch_watergap22d tests ---


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """Create a minimal run workspace."""
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "watergap22d").mkdir(parents=True)
    fabric = {
        "bbox_buffered": {
            "minx": -125.1,
            "miny": 23.9,
            "maxx": -65.9,
            "maxy": 50.1,
        }
    }
    (rd / "fabric.json").write_text(json.dumps(fabric))
    return rd


def _mock_pangaea_download(
    run_dir: Path,
    mock_pangaeapy,
    *,
    n_times: int = 12,
) -> MagicMock:
    """Set up a PanDataSet mock that returns a synthetic NC4 file.

    Returns the mock PanDataSet instance for further assertions.
    """
    cache_dir = run_dir / "pangaea_cache"
    cache_dir.mkdir(exist_ok=True)
    cached_file = cache_dir / _RAW_FILENAME
    _make_watergap_nc(cached_file, n_times=n_times)

    mock_ds = MagicMock()
    mock_data = pd.DataFrame({"File name": {30: _RAW_STEM}})
    mock_ds.data = mock_data
    mock_ds.download.return_value = [cached_file]
    mock_pangaeapy.PanDataSet = MagicMock(return_value=mock_ds)
    return mock_ds


def test_missing_fabric_raises(tmp_path: Path):
    """FileNotFoundError when fabric.json is absent."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    rd = tmp_path / "run"
    rd.mkdir()
    with pytest.raises(FileNotFoundError, match="fabric.json"):
        fetch_watergap22d(run_dir=rd, period="2000/2009")


def test_malformed_fabric_raises(tmp_path: Path):
    """ValueError when fabric.json is missing bbox_buffered."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "fabric.json").write_text(json.dumps({"wrong_key": {}}))
    with pytest.raises(ValueError, match="malformed"):
        fetch_watergap22d(run_dir=rd, period="2000/2009")


def test_skips_existing_file(run_dir: Path, _mock_pangaeapy):
    """Skip download when CF-corrected file already exists."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    output_dir = run_dir / "data" / "raw" / "watergap22d"
    cf_file = output_dir / "watergap22d_qrdif_cf.nc"
    _make_watergap_nc(output_dir / _RAW_FILENAME, n_times=12)

    # Create a pre-existing CF-corrected file
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    _cf_fixup(output_dir / _RAW_FILENAME, cf_file)

    mock_pan = MagicMock()
    _mock_pangaeapy.PanDataSet = mock_pan
    result = fetch_watergap22d(run_dir=run_dir, period="2000/2009")
    mock_pan.assert_not_called()
    assert result["source_key"] == "watergap22d"


def test_raw_exists_skips_download(run_dir: Path, _mock_pangaeapy):
    """Skip download when raw file exists but CF file does not."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    output_dir = run_dir / "data" / "raw" / "watergap22d"
    _make_watergap_nc(output_dir / _RAW_FILENAME, n_times=12)

    mock_pan = MagicMock()
    _mock_pangaeapy.PanDataSet = mock_pan
    result = fetch_watergap22d(run_dir=run_dir, period="2000/2009")

    # PanDataSet should not be called — raw file already present
    mock_pan.assert_not_called()
    # CF file should still be created from the existing raw file
    assert (output_dir / "watergap22d_qrdif_cf.nc").exists()
    assert result["source_key"] == "watergap22d"


def test_downloads_and_updates_manifest(run_dir: Path, _mock_pangaeapy):
    """Download via pangaeapy, CF fix-up, and manifest update."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    output_dir = run_dir / "data" / "raw" / "watergap22d"
    _mock_pangaea_download(run_dir, _mock_pangaeapy)

    result = fetch_watergap22d(run_dir=run_dir, period="2000/2009")

    # Verify raw file was moved to output dir
    assert (output_dir / _RAW_FILENAME).exists()
    # Verify CF-corrected file was created
    assert (output_dir / "watergap22d_qrdif_cf.nc").exists()
    # Verify manifest updated
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "watergap22d" in manifest["sources"]
    assert manifest["sources"]["watergap22d"]["license"] == "CC BY-NC 4.0"
    assert result["source_key"] == "watergap22d"


def test_preserves_original_file(run_dir: Path, _mock_pangaeapy):
    """Both original and CF-corrected files must exist after fetch."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    output_dir = run_dir / "data" / "raw" / "watergap22d"
    _mock_pangaea_download(run_dir, _mock_pangaeapy)

    fetch_watergap22d(run_dir=run_dir, period="2000/2009")

    assert (output_dir / _RAW_FILENAME).exists(), "Original file must be preserved"
    assert (output_dir / "watergap22d_qrdif_cf.nc").exists(), (
        "CF-corrected file must exist"
    )


def test_manifest_preserves_existing_sources(run_dir: Path, _mock_pangaeapy):
    """Manifest merge must not overwrite entries from other sources."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    # Pre-populate manifest with another source
    existing_manifest = {
        "sources": {
            "merra2": {"source_key": "merra2", "period": "1980/2020"},
        },
        "steps": [],
    }
    (run_dir / "manifest.json").write_text(json.dumps(existing_manifest))

    _mock_pangaea_download(run_dir, _mock_pangaeapy)
    fetch_watergap22d(run_dir=run_dir, period="2000/2009")

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "merra2" in manifest["sources"], "Existing source must be preserved"
    assert "watergap22d" in manifest["sources"], "New source must be added"


def test_corrupt_manifest_raises(run_dir: Path, _mock_pangaeapy):
    """ValueError when manifest.json exists but is corrupt."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    (run_dir / "manifest.json").write_text("not valid json{{{")

    _mock_pangaea_download(run_dir, _mock_pangaeapy)
    with pytest.raises(ValueError, match="corrupt"):
        fetch_watergap22d(run_dir=run_dir, period="2000/2009")


def test_download_failure_raises(run_dir: Path, _mock_pangaeapy):
    """RuntimeError when pangaeapy download fails."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    mock_ds = MagicMock()
    mock_data = pd.DataFrame({"File name": {30: _RAW_STEM}})
    mock_ds.data = mock_data
    mock_ds.download.side_effect = Exception("PANGAEA unavailable")
    _mock_pangaeapy.PanDataSet = MagicMock(return_value=mock_ds)

    with pytest.raises(RuntimeError, match="PANGAEA"):
        fetch_watergap22d(run_dir=run_dir, period="2000/2009")


def test_empty_download_raises(run_dir: Path, _mock_pangaeapy):
    """RuntimeError when pangaeapy returns empty file list."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    mock_ds = MagicMock()
    mock_data = pd.DataFrame({"File name": {30: _RAW_STEM}})
    mock_ds.data = mock_data
    mock_ds.download.return_value = []
    _mock_pangaeapy.PanDataSet = MagicMock(return_value=mock_ds)

    with pytest.raises(RuntimeError, match="no files"):
        fetch_watergap22d(run_dir=run_dir, period="2000/2009")


def test_wrong_file_index_raises(run_dir: Path, _mock_pangaeapy):
    """RuntimeError when file at expected index doesn't match expected filename."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    mock_ds = MagicMock()
    mock_data = pd.DataFrame({"File name": {30: "wrong_filename.nc4"}})
    mock_ds.data = mock_data
    _mock_pangaeapy.PanDataSet = MagicMock(return_value=mock_ds)

    with pytest.raises(RuntimeError, match="file index"):
        fetch_watergap22d(run_dir=run_dir, period="2000/2009")


def test_cli_watergap22d_nonexistent_run_dir(tmp_path: Path):
    """CLI exits with error for nonexistent run directory."""
    from nhf_spatial_targets.cli import app

    with pytest.raises(SystemExit) as exc_info:
        app(
            [
                "fetch",
                "watergap22d",
                "--run-dir",
                str(tmp_path / "nope"),
                "--period",
                "2000/2009",
            ],
            exit_on_error=False,
        )
    assert exc_info.value.code == 2


@pytest.mark.integration
def test_fetch_watergap22d_real_download(tmp_path: Path):
    """Integration: fetch from real PANGAEA and verify CF-compliant output.

    Requires network access. Run with: pixi run -e dev test-integration
    """
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "watergap22d").mkdir(parents=True)
    fabric = {
        "bbox_buffered": {
            "minx": -125.1,
            "miny": 23.9,
            "maxx": -65.9,
            "maxy": 50.1,
        }
    }
    (rd / "fabric.json").write_text(json.dumps(fabric))

    result = fetch_watergap22d(run_dir=rd, period="2000/2009")

    assert result["source_key"] == "watergap22d"

    cf_path = rd / "data" / "raw" / "watergap22d" / "watergap22d_qrdif_cf.nc"
    assert cf_path.exists()

    ds = xr.open_dataset(cf_path)
    assert "qrdif" in ds.data_vars
    assert ds.time.dtype == np.dtype("datetime64[ns]")
    assert ds.attrs.get("Conventions") == "CF-1.6"
    assert "crs" in ds.data_vars
    assert ds["qrdif"].attrs.get("grid_mapping") == "crs"
    ds.close()

    raw_path = rd / "data" / "raw" / "watergap22d" / _RAW_FILENAME
    assert raw_path.exists()
