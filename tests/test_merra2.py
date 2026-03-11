"""Tests for MERRA-2 fetch module."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """Create a minimal run workspace with fabric.json."""
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "merra2").mkdir(parents=True)
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


def _mock_granule(name: str) -> MagicMock:
    """Create a mock granule object."""
    g = MagicMock()
    g.__str__ = lambda self: name
    return g


def _fake_download(run_dir: Path, n: int = 1) -> list[str]:
    """Create fake downloaded files and return their paths."""
    paths = []
    for i in range(n):
        f = run_dir / "data" / "raw" / "merra2" / f"MERRA2_2010{i + 1:02d}.nc4"
        f.write_bytes(b"fake")
        paths.append(str(f))
    return paths


# ---- Authentication --------------------------------------------------------


@patch("earthaccess.login")
def test_login_called(mock_login, run_dir):
    """earthaccess.login() is called before searching."""
    mock_login.return_value = MagicMock(authenticated=True)
    with patch("earthaccess.search_data", return_value=[]):
        with pytest.raises(ValueError, match="No granules found"):
            from nhf_spatial_targets.fetch.merra2 import fetch_merra2

            fetch_merra2(run_dir=run_dir, period="2010/2010")
    mock_login.assert_called_once()


@patch("earthaccess.login")
def test_login_failure_raises(mock_login, run_dir):
    """RuntimeError raised when earthaccess.login() fails."""
    mock_login.return_value = MagicMock(authenticated=False)
    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with pytest.raises(RuntimeError, match="Earthdata"):
        fetch_merra2(run_dir=run_dir, period="2010/2010")


@patch("earthaccess.login")
def test_login_returns_none_raises(mock_login, run_dir):
    """RuntimeError raised when earthaccess.login() returns None."""
    mock_login.return_value = None
    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with pytest.raises(RuntimeError, match="Earthdata"):
        fetch_merra2(run_dir=run_dir, period="2010/2010")


# ---- Search parameters -----------------------------------------------------


@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("earthaccess.login")
def test_search_params(mock_login, mock_search, mock_dl, run_dir):
    """search_data called with correct short_name, bbox tuple, and temporal."""
    mock_login.return_value = MagicMock(authenticated=True)
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_download(run_dir)

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    fetch_merra2(run_dir=run_dir, period="2005/2006")

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["short_name"] == "M2TMNXLND"
    assert call_kwargs["bounding_box"] == (-125.1, 23.9, -65.9, 50.1)
    assert call_kwargs["temporal"] == ("2005-01-01", "2006-12-31")


# ---- No results ------------------------------------------------------------


@patch("earthaccess.login")
@patch("earthaccess.search_data", return_value=[])
def test_no_granules_raises(mock_search, mock_login, run_dir):
    """ValueError raised when search returns zero granules."""
    mock_login.return_value = MagicMock(authenticated=True)
    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with pytest.raises(ValueError, match="No granules found"):
        fetch_merra2(run_dir=run_dir, period="2010/2010")


# ---- Output directory ------------------------------------------------------


@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("earthaccess.login")
def test_output_dir(mock_login, mock_search, mock_dl, run_dir):
    """Download writes to run_dir/data/raw/merra2/."""
    mock_login.return_value = MagicMock(authenticated=True)
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_download(run_dir)

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    fetch_merra2(run_dir=run_dir, period="2010/2010")

    mock_dl.assert_called_once()
    call_args = mock_dl.call_args
    local_path = call_args[1].get("local_path") or call_args[0][1]
    assert Path(local_path) == run_dir / "data" / "raw" / "merra2"


# ---- Provenance record -----------------------------------------------------


@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("earthaccess.login")
def test_provenance_record(mock_login, mock_search, mock_dl, run_dir):
    """Returned dict has all required provenance keys."""
    mock_login.return_value = MagicMock(authenticated=True)
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_download(run_dir)

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    result = fetch_merra2(run_dir=run_dir, period="2010/2010")

    assert result["source_key"] == "merra2"
    assert "access_url" in result
    assert result["variables"] == ["SFMC", "GWETROOT"]
    assert result["period"] == "2010/2010"
    assert "bbox" in result
    assert "download_timestamp" in result
    assert isinstance(result["files"], list)
    assert len(result["files"]) == 1
    assert "path" in result["files"][0]
    assert "size_bytes" in result["files"][0]
    # Path should be relative to run_dir
    assert not Path(result["files"][0]["path"]).is_absolute()


# ---- Superseded warning ----------------------------------------------------


@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("earthaccess.login")
@patch("nhf_spatial_targets.catalog.source")
def test_superseded_warning(mock_source, mock_login, mock_search, mock_dl, run_dir):
    """DeprecationWarning emitted when catalog status is superseded."""
    mock_source.return_value = {
        "status": "superseded",
        "access": {"url": "https://example.com", "short_name": "M2TMNXLND"},
        "variables": ["SFMC"],
    }
    mock_login.return_value = MagicMock(authenticated=True)
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_download(run_dir)

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fetch_merra2(run_dir=run_dir, period="2010/2010")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "superseded" in str(dep_warnings[0].message).lower()


# ---- Period validation -----------------------------------------------------


def test_period_missing_slash(run_dir):
    """ValueError raised for period without slash separator."""
    from nhf_spatial_targets.fetch.merra2 import _parse_period

    with pytest.raises(ValueError, match="YYYY/YYYY"):
        _parse_period("2010")


def test_period_non_numeric(run_dir):
    """ValueError raised for non-integer years."""
    from nhf_spatial_targets.fetch.merra2 import _parse_period

    with pytest.raises(ValueError, match="integers"):
        _parse_period("abc/def")


def test_period_reversed(run_dir):
    """ValueError raised when end year is before start year."""
    from nhf_spatial_targets.fetch.merra2 import _parse_period

    with pytest.raises(ValueError, match="before start year"):
        _parse_period("2015/2010")


# ---- Missing fabric.json --------------------------------------------------


@patch("earthaccess.login")
def test_missing_fabric_raises(mock_login, tmp_path):
    """FileNotFoundError raised when fabric.json is missing."""
    mock_login.return_value = MagicMock(authenticated=True)
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with pytest.raises(FileNotFoundError, match="fabric.json"):
        fetch_merra2(run_dir=run_dir, period="2010/2010")


# ---- Download failures -----------------------------------------------------


@patch("earthaccess.download", return_value=[])
@patch("earthaccess.search_data")
@patch("earthaccess.login")
def test_empty_download_raises(mock_login, mock_search, mock_dl, run_dir):
    """RuntimeError raised when download returns no files."""
    mock_login.return_value = MagicMock(authenticated=True)
    mock_search.return_value = [_mock_granule("g1")]

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with pytest.raises(RuntimeError, match="returned no files"):
        fetch_merra2(run_dir=run_dir, period="2010/2010")


@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("earthaccess.login")
def test_missing_downloaded_file_raises(mock_login, mock_search, mock_dl, run_dir):
    """RuntimeError raised when downloaded file does not exist on disk."""
    mock_login.return_value = MagicMock(authenticated=True)
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = [str(run_dir / "data" / "raw" / "merra2" / "ghost.nc4")]

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with pytest.raises(RuntimeError, match="do not exist on disk"):
        fetch_merra2(run_dir=run_dir, period="2010/2010")


# ---- Integration test (requires NASA Earthdata credentials) ----------------


@pytest.mark.integration
def test_fetch_merra2_real_download(tmp_path):
    """End-to-end download of one year of MERRA-2 data."""
    import xarray as xr

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fabric = {
        "bbox_buffered": {
            "minx": -105.0,
            "miny": 39.0,
            "maxx": -104.0,
            "maxy": 40.0,
        }
    }
    (run_dir / "fabric.json").write_text(json.dumps(fabric))

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    result = fetch_merra2(run_dir=run_dir, period="2010/2010")

    assert result["source_key"] == "merra2"
    assert len(result["files"]) > 0

    first_file = run_dir / result["files"][0]["path"]
    assert first_file.exists()
    ds = xr.open_dataset(first_file)
    assert "SFMC" in ds.data_vars or "GWETROOT" in ds.data_vars
    ds.close()
