"""Tests for MERRA-2 fetch module."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_MOCK_CONSOLIDATION = {
    "consolidated_nc": "data/raw/merra2/merra2_consolidated.nc",
    "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
    "n_files": 1,
    "variables": ["GWETTOP", "GWETROOT", "GWETPROF"],
}


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


def _mock_granule(name: str, year_month: str = "201001") -> MagicMock:
    """Create a mock granule object with data_links containing a MERRA-2 URL."""
    g = MagicMock()
    g.__str__ = lambda self: name
    url = f"https://example.com/MERRA2_300.tavgM_2d_lnd_Nx.{year_month}.nc4"
    g.data_links.return_value = [url]
    return g


def _fake_download(run_dir: Path, n: int = 1) -> list[str]:
    """Create fake downloaded files and return their paths."""
    paths = []
    for i in range(n):
        f = (
            run_dir
            / "data"
            / "raw"
            / "merra2"
            / f"MERRA2_300.tavgM_2d_lnd_Nx.2010{i + 1:02d}.nc4"
        )
        f.write_bytes(b"fake")
        paths.append(str(f))
    return paths


# ---- Authentication --------------------------------------------------------


@patch("nhf_spatial_targets.fetch.merra2.earthdata_login")
def test_login_called(mock_login, run_dir):
    """earthdata_login() is called before searching."""
    with patch("earthaccess.search_data", return_value=[]):
        with pytest.raises(ValueError, match="No granules found"):
            from nhf_spatial_targets.fetch.merra2 import fetch_merra2

            fetch_merra2(run_dir=run_dir, period="2010/2010")
    mock_login.assert_called_once_with(run_dir)


# ---- Search parameters -----------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.merra2.consolidate_merra2",
    return_value=_MOCK_CONSOLIDATION,
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.merra2.earthdata_login")
def test_search_params(mock_login, mock_search, mock_dl, mock_consolidate, run_dir):
    """search_data called with correct short_name, bbox tuple, and temporal."""
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


@patch("nhf_spatial_targets.fetch.merra2.earthdata_login")
@patch("earthaccess.search_data", return_value=[])
def test_no_granules_raises(mock_search, mock_login, run_dir):
    """ValueError raised when search returns zero granules."""
    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with pytest.raises(ValueError, match="No granules found"):
        fetch_merra2(run_dir=run_dir, period="2010/2010")


# ---- Output directory ------------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.merra2.consolidate_merra2",
    return_value=_MOCK_CONSOLIDATION,
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.merra2.earthdata_login")
def test_output_dir(mock_login, mock_search, mock_dl, mock_consolidate, run_dir):
    """Download writes to run_dir/data/raw/merra2/."""
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_download(run_dir)

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    fetch_merra2(run_dir=run_dir, period="2010/2010")

    mock_dl.assert_called_once()
    call_args = mock_dl.call_args
    local_path = call_args[1].get("local_path") or call_args[0][1]
    assert Path(local_path) == run_dir / "data" / "raw" / "merra2"


# ---- Provenance record -----------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.merra2.consolidate_merra2",
    return_value=_MOCK_CONSOLIDATION,
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.merra2.earthdata_login")
def test_provenance_record(mock_login, mock_search, mock_dl, mock_consolidate, run_dir):
    """Returned dict has all required provenance keys."""
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_download(run_dir)

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    result = fetch_merra2(run_dir=run_dir, period="2010/2010")

    assert result["source_key"] == "merra2"
    assert "access_url" in result
    var_names = [v["name"] for v in result["variables"]]
    assert var_names == ["GWETTOP", "GWETROOT", "GWETPROF"]
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


@patch(
    "nhf_spatial_targets.fetch.merra2.consolidate_merra2",
    return_value=_MOCK_CONSOLIDATION,
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.merra2.earthdata_login")
@patch("nhf_spatial_targets.catalog.source")
def test_superseded_warning(
    mock_source, mock_login, mock_search, mock_dl, mock_consolidate, run_dir
):
    """DeprecationWarning emitted when catalog status is superseded."""
    mock_source.return_value = {
        "status": "superseded",
        "access": {"url": "https://example.com", "short_name": "M2TMNXLND"},
        "variables": [{"name": "SFMC"}],
    }
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_download(run_dir)

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fetch_merra2(run_dir=run_dir, period="2010/2010")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "superseded" in str(dep_warnings[0].message).lower()


# ---- Missing / malformed fabric.json ---------------------------------------


@patch("nhf_spatial_targets.fetch.merra2.earthdata_login")
def test_missing_fabric_raises(mock_login, tmp_path):
    """FileNotFoundError raised when fabric.json is missing."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with pytest.raises(FileNotFoundError, match="fabric.json"):
        fetch_merra2(run_dir=run_dir, period="2010/2010")


@patch("nhf_spatial_targets.fetch.merra2.earthdata_login")
def test_malformed_fabric_raises(mock_login, tmp_path):
    """ValueError raised when fabric.json is malformed."""
    run_dir = tmp_path / "bad_fabric_run"
    run_dir.mkdir()
    (run_dir / "fabric.json").write_text("{}")

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with pytest.raises(ValueError, match="malformed"):
        fetch_merra2(run_dir=run_dir, period="2010/2010")


# ---- Download failures -----------------------------------------------------


@patch("earthaccess.download", return_value=[])
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.merra2.earthdata_login")
def test_empty_download_raises(mock_login, mock_search, mock_dl, run_dir):
    """RuntimeError raised when download returns no files."""
    mock_search.return_value = [_mock_granule("g1")]

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with pytest.raises(RuntimeError, match="returned no files"):
        fetch_merra2(run_dir=run_dir, period="2010/2010")


# ---- Incremental fetch helpers ---------------------------------------------


@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.merra2.earthdata_login")
def test_incremental_skips_existing(mock_login, mock_search, mock_dl, run_dir):
    """Months already in manifest are not re-downloaded."""
    # Pre-populate manifest with Jan 2010 already downloaded
    manifest = {
        "sources": {
            "merra2": {
                "period": "2010/2010",
                "files": [
                    {
                        "path": "data/raw/merra2/MERRA2_300.tavgM_2d_lnd_Nx.201001.nc4",
                        "year_month": "2010-01",
                        "size_bytes": 100,
                        "downloaded_utc": "2026-01-01T00:00:00+00:00",
                    }
                ],
            }
        }
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    # Create the existing file on disk
    existing = (
        run_dir / "data" / "raw" / "merra2" / "MERRA2_300.tavgM_2d_lnd_Nx.201001.nc4"
    )
    existing.write_bytes(b"fake")

    from nhf_spatial_targets.fetch._period import months_in_period
    from nhf_spatial_targets.fetch.merra2 import _existing_months

    existing_months = _existing_months(run_dir)
    assert "2010-01" in existing_months

    requested = months_in_period("2010/2010")
    assert len(requested) == 12

    missing = [m for m in requested if m not in existing_months]
    assert "2010-01" not in missing
    assert len(missing) == 11


def test_year_month_from_filename():
    """Extract YYYY-MM from MERRA-2 filename for all collection numbers."""
    from nhf_spatial_targets.fetch.merra2 import _year_month_from_path

    assert (
        _year_month_from_path(Path("MERRA2_300.tavgM_2d_lnd_Nx.201007.nc4"))
        == "2010-07"
    )
    assert (
        _year_month_from_path(Path("MERRA2_100.tavgM_2d_lnd_Nx.198001.nc4"))
        == "1980-01"
    )
    assert (
        _year_month_from_path(Path("MERRA2_200.tavgM_2d_lnd_Nx.199506.nc4"))
        == "1995-06"
    )
    assert (
        _year_month_from_path(Path("MERRA2_400.tavgM_2d_lnd_Nx.202312.nc4"))
        == "2023-12"
    )


def test_year_month_from_invalid_filename():
    """ValueError raised for filenames without a date pattern."""
    from nhf_spatial_targets.fetch.merra2 import _year_month_from_path

    with pytest.raises(ValueError, match="Cannot extract date"):
        _year_month_from_path(Path("not_a_merra2_file.txt"))


# ---- Manifest update tests -------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.merra2.consolidate_merra2",
    return_value=_MOCK_CONSOLIDATION,
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.merra2.earthdata_login")
def test_manifest_updated(mock_login, mock_search, mock_dl, mock_consolidate, run_dir):
    """fetch_merra2 writes provenance to manifest.json."""
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_download(run_dir)

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    fetch_merra2(run_dir=run_dir, period="2010/2010")

    updated_manifest = json.loads((run_dir / "manifest.json").read_text())
    merra2_entry = updated_manifest["sources"]["merra2"]
    assert merra2_entry["period"] == "2010/2010"
    assert len(merra2_entry["files"]) > 0
    assert "year_month" in merra2_entry["files"][0]
    assert "consolidated_nc" in merra2_entry


@patch(
    "nhf_spatial_targets.fetch.merra2.consolidate_merra2",
    return_value=_MOCK_CONSOLIDATION,
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.merra2.earthdata_login")
def test_manifest_preserves_download_timestamp(
    mock_login, mock_search, mock_dl, mock_consolidate, run_dir
):
    """Re-running fetch preserves original downloaded_utc for existing files."""
    mock_search.return_value = [_mock_granule("g1")]

    # Pre-populate manifest with all 12 months already recorded so
    # no download is needed; Jan has an older timestamp to verify preservation.
    original_ts = "2026-01-01T00:00:00+00:00"
    other_ts = "2026-01-02T00:00:00+00:00"
    files_in_manifest = [
        {
            "path": f"data/raw/merra2/MERRA2_300.tavgM_2d_lnd_Nx.2010{m:02d}.nc4",
            "year_month": f"2010-{m:02d}",
            "size_bytes": 100,
            "downloaded_utc": original_ts if m == 1 else other_ts,
        }
        for m in range(1, 13)
    ]
    manifest = {"sources": {"merra2": {"files": files_in_manifest}}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    # Create all 12 fake files on disk
    for m in range(1, 13):
        f = (
            run_dir
            / "data"
            / "raw"
            / "merra2"
            / f"MERRA2_300.tavgM_2d_lnd_Nx.2010{m:02d}.nc4"
        )
        f.write_bytes(b"fake")

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    fetch_merra2(run_dir=run_dir, period="2010/2010")

    updated = json.loads((run_dir / "manifest.json").read_text())
    jan_file = next(
        f for f in updated["sources"]["merra2"]["files"] if f["year_month"] == "2010-01"
    )
    assert jan_file["downloaded_utc"] == original_ts


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
    assert len(result["files"]) == 12
    assert "consolidated_nc" in result

    # Verify consolidated file works
    nc_path = run_dir / result["consolidated_nc"]
    assert nc_path.exists()

    ds = xr.open_dataset(nc_path)
    assert "GWETTOP" in ds.data_vars
    assert "GWETROOT" in ds.data_vars
    assert "GWETPROF" in ds.data_vars
    assert "SFMC" not in ds.data_vars
    assert len(ds.time) == 12
    ds.close()

    # Verify manifest was written
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "merra2" in manifest["sources"]
