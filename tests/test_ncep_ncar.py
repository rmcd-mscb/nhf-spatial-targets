"""Tests for NCEP/NCAR Reanalysis fetch module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

_MOCK_CONSOLIDATION = {
    "consolidated_nc": "data/raw/ncep_ncar/ncep_ncar_consolidated.nc",
    "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
    "n_files": 1,
    "variables": ["soilw_0_10cm", "soilw_10_200cm"],
}


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "ncep_ncar").mkdir(parents=True)
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


def _make_daily_nc(path: Path, year: int, var_name: str = "soilw"):
    """Create a synthetic daily NetCDF file for one year."""
    import pandas as pd

    times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    lat = np.arange(-90, 91, 45.0)
    lon = np.arange(-180, 180, 60.0)
    ds = xr.Dataset(
        {
            var_name: (
                ["time", "lat", "lon"],
                np.random.rand(len(times), len(lat), len(lon)).astype(np.float32),
            )
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path, format="NETCDF3_CLASSIC")
    return ds


# ---- Filename parsing -------------------------------------------------------


def test_year_from_monthly_path():
    """Extract year from various valid monthly filenames."""
    from nhf_spatial_targets.fetch.ncep_ncar import _year_from_monthly_path

    assert _year_from_monthly_path(Path("soilw.0-10cm.gauss.2010.monthly.nc")) == "2010"
    assert (
        _year_from_monthly_path(Path("soilw.10-200cm.gauss.1995.monthly.nc")) == "1995"
    )


def test_year_from_monthly_path_invalid():
    """ValueError raised for filenames without 'monthly' marker."""
    from nhf_spatial_targets.fetch.ncep_ncar import _year_from_monthly_path

    with pytest.raises(ValueError, match="Cannot extract year"):
        _year_from_monthly_path(Path("not_a_valid_file.nc"))


# ---- Malformed fabric.json -------------------------------------------------


def test_malformed_fabric_raises(run_dir):
    """ValueError raised when fabric.json is malformed."""
    (run_dir / "fabric.json").write_text("{}")

    from nhf_spatial_targets.fetch.ncep_ncar import fetch_ncep_ncar

    with pytest.raises(ValueError, match="malformed"):
        fetch_ncep_ncar(run_dir=run_dir, period="2010/2010")


# ---- Network error handling ------------------------------------------------


def test_url_error_raises(run_dir):
    """RuntimeError raised when urlretrieve raises URLError."""
    import urllib.error

    with patch(
        "nhf_spatial_targets.fetch.ncep_ncar.urllib.request.urlretrieve",
        side_effect=urllib.error.URLError("Name or service not known"),
    ):
        from nhf_spatial_targets.fetch.ncep_ncar import fetch_ncep_ncar

        with pytest.raises(RuntimeError, match="Failed to connect"):
            fetch_ncep_ncar(run_dir=run_dir, period="2010/2010")


# ---- URL construction -------------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.ncep_ncar.consolidate_ncep_ncar",
    return_value=_MOCK_CONSOLIDATION,
)
def test_url_construction(mock_consolidate, run_dir):
    """Correct URLs built from catalog file_pattern + year."""
    year = 2010
    urlretrieve_calls = []

    def fake_urlretrieve(url, dest):
        urlretrieve_calls.append((url, dest))
        _make_daily_nc(Path(dest), year)

    with patch(
        "nhf_spatial_targets.fetch.ncep_ncar.urllib.request.urlretrieve",
        side_effect=fake_urlretrieve,
    ):
        from nhf_spatial_targets.fetch.ncep_ncar import fetch_ncep_ncar

        fetch_ncep_ncar(run_dir=run_dir, period="2010/2010")

    urls_called = [c[0] for c in urlretrieve_calls]
    assert len(urls_called) == 2
    assert any("soilw.0-10cm.gauss.2010.nc" in u for u in urls_called)
    assert any("soilw.10-200cm.gauss.2010.nc" in u for u in urls_called)
    for url in urls_called:
        assert url.startswith("https://downloads.psl.noaa.gov/")
        assert "2010" in url


# ---- Download failure -------------------------------------------------------


def test_download_failure_raises(run_dir):
    """RuntimeError raised when urlretrieve raises HTTPError."""
    import urllib.error

    with patch(
        "nhf_spatial_targets.fetch.ncep_ncar.urllib.request.urlretrieve",
        side_effect=urllib.error.HTTPError(
            url="https://example.com/soilw.0-10cm.gauss.2010.nc",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        ),
    ):
        from nhf_spatial_targets.fetch.ncep_ncar import fetch_ncep_ncar

        with pytest.raises(RuntimeError, match="404"):
            fetch_ncep_ncar(run_dir=run_dir, period="2010/2010")


# ---- Daily to monthly aggregation ------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.ncep_ncar.consolidate_ncep_ncar",
    return_value=_MOCK_CONSOLIDATION,
)
def test_daily_to_monthly(mock_consolidate, run_dir):
    """Daily files are aggregated to monthly and daily files are deleted."""
    year = 2010

    def fake_urlretrieve(url, dest):
        _make_daily_nc(Path(dest), year)

    with patch(
        "nhf_spatial_targets.fetch.ncep_ncar.urllib.request.urlretrieve",
        side_effect=fake_urlretrieve,
    ):
        from nhf_spatial_targets.fetch.ncep_ncar import fetch_ncep_ncar

        fetch_ncep_ncar(run_dir=run_dir, period="2010/2010")

    ncep_dir = run_dir / "data" / "raw" / "ncep_ncar"
    monthly_files = list(ncep_dir.glob("*.monthly.nc"))
    daily_files = [f for f in ncep_dir.glob("*.nc") if "monthly" not in f.name]

    assert len(monthly_files) == 2  # two variables
    assert len(daily_files) == 0  # daily files cleaned up


# ---- Output directory -------------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.ncep_ncar.consolidate_ncep_ncar",
    return_value=_MOCK_CONSOLIDATION,
)
def test_output_dir(mock_consolidate, run_dir):
    """Files are written to data/raw/ncep_ncar/."""
    year = 2010

    def fake_urlretrieve(url, dest):
        _make_daily_nc(Path(dest), year)

    with patch(
        "nhf_spatial_targets.fetch.ncep_ncar.urllib.request.urlretrieve",
        side_effect=fake_urlretrieve,
    ):
        from nhf_spatial_targets.fetch.ncep_ncar import fetch_ncep_ncar

        fetch_ncep_ncar(run_dir=run_dir, period="2010/2010")

    ncep_dir = run_dir / "data" / "raw" / "ncep_ncar"
    assert ncep_dir.is_dir()
    monthly_files = list(ncep_dir.glob("*.monthly.nc"))
    assert len(monthly_files) > 0


# ---- Provenance record ------------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.ncep_ncar.consolidate_ncep_ncar",
    return_value=_MOCK_CONSOLIDATION,
)
def test_provenance_record(mock_consolidate, run_dir):
    """All required keys present in returned provenance dict."""
    year = 2010

    def fake_urlretrieve(url, dest):
        _make_daily_nc(Path(dest), year)

    with patch(
        "nhf_spatial_targets.fetch.ncep_ncar.urllib.request.urlretrieve",
        side_effect=fake_urlretrieve,
    ):
        from nhf_spatial_targets.fetch.ncep_ncar import fetch_ncep_ncar

        result = fetch_ncep_ncar(run_dir=run_dir, period="2010/2010")

    assert result["source_key"] == "ncep_ncar"
    assert "access_url" in result
    assert "variables" in result
    assert "period" in result
    assert "bbox" in result
    assert "download_timestamp" in result
    assert isinstance(result["files"], list)
    assert len(result["files"]) > 0
    assert "path" in result["files"][0]
    assert "year" in result["files"][0]
    assert "size_bytes" in result["files"][0]
    # Path should be relative to run_dir
    assert not Path(result["files"][0]["path"]).is_absolute()


# ---- Missing fabric.json ---------------------------------------------------


def test_missing_fabric_raises(tmp_path):
    """FileNotFoundError raised when fabric.json is missing."""
    empty_run = tmp_path / "empty_run"
    empty_run.mkdir()

    from nhf_spatial_targets.fetch.ncep_ncar import fetch_ncep_ncar

    with pytest.raises(FileNotFoundError, match="fabric.json"):
        fetch_ncep_ncar(run_dir=empty_run, period="2010/2010")


# ---- Incremental download --------------------------------------------------


def test_incremental_skips_existing(run_dir):
    """Years already in manifest are not re-downloaded."""
    manifest = {
        "sources": {
            "ncep_ncar": {
                "period": "2010/2010",
                "files": [
                    {
                        "path": "data/raw/ncep_ncar/soilw.0-10cm.gauss.2010.monthly.nc",
                        "year": "2010",
                        "size_bytes": 100,
                        "downloaded_utc": "2026-01-01T00:00:00+00:00",
                    }
                ],
            }
        }
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    from nhf_spatial_targets.fetch._period import years_in_period
    from nhf_spatial_targets.fetch.ncep_ncar import _existing_years

    existing = _existing_years(run_dir)
    assert "2010" in existing

    requested = [str(y) for y in years_in_period("2009/2011")]
    missing = [y for y in requested if y not in existing]
    assert "2010" not in missing
    assert "2009" in missing
    assert "2011" in missing


# ---- Manifest updated -------------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.ncep_ncar.consolidate_ncep_ncar",
    return_value=_MOCK_CONSOLIDATION,
)
def test_manifest_updated(mock_consolidate, run_dir):
    """fetch_ncep_ncar writes provenance to manifest.json."""
    year = 2010

    def fake_urlretrieve(url, dest):
        _make_daily_nc(Path(dest), year)

    with patch(
        "nhf_spatial_targets.fetch.ncep_ncar.urllib.request.urlretrieve",
        side_effect=fake_urlretrieve,
    ):
        from nhf_spatial_targets.fetch.ncep_ncar import fetch_ncep_ncar

        fetch_ncep_ncar(run_dir=run_dir, period="2010/2010")

    updated = json.loads((run_dir / "manifest.json").read_text())
    entry = updated["sources"]["ncep_ncar"]
    assert entry["period"] == "2010/2010"
    assert len(entry["files"]) > 0
    assert "year" in entry["files"][0]
    assert "consolidated_nc" in entry
