"""Tests for NLDAS-2 fetch module (MOSAIC and NOAH)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_MOCK_CONSOLIDATION = {
    "consolidated_nc": "data/raw/nldas_mosaic/nldas_mosaic_consolidated.nc",
    "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
    "n_files": 1,
    "variables": ["SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_200cm"],
}

_MOCK_CONSOLIDATION_NOAH = {
    "consolidated_nc": "data/raw/nldas_noah/nldas_noah_consolidated.nc",
    "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
    "n_files": 1,
    "variables": ["SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_200cm"],
}

_CONSOLIDATE_TARGET = "nhf_spatial_targets.fetch.consolidate.consolidate_nldas"


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """Create a minimal run workspace with fabric.json."""
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "nldas_mosaic").mkdir(parents=True)
    (rd / "data" / "raw" / "nldas_noah").mkdir(parents=True)
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


def _mock_granule(name: str, year_month: str = "200101") -> MagicMock:
    """Create a mock granule with a data link containing an NLDAS-style filename."""
    g = MagicMock()
    g.__str__ = lambda self: name
    url = f"https://example.com/NLDAS_MOS0125_M.A{year_month}.002.grb.SUB.nc4"
    g.data_links.return_value = [url]
    return g


def _mock_noah_granule(name: str, year_month: str = "200101") -> MagicMock:
    """Create a mock NOAH granule with a data link."""
    g = MagicMock()
    g.__str__ = lambda self: name
    url = f"https://example.com/NLDAS_NOAH0125_M.A{year_month}.002.grb.SUB.nc4"
    g.data_links.return_value = [url]
    return g


def _fake_mosaic_download(run_dir: Path, n: int = 1) -> list[str]:
    """Create fake MOSAIC downloaded files and return their paths."""
    paths = []
    for i in range(n):
        f = (
            run_dir
            / "data"
            / "raw"
            / "nldas_mosaic"
            / f"NLDAS_MOS0125_M.A2001{i + 1:02d}.002.grb.SUB.nc4"
        )
        f.write_bytes(b"fake")
        paths.append(str(f))
    return paths


def _fake_noah_download(run_dir: Path, n: int = 1) -> list[str]:
    """Create fake NOAH downloaded files and return their paths."""
    paths = []
    for i in range(n):
        f = (
            run_dir
            / "data"
            / "raw"
            / "nldas_noah"
            / f"NLDAS_NOAH0125_M.A2001{i + 1:02d}.002.grb.SUB.nc4"
        )
        f.write_bytes(b"fake")
        paths.append(str(f))
    return paths


# ---- Filename parsing -------------------------------------------------------


def test_year_month_from_path():
    """Extract YYYY-MM from various valid NLDAS filenames."""
    from nhf_spatial_targets.fetch.nldas import _year_month_from_path

    assert (
        _year_month_from_path(Path("NLDAS_MOS0125_M.A200101.002.grb.SUB.nc4"))
        == "2001-01"
    )
    assert (
        _year_month_from_path(Path("NLDAS_NOAH0125_M.A201012.002.grb.SUB.nc4"))
        == "2010-12"
    )


def test_year_month_from_invalid_path():
    """ValueError raised for filenames without NLDAS date pattern."""
    from nhf_spatial_targets.fetch.nldas import _year_month_from_path

    with pytest.raises(ValueError, match="Cannot extract date"):
        _year_month_from_path(Path("not_an_nldas_file.nc"))


# ---- Malformed fabric.json -------------------------------------------------


@patch("nhf_spatial_targets.fetch.nldas.earthdata_login")
def test_malformed_fabric_raises(mock_login, tmp_path):
    """ValueError raised when fabric.json is malformed."""
    run_dir = tmp_path / "bad_fabric_run"
    run_dir.mkdir()
    (run_dir / "fabric.json").write_text("{}")

    with patch(_CONSOLIDATE_TARGET, return_value=_MOCK_CONSOLIDATION):
        from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic

        with pytest.raises(ValueError, match="malformed"):
            fetch_nldas_mosaic(run_dir=run_dir, period="2001/2001")


# ---- Authentication --------------------------------------------------------


@patch("nhf_spatial_targets.fetch.nldas.earthdata_login")
def test_login_called(mock_login, run_dir):
    """earthdata_login() is called before searching."""
    with patch(_CONSOLIDATE_TARGET, return_value=_MOCK_CONSOLIDATION):
        with patch("earthaccess.search_data", return_value=[]):
            with pytest.raises(ValueError, match="No granules found"):
                from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic

                fetch_nldas_mosaic(run_dir=run_dir, period="2001/2001")
    mock_login.assert_called_once_with(run_dir)


# ---- Search parameters -----------------------------------------------------


@patch(_CONSOLIDATE_TARGET, return_value=_MOCK_CONSOLIDATION)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.nldas.earthdata_login")
def test_search_params_mosaic(
    mock_login, mock_search, mock_dl, mock_consolidate, run_dir
):
    """search_data called with correct short_name for MOSAIC."""
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_mosaic_download(run_dir)

    from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic

    fetch_nldas_mosaic(run_dir=run_dir, period="2001/2002")

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["short_name"] == "NLDAS_MOS0125_M"
    assert call_kwargs["bounding_box"] == (-125.1, 23.9, -65.9, 50.1)
    assert call_kwargs["temporal"] == ("2001-01-01", "2002-12-31")


@patch(_CONSOLIDATE_TARGET, return_value=_MOCK_CONSOLIDATION_NOAH)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.nldas.earthdata_login")
def test_search_params_noah(
    mock_login, mock_search, mock_dl, mock_consolidate, run_dir
):
    """search_data called with correct short_name for NOAH."""
    mock_search.return_value = [_mock_noah_granule("g1")]
    mock_dl.return_value = _fake_noah_download(run_dir)

    from nhf_spatial_targets.fetch.nldas import fetch_nldas_noah

    fetch_nldas_noah(run_dir=run_dir, period="2001/2002")

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["short_name"] == "NLDAS_NOAH0125_M"
    assert call_kwargs["bounding_box"] == (-125.1, 23.9, -65.9, 50.1)
    assert call_kwargs["temporal"] == ("2001-01-01", "2002-12-31")


# ---- No results ------------------------------------------------------------


@patch("nhf_spatial_targets.fetch.nldas.earthdata_login")
@patch("earthaccess.search_data", return_value=[])
def test_no_granules_raises(mock_search, mock_login, run_dir):
    """ValueError raised when search returns zero granules."""
    with patch(_CONSOLIDATE_TARGET, return_value=_MOCK_CONSOLIDATION):
        from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic

        with pytest.raises(ValueError, match="No granules found"):
            fetch_nldas_mosaic(run_dir=run_dir, period="2001/2001")


# ---- Output directory ------------------------------------------------------


@patch(_CONSOLIDATE_TARGET, return_value=_MOCK_CONSOLIDATION)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.nldas.earthdata_login")
def test_output_dir(mock_login, mock_search, mock_dl, mock_consolidate, run_dir):
    """Download writes to run_dir/data/raw/nldas_mosaic/."""
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_mosaic_download(run_dir)

    from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic

    fetch_nldas_mosaic(run_dir=run_dir, period="2001/2001")

    mock_dl.assert_called_once()
    call_args = mock_dl.call_args
    local_path = call_args[1].get("local_path") or call_args[0][1]
    assert Path(local_path) == run_dir / "data" / "raw" / "nldas_mosaic"


# ---- Provenance record -----------------------------------------------------


@patch(_CONSOLIDATE_TARGET, return_value=_MOCK_CONSOLIDATION)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.nldas.earthdata_login")
def test_provenance_record(mock_login, mock_search, mock_dl, mock_consolidate, run_dir):
    """Returned dict has all required provenance keys."""
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_mosaic_download(run_dir)

    from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic

    result = fetch_nldas_mosaic(run_dir=run_dir, period="2001/2001")

    assert result["source_key"] == "nldas_mosaic"
    assert "access_url" in result
    var_names = [v["name"] for v in result["variables"]]
    assert var_names == ["SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_200cm"]
    assert result["period"] == "2001/2001"
    assert "bbox" in result
    assert "download_timestamp" in result
    assert isinstance(result["files"], list)
    assert len(result["files"]) == 1
    assert "path" in result["files"][0]
    assert "size_bytes" in result["files"][0]
    # Path should be relative to run_dir
    assert not Path(result["files"][0]["path"]).is_absolute()


# ---- Superseded warning ----------------------------------------------------


@patch(_CONSOLIDATE_TARGET, return_value=_MOCK_CONSOLIDATION)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.nldas.earthdata_login")
@patch("nhf_spatial_targets.catalog.source")
def test_superseded_warning(
    mock_source, mock_login, mock_search, mock_dl, mock_consolidate, run_dir
):
    """DeprecationWarning emitted when catalog status is superseded."""
    mock_source.return_value = {
        "status": "superseded",
        "superseded_by": "nldas_mosaic_v3",
        "access": {
            "url": "https://example.com",
            "short_name": "NLDAS_MOS0125_M",
        },
        "variables": [{"name": "SoilM_0_10cm"}],
    }
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_mosaic_download(run_dir)

    from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fetch_nldas_mosaic(run_dir=run_dir, period="2001/2001")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "superseded" in str(dep_warnings[0].message).lower()


# ---- Missing fabric.json --------------------------------------------------


@patch("nhf_spatial_targets.fetch.nldas.earthdata_login")
def test_missing_fabric_raises(mock_login, tmp_path):
    """FileNotFoundError raised when fabric.json is missing."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    with patch(_CONSOLIDATE_TARGET, return_value=_MOCK_CONSOLIDATION):
        from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic

        with pytest.raises(FileNotFoundError, match="fabric.json"):
            fetch_nldas_mosaic(run_dir=run_dir, period="2001/2001")


# ---- Download failures -----------------------------------------------------


@patch("earthaccess.download", return_value=[])
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.nldas.earthdata_login")
def test_empty_download_raises(mock_login, mock_search, mock_dl, run_dir):
    """RuntimeError raised when download returns no files."""
    mock_search.return_value = [_mock_granule("g1")]

    with patch(_CONSOLIDATE_TARGET, return_value=_MOCK_CONSOLIDATION):
        from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic

        with pytest.raises(RuntimeError, match="returned no files"):
            fetch_nldas_mosaic(run_dir=run_dir, period="2001/2001")


# ---- Incremental fetch helpers ---------------------------------------------


@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.nldas.earthdata_login")
def test_incremental_skips_existing(mock_login, mock_search, mock_dl, run_dir):
    """Months already in manifest are not re-downloaded."""
    # Pre-populate manifest with Jan 2001 already downloaded
    manifest = {
        "sources": {
            "nldas_mosaic": {
                "period": "2001/2001",
                "files": [
                    {
                        "path": "data/raw/nldas_mosaic/NLDAS_MOS0125_M.A200101.002.grb.SUB.nc4",
                        "year_month": "2001-01",
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
        run_dir
        / "data"
        / "raw"
        / "nldas_mosaic"
        / "NLDAS_MOS0125_M.A200101.002.grb.SUB.nc4"
    )
    existing.write_bytes(b"fake")

    from nhf_spatial_targets.fetch._period import months_in_period
    from nhf_spatial_targets.fetch.nldas import _existing_months

    existing_months = _existing_months(run_dir, "nldas_mosaic")
    assert "2001-01" in existing_months

    requested = months_in_period("2001/2001")
    assert len(requested) == 12

    missing = [m for m in requested if m not in existing_months]
    assert "2001-01" not in missing
    assert len(missing) == 11


# ---- Manifest update tests -------------------------------------------------


@patch(_CONSOLIDATE_TARGET, return_value=_MOCK_CONSOLIDATION)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.nldas.earthdata_login")
def test_manifest_updated(mock_login, mock_search, mock_dl, mock_consolidate, run_dir):
    """fetch_nldas_mosaic writes provenance to manifest.json."""
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_mosaic_download(run_dir)

    from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic

    fetch_nldas_mosaic(run_dir=run_dir, period="2001/2001")

    updated_manifest = json.loads((run_dir / "manifest.json").read_text())
    entry = updated_manifest["sources"]["nldas_mosaic"]
    assert entry["period"] == "2001/2001"
    assert len(entry["files"]) > 0
    assert "year_month" in entry["files"][0]
    assert "consolidated_nc" in entry


@patch(_CONSOLIDATE_TARGET, return_value=_MOCK_CONSOLIDATION)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.nldas.earthdata_login")
def test_manifest_preserves_download_timestamp(
    mock_login, mock_search, mock_dl, mock_consolidate, run_dir
):
    """Re-running fetch preserves original downloaded_utc for existing files."""
    mock_search.return_value = [_mock_granule("g1")]

    # Pre-populate manifest with all 12 months already recorded
    original_ts = "2026-01-01T00:00:00+00:00"
    other_ts = "2026-01-02T00:00:00+00:00"
    files_in_manifest = [
        {
            "path": f"data/raw/nldas_mosaic/NLDAS_MOS0125_M.A2001{m:02d}.002.grb.SUB.nc4",
            "year_month": f"2001-{m:02d}",
            "size_bytes": 100,
            "downloaded_utc": original_ts if m == 1 else other_ts,
        }
        for m in range(1, 13)
    ]
    manifest = {"sources": {"nldas_mosaic": {"files": files_in_manifest}}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    # Create all 12 fake files on disk
    for m in range(1, 13):
        f = (
            run_dir
            / "data"
            / "raw"
            / "nldas_mosaic"
            / f"NLDAS_MOS0125_M.A2001{m:02d}.002.grb.SUB.nc4"
        )
        f.write_bytes(b"fake")

    from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic

    fetch_nldas_mosaic(run_dir=run_dir, period="2001/2001")

    updated = json.loads((run_dir / "manifest.json").read_text())
    jan_file = next(
        f
        for f in updated["sources"]["nldas_mosaic"]["files"]
        if f["year_month"] == "2001-01"
    )
    assert jan_file["downloaded_utc"] == original_ts


# ---- Integration test (requires NASA Earthdata credentials) ----------------


@pytest.mark.integration
def test_fetch_nldas_mosaic_real_download(tmp_path):
    """End-to-end download of one year of NLDAS-2 MOSAIC data."""
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

    from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic

    result = fetch_nldas_mosaic(run_dir=run_dir, period="2001/2001")

    assert result["source_key"] == "nldas_mosaic"
    assert len(result["files"]) == 12
    assert "consolidated_nc" in result

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "nldas_mosaic" in manifest["sources"]
