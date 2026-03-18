"""Tests for MODIS fetch module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

_SOURCE_KEY_10 = "mod10c1_v061"

_MOCK_CONSOLIDATION_10 = {
    "consolidated_nc": "data/raw/mod10c1_v061/mod10c1_v061_2005.nc",
    "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
    "n_files": 1,
    "variables": ["Day_CMG_Snow_Cover", "Snow_Spatial_QA"],
}

_MOCK_CONSOLIDATION = {
    "consolidated_nc": "data/raw/mod16a2_v061/mod16a2_v061_2010.nc",
    "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
    "n_files": 1,
    "variables": ["ET_500m", "ET_QC_500m"],
}

_MOCK_CONSOLIDATION_FINALIZE = {
    "consolidated_nc": "data/raw/mod16a2_v061/mod16a2_v061_2010_consolidated.nc",
    "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
    "n_files": 1,
    "variables": ["ET_500m", "ET_QC_500m"],
}


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    """Create a minimal run workspace with fabric.json."""
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "mod16a2_v061").mkdir(parents=True)
    (rd / "data" / "raw" / "mod10c1_v061").mkdir(parents=True)
    fabric = {
        "bbox_buffered": {
            "minx": -125.1,
            "miny": 23.9,
            "maxx": -65.9,
            "maxy": 50.1,
        }
    }
    (rd / "fabric.json").write_text(json.dumps(fabric))
    config = {
        "fabric": {"path": "", "id_col": "nhm_id"},
        "datastore": str(rd / "data" / "raw"),
        "dir_mode": "2775",
    }
    (rd / "config.yml").write_text(yaml.dump(config))
    (rd / "manifest.json").write_text('{"sources": {}, "steps": []}')
    return rd


def _mock_granule(name: str) -> MagicMock:
    """Create a mock granule object."""
    g = MagicMock()
    g.__str__ = lambda self: name
    g.data_links.return_value = [f"https://example.com/{name}"]
    return g


def _fake_download(workdir: Path, year: int = 2010, n: int = 1) -> list[str]:
    """Create fake downloaded HDF files and return their paths."""
    paths = []
    for i in range(n):
        doy = (i + 1) * 8
        f = (
            workdir
            / "data"
            / "raw"
            / "mod16a2_v061"
            / f"MOD16A2GF.A{year}{doy:03d}.h08v04.061.2020256154955.hdf"
        )
        f.write_bytes(b"fake")
        paths.append(str(f))
    return paths


# ---- Year extraction -------------------------------------------------------


def test_year_from_mod16a2_filename():
    """Extract year from various MOD16A2 filenames."""
    from nhf_spatial_targets.fetch.modis import _year_from_path

    assert (
        _year_from_path(Path("MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf")) == 2010
    )
    assert (
        _year_from_path(Path("MOD16A2GF.A2005049.h09v05.061.2020256154955.hdf")) == 2005
    )
    assert (
        _year_from_path(Path("MOD16A2GF.A2000361.h12v04.061.2020256154955.hdf")) == 2000
    )


def test_year_from_mod10c1_filename():
    """Extract year from various MOD10C1 filenames."""
    from nhf_spatial_targets.fetch.modis import _year_from_path

    assert _year_from_path(Path("MOD10C1.A2010032.061.2020345123456.hdf")) == 2010
    assert _year_from_path(Path("MOD10C1.A2014365.061.2020345123456.hdf")) == 2014
    assert _year_from_path(Path("MOD10C1.A2001001.061.2020345123456.hdf")) == 2001


def test_year_from_conus_subset_filename():
    """Extract year from .conus.nc suffix filenames."""
    from nhf_spatial_targets.fetch.modis import _year_from_path

    assert _year_from_path(Path("MOD16A2GF.A2010001.h08v04.061.conus.nc")) == 2010
    assert _year_from_path(Path("MOD10C1.A2005180.061.conus.nc")) == 2005


def test_year_from_invalid_filename():
    """ValueError raised for filenames without AYYYYDDD pattern."""
    from nhf_spatial_targets.fetch.modis import _year_from_path

    with pytest.raises(ValueError, match="Cannot extract year"):
        _year_from_path(Path("not_a_modis_file.txt"))

    with pytest.raises(ValueError, match="Cannot extract year"):
        _year_from_path(Path("MOD16A2GF_no_dot_A2010001.hdf"))


# ---- Authentication --------------------------------------------------------


@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_login_called(mock_login, workdir):
    """earthdata_login() is called before searching."""
    with patch("earthaccess.search_data", return_value=[]):
        with pytest.raises(ValueError, match="No granules found"):
            from nhf_spatial_targets.fetch.modis import fetch_mod16a2

            fetch_mod16a2(workdir=workdir, period="2010/2010")
    mock_login.assert_called_once_with(workdir)


# ---- Search parameters -----------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2",
    return_value=_MOCK_CONSOLIDATION,
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_finalize",
    return_value=_MOCK_CONSOLIDATION_FINALIZE,
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_timestep",
    return_value=Path("/tmp/fake_tmp.nc"),
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_search_params(
    mock_login, mock_search, mock_dl, mock_ts, mock_fin, mock_consolidate, workdir
):
    """search_data called with correct short_name, bbox tuple, and temporal."""
    mock_search.return_value = [
        _mock_granule("MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf")
    ]
    mock_dl.return_value = _fake_download(workdir)

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    fetch_mod16a2(workdir=workdir, period="2010/2010")

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["short_name"] == "MOD16A2GF"
    assert call_kwargs["bounding_box"] == (-125.1, 23.9, -65.9, 50.1)
    assert call_kwargs["temporal"] == ("2010-01-01", "2010-12-31")


# ---- No results ------------------------------------------------------------


@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
@patch("earthaccess.search_data", return_value=[])
def test_mod16a2_no_granules_raises(mock_search, mock_login, workdir):
    """ValueError raised when search returns zero granules."""
    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    with pytest.raises(ValueError, match="No granules found"):
        fetch_mod16a2(workdir=workdir, period="2010/2010")


# ---- Download failures -----------------------------------------------------


@patch("earthaccess.download", return_value=[])
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_empty_download_raises(mock_login, mock_search, mock_dl, workdir):
    """RuntimeError raised when download returns no files."""
    mock_search.return_value = [
        _mock_granule("MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf")
    ]

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    with pytest.raises(RuntimeError, match="returned no files"):
        fetch_mod16a2(workdir=workdir, period="2010/2010")


# ---- Missing / malformed fabric.json ---------------------------------------


@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_missing_fabric_raises(mock_login, tmp_path):
    """FileNotFoundError raised when fabric.json is missing."""
    workdir = tmp_path / "empty_run"
    workdir.mkdir()

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    with pytest.raises(FileNotFoundError, match="config.yml"):
        fetch_mod16a2(workdir=workdir, period="2010/2010")


@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_malformed_fabric_raises(mock_login, tmp_path):
    """ValueError raised when fabric.json is malformed."""
    workdir = tmp_path / "bad_fabric_run"
    workdir.mkdir()
    (workdir / "fabric.json").write_text("{}")
    import yaml as _yaml

    _cfg = {
        "fabric": {"path": "", "id_col": "nhm_id"},
        "datastore": str(workdir),
        "dir_mode": "2775",
    }
    (workdir / "config.yml").write_text(_yaml.dump(_cfg))
    (workdir / "manifest.json").write_text('{"sources": {}, "steps": []}')

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    with pytest.raises(KeyError):
        fetch_mod16a2(workdir=workdir, period="2010/2010")


# ---- Provenance record -----------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2",
    return_value=_MOCK_CONSOLIDATION,
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_finalize",
    return_value=_MOCK_CONSOLIDATION_FINALIZE,
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_timestep",
    return_value=Path("/tmp/fake_tmp.nc"),
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_provenance_record(
    mock_login, mock_search, mock_dl, mock_ts, mock_fin, mock_consolidate, workdir
):
    """Returned dict has all required provenance keys."""
    mock_search.return_value = [
        _mock_granule("MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf")
    ]
    mock_dl.return_value = _fake_download(workdir)

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    result = fetch_mod16a2(workdir=workdir, period="2010/2010")

    assert result["source_key"] == "mod16a2_v061"
    assert "access_url" in result
    assert result["variables"] == ["ET_500m", "ET_QC_500m"]
    assert result["period"] == "2010/2010"
    assert "bbox" in result
    assert "download_timestamp" in result
    assert isinstance(result["files"], list)
    assert len(result["files"]) == 1
    assert "path" in result["files"][0]
    assert "year" in result["files"][0]
    assert "size_bytes" in result["files"][0]
    assert isinstance(result["consolidated_ncs"], dict)
    assert Path(result["files"][0]["path"]).is_absolute()


# ---- Manifest update -------------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2",
    return_value=_MOCK_CONSOLIDATION,
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_finalize",
    return_value=_MOCK_CONSOLIDATION_FINALIZE,
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_timestep",
    return_value=Path("/tmp/fake_tmp.nc"),
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_manifest_updated(
    mock_login, mock_search, mock_dl, mock_ts, mock_fin, mock_consolidate, workdir
):
    """fetch_mod16a2 writes provenance to manifest.json."""
    mock_search.return_value = [
        _mock_granule("MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf")
    ]
    mock_dl.return_value = _fake_download(workdir)

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    fetch_mod16a2(workdir=workdir, period="2010/2010")

    updated_manifest = json.loads((workdir / "manifest.json").read_text())
    entry = updated_manifest["sources"]["mod16a2_v061"]
    assert entry["period"] == "2010/2010"
    assert len(entry["files"]) > 0
    assert "year" in entry["files"][0]
    assert "consolidated_ncs" in entry
    assert "variables" in entry
    var_names = [v["name"] if isinstance(v, dict) else v for v in entry["variables"]]
    assert var_names == ["ET_500m", "ET_QC_500m"]


# ---- Incremental fetch -----------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2",
    return_value=_MOCK_CONSOLIDATION,
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_finalize",
    return_value={
        "consolidated_nc": "data/raw/mod16a2_v061/mod16a2_v061_2011_consolidated.nc",
        "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
        "n_files": 1,
        "variables": ["ET_500m", "ET_QC_500m"],
    },
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_timestep",
    return_value=Path("/tmp/fake_tmp.nc"),
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_incremental_skips_year(
    mock_login, mock_search, mock_dl, mock_ts, mock_fin, mock_consolidate, workdir
):
    """Years already in manifest are not re-downloaded."""
    manifest = {
        "sources": {
            "mod16a2_v061": {
                "period": "2010/2010",
                "files": [
                    {
                        "path": "data/raw/mod16a2_v061/MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf",
                        "year": 2010,
                        "size_bytes": 100,
                        "downloaded_utc": "2026-01-01T00:00:00+00:00",
                    }
                ],
            }
        }
    }
    (workdir / "manifest.json").write_text(json.dumps(manifest))

    existing = (
        workdir
        / "data"
        / "raw"
        / "mod16a2_v061"
        / "MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf"
    )
    existing.write_bytes(b"fake")

    mock_search.return_value = [
        _mock_granule("MOD16A2GF.A2011001.h08v04.061.2020256154955.hdf")
    ]
    mock_dl.return_value = _fake_download(workdir, year=2011)

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    fetch_mod16a2(workdir=workdir, period="2010/2011")

    assert mock_search.call_count == 1
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["temporal"] == ("2011-01-01", "2011-12-31")


# ---------------------------------------------------------------------------
# MOD10C1 helpers
# ---------------------------------------------------------------------------


def _fake_download_10c1(workdir: Path, year: int = 2005, n: int = 1) -> list[str]:
    """Create fake downloaded HDF files for MOD10C1 and return their paths."""
    paths = []
    for i in range(n):
        doy = i + 1
        f = (
            workdir
            / "data"
            / "raw"
            / _SOURCE_KEY_10
            / f"MOD10C1.A{year}{doy:03d}.061.2020345123456.hdf"
        )
        f.write_bytes(b"fake")
        paths.append(str(f))
    return paths


def _mock_subset_side_effect(hdf_path: Path, bbox=None) -> Path:
    """Side effect for _subset_to_conus: create .conus.nc, remove .hdf."""
    out_path = hdf_path.with_suffix("").with_suffix(".conus.nc")
    out_path.write_bytes(b"fake-nc")
    if hdf_path.exists():
        hdf_path.unlink()
    return out_path


# ---- MOD10C1 Authentication -----------------------------------------------


@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod10c1_login_called(mock_login, workdir):
    """earthdata_login() is called before searching."""
    with patch("earthaccess.search_data", return_value=[]):
        with pytest.raises(ValueError, match="No granules found"):
            from nhf_spatial_targets.fetch.modis import fetch_mod10c1

            fetch_mod10c1(workdir=workdir, period="2005/2005")
    mock_login.assert_called_once_with(workdir)


# ---- MOD10C1 Search parameters --------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod10c1",
    return_value=_MOCK_CONSOLIDATION_10,
)
@patch(
    "nhf_spatial_targets.fetch.modis._subset_to_conus",
    side_effect=_mock_subset_side_effect,
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod10c1_search_params(
    mock_login, mock_search, mock_dl, mock_subset, mock_consolidate, workdir
):
    """search_data called with correct short_name and temporal."""
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_download_10c1(workdir)

    from nhf_spatial_targets.fetch.modis import fetch_mod10c1

    fetch_mod10c1(workdir=workdir, period="2005/2005")

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["short_name"] == "MOD10C1"
    assert call_kwargs["temporal"] == ("2005-01-01", "2005-12-31")


# ---- MOD10C1 No granules --------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod10c1",
    return_value=_MOCK_CONSOLIDATION_10,
)
@patch("nhf_spatial_targets.fetch.modis._subset_to_conus")
@patch("earthaccess.download")
@patch("earthaccess.search_data", return_value=[])
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod10c1_no_granules_raises(
    mock_login, mock_search, mock_dl, mock_subset, mock_consolidate, workdir
):
    """ValueError raised when search returns zero granules."""
    from nhf_spatial_targets.fetch.modis import fetch_mod10c1

    with pytest.raises(ValueError, match="No granules found"):
        fetch_mod10c1(workdir=workdir, period="2005/2005")


# ---- MOD10C1 Provenance ---------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod10c1",
    return_value=_MOCK_CONSOLIDATION_10,
)
@patch(
    "nhf_spatial_targets.fetch.modis._subset_to_conus",
    side_effect=_mock_subset_side_effect,
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod10c1_provenance_record(
    mock_login, mock_search, mock_dl, mock_subset, mock_consolidate, workdir
):
    """Returned dict has all required provenance keys."""
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_download_10c1(workdir)

    from nhf_spatial_targets.fetch.modis import fetch_mod10c1

    result = fetch_mod10c1(workdir=workdir, period="2005/2005")

    assert result["source_key"] == _SOURCE_KEY_10
    assert "access_url" in result
    assert result["variables"] == ["Day_CMG_Snow_Cover", "Snow_Spatial_QA"]
    assert result["period"] == "2005/2005"
    assert "bbox" in result
    assert "download_timestamp" in result
    assert isinstance(result["files"], list)
    assert "consolidated_ncs" in result


# ---- MOD10C1 Subset called -------------------------------------------------


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod10c1",
    return_value=_MOCK_CONSOLIDATION_10,
)
@patch(
    "nhf_spatial_targets.fetch.modis._subset_to_conus",
    side_effect=_mock_subset_side_effect,
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod10c1_subset_called(
    mock_login, mock_search, mock_dl, mock_subset, mock_consolidate, workdir
):
    """_subset_to_conus called for each downloaded HDF."""
    n_files = 3
    mock_search.return_value = [_mock_granule(f"g{i}") for i in range(n_files)]
    mock_dl.return_value = _fake_download_10c1(workdir, n=n_files)

    from nhf_spatial_targets.fetch.modis import fetch_mod10c1

    fetch_mod10c1(workdir=workdir, period="2005/2005")

    assert mock_subset.call_count == n_files


# ---- Granule grouping by timestep -----------------------------------------


def test_group_granules_by_timestep():
    """Granules are grouped by AYYYYDDD token."""
    from nhf_spatial_targets.fetch.modis import _group_granules_by_timestep

    granules = []
    for doy in [1, 1, 9, 9, 9]:
        g = _mock_granule(f"MOD16A2GF.A2010{doy:03d}.h08v04.061.2020256154955.hdf")
        g.data_links.return_value = [
            f"https://example.com/MOD16A2GF.A2010{doy:03d}.h08v04.061.2020256154955.hdf"
        ]
        granules.append(g)

    groups = _group_granules_by_timestep(granules)

    assert sorted(groups.keys()) == ["2010001", "2010009"]
    assert len(groups["2010001"]) == 2
    assert len(groups["2010009"]) == 3


def test_group_granules_by_timestep_empty_data_links():
    """Granules with empty data_links are dropped."""
    from nhf_spatial_targets.fetch.modis import _group_granules_by_timestep

    g = _mock_granule("MOD16A2GF.A2010001.h08v04.061.hdf")
    g.data_links.return_value = []

    groups = _group_granules_by_timestep([g])
    assert groups == {}


def test_group_granules_by_timestep_unparseable_filename():
    """Granules with unparseable filenames are dropped."""
    from nhf_spatial_targets.fetch.modis import _group_granules_by_timestep

    g = _mock_granule("not_a_modis_file.hdf")
    g.data_links.return_value = ["https://example.com/not_a_modis_file.hdf"]

    groups = _group_granules_by_timestep([g])
    assert groups == {}


# ---- Granule bbox filtering -------------------------------------------------


def _make_umm_granule(
    name: str,
    west: float,
    south: float,
    east: float,
    north: float,
) -> dict:
    """Create a dict mimicking an earthaccess granule with UMM spatial metadata."""
    return {
        "umm": {
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {
                        "BoundingRectangles": [
                            {
                                "WestBoundingCoordinate": west,
                                "EastBoundingCoordinate": east,
                                "SouthBoundingCoordinate": south,
                                "NorthBoundingCoordinate": north,
                            }
                        ]
                    }
                }
            }
        }
    }


def test_granule_overlaps_bbox_inside():
    """Granule fully inside bbox returns True."""
    from nhf_spatial_targets.fetch.modis import _granule_overlaps_bbox

    g = _make_umm_granule("tile", west=-110, south=30, east=-100, north=40)
    assert _granule_overlaps_bbox(g, (-130, 20, -60, 55)) is True


def test_granule_overlaps_bbox_outside():
    """Granule fully outside bbox returns False."""
    from nhf_spatial_targets.fetch.modis import _granule_overlaps_bbox

    # Iceland tile — outside CONUS
    g = _make_umm_granule("tile", west=-25, south=63, east=-13, north=66)
    assert _granule_overlaps_bbox(g, (-130, 20, -60, 55)) is False


def test_granule_overlaps_bbox_partial():
    """Granule partially overlapping bbox returns True."""
    from nhf_spatial_targets.fetch.modis import _granule_overlaps_bbox

    # Overlaps western edge
    g = _make_umm_granule("tile", west=-140, south=30, east=-120, north=40)
    assert _granule_overlaps_bbox(g, (-130, 20, -60, 55)) is True


def test_granule_overlaps_bbox_edge_touching():
    """Granule touching bbox edge returns True."""
    from nhf_spatial_targets.fetch.modis import _granule_overlaps_bbox

    # East edge of granule == west edge of bbox
    g = _make_umm_granule("tile", west=-140, south=30, east=-130, north=40)
    assert _granule_overlaps_bbox(g, (-130, 20, -60, 55)) is True


def test_granule_overlaps_bbox_missing_metadata():
    """Granule with no UMM metadata returns True (fail-open)."""
    from nhf_spatial_targets.fetch.modis import _granule_overlaps_bbox

    assert _granule_overlaps_bbox({}, (-130, 20, -60, 55)) is True


def test_granule_overlaps_bbox_non_list_rects():
    """Granule with non-list BoundingRectangles returns True (fail-open)."""
    from nhf_spatial_targets.fetch.modis import _granule_overlaps_bbox

    # MagicMock-like object — BoundingRectangles is not a list
    g = {
        "umm": {
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {"BoundingRectangles": "not-a-list"}
                }
            }
        }
    }
    assert _granule_overlaps_bbox(g, (-130, 20, -60, 55)) is True


def test_granule_overlaps_bbox_missing_coord_keys():
    """Granule with incomplete bounding rect returns True (fail-open)."""
    from nhf_spatial_targets.fetch.modis import _granule_overlaps_bbox

    g = {
        "umm": {
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {
                        "BoundingRectangles": [{"WestBoundingCoordinate": -110}]
                    }
                }
            }
        }
    }
    assert _granule_overlaps_bbox(g, (-130, 20, -60, 55)) is True


def test_filter_granules_by_bbox():
    """Filter keeps overlapping granules and drops non-overlapping ones."""
    from nhf_spatial_targets.fetch.modis import _filter_granules_by_bbox

    conus = _make_umm_granule("conus", west=-110, south=30, east=-100, north=40)
    iceland = _make_umm_granule("iceland", west=-25, south=63, east=-13, north=66)
    bbox = (-130, 20, -60, 55)

    result = _filter_granules_by_bbox([conus, iceland], bbox)
    assert len(result) == 1
    assert result[0] is conus
