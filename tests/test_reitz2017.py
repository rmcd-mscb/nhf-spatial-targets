"""Tests for Reitz 2017 recharge fetch module."""

from __future__ import annotations

import json
import sys
import types
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr


def _make_reitz_tif(path: Path, *, value: float = 1.0) -> Path:
    """Create a small synthetic GeoTIFF mimicking Reitz output.

    4x4 grid in NAD83 geographic coordinates (EPSG:4269), matching the
    actual Reitz 2017 source data CRS.
    """
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    ny, nx = 4, 4
    data = np.full((ny, nx), value, dtype=np.float32)
    transform = from_bounds(-100.0, 35.0, -99.0, 36.0, nx, ny)
    da = xr.DataArray(
        data[np.newaxis, :, :],  # (band=1, y, x)
        dims=["band", "y", "x"],
    )
    da.rio.write_crs(CRS.from_epsg(4269), inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_nodata(-9999.0, inplace=True)
    da.rio.to_raster(path)
    return path


def test_consolidate_builds_nc(tmp_path: Path):
    """Consolidation reads GeoTIFFs, builds NC with both vars and time."""
    from nhf_spatial_targets.fetch.reitz2017 import _consolidate

    output_dir = tmp_path / "reitz"
    output_dir.mkdir()

    for year in [2005, 2006]:
        _make_reitz_tif(output_dir / f"TotalRecharge_{year}.tif", value=float(year))
        _make_reitz_tif(output_dir / f"EffRecharge_{year}.tif", value=float(year) + 0.5)

    nc_path = _consolidate(output_dir, "2005/2006")

    assert nc_path.exists()
    ds = xr.open_dataset(nc_path)
    assert "total_recharge" in ds.data_vars
    assert "eff_recharge" in ds.data_vars
    assert ds.sizes["time"] == 2
    # Check time values are mid-year
    assert str(ds.time.values[0])[:10] == "2005-07-01"
    assert str(ds.time.values[1])[:10] == "2006-07-01"
    # Check data values round-trip
    assert float(ds["total_recharge"].isel(time=0).mean()) == pytest.approx(2005.0)
    assert float(ds["eff_recharge"].isel(time=0).mean()) == pytest.approx(2005.5)

    # CF-compliance: CRS variable
    assert "crs" in ds.data_vars
    assert "spatial_ref" not in ds.data_vars
    crs_attrs = ds["crs"].attrs
    assert "crs_wkt" in crs_attrs
    assert "NAD83" in crs_attrs["crs_wkt"]
    assert crs_attrs["grid_mapping_name"] == "latitude_longitude"
    assert crs_attrs["semi_major_axis"] == pytest.approx(6378137.0)

    # CF-compliance: grid_mapping on data variables
    assert ds["total_recharge"].attrs["grid_mapping"] == "crs"
    assert ds["eff_recharge"].attrs["grid_mapping"] == "crs"

    # CF-compliance: variable metadata (cf_units from catalog)
    assert ds["total_recharge"].attrs["units"] == "inches yr-1"
    assert "long_name" in ds["total_recharge"].attrs
    assert ds["eff_recharge"].attrs["units"] == "inches yr-1"

    # CF-compliance: coordinates renamed to lat/lon by apply_cf_metadata
    assert "lat" in ds.dims
    assert "lon" in ds.dims
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lat.attrs["units"] == "degrees_north"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert ds.lon.attrs["units"] == "degrees_east"
    assert ds.time.attrs["standard_name"] == "time"

    # CF-compliance: global attribute
    assert ds.attrs["Conventions"] == "CF-1.6"
    ds.close()


@pytest.fixture()
def _mock_sciencebasepy():
    """Inject a fake sciencebasepy module so the lazy import resolves."""
    fake = types.ModuleType("sciencebasepy")
    fake.SbSession = MagicMock()
    original = sys.modules.get("sciencebasepy")
    sys.modules["sciencebasepy"] = fake
    yield fake
    if original is not None:
        sys.modules["sciencebasepy"] = original
    else:
        sys.modules.pop("sciencebasepy", None)


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    """Create a minimal run workspace."""
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "reitz2017").mkdir(parents=True)
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


def _make_reitz_zip(zip_path: Path, tif_name: str, *, value: float = 1.0) -> Path:
    """Create a zip file containing a synthetic GeoTIFF."""
    tif_path = zip_path.parent / tif_name
    _make_reitz_tif(tif_path, value=value)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(tif_path, tif_name)
    tif_path.unlink()
    return zip_path


def _setup_sb_mock(
    mock_sciencebasepy,
    workdir: Path,
    years: list[int],
) -> MagicMock:
    """Configure sciencebasepy mock to serve synthetic zips for given years."""
    staging = workdir / ".sb_staging"
    staging.mkdir(exist_ok=True)

    # Build file info list matching what ScienceBase returns
    file_infos = []
    for year in years:
        for prefix in ["TotalRecharge", "EffRecharge"]:
            zip_name = f"{prefix}_{year}.zip"
            tif_name = f"{prefix}_{year}.tif"
            value = float(year) if prefix == "TotalRecharge" else float(year) + 0.5
            _make_reitz_zip(staging / zip_name, tif_name, value=value)
            file_infos.append({"name": zip_name, "url": f"https://fake/{zip_name}"})

    mock_sb = MagicMock()
    mock_sb.get_item.return_value = {"id": "55d383a9e4b0518e35468e58"}
    mock_sb.get_item_file_info.return_value = file_infos

    def fake_download_file(url, name, dest):
        """Copy the staging zip to the destination directory."""
        import shutil

        src = staging / name
        shutil.copy2(str(src), str(Path(dest) / name))

    mock_sb.download_file = MagicMock(side_effect=fake_download_file)
    mock_sciencebasepy.SbSession = MagicMock(return_value=mock_sb)
    return mock_sb


def test_downloads_and_updates_manifest(workdir: Path, _mock_sciencebasepy):
    """Full download flow with mocked ScienceBase."""
    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017

    _setup_sb_mock(_mock_sciencebasepy, workdir, [2005, 2006])

    result = fetch_reitz2017(workdir=workdir, period="2005/2006")

    output_dir = workdir / "data" / "raw" / "reitz2017"
    # GeoTIFFs extracted
    assert (output_dir / "TotalRecharge_2005.tif").exists()
    assert (output_dir / "EffRecharge_2005.tif").exists()
    # Consolidated NC created
    assert (output_dir / "reitz2017_consolidated.nc").exists()
    # Manifest updated
    manifest = json.loads((workdir / "manifest.json").read_text())
    assert "reitz2017" in manifest["sources"]
    assert manifest["sources"]["reitz2017"]["license"] == "public domain (USGS)"
    assert result["source_key"] == "reitz2017"


def test_year_from_filename_valid():
    """Extract year from a standard Reitz filename."""
    from nhf_spatial_targets.fetch.reitz2017 import _year_from_filename

    assert _year_from_filename(Path("TotalRecharge_2005.tif")) == 2005
    assert _year_from_filename(Path("EffRecharge_2013.tif")) == 2013


def test_year_from_filename_no_underscore():
    """ValueError when filename has no underscore."""
    from nhf_spatial_targets.fetch.reitz2017 import _year_from_filename

    with pytest.raises(ValueError, match="Cannot extract year"):
        _year_from_filename(Path("badname.tif"))


def test_year_from_filename_non_integer():
    """ValueError when suffix after underscore is not an integer."""
    from nhf_spatial_targets.fetch.reitz2017 import _year_from_filename

    with pytest.raises(ValueError, match="Cannot extract year"):
        _year_from_filename(Path("TotalRecharge_abc.tif"))


def test_consolidate_mismatched_years_raises(tmp_path: Path):
    """RuntimeError when TotalRecharge and EffRecharge have different years."""
    from nhf_spatial_targets.fetch.reitz2017 import _consolidate

    output_dir = tmp_path / "reitz"
    output_dir.mkdir()

    # Create TotalRecharge for 2005+2006 but EffRecharge only for 2005
    _make_reitz_tif(output_dir / "TotalRecharge_2005.tif", value=2005.0)
    _make_reitz_tif(output_dir / "TotalRecharge_2006.tif", value=2006.0)
    _make_reitz_tif(output_dir / "EffRecharge_2005.tif", value=2005.5)

    with pytest.raises(RuntimeError, match="Mismatched years"):
        _consolidate(output_dir, "2005/2006")


def test_missing_fabric_raises(tmp_path: Path):
    """FileNotFoundError when fabric.json is absent."""
    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017

    rd = tmp_path / "run"
    rd.mkdir()
    with pytest.raises(FileNotFoundError, match="config.yml"):
        fetch_reitz2017(workdir=rd, period="2005/2006")


def test_skips_existing(workdir: Path, _mock_sciencebasepy):
    """Skip everything when consolidated NC exists and covers period."""
    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017

    output_dir = workdir / "data" / "raw" / "reitz2017"
    # Create synthetic GeoTIFFs and consolidate them
    for year in [2005, 2006]:
        _make_reitz_tif(output_dir / f"TotalRecharge_{year}.tif", value=float(year))
        _make_reitz_tif(output_dir / f"EffRecharge_{year}.tif", value=float(year) + 0.5)

    from nhf_spatial_targets.fetch.reitz2017 import _consolidate

    _consolidate(output_dir, "2005/2006")

    mock_sb = MagicMock()
    _mock_sciencebasepy.SbSession = mock_sb
    result = fetch_reitz2017(workdir=workdir, period="2005/2006")

    mock_sb.assert_not_called()
    assert result["source_key"] == "reitz2017"


def test_existing_nc_period_mismatch_raises(workdir: Path, _mock_sciencebasepy):
    """RuntimeError when consolidated NC exists but doesn't cover requested period."""
    from nhf_spatial_targets.fetch.reitz2017 import _consolidate, fetch_reitz2017

    output_dir = workdir / "data" / "raw" / "reitz2017"
    # Create NC covering only 2005
    _make_reitz_tif(output_dir / "TotalRecharge_2005.tif", value=2005.0)
    _make_reitz_tif(output_dir / "EffRecharge_2005.tif", value=2005.5)
    _consolidate(output_dir, "2005/2005")

    # Request broader period
    with pytest.raises(RuntimeError, match="missing years"):
        fetch_reitz2017(workdir=workdir, period="2005/2006")


def test_incremental_skips_downloaded_years(workdir: Path, _mock_sciencebasepy):
    """Only download years whose GeoTIFFs are missing."""
    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017

    output_dir = workdir / "data" / "raw" / "reitz2017"
    # Pre-create 2005 GeoTIFFs
    _make_reitz_tif(output_dir / "TotalRecharge_2005.tif", value=2005.0)
    _make_reitz_tif(output_dir / "EffRecharge_2005.tif", value=2005.5)

    mock_sb = _setup_sb_mock(_mock_sciencebasepy, workdir, [2006])

    result = fetch_reitz2017(workdir=workdir, period="2005/2006")

    # ScienceBase should only be called for 2006 files
    calls = mock_sb.download_file.call_args_list
    downloaded_names = [c[0][1] for c in calls]
    assert "TotalRecharge_2005.zip" not in downloaded_names
    assert "TotalRecharge_2006.zip" in downloaded_names
    assert result["source_key"] == "reitz2017"


def test_manifest_preserves_existing_sources(workdir: Path, _mock_sciencebasepy):
    """Manifest merge must not overwrite entries from other sources."""
    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017

    existing_manifest = {
        "sources": {
            "merra2": {"source_key": "merra2", "period": "1980/2020"},
        },
        "steps": [],
    }
    (workdir / "manifest.json").write_text(json.dumps(existing_manifest))

    _setup_sb_mock(_mock_sciencebasepy, workdir, [2005])
    fetch_reitz2017(workdir=workdir, period="2005/2005")

    manifest = json.loads((workdir / "manifest.json").read_text())
    assert "merra2" in manifest["sources"]
    assert "reitz2017" in manifest["sources"]


def test_corrupt_manifest_raises(workdir: Path, _mock_sciencebasepy):
    """ValueError when manifest.json is corrupt."""
    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017

    (workdir / "manifest.json").write_text("not valid json{{{")
    _setup_sb_mock(_mock_sciencebasepy, workdir, [2005])

    with pytest.raises(ValueError, match="corrupt"):
        fetch_reitz2017(workdir=workdir, period="2005/2005")


def test_download_failure_raises(workdir: Path, _mock_sciencebasepy):
    """RuntimeError when ScienceBase download fails."""
    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017

    mock_sb = MagicMock()
    mock_sb.get_item.return_value = {"id": "55d383a9e4b0518e35468e58"}
    mock_sb.get_item_file_info.return_value = [
        {"name": "TotalRecharge_2005.zip", "url": "https://fake/tr.zip"},
        {"name": "EffRecharge_2005.zip", "url": "https://fake/er.zip"},
    ]
    mock_sb.download_file.side_effect = Exception("ScienceBase unavailable")
    _mock_sciencebasepy.SbSession = MagicMock(return_value=mock_sb)

    with pytest.raises(RuntimeError, match="ScienceBase download failed"):
        fetch_reitz2017(workdir=workdir, period="2005/2005")


def test_period_out_of_range_raises(workdir: Path):
    """ValueError for period outside 2000-2013."""
    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017

    with pytest.raises(ValueError, match="outside"):
        fetch_reitz2017(workdir=workdir, period="1990/1995")


def test_zip_no_tif_raises(workdir: Path, _mock_sciencebasepy):
    """RuntimeError when zip contains no .tif file."""
    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017

    staging = workdir / ".sb_staging"
    staging.mkdir()
    # Create a zip with a non-tif file
    zip_path = staging / "TotalRecharge_2005.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "not a tif")

    mock_sb = MagicMock()
    mock_sb.get_item.return_value = {"id": "55d383a9e4b0518e35468e58"}
    mock_sb.get_item_file_info.return_value = [
        {"name": "TotalRecharge_2005.zip", "url": "https://fake/tr.zip"},
        {"name": "EffRecharge_2005.zip", "url": "https://fake/er.zip"},
    ]

    import shutil

    def fake_download_file(url, name, dest):
        shutil.copy2(str(staging / name), str(Path(dest) / name))

    # Create EffRecharge zip with a real tif so the error happens on TotalRecharge
    eff_zip = staging / "EffRecharge_2005.zip"
    eff_tif = staging / "EffRecharge_2005.tif"
    _make_reitz_tif(eff_tif, value=1.0)
    with zipfile.ZipFile(eff_zip, "w") as zf:
        zf.write(eff_tif, "EffRecharge_2005.tif")

    mock_sb.download_file = MagicMock(side_effect=fake_download_file)
    _mock_sciencebasepy.SbSession = MagicMock(return_value=mock_sb)

    with pytest.raises(RuntimeError, match="No .tif file"):
        fetch_reitz2017(workdir=workdir, period="2005/2005")


def test_cli_nonexistent_run_dir(tmp_path: Path):
    """CLI exits with error for nonexistent run directory."""
    from nhf_spatial_targets.cli import app

    with pytest.raises(SystemExit) as exc_info:
        app(
            [
                "fetch",
                "reitz2017",
                "--workdir",
                str(tmp_path / "nope"),
                "--period",
                "2005/2006",
            ],
            exit_on_error=False,
        )
    assert exc_info.value.code == 2


@pytest.mark.integration
def test_fetch_reitz2017_real_download(tmp_path: Path):
    """Integration: fetch a single year from real ScienceBase.

    Requires network access. Run with: pixi run -e dev test-integration
    """
    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017

    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "reitz2017").mkdir(parents=True)
    fabric = {
        "bbox_buffered": {
            "minx": -125.1,
            "miny": 23.9,
            "maxx": -65.9,
            "maxy": 50.1,
        }
    }
    (rd / "fabric.json").write_text(json.dumps(fabric))
    import yaml as _yaml

    _cfg = {
        "fabric": {"path": "", "id_col": "nhm_id"},
        "datastore": str(rd / "data" / "raw"),
        "dir_mode": "2775",
    }
    (rd / "config.yml").write_text(_yaml.dump(_cfg))
    (rd / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))

    result = fetch_reitz2017(workdir=rd, period="2005/2005")

    assert result["source_key"] == "reitz2017"

    output_dir = rd / "data" / "raw" / "reitz2017"
    # Raw GeoTIFFs exist
    assert (output_dir / "TotalRecharge_2005.tif").exists()
    assert (output_dir / "EffRecharge_2005.tif").exists()

    # Consolidated NC exists with both variables
    nc_path = output_dir / "reitz2017_consolidated.nc"
    assert nc_path.exists()

    ds = xr.open_dataset(nc_path)
    assert "total_recharge" in ds.data_vars
    assert "eff_recharge" in ds.data_vars
    assert ds.sizes["time"] == 1
    assert str(ds.time.values[0])[:10] == "2005-07-01"
    ds.close()

    # Manifest updated
    manifest = json.loads((rd / "manifest.json").read_text())
    assert "reitz2017" in manifest["sources"]
