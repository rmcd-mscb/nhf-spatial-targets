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
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr


def _make_reitz_tif(path: Path, *, value: float = 1.0) -> Path:
    """Create a small synthetic GeoTIFF mimicking Reitz output.

    4x4 grid in Albers Equal Area CONUS projection (EPSG:5070).
    """
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    ny, nx = 4, 4
    data = np.full((ny, nx), value, dtype=np.float32)
    transform = from_bounds(-2356114, 277962, -2355314, 278762, nx, ny)
    da = xr.DataArray(
        data[np.newaxis, :, :],  # (band=1, y, x)
        dims=["band", "y", "x"],
    )
    da.rio.write_crs(CRS.from_epsg(5070), inplace=True)
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
def run_dir(tmp_path: Path) -> Path:
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
    run_dir: Path,
    years: list[int],
) -> MagicMock:
    """Configure sciencebasepy mock to serve synthetic zips for given years."""
    staging = run_dir / ".sb_staging"
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


def test_downloads_and_updates_manifest(run_dir: Path, _mock_sciencebasepy):
    """Full download flow with mocked ScienceBase."""
    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017

    _setup_sb_mock(_mock_sciencebasepy, run_dir, [2005, 2006])

    result = fetch_reitz2017(run_dir=run_dir, period="2005/2006")

    output_dir = run_dir / "data" / "raw" / "reitz2017"
    # GeoTIFFs extracted
    assert (output_dir / "TotalRecharge_2005.tif").exists()
    assert (output_dir / "EffRecharge_2005.tif").exists()
    # Consolidated NC created
    assert (output_dir / "reitz2017_consolidated.nc").exists()
    # Manifest updated
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "reitz2017" in manifest["sources"]
    assert manifest["sources"]["reitz2017"]["license"] == "public domain (USGS)"
    assert result["source_key"] == "reitz2017"
