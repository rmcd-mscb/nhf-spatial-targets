"""Tests for Reitz 2017 recharge fetch module."""

from __future__ import annotations

from pathlib import Path

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
