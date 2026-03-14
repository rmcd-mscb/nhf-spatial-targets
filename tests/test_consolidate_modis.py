"""Tests for MODIS consolidation (MOD10C1 and MOD16A2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from nhf_spatial_targets.fetch.consolidate import _time_from_modis_filename


def test_time_from_modis_filename():
    """Extract date from MODIS AYYYYDDD filename."""
    t = _time_from_modis_filename(Path("MOD16A2GF.A2010001.h08v04.061.hdf"))
    assert t == pd.Timestamp("2010-01-01")

    t = _time_from_modis_filename(Path("MOD16A2GF.A2010009.h08v04.061.hdf"))
    assert t == pd.Timestamp("2010-01-09")

    t = _time_from_modis_filename(Path("MOD10C1.A2010032.061.conus.nc"))
    assert t == pd.Timestamp("2010-02-01")


def test_time_from_modis_filename_bad():
    """Raises ValueError for non-MODIS filename."""
    with pytest.raises(ValueError, match="Cannot extract date"):
        _time_from_modis_filename(Path("random_file.nc"))


@pytest.fixture()
def mod10c1_run_dir(tmp_path: Path) -> Path:
    """Create a run workspace with 3 synthetic MOD10C1 .conus.nc files.

    Files simulate DOY 1, 2, 3 of year 2010.  Each has Day_CMG_Snow_Cover
    and Snow_Spatial_QA on a small lat/lon grid (4x6) with NO time dimension,
    matching real ``_subset_to_conus`` output.
    """
    source_key = "mod10c1_v061"
    out = tmp_path / "data" / "raw" / source_key
    out.mkdir(parents=True)

    lat = np.linspace(25.0, 50.0, 4)
    lon = np.linspace(-125.0, -65.0, 6)

    for doy in range(1, 4):
        ds = xr.Dataset(
            {
                "Day_CMG_Snow_Cover": (
                    ["lat", "lon"],
                    np.random.rand(len(lat), len(lon)).astype(np.float32),
                ),
                "Snow_Spatial_QA": (
                    ["lat", "lon"],
                    np.random.randint(0, 4, (len(lat), len(lon)), dtype=np.int8),
                ),
            },
            coords={"lat": lat, "lon": lon},
        )
        fname = f"MOD10C1.A2010{doy:03d}.061.conus.nc"
        ds.to_netcdf(out / fname)

    return tmp_path


def test_consolidate_mod10c1_basic(mod10c1_run_dir: Path) -> None:
    """Output has time dim with 3 steps, both variables, correct provenance."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod10c1

    source_key = "mod10c1_v061"
    variables = ["Day_CMG_Snow_Cover", "Snow_Spatial_QA"]

    result = consolidate_mod10c1(
        run_dir=mod10c1_run_dir,
        source_key=source_key,
        variables=variables,
        year=2010,
    )

    out_path = (
        mod10c1_run_dir
        / "data"
        / "raw"
        / source_key
        / f"{source_key}_2010_consolidated.nc"
    )
    assert out_path.exists()

    ds = xr.open_dataset(out_path)
    assert "time" in ds.dims
    assert len(ds.time) == 3
    assert "Day_CMG_Snow_Cover" in ds.data_vars
    assert "Snow_Spatial_QA" in ds.data_vars
    ds.close()

    assert (
        result["consolidated_nc"]
        == f"data/raw/{source_key}/{source_key}_2010_consolidated.nc"
    )
    assert "last_consolidated_utc" in result
    assert result["n_files"] == 3
    assert result["variables"] == variables


def test_consolidate_mod10c1_no_files(tmp_path: Path) -> None:
    """FileNotFoundError raised for empty directory."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod10c1

    source_key = "mod10c1_v061"
    (tmp_path / "data" / "raw" / source_key).mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No .conus.nc files"):
        consolidate_mod10c1(
            run_dir=tmp_path,
            source_key=source_key,
            variables=["Day_CMG_Snow_Cover"],
            year=2010,
        )


def test_consolidate_mod10c1_overwrites_existing(mod10c1_run_dir: Path) -> None:
    """Re-running consolidation produces the same file path."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod10c1

    source_key = "mod10c1_v061"
    variables = ["Day_CMG_Snow_Cover"]

    result1 = consolidate_mod10c1(
        run_dir=mod10c1_run_dir,
        source_key=source_key,
        variables=variables,
        year=2010,
    )
    result2 = consolidate_mod10c1(
        run_dir=mod10c1_run_dir,
        source_key=source_key,
        variables=variables,
        year=2010,
    )

    assert result1["consolidated_nc"] == result2["consolidated_nc"]
    out_path = mod10c1_run_dir / result2["consolidated_nc"]
    assert out_path.exists()


def test_consolidate_mod10c1_filters_year(mod10c1_run_dir: Path) -> None:
    """Adding a 2011 file does not affect 2010 consolidation."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod10c1

    source_key = "mod10c1_v061"
    source_dir = mod10c1_run_dir / "data" / "raw" / source_key

    # Add a file for 2011 DOY 1
    lat = np.linspace(25.0, 50.0, 4)
    lon = np.linspace(-125.0, -65.0, 6)
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (
                ["lat", "lon"],
                np.random.rand(len(lat), len(lon)).astype(np.float32),
            ),
            "Snow_Spatial_QA": (
                ["lat", "lon"],
                np.random.randint(0, 4, (len(lat), len(lon)), dtype=np.int8),
            ),
        },
        coords={"lat": lat, "lon": lon},
    )
    ds.to_netcdf(source_dir / "MOD10C1.A2011001.061.conus.nc")

    result = consolidate_mod10c1(
        run_dir=mod10c1_run_dir,
        source_key=source_key,
        variables=["Day_CMG_Snow_Cover", "Snow_Spatial_QA"],
        year=2010,
    )

    assert result["n_files"] == 3

    out_path = mod10c1_run_dir / result["consolidated_nc"]
    ds_out = xr.open_dataset(out_path)
    assert len(ds_out.time) == 3
    ds_out.close()


# ---------------------------------------------------------------------------
# MOD16A2 consolidation tests
# ---------------------------------------------------------------------------


def _make_sinusoidal_tile(path: Path, h: int, v: int, value: int = 42) -> None:
    """Write a small sinusoidal-projected GeoTIFF (with .hdf extension)."""
    srs = CRS.from_proj4(
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
    )
    nx, ny = 4, 4
    x0 = -10_000_000 + h * 200_000
    y0 = 6_000_000 - v * 200_000
    res = 500.0
    transform = from_bounds(x0, y0 - ny * res, x0 + nx * res, y0, nx, ny)
    data = np.full((1, ny, nx), value, dtype=np.int16)
    da = xr.DataArray(data, dims=["band", "y", "x"], coords={"band": [1]})
    da.rio.write_crs(srs, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_nodata(-1, inplace=True)
    da.rio.to_raster(path, driver="GTiff")


@pytest.fixture()
def mod16a2_run_dir(tmp_path):
    """Create a run workspace with 2 timesteps × 2 tiles of synthetic HDF files."""
    rd = tmp_path / "run"
    rd.mkdir()
    source_dir = rd / "data" / "raw" / "mod16a2_v061"
    source_dir.mkdir(parents=True)
    for doy in [1, 9]:
        for h, v in [(8, 4), (9, 4)]:
            fname = f"MOD16A2GF.A2010{doy:03d}.h{h:02d}v{v:02d}.061.2020256154955.hdf"
            _make_sinusoidal_tile(source_dir / fname, h, v, value=doy * 10)
    return rd


def test_consolidate_mod16a2_no_files(tmp_path: Path) -> None:
    """FileNotFoundError raised for empty directory."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2

    source_key = "mod16a2_v061"
    (tmp_path / "data" / "raw" / source_key).mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No .hdf files"):
        consolidate_mod16a2(
            run_dir=tmp_path,
            source_key=source_key,
            variables=["ET_500m"],
            year=2010,
        )


def test_consolidate_mod16a2_synthetic(mod16a2_run_dir: Path) -> None:
    """Full pipeline: mosaic, reproject, stack along time."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2

    source_key = "mod16a2_v061"
    variables = ["ET_500m"]

    result = consolidate_mod16a2(
        run_dir=mod16a2_run_dir,
        source_key=source_key,
        variables=variables,
        year=2010,
    )

    out_path = mod16a2_run_dir / result["consolidated_nc"]
    assert out_path.exists()

    ds = xr.open_dataset(out_path)

    # Should have 2 time steps (DOY 1 and DOY 9)
    assert "time" in ds.dims
    assert len(ds.time) == 2

    # Should have lat/lon coordinates in EPSG:4326
    assert "lat" in ds.dims or "lat" in ds.coords
    assert "lon" in ds.dims or "lon" in ds.coords

    # Variable is present
    assert "ET_500m" in ds.data_vars

    ds.close()

    # Provenance checks
    assert result["n_files"] == 4
    assert result["variables"] == variables
    assert "last_consolidated_utc" in result
    assert (
        result["consolidated_nc"]
        == f"data/raw/{source_key}/{source_key}_2010_consolidated.nc"
    )


def test_consolidate_mod16a2_partial_tiles(mod16a2_run_dir: Path) -> None:
    """Consolidation succeeds when one tile is missing from a timestep."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2

    source_key = "mod16a2_v061"
    source_dir = mod16a2_run_dir / "data" / "raw" / source_key

    # Delete the h09v04 tile from DOY 009
    doy9_h09 = list(source_dir.glob("MOD16A2GF.A2010009.h09v04.*"))
    assert len(doy9_h09) == 1, "Expected exactly one h09v04 tile for DOY 009"
    doy9_h09[0].unlink()

    result = consolidate_mod16a2(
        run_dir=mod16a2_run_dir,
        source_key=source_key,
        variables=["ET_500m"],
        year=2010,
    )

    # 3 remaining HDF files (2 for DOY 001, 1 for DOY 009)
    assert result["n_files"] == 3

    out_path = mod16a2_run_dir / result["consolidated_nc"]
    ds = xr.open_dataset(out_path)
    # Both timesteps should still be present
    assert ds.sizes["time"] == 2
    ds.close()


def test_consolidate_mod16a2_overwrites_existing(mod16a2_run_dir: Path) -> None:
    """Re-running consolidation is idempotent and produces the same file."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2

    source_key = "mod16a2_v061"
    variables = ["ET_500m"]

    result1 = consolidate_mod16a2(
        run_dir=mod16a2_run_dir,
        source_key=source_key,
        variables=variables,
        year=2010,
    )
    result2 = consolidate_mod16a2(
        run_dir=mod16a2_run_dir,
        source_key=source_key,
        variables=variables,
        year=2010,
    )

    assert result1["consolidated_nc"] == result2["consolidated_nc"]
    out_path = mod16a2_run_dir / result2["consolidated_nc"]
    assert out_path.exists()
