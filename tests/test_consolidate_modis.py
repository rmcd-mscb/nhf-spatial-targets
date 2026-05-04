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

from unittest.mock import patch as _patch

from nhf_spatial_targets.fetch.consolidate import _time_from_modis_filename

# Bbox covering CONUS in EPSG:4326 — used for MOD16A2 tests where the
# synthetic sinusoidal tiles (h08v04, h09v04) reproject to ~47°N, -110°W.
_TEST_BBOX = (-130.0, 20.0, -60.0, 55.0)


def test_time_from_modis_filename():
    """Extract date from MODIS AYYYYDDD filename."""
    t = _time_from_modis_filename(Path("MOD16A2GF.A2010001.h08v04.061.hdf"))
    assert t == pd.Timestamp("2010-01-01")

    t = _time_from_modis_filename(Path("MOD16A2GF.A2010009.h08v04.061.hdf"))
    assert t == pd.Timestamp("2010-01-09")

    t = _time_from_modis_filename(Path("MOD10C1.A2010032.061.conus.nc"))
    assert t == pd.Timestamp("2010-02-01")

    # DOY 365 in a non-leap year → Dec 31
    t = _time_from_modis_filename(Path("MOD10C1.A2010365.061.conus.nc"))
    assert t == pd.Timestamp("2010-12-31")

    # DOY 366 in leap year 2000 → Dec 31
    t = _time_from_modis_filename(Path("MOD10C1.A2000366.061.conus.nc"))
    assert t == pd.Timestamp("2000-12-31")


def test_time_from_modis_filename_bad():
    """Raises ValueError for non-MODIS filename."""
    with pytest.raises(ValueError, match="Cannot extract date"):
        _time_from_modis_filename(Path("random_file.nc"))


@pytest.fixture()
def mod10c1_run_dir(tmp_path: Path) -> Path:
    """Create a source directory with 3 synthetic MOD10C1 .conus.nc files.

    Files simulate DOY 1, 2, 3 of year 2010.  Each has Day_CMG_Snow_Cover
    and Snow_Spatial_QA on a small lat/lon grid (4x6) with NO time dimension,
    matching real ``_subset_to_conus`` output.

    Returns the source directory (not the parent).
    """
    source_key = "mod10c1_v061"
    out = tmp_path / source_key
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

    return out


def test_consolidate_mod10c1_basic(mod10c1_run_dir: Path) -> None:
    """Output has time dim with 3 steps, both variables, correct provenance."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod10c1

    source_key = "mod10c1_v061"
    variables = ["Day_CMG_Snow_Cover", "Snow_Spatial_QA"]

    result = consolidate_mod10c1(
        source_dir=mod10c1_run_dir,
        source_key=source_key,
        variables=variables,
        year=2010,
    )

    out_path = mod10c1_run_dir / f"{source_key}_2010_consolidated.nc"
    assert out_path.exists()

    ds = xr.open_dataset(out_path)
    assert "time" in ds.dims
    assert len(ds.time) == 3
    assert "Day_CMG_Snow_Cover" in ds.data_vars
    assert "Snow_Spatial_QA" in ds.data_vars
    ds.close()

    assert f"{source_key}_2010_consolidated.nc" in result["consolidated_nc"]
    assert "last_consolidated_utc" in result
    assert result["n_files"] == 3
    assert result["variables"] == variables


def test_consolidate_mod10c1_no_files(tmp_path: Path) -> None:
    """FileNotFoundError raised for empty directory."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod10c1

    source_key = "mod10c1_v061"
    source_dir = tmp_path / source_key
    source_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No .conus.nc files"):
        consolidate_mod10c1(
            source_dir=source_dir,
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
        source_dir=mod10c1_run_dir,
        source_key=source_key,
        variables=variables,
        year=2010,
    )
    result2 = consolidate_mod10c1(
        source_dir=mod10c1_run_dir,
        source_key=source_key,
        variables=variables,
        year=2010,
    )

    assert result1["consolidated_nc"] == result2["consolidated_nc"]
    out_path = mod10c1_run_dir / result2["consolidated_nc"]
    assert out_path.exists()


def test_consolidate_mod10c1_missing_variable_raises(mod10c1_run_dir: Path) -> None:
    """ValueError raised when requesting a nonexistent variable."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod10c1

    with pytest.raises(ValueError, match="NONEXISTENT"):
        consolidate_mod10c1(
            source_dir=mod10c1_run_dir,
            source_key="mod10c1_v061",
            variables=["NONEXISTENT"],
            year=2010,
        )


def test_consolidate_mod10c1_filters_variables(tmp_path: Path) -> None:
    """Only requested variables appear in consolidated output."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod10c1

    source_key = "mod10c1_v061"
    out = tmp_path / source_key
    out.mkdir(parents=True)

    lat = np.linspace(25.0, 50.0, 4)
    lon = np.linspace(-125.0, -65.0, 6)

    for doy in range(1, 3):
        ds = xr.Dataset(
            {
                "Day_CMG_Snow_Cover": (
                    ["lat", "lon"],
                    np.random.rand(len(lat), len(lon)).astype(np.float32),
                ),
                "ExtraVar": (
                    ["lat", "lon"],
                    np.random.rand(len(lat), len(lon)).astype(np.float32),
                ),
            },
            coords={"lat": lat, "lon": lon},
        )
        fname = f"MOD10C1.A2010{doy:03d}.061.conus.nc"
        ds.to_netcdf(out / fname)

    consolidate_mod10c1(
        source_dir=out,
        source_key=source_key,
        variables=["Day_CMG_Snow_Cover"],
        year=2010,
    )

    out_path = out / f"{source_key}_2010_consolidated.nc"
    ds_out = xr.open_dataset(out_path)
    assert "Day_CMG_Snow_Cover" in ds_out.data_vars
    assert "ExtraVar" not in ds_out.data_vars
    ds_out.close()


def test_consolidate_mod10c1_sorts_time(mod10c1_run_dir: Path) -> None:
    """Consolidated time dimension is monotonically increasing."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod10c1

    result = consolidate_mod10c1(
        source_dir=mod10c1_run_dir,
        source_key="mod10c1_v061",
        variables=["Day_CMG_Snow_Cover"],
        year=2010,
    )

    out_path = Path(result["consolidated_nc"])
    ds = xr.open_dataset(out_path)
    assert pd.DatetimeIndex(ds.time.values).is_monotonic_increasing
    ds.close()


def test_consolidate_mod10c1_filters_year(mod10c1_run_dir: Path) -> None:
    """Adding a 2011 file does not affect 2010 consolidation."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod10c1

    source_key = "mod10c1_v061"
    source_dir = mod10c1_run_dir

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
        source_dir=mod10c1_run_dir,
        source_key=source_key,
        variables=["Day_CMG_Snow_Cover", "Snow_Spatial_QA"],
        year=2010,
    )

    assert result["n_files"] == 3

    out_path = Path(result["consolidated_nc"])
    ds_out = xr.open_dataset(out_path)
    assert len(ds_out.time) == 3
    ds_out.close()


# ---------------------------------------------------------------------------
# MOD16A2 consolidation tests
# ---------------------------------------------------------------------------


def _make_sinusoidal_tile(path: Path, h: int, v: int, value: int = 42) -> None:
    """Write a sinusoidal-projected GeoTIFF (with .hdf extension).

    Tiles are 48x48 pixels at 500m — large enough to produce a multi-pixel
    grid after reprojection to 0.04° EPSG:4326.
    """
    srs = CRS.from_proj4(
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
    )
    nx, ny = 48, 48
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
    """Create a source directory with 2 timesteps x 2 tiles of synthetic HDF files.

    Returns the source directory directly.
    """
    source_dir = tmp_path / "mod16a2_v061"
    source_dir.mkdir(parents=True)
    for doy in [1, 9]:
        for h, v in [(8, 4), (9, 4)]:
            fname = f"MOD16A2GF.A2010{doy:03d}.h{h:02d}v{v:02d}.061.2020256154955.hdf"
            _make_sinusoidal_tile(source_dir / fname, h, v, value=doy * 10)
    return source_dir


def test_consolidate_mod16a2_no_files(tmp_path: Path) -> None:
    """FileNotFoundError raised for empty directory."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2

    source_key = "mod16a2_v061"
    source_dir = tmp_path / source_key
    source_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No .hdf files"):
        consolidate_mod16a2(
            source_dir=source_dir,
            source_key=source_key,
            variables=["ET_500m"],
            year=2010,
            bbox=_TEST_BBOX,
        )


def test_consolidate_mod16a2_synthetic(mod16a2_run_dir: Path) -> None:
    """Full pipeline: mosaic, reproject, stack along time."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2

    source_key = "mod16a2_v061"
    variables = ["ET_500m"]

    result = consolidate_mod16a2(
        source_dir=mod16a2_run_dir,
        source_key=source_key,
        variables=variables,
        year=2010,
        bbox=_TEST_BBOX,
    )

    out_path = Path(result["consolidated_nc"])
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
    assert f"{source_key}_2010_consolidated.nc" in result["consolidated_nc"]


def test_consolidate_mod16a2_partial_tiles(mod16a2_run_dir: Path) -> None:
    """Consolidation succeeds when one tile is missing from a timestep."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2

    source_key = "mod16a2_v061"
    source_dir = mod16a2_run_dir

    # Delete the h09v04 tile from DOY 009
    doy9_h09 = list(source_dir.glob("MOD16A2GF.A2010009.h09v04.*"))
    assert len(doy9_h09) == 1, "Expected exactly one h09v04 tile for DOY 009"
    doy9_h09[0].unlink()

    result = consolidate_mod16a2(
        source_dir=mod16a2_run_dir,
        source_key=source_key,
        variables=["ET_500m"],
        year=2010,
        bbox=_TEST_BBOX,
    )

    # 3 remaining HDF files (2 for DOY 001, 1 for DOY 009)
    assert result["n_files"] == 3

    out_path = Path(result["consolidated_nc"])
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
        source_dir=mod16a2_run_dir,
        source_key=source_key,
        variables=variables,
        year=2010,
        bbox=_TEST_BBOX,
    )
    result2 = consolidate_mod16a2(
        source_dir=mod16a2_run_dir,
        source_key=source_key,
        variables=variables,
        year=2010,
        bbox=_TEST_BBOX,
    )

    assert result1["consolidated_nc"] == result2["consolidated_nc"]
    out_path = mod16a2_run_dir / result2["consolidated_nc"]
    assert out_path.exists()


def _make_fake_mosaic(tile_paths, variable, bbox, resolution=0.04):
    """Return a synthetic DataArray mimicking _mosaic_and_reproject_timestep."""
    lat = np.linspace(25.0, 50.0, 4)
    lon = np.linspace(-125.0, -65.0, 6)
    data = np.random.rand(1, len(lat), len(lon)).astype(np.float32)
    da = xr.DataArray(data, dims=["band", "y", "x"])
    return da


def test_consolidate_mod16a2_timestep_writes_temp(mod16a2_run_dir: Path) -> None:
    """consolidate_mod16a2_timestep writes a temp NetCDF and returns its path."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2_timestep

    source_dir = mod16a2_run_dir

    # Collect DOY 001 tiles
    tile_paths = sorted(source_dir.glob("MOD16A2GF.A2010001.*.hdf"))
    assert len(tile_paths) == 2  # h08v04 and h09v04

    with _patch(
        "nhf_spatial_targets.fetch.consolidate._mosaic_and_reproject_timestep",
        side_effect=_make_fake_mosaic,
    ):
        tmp_path = consolidate_mod16a2_timestep(
            tile_paths=tile_paths,
            variables=["ET_500m"],
            source_dir=source_dir,
            ydoy="2010001",
            bbox=_TEST_BBOX,
        )

    assert tmp_path.exists()
    assert tmp_path.name.startswith("_tmp_")
    assert "A2010001" in tmp_path.name
    assert tmp_path.suffix == ".nc"

    ds = xr.open_dataset(tmp_path)
    assert "time" in ds.dims
    assert len(ds.time) == 1
    assert "ET_500m" in ds.data_vars
    assert "lat" in ds.dims
    assert "lon" in ds.dims
    ds.close()

    # Clean up
    tmp_path.unlink()


def test_consolidate_mod16a2_finalize_concats_and_cleans(tmp_path: Path) -> None:
    """finalize lazy-concats temp files, writes consolidated, cleans up temps."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2_finalize

    source_dir = tmp_path / "mod16a2_v061"
    source_dir.mkdir(parents=True)

    lat = np.linspace(25.0, 50.0, 4)
    lon = np.linspace(-125.0, -65.0, 6)

    tmp_paths = []
    for doy in [1, 9]:
        ts = pd.Timestamp(year=2010, month=1, day=1) + pd.Timedelta(days=doy - 1)
        ds = xr.Dataset(
            {
                "ET_500m": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
            },
            coords={"time": [ts], "lat": lat, "lon": lon},
        )
        p = source_dir / f"_tmp_99999_A2010{doy:03d}.nc"
        ds.to_netcdf(p)
        tmp_paths.append(p)

    out_path = source_dir / "mod16a2_v061_2010_consolidated.nc"
    result = consolidate_mod16a2_finalize(
        tmp_paths=tmp_paths,
        variables=["ET_500m"],
        out_path=out_path,
    )

    # Final file exists
    assert out_path.exists()
    ds_out = xr.open_dataset(out_path)
    assert len(ds_out.time) == 2
    assert "ET_500m" in ds_out.data_vars
    assert pd.DatetimeIndex(ds_out.time.values).is_monotonic_increasing
    ds_out.close()

    # Temp files cleaned up
    for p in tmp_paths:
        assert not p.exists()

    # Provenance
    assert "consolidated_nc" in result
    assert result["n_files"] == 2


def test_consolidate_mod16a2_finalize_cleans_on_failure(tmp_path: Path) -> None:
    """Temp files are cleaned up even when the final write fails."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2_finalize

    source_dir = tmp_path / "mod16a2_v061"
    source_dir.mkdir(parents=True)

    # Create a temp file that cannot be opened as NetCDF
    bad_tmp = source_dir / "_tmp_99999_A2010001.nc"
    bad_tmp.write_bytes(b"not-netcdf")

    out_path = source_dir / "mod16a2_v061_2010_consolidated.nc"
    with pytest.raises(RuntimeError):
        consolidate_mod16a2_finalize(
            tmp_paths=[bad_tmp],
            variables=["ET_500m"],
            out_path=out_path,
        )

    # Temp file should be cleaned up
    assert not bad_tmp.exists()


def test_consolidate_mod16a2_finalize_empty_raises(tmp_path: Path) -> None:
    """ValueError raised when tmp_paths is empty."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2_finalize

    out_path = tmp_path / "consolidated.nc"
    with pytest.raises(ValueError, match="No temp files to finalize"):
        consolidate_mod16a2_finalize(
            tmp_paths=[],
            variables=["ET_500m"],
            out_path=out_path,
        )


def test_consolidate_mod16a2_bbox_clips_output(mod16a2_run_dir: Path) -> None:
    """Output grid is clipped to the bbox, not the full reprojected extent."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2

    source_key = "mod16a2_v061"

    # Use a tight bbox that is smaller than the tiles' reprojected footprint
    tight_bbox = (-112.0, 46.0, -108.0, 48.0)

    result = consolidate_mod16a2(
        source_dir=mod16a2_run_dir,
        source_key=source_key,
        variables=["ET_500m"],
        year=2010,
        bbox=tight_bbox,
    )

    out_path = Path(result["consolidated_nc"])
    ds = xr.open_dataset(out_path)

    # Verify the spatial extent is bounded by the tight bbox
    lons = ds.lon.values
    lats = ds.lat.values
    assert lons.min() >= tight_bbox[0] - 0.04  # allow one pixel tolerance
    assert lons.max() <= tight_bbox[2] + 0.04
    assert lats.min() >= tight_bbox[1] - 0.04
    assert lats.max() <= tight_bbox[3] + 0.04

    # Grid should be much smaller than a global grid
    assert ds.sizes["lon"] < 200
    assert ds.sizes["lat"] < 200
    ds.close()


def test_consolidate_mod16a2_cf_metadata(mod16a2_run_dir: Path) -> None:
    """MOD16A2 consolidated file has CF-1.6 metadata with crs (not spatial_ref)."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2

    source_key = "mod16a2_v061"
    result = consolidate_mod16a2(
        source_dir=mod16a2_run_dir,
        source_key=source_key,
        variables=["ET_500m"],
        year=2010,
        bbox=_TEST_BBOX,
    )

    out_path = Path(result["consolidated_nc"])
    ds = xr.open_dataset(out_path)
    assert ds.attrs["Conventions"] == "CF-1.6"
    assert "crs" in ds.data_vars
    assert "spatial_ref" not in ds.data_vars
    assert "spatial_ref" not in ds.coords
    assert ds["ET_500m"].attrs["grid_mapping"] == "crs"
    assert ds["ET_500m"].attrs["units"] == "kg m-2"
    assert ds["ET_500m"].attrs["long_name"] == "actual evapotranspiration"
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    ds.close()


def test_consolidate_mod10c1_cf_metadata(mod10c1_run_dir: Path) -> None:
    """MOD10C1 consolidated file has CF-1.6 metadata."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod10c1

    source_key = "mod10c1_v061"
    consolidate_mod10c1(
        source_dir=mod10c1_run_dir,
        source_key=source_key,
        variables=["Day_CMG_Snow_Cover", "Snow_Spatial_QA"],
        year=2010,
    )

    out_path = mod10c1_run_dir / f"{source_key}_2010_consolidated.nc"
    ds = xr.open_dataset(out_path)
    assert ds.attrs["Conventions"] == "CF-1.6"
    assert "crs" in ds.data_vars
    assert "spatial_ref" not in ds.data_vars
    assert ds["Day_CMG_Snow_Cover"].attrs["grid_mapping"] == "crs"
    assert ds["Day_CMG_Snow_Cover"].attrs["units"] == "percent"
    assert ds["Day_CMG_Snow_Cover"].attrs["long_name"] == "daily snow-covered area"
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert "time_bnds" not in ds.data_vars  # daily data, no time_bnds
    ds.close()


def test_log_memory_does_not_raise():
    """log_memory runs without error on any platform."""
    from nhf_spatial_targets.fetch.consolidate import log_memory

    # Should not raise regardless of platform
    log_memory("test checkpoint")


# ---------------------------------------------------------------------------
# MOD16A2 fill-value masking (regression for fill-contamination bug)
# ---------------------------------------------------------------------------


def test_mask_modis_et_fills_scaled_input():
    """Scaled tile (rioxarray decoded) — special codes 3276.x are masked."""
    from nhf_spatial_targets.fetch.consolidate import _mask_modis_et_fills

    da = xr.DataArray(
        np.array([0.0, 50.0, 3000.0, 3270.0, 3276.1, 3276.6, 3276.7], dtype=np.float32),
        dims=["x"],
    )
    out = _mask_modis_et_fills(da)

    expected_finite = np.array([0.0, 50.0, 3000.0, 3270.0])
    finite = out.values[np.isfinite(out.values)]
    np.testing.assert_array_equal(finite, expected_finite)
    # The 3 fill codes become NaN
    assert np.isnan(out.values[4])
    assert np.isnan(out.values[5])
    assert np.isnan(out.values[6])


def test_mask_modis_et_fills_raw_input():
    """Raw tile (rioxarray did not decode) — special codes 32761+ are masked."""
    from nhf_spatial_targets.fetch.consolidate import _mask_modis_et_fills

    da = xr.DataArray(
        np.array(
            [0.0, 500.0, 30000.0, 32700.0, 32761.0, 32766.0, 32767.0], dtype=np.float32
        ),
        dims=["x"],
    )
    out = _mask_modis_et_fills(da)

    expected_finite = np.array([0.0, 500.0, 30000.0, 32700.0])
    finite = out.values[np.isfinite(out.values)]
    np.testing.assert_array_equal(finite, expected_finite)
    assert np.isnan(out.values[4])
    assert np.isnan(out.values[5])
    assert np.isnan(out.values[6])


def test_mask_modis_et_fills_all_valid_scaled():
    """All-valid scaled tile is unchanged."""
    from nhf_spatial_targets.fetch.consolidate import _mask_modis_et_fills

    da = xr.DataArray(np.array([0.0, 50.0, 100.0, 200.0], dtype=np.float32), dims=["x"])
    out = _mask_modis_et_fills(da)

    np.testing.assert_array_equal(out.values, da.values)


def test_mask_modis_et_fills_all_nan():
    """All-NaN tile passes through without raising."""
    from nhf_spatial_targets.fetch.consolidate import _mask_modis_et_fills

    da = xr.DataArray(np.full(5, np.nan, dtype=np.float32), dims=["x"])
    out = _mask_modis_et_fills(da)

    assert np.all(np.isnan(out.values))


def test_mask_modis_et_fills_empty():
    """Empty tile passes through without raising."""
    from nhf_spatial_targets.fetch.consolidate import _mask_modis_et_fills

    da = xr.DataArray(np.empty((0,), dtype=np.float32), dims=["x"])
    out = _mask_modis_et_fills(da)

    assert out.size == 0


def _make_sinusoidal_tile_mixed(
    path: Path, h: int, v: int, valid_value: int = 100, fill_value: int = 32766
) -> None:
    """Write a sinusoidal-projected GeoTIFF with half valid, half fill pixels.

    The left half of the tile carries ``valid_value`` (raw) pixels and the
    right half carries ``fill_value`` pixels.  Drives the regression test
    for the consolidation reprojection: ``Resampling.average`` would otherwise
    produce contaminated 4 km cells whose averaged values fall in the
    physically impossible (100, 3270) scaled range at the valid/fill seam.
    """
    srs = CRS.from_proj4(
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
    )
    nx, ny = 48, 48
    x0 = -10_000_000 + h * 200_000
    y0 = 6_000_000 - v * 200_000
    res = 500.0
    transform = from_bounds(x0, y0 - ny * res, x0 + nx * res, y0, nx, ny)
    data = np.full((1, ny, nx), valid_value, dtype=np.int16)
    data[:, :, nx // 2 :] = fill_value
    da = xr.DataArray(data, dims=["band", "y", "x"], coords={"band": [1]})
    da.rio.write_crs(srs, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_nodata(-1, inplace=True)
    da.rio.to_raster(path, driver="GTiff")


def test_mosaic_and_reproject_timestep_no_fill_contamination(tmp_path: Path) -> None:
    """Reprojected tile has no contaminated cells in the (100, 3270) scaled band.

    Regression for the MOD16A2 v061 flat-on-CONUS+ bug. Before the fix,
    ``Resampling.average`` blended valid (raw 100) with fill-coded
    (raw 32766) pixels at the boundary, producing scaled values of
    ~1660 mm/8day that survived the post-aggregate ``<= 3270`` mask.
    After the fix, fill codes are masked to NaN before reprojection, so
    boundary cells are either real averages of valid pixels (≈ 10 scaled)
    or NaN — never a contaminated middle value.
    """
    from nhf_spatial_targets.fetch.consolidate import _mosaic_and_reproject_timestep

    source_dir = tmp_path / "mod16a2_v061"
    source_dir.mkdir(parents=True)

    # One mixed tile: left half = raw 100 (valid, scaled 10), right half = 32766 (fill)
    tile_path = source_dir / "MOD16A2GF.A2010001.h08v04.061.test.hdf"
    _make_sinusoidal_tile_mixed(tile_path, h=8, v=4, valid_value=100, fill_value=32766)

    out = _mosaic_and_reproject_timestep(
        tile_paths=[tile_path],
        variable="ET_500m",
        bbox=(-130.0, 20.0, -60.0, 55.0),
        resolution=0.04,
    )

    vals = out.values[np.isfinite(out.values)]
    # If raw values came through, max would be near 100 (raw → scaled 10);
    # if scaled, max would be near 10 (scaled). In either case the contaminated
    # band — present pre-fix as values 100..3270 in scaled units, or
    # 1000..32700 in raw — must be empty.
    contaminated_scaled = ((vals > 100.0) & (vals < 3270.0)).sum()
    contaminated_raw = ((vals > 1000.0) & (vals < 32700.0)).sum()
    assert contaminated_scaled == 0, (
        f"Found {contaminated_scaled} cells in the contaminated scaled band "
        f"(100, 3270) — fill values reached Resampling.average"
    )
    assert contaminated_raw == 0, (
        f"Found {contaminated_raw} cells in the contaminated raw band "
        f"(1000, 32700) — fill values reached Resampling.average"
    )
