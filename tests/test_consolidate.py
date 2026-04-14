"""Tests for NetCDF consolidation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture()
def merra2_dir(tmp_path: Path) -> Path:
    """Create a directory with small synthetic MERRA-2 NetCDF files."""
    out = tmp_path / "merra2"
    out.mkdir(parents=True)

    lat = np.arange(-90, 91, 45.0)
    lon = np.arange(-180, 180, 60.0)

    for month in range(1, 4):  # 3 months
        time = np.array(
            [f"2010-{month:02d}-01T00:30:00"],
            dtype="datetime64[ns]",
        )
        ds = xr.Dataset(
            {
                "GWETTOP": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
                "GWETROOT": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
                "GWETPROF": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
                "SFMC": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
                "BASEFLOW": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )
        fname = f"MERRA2_300.tavgM_2d_lnd_Nx.2010{month:02d}.nc4"
        ds.to_netcdf(out / fname)

    return out


def test_filter_variables(merra2_dir):
    """Consolidated file contains only requested variables plus coordinates."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    consolidate_merra2(
        source_dir=merra2_dir,
        variables=["GWETTOP", "GWETROOT", "GWETPROF"],
    )

    nc_path = merra2_dir / "merra2_consolidated.nc"
    assert nc_path.exists()

    ds = xr.open_dataset(nc_path)
    assert "GWETTOP" in ds.data_vars
    assert "GWETROOT" in ds.data_vars
    assert "GWETPROF" in ds.data_vars
    assert "SFMC" not in ds.data_vars
    assert "BASEFLOW" not in ds.data_vars
    assert len(ds.time) == 3
    ds.close()


def test_time_midmonth(merra2_dir):
    """Timestamps are shifted to the 15th of each month."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    consolidate_merra2(source_dir=merra2_dir, variables=["GWETTOP"])

    ds = xr.open_dataset(merra2_dir / "merra2_consolidated.nc")
    for t in pd.DatetimeIndex(ds.time.values):
        assert t.day == 15
        assert t.hour == 0
    ds.close()


def test_time_bounds(merra2_dir):
    """time_bnds spans first-of-month to first-of-next-month."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    consolidate_merra2(source_dir=merra2_dir, variables=["GWETTOP"])

    ds = xr.open_dataset(merra2_dir / "merra2_consolidated.nc")
    assert "time_bnds" in ds.data_vars
    assert ds.time.attrs.get("bounds") == "time_bnds"
    assert ds.time.attrs.get("cell_methods") == "time: mean"
    # First month: Jan 2010
    bnds = pd.DatetimeIndex(ds.time_bnds.values[0])
    assert bnds[0] == pd.Timestamp("2010-01-01")
    assert bnds[1] == pd.Timestamp("2010-02-01")
    ds.close()


def test_global_attributes(merra2_dir):
    """Consolidated file has CF and provenance global attributes."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    consolidate_merra2(source_dir=merra2_dir, variables=["GWETTOP"])

    ds = xr.open_dataset(merra2_dir / "merra2_consolidated.nc")
    # CF-1.6 compliance
    assert ds.attrs["Conventions"] == "CF-1.6"
    assert "crs" in ds.data_vars
    assert "spatial_ref" not in ds.data_vars
    assert ds["crs"].attrs["grid_mapping_name"] == "latitude_longitude"
    assert ds["GWETTOP"].attrs["grid_mapping"] == "crs"
    assert ds["GWETTOP"].attrs["units"] == "1"
    assert ds["GWETTOP"].attrs["long_name"] == "surface_soil_wetness"
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert ds.time.attrs["standard_name"] == "time"
    # Provenance attrs preserved
    assert "nhf-spatial-targets" in ds.attrs["history"]
    assert "M2TMNXLND" in ds.attrs["source"]
    assert "time_modification_note" in ds.attrs
    assert "references" in ds.attrs
    ds.close()


def test_no_nc4_files_raises(tmp_path):
    """FileNotFoundError raised when no .nc4 files exist."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    source_dir = tmp_path / "merra2"
    source_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No .nc4 files"):
        consolidate_merra2(source_dir=source_dir, variables=["GWETTOP"])


@pytest.fixture()
def merra2_dir_year_boundary(tmp_path: Path) -> Path:
    """Create synthetic MERRA-2 files spanning a year boundary (Nov-Dec-Jan)."""
    out = tmp_path / "merra2"
    out.mkdir(parents=True)

    lat = np.arange(-90, 91, 45.0)
    lon = np.arange(-180, 180, 60.0)

    months = [(2010, 11), (2010, 12), (2011, 1)]
    for year, month in months:
        time = np.array(
            [f"{year}-{month:02d}-01T00:30:00"],
            dtype="datetime64[ns]",
        )
        ds = xr.Dataset(
            {
                "GWETTOP": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )
        fname = f"MERRA2_300.tavgM_2d_lnd_Nx.{year}{month:02d}.nc4"
        ds.to_netcdf(out / fname)

    return out


def test_time_bounds_december_year_boundary(merra2_dir_year_boundary):
    """time_bnds correctly crosses year boundary for December."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    consolidate_merra2(source_dir=merra2_dir_year_boundary, variables=["GWETTOP"])

    ds = xr.open_dataset(merra2_dir_year_boundary / "merra2_consolidated.nc")
    # December 2010: bounds should be [2010-12-01, 2011-01-01]
    dec_idx = 1  # Nov=0, Dec=1, Jan=2
    bnds = pd.DatetimeIndex(ds.time_bnds.values[dec_idx])
    assert bnds[0] == pd.Timestamp("2010-12-01")
    assert bnds[1] == pd.Timestamp("2011-01-01")
    ds.close()


def test_provenance_return(merra2_dir):
    """consolidate_merra2 returns a dict with provenance keys."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    result = consolidate_merra2(
        source_dir=merra2_dir,
        variables=["GWETTOP", "GWETROOT", "GWETPROF"],
    )

    assert "merra2_consolidated.nc" in result["consolidated_nc"]
    assert "last_consolidated_utc" in result
    assert result["n_files"] == 3
    assert result["variables"] == ["GWETTOP", "GWETROOT", "GWETPROF"]


@pytest.fixture()
def nldas_dir(tmp_path: Path) -> Path:
    """Create synthetic NLDAS NetCDF4 files."""
    out = tmp_path / "nldas_mosaic"
    out.mkdir(parents=True)

    lat = np.arange(25.0, 50.0, 5.0)
    lon = np.arange(-125.0, -65.0, 10.0)

    for month in range(1, 4):
        time = np.array([f"2010-{month:02d}-15T00:00:00"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {
                "SoilM_0_10cm": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
                "SoilM_10_40cm": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
                "SoilM_40_200cm": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
                "EXTRA_VAR": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )
        fname = f"NLDAS_MOS0125_M.A2010{month:02d}.002.grb.SUB.nc4"
        ds.to_netcdf(out / fname)

    return out


def test_nldas_filter_variables(nldas_dir):
    from nhf_spatial_targets.fetch.consolidate import consolidate_nldas

    consolidate_nldas(
        source_dir=nldas_dir,
        source_key="nldas_mosaic",
        variables=["SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_200cm"],
    )

    nc_path = nldas_dir / "nldas_mosaic_consolidated.nc"
    assert nc_path.exists()

    ds = xr.open_dataset(nc_path)
    assert "SoilM_0_10cm" in ds.data_vars
    assert "SoilM_10_40cm" in ds.data_vars
    assert "SoilM_40_200cm" in ds.data_vars
    assert "EXTRA_VAR" not in ds.data_vars
    assert len(ds.time) == 3
    ds.close()


def test_nldas_provenance_return(nldas_dir):
    from nhf_spatial_targets.fetch.consolidate import consolidate_nldas

    result = consolidate_nldas(
        source_dir=nldas_dir,
        source_key="nldas_mosaic",
        variables=["SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_200cm"],
    )
    assert "nldas_mosaic_consolidated.nc" in result["consolidated_nc"]
    assert "last_consolidated_utc" in result
    assert result["n_files"] == 3


def test_nldas_no_files_raises(tmp_path):
    from nhf_spatial_targets.fetch.consolidate import consolidate_nldas

    source_dir = tmp_path / "nldas_mosaic"
    source_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        consolidate_nldas(
            source_dir=source_dir,
            source_key="nldas_mosaic",
            variables=["SoilM_0_10cm"],
        )


def test_nldas_cf_metadata(nldas_dir):
    """NLDAS consolidated file has CF-1.6 metadata."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_nldas

    consolidate_nldas(
        source_dir=nldas_dir,
        source_key="nldas_mosaic",
        variables=["SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_200cm"],
    )

    ds = xr.open_dataset(nldas_dir / "nldas_mosaic_consolidated.nc")
    assert ds.attrs["Conventions"] == "CF-1.6"
    assert "crs" in ds.data_vars
    assert "spatial_ref" not in ds.data_vars
    assert ds["crs"].attrs["grid_mapping_name"] == "latitude_longitude"
    assert ds["SoilM_0_10cm"].attrs["grid_mapping"] == "crs"
    assert ds["SoilM_0_10cm"].attrs["units"] == "kg/m2"
    assert ds["SoilM_0_10cm"].attrs["long_name"] == "soil moisture 0-10 cm"
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert ds.time.attrs["standard_name"] == "time"
    assert "time_bnds" in ds.data_vars
    assert ds.time.attrs.get("bounds") == "time_bnds"
    ds.close()


@pytest.fixture()
def ncep_dir(tmp_path: Path) -> Path:
    """Create synthetic NCEP/NCAR monthly NetCDF3 files."""
    out = tmp_path / "ncep_ncar"
    out.mkdir(parents=True)

    lat = np.arange(-90, 91, 45.0)
    lon = np.arange(-180, 180, 60.0)

    for month in range(1, 4):
        time = np.array([f"2010-{month:02d}-15T00:00:00"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {
                "soilw": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
                "EXTRA": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )
        fname = f"soilw.0-10cm.gauss.2010-{month:02d}.monthly.nc"
        ds.to_netcdf(out / fname, format="NETCDF3_CLASSIC")

    return out


def test_ncep_filter_variables(ncep_dir):
    from nhf_spatial_targets.fetch.consolidate import consolidate_ncep_ncar

    consolidate_ncep_ncar(source_dir=ncep_dir, variables=["soilw"])

    nc_path = ncep_dir / "ncep_ncar_consolidated.nc"
    assert nc_path.exists()

    ds = xr.open_dataset(nc_path)
    assert "soilw" in ds.data_vars
    assert "EXTRA" not in ds.data_vars
    ds.close()


def test_ncep_provenance_return(ncep_dir):
    from nhf_spatial_targets.fetch.consolidate import consolidate_ncep_ncar

    result = consolidate_ncep_ncar(source_dir=ncep_dir, variables=["soilw"])
    assert "ncep_ncar_consolidated.nc" in result["consolidated_nc"]
    assert result["n_files"] == 3


def test_ncep_no_files_raises(tmp_path):
    from nhf_spatial_targets.fetch.consolidate import consolidate_ncep_ncar

    source_dir = tmp_path / "ncep_ncar"
    source_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        consolidate_ncep_ncar(source_dir=source_dir, variables=["soilw"])


def test_ncep_cf_metadata(ncep_dir):
    """NCEP/NCAR consolidated file has CF-1.6 metadata."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_ncep_ncar

    consolidate_ncep_ncar(source_dir=ncep_dir, variables=["soilw"])

    ds = xr.open_dataset(ncep_dir / "ncep_ncar_consolidated.nc")
    assert ds.attrs["Conventions"] == "CF-1.6"
    assert "crs" in ds.data_vars
    assert "spatial_ref" not in ds.data_vars
    assert ds["soilw"].attrs["grid_mapping"] == "crs"
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert "time_bnds" in ds.data_vars
    ds.close()


@pytest.fixture()
def merra2_dir_unsorted(tmp_path: Path) -> Path:
    """Create MERRA-2 files in reverse chronological order."""
    out = tmp_path / "merra2"
    out.mkdir(parents=True)

    lat = np.arange(-90, 91, 45.0)
    lon = np.arange(-180, 180, 60.0)

    # Write March first, then Jan — reversed order
    for month in [3, 1, 2]:
        time = np.array(
            [f"2010-{month:02d}-01T00:30:00"],
            dtype="datetime64[ns]",
        )
        ds = xr.Dataset(
            {
                "GWETTOP": (
                    ["time", "lat", "lon"],
                    np.full((1, len(lat), len(lon)), month, dtype=np.float32),
                ),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )
        fname = f"MERRA2_300.tavgM_2d_lnd_Nx.2010{month:02d}.nc4"
        ds.to_netcdf(out / fname)

    return out


def test_time_sorting(merra2_dir_unsorted):
    """Consolidated output has monotonically increasing time regardless of input order."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    consolidate_merra2(source_dir=merra2_dir_unsorted, variables=["GWETTOP"])

    ds = xr.open_dataset(merra2_dir_unsorted / "merra2_consolidated.nc")
    times = pd.DatetimeIndex(ds.time.values)
    assert times.is_monotonic_increasing
    # Verify data follows the time order (month value encoded in data)
    assert float(ds["GWETTOP"].isel(time=0).mean()) == pytest.approx(1.0)
    assert float(ds["GWETTOP"].isel(time=1).mean()) == pytest.approx(2.0)
    assert float(ds["GWETTOP"].isel(time=2).mean()) == pytest.approx(3.0)
    ds.close()


@pytest.fixture()
def ncep_multi_var_dir(tmp_path: Path) -> Path:
    """Create NCEP/NCAR files with different variables in separate files."""
    out = tmp_path / "ncep_ncar"
    out.mkdir(parents=True)

    lat = np.arange(-90, 91, 45.0)
    lon = np.arange(-180, 180, 60.0)

    # Variable group 1: soilw_0_10cm (2 years)
    for year in [2010, 2011]:
        for month in range(1, 4):
            time = np.array([f"{year}-{month:02d}-15T00:00:00"], dtype="datetime64[ns]")
            ds = xr.Dataset(
                {
                    "soilw_0_10cm": (
                        ["time", "lat", "lon"],
                        np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                    ),
                },
                coords={"time": time, "lat": lat, "lon": lon},
            )
            fname = f"soilw.0-10cm.gauss.{year}-{month:02d}.monthly.nc"
            ds.to_netcdf(out / fname, format="NETCDF3_CLASSIC")

    # Variable group 2: soilw_10_200cm (same 2 years)
    for year in [2010, 2011]:
        for month in range(1, 4):
            time = np.array([f"{year}-{month:02d}-15T00:00:00"], dtype="datetime64[ns]")
            ds = xr.Dataset(
                {
                    "soilw_10_200cm": (
                        ["time", "lat", "lon"],
                        np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                    ),
                },
                coords={"time": time, "lat": lat, "lon": lon},
            )
            fname = f"soilw.10-200cm.gauss.{year}-{month:02d}.monthly.nc"
            ds.to_netcdf(out / fname, format="NETCDF3_CLASSIC")

    return out


def test_ncep_multi_variable_merge(ncep_multi_var_dir):
    """Files with different variables are grouped, concatenated, then merged."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_ncep_ncar

    result = consolidate_ncep_ncar(
        source_dir=ncep_multi_var_dir, variables=["soilw_0_10cm", "soilw_10_200cm"]
    )

    nc_path = ncep_multi_var_dir / "ncep_ncar_consolidated.nc"
    assert nc_path.exists()

    ds = xr.open_dataset(nc_path)
    assert "soilw_0_10cm" in ds.data_vars
    assert "soilw_10_200cm" in ds.data_vars
    # 2 years x 3 months = 6 timesteps (no duplicates)
    assert len(ds.time) == 6
    assert pd.DatetimeIndex(ds.time.values).is_monotonic_increasing
    ds.close()

    assert result["n_files"] == 12  # 6 per variable group


@pytest.fixture()
def nldas_noah_dir(tmp_path: Path) -> Path:
    """Create synthetic NLDAS NOAH NetCDF4 files with 4 soil layers."""
    out = tmp_path / "nldas_noah"
    out.mkdir(parents=True)

    lat = np.arange(25.0, 50.0, 5.0)
    lon = np.arange(-125.0, -65.0, 10.0)

    for month in range(1, 4):
        time = np.array([f"2010-{month:02d}-15T00:00:00"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {
                "SoilM_0_10cm": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
                "SoilM_10_40cm": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
                "SoilM_40_100cm": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
                "SoilM_100_200cm": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
                "EXTRA_VAR": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )
        fname = f"NLDAS_NOAH0125_M.A2010{month:02d}.002.grb.SUB.nc4"
        ds.to_netcdf(out / fname)

    return out


def test_nldas_noah_filter_variables(nldas_noah_dir):
    """NOAH consolidation with 4 soil layers filters correctly."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_nldas

    noah_vars = ["SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_100cm", "SoilM_100_200cm"]
    consolidate_nldas(
        source_dir=nldas_noah_dir,
        source_key="nldas_noah",
        variables=noah_vars,
    )

    nc_path = nldas_noah_dir / "nldas_noah_consolidated.nc"
    assert nc_path.exists()

    ds = xr.open_dataset(nc_path)
    for var in noah_vars:
        assert var in ds.data_vars
    assert "EXTRA_VAR" not in ds.data_vars
    assert len(ds.time) == 3
    ds.close()


def test_missing_variable_raises(merra2_dir):
    """ValueError raised when a requested variable does not exist in the data."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    with pytest.raises(ValueError, match="not found"):
        consolidate_merra2(
            source_dir=merra2_dir,
            variables=["GWETTOP", "NONEXISTENT_VAR"],
        )


def test_open_consolidated(merra2_dir):
    """open_consolidated returns a readable xr.Dataset."""
    from nhf_spatial_targets.fetch.consolidate import (
        consolidate_merra2,
        open_consolidated,
    )

    consolidate_merra2(source_dir=merra2_dir, variables=["GWETTOP"])

    nc_path = merra2_dir / "merra2_consolidated.nc"
    ds = open_consolidated(nc_path)
    assert "GWETTOP" in ds.data_vars
    assert len(ds.time) == 3
    ds.close()


def test_apply_cf_metadata_monthly():
    """apply_cf_metadata adds all CF-1.6 metadata for monthly data."""
    from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

    # Build a minimal dataset with y/x coords, spatial_ref, no CF metadata
    lat = np.arange(25.0, 50.0, 5.0)
    lon = np.arange(-125.0, -65.0, 10.0)
    time = pd.date_range("2010-01-15", periods=3, freq="MS")
    ds = xr.Dataset(
        {
            "SoilM_0_10cm": (
                ["time", "y", "x"],
                np.random.rand(3, len(lat), len(lon)).astype(np.float32),
            ),
            "spatial_ref": xr.DataArray(np.int32(0)),
        },
        coords={"time": time, "y": lat, "x": lon},
    )

    result = apply_cf_metadata(ds, "nldas_mosaic", "monthly")

    # Coordinates renamed to lat/lon
    assert "lat" in result.dims
    assert "lon" in result.dims
    assert "y" not in result.dims
    assert "x" not in result.dims

    # Dimension order
    assert result["SoilM_0_10cm"].dims == ("time", "lat", "lon")

    # CRS variable
    assert "crs" in result.data_vars
    assert result["crs"].attrs["grid_mapping_name"] == "latitude_longitude"
    assert result["crs"].attrs["semi_major_axis"] == pytest.approx(6378137.0)
    assert result["crs"].attrs["inverse_flattening"] == pytest.approx(298.257223563)
    assert "crs_wkt" in result["crs"].attrs

    # No spatial_ref
    assert "spatial_ref" not in result.data_vars
    assert "spatial_ref" not in result.coords

    # grid_mapping on data vars
    assert result["SoilM_0_10cm"].attrs["grid_mapping"] == "crs"

    # Variable metadata from catalog
    assert result["SoilM_0_10cm"].attrs["units"] == "kg/m2"
    assert result["SoilM_0_10cm"].attrs["long_name"] == "soil moisture 0-10 cm"

    # Coordinate attrs
    assert result.lat.attrs["standard_name"] == "latitude"
    assert result.lat.attrs["units"] == "degrees_north"
    assert result.lat.attrs["axis"] == "Y"
    assert result.lon.attrs["standard_name"] == "longitude"
    assert result.lon.attrs["units"] == "degrees_east"
    assert result.lon.attrs["axis"] == "X"
    assert result.time.attrs["standard_name"] == "time"
    assert result.time.attrs["axis"] == "T"

    # time_bnds for monthly
    assert "time_bnds" in result.data_vars
    assert result.time.attrs.get("bounds") == "time_bnds"

    # Conventions
    assert result.attrs["Conventions"] == "CF-1.6"


def test_apply_cf_metadata_daily_no_time_bnds():
    """apply_cf_metadata does not add time_bnds for daily data."""
    from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

    lat = np.linspace(25.0, 50.0, 4)
    lon = np.linspace(-125.0, -65.0, 6)
    time = pd.date_range("2010-01-01", periods=3, freq="D")
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (
                ["time", "lat", "lon"],
                np.random.rand(3, len(lat), len(lon)).astype(np.float32),
            ),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )

    result = apply_cf_metadata(ds, "mod10c1_v061", "daily")

    assert "time_bnds" not in result.data_vars
    assert "crs" in result.data_vars
    assert result.attrs["Conventions"] == "CF-1.6"


def test_apply_cf_metadata_latitude_longitude_rename():
    """apply_cf_metadata renames latitude/longitude to lat/lon."""
    from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

    ds = xr.Dataset(
        {
            "var": (
                ["time", "latitude", "longitude"],
                np.random.rand(2, 3, 4).astype(np.float32),
            ),
        },
        coords={
            "time": pd.date_range("2010-01-01", periods=2, freq="MS"),
            "latitude": np.linspace(25.0, 50.0, 3),
            "longitude": np.linspace(-125.0, -65.0, 4),
        },
    )

    result = apply_cf_metadata(ds, "ncep_ncar", "monthly")

    assert "lat" in result.dims
    assert "lon" in result.dims
    assert "latitude" not in result.dims
    assert "longitude" not in result.dims


def test_apply_cf_metadata_skips_existing_time_bnds():
    """apply_cf_metadata skips time_bnds if already present (MERRA-2 case)."""
    from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

    lat = np.arange(-90, 91, 45.0)
    lon = np.arange(-180, 180, 60.0)
    time = pd.date_range("2010-01-15", periods=2, freq="MS")
    ds = xr.Dataset(
        {
            "GWETTOP": (
                ["time", "lat", "lon"],
                np.random.rand(2, len(lat), len(lon)).astype(np.float32),
            ),
            "time_bnds": (
                ["time", "nv"],
                np.array([[0, 31], [31, 59]], dtype="<i8"),
                {"units": "days since 1970-01-01", "calendar": "standard"},
            ),
        },
        coords={"time": time, "lat": lat, "lon": lon, "nv": [0, 1]},
    )

    result = apply_cf_metadata(ds, "merra2", "monthly")

    # Should keep existing time_bnds, not add a second one
    assert "time_bnds" in result.data_vars
    # Original values preserved
    np.testing.assert_array_equal(
        result["time_bnds"].values, np.array([[0, 31], [31, 59]])
    )


def test_apply_cf_metadata_custom_crs_wkt():
    """apply_cf_metadata uses pyproj to extract ellipsoid from custom CRS WKT."""
    from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

    lat = np.linspace(25.0, 50.0, 4)
    lon = np.linspace(-125.0, -65.0, 6)
    time = pd.date_range("2005-07-01", periods=2, freq="YS")
    ds = xr.Dataset(
        {
            "total_recharge": (
                ["time", "y", "x"],
                np.random.rand(2, len(lat), len(lon)).astype(np.float32),
            ),
        },
        coords={"time": time, "y": lat, "x": lon},
    )

    # NAD83 WKT
    from pyproj import CRS as _CRS

    nad83_wkt = _CRS.from_epsg(4269).to_wkt()

    result = apply_cf_metadata(ds, "reitz2017", "annual", crs_wkt=nad83_wkt)

    assert result["crs"].attrs["grid_mapping_name"] == "latitude_longitude"
    # NAD83 uses GRS 1980 ellipsoid
    assert result["crs"].attrs["inverse_flattening"] == pytest.approx(298.257222101)
    assert "NAD" in result["crs"].attrs["crs_wkt"]
    assert "time_bnds" not in result.data_vars
    # y/x renamed to lat/lon
    assert "lat" in result.dims
    assert "lon" in result.dims


# ---------------------------------------------------------------------------
# _write_netcdf — atomic write tests
# ---------------------------------------------------------------------------


def test_write_netcdf_atomic_success(tmp_path):
    """Successful write produces final file with no temp files left."""
    from nhf_spatial_targets.fetch.consolidate import _write_netcdf

    ds = xr.Dataset({"x": (["t"], [1.0, 2.0])})
    out = tmp_path / "output.nc"
    _write_netcdf(ds, out)

    assert out.exists()
    # No leftover temp files
    assert not list(tmp_path.glob("*.nc.tmp"))
    # Verify contents
    result = xr.open_dataset(out)
    assert "x" in result.data_vars
    result.close()


def test_write_netcdf_atomic_failure_no_partial(tmp_path):
    """Failed write leaves no file at the output path."""
    from unittest.mock import patch

    from nhf_spatial_targets.fetch.consolidate import _write_netcdf

    ds = xr.Dataset({"x": (["t"], [1.0, 2.0])})
    out = tmp_path / "output.nc"

    with patch("xarray.Dataset.to_netcdf", side_effect=OSError("disk full")):
        with pytest.raises(RuntimeError, match="disk full"):
            _write_netcdf(ds, out)

    assert not out.exists()
    assert not list(tmp_path.glob("*.nc.tmp"))


def test_apply_cf_metadata_raises_when_time_missing():
    """apply_cf_metadata for a time-stepped product must have a time dim."""
    import numpy as np
    import xarray as xr
    from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

    # Dataset with lat/lon but no time (or valid_time) dim.
    ds = xr.Dataset(
        {"foo": (("lat", "lon"), np.zeros((2, 2)))},
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    with pytest.raises(ValueError, match="no 'time' or 'valid_time' dim"):
        apply_cf_metadata(ds, "era5_land", "daily")


def test_time_bnds_roundtrip_no_warning(tmp_path):
    """Monthly datasets written by apply_cf_metadata decode time_bnds correctly.

    Guards the CF-1.6 §7.1 inheritance contract:
    - time.encoding carries the units/calendar reference
    - time_bnds is written as raw int64 days-since-epoch (no duplicate attrs)
    - on read-back, xarray correctly decodes time_bnds to datetime64 via
      the parent time coord
    - no SerializationWarning is emitted during write
    """
    import warnings

    import numpy as np
    import pandas as pd
    import xarray as xr

    from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

    times = pd.date_range("2020-01-01", periods=3, freq="1MS")
    ds = xr.Dataset(
        {"foo": (("time", "lat", "lon"), np.zeros((3, 2, 2)))},
        coords={"time": times, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    ds = apply_cf_metadata(ds, "era5_land", "monthly")

    out = tmp_path / "roundtrip.nc"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds.to_netcdf(out)

    # No time-related SerializationWarning — any such warning indicates the
    # units pinning has regressed.
    time_warnings = [
        w
        for w in caught
        if "time" in str(w.message).lower() and "bounds" in str(w.message).lower()
    ]
    assert not time_warnings, f"Unexpected time/bounds warnings: {time_warnings}"

    # Read back with full CF decoding; time_bnds should be datetime64.
    decoded = xr.open_dataset(out)
    try:
        assert decoded.time_bnds.dtype == np.dtype("datetime64[ns]")
        assert decoded.time.attrs["bounds"] == "time_bnds"
        # Bounds must bracket their time value.
        t0 = decoded.time.values[0]
        lo, hi = decoded.time_bnds.values[0]
        assert lo <= t0 < hi
    finally:
        decoded.close()

    # Raw on-disk: time has explicit units; time_bnds has raw ints with no
    # units attr (xarray strips it per CF inheritance). Pin this behavior.
    raw = xr.open_dataset(out, decode_times=False)
    try:
        assert raw.time.attrs.get("units") == "days since 1970-01-01"
        assert raw.time.attrs.get("calendar") == "standard"
        assert raw.time_bnds.dtype.kind == "i"
        # Bounds variable itself has no units attr per CF-1.6 §7.1 inheritance.
        assert "units" not in raw.time_bnds.attrs
    finally:
        raw.close()


def test_resolve_license_returns_catalog_value_when_present():
    """When the catalog provides a license, it's returned verbatim."""
    from nhf_spatial_targets.fetch.consolidate import resolve_license

    meta = {"license": "public domain (NASA)"}
    assert resolve_license(meta, "some_source") == "public domain (NASA)"


def test_resolve_license_falls_back_with_warning_when_missing(caplog):
    """Missing license → UNKNOWN sentinel + WARNING log mentioning the source."""
    import logging

    from nhf_spatial_targets.fetch.consolidate import resolve_license

    with caplog.at_level(logging.WARNING):
        result = resolve_license({}, "test_source")
    assert result == "UNKNOWN — see catalog/sources.yml"
    assert any(
        "test_source" in rec.message and "license" in rec.message
        for rec in caplog.records
    )


def test_resolve_license_falls_back_when_empty_string(caplog):
    """An explicit empty string triggers the fallback too."""
    import logging

    from nhf_spatial_targets.fetch.consolidate import resolve_license

    with caplog.at_level(logging.WARNING):
        result = resolve_license({"license": ""}, "test_source")
    assert result == "UNKNOWN — see catalog/sources.yml"
    assert len(caplog.records) >= 1
