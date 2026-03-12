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
    out = tmp_path / "data" / "raw" / "merra2"
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

    run_dir = merra2_dir.parent.parent.parent
    consolidate_merra2(
        run_dir=run_dir,
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

    run_dir = merra2_dir.parent.parent.parent
    consolidate_merra2(run_dir=run_dir, variables=["GWETTOP"])

    ds = xr.open_dataset(merra2_dir / "merra2_consolidated.nc")
    for t in pd.DatetimeIndex(ds.time.values):
        assert t.day == 15
        assert t.hour == 0
    ds.close()


def test_time_bounds(merra2_dir):
    """time_bnds spans first-of-month to first-of-next-month."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    run_dir = merra2_dir.parent.parent.parent
    consolidate_merra2(run_dir=run_dir, variables=["GWETTOP"])

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

    run_dir = merra2_dir.parent.parent.parent
    consolidate_merra2(run_dir=run_dir, variables=["GWETTOP"])

    ds = xr.open_dataset(merra2_dir / "merra2_consolidated.nc")
    assert ds.attrs["Conventions"] == "CF-1.8"
    assert "nhf-spatial-targets" in ds.attrs["history"]
    assert "M2TMNXLND" in ds.attrs["source"]
    assert "time_modification_note" in ds.attrs
    assert "references" in ds.attrs
    ds.close()


def test_no_nc4_files_raises(tmp_path):
    """FileNotFoundError raised when no .nc4 files exist."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    run_dir = tmp_path
    (run_dir / "data" / "raw" / "merra2").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No .nc4 files"):
        consolidate_merra2(run_dir=run_dir, variables=["GWETTOP"])


@pytest.fixture()
def merra2_dir_year_boundary(tmp_path: Path) -> Path:
    """Create synthetic MERRA-2 files spanning a year boundary (Nov-Dec-Jan)."""
    out = tmp_path / "data" / "raw" / "merra2"
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

    run_dir = merra2_dir_year_boundary.parent.parent.parent
    consolidate_merra2(run_dir=run_dir, variables=["GWETTOP"])

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

    run_dir = merra2_dir.parent.parent.parent
    result = consolidate_merra2(
        run_dir=run_dir,
        variables=["GWETTOP", "GWETROOT", "GWETPROF"],
    )

    assert result["consolidated_nc"] == "data/raw/merra2/merra2_consolidated.nc"
    assert "last_consolidated_utc" in result
    assert result["n_files"] == 3
    assert result["variables"] == ["GWETTOP", "GWETROOT", "GWETPROF"]


@pytest.fixture()
def nldas_dir(tmp_path: Path) -> Path:
    """Create synthetic NLDAS NetCDF4 files."""
    out = tmp_path / "data" / "raw" / "nldas_mosaic"
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

    run_dir = nldas_dir.parent.parent.parent
    consolidate_nldas(
        run_dir=run_dir,
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

    run_dir = nldas_dir.parent.parent.parent
    result = consolidate_nldas(
        run_dir=run_dir,
        source_key="nldas_mosaic",
        variables=["SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_200cm"],
    )
    assert (
        result["consolidated_nc"]
        == "data/raw/nldas_mosaic/nldas_mosaic_consolidated.nc"
    )
    assert "last_consolidated_utc" in result
    assert result["n_files"] == 3


def test_nldas_no_files_raises(tmp_path):
    from nhf_spatial_targets.fetch.consolidate import consolidate_nldas

    (tmp_path / "data" / "raw" / "nldas_mosaic").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        consolidate_nldas(
            run_dir=tmp_path,
            source_key="nldas_mosaic",
            variables=["SoilM_0_10cm"],
        )


@pytest.fixture()
def ncep_dir(tmp_path: Path) -> Path:
    """Create synthetic NCEP/NCAR monthly NetCDF3 files."""
    out = tmp_path / "data" / "raw" / "ncep_ncar"
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

    run_dir = ncep_dir.parent.parent.parent
    consolidate_ncep_ncar(run_dir=run_dir, variables=["soilw"])

    nc_path = ncep_dir / "ncep_ncar_consolidated.nc"
    assert nc_path.exists()

    ds = xr.open_dataset(nc_path)
    assert "soilw" in ds.data_vars
    assert "EXTRA" not in ds.data_vars
    ds.close()


def test_ncep_provenance_return(ncep_dir):
    from nhf_spatial_targets.fetch.consolidate import consolidate_ncep_ncar

    run_dir = ncep_dir.parent.parent.parent
    result = consolidate_ncep_ncar(run_dir=run_dir, variables=["soilw"])
    assert result["consolidated_nc"] == "data/raw/ncep_ncar/ncep_ncar_consolidated.nc"
    assert result["n_files"] == 3


def test_ncep_no_files_raises(tmp_path):
    from nhf_spatial_targets.fetch.consolidate import consolidate_ncep_ncar

    (tmp_path / "data" / "raw" / "ncep_ncar").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        consolidate_ncep_ncar(run_dir=tmp_path, variables=["soilw"])


def test_open_consolidated(merra2_dir):
    """open_consolidated returns a readable xr.Dataset."""
    from nhf_spatial_targets.fetch.consolidate import (
        consolidate_merra2,
        open_consolidated,
    )

    run_dir = merra2_dir.parent.parent.parent
    consolidate_merra2(run_dir=run_dir, variables=["GWETTOP"])

    nc_path = merra2_dir / "merra2_consolidated.nc"
    ds = open_consolidated(nc_path)
    assert "GWETTOP" in ds.data_vars
    assert len(ds.time) == 3
    ds.close()
