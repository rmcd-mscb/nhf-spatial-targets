"""Tests for the runoff target builder end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml


def _write_year_nc(
    path: Path, year: int, var: str, value: float, id_col: str = "nhm_id"
):
    """Write a per-year aggregated NC with the given constant value."""
    times = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    hrus = [1, 2, 3]
    arr = np.full((len(times), len(hrus)), value, dtype=np.float32)
    ds = xr.Dataset(
        {var: (("time", id_col), arr)},
        coords={"time": times, id_col: hrus},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _write_synthetic_fabric(path: Path, id_col: str = "nhm_id"):
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {id_col: [1, 2, 3]},
        geometry=[
            box(-105.0, 40.0, -104.9, 40.1),
            box(-104.9, 40.0, -104.8, 40.1),
            box(-104.8, 40.0, -104.7, 40.1),
        ],
        crs="EPSG:4326",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")


def _make_runoff_project(
    tmp_path: Path,
    period: str = "2000-01-01/2001-12-31",
    nn_fill: bool = True,
    sources_per_year: dict | None = None,
) -> Path:
    """Build a project skeleton with synthetic fabric + per-year aggregated NCs.

    sources_per_year: dict[source_key -> dict[year -> (var, value)]]
    Default: era5_land ro=0.05 m/month for 2000-2001;
             gldas_noah_v21_monthly runoff_total=2.0 kg/m2 for 2000-2001;
             mwbm_climgrid runoff=30.0 mm/month for 2000 only (partial period).
    """
    workdir = tmp_path / "proj"
    workdir.mkdir()
    fabric_path = tmp_path / "fabric.gpkg"
    _write_synthetic_fabric(fabric_path)
    (workdir / "fabric.json").write_text(json.dumps({"id_col": "nhm_id"}))

    cfg = {
        "datastore": str(tmp_path / "store"),
        "fabric": {"path": str(fabric_path), "id_col": "nhm_id"},
        "targets": {
            "runoff": {
                "period": period,
                "nn_fill": nn_fill,
            }
        },
    }
    (workdir / "config.yml").write_text(yaml.safe_dump(cfg))

    plan = sources_per_year or {
        "era5_land": {2000: ("ro", 0.05), 2001: ("ro", 0.05)},
        "gldas_noah_v21_monthly": {
            2000: ("runoff_total", 2.0),
            2001: ("runoff_total", 2.0),
        },
        "mwbm_climgrid": {2000: ("runoff", 30.0)},
    }
    agg_dir = workdir / "data" / "aggregated"
    for src, year_map in plan.items():
        for year, (var, value) in year_map.items():
            _write_year_nc(agg_dir / src / f"{src}_{year}_agg.nc", year, var, value)
    return workdir


def test_build_writes_unfilled_and_filled_files(tmp_path: Path):
    from nhf_spatial_targets.targets.run import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_runoff_project(tmp_path)
    project = load(workdir)
    build(project)
    assert (project.targets_dir() / "runoff_targets.nc").exists()
    assert (project.targets_dir() / "runoff_targets_nn_filled.nc").exists()


def test_build_period_union_n_sources_diagnostic(tmp_path: Path):
    """MWBM only covers 2000 -> 2001 cells should have n_sources=2."""
    from nhf_spatial_targets.targets.run import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_runoff_project(tmp_path, nn_fill=False)
    project = load(workdir)
    build(project)
    out = project.targets_dir() / "runoff_targets.nc"
    with xr.open_dataset(out, decode_cf=True) as ds:
        n = ds["n_sources"].values
        # 24 months total: months 0..11 = 2000 (3 sources), 12..23 = 2001 (2 sources)
        assert (n[:12] == 3).all()
        assert (n[12:] == 2).all()


def test_build_output_schema(tmp_path: Path):
    from nhf_spatial_targets.targets.run import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_runoff_project(tmp_path)
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "runoff_targets.nc") as ds:
        assert "lower_bound" in ds and "upper_bound" in ds
        assert "n_sources" in ds
        # centroid_lat/lon may be promoted to coords by decode_cf:
        assert "centroid_lat" in ds.coords or "centroid_lat" in ds.variables
        assert "centroid_lon" in ds.coords or "centroid_lon" in ds.variables
        assert ds["lower_bound"].attrs["units"] == "cfs"
        assert ds.attrs["Conventions"] == "CF-1.6"
        assert ds["time"].attrs["bounds"] == "time_bnds"
        assert "time_bnds" in ds.variables


def test_build_unit_chain_positive_and_ordered(tmp_path: Path):
    """Positive inputs produce positive bounds with lower <= upper everywhere."""
    from nhf_spatial_targets.targets.run import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_runoff_project(
        tmp_path,
        sources_per_year={
            "era5_land": {2000: ("ro", 0.05)},
            "gldas_noah_v21_monthly": {2000: ("runoff_total", 6.25)},
            "mwbm_climgrid": {2000: ("runoff", 30.0)},
        },
        period="2000-01-01/2000-12-31",
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "runoff_targets.nc") as ds:
        assert (ds["lower_bound"].values > 0).all()
        assert (ds["upper_bound"].values >= ds["lower_bound"].values).all()


def test_build_hru_mismatch_raises(tmp_path: Path):
    """Sources aggregated against different HRU sets -> raise."""
    from nhf_spatial_targets.targets.run import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_runoff_project(tmp_path, nn_fill=False)
    bad = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    times = pd.date_range("2000-01-01", "2000-12-01", freq="MS")
    hrus = [1, 2, 99]  # 99 instead of 3
    ds = xr.Dataset(
        {"ro": (("time", "nhm_id"), np.full((12, 3), 0.05, dtype=np.float32))},
        coords={"time": times, "nhm_id": hrus},
    )
    bad.unlink()
    ds.to_netcdf(bad)

    project = load(workdir)
    with pytest.raises(ValueError, match="HRU coords differ"):
        build(project)


def test_build_source_attr_reflects_active_sources(tmp_path: Path):
    """Global 'source' attr names exactly the sources actually consumed."""
    from nhf_spatial_targets.targets.run import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_runoff_project(
        tmp_path,
        # Only era5_land and gldas, drop mwbm:
        sources_per_year={
            "era5_land": {2000: ("ro", 0.05)},
            "gldas_noah_v21_monthly": {2000: ("runoff_total", 2.0)},
        },
        period="2000-01-01/2000-12-31",
        nn_fill=False,
    )
    # Override config to only enable the two sources we have:
    cfg_path = workdir / "config.yml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["targets"]["runoff"]["sources"] = ["era5_land", "gldas_noah_v21_monthly"]
    cfg_path.write_text(yaml.safe_dump(cfg))

    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "runoff_targets.nc") as ds:
        src_attr = ds.attrs["source"]
        assert "ERA5-Land" in src_attr
        assert "GLDAS" in src_attr
        assert "MWBM" not in src_attr  # not in active sources list
