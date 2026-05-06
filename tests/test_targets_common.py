"""Tests for shared multi-source-minmax target machinery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from nhf_spatial_targets.workspace import load
from tests.conftest import make_minimal_project, write_year_nc


def test_read_aggregated_source_concats_per_year_nc(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    write_year_nc(src_dir / f"{src}_2000_agg.nc", 2000, var)
    write_year_nc(src_dir / f"{src}_2001_agg.nc", 2001, var)

    project = load(workdir)
    da = read_aggregated_source(
        project, src, var, period=("2000-01-01", "2001-12-31"), chunks={"time": 12}
    )
    assert da.dims == ("time", "nhm_id")
    assert len(da.time) == 24
    assert da.time.values[0] == np.datetime64("2000-01-01")
    assert da.time.values[-1] == np.datetime64("2001-12-01")


def test_read_aggregated_source_slices_to_period(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    for y in (1999, 2000, 2001, 2002):
        write_year_nc(src_dir / f"{src}_{y}_agg.nc", y, var)

    project = load(workdir)
    da = read_aggregated_source(
        project, src, var, period=("2000-06-01", "2001-06-30"), chunks={"time": 12}
    )
    # months 2000-06 .. 2001-06 inclusive -> 13 months
    assert len(da.time) == 13


def test_read_aggregated_source_raises_when_dir_empty(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    project = load(workdir)
    with pytest.raises(FileNotFoundError, match="No aggregated NC files found"):
        read_aggregated_source(
            project, "era5_land", "ro", period=("2000-01-01", "2001-12-31")
        )


def test_read_aggregated_source_raises_when_period_outside_coverage(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    write_year_nc(src_dir / f"{src}_2000_agg.nc", 2000, var)

    project = load(workdir)
    with pytest.raises(ValueError, match="entirely outside source coverage"):
        read_aggregated_source(project, src, var, period=("2010-01-01", "2010-12-31"))


def test_read_aggregated_source_raises_diagnostic_on_missing_var(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    src = "era5_land"
    src_dir = workdir / "data" / "aggregated" / src
    write_year_nc(src_dir / f"{src}_2000_agg.nc", 2000, "ro")

    project = load(workdir)
    with pytest.raises(KeyError, match="Available variables"):
        read_aggregated_source(
            project, src, "not_a_var", period=("2000-01-01", "2000-12-31")
        )


def _da_with_time(times, hrus=(1, 2, 3), values=None) -> xr.DataArray:
    if values is None:
        values = np.arange(len(times) * len(hrus), dtype=np.float32).reshape(
            len(times), len(hrus)
        )
    return xr.DataArray(
        values,
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(times), "nhm_id": list(hrus)},
    )


def test_reindex_to_month_start_maps_eom_to_ms():
    from nhf_spatial_targets.targets._common import reindex_to_month_start

    eom = _da_with_time(["2000-01-31", "2000-02-29", "2000-03-31"])
    master = pd.date_range("2000-01-01", "2000-03-01", freq="MS")
    reindexed = reindex_to_month_start(eom, master)
    assert list(reindexed.time.values) == list(master.values)
    np.testing.assert_array_equal(reindexed.values, eom.values)


def test_reindex_to_month_start_maps_mid_month_to_ms():
    from nhf_spatial_targets.targets._common import reindex_to_month_start

    mid = _da_with_time(["2000-01-15", "2000-02-15", "2000-03-15"])
    master = pd.date_range("2000-01-01", "2000-03-01", freq="MS")
    reindexed = reindex_to_month_start(mid, master)
    np.testing.assert_array_equal(reindexed.values, mid.values)


def test_reindex_to_month_start_pads_missing_months_with_nan():
    """Months in master_index but absent from the source come out as NaN."""
    from nhf_spatial_targets.targets._common import reindex_to_month_start

    partial = _da_with_time(["2000-01-01", "2000-02-01"])
    master = pd.date_range("2000-01-01", "2000-04-01", freq="MS")
    reindexed = reindex_to_month_start(partial, master)
    assert len(reindexed.time) == 4
    assert np.isnan(reindexed.values[2:]).all()
    np.testing.assert_array_equal(reindexed.values[:2], partial.values)


def test_reindex_to_month_start_already_ms_is_idempotent():
    from nhf_spatial_targets.targets._common import reindex_to_month_start

    ms = _da_with_time(["2000-01-01", "2000-02-01", "2000-03-01"])
    master = pd.date_range("2000-01-01", "2000-03-01", freq="MS")
    reindexed = reindex_to_month_start(ms, master)
    np.testing.assert_array_equal(reindexed.values, ms.values)


def test_multi_source_nanminmax_three_finite_sources():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(["2000-01-01"], values=np.array([[10, 20, 30]], dtype=np.float32))
    b = _da_with_time(["2000-01-01"], values=np.array([[15, 25, 35]], dtype=np.float32))
    c = _da_with_time(["2000-01-01"], values=np.array([[20, 30, 40]], dtype=np.float32))
    lower, upper, n = multi_source_nanminmax({"a": a, "b": b, "c": c})
    np.testing.assert_array_equal(lower.values, [[10, 20, 30]])
    np.testing.assert_array_equal(upper.values, [[20, 30, 40]])
    np.testing.assert_array_equal(n.values, [[3, 3, 3]])


def test_multi_source_nanminmax_partial_nan_uses_finite_only():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(
        ["2000-01-01"], values=np.array([[10.0, np.nan, 30.0]], dtype=np.float32)
    )
    b = _da_with_time(
        ["2000-01-01"], values=np.array([[15.0, 25.0, np.nan]], dtype=np.float32)
    )
    lower, upper, n = multi_source_nanminmax({"a": a, "b": b})
    np.testing.assert_array_equal(lower.values, [[10, 25, 30]])
    np.testing.assert_array_equal(upper.values, [[15, 25, 30]])
    np.testing.assert_array_equal(n.values, [[2, 1, 1]])


def test_multi_source_nanminmax_all_nan_returns_nan_and_zero():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(
        ["2000-01-01"], hrus=(1,), values=np.array([[np.nan]], dtype=np.float32)
    )
    b = _da_with_time(
        ["2000-01-01"], hrus=(1,), values=np.array([[np.nan]], dtype=np.float32)
    )
    lower, upper, n = multi_source_nanminmax({"a": a, "b": b})
    assert np.isnan(lower.values[0, 0])
    assert np.isnan(upper.values[0, 0])
    assert n.values[0, 0] == 0


def test_multi_source_nanminmax_n_sources_is_int8():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(["2000-01-01"], values=np.array([[10.0, 20.0, 30.0]], np.float32))
    _, _, n = multi_source_nanminmax({"a": a})
    assert n.dtype == np.int8


def test_multi_source_nanminmax_raises_on_hru_mismatch():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(["2000-01-01"], hrus=(1, 2, 3))
    b = _da_with_time(["2000-01-01"], hrus=(1, 2, 4))
    with pytest.raises(ValueError, match="HRU coords differ"):
        multi_source_nanminmax({"a": a, "b": b})


def _write_synthetic_fabric(path: Path, id_col: str = "nhm_id"):
    """Write a 3-polygon GeoPackage in EPSG:4326."""
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {id_col: [1, 2, 3]},
        geometry=[
            box(-105.0, 40.0, -104.9, 40.1),  # ~10x10 km in mid-CONUS
            box(-104.9, 40.0, -104.8, 40.1),
            box(-104.8, 40.0, -104.7, 40.1),
        ],
        crs="EPSG:4326",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")


def _make_project_with_fabric(tmp_path: Path) -> Path:
    workdir = tmp_path / "proj"
    workdir.mkdir()
    fabric_path = tmp_path / "fabric.gpkg"
    _write_synthetic_fabric(fabric_path)
    (workdir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "datastore": str(tmp_path / "store"),
                "fabric": {"path": str(fabric_path), "id_col": "nhm_id"},
            }
        )
    )
    (workdir / "fabric.json").write_text(json.dumps({"id_col": "nhm_id"}))
    return workdir


def test_compute_hru_area_and_centroids_returns_expected_columns(tmp_path: Path):
    from nhf_spatial_targets.targets._common import compute_hru_area_and_centroids

    workdir = _make_project_with_fabric(tmp_path)
    project = load(workdir)
    df = compute_hru_area_and_centroids(project)
    assert df.index.name == "nhm_id"
    assert set(df.columns) == {
        "area_m2",
        "centroid_x",
        "centroid_y",
        "centroid_lat",
        "centroid_lon",
    }
    assert len(df) == 3


def test_compute_hru_area_and_centroids_areas_within_1pct(tmp_path: Path):
    """Each polygon is ~0.1 deg square at lat 40 N ≈ ~85 km² (varies by lat)."""
    from nhf_spatial_targets.targets._common import compute_hru_area_and_centroids

    workdir = _make_project_with_fabric(tmp_path)
    project = load(workdir)
    df = compute_hru_area_and_centroids(project)
    # All three polygons are the same size at this latitude.
    areas = df["area_m2"].values
    assert (areas > 50e6).all() and (areas < 150e6).all()
    # Adjacent polygons should agree to within 1% in area.
    assert abs(areas[0] - areas[1]) / areas[0] < 0.01


def test_compute_hru_area_and_centroids_lat_lon_in_range(tmp_path: Path):
    from nhf_spatial_targets.targets._common import compute_hru_area_and_centroids

    workdir = _make_project_with_fabric(tmp_path)
    project = load(workdir)
    df = compute_hru_area_and_centroids(project)
    assert df["centroid_lon"].between(-180, 180).all()
    assert df["centroid_lat"].between(-90, 90).all()
    # Should be near 40N, -105E given the polygons:
    assert df["centroid_lat"].between(39, 41).all()
    assert df["centroid_lon"].between(-106, -104).all()


def _toy_target_dataset(
    id_col: str = "nhm_id",
) -> xr.Dataset:
    """Toy 3-month / 3-HRU dataset with the bound vars and an n_sources."""
    times = pd.date_range("2000-01-01", "2000-03-01", freq="MS")
    bnds = np.stack([times.values, (times + pd.offsets.MonthBegin(1)).values], axis=1)
    hrus = np.array([1, 2, 3])
    lower = np.array(
        [[1.0, 2.0, np.nan], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    upper = lower + 1.0
    n = np.array([[2, 2, 0], [2, 2, 2], [2, 2, 2]], dtype=np.int8)
    ds = xr.Dataset(
        {
            "lower_bound": (("time", id_col), lower),
            "upper_bound": (("time", id_col), upper),
            "n_sources": (("time", id_col), n),
        },
        coords={
            "time": times,
            id_col: hrus,
            "time_bnds": (("time", "nv"), bnds),
            "centroid_lat": ((id_col,), np.array([40.0, 40.1, 40.2])),
            "centroid_lon": ((id_col,), np.array([-105.0, -104.9, -104.8])),
        },
    )
    ds["time"].attrs["bounds"] = "time_bnds"
    for v in ("lower_bound", "upper_bound", "n_sources"):
        ds[v].attrs["coordinates"] = "centroid_lat centroid_lon"
    return ds


def test_write_target_nc_round_trips_via_xarray(tmp_path: Path):
    from nhf_spatial_targets.targets._common import write_target_nc

    ds = _toy_target_dataset()
    # Set units on lower_bound so the round-trip can verify it survives.
    ds["lower_bound"].attrs["units"] = "cfs"
    out = tmp_path / "runoff_targets.nc"
    write_target_nc(ds, out, title="Test runoff target")
    assert out.exists()
    with xr.open_dataset(out, decode_cf=True) as got:
        assert got.attrs["Conventions"] == "CF-1.6"
        assert got.attrs["title"] == "Test runoff target"
        assert "lower_bound" in got.data_vars
        assert "upper_bound" in got.data_vars
        assert "n_sources" in got.data_vars
        assert got["lower_bound"].dtype == np.float32
        assert got["n_sources"].dtype == np.int8
        assert got["time"].attrs["bounds"] == "time_bnds"
        assert "time_bnds" in got.variables
        assert got["lower_bound"].attrs["units"] == "cfs"
        # CF aux-coord linkage must round-trip on every bound + diagnostic var.
        # xarray (decode_cf=True) consumes the raw "coordinates" attr and promotes
        # the named variables to Dataset coords — so we check that the coords are
        # reachable via the Dataset, not via the per-var attr string (which xarray
        # strips during decode).  The raw attr IS written to disk (verified via
        # netCDF4 directly); this is the decoded equivalent.
        for v in ("lower_bound", "upper_bound", "n_sources"):
            coords_attr = got[v].attrs.get("coordinates", "")
            linked_via_attr = (
                "centroid_lat" in coords_attr and "centroid_lon" in coords_attr
            )
            linked_via_dataset = (
                "centroid_lat" in got.coords and "centroid_lon" in got.coords
            )
            assert linked_via_attr or linked_via_dataset, (
                f"{v}: CF aux-coord linkage lost after round-trip — "
                f"coordinates attr={coords_attr!r}, "
                f"centroid_lat in coords={('centroid_lat' in got.coords)}, "
                f"centroid_lon in coords={('centroid_lon' in got.coords)}"
            )
        # NaN preserved (decode_cf maps _FillValue back to NaN):
        assert np.isnan(got["lower_bound"].values[0, 2])


def test_write_target_nc_atomic_no_partial_on_failure(tmp_path: Path, monkeypatch):
    """If to_netcdf raises, the final path must not exist (tempfile cleanup)."""
    from nhf_spatial_targets.targets._common import write_target_nc

    out = tmp_path / "runoff_targets.nc"

    def _boom(self, *a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _boom)
    with pytest.raises(RuntimeError, match="disk full"):
        write_target_nc(_toy_target_dataset(), out, title="x")
    assert not out.exists()


def test_reindex_to_month_start_rejects_non_ms_freq():
    """A freq='ME' master_index must raise, not silently produce all-NaN."""
    from nhf_spatial_targets.targets._common import reindex_to_month_start

    da = _da_with_time(["2000-01-01", "2000-02-01"])
    bad = pd.date_range("2000-01-31", "2000-02-29", freq="ME")
    with pytest.raises(ValueError, match="freq='MS'"):
        reindex_to_month_start(da, bad)


def test_compute_hru_area_and_centroids_raises_on_duplicate_id_col(tmp_path: Path):
    """A fabric with duplicate HRU IDs must raise, not silently produce a
    non-unique-index DataFrame."""
    import geopandas as gpd
    from shapely.geometry import box

    from nhf_spatial_targets.targets._common import compute_hru_area_and_centroids

    fabric_path = tmp_path / "fabric.gpkg"
    gdf = gpd.GeoDataFrame(
        {"nhm_id": [1, 2, 1]},  # duplicate
        geometry=[
            box(-105.0, 40.0, -104.9, 40.1),
            box(-104.9, 40.0, -104.8, 40.1),
            box(-104.8, 40.0, -104.7, 40.1),
        ],
        crs="EPSG:4326",
    )
    fabric_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(fabric_path, driver="GPKG")

    workdir = tmp_path / "proj"
    workdir.mkdir()
    (workdir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "datastore": str(tmp_path / "store"),
                "fabric": {"path": str(fabric_path), "id_col": "nhm_id"},
            }
        )
    )
    (workdir / "fabric.json").write_text(json.dumps({"id_col": "nhm_id"}))

    project = load(workdir)
    with pytest.raises(ValueError, match="duplicate"):
        compute_hru_area_and_centroids(project)
