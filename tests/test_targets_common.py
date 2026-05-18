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


def test_compute_hru_centroids_returns_centroids_only(tmp_path: Path):
    """centroids-only helper omits area_m2."""
    from nhf_spatial_targets.targets._common import compute_hru_centroids

    workdir = _make_project_with_fabric(tmp_path)
    project = load(workdir)
    df = compute_hru_centroids(project)
    assert df.index.name == "nhm_id"
    assert set(df.columns) == {
        "centroid_x",
        "centroid_y",
        "centroid_lat",
        "centroid_lon",
    }
    assert "area_m2" not in df.columns
    assert len(df) == 3


def test_compute_hru_centroids_matches_combined_helper(tmp_path: Path):
    """Centroid columns from the lighter helper match the combined helper."""
    from nhf_spatial_targets.targets._common import (
        compute_hru_area_and_centroids,
        compute_hru_centroids,
    )

    workdir = _make_project_with_fabric(tmp_path)
    project = load(workdir)
    light = compute_hru_centroids(project)
    full = compute_hru_area_and_centroids(project)
    for col in ("centroid_x", "centroid_y", "centroid_lat", "centroid_lon"):
        np.testing.assert_array_equal(light[col].values, full[col].values)
    np.testing.assert_array_equal(light.index.values, full.index.values)


def test_compute_hru_areas_returns_area_only(tmp_path: Path):
    """area-only helper omits centroid columns."""
    from nhf_spatial_targets.targets._common import compute_hru_areas

    workdir = _make_project_with_fabric(tmp_path)
    project = load(workdir)
    df = compute_hru_areas(project)
    assert df.index.name == "nhm_id"
    assert set(df.columns) == {"area_m2"}
    assert len(df) == 3
    assert (df["area_m2"] > 0).all()


def test_compute_hru_areas_matches_combined_helper(tmp_path: Path):
    """area_m2 from the lighter helper matches the combined helper exactly."""
    from nhf_spatial_targets.targets._common import (
        compute_hru_area_and_centroids,
        compute_hru_areas,
    )

    workdir = _make_project_with_fabric(tmp_path)
    project = load(workdir)
    light = compute_hru_areas(project)
    full = compute_hru_area_and_centroids(project)
    np.testing.assert_array_equal(light["area_m2"].values, full["area_m2"].values)
    np.testing.assert_array_equal(light.index.values, full.index.values)


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


def test_write_target_nc_sort_dim_canonicalizes_hru_order(tmp_path: Path):
    """sort_dim sorts the HRU dim ascending at emission (issue #93)."""
    from nhf_spatial_targets.targets._common import write_target_nc

    ds = _toy_target_dataset().isel(nhm_id=[2, 0, 1])  # shuffle to [3, 1, 2]
    out = tmp_path / "shuffled.nc"
    write_target_nc(ds, out, title="sort test", sort_dim="nhm_id")
    with xr.open_dataset(out) as got:
        np.testing.assert_array_equal(got["nhm_id"].values, [1, 2, 3])
        # Values move with the coord, not by position.
        original_for_hru_1 = ds["lower_bound"].sel(nhm_id=1, time=ds["time"].values[1])
        round_tripped = got["lower_bound"].sel(nhm_id=1, time=got["time"].values[1])
        assert float(original_for_hru_1) == float(round_tripped)


def test_write_target_nc_no_sort_dim_preserves_caller_order(tmp_path: Path):
    """Without sort_dim, the writer leaves caller order untouched (#93 opt-in)."""
    from nhf_spatial_targets.targets._common import write_target_nc

    ds = _toy_target_dataset().isel(nhm_id=[2, 0, 1])  # [3, 1, 2]
    out = tmp_path / "no_sort.nc"
    write_target_nc(ds, out, title="no-sort test")
    with xr.open_dataset(out) as got:
        np.testing.assert_array_equal(got["nhm_id"].values, [3, 1, 2])


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


def test_reindex_to_day_start_floors_noon_timestamps():
    """Daymet-style noon timestamps land on the same day as midnight on master."""
    from nhf_spatial_targets.targets._common import reindex_to_day_start

    noon = _da_with_time(["2003-12-15 12:00:00", "2003-12-16 12:00:00"])
    master = pd.date_range("2003-12-15", "2003-12-16", freq="D")
    reindexed = reindex_to_day_start(noon, master)
    assert list(reindexed.time.values) == list(master.values)
    np.testing.assert_array_equal(reindexed.values, noon.values)


def test_reindex_to_day_start_already_midnight_is_idempotent():
    from nhf_spatial_targets.targets._common import reindex_to_day_start

    midnight = _da_with_time(["2003-12-15", "2003-12-16", "2003-12-17"])
    master = pd.date_range("2003-12-15", "2003-12-17", freq="D")
    reindexed = reindex_to_day_start(midnight, master)
    np.testing.assert_array_equal(reindexed.values, midnight.values)


def test_reindex_to_day_start_pads_missing_days_with_nan():
    """Days in master_index but absent from the source come out as NaN.

    This is the SWE target's period-union semantics — a SNODAS source
    that starts in 2003-10 contributes nothing for earlier days in a
    larger master index, but the bound is still defined wherever
    another source is finite.
    """
    from nhf_spatial_targets.targets._common import reindex_to_day_start

    partial = _da_with_time(["2003-12-15", "2003-12-16"])
    master = pd.date_range("2003-12-15", "2003-12-18", freq="D")
    reindexed = reindex_to_day_start(partial, master)
    assert len(reindexed.time) == 4
    assert np.isnan(reindexed.values[2:]).all()
    np.testing.assert_array_equal(reindexed.values[:2], partial.values)


def test_reindex_to_day_start_rejects_non_d_freq():
    """A freq='MS' master_index must raise, not silently produce all-NaN."""
    from nhf_spatial_targets.targets._common import reindex_to_day_start

    da = _da_with_time(["2003-12-15", "2003-12-16"])
    bad = pd.date_range("2003-12-01", "2003-12-31", freq="MS")
    with pytest.raises(ValueError, match="freq='D'"):
        reindex_to_day_start(da, bad)


def test_reindex_to_day_start_rejects_cftime_decoded_time():
    """Non-datetime64 time coords (e.g. cftime from non-standard calendars)
    raise rather than silently losing calendar info via DatetimeIndex coercion.
    """
    import cftime

    from nhf_spatial_targets.targets._common import reindex_to_day_start

    times = np.array(
        [cftime.DatetimeNoLeap(2003, 12, 15), cftime.DatetimeNoLeap(2003, 12, 16)],
        dtype=object,
    )
    da = xr.DataArray(
        np.array([[1.0], [2.0]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
    )
    master = pd.date_range("2003-12-15", "2003-12-16", freq="D")
    with pytest.raises(TypeError, match="datetime64-decoded time"):
        reindex_to_day_start(da, master)


def _write_year_nc_unsorted(
    path: Path,
    year: int,
    var: str,
    hrus: list[int],
    id_col: str = "nhm_id",
):
    """Write a per-year NC with HRUs in caller-specified (possibly unsorted) order."""
    times = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    data = np.arange(len(times) * len(hrus), dtype=np.float32).reshape(
        len(times), len(hrus)
    )
    ds = xr.Dataset(
        {var: (("time", id_col), data)},
        coords={"time": times, id_col: hrus},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def test_read_aggregated_source_sorts_hru_ascending(tmp_path: Path):
    """Source NCs written in non-monotonic HRU order come back canonically sorted.

    Mirrors the GFv2 era5_land case (#94) where gdptools wrote rows in a VPU-
    grouped (non-monotonic) order; downstream alignment against a sorted
    fabric requires the reader to canonicalise the order.
    """
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    _write_year_nc_unsorted(src_dir / f"{src}_2000_agg.nc", 2000, var, hrus=[3, 1, 2])

    project = load(workdir)
    da = read_aggregated_source(
        project, src, var, period=("2000-01-01", "2000-12-31"), chunks={"time": 12}
    )
    np.testing.assert_array_equal(da["nhm_id"].values, [1, 2, 3])
    # Values must follow the coord — i.e. rows are reordered, not just relabelled.
    # Original month-0 row was [0, 1, 2] (HRUs [3, 1, 2]); after sort it should
    # be [1, 2, 0] (values for HRUs [1, 2, 3]).
    np.testing.assert_array_equal(da.values[0], [1.0, 2.0, 0.0])


def test_compute_hru_area_and_centroids_returns_sorted_index(tmp_path: Path):
    """A fabric with rows in non-monotonic id_col order returns a sorted DataFrame.

    Mirrors gfv2_nhru_merged.gpkg (#94), which is a permutation of 1..N with
    a handful of out-of-order rows in the middle.
    """
    import geopandas as gpd
    from shapely.geometry import box

    from nhf_spatial_targets.targets._common import compute_hru_area_and_centroids

    fabric_path = tmp_path / "fabric.gpkg"
    gdf = gpd.GeoDataFrame(
        {"nhm_id": [3, 1, 2]},  # deliberately unsorted
        geometry=[
            box(-104.8, 40.0, -104.7, 40.1),
            box(-105.0, 40.0, -104.9, 40.1),
            box(-104.9, 40.0, -104.8, 40.1),
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
    df = compute_hru_area_and_centroids(project)
    np.testing.assert_array_equal(df.index.values, [1, 2, 3])
    # The centroid_lon for HRU 1 should match the polygon at lon ~-104.95
    # (the second row of the input GeoDataFrame).
    assert -105.0 < df.loc[1, "centroid_lon"] < -104.9
    assert -104.9 < df.loc[2, "centroid_lon"] < -104.8
    assert -104.8 < df.loc[3, "centroid_lon"] < -104.7


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


# ---------------------------------------------------------------------------
# SourceShim / shims_by_key
# ---------------------------------------------------------------------------


def _identity_shim(da: xr.DataArray) -> xr.DataArray:
    return da


def test_source_shim_is_frozen_dataclass():
    """SourceShim instances are immutable so SHIMS tuples can be module-level."""
    from dataclasses import FrozenInstanceError

    from nhf_spatial_targets.targets._common import SourceShim

    shim = SourceShim(
        source_key="foo",
        aggregated_var="x",
        description="foo",
        to_common_units=_identity_shim,
    )
    with pytest.raises(FrozenInstanceError):
        shim.source_key = "bar"  # type: ignore[misc]


def test_shims_by_key_returns_keyed_lookup():
    """shims_by_key indexes a SHIMS tuple by source_key."""
    from nhf_spatial_targets.targets._common import SourceShim, shims_by_key

    a = SourceShim("a", "va", "A", _identity_shim)
    b = SourceShim("b", "vb", "B", _identity_shim)
    out = shims_by_key((a, b))
    assert out == {"a": a, "b": b}


def test_shims_by_key_raises_on_duplicate_source_key():
    """Two shims with the same key would silently shadow; must raise."""
    from nhf_spatial_targets.targets._common import SourceShim, shims_by_key

    a = SourceShim("dup", "va", "A", _identity_shim)
    b = SourceShim("dup", "vb", "B", _identity_shim)
    with pytest.raises(ValueError, match="Duplicate SourceShim.source_key='dup'"):
        shims_by_key((a, b))


def test_shims_by_key_empty_tuple_returns_empty_dict():
    """Empty SHIMS tuple is a degenerate but valid input."""
    from nhf_spatial_targets.targets._common import shims_by_key

    assert shims_by_key(()) == {}


# ---------------------------------------------------------------------------
# validate_source_units (issue #130 — catalog-vs-shim unit drift guard)
# ---------------------------------------------------------------------------


def test_source_shim_accepts_expected_cf_units_field():
    """SourceShim grew an optional `expected_cf_units` field (issue #130)."""
    from nhf_spatial_targets.targets._common import SourceShim

    shim = SourceShim(
        source_key="era5_land",
        aggregated_var="ro",
        description="test",
        to_common_units=_identity_shim,
        expected_cf_units="m",
    )
    assert shim.expected_cf_units == "m"


def test_source_shim_expected_cf_units_defaults_to_none():
    """Defaulting to None preserves opt-out semantics for legacy shims."""
    from nhf_spatial_targets.targets._common import SourceShim

    shim = SourceShim(
        source_key="x",
        aggregated_var="y",
        description="t",
        to_common_units=_identity_shim,
    )
    assert shim.expected_cf_units is None


def test_validate_source_units_passes_when_catalog_matches_shim():
    """Happy path: real SHIMS registries pass against the real catalog."""
    from nhf_spatial_targets.targets._common import validate_source_units
    from nhf_spatial_targets.targets.run import SHIMS as RUN_SHIMS

    # Should not raise.
    validate_source_units(
        RUN_SHIMS, ["era5_land", "gldas_noah_v21_monthly", "mwbm_climgrid"]
    )


def test_validate_source_units_raises_on_catalog_drift():
    """A shim declaring different expected_cf_units from catalog raises."""
    from nhf_spatial_targets.targets._common import SourceShim, validate_source_units

    drifted = (
        SourceShim(
            source_key="era5_land",
            aggregated_var="ro",
            description="d",
            to_common_units=_identity_shim,
            expected_cf_units="DRIFTED",
        ),
    )
    with pytest.raises(ValueError, match="Catalog cf_units drift"):
        validate_source_units(drifted, ["era5_land"])


def test_validate_source_units_message_includes_both_units():
    """The error message must surface both strings so the operator can decide."""
    from nhf_spatial_targets.targets._common import SourceShim, validate_source_units

    drifted = (
        SourceShim(
            source_key="era5_land",
            aggregated_var="ro",
            description="d",
            to_common_units=_identity_shim,
            expected_cf_units="kg m-2",
        ),
    )
    with pytest.raises(ValueError) as exc_info:
        validate_source_units(drifted, ["era5_land"])
    msg = str(exc_info.value)
    assert "'m'" in msg  # catalog says "m"
    assert "'kg m-2'" in msg  # shim claims "kg m-2"


def test_validate_source_units_skips_when_expected_is_none():
    """expected_cf_units=None opts out of the check entirely."""
    from nhf_spatial_targets.targets._common import SourceShim, validate_source_units

    # If validation ran, this would raise (catalog says "m", not "wrong").
    # expected_cf_units=None means the check is skipped.
    opted_out = (
        SourceShim(
            source_key="era5_land",
            aggregated_var="ro",
            description="d",
            to_common_units=_identity_shim,
            expected_cf_units=None,
        ),
    )
    validate_source_units(opted_out, ["era5_land"])


def test_validate_source_units_raises_on_unknown_source_label():
    """An unknown config label (typo) surfaces as a startup error."""
    from nhf_spatial_targets.targets._common import SourceShim, validate_source_units

    shims = (
        SourceShim(
            source_key="x",
            aggregated_var="v",
            description="d",
            to_common_units=_identity_shim,
            expected_cf_units="m",
        ),
    )
    with pytest.raises(ValueError, match="no matching SourceShim"):
        validate_source_units(shims, ["not_in_registry"])


def test_validate_source_units_uses_config_label_for_catalog_lookup():
    """When config_label is set, catalog lookup goes through that key.

    SWE's era5_land_sd shim stores under a synthetic source_key but the
    catalog key is the canonical "era5_land"; validate_source_units must
    look up against the catalog under config_label, not source_key.
    """
    from nhf_spatial_targets.targets._common import validate_source_units
    from nhf_spatial_targets.targets.swe import SHIMS as SWE_SHIMS

    # The label "era5_land" resolves to the era5_land_sd shim; catalog
    # cf_units for era5_land/sd is "m", which matches expected_cf_units="m".
    validate_source_units(SWE_SHIMS, ["era5_land"])


def test_validate_source_units_raises_when_catalog_lacks_cf_units():
    """A shim pointing at a flat-string catalog entry raises with guidance."""
    from nhf_spatial_targets.targets._common import SourceShim, validate_source_units

    # merra_land is superseded and uses flat-list `variables: [SFMC]` (no
    # cf_units). A target that pointed at it must either gain catalog
    # cf_units or opt out via expected_cf_units=None.
    shims = (
        SourceShim(
            source_key="merra_land",
            aggregated_var="SFMC",
            description="d",
            to_common_units=_identity_shim,
            expected_cf_units="m3/m3",
        ),
    )
    with pytest.raises(ValueError, match="cannot resolve cf_units"):
        validate_source_units(shims, ["merra_land"])


def test_run_target_module_exposes_well_formed_shims():
    """targets/run.py SHIMS registry has the expected source keys and aggregated vars."""
    from nhf_spatial_targets.targets.run import SHIMS

    by_key = {s.source_key: s for s in SHIMS}
    assert set(by_key) == {"era5_land", "gldas_noah_v21_monthly", "mwbm_climgrid"}
    assert by_key["era5_land"].aggregated_var == "ro"
    assert by_key["gldas_noah_v21_monthly"].aggregated_var == "runoff_total"
    assert by_key["mwbm_climgrid"].aggregated_var == "runoff"
    # Each shim's to_common_units must be callable on a synthetic DataArray.
    da = xr.DataArray(
        np.array([[1.0]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2000-01-01"]), "nhm_id": [1]},
        attrs={"units": "anything"},
    )
    for shim in SHIMS:
        out = shim.to_common_units(da)
        assert out.attrs.get("units") == "mm"


def test_run_target_shims_declare_expected_cf_units():
    """Issue #130: every runoff shim pins its expected catalog cf_units."""
    from nhf_spatial_targets.targets.run import SHIMS

    by_key = {s.source_key: s for s in SHIMS}
    assert by_key["era5_land"].expected_cf_units == "m"
    assert by_key["gldas_noah_v21_monthly"].expected_cf_units == "kg m-2"
    assert by_key["mwbm_climgrid"].expected_cf_units == "mm"


def test_aet_target_module_exposes_well_formed_shims():
    """targets/aet.py SHIMS registry has the expected source keys and aggregated vars."""
    from nhf_spatial_targets.targets.aet import SHIMS

    by_key = {s.source_key: s for s in SHIMS}
    assert set(by_key) == {"mod16a2_v061", "ssebop", "mwbm_climgrid"}
    assert by_key["mod16a2_v061"].aggregated_var == "ET_500m"
    assert by_key["ssebop"].aggregated_var == "et"
    assert by_key["mwbm_climgrid"].aggregated_var == "aet"
    # SSEBop and MWBM are pass-throughs; verify they accept a synthetic DataArray.
    da = xr.DataArray(
        np.array([[1.0]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2000-01-01"]), "nhm_id": [1]},
        attrs={"units": "anything"},
    )
    for key in ("ssebop", "mwbm_climgrid"):
        out = by_key[key].to_common_units(da)
        assert out.attrs.get("units") == "mm"


def test_aet_target_shims_declare_expected_cf_units():
    """Issue #130: every AET shim pins its expected catalog cf_units."""
    from nhf_spatial_targets.targets.aet import SHIMS

    by_key = {s.source_key: s for s in SHIMS}
    assert by_key["mod16a2_v061"].expected_cf_units == "kg m-2"
    assert by_key["ssebop"].expected_cf_units == "mm"
    assert by_key["mwbm_climgrid"].expected_cf_units == "mm"


def test_swe_target_shims_declare_expected_cf_units():
    """Issue #130: every SWE shim pins its expected catalog cf_units."""
    from nhf_spatial_targets.targets.swe import SHIMS

    by_key = {s.source_key: s for s in SHIMS}
    assert by_key["daymet"].expected_cf_units == "kg m-2"
    assert by_key["snodas"].expected_cf_units == "kg m-2"
    assert by_key["era5_land_sd"].expected_cf_units == "m"
    assert by_key["margulis_wus_sr"].expected_cf_units == "m"


# ---------------------------------------------------------------------------
# parse_period
# ---------------------------------------------------------------------------


def test_parse_period_splits_endpoints():
    from nhf_spatial_targets.targets._common import parse_period

    assert parse_period("2000-01-01/2009-12-31") == ("2000-01-01", "2009-12-31")
    assert parse_period("2000/2009") == ("2000", "2009")


def test_parse_period_strips_whitespace():
    from nhf_spatial_targets.targets._common import parse_period

    assert parse_period("  2000-01-01 / 2009-12-31  ") == (
        "2000-01-01",
        "2009-12-31",
    )


def test_parse_period_raises_on_missing_slash():
    from nhf_spatial_targets.targets._common import parse_period

    with pytest.raises(ValueError, match="Expected 'YYYY-MM-DD"):
        parse_period("2000-01-01")


# ---------------------------------------------------------------------------
# check_hru_coords  (consider-2 follow-up; nit-5 test coverage)
# ---------------------------------------------------------------------------


def test_check_hru_coords_returns_none_when_coords_match():
    """Exact-match returns None (no raise)."""
    from nhf_spatial_targets.targets._common import check_hru_coords

    fabric_ids = np.array([1, 2, 3], dtype=np.int32)
    da = xr.DataArray(
        np.zeros(3, dtype=np.float32),
        dims=("nhm_id",),
        coords={"nhm_id": fabric_ids},
    )
    # Returning None is the contract; no exception means success.
    assert check_hru_coords(da, fabric_ids, "nhm_id", "test_source") is None


def test_check_hru_coords_raises_on_same_set_different_order():
    """Permuted-but-same-set raises a 'canonical-sort regression' message."""
    from nhf_spatial_targets.targets._common import check_hru_coords

    fabric_ids = np.array([1, 2, 3], dtype=np.int32)
    da = xr.DataArray(
        np.zeros(3, dtype=np.float32),
        dims=("nhm_id",),
        coords={"nhm_id": np.array([3, 1, 2], dtype=np.int32)},  # permuted
    )
    with pytest.raises(ValueError, match="canonical-sort invariant"):
        check_hru_coords(da, fabric_ids, "nhm_id", "test_source")


def test_check_hru_coords_raises_on_different_sets():
    """Different sets raise a 'Re-aggregate' message."""
    from nhf_spatial_targets.targets._common import check_hru_coords

    fabric_ids = np.array([1, 2, 3], dtype=np.int32)
    da = xr.DataArray(
        np.zeros(3, dtype=np.float32),
        dims=("nhm_id",),
        coords={"nhm_id": np.array([1, 2, 99], dtype=np.int32)},
    )
    with pytest.raises(ValueError, match="Re-aggregate 'test_source'"):
        check_hru_coords(da, fabric_ids, "nhm_id", "test_source")


def test_check_hru_coords_raises_on_different_length():
    """Different-length coords go through the 'different sets' branch."""
    from nhf_spatial_targets.targets._common import check_hru_coords

    fabric_ids = np.array([1, 2, 3], dtype=np.int32)
    da = xr.DataArray(
        np.zeros(2, dtype=np.float32),
        dims=("nhm_id",),
        coords={"nhm_id": np.array([1, 2], dtype=np.int32)},
    )
    with pytest.raises(ValueError, match="differ between fabric and source"):
        check_hru_coords(da, fabric_ids, "nhm_id", "test_source")


# ---------------------------------------------------------------------------
# build_n_sources_attrs  (consider-2 follow-up; nit-1 docstring/count check)
# ---------------------------------------------------------------------------


def test_build_n_sources_attrs_three_sources():
    """3-source target → flag_values=[0,1,2,3] and meanings 'none one two three'."""
    from nhf_spatial_targets.targets._common import build_n_sources_attrs

    attrs = build_n_sources_attrs(3)
    assert attrs["units"] == "1"
    assert attrs["long_name"] == "number of finite source contributions"
    assert attrs["flag_values"] == [0, 1, 2, 3]
    assert attrs["flag_meanings"] == "none one two three"
    assert attrs["coordinates"] == "centroid_lat centroid_lon"


def test_build_n_sources_attrs_one_source():
    """Single-source target → flag_values=[0,1], meanings 'none one'."""
    from nhf_spatial_targets.targets._common import build_n_sources_attrs

    attrs = build_n_sources_attrs(1)
    assert attrs["flag_values"] == [0, 1]
    assert attrs["flag_meanings"] == "none one"


def test_build_n_sources_attrs_custom_coords():
    from nhf_spatial_targets.targets._common import build_n_sources_attrs

    attrs = build_n_sources_attrs(2, ancillary_coords="lat lon")
    assert attrs["coordinates"] == "lat lon"


def test_build_n_sources_attrs_raises_when_count_exceeds_label_vocab():
    """6+ sources outpace the 'none..five' vocabulary and must raise."""
    from nhf_spatial_targets.targets._common import build_n_sources_attrs

    with pytest.raises(ValueError, match="5-source label vocabulary"):
        build_n_sources_attrs(6)


# ---------------------------------------------------------------------------
# stitch_year_chunks_to_target unit tests (year-chunked SWE pattern, PR #139)
# ---------------------------------------------------------------------------


def _write_year_chunk_nc(
    path: Path,
    year: int,
    *,
    hrus: list[int] | None = None,
    bound_value: float = 1.0,
    n_sources: int = 3,
    extra_attrs: dict | None = None,
) -> None:
    """Write a synthetic per-year SWE-style intermediate NC.

    Mirrors what `targets/swe.py:_build_year` emits via
    `write_bounds_target`: ``lower_bound``, ``upper_bound``, ``n_sources``
    data vars over ``(time, nhm_id)`` with a daily time index for that
    calendar year, plus the ``year_chunk`` global attr that the stitch
    helper must strip.
    """
    if hrus is None:
        hrus = [1, 2, 3]
    times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    shape = (len(times), len(hrus))
    ds = xr.Dataset(
        {
            "lower_bound": (
                ("time", "nhm_id"),
                np.full(shape, bound_value, dtype=np.float32),
            ),
            "upper_bound": (
                ("time", "nhm_id"),
                np.full(shape, bound_value + 1.0, dtype=np.float32),
            ),
            "n_sources": (
                ("time", "nhm_id"),
                np.full(shape, n_sources, dtype=np.int8),
            ),
        },
        coords={"time": times, "nhm_id": hrus},
        attrs={
            "Conventions": "CF-1.6",
            "title": f"per-year SWE chunk {year}",
            "year_chunk": year,
            **(extra_attrs or {}),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def test_stitch_year_chunks_combines_contiguous_time(tmp_path: Path):
    """Two adjacent year-chunks stitch into a single contiguous time axis."""
    from nhf_spatial_targets.targets._common import stitch_year_chunks_to_target

    inter = tmp_path / "intermediates"
    _write_year_chunk_nc(inter / "swe_targets_2003.nc", 2003)
    _write_year_chunk_nc(inter / "swe_targets_2004.nc", 2004)

    out = tmp_path / "swe_targets.nc"
    stitch_year_chunks_to_target(
        sorted(inter.glob("swe_targets_*.nc")),
        out,
        title="test stitched",
        extra_global_attrs={"period": "2003-01-01/2004-12-31"},
        sort_dim="nhm_id",
    )

    with xr.open_dataset(out) as ds:
        times = pd.DatetimeIndex(ds["time"].values)
        # 365 (2003) + 366 (2004 leap) = 731 days
        assert len(times) == 731
        assert times[0] == pd.Timestamp("2003-01-01")
        assert times[-1] == pd.Timestamp("2004-12-31")


def test_stitch_strips_year_chunk_attr(tmp_path: Path):
    """The per-year `year_chunk` attr must not leak into the stitched
    canonical file (PR #139 review must-fix). open_mfdataset's default
    combine_attrs='override' would otherwise carry over the first
    year's value and silently mislead about the file's actual scope.
    """
    from nhf_spatial_targets.targets._common import stitch_year_chunks_to_target

    inter = tmp_path / "intermediates"
    _write_year_chunk_nc(inter / "swe_targets_2003.nc", 2003)
    _write_year_chunk_nc(inter / "swe_targets_2004.nc", 2004)

    out = tmp_path / "swe_targets.nc"
    stitch_year_chunks_to_target(
        sorted(inter.glob("swe_targets_*.nc")),
        out,
        title="t",
        extra_global_attrs={"period": "2003/2004"},
        sort_dim="nhm_id",
    )
    with xr.open_dataset(out) as ds:
        assert "year_chunk" not in ds.attrs

    # Regression guard: the per-year files themselves must STILL carry
    # year_chunk for forensic value when inspected independently.
    with xr.open_dataset(inter / "swe_targets_2003.nc") as ds_y:
        assert ds_y.attrs["year_chunk"] == 2003


def test_stitch_applies_canonical_attrs_and_history(tmp_path: Path):
    """The stitched file gets CF-1.6, title, history, institution,
    and extra_global_attrs overlay."""
    from nhf_spatial_targets.targets._common import stitch_year_chunks_to_target

    inter = tmp_path / "intermediates"
    _write_year_chunk_nc(inter / "swe_targets_2003.nc", 2003)

    out = tmp_path / "swe_targets.nc"
    stitch_year_chunks_to_target(
        [inter / "swe_targets_2003.nc"],
        out,
        title="canonical SWE target",
        extra_global_attrs={"period": "2003/2003", "source": "x; y"},
        sort_dim="nhm_id",
    )
    with xr.open_dataset(out) as ds:
        assert ds.attrs["title"] == "canonical SWE target"
        assert ds.attrs["period"] == "2003/2003"
        assert ds.attrs["source"] == "x; y"
        assert ds.attrs["Conventions"] == "CF-1.6"
        assert ds.attrs["institution"] == "USGS"
        assert "stitched from 1 per-year NCs" in ds.attrs["history"]


def test_stitch_fails_loud_on_hru_coord_mismatch(tmp_path: Path):
    """join='exact' must raise when per-year files don't share HRU
    coords — this catches per-year-build corruption (fabric drift,
    weight-cache poisoning) rather than silently broadcasting NaN.
    """
    from nhf_spatial_targets.targets._common import stitch_year_chunks_to_target

    inter = tmp_path / "intermediates"
    _write_year_chunk_nc(inter / "swe_targets_2003.nc", 2003, hrus=[1, 2, 3])
    _write_year_chunk_nc(
        inter / "swe_targets_2004.nc", 2004, hrus=[1, 2, 99]
    )  # HRU 99 instead of 3

    out = tmp_path / "swe_targets.nc"
    with pytest.raises((ValueError, Exception)):
        stitch_year_chunks_to_target(
            sorted(inter.glob("swe_targets_*.nc")),
            out,
            title="t",
            extra_global_attrs=None,
            sort_dim="nhm_id",
        )


def test_stitch_raises_on_empty_input(tmp_path: Path):
    from nhf_spatial_targets.targets._common import stitch_year_chunks_to_target

    out = tmp_path / "swe_targets.nc"
    with pytest.raises(ValueError, match="intermediate_files is empty"):
        stitch_year_chunks_to_target(
            [], out, title="t", extra_global_attrs=None, sort_dim="nhm_id"
        )


def test_stitch_atomic_write_no_partial_on_failure(tmp_path: Path, monkeypatch):
    """If to_netcdf raises mid-write, the tempfile is cleaned up and
    the final path doesn't exist — same atomic-write contract as
    write_target_nc.
    """
    from nhf_spatial_targets.targets import _common as _c

    inter = tmp_path / "intermediates"
    _write_year_chunk_nc(inter / "swe_targets_2003.nc", 2003)

    out = tmp_path / "swe_targets.nc"

    real_to_netcdf = xr.Dataset.to_netcdf

    def _boom(self, *args, **kwargs):
        # Simulate a mid-write IO error (disk full, killed worker, etc.)
        raise OSError("simulated to_netcdf failure")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _boom)
    with pytest.raises(OSError, match="simulated to_netcdf failure"):
        _c.stitch_year_chunks_to_target(
            [inter / "swe_targets_2003.nc"],
            out,
            title="t",
            extra_global_attrs=None,
            sort_dim="nhm_id",
        )
    assert not out.exists(), "final output should not exist after failure"
    leftover = list(tmp_path.glob("*.tmp"))
    assert not leftover, f"tempfile leak: {leftover}"
    # restore (paranoia — monkeypatch undoes it but be explicit)
    monkeypatch.setattr(xr.Dataset, "to_netcdf", real_to_netcdf)
