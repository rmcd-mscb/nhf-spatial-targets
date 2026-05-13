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


def test_build_emits_id_col_sorted_target_ncs(tmp_path: Path):
    """Both unfilled and NN-filled target NCs emerge sorted ascending by id_col (#93).

    Distinct from test_build_succeeds_when_source_hru_order_differs_from_fabric:
    that test exercises end-to-end recovery from upstream disorder; this one
    pins the emission-boundary invariant on the happy path (sorted inputs in,
    sorted outputs out) for both target files.
    """
    from nhf_spatial_targets.targets.run import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_runoff_project(tmp_path)
    project = load(workdir)
    build(project)

    for fname in ("runoff_targets.nc", "runoff_targets_nn_filled.nc"):
        with xr.open_dataset(project.targets_dir() / fname) as ds:
            ids = ds["nhm_id"].values
            assert np.all(np.diff(ids) > 0), (
                f"{fname}: HRU dim not strictly ascending; got {ids}"
            )


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
    with pytest.raises(ValueError, match="HRU coords differ.*as sets"):
        build(project)


def test_build_succeeds_when_source_hru_order_differs_from_fabric(tmp_path: Path):
    """Source NCs with HRUs in non-monotonic order must not break the build.

    Regression test for #94: gfv2's era5_land aggregated NCs ship rows in a
    VPU-grouped order while the fabric is (near-)sorted by nat_hru_id. The
    target builder must canonicalise both sides via id_col-ascending sort
    rather than relying on positional alignment.

    All per-year NCs for a single source share the same (permuted) order in
    production (one gdptools run per source); the test mirrors that — a
    cross-year ordering mismatch would trip xr.open_mfdataset before the
    sort could run.
    """
    from nhf_spatial_targets.targets.run import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_runoff_project(
        tmp_path,
        period="2000-01-01/2000-12-31",
        sources_per_year={
            "era5_land": {2000: ("ro", 0.05)},
            "gldas_noah_v21_monthly": {2000: ("runoff_total", 6.25)},
            "mwbm_climgrid": {2000: ("runoff", 30.0)},
        },
        nn_fill=False,
    )
    # Rewrite the era5_land NC with HRUs in a permuted (non-monotonic) order.
    # Per-HRU values are distinct so a positional misalignment produces a wrong
    # answer rather than just rearranged metadata.
    bad = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    times = pd.date_range("2000-01-01", "2000-12-01", freq="MS")
    permuted_hrus = [3, 1, 2]
    per_hru_value = {1: 0.05, 2: 0.10, 3: 0.20}  # m/month
    arr = np.array(
        [[per_hru_value[h] for h in permuted_hrus]] * len(times), dtype=np.float32
    )
    ds = xr.Dataset(
        {"ro": (("time", "nhm_id"), arr)},
        coords={"time": times, "nhm_id": permuted_hrus},
    )
    bad.unlink()
    ds.to_netcdf(bad)

    project = load(workdir)
    build(project)

    out = project.targets_dir() / "runoff_targets.nc"
    assert out.exists()
    with xr.open_dataset(out) as got:
        # HRU dim must be canonically sorted in the target output, regardless
        # of input order.
        np.testing.assert_array_equal(got["nhm_id"].values, [1, 2, 3])
        # ERA5-Land contributes the lower bound here (gldas/mwbm are scaled to
        # ~roughly the same cfs magnitude, but HRU-3's ERA5 value of 0.20
        # m/month is the largest of the three; HRU-1's 0.05 is the smallest).
        # If positional alignment had silently mismatched, HRU-3's bound would
        # not reflect 0.20 m/month.
        lb = got["lower_bound"].sel(nhm_id=[1, 2, 3]).isel(time=0).values
        assert (lb > 0).all()
        assert lb[2] > lb[0]  # HRU 3 ERA5 (0.20) > HRU 1 (0.05) m/month


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


# ---------------------------------------------------------------------------
# C1 — Numeric unit-chain tests
# ---------------------------------------------------------------------------


def test_era5_to_mm_per_month_exact():
    """ERA5-Land ro is m/month -> mm/month is x1000."""
    from nhf_spatial_targets.targets.run import era5_to_mm_per_month

    times = pd.date_range("2000-01-01", "2000-03-01", freq="MS")
    da = xr.DataArray(
        np.array([[0.05], [0.10], [0.025]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
        attrs={"units": "m"},
    )
    out = era5_to_mm_per_month(da)
    np.testing.assert_allclose(out.values, [[50.0], [100.0], [25.0]], rtol=1e-6)
    assert out.attrs["units"] == "mm"


def test_gldas_to_mm_per_month_exact_jan_and_leap_feb():
    """GLDAS runoff_total is mean of 3-hourly accums -> mm/month is x8 x days_in_month.

    Jan 2000: 31 days -> factor 8*31 = 248
    Feb 2000: 29 days (leap) -> factor 8*29 = 232
    """
    from nhf_spatial_targets.targets.run import gldas_to_mm_per_month

    times = pd.date_range("2000-01-01", "2000-02-01", freq="MS")
    da = xr.DataArray(
        np.array([[0.20], [0.20]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
        attrs={"units": "kg m-2"},
    )
    out = gldas_to_mm_per_month(da)
    # 0.20 * 8 * 31 = 49.6 ; 0.20 * 8 * 29 = 46.4 (leap-year February)
    np.testing.assert_allclose(out.values, [[49.6], [46.4]], rtol=1e-6)
    assert out.attrs["units"] == "mm"


def test_gldas_to_mm_per_month_non_leap_february():
    """Non-leap February (28 days) -> factor 8*28 = 224."""
    from nhf_spatial_targets.targets.run import gldas_to_mm_per_month

    da = xr.DataArray(
        np.array([[0.10]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2001-02-01"]), "nhm_id": [1]},
        attrs={"units": "kg m-2"},
    )
    out = gldas_to_mm_per_month(da)
    # 0.10 * 8 * 28 = 22.4
    np.testing.assert_allclose(out.values, [[22.4]], rtol=1e-6)


def test_mwbm_to_mm_per_month_passthrough():
    """MWBM runoff is already mm/month; values unchanged, units re-stamped."""
    from nhf_spatial_targets.targets.run import mwbm_to_mm_per_month

    da = xr.DataArray(
        np.array([[10.5, 20.0, 5.25]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2000-01-01"]), "nhm_id": [1, 2, 3]},
        attrs={"units": "mm"},
    )
    out = mwbm_to_mm_per_month(da)
    np.testing.assert_array_equal(out.values, da.values)
    assert out.attrs["units"] == "mm"


def test_mm_per_month_to_cfs_exact_january():
    """31 mm/mo over a 1 km2 (1e6 m2) HRU in January.

    cfs = (31 * 1e-3 / 31) * 1e6 * 35.3146667 / 86400
        = 1e-3 * 1e6 * 35.3146667 / 86400
        = 1000 * 35.3146667 / 86400
        ~= 0.408734
    """
    from nhf_spatial_targets.targets.run import mm_per_month_to_cfs

    da = xr.DataArray(
        np.array([[31.0]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2000-01-01"]), "nhm_id": [1]},
        attrs={"units": "mm"},
    )
    area = xr.DataArray(
        np.array([1e6], dtype=np.float64),
        dims=("nhm_id",),
        coords={"nhm_id": [1]},
    )
    out = mm_per_month_to_cfs(da, area)
    expected = (31 * 1e-3 / 31) * 1e6 * 35.3146667 / 86400
    np.testing.assert_allclose(out.values, [[expected]], rtol=1e-4)
    assert out.attrs["units"] == "cfs"


def test_mm_per_month_to_cfs_february_uses_28_or_29_days():
    """February's days_in_month must come from the time coord, not be hardcoded."""
    from nhf_spatial_targets.targets.run import mm_per_month_to_cfs

    # Non-leap Feb: 28 days
    da_2001 = xr.DataArray(
        np.array([[28.0]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2001-02-01"]), "nhm_id": [1]},
        attrs={"units": "mm"},
    )
    # Leap Feb: 29 days
    da_2000 = xr.DataArray(
        np.array([[29.0]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2000-02-01"]), "nhm_id": [1]},
        attrs={"units": "mm"},
    )
    area = xr.DataArray(
        np.array([1e6], dtype=np.float64),
        dims=("nhm_id",),
        coords={"nhm_id": [1]},
    )
    # Both should produce identical cfs (numerator and denominator scale together):
    # 28 mm / 28 days = 1 mm/day  ;  29 mm / 29 days = 1 mm/day  -> same cfs.
    out_2001 = mm_per_month_to_cfs(da_2001, area)
    out_2000 = mm_per_month_to_cfs(da_2000, area)
    np.testing.assert_allclose(out_2001.values, out_2000.values, rtol=1e-6)


# ---------------------------------------------------------------------------
# C2 — End-to-end NN-fill path
# ---------------------------------------------------------------------------


def test_build_nn_fill_actually_fills_nan_cells(tmp_path: Path):
    """End-to-end NN-fill: an aggregated NC with NaN at one HRU/month
    must produce a *_nn_filled.nc with that cell filled and nn_filled=1."""
    import json

    from nhf_spatial_targets.targets.run import build
    from nhf_spatial_targets.workspace import load

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
                "period": "2000-01-01/2000-03-31",
                "nn_fill": True,
                "sources": ["era5_land"],  # single source so any NaN in source
                # propagates to NaN bounds
            }
        },
    }
    (workdir / "config.yml").write_text(yaml.safe_dump(cfg))

    # Build a synthetic ERA5 NC where HRU 2 is NaN at all 3 months but HRU 1
    # and HRU 3 are finite — the NN-fill should adopt HRU 1's values for
    # HRU 2 (HRU 1 is the geometrically-nearest finite donor at x=0 vs HRU
    # 3 at x=2 in the synthetic fabric layout where polygons are adjacent
    # at lon=-105, -104.9, -104.8).
    src = "era5_land"
    src_dir = workdir / "data" / "aggregated" / src
    src_dir.mkdir(parents=True)
    times = pd.date_range("2000-01-01", "2000-03-01", freq="MS")
    arr = np.full((3, 3), 0.05, dtype=np.float32)  # 50 mm/mo
    arr[:, 1] = np.nan  # HRU 2 NaN throughout
    ds = xr.Dataset(
        {"ro": (("time", "nhm_id"), arr)},
        coords={"time": times, "nhm_id": [1, 2, 3]},
    )
    ds.to_netcdf(src_dir / f"{src}_2000_agg.nc")

    project = load(workdir)
    build(project)

    # Honest-NaN file: HRU 2 stays NaN, n_sources=0 there.
    with xr.open_dataset(project.targets_dir() / "runoff_targets.nc") as out:
        assert np.isnan(out["lower_bound"].values[:, 1]).all()
        assert (out["n_sources"].values[:, 1] == 0).all()

    # NN-filled file: HRU 2 now has finite values, nn_filled=1 there.
    nn_path = project.targets_dir() / "runoff_targets_nn_filled.nc"
    assert nn_path.exists()
    with xr.open_dataset(nn_path) as filled:
        assert "nn_filled" in filled.data_vars
        assert np.isfinite(filled["lower_bound"].values[:, 1]).all()
        assert (filled["nn_filled"].values[:, 1] == 1).all()
        # HRUs 1 and 3 were already finite; nn_filled=0 there:
        assert (filled["nn_filled"].values[:, 0] == 0).all()
        assert (filled["nn_filled"].values[:, 2] == 0).all()
        # The filled value should match HRU 1's (geometrically closer than HRU 3):
        np.testing.assert_allclose(
            filled["lower_bound"].values[:, 1],
            filled["lower_bound"].values[:, 0],
        )
