"""Tests for the recharge target builder end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml


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


def _write_reitz_year_nc(
    path: Path, year: int, value: float, id_col: str = "nhm_id"
) -> None:
    """Reitz native: 1 timestep per year at mid-year, m/year."""
    times = pd.DatetimeIndex([f"{year}-07-01"])
    hrus = [1, 2, 3]
    arr = np.full((1, len(hrus)), value, dtype=np.float32)
    ds = xr.Dataset(
        {"total_recharge": (("time", id_col), arr)},
        coords={"time": times, id_col: hrus},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _write_watergap_year_nc(
    path: Path, year: int, value: float, id_col: str = "nhm_id"
) -> None:
    """WaterGAP native: 12 monthly mean rates at month-start, kg m-2 s-1."""
    times = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    hrus = [1, 2, 3]
    arr = np.full((len(times), len(hrus)), value, dtype=np.float32)
    ds = xr.Dataset(
        {"qrdif": (("time", id_col), arr)},
        coords={"time": times, id_col: hrus},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _write_era5_year_nc(
    path: Path, year: int, value: float, id_col: str = "nhm_id"
) -> None:
    """ERA5-Land native: 12 monthly accumulations at month-end, m water-eq."""
    times = pd.date_range(f"{year}-01-31", f"{year}-12-31", freq="ME")
    hrus = [1, 2, 3]
    arr = np.full((len(times), len(hrus)), value, dtype=np.float32)
    ds = xr.Dataset(
        {"ssro": (("time", id_col), arr)},
        coords={"time": times, id_col: hrus},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _make_rch_project(
    tmp_path: Path,
    *,
    period: str = "2000-01-01/2009-12-31",
    normalize_period: str | None = None,
    sources: list[str] | None = None,
    nn_fill: bool = True,
    write_reitz: bool = True,
    write_watergap: bool = True,
    write_era5: bool = True,
    reitz_value: float = 0.05,  # m/yr -> 50 mm/yr
    watergap_value: float = 1e-6,  # kg/m²/s -> ~31.5 mm/yr (linear in days_in_month)
    era5_value: float = 0.001,  # m/month -> 1 mm/month -> 12 mm/yr
) -> Path:
    """Build a project skeleton with synthetic fabric + per-year aggregated NCs."""
    if sources is None:
        sources = ["reitz2017", "watergap22d", "era5_land"]
    if normalize_period is None:
        normalize_period = period

    workdir = tmp_path / "proj"
    workdir.mkdir()
    fabric_path = tmp_path / "fabric.gpkg"
    _write_synthetic_fabric(fabric_path)
    (workdir / "fabric.json").write_text(json.dumps({"id_col": "nhm_id"}))

    cfg = {
        "datastore": str(tmp_path / "store"),
        "fabric": {"path": str(fabric_path), "id_col": "nhm_id"},
        "targets": {
            "recharge": {
                "period": period,
                "normalize_period": normalize_period,
                "sources": sources,
                "nn_fill": nn_fill,
            },
            "runoff": {"enabled": False},
            "aet": {"enabled": False},
            "soil_moisture": {"enabled": False},
            "snow_covered_area": {"enabled": False},
            "snow_water_equivalent": {"enabled": False},
        },
    }
    (workdir / "config.yml").write_text(yaml.safe_dump(cfg))

    agg_dir = workdir / "data" / "aggregated"
    years = list(
        range(
            pd.Timestamp(period.split("/")[0]).year,
            pd.Timestamp(period.split("/")[1]).year + 1,
        )
    )
    # Make the source values vary year-over-year so the normalize bound is
    # non-degenerate. Use a deterministic ramp (multiplier in [0.5, 1.5]).
    for i, year in enumerate(years):
        scale = 0.5 + (i / max(len(years) - 1, 1))  # 0.5 ramp to 1.5
        if write_reitz and "reitz2017" in sources:
            _write_reitz_year_nc(
                agg_dir / "reitz2017" / f"reitz2017_{year}_agg.nc",
                year,
                reitz_value * scale,
            )
        if write_watergap and "watergap22d" in sources:
            _write_watergap_year_nc(
                agg_dir / "watergap22d" / f"watergap22d_{year}_agg.nc",
                year,
                watergap_value * scale,
            )
        if write_era5 and "era5_land" in sources:
            _write_era5_year_nc(
                agg_dir / "era5_land" / f"era5_land_{year}_agg.nc",
                year,
                era5_value * scale,
            )
    return workdir


# ---------------------------------------------------------------------------
# Per-source unit shims
# ---------------------------------------------------------------------------


def test_reitz_to_mm_per_year_x1000_and_year_start():
    """reitz 0.05 m/yr → 50 mm/yr; time canonicalized to year-start."""
    from nhf_spatial_targets.targets.rch import reitz_to_mm_per_year

    da = xr.DataArray(
        np.array([[0.05, 0.10, 0.20]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2000-07-01"]), "nhm_id": [1, 2, 3]},
        attrs={"units": "m yr-1"},
    )
    out = reitz_to_mm_per_year(da)
    np.testing.assert_allclose(out.values, [[50.0, 100.0, 200.0]], rtol=1e-6)
    assert pd.Timestamp(out.time.values[0]) == pd.Timestamp("2000-01-01")
    assert out.attrs["units"] == "mm"


def test_watergap22d_to_mm_per_year_constant_rate():
    """1e-6 kg/m²/s × seconds_in_year (non-leap) = ~31.5 mm/yr."""
    from nhf_spatial_targets.targets.rch import watergap22d_to_mm_per_year

    times = pd.date_range("2001-01-01", "2001-12-01", freq="MS")
    da = xr.DataArray(
        np.full((12, 1), 1e-6, dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
        attrs={"units": "kg m-2 s-1"},
    )
    out = watergap22d_to_mm_per_year(da)
    expected = 1e-6 * 365 * 86400.0  # non-leap year: 365 days
    np.testing.assert_allclose(out.values, [[expected]], rtol=1e-5)
    assert pd.Timestamp(out.time.values[0]) == pd.Timestamp("2001-01-01")
    assert out.attrs["units"] == "mm"


def test_watergap22d_to_mm_per_year_leap_year():
    """Leap year (2000) → 366 days × 86400 s × rate."""
    from nhf_spatial_targets.targets.rch import watergap22d_to_mm_per_year

    times = pd.date_range("2000-01-01", "2000-12-01", freq="MS")
    da = xr.DataArray(
        np.full((12, 1), 1e-6, dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
    )
    out = watergap22d_to_mm_per_year(da)
    expected = 1e-6 * 366 * 86400.0
    np.testing.assert_allclose(out.values, [[expected]], rtol=1e-5)


def test_era5_ssro_to_mm_per_year_constant_accumulation():
    """0.001 m/month × 1000 × 12 months = 12 mm/yr."""
    from nhf_spatial_targets.targets.rch import era5_ssro_to_mm_per_year

    times = pd.date_range("2005-01-31", "2005-12-31", freq="ME")
    da = xr.DataArray(
        np.full((12, 1), 0.001, dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
        attrs={"units": "m"},
    )
    out = era5_ssro_to_mm_per_year(da)
    np.testing.assert_allclose(out.values, [[12.0]], rtol=1e-6)
    assert pd.Timestamp(out.time.values[0]) == pd.Timestamp("2005-01-01")
    assert out.attrs["units"] == "mm"


def test_shims_registered_and_well_formed():
    """SHIMS exposes the three expected source keys with correct aggregated_vars."""
    from nhf_spatial_targets.targets.rch import SHIMS

    by_key = {s.source_key: s for s in SHIMS}
    assert set(by_key) == {"reitz2017", "watergap22d", "era5_land"}
    assert by_key["reitz2017"].aggregated_var == "total_recharge"
    assert by_key["watergap22d"].aggregated_var == "qrdif"
    assert by_key["era5_land"].aggregated_var == "ssro"


# ---------------------------------------------------------------------------
# End-to-end build
# ---------------------------------------------------------------------------


def test_build_writes_unfilled_and_filled_files(tmp_path: Path):
    from nhf_spatial_targets.targets.rch import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_rch_project(tmp_path, period="2000-01-01/2009-12-31")
    project = load(workdir)
    build(project)
    assert (project.targets_dir() / "recharge_targets.nc").exists()
    assert (project.targets_dir() / "recharge_targets_nn_filled.nc").exists()


def test_build_output_schema(tmp_path: Path):
    from nhf_spatial_targets.targets.rch import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_rch_project(tmp_path, period="2000-01-01/2009-12-31", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "recharge_targets.nc") as ds:
        assert "lower_bound" in ds and "upper_bound" in ds
        assert "n_sources" in ds
        assert ds["lower_bound"].attrs["units"] == "1"
        assert ds["upper_bound"].attrs["units"] == "1"
        assert ds.attrs["Conventions"] == "CF-1.6"
        assert ds["time"].attrs["bounds"] == "time_bnds"
        assert "time_bnds" in ds.variables
        # 10 annual timesteps for 2000-2009
        assert ds.sizes["time"] == 10


def test_build_bounds_in_0_1_range(tmp_path: Path):
    """Normalized output: 0 <= lower <= upper <= 1 everywhere (or NaN)."""
    from nhf_spatial_targets.targets.rch import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_rch_project(tmp_path, period="2000-01-01/2009-12-31", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "recharge_targets.nc") as ds:
        finite_lo = ds["lower_bound"].values[np.isfinite(ds["lower_bound"].values)]
        finite_up = ds["upper_bound"].values[np.isfinite(ds["upper_bound"].values)]
        assert (finite_lo >= 0).all()
        assert (finite_lo <= 1).all()
        assert (finite_up >= 0).all()
        assert (finite_up <= 1).all()
        finite_pair = np.isfinite(ds["lower_bound"].values) & np.isfinite(
            ds["upper_bound"].values
        )
        assert (
            ds["upper_bound"].values[finite_pair]
            >= ds["lower_bound"].values[finite_pair]
        ).all()


def test_build_n_sources_full_period(tmp_path: Path):
    """All three sources present at every year-step → n_sources=3 everywhere."""
    from nhf_spatial_targets.targets.rch import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_rch_project(tmp_path, period="2000-01-01/2009-12-31", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "recharge_targets.nc") as ds:
        assert (ds["n_sources"].values == 3).all()


def test_build_normalize_uses_window(tmp_path: Path):
    """A flat-rate source on a year-varying scale normalizes to span [0, 1].

    The synthetic data ramps from 0.5x to 1.5x across years, so the min is
    at year 0 and max at year N-1. After normalization, year 0 should be 0
    and year N-1 should be 1 (per HRU, per source).
    """
    from nhf_spatial_targets.targets.rch import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_rch_project(tmp_path, period="2000-01-01/2009-12-31", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "recharge_targets.nc") as ds:
        # All three sources scale together → the min/max bound at each year
        # is the SAME across the three normalized sources (so lower == upper)
        # AND walks 0 -> 1 across the 10-year span.
        lower = ds["lower_bound"].values
        upper = ds["upper_bound"].values
        # First year: 0.0 (smallest scale value); last year: 1.0
        np.testing.assert_allclose(lower[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(upper[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(lower[-1], 1.0, atol=1e-6)
        np.testing.assert_allclose(upper[-1], 1.0, atol=1e-6)


def test_build_source_attr_reflects_active_sources(tmp_path: Path):
    """Dropping a source from config removes it from the output's source attr."""
    from nhf_spatial_targets.targets.rch import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_rch_project(
        tmp_path,
        period="2000-01-01/2009-12-31",
        sources=["reitz2017", "era5_land"],
        write_watergap=False,
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "recharge_targets.nc") as ds:
        src_attr = ds.attrs["source"]
        assert "Reitz 2017" in src_attr
        assert "ERA5-Land" in src_attr
        assert "WaterGAP" not in src_attr


def test_build_unknown_source_raises(tmp_path: Path):
    """recharge.sources with an unknown key raises before any IO."""
    from nhf_spatial_targets.targets.rch import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_rch_project(
        tmp_path,
        period="2000-01-01/2009-12-31",
        sources=["reitz2017", "not_a_real_source"],
        write_watergap=False,
        write_era5=False,
        nn_fill=False,
    )
    project = load(workdir)
    with pytest.raises(ValueError, match="unknown source 'not_a_real_source'"):
        build(project)


def test_build_hru_mismatch_raises(tmp_path: Path):
    """Source aggregated to a different HRU set than the fabric → raise."""
    from nhf_spatial_targets.targets.rch import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_rch_project(tmp_path, period="2000-01-01/2009-12-31", nn_fill=False)
    # Overwrite the reitz2017 2005 NC with a bad HRU set.
    bad = workdir / "data" / "aggregated" / "reitz2017" / "reitz2017_2005_agg.nc"
    times = pd.DatetimeIndex(["2005-07-01"])
    hrus = [1, 2, 99]  # 99 instead of 3
    ds = xr.Dataset(
        {
            "total_recharge": (
                ("time", "nhm_id"),
                np.full((1, 3), 0.05, dtype=np.float32),
            )
        },
        coords={"time": times, "nhm_id": hrus},
    )
    bad.unlink()
    ds.to_netcdf(bad)

    project = load(workdir)
    with pytest.raises(ValueError, match="HRU coords differ.*as sets"):
        build(project)


def test_build_nn_fill_actually_fills_nan_cells(tmp_path: Path):
    """End-to-end NN-fill: a single-source build with NaN at HRU 2 → filled file
    has HRU 2 finite and nn_filled=1 there.
    """
    from nhf_spatial_targets.targets.rch import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_rch_project(
        tmp_path,
        period="2000-01-01/2002-12-31",
        sources=["reitz2017"],
        write_watergap=False,
        write_era5=False,
        nn_fill=True,
    )
    # Rewrite all three reitz NCs so HRU 2 is NaN throughout.
    for year in (2000, 2001, 2002):
        bad = workdir / "data" / "aggregated" / "reitz2017" / f"reitz2017_{year}_agg.nc"
        bad.unlink()
        times = pd.DatetimeIndex([f"{year}-07-01"])
        vals = np.array(
            [[0.05 * (1 + (year - 2000) * 0.5), np.nan, 0.20]], dtype=np.float32
        )
        ds = xr.Dataset(
            {"total_recharge": (("time", "nhm_id"), vals)},
            coords={"time": times, "nhm_id": [1, 2, 3]},
        )
        ds.to_netcdf(bad)

    project = load(workdir)
    build(project)

    with xr.open_dataset(project.targets_dir() / "recharge_targets.nc") as out:
        assert np.isnan(out["lower_bound"].values[:, 1]).all()
        assert (out["n_sources"].values[:, 1] == 0).all()

    nn_path = project.targets_dir() / "recharge_targets_nn_filled.nc"
    assert nn_path.exists()
    with xr.open_dataset(nn_path) as filled:
        assert "nn_filled" in filled.data_vars
        assert np.isfinite(filled["lower_bound"].values[:, 1]).all()
        assert (filled["nn_filled"].values[:, 1] == 1).all()
        assert (filled["nn_filled"].values[:, 0] == 0).all()
