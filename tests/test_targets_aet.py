"""Tests for the AET target builder end-to-end."""

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


def _write_monthly_nc(
    path: Path,
    year: int,
    var: str,
    value: float,
    id_col: str = "nhm_id",
) -> None:
    """Write a per-year aggregated NC with monthly cadence (12 timesteps)."""
    times = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    hrus = [1, 2, 3]
    arr = np.full((len(times), len(hrus)), value, dtype=np.float32)
    ds = xr.Dataset(
        {var: (("time", id_col), arr)},
        coords={"time": times, id_col: hrus},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _write_mod16a2_8day_nc(
    path: Path,
    year: int,
    value: float,
    id_col: str = "nhm_id",
) -> None:
    """Write a synthetic per-year MOD16A2 aggregated NC at 8-day cadence.

    LP DAAC ships 46 composites per year on DOYs 1, 9, 17, ..., 361. The
    composite at DOY 361 covers only 5-6 days (Dec 27 → Jan 1). With a
    constant ``value`` representing the composite-window ET (mm of water),
    a calendar month fully covered by composites integrates to ~total ET
    over that month under the overlap-weighted resample.
    """
    doys = list(range(1, 366, 8))  # 1, 9, ..., 361
    times = pd.DatetimeIndex(
        [
            pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=d - 1)
            for d in doys
        ]
    )
    hrus = [1, 2, 3]
    arr = np.full((len(times), len(hrus)), value, dtype=np.float32)
    ds = xr.Dataset(
        {"ET_500m": (("time", id_col), arr)},
        coords={"time": times, id_col: hrus},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _make_aet_project(
    tmp_path: Path,
    *,
    period: str = "2000-01-01/2001-12-31",
    sources: list[str] | None = None,
    nn_fill: bool = True,
    write_mod16a2: bool = True,
    write_ssebop: bool = True,
    write_mwbm: bool = True,
    mod16a2_value: float = 8.0,  # mm per 8-day composite => 1 mm/day rate
    ssebop_value: float = 30.0,  # mm/month
    mwbm_value: float = 25.0,  # mm/month
) -> Path:
    """Build a project skeleton with synthetic fabric + per-year aggregated NCs."""
    if sources is None:
        sources = ["mod16a2_v061", "ssebop", "mwbm_climgrid"]

    workdir = tmp_path / "proj"
    workdir.mkdir()
    fabric_path = tmp_path / "fabric.gpkg"
    _write_synthetic_fabric(fabric_path)
    (workdir / "fabric.json").write_text(json.dumps({"id_col": "nhm_id"}))

    cfg = {
        "datastore": str(tmp_path / "store"),
        "fabric": {"path": str(fabric_path), "id_col": "nhm_id"},
        "targets": {
            "aet": {
                "period": period,
                "sources": sources,
                "nn_fill": nn_fill,
            },
            # Disable other targets so defaults-merge doesn't try to run them.
            "runoff": {"enabled": False},
            "recharge": {"enabled": False},
            "soil_moisture": {"enabled": False},
            "snow_covered_area": {"enabled": False},
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
    for year in years:
        if write_mod16a2 and "mod16a2_v061" in sources:
            _write_mod16a2_8day_nc(
                agg_dir / "mod16a2_v061" / f"mod16a2_v061_{year}_agg.nc",
                year,
                mod16a2_value,
            )
        if write_ssebop and "ssebop" in sources:
            _write_monthly_nc(
                agg_dir / "ssebop" / f"ssebop_{year}_agg.nc",
                year,
                "et",
                ssebop_value,
            )
        if write_mwbm and "mwbm_climgrid" in sources:
            _write_monthly_nc(
                agg_dir / "mwbm_climgrid" / f"mwbm_climgrid_{year}_agg.nc",
                year,
                "aet",
                mwbm_value,
            )
    return workdir


# ---------------------------------------------------------------------------
# Per-source unit shims
# ---------------------------------------------------------------------------


def test_ssebop_to_mm_per_month_passthrough():
    from nhf_spatial_targets.targets.aet import ssebop_to_mm_per_month

    da = xr.DataArray(
        np.array([[10.5, 20.0, 5.25]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2000-01-01"]), "nhm_id": [1, 2, 3]},
        attrs={"units": "mm"},
    )
    out = ssebop_to_mm_per_month(da)
    np.testing.assert_array_equal(out.values, da.values)
    assert out.attrs["units"] == "mm"


def test_mwbm_to_mm_per_month_passthrough():
    from nhf_spatial_targets.targets.aet import mwbm_to_mm_per_month

    da = xr.DataArray(
        np.array([[15.0, 22.5]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2000-06-01"]), "nhm_id": [1, 2]},
        attrs={"units": "mm"},
    )
    out = mwbm_to_mm_per_month(da)
    np.testing.assert_array_equal(out.values, da.values)
    assert out.attrs["units"] == "mm"


def test_mod16a2_overlap_weighted_january_full_month():
    """Constant 1 mm/day rate over 8-day composites integrates to 31 mm in Jan.

    Each composite carries 8.0 mm of ET over its 8-day window (≡ 1 mm/day).
    January is fully covered by the four composites starting on DOY 1, 9, 17, 25
    (DOY 25's window extends into early February). The overlap-weighted sum
    must therefore equal 31 days × 1 mm/day = 31 mm.
    """
    from nhf_spatial_targets.targets.aet import mod16a2_to_mm_per_month

    doys = list(range(1, 366, 8))
    times = pd.DatetimeIndex(
        [
            pd.Timestamp(year=2001, month=1, day=1) + pd.Timedelta(days=d - 1)
            for d in doys
        ]
    )
    da = xr.DataArray(
        np.full((len(times), 1), 8.0, dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
        attrs={"units": "kg m-2", "scale_factor": 0.1},
    )
    out = mod16a2_to_mm_per_month(da)
    # First month-start in the output should be 2001-01-01.
    assert pd.Timestamp(out["time"].values[0]) == pd.Timestamp("2001-01-01")
    np.testing.assert_allclose(out.sel(time="2001-01-01").values, [31.0], rtol=1e-6)


def test_mod16a2_overlap_weighted_february_non_leap():
    """Non-leap February (28 days) at 1 mm/day -> 28 mm."""
    from nhf_spatial_targets.targets.aet import mod16a2_to_mm_per_month

    doys = list(range(1, 366, 8))
    times = pd.DatetimeIndex(
        [
            pd.Timestamp(year=2001, month=1, day=1) + pd.Timedelta(days=d - 1)
            for d in doys
        ]
    )
    da = xr.DataArray(
        np.full((len(times), 1), 8.0, dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
    )
    out = mod16a2_to_mm_per_month(da)
    np.testing.assert_allclose(out.sel(time="2001-02-01").values, [28.0], rtol=1e-6)


def test_mod16a2_overlap_weighted_february_leap():
    """Leap February (29 days) at 1 mm/day -> 29 mm."""
    from nhf_spatial_targets.targets.aet import mod16a2_to_mm_per_month

    doys = list(range(1, 366, 8))
    times = pd.DatetimeIndex(
        [
            pd.Timestamp(year=2000, month=1, day=1) + pd.Timedelta(days=d - 1)
            for d in doys
        ]
    )
    da = xr.DataArray(
        np.full((len(times), 1), 8.0, dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
    )
    out = mod16a2_to_mm_per_month(da)
    np.testing.assert_allclose(out.sel(time="2000-02-01").values, [29.0], rtol=1e-6)


def test_mod16a2_year_end_composite_short_window():
    """The DOY 361 composite is capped at 5-6 days (Dec 27 -> Jan 1).

    Each composite is given a value equal to (composite_length × 1 mm/day),
    so a faithful 1 mm/day rate holds across the year-end boundary. Under
    that rate, December must integrate to 31 mm — which is only true when
    the short year-end composite is recognised as a 5-day window (carrying
    5 mm), not stretched to 8 days. If the cap were missing, the 8 mm in
    the year-end composite would spill 3 mm into the following January
    (LP DAAC's reason for capping at Jan 1).
    """
    from nhf_spatial_targets.targets.aet import mod16a2_to_mm_per_month

    # Two years of composites so December's coverage is well-defined.
    doys_2001 = list(range(1, 366, 8))
    doys_2002 = list(range(1, 17, 8))  # only first two of 2002 (Jan 1, Jan 9)
    pairs = [(2001, d) for d in doys_2001] + [(2002, d) for d in doys_2002]
    times = pd.DatetimeIndex(
        [
            pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=d - 1)
            for (y, d) in pairs
        ]
    )
    # Per-composite value = composite length in days (≡ 1 mm/day rate).
    # The year-end 2001 composite (DOY 361) starts Dec 27 2001 and is
    # capped at Jan 1 2002 -> 5 days, so its value is 5.
    values = []
    for i, t in enumerate(times):
        next_t = times[i + 1] if i + 1 < len(times) else None
        year_end = pd.Timestamp(year=t.year + 1, month=1, day=1)
        nominal_end = t + pd.Timedelta(days=8)
        end = min(nominal_end, year_end)
        if next_t is not None:
            end = min(end, next_t)
        values.append(float((end - t).days))
    da = xr.DataArray(
        np.array(values, dtype=np.float32).reshape(-1, 1),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1]},
    )
    out = mod16a2_to_mm_per_month(da)
    # December 2001: 31 days × 1 mm/day = 31 mm. The 5-day year-end
    # composite contributes its full 5 mm to December (none spills into
    # 2002 January because the cap shortens the window).
    np.testing.assert_allclose(out.sel(time="2001-12-01").values, [31.0], rtol=1e-6)


def test_mm_per_month_to_inches_per_day_january():
    """25.4 mm in January -> 1 inch over 31 days -> 1/31 inches/day."""
    from nhf_spatial_targets.targets.aet import mm_per_month_to_inches_per_day

    da = xr.DataArray(
        np.array([[25.4]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2001-01-01"]), "nhm_id": [1]},
        attrs={"units": "mm"},
    )
    out = mm_per_month_to_inches_per_day(da)
    np.testing.assert_allclose(out.values, [[1.0 / 31.0]], rtol=1e-6)
    assert out.attrs["units"] == "inches/day"


def test_mm_per_month_to_inches_per_day_february_leap_vs_non_leap():
    """Same mm/month -> different inches/day for 28- vs 29-day Februaries."""
    from nhf_spatial_targets.targets.aet import mm_per_month_to_inches_per_day

    da_2001 = xr.DataArray(
        np.array([[25.4]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2001-02-01"]), "nhm_id": [1]},
    )
    da_2000 = xr.DataArray(
        np.array([[25.4]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2000-02-01"]), "nhm_id": [1]},
    )
    np.testing.assert_allclose(
        mm_per_month_to_inches_per_day(da_2001).values, [[1.0 / 28.0]], rtol=1e-6
    )
    np.testing.assert_allclose(
        mm_per_month_to_inches_per_day(da_2000).values, [[1.0 / 29.0]], rtol=1e-6
    )


# ---------------------------------------------------------------------------
# End-to-end build
# ---------------------------------------------------------------------------


def test_build_writes_unfilled_and_filled_files(tmp_path: Path):
    from nhf_spatial_targets.targets.aet import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_aet_project(tmp_path, period="2000-01-01/2000-12-31")
    project = load(workdir)
    build(project)
    assert (project.targets_dir() / "aet_targets.nc").exists()
    assert (project.targets_dir() / "aet_targets_nn_filled.nc").exists()


def test_build_output_schema(tmp_path: Path):
    from nhf_spatial_targets.targets.aet import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_aet_project(tmp_path, period="2000-01-01/2000-12-31", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "aet_targets.nc") as ds:
        assert "lower_bound" in ds and "upper_bound" in ds
        assert "n_sources" in ds
        assert "centroid_lat" in ds.coords or "centroid_lat" in ds.variables
        assert "centroid_lon" in ds.coords or "centroid_lon" in ds.variables
        assert ds["lower_bound"].attrs["units"] == "inches/day"
        assert ds["upper_bound"].attrs["units"] == "inches/day"
        assert ds.attrs["Conventions"] == "CF-1.6"
        assert ds["time"].attrs["bounds"] == "time_bnds"
        assert "time_bnds" in ds.variables


def test_build_emits_id_col_sorted_target_ncs(tmp_path: Path):
    """Both unfilled and NN-filled NCs come out sorted ascending by id_col (#93)."""
    from nhf_spatial_targets.targets.aet import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_aet_project(tmp_path, period="2000-01-01/2000-12-31")
    project = load(workdir)
    build(project)
    for fname in ("aet_targets.nc", "aet_targets_nn_filled.nc"):
        with xr.open_dataset(project.targets_dir() / fname) as ds:
            ids = ds["nhm_id"].values
            assert np.all(np.diff(ids) > 0), (
                f"{fname}: HRU dim not strictly ascending; got {ids}"
            )


def test_build_unit_chain_positive_and_ordered(tmp_path: Path):
    """Positive inputs produce positive bounds with lower <= upper everywhere."""
    from nhf_spatial_targets.targets.aet import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_aet_project(tmp_path, period="2000-01-01/2000-12-31", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "aet_targets.nc") as ds:
        assert (ds["lower_bound"].values > 0).all()
        assert (ds["upper_bound"].values >= ds["lower_bound"].values).all()


def test_build_n_sources_full_period(tmp_path: Path):
    """All three sources present -> n_sources=3 at every cell."""
    from nhf_spatial_targets.targets.aet import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_aet_project(tmp_path, period="2000-01-01/2000-12-31", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "aet_targets.nc") as ds:
        assert (ds["n_sources"].values == 3).all()


def test_build_source_attr_reflects_active_sources(tmp_path: Path):
    """Global 'source' attr names exactly the sources actually consumed.

    Mirrors targets/run.py: dropping a source from `aet.sources` must drop
    it from the global attr too. This is the lever recipes §2 calls for to
    resolve the MOD16A2-v061 inclusion debate without a code edit.
    """
    from nhf_spatial_targets.targets.aet import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_aet_project(
        tmp_path,
        period="2000-01-01/2000-12-31",
        sources=["ssebop", "mwbm_climgrid"],
        write_mod16a2=False,
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "aet_targets.nc") as ds:
        src_attr = ds.attrs["source"]
        assert "SSEBop" in src_attr
        assert "MWBM" in src_attr
        assert "MOD16A2" not in src_attr


def test_build_unknown_source_raises(tmp_path: Path):
    """aet.sources with an unknown key raises before any IO."""
    from nhf_spatial_targets.targets.aet import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_aet_project(
        tmp_path,
        period="2000-01-01/2000-12-31",
        sources=["ssebop", "not_a_real_source"],
        write_mod16a2=False,
        write_mwbm=False,
        nn_fill=False,
    )
    project = load(workdir)
    with pytest.raises(ValueError, match="unknown source 'not_a_real_source'"):
        build(project)


def test_build_hru_mismatch_raises(tmp_path: Path):
    """SSEBop aggregated to a different HRU set than the fabric -> raise."""
    from nhf_spatial_targets.targets.aet import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_aet_project(tmp_path, period="2000-01-01/2000-12-31", nn_fill=False)
    bad = workdir / "data" / "aggregated" / "ssebop" / "ssebop_2000_agg.nc"
    times = pd.date_range("2000-01-01", "2000-12-01", freq="MS")
    hrus = [1, 2, 99]  # 99 instead of 3
    ds = xr.Dataset(
        {"et": (("time", "nhm_id"), np.full((12, 3), 30.0, dtype=np.float32))},
        coords={"time": times, "nhm_id": hrus},
    )
    bad.unlink()
    ds.to_netcdf(bad)

    project = load(workdir)
    with pytest.raises(ValueError, match="HRU coords differ.*as sets"):
        build(project)


def test_build_nn_fill_actually_fills_nan_cells(tmp_path: Path):
    """End-to-end NN-fill: aggregated NC with NaN at one HRU/month produces
    a *_nn_filled.nc with that cell filled and nn_filled=1.
    """
    from nhf_spatial_targets.targets.aet import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_aet_project(
        tmp_path,
        period="2000-01-01/2000-03-31",
        sources=["ssebop"],  # single source -> any NaN propagates to bound NaN
        write_mod16a2=False,
        write_mwbm=False,
        nn_fill=True,
    )
    # Overwrite the SSEBop NC so HRU 2 is NaN at all 3 months.
    src_dir = workdir / "data" / "aggregated" / "ssebop"
    times = pd.date_range("2000-01-01", "2000-03-01", freq="MS")
    arr = np.full((3, 3), 30.0, dtype=np.float32)
    arr[:, 1] = np.nan
    ds = xr.Dataset(
        {"et": (("time", "nhm_id"), arr)},
        coords={"time": times, "nhm_id": [1, 2, 3]},
    )
    (src_dir / "ssebop_2000_agg.nc").unlink()
    ds.to_netcdf(src_dir / "ssebop_2000_agg.nc")

    project = load(workdir)
    build(project)

    with xr.open_dataset(project.targets_dir() / "aet_targets.nc") as out:
        assert np.isnan(out["lower_bound"].values[:, 1]).all()
        assert (out["n_sources"].values[:, 1] == 0).all()

    nn_path = project.targets_dir() / "aet_targets_nn_filled.nc"
    assert nn_path.exists()
    with xr.open_dataset(nn_path) as filled:
        assert "nn_filled" in filled.data_vars
        assert np.isfinite(filled["lower_bound"].values[:, 1]).all()
        assert (filled["nn_filled"].values[:, 1] == 1).all()
        assert (filled["nn_filled"].values[:, 0] == 0).all()
        assert (filled["nn_filled"].values[:, 2] == 0).all()
