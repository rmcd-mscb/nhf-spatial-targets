"""Tests for the SWE target builder end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml


# Map config-label → on-disk per-source dir + aggregated variable name.
# Mirrors targets/swe.py:_CONFIG_LABEL_TO_SHIM_KEY + SHIMS.aggregated_var.
_SOURCE_DIRS_AND_VARS = {
    "daymet": ("daymet", "swe"),
    "snodas": ("snodas", "swe"),
    "era5_land": ("era5_land_sd", "sd"),
    "margulis_wus_sr": ("margulis_wus_sr", "SWE"),
}


def _write_synthetic_fabric(path: Path, id_col: str = "nhm_id"):
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {id_col: [1, 2, 3]},
        geometry=[
            box(-122.0, 44.0, -121.9, 44.1),
            box(-121.9, 44.0, -121.8, 44.1),
            box(-121.8, 44.0, -121.7, 44.1),
        ],
        crs="EPSG:4326",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")


def _write_daily_nc(
    path: Path,
    year: int,
    var: str,
    value: float,
    id_col: str = "nhm_id",
) -> None:
    """Write a per-year aggregated NC with daily cadence (365/366 timesteps)."""
    times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    hrus = [1, 2, 3]
    arr = np.full((len(times), len(hrus)), value, dtype=np.float32)
    ds = xr.Dataset(
        {var: (("time", id_col), arr)},
        coords={"time": times, id_col: hrus},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _make_swe_project(
    tmp_path: Path,
    *,
    period: str = "2003-10-01/2003-12-31",
    sources: list[str] | None = None,
    nn_fill: bool = True,
    fabric_token: str | None = None,
    write_daymet: bool = True,
    write_snodas: bool = True,
    write_era5_sd: bool = True,
    write_margulis: bool = True,
    # All in their native units; the builder converts to inches at the
    # tail end. Constants chosen so multi-source min/max comes back
    # ordered (daymet < snodas < era5 < margulis in inches).
    daymet_value_mm: float = 50.0,  # 50 mm ≈ 1.97 in
    snodas_value_mm: float = 80.0,  # 80 mm ≈ 3.15 in
    era5_sd_value_m: float = 0.1,  # 100 mm ≈ 3.94 in
    margulis_value_m: float = 0.2,  # 200 mm ≈ 7.87 in
) -> Path:
    """Build a project skeleton with synthetic fabric + per-year aggregated NCs."""
    if sources is None:
        sources = ["daymet", "snodas", "era5_land", "margulis_wus_sr"]

    workdir = tmp_path / "proj"
    workdir.mkdir()
    fabric_path = tmp_path / "fabric.gpkg"
    _write_synthetic_fabric(fabric_path)
    (workdir / "fabric.json").write_text(json.dumps({"id_col": "nhm_id"}))

    cfg = {
        "datastore": str(tmp_path / "store"),
        "fabric": {
            "path": str(fabric_path),
            "id_col": "nhm_id",
            "token": fabric_token,
        },
        "targets": {
            "snow_water_equivalent": {
                "period": period,
                "sources": sources,
                "nn_fill": nn_fill,
            },
            "runoff": {"enabled": False},
            "aet": {"enabled": False},
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
    per_source_writes = (
        ("daymet", write_daymet, daymet_value_mm),
        ("snodas", write_snodas, snodas_value_mm),
        ("era5_land", write_era5_sd, era5_sd_value_m),
        ("margulis_wus_sr", write_margulis, margulis_value_m),
    )
    for src_label, do_write, value in per_source_writes:
        if not do_write or src_label not in sources:
            continue
        on_disk_key, var = _SOURCE_DIRS_AND_VARS[src_label]
        for year in years:
            _write_daily_nc(
                agg_dir / on_disk_key / f"{on_disk_key}_{year}_agg.nc",
                year,
                var,
                value,
            )
    return workdir


# ---------------------------------------------------------------------------
# Per-source unit shims
# ---------------------------------------------------------------------------


def test_daymet_to_mm_passthrough():
    from nhf_spatial_targets.targets.swe import daymet_to_mm

    da = xr.DataArray(
        np.array([[12.5, 20.0]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2003-12-15"]), "nhm_id": [1, 2]},
    )
    out = daymet_to_mm(da)
    np.testing.assert_array_equal(out.values, da.values)
    assert out.attrs["units"] == "mm"


def test_snodas_to_mm_passthrough():
    from nhf_spatial_targets.targets.swe import snodas_to_mm

    da = xr.DataArray(
        np.array([[12.5, 20.0]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2003-12-15"]), "nhm_id": [1, 2]},
    )
    out = snodas_to_mm(da)
    np.testing.assert_array_equal(out.values, da.values)
    assert out.attrs["units"] == "mm"


def test_era5_sd_metres_to_mm():
    from nhf_spatial_targets.targets.swe import era5_sd_to_mm

    da = xr.DataArray(
        np.array([[0.1, 0.25]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2003-12-15"]), "nhm_id": [1, 2]},
    )
    out = era5_sd_to_mm(da)
    np.testing.assert_allclose(out.values, [[100.0, 250.0]], rtol=1e-6)
    assert out.attrs["units"] == "mm"


def test_margulis_metres_to_mm():
    from nhf_spatial_targets.targets.swe import margulis_to_mm

    da = xr.DataArray(
        np.array([[0.5]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2003-12-15"]), "nhm_id": [1]},
    )
    out = margulis_to_mm(da)
    np.testing.assert_allclose(out.values, [[500.0]], rtol=1e-6)
    assert out.attrs["units"] == "mm"


def test_mm_to_inches_linear():
    from nhf_spatial_targets.targets.swe import mm_to_inches

    da = xr.DataArray(
        np.array([[25.4, 50.8]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2003-12-15"]), "nhm_id": [1, 2]},
    )
    out = mm_to_inches(da)
    np.testing.assert_allclose(out.values, [[1.0, 2.0]], rtol=1e-6)
    assert out.attrs["units"] == "inches"


# ---------------------------------------------------------------------------
# Fabric scope filter (logic-level)
# ---------------------------------------------------------------------------


def test_fabric_scope_filter_keeps_scoped_source_when_token_matches():
    from nhf_spatial_targets.targets.swe import _filter_sources_by_fabric_scope

    kept = _filter_sources_by_fabric_scope(
        ["daymet", "snodas", "era5_land", "margulis_wus_sr"], "or"
    )
    assert "margulis_wus_sr" in kept
    assert kept == ["daymet", "snodas", "era5_land", "margulis_wus_sr"]


def test_fabric_scope_filter_drops_scoped_source_when_token_mismatches():
    from nhf_spatial_targets.targets.swe import _filter_sources_by_fabric_scope

    kept = _filter_sources_by_fabric_scope(
        ["daymet", "snodas", "era5_land", "margulis_wus_sr"], None
    )
    assert "margulis_wus_sr" not in kept
    assert kept == ["daymet", "snodas", "era5_land"]


# ---------------------------------------------------------------------------
# End-to-end build
# ---------------------------------------------------------------------------


def test_build_writes_unfilled_and_filled_files(tmp_path: Path):
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(tmp_path, fabric_token="or")
    project = load(workdir)
    build(project)
    assert (project.targets_dir() / "swe_targets.nc").exists()
    assert (project.targets_dir() / "swe_targets_nn_filled.nc").exists()


def test_build_output_schema(tmp_path: Path):
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(tmp_path, fabric_token="or", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        assert "lower_bound" in ds and "upper_bound" in ds
        assert "n_sources" in ds
        assert "centroid_lat" in ds.coords or "centroid_lat" in ds.variables
        assert "centroid_lon" in ds.coords or "centroid_lon" in ds.variables
        assert ds["lower_bound"].attrs["units"] == "inches"
        assert ds["upper_bound"].attrs["units"] == "inches"
        assert ds["lower_bound"].attrs["cell_methods"] == "time: point"
        assert ds.attrs["Conventions"] == "CF-1.6"
        assert ds["time"].attrs["bounds"] == "time_bnds"
        assert "time_bnds" in ds.variables


def test_build_daily_time_index(tmp_path: Path):
    """Time axis is one timestamp per day across the requested period."""
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        period="2003-12-01/2003-12-31",
        fabric_token="or",
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        times = pd.DatetimeIndex(ds["time"].values)
        assert len(times) == 31
        assert times[0] == pd.Timestamp("2003-12-01")
        assert times[-1] == pd.Timestamp("2003-12-31")


def test_build_unit_chain_min_max_ordered(tmp_path: Path):
    """Daymet=50mm, SNODAS=80mm, ERA5=100mm, Margulis=200mm → bounds in inches.

    Lower bound = daymet = 50/25.4 ≈ 1.969 in; upper = margulis = 200/25.4 ≈
    7.874 in. Verifies the per-source unit shims compose correctly.
    """
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        period="2003-12-15/2003-12-15",
        fabric_token="or",
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        np.testing.assert_allclose(ds["lower_bound"].values, 50.0 / 25.4, rtol=1e-5)
        np.testing.assert_allclose(ds["upper_bound"].values, 200.0 / 25.4, rtol=1e-5)
        assert (ds["n_sources"].values == 4).all()


def test_build_oregon_includes_margulis_in_source_attr(tmp_path: Path):
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(tmp_path, fabric_token="or", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        src_attr = ds.attrs["source"]
        assert "Margulis" in src_attr
        assert "Daymet" in src_attr
        assert "SNODAS" in src_attr
        assert "ERA5-Land" in src_attr
        assert ds.attrs["fabric_token"] == "or"


def test_build_no_token_drops_margulis(tmp_path: Path):
    """Fabric token unset (e.g. gfv2-style CONUS project) → Margulis is
    silently dropped via fabric_scope filter; bound reduces to 3 sources.
    """
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        fabric_token=None,
        nn_fill=False,
        write_margulis=False,  # operator wouldn't fetch margulis here
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        assert (ds["n_sources"].values == 3).all()
        assert "Margulis" not in ds.attrs["source"]


def test_build_invalid_fabric_token_raises(tmp_path: Path):
    """A typo like fabric.token=oregon (instead of `or`) must fail loudly
    rather than silently filter every fabric-scoped source out."""
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path, fabric_token="oregon", write_margulis=False, nn_fill=False
    )
    project = load(workdir)
    with pytest.raises(ValueError, match="fabric.token='oregon'"):
        build(project)


def test_build_unknown_source_raises(tmp_path: Path):
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        fabric_token="or",
        sources=["daymet", "not_a_real_source"],
        write_snodas=False,
        write_era5_sd=False,
        write_margulis=False,
        nn_fill=False,
    )
    project = load(workdir)
    with pytest.raises(ValueError, match="unknown source 'not_a_real_source'"):
        build(project)


def test_build_emits_id_col_sorted_target_ncs(tmp_path: Path):
    """Both unfilled and NN-filled NCs come out sorted ascending by id_col (#93)."""
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(tmp_path, fabric_token="or")
    project = load(workdir)
    build(project)
    for fname in ("swe_targets.nc", "swe_targets_nn_filled.nc"):
        with xr.open_dataset(project.targets_dir() / fname) as ds:
            ids = ds["nhm_id"].values
            assert np.all(np.diff(ids) > 0), (
                f"{fname}: HRU dim not strictly ascending; got {ids}"
            )


def test_build_nn_fill_actually_fills_nan_cells(tmp_path: Path):
    """End-to-end NN-fill: aggregated NC with NaN at one HRU/day produces
    a *_nn_filled.nc with that cell filled and nn_filled=1.
    """
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    # Single source so any NaN propagates to bound NaN.
    workdir = _make_swe_project(
        tmp_path,
        period="2003-12-01/2003-12-03",
        sources=["daymet"],
        fabric_token=None,
        write_snodas=False,
        write_era5_sd=False,
        write_margulis=False,
        nn_fill=True,
    )
    # Overwrite Daymet NC so HRU 2 is NaN at all 3 days.
    src_dir = workdir / "data" / "aggregated" / "daymet"
    times = pd.date_range("2003-12-01", "2003-12-03", freq="D")
    arr = np.full((3, 3), 50.0, dtype=np.float32)
    arr[:, 1] = np.nan
    ds = xr.Dataset(
        {"swe": (("time", "nhm_id"), arr)},
        coords={"time": times, "nhm_id": [1, 2, 3]},
    )
    (src_dir / "daymet_2003_agg.nc").unlink()
    ds.to_netcdf(src_dir / "daymet_2003_agg.nc")

    project = load(workdir)
    build(project)

    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as out:
        assert np.isnan(out["lower_bound"].values[:, 1]).all()
        assert (out["n_sources"].values[:, 1] == 0).all()

    nn_path = project.targets_dir() / "swe_targets_nn_filled.nc"
    assert nn_path.exists()
    with xr.open_dataset(nn_path) as filled:
        assert "nn_filled" in filled.data_vars
        assert np.isfinite(filled["lower_bound"].values[:, 1]).all()
        assert (filled["nn_filled"].values[:, 1] == 1).all()
        assert (filled["nn_filled"].values[:, 0] == 0).all()
        assert (filled["nn_filled"].values[:, 2] == 0).all()
