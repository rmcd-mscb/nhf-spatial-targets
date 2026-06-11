"""Tests for the SWE target builder end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml


# Map config-label → (on-disk per-source dir, aggregated variable name).
# Derived from targets/swe.py:SHIMS so there is no parallel dict to drift
# from the real registry (PR #135 review consider 4).
def _source_dirs_and_vars() -> dict[str, tuple[str, str]]:
    from nhf_spatial_targets.targets._shims import shims_by_config_label
    from nhf_spatial_targets.targets.swe import SHIMS

    return {
        label: (shim.source_key, shim.aggregated_var)
        for label, shim in shims_by_config_label(SHIMS).items()
    }


_SOURCE_DIRS_AND_VARS = _source_dirs_and_vars()


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
    write_daymet: bool = True,
    write_snodas: bool = True,
    write_era5_sd: bool = True,
    write_margulis: bool = True,
    write_ua_swe: bool = True,
    # All in their native units; the builder converts to inches at the
    # tail end. Constants chosen so multi-source min/max comes back
    # ordered (daymet < snodas < era5 < margulis < ua_swe in inches).
    daymet_value_mm: float = 50.0,  # 50 mm ≈ 1.97 in
    snodas_value_mm: float = 80.0,  # 80 mm ≈ 3.15 in
    era5_sd_value_m: float = 0.1,  # 100 mm ≈ 3.94 in
    margulis_value_m: float = 0.2,  # 200 mm ≈ 7.87 in
    ua_swe_value_mm: float = 300.0,  # 300 mm ≈ 11.81 in (kg m-2 ≡ mm)
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
        ("ua_swe", write_ua_swe, ua_swe_value_mm),
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


def test_ua_swe_to_mm_passthrough():
    """ua_swe `swe` is kg/m² ≡ mm — identity, NOT a ×1000 metres conversion."""
    from nhf_spatial_targets.targets.swe import ua_swe_to_mm

    da = xr.DataArray(
        np.array([[40.0, 300.0]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2003-12-15"]), "nhm_id": [1, 2]},
    )
    out = ua_swe_to_mm(da)
    np.testing.assert_array_equal(out.values, da.values)
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
# Availability filter (logic-level + end-to-end)
# ---------------------------------------------------------------------------


def test_availability_filter_drops_unaggregated_source(tmp_path: Path, caplog):
    """A requested source whose aggregated dir is empty is dropped with a
    WARNING; aggregated sources pass through unchanged."""
    import logging

    from nhf_spatial_targets.targets._shims import shims_by_config_label
    from nhf_spatial_targets.targets.swe import (
        SHIMS,
        _filter_sources_by_availability,
    )
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        write_margulis=False,
        nn_fill=False,
    )
    project = load(workdir)
    shims = shims_by_config_label(SHIMS)
    with caplog.at_level(logging.WARNING, logger="nhf_spatial_targets.targets.swe"):
        kept = _filter_sources_by_availability(
            project, ["daymet", "snodas", "era5_land", "margulis_wus_sr"], shims
        )
    assert kept == ["daymet", "snodas", "era5_land"]
    assert "margulis_wus_sr" in caplog.text
    assert "no aggregated NCs found" in caplog.text


def test_build_oregon_without_margulis_succeeds_with_three_sources(
    tmp_path: Path, caplog
):
    """End-to-end: all 4 sources requested, but Margulis
    has never been aggregated. Build succeeds against the other 3 with a
    WARNING; n_sources stays at 3 everywhere, source attr drops Margulis."""
    import logging

    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        write_margulis=False,
        nn_fill=False,
    )
    project = load(workdir)
    with caplog.at_level(logging.WARNING, logger="nhf_spatial_targets.targets.swe"):
        build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        assert (ds["n_sources"].values == 3).all()
        assert "Margulis" not in ds.attrs["source"]
    assert "margulis_wus_sr" in caplog.text


def test_build_all_sources_unaggregated_raises(tmp_path: Path):
    """If every requested source is unaggregated, the build raises a clear
    error pointing the operator at the right ``agg`` command."""
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        write_daymet=False,
        write_snodas=False,
        write_era5_sd=False,
        write_margulis=False,
        nn_fill=False,
    )
    project = load(workdir)
    with pytest.raises(ValueError, match="zero sources after dropping unaggregated"):
        build(project)


# ---------------------------------------------------------------------------
# End-to-end build
# ---------------------------------------------------------------------------


def test_build_writes_unfilled_and_filled_files(tmp_path: Path):
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
    )
    project = load(workdir)
    build(project)
    assert (project.targets_dir() / "swe_targets.nc").exists()
    assert (project.targets_dir() / "swe_targets_nn_filled.nc").exists()


def test_build_output_schema(tmp_path: Path):
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(tmp_path, nn_fill=False)
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
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        np.testing.assert_allclose(ds["lower_bound"].values, 50.0 / 25.4, rtol=1e-5)
        np.testing.assert_allclose(ds["upper_bound"].values, 200.0 / 25.4, rtol=1e-5)
        assert (ds["n_sources"].values == 4).all()


def test_build_ua_swe_participates_in_envelope_and_count(tmp_path: Path):
    """ua_swe (kg/m² ≡ mm, identity-to-mm) joins the min/max envelope (#237 PR-C).

    Margulis is omitted from the requested sources; the rest are
    daymet=50mm, snodas=80mm, era5=100mm, ua_swe=300mm. ua_swe is the new
    maximum, so the upper bound must move to 300/25.4 in, n_sources must
    count it (4), and its description must appear in the source attr.
    """
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        period="2003-12-15/2003-12-15",
        sources=["daymet", "snodas", "era5_land", "ua_swe"],
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        np.testing.assert_allclose(ds["lower_bound"].values, 50.0 / 25.4, rtol=1e-5)
        np.testing.assert_allclose(ds["upper_bound"].values, 300.0 / 25.4, rtol=1e-5)
        assert (ds["n_sources"].values == 4).all()
        assert "UA SWE" in ds.attrs["source"]


def test_build_five_source_envelope_with_ua_swe(tmp_path: Path):
    """All five sources requested and aggregated → n_sources=5 envelope.

    This is the shape the regenerated FGDC/MCF release fixtures advertise:
    daymet=50, snodas=80, era5=100, margulis=200, ua_swe=300 (mm/inches
    ordered). Any fabric where all five sources have aggregated coverage
    yields this; the source attr must name both Margulis and UA SWE and
    the count must reach 5 (#309: no token gate — coverage is the only
    thing that matters).
    """
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        period="2003-12-15/2003-12-15",
        sources=["daymet", "snodas", "era5_land", "margulis_wus_sr", "ua_swe"],
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        np.testing.assert_allclose(ds["lower_bound"].values, 50.0 / 25.4, rtol=1e-5)
        np.testing.assert_allclose(ds["upper_bound"].values, 300.0 / 25.4, rtol=1e-5)
        assert (ds["n_sources"].values == 5).all()
        assert "Margulis" in ds.attrs["source"]
        assert "UA SWE" in ds.attrs["source"]


def test_build_oregon_includes_margulis_in_source_attr(tmp_path: Path):
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(tmp_path, nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        src_attr = ds.attrs["source"]
        assert "Margulis" in src_attr
        assert "Daymet" in src_attr
        assert "SNODAS" in src_attr
        assert "ERA5-Land" in src_attr


def test_build_unknown_source_raises(tmp_path: Path):
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        sources=["daymet", "not_a_real_source"],
        write_snodas=False,
        write_era5_sd=False,
        write_margulis=False,
        nn_fill=False,
    )
    project = load(workdir)
    with pytest.raises(
        ValueError, match="no matching SourceShim for source 'not_a_real_source'"
    ):
        build(project)


def test_build_emits_id_col_sorted_target_ncs(tmp_path: Path):
    """Both unfilled and NN-filled NCs come out sorted ascending by id_col (#93)."""
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
    )
    project = load(workdir)
    build(project)
    for fname in ("swe_targets.nc", "swe_targets_nn_filled.nc"):
        with xr.open_dataset(project.targets_dir() / fname) as ds:
            ids = ds["nhm_id"].values
            assert np.all(np.diff(ids) > 0), (
                f"{fname}: HRU dim not strictly ascending; got {ids}"
            )


# ---------------------------------------------------------------------------
# Year-chunked build (PR #139)
# ---------------------------------------------------------------------------


def test_build_writes_per_year_intermediates(tmp_path: Path):
    """Year-chunked build leaves per-year NCs under .swe_intermediates/
    for forensic inspection after the stitch.
    """
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        period="2003-12-01/2004-01-31",  # spans two calendar years
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    inter = project.targets_dir() / ".swe_intermediates"
    assert inter.is_dir()
    year_files = sorted(inter.glob("swe_targets_*.nc"))
    assert [p.name for p in year_files] == [
        "swe_targets_2003.nc",
        "swe_targets_2004.nc",
    ]


def test_build_stitched_time_index_is_contiguous_across_year_boundary(
    tmp_path: Path,
):
    """Stitched output has every day from period_start to period_end
    with no gap at the year boundary.
    """
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        period="2003-12-30/2004-01-02",
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        times = pd.DatetimeIndex(ds["time"].values)
        expected = pd.date_range("2003-12-30", "2004-01-02", freq="D")
        assert list(times) == list(expected)
        # All 4 sources present in both years; n_sources stays at 4.
        assert (ds["n_sources"].values == 4).all()


def test_build_per_year_n_sources_varies_with_source_coverage(tmp_path: Path):
    """When SNODAS only covers 2004 (not 2003), the per-year build
    drops it for 2003 (n_sources=3) and includes it for 2004 (n_sources=4).
    Verifies the per-year period-union semantics work as advertised.
    """
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        period="2003-12-30/2004-01-02",
        nn_fill=False,
    )
    # Remove the SNODAS 2003 NC so 2003 has only 3 sources.
    snodas_2003 = workdir / "data" / "aggregated" / "snodas" / "snodas_2003_agg.nc"
    snodas_2003.unlink()

    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        ns_2003 = ds["n_sources"].sel(time="2003-12-31").values
        ns_2004 = ds["n_sources"].sel(time="2004-01-01").values
        assert (ns_2003 == 3).all(), (
            f"2003-12-31 should have 3 sources (snodas missing), got {ns_2003}"
        )
        assert (ns_2004 == 4).all(), f"2004-01-01 should have 4 sources, got {ns_2004}"


def test_build_year_chunked_idempotent_skips_existing_intermediates(
    tmp_path: Path,
):
    """A re-run after partial completion (or mid-OOM) skips per-year
    NCs that already exist. Useful for recovering from OOM mid-build
    without re-doing every year.
    """
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        period="2003-12-30/2004-01-02",
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    inter = project.targets_dir() / ".swe_intermediates"
    # Capture mtimes; re-running build must NOT re-touch them.
    pre_mtimes = {p.name: p.stat().st_mtime_ns for p in inter.glob("swe_targets_*.nc")}
    build(project)
    post_mtimes = {p.name: p.stat().st_mtime_ns for p in inter.glob("swe_targets_*.nc")}
    assert pre_mtimes == post_mtimes, (
        "Per-year intermediates were re-touched on idempotent re-build"
    )


def test_iter_period_years_clips_to_period_bounds():
    """First and last year ranges are clipped to mid-year period bounds."""
    from nhf_spatial_targets.targets._intermediates import iter_period_years

    out = iter_period_years("1980-06-15", "1982-03-20")
    assert out == [
        (1980, "1980-06-15", "1980-12-31"),
        (1981, "1981-01-01", "1981-12-31"),
        (1982, "1982-01-01", "1982-03-20"),
    ]


def test_iter_period_years_single_year():
    from nhf_spatial_targets.targets._intermediates import iter_period_years

    assert iter_period_years("2020-03-01", "2020-04-30") == [
        (2020, "2020-03-01", "2020-04-30"),
    ]


def test_iter_period_years_rejects_reversed_period():
    from nhf_spatial_targets.targets._intermediates import iter_period_years

    with pytest.raises(ValueError, match="precedes start"):
        iter_period_years("2025-01-01", "2024-12-31")


def test_stitched_output_global_attrs_carry_target_metadata(tmp_path: Path):
    """The stitch step overlays the target's `extra_global_attrs` on
    top of the per-year files' attrs, so the canonical output keeps the
    PR-#135 metadata (source, period, etc) and strips
    per-year-only attrs that would mislead about the file's scope.
    """
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        period="2003-12-30/2004-01-02",
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        assert ds.attrs["period"] == "2003-12-30/2004-01-02"
        assert "Margulis" in ds.attrs["source"]
        assert "stitched from" in ds.attrs["history"]
        # PR #139 review must-fix: year_chunk is set on every per-year
        # intermediate by _build_year; xr.open_mfdataset's default
        # combine_attrs='override' would leak the first year's value
        # into the stitched canonical file. The stitch helper must
        # pop it so the canonical file doesn't mislead about its scope.
        assert "year_chunk" not in ds.attrs, (
            "year_chunk is a per-year-intermediate attr; must not leak "
            "to the canonical stitched file"
        )

    # And: the per-year intermediates DO carry year_chunk (regression
    # guard — losing this would hide forensic info on the per-year files).
    inter = project.targets_dir() / ".swe_intermediates"
    with xr.open_dataset(inter / "swe_targets_2003.nc") as ds_2003:
        assert ds_2003.attrs.get("year_chunk") == 2003
    with xr.open_dataset(inter / "swe_targets_2004.nc") as ds_2004:
        assert ds_2004.attrs.get("year_chunk") == 2004


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


# ---------------------------------------------------------------------------
# Cache invalidation by fingerprint (#213)
# ---------------------------------------------------------------------------


def test_fingerprint_mismatch_rebuilds_year(tmp_path: Path, caplog):
    """Edit the SWE sources list between two builds → the cached
    intermediate is automatically rebuilt with the new source set,
    no manual ``rm`` needed.

    Regression guard for #213's core promise: a config change invalidates
    the cache without operator intervention.
    """
    import logging

    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    # First build: all 4 OR sources (period 2003-10 → SNODAS-era 2003+
    # so all four contribute). _make_swe_project default values produce
    # well-ordered bounds.
    workdir = _make_swe_project(
        tmp_path,
        period="2003-12-15/2003-12-15",
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        first_fp = ds.attrs["config_fingerprint"]
        # All 4 sources → n_sources=4 by construction in the test fixture.
        assert (ds["n_sources"].values == 4).all()

    # Mutate sources list: drop margulis_wus_sr. The fingerprint must
    # change, the year intermediate must be deleted, the rebuild must
    # produce n_sources=3.
    cfg_path = workdir / "config.yml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["targets"]["snow_water_equivalent"]["sources"] = [
        "daymet",
        "snodas",
        "era5_land",
    ]
    cfg_path.write_text(yaml.safe_dump(cfg))
    # Delete stitched output so we can confirm the rebuild wrote a fresh one.
    (project.targets_dir() / "swe_targets.nc").unlink()

    project = load(workdir)
    caplog.clear()
    with caplog.at_level(
        logging.WARNING, logger="nhf_spatial_targets.targets._intermediates"
    ):
        build(project)
    # Skip predicate logged a fingerprint-mismatch warning under the
    # shared _intermediates logger.
    assert any("mismatch" in r.message for r in caplog.records), (
        "fingerprint mismatch on cached intermediate must log a WARNING"
    )
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        second_fp = ds.attrs["config_fingerprint"]
        # Rebuild reflects the trimmed sources list.
        assert (ds["n_sources"].values == 3).all()
        assert "Margulis" not in ds.attrs["source"]
    assert first_fp != second_fp


def test_pre_213_intermediate_triggers_rebuild(tmp_path: Path, caplog):
    """A legacy intermediate without fingerprint attrs (from before
    #213) must be detected, deleted, and rebuilt rather than silently
    reused. Required for the upgrade transition."""
    import logging

    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        period="2003-12-15/2003-12-15",
        nn_fill=False,
    )
    # Pre-populate intermediates_dir with a fake legacy file (no fingerprint).
    intermediates_dir = workdir / "targets" / ".swe_intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = intermediates_dir / "swe_targets_2003.nc"
    legacy = xr.Dataset(
        {"placeholder": (("time",), np.array([0.0], dtype=np.float32))},
        coords={"time": pd.date_range("2003-12-15", periods=1, freq="D")},
    )
    legacy.to_netcdf(legacy_path)
    project = load(workdir)
    caplog.clear()
    with caplog.at_level(
        logging.WARNING, logger="nhf_spatial_targets.targets._intermediates"
    ):
        build(project)
    assert any(
        ("predates" in r.message or "missing" in r.message) for r in caplog.records
    )
    rebuilt = xr.open_dataset(legacy_path)
    try:
        assert "config_fingerprint" in rebuilt.attrs
        assert "lower_bound" in rebuilt.data_vars
    finally:
        rebuilt.close()


# ---------------------------------------------------------------------------
# Period-shrink defense (#211)
# ---------------------------------------------------------------------------


def test_period_shrink_prunes_orphan_intermediates(tmp_path: Path, caplog):
    """Shrinking the active period leaves out-of-period intermediates on
    disk from the wider previous build. The next build must prune them
    AND the stitched output must cover only the new (smaller) period.

    Regression guard for #211, the original symptom that motivated this
    fix: OR SWE's period shrunk from 1980-2025 to 1980-2024 and the
    stitched ``swe_targets.nc`` still covered through 2025 because the
    stitcher's directory glob silently included the orphan 2025
    intermediate.
    """
    import logging

    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    # First build: 2 years wide (2003-2004 — SNODAS era so all 4 OR
    # sources contribute).
    workdir = _make_swe_project(
        tmp_path,
        period="2003-12-01/2004-01-31",
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    intermediates_dir = project.targets_dir() / ".swe_intermediates"
    assert (intermediates_dir / "swe_targets_2003.nc").exists()
    assert (intermediates_dir / "swe_targets_2004.nc").exists()
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        years_first = sorted(set(pd.DatetimeIndex(ds["time"].values).year))
    assert years_first == [2003, 2004]

    # Shrink period: drop 2004. Delete the stitched output so we can
    # confirm the rebuild writes a fresh one matching the new period.
    cfg_path = workdir / "config.yml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["targets"]["snow_water_equivalent"]["period"] = "2003-12-01/2003-12-31"
    cfg_path.write_text(yaml.safe_dump(cfg))
    (project.targets_dir() / "swe_targets.nc").unlink()

    project = load(workdir)
    caplog.clear()
    # The prune WARNING is emitted under the caller's logger
    # (the swe module logger): the helper takes the caller's logger
    # as a parameter so the message appears under the right name in
    # operator logs.
    with caplog.at_level(logging.WARNING, logger="nhf_spatial_targets.targets.swe"):
        build(project)

    # 1. Orphan was pruned with a WARNING naming year 2004 + the
    # target label so operators can filter logs by target.
    assert any(
        "pruned" in r.message and "2004" in r.message and "swe" in r.message
        for r in caplog.records
    ), "prune WARNING must name the orphan year(s) and the target label"
    assert not (intermediates_dir / "swe_targets_2004.nc").exists()

    # 2. Stitched output covers ONLY 2003, not 2003-2004.
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        years_second = sorted(set(pd.DatetimeIndex(ds["time"].values).year))
    assert years_second == [2003], (
        f"stitched output must match shrunk period; got years {years_second}"
    )


def test_stitch_input_computed_from_period_not_glob(tmp_path: Path):
    """Defense-in-depth (parallels the SCA test by the same name):
    even if a stale orphan with a non-canonical filename slips past
    the prune regex, the stitch input must be computed from
    iter_period_years so the orphan cannot leak into the canonical
    output.
    """
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path,
        period="2003-12-15/2003-12-15",
        nn_fill=False,
    )
    project = load(workdir)
    intermediates_dir = project.targets_dir() / ".swe_intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    # 5-digit "year" — prune regex (anchored at \d{4}) won't match,
    # so the file stays on disk after the build.
    orphan = intermediates_dir / "swe_targets_99999.nc"
    orphan_ds = xr.Dataset(
        {"placeholder": (("time",), np.array([0.0], dtype=np.float32))},
        coords={"time": pd.date_range("9999-01-01", periods=1, freq="D")},
    )
    orphan_ds.to_netcdf(orphan)

    build(project)

    # Orphan survives the prune (regex didn't match).
    assert orphan.exists()
    # Stitched output excludes the orphan because the stitch input list
    # is computed from iter_period_years, not from a directory glob.
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as ds:
        years = sorted(set(pd.DatetimeIndex(ds["time"].values).year))
    assert years == [2003]


def test_build_partial_coverage_source_contributes_only_where_finite(tmp_path: Path):
    """#309: a partial-coverage source (NaN rows at uncovered HRUs, as the
    aggregation driver now emits) joins the bound only where finite; the
    bound falls back to the remaining sources elsewhere and n_sources
    drops by one at the uncovered HRU."""
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(tmp_path, period="2003-12-15/2003-12-15", nn_fill=False)
    nc = (
        workdir
        / "data"
        / "aggregated"
        / "margulis_wus_sr"
        / "margulis_wus_sr_2003_agg.nc"
    )
    with xr.open_dataset(nc) as ds:
        patched = ds.load()
    patched["SWE"].values[:, 0] = np.nan  # HRU 0 "outside the source grid"
    nc.unlink()
    patched.to_netcdf(nc)

    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as out:
        n0 = out["n_sources"].isel(nhm_id=0).values
        nrest = out["n_sources"].isel(nhm_id=slice(1, None)).values
        assert (n0 == 3).all()
        assert (nrest == 4).all()
        # margulis (200 mm) sets the upper bound only where it has data;
        # at HRU 0 the bound falls back to era5 (100 mm).
        np.testing.assert_allclose(
            out["upper_bound"].isel(nhm_id=0).values, 100.0 / 25.4, rtol=1e-5
        )
        np.testing.assert_allclose(
            out["upper_bound"].isel(nhm_id=slice(1, None)).values,
            200.0 / 25.4,
            rtol=1e-5,
        )
