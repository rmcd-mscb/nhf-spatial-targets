"""Unit tests for notebooks/aggregated/_helpers.py.

The helper module lives outside the package, so we load it via importlib
rather than a regular import.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPERS_PATH = REPO_ROOT / "notebooks" / "aggregated" / "_helpers.py"


@pytest.fixture(scope="session")
def helpers():
    spec = importlib.util.spec_from_file_location("aggregated_helpers", HELPERS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_loads_with_default_save_figures_off(helpers):
    assert helpers.SAVE_FIGURES is False
    assert helpers.FIGURES_DIR == Path("docs/figures/aggregated/")


def test_unit_from_catalog_dict_variables(helpers):
    # gldas_noah_v21_monthly has dict-form variables with cf_units
    units = helpers.unit_from_catalog("gldas_noah_v21_monthly", "Qs_acc")
    assert units == "kg m-2"


def test_unit_from_catalog_flat_variables(helpers):
    # ssebop has a flat variables list and a source-level units field
    units = helpers.unit_from_catalog("ssebop", "et")
    assert units == "mm/month"


def test_unit_from_catalog_unknown_variable_raises(helpers):
    with pytest.raises(KeyError):
        helpers.unit_from_catalog("ssebop", "nonexistent_var")


def test_unit_from_catalog_file_variable_lookup(helpers):
    # reitz2017: name=total_recharge, file_variable=TotalRecharge
    # Caller passes the on-disk variable name; should resolve via file_variable path
    units = helpers.unit_from_catalog("reitz2017", "TotalRecharge")
    assert units == "m yr-1"


def test_unit_from_catalog_cf_units_takes_precedence(helpers):
    # reitz2017 total_recharge: cf_units="m yr-1", units="m/year" — cf_units wins
    units = helpers.unit_from_catalog("reitz2017", "total_recharge")
    assert units == "m yr-1"


def test_select_month_start_of_month(helpers):
    # GLDAS/NLDAS convention: start-of-month
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    da = xr.DataArray(np.arange(12), coords={"time": times}, dims=["time"])
    result = helpers.select_month(da, 2000, 3)
    assert pd.Timestamp(result.time.values) == pd.Timestamp("2000-03-01")
    assert int(result.values) == 2  # March = index 2


def test_select_month_end_of_month(helpers):
    # NCEP/NCAR convention: end-of-month — "nearest" to mid-month silently
    # picks the wrong calendar month here; select_month gets it right.
    times = pd.date_range("2000-01-31", periods=12, freq="ME")
    da = xr.DataArray(np.arange(12), coords={"time": times}, dims=["time"])
    result = helpers.select_month(da, 2000, 3)
    assert pd.Timestamp(result.time.values) == pd.Timestamp("2000-03-31")
    assert int(result.values) == 2


def test_select_month_mid_month(helpers):
    # MERRA-2 convention: mid-month
    times = pd.DatetimeIndex(
        [pd.Timestamp(year=2000, month=m, day=15) for m in range(1, 13)]
    )
    da = xr.DataArray(np.arange(12), coords={"time": times}, dims=["time"])
    result = helpers.select_month(da, 2000, 3)
    assert pd.Timestamp(result.time.values) == pd.Timestamp("2000-03-15")


def test_select_month_no_data_raises(helpers):
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    da = xr.DataArray(np.arange(12), coords={"time": times}, dims=["time"])
    with pytest.raises(IndexError):
        helpers.select_month(da, 2099, 1)


def test_discover_aggregated_returns_sorted_paths(helpers, tmp_path):
    src = "era5_land"
    agg_dir = tmp_path / "data" / "aggregated" / src
    agg_dir.mkdir(parents=True)
    (agg_dir / f"{src}_2002_agg.nc").touch()
    (agg_dir / f"{src}_2000_agg.nc").touch()
    (agg_dir / f"{src}_2001_agg.nc").touch()

    paths = helpers.discover_aggregated(tmp_path, src)
    assert paths is not None
    assert [p.name for p in paths] == [
        f"{src}_2000_agg.nc",
        f"{src}_2001_agg.nc",
        f"{src}_2002_agg.nc",
    ]


def test_discover_aggregated_returns_none_when_dir_missing(helpers, tmp_path):
    assert helpers.discover_aggregated(tmp_path, "no_such_source") is None


def test_discover_aggregated_returns_none_when_dir_empty(helpers, tmp_path):
    (tmp_path / "data" / "aggregated" / "src1").mkdir(parents=True)
    assert helpers.discover_aggregated(tmp_path, "src1") is None


def test_load_project_paths_reads_config_yml(helpers, tmp_path):
    cfg = {
        "datastore": "/mnt/d/nhf-datastore",
        "fabric": {
            "path": "/mnt/d/fabric/gfv2.gpkg",
            "id_col": "nhm_id",
            "crs": "EPSG:4326",
        },
    }
    (tmp_path / "config.yml").write_text(yaml.safe_dump(cfg))

    project_dir, datastore_dir, fabric_cfg = helpers.load_project_paths(tmp_path)
    assert project_dir == tmp_path
    assert datastore_dir == Path("/mnt/d/nhf-datastore")
    assert fabric_cfg["path"] == "/mnt/d/fabric/gfv2.gpkg"
    assert fabric_cfg["id_col"] == "nhm_id"


def test_load_project_paths_missing_config_raises(helpers, tmp_path):
    with pytest.raises(FileNotFoundError):
        helpers.load_project_paths(tmp_path)


def _three_hru_fabric() -> gpd.GeoDataFrame:
    """Three square HRUs in EPSG:4326 with known relative areas."""
    geoms = [
        box(-100, 40, -99, 41),  # ~111 x 85 km
        box(-99, 40, -98, 41),
        box(-98, 40, -97, 41),
    ]
    return gpd.GeoDataFrame(
        {"hru_id": [1, 2, 3]},
        geometry=geoms,
        crs="EPSG:4326",
    ).set_index("hru_id")


def test_area_weighted_mean_equal_areas(helpers):
    fabric = _three_hru_fabric()
    values = pd.Series([10.0, 20.0, 30.0], index=[1, 2, 3])
    result = helpers.area_weighted_mean(values, fabric)
    # Areas in EPSG:5070 are very nearly equal here; expect ~20.0
    assert abs(result - 20.0) < 0.5


def test_area_weighted_mean_skips_nan(helpers):
    fabric = _three_hru_fabric()
    values = pd.Series([10.0, np.nan, 30.0], index=[1, 2, 3])
    result = helpers.area_weighted_mean(values, fabric)
    # Should average HRUs 1 and 3 only
    assert abs(result - 20.0) < 0.5


def test_nan_hru_count(helpers):
    values = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
    assert helpers.nan_hru_count(values) == 2


def test_lookup_hrus_by_points_resolves_inside(helpers):
    fabric = _three_hru_fabric()
    points = {
        "left": (-99.5, 40.5),  # inside HRU 1
        "middle": (-98.5, 40.5),  # inside HRU 2
        "right": (-97.5, 40.5),  # inside HRU 3
    }
    result = helpers.lookup_hrus_by_points(fabric, points)
    assert result == {"left": 1, "middle": 2, "right": 3}


def test_lookup_hrus_by_points_outside_raises(helpers):
    fabric = _three_hru_fabric()
    points = {"in_ocean": (-50.0, 40.5)}
    with pytest.raises(ValueError):
        helpers.lookup_hrus_by_points(fabric, points)


def _toy_sca_dataset():
    """Build a 5-day × 4-HRU dataset that mimics MOD10C1 schema enough to
    exercise daily_coverage_summary / find_best_day."""
    times = pd.date_range("2010-01-15", periods=5, freq="D")
    sca = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],  # day 0: all NaN
            [10.0, 0.0, np.nan, np.nan],  # day 1: 1 above threshold, 1 zero
            [50.0, 30.0, 20.0, np.nan],  # day 2: 3 above
            [70.0, 60.0, 50.0, 40.0],  # day 3: 4 above (full coverage)
            [10.0, 0.0, 0.0, 0.0],  # day 4: 1 above
        ]
    )
    vaf = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],  # day 0: no covered HRUs
            [0.9, 0.9, 0.0, 0.0],  # day 1: 2 covered
            [0.9, 0.5, 0.9, 0.0],  # day 2: 2 covered (HRU1 below threshold)
            [0.9, 0.9, 0.9, 0.9],  # day 3: 4 covered (the BEST day)
            [0.9, 0.9, 0.9, 0.9],  # day 4: 4 covered, but only 1 above threshold
        ]
    )
    return xr.Dataset(
        {
            "sca": (["time", "hru_id"], sca),
            "vaf": (["time", "hru_id"], vaf),
        },
        coords={"time": times, "hru_id": [0, 1, 2, 3]},
    )


def test_daily_coverage_summary_counts_finite_above_and_overlap(helpers):
    ds = _toy_sca_dataset()
    df = helpers.daily_coverage_summary(
        ds, "sca", coverage_var="vaf", coverage_threshold=0.7
    )
    assert list(df.columns) == ["n_finite", "n_above", "n_covered", "n_overlap"]
    assert df["n_finite"].tolist() == [0, 2, 3, 4, 4]
    assert df["n_above"].tolist() == [0, 1, 3, 4, 1]
    assert df["n_covered"].tolist() == [0, 2, 2, 4, 4]
    # day 3: all 4 HRUs above threshold AND covered. Day 4: only 1 above.
    assert df["n_overlap"].tolist() == [0, 1, 2, 4, 1]


def test_daily_coverage_summary_omits_coverage_columns_when_no_var(helpers):
    ds = _toy_sca_dataset()
    df = helpers.daily_coverage_summary(ds, "sca")
    assert list(df.columns) == ["n_finite", "n_above"]


def test_find_best_day_picks_overlap_max(helpers):
    ds = _toy_sca_dataset()
    best = helpers.find_best_day(ds, "sca", coverage_var="vaf")
    assert best == pd.Timestamp("2010-01-18")  # day 3, n_overlap=4


def test_find_best_day_falls_back_to_n_above_without_coverage(helpers):
    ds = _toy_sca_dataset()
    best = helpers.find_best_day(ds, "sca")
    # Without coverage_var, ranks by n_above. Day 3 (4) is still the max.
    assert best == pd.Timestamp("2010-01-18")


def test_find_best_day_respects_month_filter(helpers):
    """Add a Feb day with smaller overlap; month filter should pick that."""
    ds = _toy_sca_dataset()
    feb_times = pd.date_range("2010-02-10", periods=2, freq="D")
    feb_sca = np.array([[5.0, np.nan, np.nan, np.nan], [np.nan] * 4])
    feb_vaf = np.array([[0.9, 0.0, 0.0, 0.0], [0.0] * 4])
    feb_ds = xr.Dataset(
        {
            "sca": (["time", "hru_id"], feb_sca),
            "vaf": (["time", "hru_id"], feb_vaf),
        },
        coords={"time": feb_times, "hru_id": [0, 1, 2, 3]},
    )
    combined = xr.concat([ds, feb_ds], dim="time")
    best_feb = helpers.find_best_day(combined, "sca", coverage_var="vaf", month=2)
    assert best_feb == pd.Timestamp("2010-02-10")


def test_find_best_day_raises_when_month_has_no_data(helpers):
    ds = _toy_sca_dataset()  # all timestamps in Jan
    with pytest.raises(ValueError, match="month 7"):
        helpers.find_best_day(ds, "sca", month=7)


def test_save_figure_no_op_when_disabled(helpers, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(helpers, "SAVE_FIGURES", False)
    monkeypatch.setattr(helpers, "FIGURES_DIR", tmp_path / "figures")
    fig = plt.figure()
    helpers.save_figure(fig, "test_disabled")
    plt.close(fig)
    assert not (tmp_path / "figures").exists()


def test_save_figure_writes_png_when_enabled(helpers, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(helpers, "SAVE_FIGURES", True)
    monkeypatch.setattr(helpers, "FIGURES_DIR", tmp_path / "figures")
    fig = plt.figure()
    helpers.save_figure(fig, "test_enabled")
    plt.close(fig)
    assert (tmp_path / "figures" / "test_enabled.png").exists()


def test_save_figure_writes_under_project_subdir(helpers, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(helpers, "SAVE_FIGURES", True)
    monkeypatch.setattr(helpers, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(helpers, "PROJECT", "gfv2-spatial-targets")
    fig = plt.figure()
    helpers.save_figure(fig, "test_project")
    plt.close(fig)
    assert (tmp_path / "figures" / "gfv2-spatial-targets" / "test_project.png").exists()


def test_save_figure_no_subdir_when_project_none(helpers, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(helpers, "SAVE_FIGURES", True)
    monkeypatch.setattr(helpers, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(helpers, "PROJECT", None)
    fig = plt.figure()
    helpers.save_figure(fig, "test_no_project")
    plt.close(fig)
    assert (tmp_path / "figures" / "test_no_project.png").exists()
