"""Unit tests for notebooks/inspect_aggregated/_helpers.py.

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
HELPERS_PATH = REPO_ROOT / "notebooks" / "inspect_aggregated" / "_helpers.py"


@pytest.fixture(scope="session")
def helpers():
    spec = importlib.util.spec_from_file_location(
        "inspect_aggregated_helpers", HELPERS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_loads_with_default_save_figures_off(helpers):
    assert helpers.SAVE_FIGURES is False
    assert helpers.FIGURES_DIR == Path("docs/figures/inspect_aggregated/")


def test_unit_from_catalog_dict_variables(helpers):
    # gldas_noah_v21_monthly has dict-form variables with cf_units
    units = helpers.unit_from_catalog("gldas_noah_v21_monthly", "Qs_acc")
    assert units == "kg m-2"


def test_unit_from_catalog_flat_variables(helpers):
    # ssebop has a flat variables list and a source-level units field
    units = helpers.unit_from_catalog("ssebop", "actual_et")
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
