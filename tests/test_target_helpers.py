"""Unit tests for notebooks/targets/_helpers.py.

The helper module lives outside the package (it is a sibling of the
``inspect_target_*.ipynb`` notebooks), so we load it via importlib
rather than a regular import — same approach as
``tests/test_aggregated_helpers.py``.

Focus here is ``open_target_nc``'s on-disk time subsetting (issue #163):
the daily SWE target is ~11 GB, so the notebook subsets a single water
year before ``.load()`` rather than materialising the whole file.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPERS_PATH = REPO_ROOT / "notebooks" / "targets" / "_helpers.py"


@pytest.fixture(scope="session")
def helpers():
    spec = importlib.util.spec_from_file_location("target_helpers", HELPERS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def daily_target_nc(tmp_path: Path) -> Path:
    """A small synthetic daily target NC mirroring the SWE target schema."""
    time = pd.date_range("2009-10-01", "2011-09-30", freq="D")
    hru = np.arange(5, dtype="int64")
    shape = (time.size, hru.size)
    lower = np.zeros(shape, dtype="float32")
    upper = np.ones(shape, dtype="float32")
    n_sources = np.full(shape, 3, dtype="int8")
    ds = xr.Dataset(
        {
            "lower_bound": (("time", "nat_hru_id"), lower),
            "upper_bound": (("time", "nat_hru_id"), upper),
            "n_sources": (("time", "nat_hru_id"), n_sources),
        },
        coords={"time": time, "nat_hru_id": hru},
    )
    path = tmp_path / "swe_targets.nc"
    ds.to_netcdf(path)
    ds.close()
    return path


def test_open_target_nc_default_loads_full_range(helpers, daily_target_nc):
    """time=None is backwards-compatible: the whole file is loaded."""
    ds = helpers.open_target_nc(daily_target_nc)
    assert ds.sizes["time"] == pd.date_range("2009-10-01", "2011-09-30", freq="D").size
    # Detached from the handle: data is in memory, not a lazy/dask array.
    assert isinstance(ds["lower_bound"].data, np.ndarray)


def test_open_target_nc_tuple_window_is_inclusive(helpers, daily_target_nc):
    """A 2-tuple is treated as inclusive slice endpoints (label-based)."""
    ds = helpers.open_target_nc(daily_target_nc, time=("2009-10-01", "2010-09-30"))
    times = pd.DatetimeIndex(ds["time"].values)
    assert times.min() == pd.Timestamp("2009-10-01")
    assert times.max() == pd.Timestamp("2010-09-30")
    # WY2010 (Oct 1 2009 – Sep 30 2010) is 365 days.
    assert ds.sizes["time"] == 365
    # The windowed path is also eagerly materialised (handle detached).
    assert isinstance(ds["lower_bound"].data, np.ndarray)


def test_open_target_nc_slice_window_matches_tuple(helpers, daily_target_nc):
    """An explicit slice gives the same result as the tuple form."""
    window = ("2010-10-01", "2011-09-30")
    ds_tuple = helpers.open_target_nc(daily_target_nc, time=window)
    ds_slice = helpers.open_target_nc(daily_target_nc, time=slice(*window))
    assert ds_tuple.sizes["time"] == ds_slice.sizes["time"]
    assert (ds_tuple["time"].values == ds_slice["time"].values).all()


def test_open_target_nc_window_contains_target_date(helpers, daily_target_nc):
    """The notebook's derived water-year window contains TARGET_DATE."""
    target_date = pd.Timestamp("2010-03-01")
    wy_start = target_date.year - 1 if target_date.month < 10 else target_date.year
    window = (f"{wy_start}-10-01", f"{wy_start + 1}-09-30")
    ds = helpers.open_target_nc(daily_target_nc, time=window)
    # The at-date choropleth panels do ds.sel(time=TARGET_DATE) — must hit.
    sel = ds.sel(time=target_date)
    assert sel.sizes == {"nat_hru_id": 5}


def test_open_target_nc_subsets_before_load(helpers, daily_target_nc, monkeypatch):
    """The window is applied on-disk *before* .load() — the point of #163.

    Shape assertions alone can't catch a load-first regression
    (``ds.load().sel(...)`` returns the same window), so spy on
    ``xr.Dataset.load`` and assert it ran against the already-subset
    dataset (365 days) rather than the full file (730 days).
    """
    seen: dict[str, int] = {}
    orig_load = xr.Dataset.load

    def spy_load(self, *args, **kwargs):
        seen["time_size"] = self.sizes.get("time", 0)
        return orig_load(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "load", spy_load)
    helpers.open_target_nc(daily_target_nc, time=("2009-10-01", "2010-09-30"))
    assert seen["time_size"] == 365  # not the full 730-day file


def test_open_target_nc_window_outside_range_is_empty(helpers, daily_target_nc):
    """A window entirely outside the file range clips to an empty time dim.

    Pins the silent-empty contract: callers (e.g. an out-of-range
    TARGET_DATE) get a 0-length time axis rather than an exception here
    — the KeyError surfaces later at the at-date ``sel``, not in the
    loader.
    """
    ds = helpers.open_target_nc(daily_target_nc, time=("2030-01-01", "2030-12-31"))
    assert ds.sizes["time"] == 0


def test_open_target_nc_window_clips_to_available_range(helpers, daily_target_nc):
    """A window straddling the file's lower edge clips, not raises."""
    # File starts 2009-10-01; ask from before that.
    ds = helpers.open_target_nc(daily_target_nc, time=("2009-01-01", "2009-10-31"))
    times = pd.DatetimeIndex(ds["time"].values)
    assert times.min() == pd.Timestamp("2009-10-01")
    assert times.max() == pd.Timestamp("2009-10-31")


# --- load_fabric: parquet vs gpkg dispatch (Oregon fabric is parquet) -------


def _tiny_fabric_gdf():
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"nhm_id": [10, 20, 30]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:4326",
    )


def test_load_fabric_reads_parquet_without_gdal_plugin(helpers, tmp_path):
    """Oregon's fabric is .parquet — must use gpd.read_parquet (geopandas-
    native), not gpd.read_file. Regression for #182 render_or.slurm crash."""
    path = tmp_path / "fabric.parquet"
    _tiny_fabric_gdf().to_parquet(path)
    gdf = helpers.load_fabric({"path": str(path), "id_col": "nhm_id"})
    assert list(gdf.index) == [10, 20, 30]


def test_load_fabric_still_reads_gpkg(helpers, tmp_path):
    """gfv2's .gpkg path must keep working (read_file/pyogrio)."""
    path = tmp_path / "fabric.gpkg"
    _tiny_fabric_gdf().to_file(path, driver="GPKG")
    gdf = helpers.load_fabric({"path": str(path), "id_col": "nhm_id"})
    assert list(gdf.index) == [10, 20, 30]


# --- load_representative_points: per-project override (#182) ----------------


def _write_or_config(tmp_path, **extra_blocks):
    cfg = {"fabric": {"path": "x", "id_col": "nhm_id"}, "datastore": "/x"}
    cfg.update(extra_blocks)
    (tmp_path / "config.yml").write_text(yaml.safe_dump(cfg))
    return tmp_path


def test_load_representative_points_reads_per_target_map(helpers, tmp_path):
    project_dir = _write_or_config(
        tmp_path,
        representative_points={
            "swe": {"Mt Hood": [-121.7, 45.4], "Steens": [-118.6, 42.7]},
        },
    )
    pts = helpers.load_representative_points(project_dir, "swe")
    assert pts == {"Mt Hood": (-121.7, 45.4), "Steens": (-118.6, 42.7)}
    assert all(
        isinstance(v, tuple) and all(isinstance(x, float) for x in v)
        for v in pts.values()
    )


def test_load_representative_points_returns_none_when_target_absent(helpers, tmp_path):
    project_dir = _write_or_config(
        tmp_path, representative_points={"aet": {"X": [-120.0, 45.0]}}
    )
    assert helpers.load_representative_points(project_dir, "swe") is None


def test_load_representative_points_returns_none_without_block(helpers, tmp_path):
    project_dir = _write_or_config(tmp_path)
    assert helpers.load_representative_points(project_dir, "aet") is None


def test_load_representative_points_returns_none_when_config_missing(helpers, tmp_path):
    assert helpers.load_representative_points(tmp_path, "aet") is None


# --------------------------------------------------------------------------
# Ensemble member helpers (issue #351)
# --------------------------------------------------------------------------


@pytest.fixture
def member_target_nc(tmp_path: Path) -> Path:
    """A monthly target NC carrying three ensemble members.

    Mirrors the runoff target schema written by
    ``targets/_writers.write_bounds_target`` with ``emit_members=True``:
    bounds + ``n_sources`` + one variable per source key + the derived
    ``ensemble_mean`` / ``ensemble_std``, stamped with ``members_emitted``
    and ``member_keys``.
    """
    time = pd.date_range("2005-01-01", periods=4, freq="MS")
    hru = np.arange(3, dtype="int64")
    shape = (time.size, hru.size)

    def _const(value: float) -> np.ndarray:
        return np.full(shape, value, dtype="float32")

    ds = xr.Dataset(
        {
            "lower_bound": (("time", "nhm_id"), _const(1.0)),
            "upper_bound": (("time", "nhm_id"), _const(3.0)),
            "n_sources": (("time", "nhm_id"), np.full(shape, 3, dtype="int8")),
            "ensemble_mean": (("time", "nhm_id"), _const(2.0)),
            "ensemble_std": (("time", "nhm_id"), _const(0.8)),
            "era5_land": (("time", "nhm_id"), _const(1.0)),
            "gldas_noah_v21_monthly": (("time", "nhm_id"), _const(2.0)),
            "mwbm_climgrid": (("time", "nhm_id"), _const(3.0)),
        },
        coords={"time": time, "nhm_id": hru},
        attrs={
            "members_emitted": "true",
            "member_keys": "era5_land,gldas_noah_v21_monthly,mwbm_climgrid",
        },
    )
    path = tmp_path / "runoff_targets.nc"
    ds.to_netcdf(path)
    ds.close()
    return path


def test_member_keys_reads_the_member_keys_attr(helpers, member_target_nc):
    with xr.open_dataset(member_target_nc) as ds:
        assert helpers.member_keys(ds) == [
            "era5_land",
            "gldas_noah_v21_monthly",
            "mwbm_climgrid",
        ]


def test_member_keys_empty_when_members_not_emitted(helpers, member_target_nc):
    """The SCA case: bounds are a CI, not a member min/max."""
    with xr.open_dataset(member_target_nc) as ds:
        ds.attrs["members_emitted"] = "false"
        assert helpers.member_keys(ds) == []


def test_member_keys_empty_when_attrs_absent(helpers, member_target_nc):
    """A pre-#338 target NC carries neither attr."""
    with xr.open_dataset(member_target_nc) as ds:
        ds.attrs.pop("members_emitted")
        ds.attrs.pop("member_keys")
        assert helpers.member_keys(ds) == []


def test_member_keys_drops_keys_with_no_matching_variable(helpers, member_target_nc):
    """Guard a truncated file: the attr promises a var that is not there."""
    with xr.open_dataset(member_target_nc) as ds:
        ds.attrs["member_keys"] = "era5_land,not_on_disk"
        with pytest.warns(UserWarning, match="not_on_disk"):
            assert helpers.member_keys(ds) == ["era5_land"]


def test_member_colors_assigns_one_validated_hue_per_key(helpers):
    keys = ["snodas", "era5_land", "margulis_wus_sr", "ua_swe"]
    colors = helpers.member_colors(keys)
    assert list(colors) == keys
    assert len(set(colors.values())) == 4
    assert all(c.startswith("#") for c in colors.values())


def test_member_colors_is_deterministic(helpers):
    keys = ["a", "b", "c"]
    assert helpers.member_colors(keys) == helpers.member_colors(keys)


def test_member_colors_warns_past_the_all_pairs_safe_count(helpers):
    """Slots 5+ are not all-pairs colorblind-separable; warn, do not cycle."""
    with pytest.warns(UserWarning, match="all-pairs"):
        colors = helpers.member_colors([f"s{i}" for i in range(5)])
    assert len(set(colors.values())) == 5


def test_member_colors_rejects_more_keys_than_validated_hues(helpers):
    with pytest.raises(ValueError, match="9"):
        helpers.member_colors([f"s{i}" for i in range(9)])


def test_member_frame_at_time_is_hru_by_member(helpers, member_target_nc):
    with xr.open_dataset(member_target_nc) as ds:
        keys = helpers.member_keys(ds)
        frame = helpers.member_frame_at_time(ds, keys, "2005-01-01", "nhm_id")
    assert list(frame.columns) == keys
    assert len(frame) == 3
    assert frame["mwbm_climgrid"].iloc[0] == pytest.approx(3.0)


def test_member_frame_at_hru_is_time_by_member(helpers, member_target_nc):
    with xr.open_dataset(member_target_nc) as ds:
        keys = helpers.member_keys(ds)
        frame = helpers.member_frame_at_hru(ds, keys, 1, "nhm_id")
    assert list(frame.columns) == keys
    assert len(frame) == 4
    assert isinstance(frame.index, pd.DatetimeIndex)


def test_member_argextreme_picks_the_driving_member(helpers):
    frame = pd.DataFrame(
        {"a": [1.0, 5.0], "b": [2.0, 1.0], "c": [3.0, 2.0]}, index=[10, 11]
    )
    codes = helpers.member_argextreme(frame, how="max")
    assert codes.tolist() == [2, 0]


def test_member_argextreme_min_picks_the_lower_driver(helpers):
    frame = pd.DataFrame(
        {"a": [1.0, 5.0], "b": [2.0, 1.0], "c": [3.0, 2.0]}, index=[10, 11]
    )
    codes = helpers.member_argextreme(frame, how="min")
    assert codes.tolist() == [0, 1]


def test_member_argextreme_flags_ties_instead_of_inventing_a_driver(helpers):
    """The summer-SWE case: every source reads 0.0, so nothing drives.

    ``np.argmax`` would silently return member 0 and paint the whole
    map as that source's, which is the bug this code exists to avoid.
    """
    frame = pd.DataFrame({"a": [0.0], "b": [0.0], "c": [0.0]}, index=[10])
    codes = helpers.member_argextreme(frame, how="max")
    assert codes.tolist() == [helpers.NO_SPREAD_CODE]


def test_member_argextreme_flags_single_source_cells(helpers):
    """One finite member is a coverage story, not a driver story."""
    frame = pd.DataFrame({"a": [np.nan], "b": [4.0], "c": [np.nan]}, index=[10])
    codes = helpers.member_argextreme(frame, how="max")
    assert codes.tolist() == [helpers.SINGLE_SOURCE_CODE]


def test_member_argextreme_is_nan_where_no_member_is_finite(helpers):
    frame = pd.DataFrame({"a": [np.nan], "b": [np.nan], "c": [np.nan]}, index=[10])
    codes = helpers.member_argextreme(frame, how="max")
    assert bool(np.isnan(codes.iloc[0]))


def test_member_argextreme_ignores_nan_members_when_ranking(helpers):
    frame = pd.DataFrame({"a": [1.0], "b": [np.nan], "c": [3.0]}, index=[10])
    codes = helpers.member_argextreme(frame, how="max")
    assert codes.tolist() == [2]


def test_member_argextreme_rejects_an_unknown_how(helpers):
    frame = pd.DataFrame({"a": [1.0]}, index=[10])
    with pytest.raises(ValueError, match="how"):
        helpers.member_argextreme(frame, how="median")


@pytest.fixture
def tiny_fabric():
    """A 3-cell fabric; enough geometry for the plotting helpers."""
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"geometry": [box(i, 0, i + 1, 1) for i in range(3)]},
        index=pd.Index([0, 1, 2], name="nhm_id"),
        crs="EPSG:4326",
    )


def test_plot_hru_choropleth_labels_the_nan_class_when_asked(helpers, tiny_fabric):
    """ensemble_std's grey needs a reason, not a mystery."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    helpers.plot_hru_choropleth(
        ax,
        tiny_fabric,
        pd.Series([1.0, np.nan, 3.0], index=tiny_fabric.index),
        nan_label="n_sources < 2",
    )
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close(fig)
    assert any("n_sources < 2" in t for t in legend_texts)
    assert any("1" in t for t in legend_texts)  # the count of masked HRUs


def test_plot_hru_choropleth_has_no_nan_legend_by_default(helpers, tiny_fabric):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    helpers.plot_hru_choropleth(
        ax,
        tiny_fabric,
        pd.Series([1.0, np.nan, 3.0], index=tiny_fabric.index),
    )
    legend = ax.get_legend()
    plt.close(fig)
    assert legend is None


def test_plot_member_panels_shares_one_color_scale_across_panels(helpers, tiny_fabric):
    """A per-panel scale would make the members look falsely alike."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = {
        "era5_land": pd.Series([1.0, 2.0, 3.0], index=tiny_fabric.index),
        "snodas": pd.Series([10.0, 20.0, 30.0], index=tiny_fabric.index),
    }
    fig, (vmin, vmax) = helpers.plot_member_panels(tiny_fabric, panels, units="mm")
    plt.close(fig)
    # Pooled across both panels: had each panel autoscaled, era5_land's
    # scale would top out near 3.0 and the two members would render as
    # near-identical maps despite differing by an order of magnitude.
    assert vmax > 3.0
    assert vmin < 10.0


def test_plot_member_panels_makes_one_panel_per_member(helpers, tiny_fabric):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = {
        k: pd.Series([1.0, 2.0, 3.0], index=tiny_fabric.index)
        for k in ("a", "b", "c", "d", "e")
    }
    fig, _ = helpers.plot_member_panels(tiny_fabric, panels, units="mm", ncols=3)
    titled = [ax.get_title() for ax in fig.axes if ax.get_title()]
    plt.close(fig)
    assert set(panels) <= set(titled)


def test_plot_member_panels_rejects_an_empty_panel_map(helpers, tiny_fabric):
    with pytest.raises(ValueError, match="no panels"):
        helpers.plot_member_panels(tiny_fabric, {}, units="mm")


def test_member_frame_at_time_collapses_a_multi_match_selection(
    helpers, member_target_nc
):
    """A partial time string may match >1 step; take the first, like select_month.

    Lets every notebook use one idiom regardless of whether its target
    is annual, monthly or daily.
    """
    with xr.open_dataset(member_target_nc) as ds:
        keys = helpers.member_keys(ds)
        frame = helpers.member_frame_at_time(ds, keys, "2005", "nhm_id")
    assert list(frame.columns) == keys
    assert len(frame) == 3
    assert frame.index.name == "nhm_id"


def test_plot_member_panels_draws_exactly_one_shared_colorbar(helpers, tiny_fabric):
    """One scale must show one colorbar; three would imply three scales."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = {
        k: pd.Series([1.0, 2.0, 3.0], index=tiny_fabric.index) for k in ("a", "b", "c")
    }
    fig, _ = helpers.plot_member_panels(tiny_fabric, panels, units="mm", ncols=3)
    n_axes = len(fig.axes)
    plt.close(fig)
    assert n_axes == len(panels) + 1


def test_plot_hru_choropleth_labels_projected_axes_in_metres(helpers):
    """The Oregon fabric is EPSG:5070 Albers — 'Longitude' would be a lie."""
    import geopandas as gpd
    import matplotlib
    from shapely.geometry import box

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gdf = gpd.GeoDataFrame(
        {"geometry": [box(i, 0, i + 1, 1) for i in range(3)]},
        index=pd.Index([0, 1, 2], name="nhm_id"),
        crs="EPSG:5070",
    )
    fig, ax = plt.subplots()
    helpers.plot_hru_choropleth(ax, gdf, pd.Series([1.0, 2.0, 3.0], index=gdf.index))
    xlabel, ylabel = ax.get_xlabel(), ax.get_ylabel()
    plt.close(fig)
    assert "Longitude" not in xlabel
    assert "Easting" in xlabel and "m" in xlabel
    assert "Northing" in ylabel


def test_plot_hru_choropleth_still_says_longitude_for_geographic_crs(
    helpers, tiny_fabric
):
    """gfv2's fabric is EPSG:4326 — degrees really are lon/lat."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    helpers.plot_hru_choropleth(
        ax, tiny_fabric, pd.Series([1.0, 2.0, 3.0], index=tiny_fabric.index)
    )
    xlabel = ax.get_xlabel()
    plt.close(fig)
    assert xlabel == "Longitude"


def test_member_argextreme_treats_negligible_spread_as_no_spread(helpers):
    """Float noise must not be promoted to a driver.

    Measured on the real Oregon SWE target: on 2010-08-15, 14239 of
    16814 multi-source cells carry a spread strictly between 0 and
    1e-9 inches, against a field max of 108 inches — physically
    snow-free cells that unit conversion left numerically unequal. An
    exact ``spread == 0`` test catches only 239 of them, so ~85% of the
    state would be attributed to a driver chosen by a nanometre of SWE.
    """
    frame = pd.DataFrame(
        {"a": [0.0, 50.0], "b": [1e-12, 10.0], "c": [-2.9e-23, 30.0]},
        index=[10, 11],
    )
    codes = helpers.member_argextreme(frame, how="max")
    assert codes.iloc[0] == helpers.NO_SPREAD_CODE  # noise, not a driver
    assert codes.iloc[1] == 0  # a real 40-unit spread still resolves


def test_member_argextreme_tolerance_scales_with_the_field(helpers):
    """The same absolute spread is meaningful in a small field, noise in a big one."""
    small = pd.DataFrame({"a": [0.0], "b": [1e-4]}, index=[10])
    big = pd.DataFrame({"a": [0.0, 1e6], "b": [1e-4, 0.0]}, index=[10, 11])
    assert helpers.member_argextreme(small, how="max").iloc[0] == 1
    assert helpers.member_argextreme(big, how="max").iloc[0] == helpers.NO_SPREAD_CODE


def test_member_argextreme_rtol_zero_restores_exact_comparison(helpers):
    frame = pd.DataFrame({"a": [0.0], "b": [1e-12]}, index=[10])
    assert helpers.member_argextreme(frame, how="max", rtol=0.0).iloc[0] == 1
