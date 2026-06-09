"""Tests for the UA SWE aggregator (``aggregate/ua_swe.py``).

Covers the derived ``snow_covered_fraction`` binary hook (the NaN policy
that is the OPPOSITE of mod10c1's ``valid_mask`` — fill pixels must become
NaN, never a phantom 0.0), the depth-threshold provenance stamp, the static
adapter contract, the post-aggregate re-stamp, and config-driven threshold
resolution. Heavy gdptools aggregation is left to the integration suite;
these are pure-function unit tests on the hooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nhf_spatial_targets import catalog
from nhf_spatial_targets.aggregate.ua_swe import (
    ADAPTER,
    DEFAULT_DEPTH_THRESHOLD_MM,
    aggregate_ua_swe,
    build_adapter,
    make_post_aggregate_hook,
)


@pytest.fixture()
def raw_ua_swe():
    """One-day synthetic UA grid: snowy, bare, and off-CONUS fill (NaN) pixels.

    snow_depth in mm; 2x3 grid. Row 0: 50, 50, 0 (two snowy, one bare).
    Row 1: 0, NaN, NaN (one bare, two fill).
    """
    times = pd.date_range("2000-01-15", periods=1, freq="D")
    depth = np.array([[[50.0, 50.0, 0.0], [0.0, np.nan, np.nan]]])
    swe = np.array([[[10.0, 10.0, 0.0], [0.0, np.nan, np.nan]]])
    return xr.Dataset(
        {
            "swe": (["time", "y", "x"], swe),
            "snow_depth": (["time", "y", "x"], depth),
        },
        coords={"time": times, "y": [0.0, 4000.0], "x": [0.0, 4000.0, 8000.0]},
    )


def test_adapter_static_contract():
    assert ADAPTER.source_key == "ua_swe"
    assert ADAPTER.output_name == "ua_swe_agg.nc"
    assert ADAPTER.variables == ("swe", "snow_depth", "snow_covered_fraction")
    assert ADAPTER.source_crs == "EPSG:5070"
    assert ADAPTER.files_glob == "daily/ua_swe_daily_*.nc"
    assert ADAPTER.stat_method == "masked_mean"
    assert ADAPTER.output_cadence == "daily"
    assert ADAPTER.pre_aggregate_hook is not None
    assert ADAPTER.post_aggregate_hook is not None
    # grid_variable defaults to "swe" (a genuine raw var), so raw_grid_variable
    # resolves correctly without an explicit override.
    assert ADAPTER.grid_variable == "swe"
    assert ADAPTER.raw_grid_variable == "swe"


def test_snow_covered_fraction_is_a_catalog_variable():
    """CF attrs the hook stamps must line up with the catalog declaration."""
    names = {v.get("name") for v in catalog.source("ua_swe")["variables"]}
    assert "snow_covered_fraction" in names


def test_hook_derives_binary_with_nan_on_fill(raw_ua_swe):
    out = ADAPTER.pre_aggregate_hook(raw_ua_swe)
    assert "snow_covered_fraction" in out.data_vars
    scf = out["snow_covered_fraction"].isel(time=0).values
    assert scf[0, 0] == 1.0  # depth 50 > 1
    assert scf[0, 1] == 1.0  # depth 50 > 1
    assert scf[0, 2] == 0.0  # depth 0 -> 0 (real "no snow")
    assert scf[1, 0] == 0.0  # depth 0 -> 0
    assert np.isnan(scf[1, 1])  # fill -> NaN (NOT a phantom 0.0)
    assert np.isnan(scf[1, 2])  # fill -> NaN
    # swe / snow_depth pass through untouched.
    np.testing.assert_array_equal(
        out["snow_depth"].values, raw_ua_swe["snow_depth"].values
    )


def test_all_nan_depth_hru_yields_nan_not_zero():
    """THE load-bearing test: an all-fill footprint must be NaN, never 0.

    Without ``.where(snow_depth.notnull())``, ``NaN > 1`` is False, so every
    fill pixel would become a hard 0.0 "no snow" vote and an all-fill HRU
    would read 0% covered instead of unobserved.
    """
    times = pd.date_range("2000-01-15", periods=1, freq="D")
    depth = np.full((1, 2, 2), np.nan)
    ds = xr.Dataset(
        {"snow_depth": (["time", "y", "x"], depth)},
        coords={"time": times, "y": [0.0, 4000.0], "x": [0.0, 4000.0]},
    )
    scf = ADAPTER.pre_aggregate_hook(ds)["snow_covered_fraction"].values
    assert np.isnan(scf).all()
    assert not (scf == 0.0).any()


def test_half_snow_footprint_mean_is_half():
    """Pre-aggregation binary mean over a half-snowy footprint is 0.5.

    This is the count-of-snowy-pixels-before-averaging quantity that makes
    the binary belong in the hook rather than in targets/sca.py.
    """
    times = pd.date_range("2000-01-15", periods=1, freq="D")
    depth = np.array([[[50.0, 50.0, 0.0, 0.0]]])
    ds = xr.Dataset(
        {"snow_depth": (["time", "y", "x"], depth)},
        coords={"time": times, "y": [0.0], "x": [0.0, 4000.0, 8000.0, 12000.0]},
    )
    scf = ADAPTER.pre_aggregate_hook(ds)["snow_covered_fraction"].isel(time=0).values
    assert np.nanmean(scf) == 0.5


def test_depth_threshold_attr_stamped_default(raw_ua_swe):
    scf = ADAPTER.pre_aggregate_hook(raw_ua_swe)["snow_covered_fraction"]
    assert scf.attrs["depth_threshold_mm"] == DEFAULT_DEPTH_THRESHOLD_MM


def test_cf_var_attrs_match_catalog(raw_ua_swe):
    """(d) CF-1.6: snow_covered_fraction carries catalog units/long_name/cell_methods."""
    cat_scf = next(
        v
        for v in catalog.source("ua_swe")["variables"]
        if v.get("name") == "snow_covered_fraction"
    )
    attrs = ADAPTER.pre_aggregate_hook(raw_ua_swe)["snow_covered_fraction"].attrs
    assert attrs["units"] == cat_scf["cf_units"]
    assert attrs["long_name"] == cat_scf["long_name"]
    assert attrs["cell_methods"] == cat_scf["cell_methods"]


def test_custom_threshold_closure_flips_binary_and_stamp():
    """build_adapter(5.0) must gate at 5 mm and stamp 5.0, proving the closure."""
    times = pd.date_range("2000-01-15", periods=1, freq="D")
    depth = np.array([[[3.0, 10.0]]])  # 3 mm: bare at t=5; 10 mm: snowy at t=5
    ds = xr.Dataset(
        {"snow_depth": (["time", "y", "x"], depth)},
        coords={"time": times, "y": [0.0], "x": [0.0, 4000.0]},
    )
    out = build_adapter(5.0).pre_aggregate_hook(ds)["snow_covered_fraction"]
    vals = out.isel(time=0).values.ravel()
    assert vals[0] == 0.0  # 3 > 5 is False
    assert vals[1] == 1.0  # 10 > 5 is True
    assert out.attrs["depth_threshold_mm"] == 5.0


def test_post_aggregate_hook_restamps_attrs():
    """gdptools may drop synthesized-var attrs; the post hook re-asserts them."""
    hru = xr.Dataset(
        {"snow_covered_fraction": (["nhm_id"], [0.5, 0.2])},
        coords={"nhm_id": [1, 2]},
    )
    hru["snow_covered_fraction"].attrs = {}  # simulate attr loss through agg
    out = make_post_aggregate_hook(1.0)(hru)
    assert out["snow_covered_fraction"].attrs["depth_threshold_mm"] == 1.0
    assert out["snow_covered_fraction"].attrs["units"] == "1"


def test_aggregate_ua_swe_reads_default_threshold(tmp_path, monkeypatch):
    """No depth_threshold_mm in config -> default 1.0 bound into the hook."""
    (tmp_path / "config.yml").write_text("fabric:\n  path: /x\n  id_col: nhm_id\n")
    captured = {}

    def fake_agg_source(adapter, *args, **kwargs):
        captured["adapter"] = adapter

    monkeypatch.setattr(
        "nhf_spatial_targets.aggregate.ua_swe.aggregate_source", fake_agg_source
    )
    aggregate_ua_swe("/fab", "nhm_id", tmp_path, 500)
    times = pd.date_range("2000-01-15", periods=1, freq="D")
    ds = xr.Dataset(
        {"snow_depth": (["time", "y", "x"], np.array([[[50.0]]]))},
        coords={"time": times, "y": [0.0], "x": [0.0]},
    )
    scf = captured["adapter"].pre_aggregate_hook(ds)["snow_covered_fraction"]
    assert scf.attrs["depth_threshold_mm"] == 1.0


def test_aggregate_ua_swe_reads_custom_threshold(tmp_path, monkeypatch):
    """depth_threshold_mm: 5.0 under targets.snow_covered_area is honored."""
    (tmp_path / "config.yml").write_text(
        "fabric:\n  path: /x\n  id_col: nhm_id\n"
        "targets:\n  snow_covered_area:\n    depth_threshold_mm: 5.0\n"
    )
    captured = {}

    def fake_agg_source(adapter, *args, **kwargs):
        captured["adapter"] = adapter

    monkeypatch.setattr(
        "nhf_spatial_targets.aggregate.ua_swe.aggregate_source", fake_agg_source
    )
    aggregate_ua_swe("/fab", "nhm_id", tmp_path, 500)
    times = pd.date_range("2000-01-15", periods=1, freq="D")
    ds = xr.Dataset(
        {"snow_depth": (["time", "y", "x"], np.array([[[3.0]]]))},
        coords={"time": times, "y": [0.0], "x": [0.0]},
    )
    scf = captured["adapter"].pre_aggregate_hook(ds)["snow_covered_fraction"]
    assert scf.isel(time=0).values[0] == 0.0  # 3 > 5 False
    assert scf.attrs["depth_threshold_mm"] == 5.0


def test_aggregate_ua_swe_is_callable_with_period_kwarg():
    """The CLI dispatcher forwards --period via kwargs; the function must
    accept it for the _run_tier_agg(period=...) path."""
    import inspect

    sig = inspect.signature(aggregate_ua_swe)
    assert "period" in sig.parameters
    # Default None so omitting --period aggregates every year on disk.
    assert sig.parameters["period"].default is None


def test_cli_exposes_aggregate_ua_swe():
    """The agg CLI resolves aggregators via the cli package namespace, so
    aggregate_ua_swe must be importable from it and listed in __all__."""
    from nhf_spatial_targets import cli

    assert cli.aggregate_ua_swe is aggregate_ua_swe
    assert "aggregate_ua_swe" in cli.__all__


def test_cli_registers_agg_ua_swe():
    """``nhf-targets agg ua-swe`` must be wired in the agg sub-app."""
    from nhf_spatial_targets.cli.agg import agg_ua_swe_cmd

    assert callable(agg_ua_swe_cmd)
