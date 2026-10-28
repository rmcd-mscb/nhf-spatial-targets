"""Tests for the Margulis WUS-SR daily SWE aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets import catalog as cat
from nhf_spatial_targets.aggregate.margulis_wus_sr import (
    ADAPTER,
    aggregate_margulis_wus_sr,
)


def test_adapter_declares_swe_only():
    assert ADAPTER.source_key == "margulis_wus_sr"
    assert ADAPTER.output_name == "margulis_wus_sr_agg.nc"
    assert ADAPTER.variables == ("SWE",)


def test_adapter_uses_daily_files_glob():
    """Consolidated NCs land under <datastore>/margulis_wus_sr/daily/."""
    assert ADAPTER.files_glob == "daily/margulis_wus_sr_daily_*.nc"


def test_adapter_wgs84_and_plain_mean():
    """Consolidator reprojects native EASE-Grid to regular WGS84 lat/lon;
    no per-pixel mask in a pre_aggregate_hook means default ``mean`` is
    correct (same as SNODAS)."""
    assert ADAPTER.source_crs == "EPSG:4326"
    assert ADAPTER.stat_method == "mean"


def test_adapter_has_no_hooks():
    """Per-pixel NaN over WUS-domain edges decodes via mask_and_scale;
    no pre/post hook needed."""
    assert ADAPTER.pre_aggregate_hook is None
    assert ADAPTER.post_aggregate_hook is None


def test_adapter_variable_matches_catalog():
    """The adapter's declared SWE var must appear in catalog/sources.yml
    so the cf attrs applied by the driver line up with the consolidated NCs.
    """
    entry = cat.source("margulis_wus_sr")
    catalog_vars = {v["name"] for v in entry["variables"]}
    assert set(ADAPTER.variables).issubset(catalog_vars)


def test_aggregate_margulis_is_callable_with_period_kwarg():
    """CLI dispatcher forwards --period via kwargs; the function must
    accept it for the _run_tier_agg(period=...) path."""
    import inspect

    sig = inspect.signature(aggregate_margulis_wus_sr)
    assert "period" in sig.parameters
    assert sig.parameters["period"].default is None


def test_cli_registers_agg_margulis():
    """``nhf-targets agg margulis-wus-sr`` must be wired in cli.py."""
    from nhf_spatial_targets import cli

    assert cli.aggregate_margulis_wus_sr is aggregate_margulis_wus_sr


def test_aggregate_margulis_missing_raw_raises(tmp_path):
    """#309 removed the fabric_scope token gate: margulis behaves like every
    other source — aggregating without fetched raw data raises
    FileNotFoundError pointing at the fetch command, instead of silently
    skipping based on a catalog token. (Zero spatial overlap with a fabric
    is now handled geometrically by the driver's coverage guard.)"""
    import json

    import pytest
    import yaml

    datastore = tmp_path / "datastore"
    datastore.mkdir()
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": str(tmp_path / "fake.gpkg"), "id_col": "hru_id"},
                "datastore": str(datastore),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))

    with pytest.raises(FileNotFoundError, match="fetch margulis_wus_sr"):
        aggregate_margulis_wus_sr(
            fabric_path=tmp_path / "fake.gpkg",
            id_col="hru_id",
            workdir=tmp_path,
        )


# ---------------------------------------------------------------------------
# Perennial-snow mask (#363)
# ---------------------------------------------------------------------------

import logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from nhf_spatial_targets.aggregate.margulis_wus_sr import (  # noqa: E402
    MaskRule,
    classify_perennial,
    ensure_perennial_snow_mask,
    make_perennial_weight_filter,
    mask_fingerprint,
    mask_stamp_attrs,
)

_LAT = np.array([40.0, 40.1, 40.2])
_LON = np.array([-120.0, -119.9, -119.8, -119.7])
_RULE = MaskRule(swe_threshold_m=0.1, check_month_day="09-30", min_year_fraction=0.5)


def _write_year(
    datastore, year, swe_sep30, *, last_day="09-30", lat=_LAT, lon=_LON
) -> None:
    daily = datastore / "margulis_wus_sr" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    times = pd.to_datetime([f"{year}-09-29", f"{year}-{last_day}"])
    if last_day != "09-30":
        times = pd.to_datetime([f"{year}-06-29", f"{year}-{last_day}"])
    data = np.stack([np.zeros_like(swe_sep30), swe_sep30]).astype("float32")
    ds = xr.Dataset(
        {"SWE": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(daily / f"margulis_wus_sr_daily_{year}.nc")


def _synthetic_store(tmp_path):
    """4 years, 3x4 grid. Pixel (0,0): held 4/4; (0,1): held 2/4 (tie);
    (0,2): held 3/4; (0,3): NaN all years; (1,0): held 2 of 3 valid."""
    base = np.zeros((3, 4))
    held = {
        1985: [(0, 0), (0, 1), (0, 2), (1, 0)],
        1986: [(0, 0), (0, 1), (0, 2), (1, 0)],
        1987: [(0, 0), (0, 2)],
        1988: [(0, 0)],
    }
    for yr, cells in held.items():
        a = base.copy()
        for c in cells:
            a[c] = 0.5
        a[0, 3] = np.nan
        if yr == 1988:
            a[1, 0] = np.nan  # (1,0): 2 held of 3 valid
        if yr == 1987:
            a[0, 2] = 0.1  # exactly at threshold -> held (>=)
        _write_year(tmp_path, yr, a)
    return tmp_path


def test_classify_exactly_half_not_perennial():
    assert classify_perennial(held=2, valid=4, min_year_fraction=0.5) == 0


def test_classify_more_than_half_perennial():
    assert classify_perennial(held=3, valid=4, min_year_fraction=0.5) == 1


def test_classify_nan_years_excluded_from_denominator():
    assert classify_perennial(held=2, valid=3, min_year_fraction=0.5) == 1


def test_classify_all_nan_not_perennial():
    assert classify_perennial(held=0, valid=0, min_year_fraction=0.5) == 0


def test_classify_array_returns_int8():
    out = classify_perennial(
        np.array([[3, 2], [0, 2]]), np.array([[4, 4], [0, 3]]), 0.5
    )
    assert out.dtype == np.int8
    assert out.tolist() == [[1, 0], [0, 1]]


def test_mask_rule_from_catalog_values():
    r = MaskRule.from_catalog()
    assert (r.swe_threshold_m, r.check_month_day, r.min_year_fraction) == (
        0.1,
        "09-30",
        0.5,
    )


def test_mask_fingerprint_stable_and_sensitive():
    r = _RULE
    assert mask_fingerprint(r, [1985, 1986]) == mask_fingerprint(r, [1986, 1985])
    assert mask_fingerprint(r, [1985]) != mask_fingerprint(r, [1985, 1986])
    r2 = MaskRule(0.05, "09-30", 0.5)
    assert mask_fingerprint(r2, [1985]) != mask_fingerprint(r, [1985])
    assert len(mask_fingerprint(r, [1985])) == 12


def test_ensure_mask_builds_expected_cells_and_attrs(tmp_path):
    store = _synthetic_store(tmp_path)
    path = ensure_perennial_snow_mask(store, _RULE)
    assert path == store / "margulis_wus_sr" / "derived" / "perennial_snow_mask.nc"
    with xr.open_dataset(path) as ds:
        per = ds["perennial"].values
        assert ds["perennial"].dtype == np.int8
        assert ds["n_years_held"].dtype == np.int16
        assert ds["n_years_valid"].dtype == np.int16
        assert per[0, 0] == 1  # 4/4
        assert per[0, 1] == 0  # 2/4 tie
        assert per[0, 2] == 1  # 3/4 incl. exactly-at-threshold year
        assert per[0, 3] == 0  # all NaN
        assert per[1, 0] == 1  # 2 of 3 valid
        assert int(ds["n_years_valid"].values[1, 0]) == 3
        assert int(ds["n_years_valid"].values[0, 3]) == 0
        np.testing.assert_array_equal(ds["lat"].values, _LAT)
        np.testing.assert_array_equal(ds["lon"].values, _LON)
        assert ds.attrs["perennial_snow_mask_years"] == "1985,1986,1987,1988"
        assert ds.attrs["perennial_snow_mask_n_cells"] == 3
        assert ds.attrs["perennial_snow_mask_area_km2"] > 0
        assert ds.attrs["perennial_snow_mask_fingerprint"] == mask_fingerprint(
            _RULE, [1985, 1986, 1987, 1988]
        )


def test_ensure_mask_reuses_matching_file(tmp_path):
    store = _synthetic_store(tmp_path)
    path = ensure_perennial_snow_mask(store, _RULE)
    m1 = path.stat().st_mtime_ns
    assert ensure_perennial_snow_mask(store, _RULE) == path
    assert path.stat().st_mtime_ns == m1


def test_ensure_mask_rebuilds_on_threshold_change(tmp_path):
    store = _synthetic_store(tmp_path)
    path = ensure_perennial_snow_mask(store, _RULE)
    m1 = path.stat().st_mtime_ns
    ensure_perennial_snow_mask(store, MaskRule(0.05, "09-30", 0.5))
    assert path.stat().st_mtime_ns != m1
    assert mask_stamp_attrs(path)["perennial_snow_mask_swe_threshold_m"] == 0.05


def test_ensure_mask_rebuilds_when_year_added(tmp_path):
    store = _synthetic_store(tmp_path)
    path = ensure_perennial_snow_mask(store, _RULE)
    fp1 = mask_stamp_attrs(path)["perennial_snow_mask_fingerprint"]
    _write_year(store, 1989, np.full((3, 4), 0.5))
    ensure_perennial_snow_mask(store, _RULE)
    assert mask_stamp_attrs(path)["perennial_snow_mask_fingerprint"] != fp1


def test_ensure_mask_skips_year_without_check_date(tmp_path, caplog):
    store = _synthetic_store(tmp_path)
    _write_year(store, 2020, np.full((3, 4), 0.5), last_day="06-30")
    with caplog.at_level(logging.INFO):
        path = ensure_perennial_snow_mask(store, _RULE)
    with xr.open_dataset(path) as ds:
        assert "2020" not in ds.attrs["perennial_snow_mask_years"]
    assert any(
        "2020" in r.message and r.levelno == logging.INFO for r in caplog.records
    )


def test_ensure_mask_mismatched_grid_raises(tmp_path):
    store = _synthetic_store(tmp_path)
    _write_year(store, 1989, np.zeros((3, 4)), lat=_LAT + 0.01)
    with pytest.raises(ValueError, match="lat"):
        ensure_perennial_snow_mask(store, _RULE)


def test_ensure_mask_cf_attrs(tmp_path):
    path = ensure_perennial_snow_mask(_synthetic_store(tmp_path), _RULE)
    with xr.open_dataset(path) as ds:
        assert ds.attrs["Conventions"] == "CF-1.6"
        assert ds["lat"].attrs["standard_name"] == "latitude"
        assert ds["lat"].attrs["units"] == "degrees_north"
        assert ds["lat"].attrs["axis"] == "Y"
        assert ds["lon"].attrs["standard_name"] == "longitude"
        assert ds["lon"].attrs["units"] == "degrees_east"
        assert ds["lon"].attrs["axis"] == "X"
        assert list(ds["perennial"].attrs["flag_values"]) == [0, 1]
        assert ds["perennial"].attrs["flag_meanings"] == "seasonal perennial"
        assert ds["perennial"].attrs["grid_mapping"] == "crs"
        assert "crs" in ds.variables
        assert "history" in ds.attrs


def test_mask_stamp_attrs_types(tmp_path):
    path = ensure_perennial_snow_mask(_synthetic_store(tmp_path), _RULE)
    a = mask_stamp_attrs(path)
    assert set(a) == {
        "perennial_snow_mask_fingerprint",
        "perennial_snow_mask_swe_threshold_m",
        "perennial_snow_mask_check_month_day",
        "perennial_snow_mask_min_year_fraction",
    }
    assert isinstance(a["perennial_snow_mask_fingerprint"], str)
    assert isinstance(a["perennial_snow_mask_check_month_day"], str)
    assert isinstance(a["perennial_snow_mask_swe_threshold_m"], float)
    assert isinstance(a["perennial_snow_mask_min_year_fraction"], float)


def test_perennial_weight_filter_drops_masked_rows(tmp_path):
    path = ensure_perennial_snow_mask(_synthetic_store(tmp_path), _RULE)
    filt = make_perennial_weight_filter(path)
    w = pd.DataFrame(
        {"hru_id": [1, 1, 2, 2], "wght": [0.5, 0.5, 0.5, 0.5], "i": [0, 1, 2, 3]}
    )
    # (0,0) perennial, (0,1) seasonal, (1,0) perennial, (2,3) seasonal
    cy = np.array([_LAT[0], _LAT[0], _LAT[1], _LAT[2]])
    cx = np.array([_LON[0], _LON[1], _LON[0], _LON[3]])
    out = filt(w, cy, cx)
    assert list(out.columns) == list(w.columns)
    assert out["i"].tolist() == [1, 3]


def test_perennial_weight_filter_unmatched_coords_raise(tmp_path):
    path = ensure_perennial_snow_mask(_synthetic_store(tmp_path), _RULE)
    filt = make_perennial_weight_filter(path)
    w = pd.DataFrame({"hru_id": [1], "wght": [1.0]})
    with pytest.raises(ValueError, match="absent from the perennial mask grid"):
        filt(w, np.array([50.0]), np.array([_LON[0]]))
