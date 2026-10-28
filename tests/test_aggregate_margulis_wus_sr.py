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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from nhf_spatial_targets.aggregate.margulis_wus_sr import (  # noqa: E402
    MaskRule,
    classify_perennial,
    ensure_perennial_snow_mask,
    make_perennial_weight_filter,
    mask_build_key,
    mask_content_fingerprint,
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


def test_classify_non_dyadic_fraction_follows_spec_ratio_form():
    # held / valid > frac is the spec rule; 57/100 is not > 0.57.
    assert classify_perennial(held=57, valid=100, min_year_fraction=0.57) == 0
    assert classify_perennial(held=58, valid=100, min_year_fraction=0.57) == 1


def test_mask_rule_from_catalog_rejects_nonfinite(monkeypatch):
    from nhf_spatial_targets.aggregate import margulis_wus_sr as m

    real = m.catalog.source("margulis_wus_sr")
    for field in ("swe_threshold_m", "min_year_fraction"):
        bad = {**real, "perennial_snow_mask": {**real["perennial_snow_mask"]}}
        bad["perennial_snow_mask"][field] = float("nan")
        monkeypatch.setattr(m.catalog, "source", lambda key, _b=bad: _b)
        with pytest.raises(ValueError, match="finite|must be"):
            MaskRule.from_catalog()


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


def test_mask_build_key_stable_and_sensitive():
    r = _RULE
    assert mask_build_key(r, [1985, 1986]) == mask_build_key(r, [1986, 1985])
    assert mask_build_key(r, [1985]) != mask_build_key(r, [1985, 1986])
    r2 = MaskRule(0.05, "09-30", 0.5)
    assert mask_build_key(r2, [1985]) != mask_build_key(r, [1985])
    assert len(mask_build_key(r, [1985])) == 12


def _per():
    p = np.zeros((3, 4), dtype=np.int8)
    p[0, 0] = p[1, 2] = 1
    return p


def test_content_fingerprint_deterministic_and_dtype_invariant():
    a = mask_content_fingerprint(_RULE, _LAT, _LON, _per())
    assert a == mask_content_fingerprint(_RULE, _LAT, _LON, _per())
    assert len(a) == 12
    # float32 coords that are exactly representable hash like float64
    lat32, lon32 = _LAT.astype("float32"), _LON.astype("float32")
    f64 = mask_content_fingerprint(
        _RULE, lat32.astype("float64"), lon32.astype("float64"), _per()
    )
    assert mask_content_fingerprint(_RULE, lat32, lon32, _per()) == f64
    assert mask_content_fingerprint(_RULE, _LAT, _LON, _per().astype(bool)) == a


def test_content_fingerprint_sensitive():
    base = mask_content_fingerprint(_RULE, _LAT, _LON, _per())
    flipped = _per()
    flipped[2, 3] = 1
    assert mask_content_fingerprint(_RULE, _LAT, _LON, flipped) != base
    assert (
        mask_content_fingerprint(MaskRule(0.05, "09-30", 0.5), _LAT, _LON, _per())
        != base
    )
    # same bytes, different shape/orientation
    sq_lat = np.array([40.0, 40.1, 40.2])
    sq_lon = np.array([-120.0, -119.9, -119.8])
    sq = np.zeros((3, 3), dtype=np.int8)
    sq[0, 1] = 1
    assert mask_content_fingerprint(_RULE, sq_lat, sq_lon, sq) != (
        mask_content_fingerprint(_RULE, sq_lat, sq_lon, sq.T.copy())
    )
    assert mask_content_fingerprint(_RULE, sq_lat, sq_lon, sq) != (
        mask_content_fingerprint(_RULE, sq_lon, sq_lat, sq)
    )


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
        assert ds.attrs["perennial_snow_mask_build_key"] == mask_build_key(
            _RULE, [1985, 1986, 1987, 1988]
        )
        assert ds.attrs["perennial_snow_mask_fingerprint"] == (
            mask_content_fingerprint(_RULE, _LAT, _LON, per)
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


def test_ensure_mask_year_added_no_cell_change_keeps_fingerprint(tmp_path):
    store = _synthetic_store(tmp_path)
    path = ensure_perennial_snow_mask(store, _RULE)
    m1 = path.stat().st_mtime_ns
    fp1 = mask_stamp_attrs(path)["perennial_snow_mask_fingerprint"]
    # All-NaN year: adds to neither held nor valid, so no cell can flip.
    _write_year(store, 1989, np.full((3, 4), np.nan))
    ensure_perennial_snow_mask(store, _RULE)
    assert path.stat().st_mtime_ns != m1  # rebuilt
    with xr.open_dataset(path) as ds:
        assert ds.attrs["perennial_snow_mask_years"].endswith("1989")
    assert mask_stamp_attrs(path)["perennial_snow_mask_fingerprint"] == fp1


def test_ensure_mask_year_added_cell_flip_changes_fingerprint(tmp_path):
    store = _synthetic_store(tmp_path)
    path = ensure_perennial_snow_mask(store, _RULE)
    fp1 = mask_stamp_attrs(path)["perennial_snow_mask_fingerprint"]
    # (0,1) was 2/4 (not perennial); held in 1989 and 1990 -> 4/6 > 0.5.
    a = np.zeros((3, 4))
    a[0, 1] = 0.5
    _write_year(store, 1989, a)
    _write_year(store, 1990, a)
    ensure_perennial_snow_mask(store, _RULE)
    assert mask_stamp_attrs(path)["perennial_snow_mask_fingerprint"] != fp1


def test_ensure_mask_threshold_change_changes_both(tmp_path):
    store = _synthetic_store(tmp_path)
    path = ensure_perennial_snow_mask(store, _RULE)
    with xr.open_dataset(path) as ds:
        key1 = ds.attrs["perennial_snow_mask_build_key"]
    fp1 = mask_stamp_attrs(path)["perennial_snow_mask_fingerprint"]
    ensure_perennial_snow_mask(store, MaskRule(0.05, "09-30", 0.5))
    with xr.open_dataset(path) as ds:
        assert ds.attrs["perennial_snow_mask_build_key"] != key1
    assert mask_stamp_attrs(path)["perennial_snow_mask_fingerprint"] != fp1


def test_ensure_mask_without_build_key_is_rebuilt(tmp_path):
    store = _synthetic_store(tmp_path)
    path = ensure_perennial_snow_mask(store, _RULE)
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    del ds.attrs["perennial_snow_mask_build_key"]
    path.unlink()
    ds.to_netcdf(path)
    ds.close()
    m1 = path.stat().st_mtime_ns
    ensure_perennial_snow_mask(store, _RULE)
    assert path.stat().st_mtime_ns != m1
    with xr.open_dataset(path) as d2:
        assert "perennial_snow_mask_build_key" in d2.attrs


def test_ensure_mask_stored_fingerprint_matches_recomputed(tmp_path):
    path = ensure_perennial_snow_mask(_synthetic_store(tmp_path), _RULE)
    with xr.open_dataset(path) as ds:
        rule = MaskRule(
            float(ds.attrs["perennial_snow_mask_swe_threshold_m"]),
            str(ds.attrs["perennial_snow_mask_check_month_day"]),
            float(ds.attrs["perennial_snow_mask_min_year_fraction"]),
        )
        fp = mask_content_fingerprint(
            rule, ds["lat"].values, ds["lon"].values, ds["perennial"].values
        )
        assert ds.attrs["perennial_snow_mask_fingerprint"] == fp


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


def test_perennial_weight_filter_float32_coords(tmp_path):
    """Real consolidated lat/lon are float32; mask and cell coords must match."""
    lat32, lon32 = _LAT.astype("float32"), _LON.astype("float32")
    base = np.zeros((3, 4))
    for yr in (1985, 1986, 1987):
        a = base.copy()
        a[0, 0] = 0.5  # perennial every year
        _write_year(tmp_path, yr, a, lat=lat32, lon=lon32)
    path = ensure_perennial_snow_mask(tmp_path, _RULE)
    filt = make_perennial_weight_filter(path)
    w = pd.DataFrame({"hru_id": [1, 1], "wght": [0.5, 0.5], "i": [0, 1]})
    cy = np.array([lat32[0], lat32[0]])
    cx = np.array([lon32[0], lon32[1]])
    assert cy.dtype == np.float32
    assert filt(w, cy, cx)["i"].tolist() == [1]


def test_perennial_weight_filter_unmatched_coords_raise(tmp_path):
    path = ensure_perennial_snow_mask(_synthetic_store(tmp_path), _RULE)
    filt = make_perennial_weight_filter(path)
    w = pd.DataFrame({"hru_id": [1], "wght": [1.0]})
    with pytest.raises(ValueError, match="absent from the perennial mask grid"):
        filt(w, np.array([50.0]), np.array([_LON[0]]))


def test_ensure_mask_reuse_reads_no_swe(tmp_path, monkeypatch):
    from nhf_spatial_targets.aggregate import margulis_wus_sr as m

    store = _synthetic_store(tmp_path)
    ensure_perennial_snow_mask(store, _RULE)

    def _boom(*a, **k):
        raise AssertionError("SWE loaded on the reuse path")

    monkeypatch.setattr(m, "_load_check_slice", _boom)
    ensure_perennial_snow_mask(store, _RULE)


# ---------------------------------------------------------------------------
# End-to-end with real gdptools (#363)
# ---------------------------------------------------------------------------
# 12x12 grid so the gdptools agg subset has non-zero offsets. Cell (lat, lon)
# centres: lat 11.5..0.5 descending, lon -3.5..7.5.
_E_LAT = np.arange(11.5, -0.5, -1.0)
_E_LON = np.arange(-3.5, 8.5, 1.0)
_E_YEARS = (1985, 1986, 1987, 1988)
_E_A, _E_B, _E_C, _E_D = 1, 2, 3, 4
# A: lat 3.5, lon 2.5-4.5 (all perennial). B: lat 1.5-2.5, lon 5.5-6.5 (lat 1.5
# row perennial). C: lat 4.5-5.5, lon 1.5-2.5 (none perennial; (5.5, 1.5) NaN).
# D: lat 2.5, lon 2.5-3.5 (no perennial cell, no NaN; directly below A's row, so
# it borders perennial cells and must keep all its weight rows).
_E_PERENNIAL = {(3.5, 2.5), (3.5, 3.5), (3.5, 4.5), (1.5, 5.5), (1.5, 6.5)}
_E_SEASONAL_B = [(2.5, 5.5), (2.5, 6.5)]


def _e_val(lat, lon):
    """Seasonal SWE (m) on 09-29; < 0.1 everywhere so it never trips the mask."""
    r = int(np.where(_E_LAT == lat)[0][0])
    c = int(np.where(_E_LON == lon)[0][0])
    return 0.0005 * (r * 12 + c + 1)


def _e_write_store(datastore):
    daily = datastore / "margulis_wus_sr" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    for year in _E_YEARS:
        data = np.empty((2, _E_LAT.size, _E_LON.size))
        for r, lat in enumerate(_E_LAT):
            for c, lon in enumerate(_E_LON):
                if (float(lat), float(lon)) in _E_PERENNIAL:
                    data[:, r, c] = 0.5
                else:
                    data[0, r, c] = _e_val(lat, lon)
                    data[1, r, c] = _e_val(lat, lon) + 0.001
        data[:, 6, 5] = np.nan  # lat 5.5, lon 1.5, inside HRU C
        ds = xr.Dataset(
            {"SWE": (("time", "lat", "lon"), data.astype("float32"))},
            coords={
                "time": (
                    "time",
                    pd.to_datetime([f"{year}-09-29", f"{year}-09-30"]),
                    {"standard_name": "time"},
                ),
                "lat": (
                    "lat",
                    _E_LAT,
                    {"standard_name": "latitude", "units": "degrees_north"},
                ),
                "lon": (
                    "lon",
                    _E_LON,
                    {"standard_name": "longitude", "units": "degrees_east"},
                ),
            },
        )
        ds.to_netcdf(daily / f"margulis_wus_sr_daily_{year}.nc")


def _e_project(root, datastore):
    import json

    import geopandas as gpd
    import yaml
    from shapely.geometry import box

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": str(root / "fabric.gpkg"), "id_col": "hru_id"},
                "datastore": str(datastore),
            }
        )
    )
    (root / "fabric.json").write_text(json.dumps({"sha256": "abc123"}))
    (root / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    gdf = gpd.GeoDataFrame(
        {"hru_id": [_E_A, _E_B, _E_C, _E_D]},
        geometry=[
            box(2.05, 3.05, 4.95, 3.95),
            box(5.05, 1.05, 6.95, 2.95),
            box(1.05, 4.05, 2.95, 5.95),
            box(2.05, 2.05, 3.95, 2.95),
        ],
        crs="EPSG:4326",
    )
    gdf.to_file(root / "fabric.gpkg", driver="GPKG")
    return root


def _e_year_nc(root, year=1985):
    return (
        root
        / "data"
        / "aggregated"
        / "margulis_wus_sr"
        / f"margulis_wus_sr_{year}_agg.nc"
    )


def _e_run_masked(root):
    aggregate_margulis_wus_sr(root / "fabric.gpkg", "hru_id", root)


def _e_run_plain(root):
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    aggregate_source(ADAPTER, root / "fabric.gpkg", "hru_id", root, 500)


def _e_datastore_of(root):
    import yaml

    return Path(yaml.safe_load((root / "config.yml").read_text())["datastore"])


def _e_content_fp(datastore):
    path = datastore / "margulis_wus_sr" / "derived" / "perennial_snow_mask.nc"
    with xr.open_dataset(path) as ds:
        return mask_content_fingerprint(
            MaskRule.from_catalog(),
            ds["lat"].values,
            ds["lon"].values,
            ds["perennial"].values,
        )


def _e_write_extra_year(datastore, year, *, flip):
    """Write one more year. flip=False: seasonal values only (no cell flips).
    flip=True: a seasonal cell (lat 2.5, lon 3.5, HRU D) is held in the new year."""
    daily = datastore / "margulis_wus_sr" / "daily"
    data = np.empty((2, _E_LAT.size, _E_LON.size))
    for r, lat in enumerate(_E_LAT):
        for c, lon in enumerate(_E_LON):
            if (float(lat), float(lon)) in _E_PERENNIAL:
                data[:, r, c] = 0.5
            else:
                data[0, r, c] = _e_val(lat, lon)
                data[1, r, c] = _e_val(lat, lon) + 0.001
    data[:, 6, 5] = np.nan
    if flip:
        # Flipping needs > half of valid years; use several new years (caller).
        r = int(np.where(_E_LAT == 2.5)[0][0])
        c = int(np.where(_E_LON == 3.5)[0][0])
        data[:, r, c] = 0.5
    ds = xr.Dataset(
        {"SWE": (("time", "lat", "lon"), data.astype("float32"))},
        coords={
            "time": (
                "time",
                pd.to_datetime([f"{year}-09-29", f"{year}-09-30"]),
                {"standard_name": "time"},
            ),
            "lat": (
                "lat",
                _E_LAT,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                "lon",
                _E_LON,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        },
    )
    ds.to_netcdf(daily / f"margulis_wus_sr_daily_{year}.nc")


def test_end_to_end_year_added_no_cell_change_reaggregates_nothing(tmp_path):
    datastore = tmp_path / "datastore"
    _e_write_store(datastore)
    root = _e_project(tmp_path / "proj", datastore)
    _e_run_masked(root)
    before = {y: _e_year_nc(root, y).stat().st_mtime_ns for y in _E_YEARS}
    fp = _e_content_fp(datastore)
    _e_write_extra_year(datastore, 1989, flip=False)
    _e_run_masked(root)
    assert _e_content_fp(datastore) == fp  # mask rebuilt, content identical
    for y in _E_YEARS:
        assert _e_year_nc(root, y).stat().st_mtime_ns == before[y]
    with xr.open_dataset(_e_year_nc(root, 1989)) as ds:
        assert ds.attrs["perennial_snow_mask_fingerprint"] == fp


def test_end_to_end_cell_flipping_years_reaggregate_everything(tmp_path):
    datastore = tmp_path / "datastore"
    _e_write_store(datastore)
    root = _e_project(tmp_path / "proj", datastore)
    _e_run_masked(root)
    before = {y: _e_year_nc(root, y).stat().st_mtime_ns for y in _E_YEARS}
    fp = _e_content_fp(datastore)
    for y in (1989, 1990, 1991, 1992, 1993):  # 5/9 held > 0.5
        _e_write_extra_year(datastore, y, flip=True)
    _e_run_masked(root)
    new_fp = _e_content_fp(datastore)
    assert new_fp != fp
    for y in _E_YEARS:
        assert _e_year_nc(root, y).stat().st_mtime_ns != before[y]
        with xr.open_dataset(_e_year_nc(root, y)) as ds:
            assert ds.attrs["perennial_snow_mask_fingerprint"] == new_fp


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    base = tmp_path_factory.mktemp("margulis_e2e")
    datastore = base / "datastore"
    _e_write_store(datastore)
    masked = _e_project(base / "masked", datastore)
    plain = _e_project(base / "plain", datastore)
    _e_run_masked(masked)
    _e_run_plain(plain)
    return datastore, masked, plain


def test_end_to_end_fully_perennial_hru_nan(e2e):
    _, masked, _ = e2e
    with xr.open_dataset(_e_year_nc(masked)) as ds:
        a = ds["SWE"].sel(hru_id=_E_A)
        assert np.isnan(a).all()  # never 0.0
        assert not (a == 0.0).any()


def test_end_to_end_partial_hru_mean_of_seasonal(e2e):
    _, masked, _ = e2e
    with xr.open_dataset(_e_year_nc(masked)) as ds:
        got = ds["SWE"].sel(hru_id=_E_B).values
    expected = np.array(
        [
            np.mean([_e_val(lat, lon) for lat, lon in _E_SEASONAL_B]),
            np.mean([_e_val(lat, lon) + 0.001 for lat, lon in _E_SEASONAL_B]),
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_end_to_end_unmasked_hru_bit_identical(e2e):
    _, masked, plain = e2e
    for year in _E_YEARS:
        with (
            xr.open_dataset(_e_year_nc(masked, year)) as m,
            xr.open_dataset(_e_year_nc(plain, year)) as p,
        ):
            mc = m["SWE"].sel(hru_id=_E_C).values
            pc = p["SWE"].sel(hru_id=_E_C).values
        np.testing.assert_array_equal(mc, pc)
        assert np.isnan(mc).all()  # the in-grid NaN pixel poisons plain "mean"


def test_end_to_end_finite_unmasked_hru_bit_identical(e2e):
    """HRU D borders perennial cells but has none itself and no NaN pixel."""
    _, masked, plain = e2e
    for year in _E_YEARS:
        with (
            xr.open_dataset(_e_year_nc(masked, year)) as m,
            xr.open_dataset(_e_year_nc(plain, year)) as p,
        ):
            md = m["SWE"].sel(hru_id=_E_D).values
            pd_ = p["SWE"].sel(hru_id=_E_D).values
        assert np.isfinite(md).all()
        np.testing.assert_array_equal(md, pd_)


def test_end_to_end_stamps_fingerprint(e2e):
    _, masked, plain = e2e
    expected = _e_content_fp(_e_datastore_of(masked))
    for year in _E_YEARS:
        with xr.open_dataset(_e_year_nc(masked, year)) as ds:
            assert ds.attrs["perennial_snow_mask_fingerprint"] == expected
    with xr.open_dataset(_e_year_nc(plain)) as ds:
        assert "perennial_snow_mask_fingerprint" not in ds.attrs


def test_end_to_end_premask_year_reaggregated(tmp_path, caplog):
    datastore = tmp_path / "datastore"
    _e_write_store(datastore)
    root = _e_project(tmp_path / "proj", datastore)
    _e_run_plain(root)
    with xr.open_dataset(_e_year_nc(root)) as ds:
        assert "perennial_snow_mask_fingerprint" not in ds.attrs
    with caplog.at_level(logging.WARNING):
        _e_run_masked(root)
    assert "stale" in caplog.text
    with xr.open_dataset(_e_year_nc(root)) as ds:
        assert "perennial_snow_mask_fingerprint" in ds.attrs
        assert np.isnan(ds["SWE"].sel(hru_id=_E_A)).all()


def test_module_adapter_stays_plain():
    assert ADAPTER.weight_filter is None
    assert not ADAPTER.stamp_attrs
    assert ADAPTER.stat_method == "mean"
