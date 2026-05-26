"""Tests for the SCA target builder end-to-end.

Covers the CI-bounded formula from ``PRMSobjfun.f90:calcSCA``:

  - ``ci >= ci_threshold`` gate (HRU-mean CI, in addition to the
    per-pixel CI > 70 gate already applied at aggregation time);
  - ``lower = ci * sca_obs``, ``upper = lower + (1 - ci)`` (both inputs
    on fractional [0, 1] after /100);
  - July/August forced to ``(0, 0)`` wherever the CI gate passes;
  - single-source ``n_sources`` binary diagnostic (1 = valid, 0 = NaN).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml


def _write_synthetic_fabric(path: Path, id_col: str = "nhm_id") -> None:
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


def _write_mod10c1_year(
    path: Path,
    year: int,
    *,
    snow_native: float | np.ndarray,
    ci_native: float | np.ndarray,
    id_col: str = "nhm_id",
) -> None:
    """Write a per-year MOD10C1 aggregated NC.

    Values are in MOD10C1's native 0-100 integer scale ("percent");
    builder divides by 100 to fractional [0, 1]. Scalars broadcast over
    (time, hru); arrays must already be (time, hru)-shaped.
    """
    times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    hrus = [1, 2, 3]
    if np.isscalar(snow_native):
        snow_arr = np.full((len(times), len(hrus)), snow_native, dtype=np.float32)
    else:
        snow_arr = np.asarray(snow_native, dtype=np.float32)
    if np.isscalar(ci_native):
        ci_arr = np.full((len(times), len(hrus)), ci_native, dtype=np.float32)
    else:
        ci_arr = np.asarray(ci_native, dtype=np.float32)
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (("time", id_col), snow_arr),
            "Day_CMG_Clear_Index": (("time", id_col), ci_arr),
        },
        coords={"time": times, id_col: hrus},
    )
    ds["Day_CMG_Snow_Cover"].attrs["units"] = "percent"
    ds["Day_CMG_Clear_Index"].attrs["units"] = "percent"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _make_sca_project(
    tmp_path: Path,
    *,
    period: str = "2005-01-01/2005-12-31",
    ci_threshold: float = 0.70,
    nn_fill: bool = False,
    snow_native: float | np.ndarray = 60.0,
    ci_native: float | np.ndarray = 90.0,
) -> Path:
    """Build a project skeleton with synthetic fabric + MOD10C1 NCs.

    Default values: snow=60% (sca_obs=0.60), ci=90% (ci=0.90), passing
    threshold. Override for gate / July-Aug / NaN-coverage scenarios.
    """
    workdir = tmp_path / "proj"
    workdir.mkdir()
    fabric_path = tmp_path / "fabric.gpkg"
    _write_synthetic_fabric(fabric_path)
    (workdir / "fabric.json").write_text(json.dumps({"id_col": "nhm_id"}))

    cfg = {
        "datastore": str(tmp_path / "store"),
        "fabric": {"path": str(fabric_path), "id_col": "nhm_id", "token": "or"},
        "targets": {
            "snow_covered_area": {
                "period": period,
                "ci_threshold": ci_threshold,
                "nn_fill": nn_fill,
            },
            "runoff": {"enabled": False},
            "aet": {"enabled": False},
            "recharge": {"enabled": False},
            "soil_moisture": {"enabled": False},
            "snow_water_equivalent": {"enabled": False},
        },
    }
    (workdir / "config.yml").write_text(yaml.safe_dump(cfg))

    agg_dir = workdir / "data" / "aggregated" / "mod10c1_v061"
    years = list(
        range(
            pd.Timestamp(period.split("/")[0]).year,
            pd.Timestamp(period.split("/")[1]).year + 1,
        )
    )
    for year in years:
        _write_mod10c1_year(
            agg_dir / f"mod10c1_v061_{year}_agg.nc",
            year,
            snow_native=snow_native,
            ci_native=ci_native,
        )
    return workdir


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_build_rejects_multi_source_config(tmp_path: Path):
    """The modis_ci range method requires exactly mod10c1_v061."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(tmp_path)
    cfg_path = workdir / "config.yml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["targets"]["snow_covered_area"]["sources"] = [
        "mod10c1_v061",
        "some_other_source",
    ]
    cfg_path.write_text(yaml.safe_dump(cfg))
    project = load(workdir)
    with pytest.raises(ValueError, match="modis_ci range method requires"):
        build(project)


def test_build_rejects_out_of_range_threshold(tmp_path: Path):
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(tmp_path, ci_threshold=70.0)  # native, not fractional
    project = load(workdir)
    with pytest.raises(ValueError, match=r"ci_threshold=.*\[0\.0, 1\.0\]"):
        build(project)


# ---------------------------------------------------------------------------
# End-to-end build: output files + schema
# ---------------------------------------------------------------------------


def test_build_writes_unfilled_and_filled_files(tmp_path: Path):
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(tmp_path, period="2005-01-01/2005-01-31", nn_fill=True)
    project = load(workdir)
    build(project)
    assert (project.targets_dir() / "sca_targets.nc").exists()
    assert (project.targets_dir() / "sca_targets_nn_filled.nc").exists()


def test_build_output_schema(tmp_path: Path):
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(tmp_path, period="2005-01-01/2005-01-31")
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        assert "lower_bound" in ds and "upper_bound" in ds
        assert "n_sources" in ds
        assert ds["lower_bound"].attrs["units"] == "1"
        assert ds["upper_bound"].attrs["units"] == "1"
        assert ds["lower_bound"].attrs["cell_methods"] == "time: point"
        assert ds.attrs["Conventions"] == "CF-1.6"
        assert ds["time"].attrs["bounds"] == "time_bnds"
        assert "time_bnds" in ds.variables
        assert ds["n_sources"].dtype == np.int8
        # The n_sources flag attrs are int8 (CF §3.5).
        assert ds["n_sources"].attrs["flag_values"].dtype == np.int8
        # Single-source SCA: only "none"/"one" labels.
        assert ds["n_sources"].attrs["flag_meanings"] == "none one"
        assert ds.attrs["ci_threshold"] == 0.70
        assert ds.attrs["summer_zero_months"] == "7,8"


def test_build_daily_time_index(tmp_path: Path):
    """One timestamp per day across the requested period (stitched)."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(tmp_path, period="2005-03-01/2005-03-31")
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        times = pd.DatetimeIndex(ds["time"].values)
        assert len(times) == 31
        assert times[0] == pd.Timestamp("2005-03-01")
        assert times[-1] == pd.Timestamp("2005-03-31")


# ---------------------------------------------------------------------------
# CI gate
# ---------------------------------------------------------------------------


def test_ci_above_threshold_passes_and_computes_bounds(tmp_path: Path):
    """ci=0.90, sca_obs=0.60 → lower=0.54, upper=0.54+0.10=0.64."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    # Limit to a single non-summer day so July/Aug zero-forcing doesn't
    # confound the formula check.
    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-15/2005-03-15",
        snow_native=60.0,
        ci_native=90.0,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        np.testing.assert_allclose(ds["lower_bound"].values, 0.54, rtol=1e-5)
        np.testing.assert_allclose(ds["upper_bound"].values, 0.64, rtol=1e-5)
        assert (ds["n_sources"].values == 1).all()


def test_ci_below_threshold_yields_nan_bounds(tmp_path: Path):
    """ci=0.65 < 0.70 threshold → lower/upper NaN, n_sources=0."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-15/2005-03-15",
        snow_native=60.0,
        ci_native=65.0,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        assert np.isnan(ds["lower_bound"].values).all()
        assert np.isnan(ds["upper_bound"].values).all()
        assert (ds["n_sources"].values == 0).all()


def test_ci_at_threshold_passes(tmp_path: Path):
    """ci=0.70 == threshold — Fortran uses `>=`, so this must pass."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-15/2005-03-15",
        snow_native=50.0,
        ci_native=70.0,  # exactly threshold
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        # lower = 0.70 * 0.50 = 0.35; upper = 0.35 + 0.30 = 0.65
        np.testing.assert_allclose(ds["lower_bound"].values, 0.35, rtol=1e-5)
        np.testing.assert_allclose(ds["upper_bound"].values, 0.65, rtol=1e-5)
        assert (ds["n_sources"].values == 1).all()


def test_full_ci_collapses_bounds_to_point(tmp_path: Path):
    """ci=1.00 → lower=upper=sca_obs (point estimate at full confidence)."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-15/2005-03-15",
        snow_native=40.0,
        ci_native=100.0,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        np.testing.assert_allclose(ds["lower_bound"].values, 0.40, rtol=1e-5)
        np.testing.assert_allclose(ds["upper_bound"].values, 0.40, rtol=1e-5)


# ---------------------------------------------------------------------------
# July/August zero forcing
# ---------------------------------------------------------------------------


def test_july_august_forced_to_zero_when_ci_passes(tmp_path: Path):
    """A July day with passing CI: bounds forced to (0, 0)."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-07-15/2005-07-15",
        snow_native=60.0,
        ci_native=90.0,  # CI gate passes; July rule forces 0
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        np.testing.assert_array_equal(ds["lower_bound"].values, 0.0)
        np.testing.assert_array_equal(ds["upper_bound"].values, 0.0)
        # CI gate still records this day as a valid (n_sources=1) bound —
        # zero is the bound value, not "missing."
        assert (ds["n_sources"].values == 1).all()


def test_august_forced_to_zero_when_ci_passes(tmp_path: Path):
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-08-10/2005-08-10",
        snow_native=80.0,
        ci_native=95.0,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        np.testing.assert_array_equal(ds["lower_bound"].values, 0.0)
        np.testing.assert_array_equal(ds["upper_bound"].values, 0.0)


def test_july_with_failing_ci_stays_nan(tmp_path: Path):
    """A July day with CI < threshold remains NaN, not forced to zero.

    Distinguishes "observed bare ground" from "no observation" in the
    output — the zero-forcing only kicks in when the CI gate actually
    passed.
    """
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-07-15/2005-07-15",
        snow_native=60.0,
        ci_native=50.0,  # below threshold
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        assert np.isnan(ds["lower_bound"].values).all()
        assert np.isnan(ds["upper_bound"].values).all()
        assert (ds["n_sources"].values == 0).all()


def test_june_with_passing_ci_is_not_zeroed(tmp_path: Path):
    """Boundary check: June (month 6) is NOT in the zero-forced months."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-06-30/2005-06-30",
        snow_native=60.0,
        ci_native=90.0,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        np.testing.assert_allclose(ds["lower_bound"].values, 0.54, rtol=1e-5)
        np.testing.assert_allclose(ds["upper_bound"].values, 0.64, rtol=1e-5)


def test_september_with_passing_ci_is_not_zeroed(tmp_path: Path):
    """Boundary check: September (month 9) is NOT in the zero-forced months."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-09-01/2005-09-01",
        snow_native=60.0,
        ci_native=90.0,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        np.testing.assert_allclose(ds["lower_bound"].values, 0.54, rtol=1e-5)
        np.testing.assert_allclose(ds["upper_bound"].values, 0.64, rtol=1e-5)


# ---------------------------------------------------------------------------
# Year-chunked build mechanics
# ---------------------------------------------------------------------------


def test_intermediates_written_per_year(tmp_path: Path):
    """Two-year period → two per-year intermediates exist after build."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    # Pick a non-summer window in each year so the intermediates contain
    # non-zero bounds to inspect.
    workdir = _make_sca_project(tmp_path, period="2005-03-01/2006-03-31")
    project = load(workdir)
    build(project)
    intermediates_dir = project.targets_dir() / ".sca_intermediates"
    yearly = sorted(intermediates_dir.glob("sca_targets_[0-9][0-9][0-9][0-9].nc"))
    assert [p.name for p in yearly] == [
        "sca_targets_2005.nc",
        "sca_targets_2006.nc",
    ]


def test_idempotent_skip_existing_year(tmp_path: Path, caplog):
    """Re-running with intermediates already on disk skips the per-year work."""
    import logging

    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(tmp_path, period="2005-03-01/2005-03-31")
    project = load(workdir)
    build(project)
    # Second build: intermediates valid (matching fingerprints) → skip.
    # New (#213) log line names the active config + code fingerprints
    # so operators can cross-check against the cached values without
    # opening the file by hand.
    with caplog.at_level(logging.INFO, logger="nhf_spatial_targets.targets.sca"):
        build(project)
    assert "skipping" in caplog.text
    assert "config=" in caplog.text
    assert "code=" in caplog.text


# ---------------------------------------------------------------------------
# Upper-bound algebraic cap (never exceeds 1.0)
# ---------------------------------------------------------------------------


def test_upper_bound_never_exceeds_one(tmp_path: Path):
    """upper = ci*sca + (1-ci) ≤ ci*1 + (1-ci) = 1, for any sca in [0,1]."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-15/2005-03-15",
        snow_native=100.0,  # fully snowy
        ci_native=70.0,  # min passing CI
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        # lower = 0.70 * 1.00 = 0.70; upper = 0.70 + 0.30 = 1.00
        np.testing.assert_allclose(ds["lower_bound"].values, 0.70, rtol=1e-5)
        np.testing.assert_allclose(ds["upper_bound"].values, 1.00, rtol=1e-5)
        assert (ds["upper_bound"].values <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# Heterogeneous coverage: mixed CI across HRUs in a single build
# ---------------------------------------------------------------------------


def test_heterogeneous_ci_across_hrus(tmp_path: Path):
    """One day, three HRUs with different CI/snow values.

    Exercises the dim-broadcasting path that scalar-fill tests skip:
    HRU 1 has passing CI, HRU 2 has failing CI, HRU 3 has NaN CI from
    upstream (the pre-aggregation gate dropped every pixel). Verifies
    per-HRU outcomes differ correctly in a single open.
    """
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    # (time=1, hru=3) arrays: HRU 1 = passing CI, HRU 2 = failing,
    # HRU 3 = NaN CI from upstream.
    snow = np.array([[60.0, 60.0, np.nan]], dtype=np.float32)
    ci = np.array([[90.0, 50.0, np.nan]], dtype=np.float32)
    # Single-day window: _write_mod10c1_year writes a full year, so we
    # rely on the broadcaster to fill all 365 days with the same row.
    # Reshape arrays to a full-year shape.
    times_per_year = pd.date_range("2005-01-01", "2005-12-31", freq="D")
    snow_full = np.tile(snow, (len(times_per_year), 1))
    ci_full = np.tile(ci, (len(times_per_year), 1))

    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-15/2005-03-15",
        snow_native=snow_full,
        ci_native=ci_full,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        # Explicit time-dim sanity: a single-day period must produce
        # exactly one time step on disk (catches a regression where the
        # period filter silently collapses or drops the dim).
        assert ds.sizes["time"] == 1
        # HRU 1: ci=0.90, sca=0.60 → lower=0.54, upper=0.64
        np.testing.assert_allclose(
            ds["lower_bound"].sel(nhm_id=1).values, 0.54, rtol=1e-5
        )
        np.testing.assert_allclose(
            ds["upper_bound"].sel(nhm_id=1).values, 0.64, rtol=1e-5
        )
        assert ds["n_sources"].sel(nhm_id=1).values == 1
        # HRU 2: ci=0.50 < threshold → NaN bounds, n_sources=0
        assert np.isnan(ds["lower_bound"].sel(nhm_id=2).values).all()
        assert np.isnan(ds["upper_bound"].sel(nhm_id=2).values).all()
        assert ds["n_sources"].sel(nhm_id=2).values == 0
        # HRU 3: NaN CI → NaN bounds, n_sources=0
        assert np.isnan(ds["lower_bound"].sel(nhm_id=3).values).all()
        assert np.isnan(ds["upper_bound"].sel(nhm_id=3).values).all()
        assert ds["n_sources"].sel(nhm_id=3).values == 0


def test_n_sources_contract_when_ci_passes_but_snow_nan(tmp_path: Path):
    """CI passes at HRU scale but snow is NaN (pre-agg gate killed pixels).

    The n_sources contract is "1 = a finite bound was produced." Without
    the ``valid & sca_obs.notnull()`` guard, this scenario would write
    n_sources=1 with a NaN bound — a contract violation.
    """
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-15/2005-03-15",
        snow_native=np.nan,
        ci_native=90.0,  # CI gate passes
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        assert np.isnan(ds["lower_bound"].values).all()
        assert np.isnan(ds["upper_bound"].values).all()
        # Critical: n_sources must be 0, not 1 — no finite bound exists.
        assert (ds["n_sources"].values == 0).all()


def test_n_sources_value_set_is_zero_or_one(tmp_path: Path):
    """Defensive: n_sources values are exactly {0, 1} on disk, no sentinel."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    # Mixed scenario to cover both values.
    ci = np.array([[90.0, 50.0, 80.0]], dtype=np.float32)
    snow = np.array([[60.0, 60.0, 70.0]], dtype=np.float32)
    times_per_year = pd.date_range("2005-01-01", "2005-12-31", freq="D")
    snow_full = np.tile(snow, (len(times_per_year), 1))
    ci_full = np.tile(ci, (len(times_per_year), 1))

    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-15/2005-03-15",
        snow_native=snow_full,
        ci_native=ci_full,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        unique_vals = set(np.unique(ds["n_sources"].values).tolist())
        assert unique_vals <= {0, 1}, f"unexpected n_sources values: {unique_vals}"


# ---------------------------------------------------------------------------
# NN-fill companion content (not just file existence)
# ---------------------------------------------------------------------------


def test_nn_fill_companion_actually_fills_nan_cells(tmp_path: Path):
    """The ``_nn_filled.nc`` companion replaces NaN bounds with neighbor values.

    HRU 1 and HRU 3 have finite bounds; HRU 2 has NaN (CI < threshold).
    The NN fill should propagate a neighbor's bound to HRU 2 in the
    companion file, while the unfilled file retains HRU 2's NaN.

    The fabric in ``_write_synthetic_fabric`` places HRUs 1 and 3 as
    equidistant neighbors of HRU 2; both donors carry ``lower=0.54``,
    so the asserted value is tie-safe by construction — the cKDTree
    donor walk in ``normalize/methods.py:nn_fill_bounds`` may pick
    either, but both produce the same result.
    """
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    snow = np.array([[60.0, 60.0, 60.0]], dtype=np.float32)
    ci = np.array([[90.0, 50.0, 90.0]], dtype=np.float32)  # HRU 2 fails gate
    times_per_year = pd.date_range("2005-01-01", "2005-12-31", freq="D")
    snow_full = np.tile(snow, (len(times_per_year), 1))
    ci_full = np.tile(ci, (len(times_per_year), 1))

    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-15/2005-03-15",
        snow_native=snow_full,
        ci_native=ci_full,
        nn_fill=True,
    )
    project = load(workdir)
    build(project)
    # Unfilled: HRU 2 still NaN.
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        assert np.isnan(ds["lower_bound"].sel(nhm_id=2).values).all()
    # NN-filled: HRU 2 picks up a finite value from a neighbor (HRU 1
    # or HRU 3, both with lower=0.54).
    with xr.open_dataset(project.targets_dir() / "sca_targets_nn_filled.nc") as ds:
        filled = ds["lower_bound"].sel(nhm_id=2).values
        assert not np.isnan(filled).any(), "NN-fill should have replaced NaN"
        np.testing.assert_allclose(filled, 0.54, rtol=1e-5)
        # nn_filled flag variable should record HRU 2 as filled.
        assert "nn_filled" in ds.variables
        assert ds["nn_filled"].sel(nhm_id=2).values.any()


# ---------------------------------------------------------------------------
# Period spanning July/August boundary
# ---------------------------------------------------------------------------


def test_period_spanning_summer_boundary(tmp_path: Path):
    """A multi-month period exercises the time-aware xor between summer
    zero-forcing and the formula across month boundaries within one
    per-year NC. Catches off-by-one or wrong-dim broadcasts that
    single-day periods can't reach.
    """
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-06-28/2005-08-03",
        snow_native=60.0,
        ci_native=90.0,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        # June 28-30 (3 days): formula values (lower=0.54, upper=0.64).
        june_slice = ds.sel(time=slice("2005-06-28", "2005-06-30"))
        np.testing.assert_allclose(june_slice["lower_bound"].values, 0.54, rtol=1e-5)
        np.testing.assert_allclose(june_slice["upper_bound"].values, 0.64, rtol=1e-5)
        # July (31 days) + August 1-3 (3 days): forced to zero.
        zero_slice = ds.sel(time=slice("2005-07-01", "2005-08-03"))
        assert (zero_slice["lower_bound"].values == 0.0).all()
        assert (zero_slice["upper_bound"].values == 0.0).all()
        # n_sources stays 1 throughout — zero is the bound value, not a flag.
        assert (ds["n_sources"].values == 1).all()


# ---------------------------------------------------------------------------
# Multi-year stitch continuity
# ---------------------------------------------------------------------------


def test_multi_year_stitch_is_continuous_and_sorted(tmp_path: Path):
    """A period crossing Dec 31/Jan 1 must stitch with no gap, duplicate,
    or time-coord reorder, and HRU axis stays sorted ascending."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(tmp_path, period="2005-12-28/2006-01-05")
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        times = pd.DatetimeIndex(ds["time"].values)
        # 9 consecutive days across the year boundary.
        assert len(times) == 9
        assert times[0] == pd.Timestamp("2005-12-28")
        assert times[-1] == pd.Timestamp("2006-01-05")
        # Time monotonic increasing.
        assert (
            np.diff(times.values).astype("timedelta64[D]") == np.timedelta64(1, "D")
        ).all()
        # HRU axis sorted ascending by id_col (issue #93 invariant).
        hrus = ds["nhm_id"].values
        assert (np.diff(hrus) > 0).all()


# ---------------------------------------------------------------------------
# Idempotent skip with nn_fill=True
# ---------------------------------------------------------------------------


def test_idempotent_skip_with_nn_fill(tmp_path: Path, caplog):
    """The compound skip predicate has two branches; the nn_fill=True
    branch is exercised here. A second build with both intermediates
    present must log "skipping"; if only the nn intermediate is missing,
    the year must rebuild."""
    import logging

    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(tmp_path, period="2005-03-01/2005-03-15", nn_fill=True)
    project = load(workdir)
    build(project)
    # Second build with both intermediates present → skip, naming the
    # active fingerprints so a re-running operator can cross-check
    # (#213).
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="nhf_spatial_targets.targets.sca"):
        build(project)
    assert "skipping" in caplog.text
    assert "config=" in caplog.text
    assert "code=" in caplog.text
    # Delete only the nn intermediate; the predicate must fall through
    # and re-do the year (which regenerates the nn file).
    nn_intermediate = (
        project.targets_dir() / ".sca_intermediates" / "sca_targets_2005_nn_filled.nc"
    )
    nn_intermediate.unlink()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="nhf_spatial_targets.targets.sca"):
        build(project)
    assert "skipping" not in caplog.text
    assert nn_intermediate.exists()


# ---------------------------------------------------------------------------
# Failure-path clarity
# ---------------------------------------------------------------------------


def test_missing_aggregated_file_surfaces_clear_error(tmp_path: Path):
    """If MOD10C1 aggregated NCs are missing, the build raises with a
    message that names the source key and points the operator at the
    right ``agg`` command."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    # _make_sca_project writes the aggregated NCs; delete them to
    # simulate a project with no mod10c1 aggregation yet.
    workdir = _make_sca_project(tmp_path, period="2005-03-01/2005-03-31")
    agg_dir = workdir / "data" / "aggregated" / "mod10c1_v061"
    for nc in agg_dir.glob("*.nc"):
        nc.unlink()
    project = load(workdir)
    # The error must name the source key AND the corrective ``agg``
    # command so the operator can act on the message directly.
    with pytest.raises(FileNotFoundError, match="mod10c1_v061") as exc_info:
        build(project)
    assert "agg" in str(exc_info.value)


def test_hru_coord_mismatch_surfaces_clear_error(tmp_path: Path):
    """An aggregated NC built against a different HRU set must fail
    loudly via check_hru_coords rather than silently mis-aligning."""
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(tmp_path, period="2005-03-01/2005-03-31")
    # Rewrite the 2005 aggregated NC with a different HRU set (1, 2, 4
    # instead of 1, 2, 3).
    nc_path = (
        workdir / "data" / "aggregated" / "mod10c1_v061" / "mod10c1_v061_2005_agg.nc"
    )
    times = pd.date_range("2005-01-01", "2005-12-31", freq="D")
    bad_hrus = [1, 2, 4]
    snow = np.full((len(times), len(bad_hrus)), 60.0, dtype=np.float32)
    ci = np.full((len(times), len(bad_hrus)), 90.0, dtype=np.float32)
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (("time", "nhm_id"), snow),
            "Day_CMG_Clear_Index": (("time", "nhm_id"), ci),
        },
        coords={"time": times, "nhm_id": bad_hrus},
    )
    nc_path.unlink()
    ds.to_netcdf(nc_path)
    project = load(workdir)
    with pytest.raises(ValueError, match="HRU coords"):
        build(project)


# ---------------------------------------------------------------------------
# Operator-visible low-coverage warning
# ---------------------------------------------------------------------------


def test_low_valid_coverage_emits_warning(tmp_path: Path, caplog):
    """A year with ~zero passing CI logs a WARNING (mirrors the per-year
    warning in aggregate/mod10c1.py:_log_low_valid_coverage)."""
    import logging

    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-01/2005-03-31",
        snow_native=60.0,
        ci_native=10.0,  # fully below threshold for the whole year
    )
    project = load(workdir)
    with caplog.at_level(logging.WARNING, logger="nhf_spatial_targets.targets.sca"):
        build(project)
    assert any("valid CI-passing bound" in r.message for r in caplog.records)


def test_healthy_coverage_does_not_emit_warning(tmp_path: Path, caplog):
    """Negative-case guard: a healthy year (all-passing CI) must NOT log
    the low-coverage warning. Without this, a refactor that inverted the
    comparison (``>`` instead of ``<``) or dropped the threshold guard
    would flip the warning into an always-on fire and still pass the
    positive-case test above."""
    import logging

    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-01/2005-03-31",
        snow_native=60.0,
        ci_native=90.0,  # fully above threshold for the whole year
    )
    project = load(workdir)
    with caplog.at_level(logging.WARNING, logger="nhf_spatial_targets.targets.sca"):
        build(project)
    assert not any("valid CI-passing bound" in r.message for r in caplog.records)


def test_low_valid_coverage_partial_pass_no_warning(tmp_path: Path, caplog):
    """The warning predicate is strict ``<`` against ``_LOW_VALID_WARN_FRACTION``
    (0.01). 1/3 of HRUs passing on every day produces ~33% coverage —
    well above 1% — so the warning must NOT fire. Catches an off-by-one
    boundary regression (``>`` for ``<``) that would silently flip the
    warning to fire on every healthy year.
    """
    import logging

    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    # 1/3 HRUs pass on every day → frac_valid ≈ 0.33, well above 0.01.
    ci = np.array([[90.0, 50.0, 50.0]], dtype=np.float32)
    snow = np.array([[60.0, 60.0, 60.0]], dtype=np.float32)
    times_per_year = pd.date_range("2005-01-01", "2005-12-31", freq="D")
    ci_full = np.tile(ci, (len(times_per_year), 1))
    snow_full = np.tile(snow, (len(times_per_year), 1))

    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-01/2005-03-31",
        snow_native=snow_full,
        ci_native=ci_full,
    )
    project = load(workdir)
    with caplog.at_level(logging.WARNING, logger="nhf_spatial_targets.targets.sca"):
        build(project)
    assert not any("valid CI-passing bound" in r.message for r in caplog.records), (
        "1/3 coverage should clear the 1% threshold without warning"
    )


# ---------------------------------------------------------------------------
# Cache invalidation by fingerprint (#213)
# ---------------------------------------------------------------------------


def test_fingerprint_mismatch_rebuilds_year(tmp_path: Path, caplog):
    """Edit ci_threshold between two builds → the cached intermediate is
    automatically rebuilt with the new threshold, no manual ``rm`` needed.

    Regression guard for #213's core promise: a config change invalidates
    the cache without operator intervention.
    """
    import logging

    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    # First build: ci_threshold=0.70 (default). 60/90 input → lower=0.54.
    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-15/2005-03-15",
        snow_native=60.0,
        ci_native=90.0,
    )
    project = load(workdir)
    build(project)
    # Sanity: first build wrote bound 0.54.
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        np.testing.assert_allclose(ds["lower_bound"].values, 0.54, rtol=1e-5)
        first_fp = ds.attrs["config_fingerprint"]

    # Mutate ci_threshold to 0.95 (now strictly above the synthetic CI=0.90).
    cfg_path = workdir / "config.yml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["targets"]["snow_covered_area"]["ci_threshold"] = 0.95
    cfg_path.write_text(yaml.safe_dump(cfg))
    # Also delete the stitched output so we can confirm the rebuild
    # writes a fresh one consistent with the new threshold.
    (project.targets_dir() / "sca_targets.nc").unlink()

    project = load(workdir)
    caplog.clear()
    with caplog.at_level(
        logging.WARNING, logger="nhf_spatial_targets.targets._intermediates"
    ):
        build(project)

    # The fingerprint mismatch surfaced as a WARNING from the shared
    # skip predicate (logged under the _intermediates logger).
    assert any("mismatch" in r.message for r in caplog.records), (
        "fingerprint mismatch on cached intermediate must log a WARNING"
    )
    # Rebuild applied the new threshold: ci=0.90 < 0.95 → NaN bound.
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        assert np.isnan(ds["lower_bound"].values).all(), (
            "rebuilt year must reflect the new ci_threshold (0.95)"
        )
        second_fp = ds.attrs["config_fingerprint"]
    assert first_fp != second_fp


def test_pre_213_intermediate_triggers_rebuild(tmp_path: Path, caplog):
    """An on-disk intermediate without fingerprint attrs (legacy file
    from before #213) must be detected, deleted, and rebuilt rather
    than silently reused.

    Required for the upgrade transition.
    """
    import logging

    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-01/2005-03-15",
        snow_native=60.0,
        ci_native=90.0,
    )
    # Pre-populate intermediates_dir with a fake legacy file (no fingerprint).
    intermediates_dir = workdir / "targets" / ".sca_intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = intermediates_dir / "sca_targets_2005.nc"
    # Minimal valid NC with no fingerprint attrs.
    legacy = xr.Dataset(
        {"placeholder": (("time",), np.array([0.0], dtype=np.float32))},
        coords={"time": pd.date_range("2005-01-01", periods=1, freq="D")},
    )
    legacy.to_netcdf(legacy_path)
    project = load(workdir)
    caplog.clear()
    with caplog.at_level(
        logging.WARNING, logger="nhf_spatial_targets.targets._intermediates"
    ):
        build(project)
    # WARN + rebuild: the legacy file was deleted and a fresh
    # fingerprinted intermediate written in its place.
    assert any(
        ("predates" in r.message or "missing" in r.message) for r in caplog.records
    )
    rebuilt = xr.open_dataset(legacy_path)
    try:
        assert "config_fingerprint" in rebuilt.attrs
        assert "lower_bound" in rebuilt.data_vars
    finally:
        rebuilt.close()


def test_low_coverage_warning_reemits_on_cached_skip(tmp_path: Path, caplog):
    """The low-coverage WARNING must fire on EVERY build, including
    cached-skip re-runs (#213).

    First build writes a low-coverage year; second build skips the
    rebuild (fingerprints match) but reads ``frac_valid_bound`` from
    cached attrs and re-emits the warning so the operator sees the
    alert on every run, not just the first.

    Explicitly asserts the second build took the SKIP path (via the
    "intermediates valid" INFO log) so a regression that accidentally
    re-runs the year and emits the WARNING from the live computation
    would fail this test instead of silently passing.
    """
    import logging

    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(
        tmp_path,
        period="2005-03-01/2005-03-31",
        snow_native=60.0,
        ci_native=10.0,  # fully below threshold → low coverage warning
    )
    project = load(workdir)
    # First build: warning fires from the live computation.
    with caplog.at_level(logging.WARNING, logger="nhf_spatial_targets.targets.sca"):
        build(project)
    assert any("valid CI-passing bound" in r.message for r in caplog.records)
    first_warning_count = sum(
        1 for r in caplog.records if "valid CI-passing bound" in r.message
    )
    # Second build: intermediates skip-eligible. Warning must STILL fire
    # from the cached frac_valid_bound attr. Capture INFO too so we can
    # assert the skip path was actually taken (not a silent rebuild).
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="nhf_spatial_targets.targets.sca"):
        build(project)
    second_warning_count = sum(
        1
        for r in caplog.records
        if r.levelno >= logging.WARNING and "valid CI-passing bound" in r.message
    )
    assert second_warning_count >= 1, (
        "low-coverage WARNING must re-emit on cached-skip path"
    )
    # The two should be the same magnitude (1 warning per year both times).
    assert first_warning_count == second_warning_count
    # Regression guard: the second build must have hit the skip branch,
    # not silently re-run the year. The skip log line includes
    # "intermediates valid" + "skipping"; the rebuild path does not.
    assert any(
        "intermediates valid" in r.message and "skipping" in r.message
        for r in caplog.records
    ), "second build must take the cached-skip path, not silently rebuild"


def test_frac_valid_bound_not_in_stitched_output(tmp_path: Path):
    """The per-year ``frac_valid_bound`` attr must NOT appear on the
    stitched ``sca_targets.nc``.

    ``combine_attrs="override"`` in stitch picks the first per-year
    file's attrs by default, which would leak a single-year coverage
    fraction onto the multi-year stitched output as if it were a
    multi-year statistic. ``stitch_year_chunks_to_target`` strips it
    alongside ``year_chunk`` so an operator inspecting the stitched
    NC's attrs cannot accidentally trust a misleading single-year
    number.
    """
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    # Two-year period so the stitch actually runs.
    workdir = _make_sca_project(tmp_path, period="2005-01-01/2006-12-31")
    project = load(workdir)
    build(project)
    # Per-year intermediate carries the attr (forensic value).
    per_year = project.targets_dir() / ".sca_intermediates" / "sca_targets_2005.nc"
    with xr.open_dataset(per_year) as ds:
        assert "frac_valid_bound" in ds.attrs
    # Stitched output must NOT carry the per-year attr.
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        assert "frac_valid_bound" not in ds.attrs
        # Sanity: stitched output still carries the active fingerprints.
        assert "config_fingerprint" in ds.attrs
        assert "code_version" in ds.attrs


# ---------------------------------------------------------------------------
# Period-shrink defense (#211)
# ---------------------------------------------------------------------------


def test_period_shrink_prunes_orphan_intermediates(tmp_path: Path, caplog):
    """Shrinking the active period leaves out-of-period intermediates on
    disk from the wider previous build. The next build must prune them
    AND the stitched output must cover only the new (smaller) period.

    Regression guard for #211: without the prune + computed stitch
    input, the stitcher's directory glob would silently include the
    orphan 2006 file and produce a 2-year stitched NC instead of the
    1-year output the config requests.
    """
    import logging

    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    # First build: 2 years wide.
    workdir = _make_sca_project(tmp_path, period="2005-01-01/2006-12-31")
    project = load(workdir)
    build(project)
    intermediates_dir = project.targets_dir() / ".sca_intermediates"
    assert (intermediates_dir / "sca_targets_2005.nc").exists()
    assert (intermediates_dir / "sca_targets_2006.nc").exists()
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        years_first = sorted(set(pd.DatetimeIndex(ds["time"].values).year))
    assert years_first == [2005, 2006]

    # Shrink period: drop 2006. Delete the stitched output so we can
    # confirm the rebuild writes a fresh one matching the new period.
    cfg_path = workdir / "config.yml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["targets"]["snow_covered_area"]["period"] = "2005-01-01/2005-12-31"
    cfg_path.write_text(yaml.safe_dump(cfg))
    (project.targets_dir() / "sca_targets.nc").unlink()

    project = load(workdir)
    caplog.clear()
    # The prune WARNING is emitted under the caller's logger
    # (the sca module logger): the helper takes the caller's logger
    # as a parameter so the message appears under the right name in
    # operator logs.
    with caplog.at_level(logging.WARNING, logger="nhf_spatial_targets.targets.sca"):
        build(project)

    # 1. Orphan was pruned with a WARNING naming year 2006 + the
    # target label so operators can filter logs by target.
    assert any(
        "pruned" in r.message and "2006" in r.message and "sca" in r.message
        for r in caplog.records
    ), "prune WARNING must name the orphan year(s) and the target label"
    assert not (intermediates_dir / "sca_targets_2006.nc").exists()

    # 2. Stitched output covers ONLY 2005, not 2005-2006.
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        years_second = sorted(set(pd.DatetimeIndex(ds["time"].values).year))
    assert years_second == [2005], (
        f"stitched output must match shrunk period; got years {years_second}"
    )


def test_stitch_input_computed_from_period_not_glob(tmp_path: Path):
    """Even if a stale orphan intermediate slipped past the prune step
    (e.g. mid-pattern filename a future hash variant didn't strip), the
    stitch input must be computed from iter_period_years so the orphan
    cannot leak into the canonical output.

    Defense in depth: this test plants an orphan with a filename pattern
    the prune helper does NOT match (a 5-digit "year") and asserts the
    stitched output still excludes it.
    """
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_sca_project(tmp_path, period="2005-01-01/2005-12-31")
    project = load(workdir)
    # Pre-plant a misnamed orphan in the intermediates dir before build.
    intermediates_dir = project.targets_dir() / ".sca_intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    orphan = intermediates_dir / "sca_targets_99999.nc"
    # Realistic-looking but out-of-period file the prune regex won't catch.
    ds = xr.Dataset(
        {"placeholder": (("time",), np.array([0.0], dtype=np.float32))},
        coords={"time": pd.date_range("9999-01-01", periods=1, freq="D")},
    )
    ds.to_netcdf(orphan)
    project = load(workdir)
    build(project)
    # Orphan unaffected by the prune (its name doesn't match the
    # canonical year pattern), but the stitch input is computed from
    # year_specs so it must not leak.
    assert orphan.exists()
    with xr.open_dataset(project.targets_dir() / "sca_targets.nc") as ds:
        years = sorted(set(pd.DatetimeIndex(ds["time"].values).year))
    assert years == [2005]


def test_stitch_fails_loud_on_missing_in_period_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """If an in-period per-year intermediate is missing at stitch time
    (e.g. operator manually deleted one, or build crashed mid-loop and
    was restarted into the skip branch), the stitch must raise a clear
    error rather than silently producing a year-gapped stitched NC.

    Before #211 the stitcher globbed the directory, so a missing
    in-period file silently produced a year-gapped output. With the
    computed-from-year_specs input list, the missing file surfaces
    as an error naming the path.
    """
    import nhf_spatial_targets.targets._driver as driver_mod
    from nhf_spatial_targets.targets.sca import build
    from nhf_spatial_targets.workspace import load

    # Build a 3-year target so we have multiple intermediates to choose from.
    workdir = _make_sca_project(tmp_path, period="2005-01-01/2007-12-31")
    project = load(workdir)
    build(project)
    intermediates_dir = project.targets_dir() / ".sca_intermediates"
    # Delete the 2006 intermediate + the stitched output. Then patch the
    # driver's per-year write to a no-op so the missing 2006 intermediate
    # stays missing through the second build's stitch call (2005 + 2007
    # hit should_skip_year_build's healthy-cache branch and don't write;
    # 2006 would otherwise be rebuilt, but the no-op write swallows it).
    (intermediates_dir / "sca_targets_2006.nc").unlink()
    (project.targets_dir() / "sca_targets.nc").unlink()
    monkeypatch.setattr(driver_mod, "write_bounds_target", lambda **_kw: None)

    project = load(workdir)
    with pytest.raises((FileNotFoundError, OSError), match="2006"):
        build(project)
