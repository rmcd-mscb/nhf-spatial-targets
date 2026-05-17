"""Tests for the soil moisture target builder end-to-end (monthly + annual)."""

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
    timestamp_style: str,
    values: np.ndarray,
    id_col: str = "nhm_id",
) -> None:
    """Write a per-year aggregated NC at monthly cadence.

    ``timestamp_style`` is one of:
      - ``"month_start"`` — 12 timestamps at YYYY-MM-01 (nldas_mosaic, nldas_noah)
      - ``"month_end"`` — 12 timestamps at YYYY-MM-(last) (ncep_ncar)
      - ``"mid_month"`` — 12 timestamps at YYYY-MM-15 (merra2)
    """
    if timestamp_style == "month_start":
        times = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    elif timestamp_style == "month_end":
        times = pd.date_range(f"{year}-01-31", f"{year}-12-31", freq="ME")
    elif timestamp_style == "mid_month":
        times = pd.DatetimeIndex([f"{year}-{m:02d}-15" for m in range(1, 13)])
    else:
        raise ValueError(f"Unknown timestamp_style: {timestamp_style!r}")
    assert values.shape[0] == 12
    ds = xr.Dataset(
        {var: (("time", id_col), values)},
        coords={"time": times, id_col: [1, 2, 3]},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _make_som_project(
    tmp_path: Path,
    *,
    period: str = "2000-01-01/2002-12-31",
    sources: list[str] | None = None,
    nn_fill: bool = True,
    normalize_period: str | None = None,
) -> Path:
    """Build a project skeleton with synthetic fabric + per-year aggregated NCs.

    Each source gets a year-by-year linear ramp at every month so the
    per-calendar-month normalization is exercised (multiple years per month).
    """
    if sources is None:
        sources = ["merra2", "ncep_ncar", "nldas_mosaic", "nldas_noah"]

    workdir = tmp_path / "proj"
    workdir.mkdir()
    fabric_path = tmp_path / "fabric.gpkg"
    _write_synthetic_fabric(fabric_path)
    (workdir / "fabric.json").write_text(json.dumps({"id_col": "nhm_id"}))

    cfg = {
        "datastore": str(tmp_path / "store"),
        "fabric": {"path": str(fabric_path), "id_col": "nhm_id"},
        "targets": {
            "soil_moisture": {
                "period": period,
                "normalize_period": normalize_period,
                "sources": sources,
                "nn_fill": nn_fill,
            },
            "runoff": {"enabled": False},
            "aet": {"enabled": False},
            "recharge": {"enabled": False},
            "snow_covered_area": {"enabled": False},
            "snow_water_equivalent": {"enabled": False},
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

    spec = {
        "merra2": ("GWETTOP", "mid_month", 0.5),
        "ncep_ncar": ("soilw_0_10cm", "month_end", 0.3),
        "nldas_mosaic": ("SoilM_0_10cm", "month_start", 30.0),
        "nldas_noah": ("SoilM_0_10cm", "month_start", 25.0),
    }
    for src in sources:
        if src not in spec:
            # Unknown source — caller is exercising the "unknown source" error
            # path; skip writing NCs we have no synthetic data spec for.
            continue
        var, style, base = spec[src]
        for i, year in enumerate(years):
            # Year-over-year ramp 1.0 → 1.5 to give per-calendar-month
            # normalization a non-degenerate min/max range. Per HRU, all 3
            # HRUs share the same value (synthetic uniform field).
            scale = 1.0 + 0.5 * (i / max(len(years) - 1, 1))
            arr = np.full((12, 3), base * scale, dtype=np.float32)
            _write_monthly_nc(
                agg_dir / src / f"{src}_{year}_agg.nc",
                year,
                var,
                style,
                arr,
            )
    return workdir


# ---------------------------------------------------------------------------
# Per-source shim + registry
# ---------------------------------------------------------------------------


def test_som_passthrough_returns_input_unchanged():
    from nhf_spatial_targets.targets.som import som_passthrough

    da = xr.DataArray(
        np.array([[0.5, 0.6, 0.7]], dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(["2000-01-15"]), "nhm_id": [1, 2, 3]},
        attrs={"units": "1"},
    )
    out = som_passthrough(da)
    np.testing.assert_array_equal(out.values, da.values)
    # Identity contract: same object reference is fine; alternatively assert
    # values + attrs equality.
    assert out.attrs == da.attrs


def test_shims_registered_and_well_formed():
    """SHIMS exposes all four expected source keys with the right aggregated_vars."""
    from nhf_spatial_targets.targets.som import SHIMS

    by_key = {s.source_key: s for s in SHIMS}
    assert set(by_key) == {"merra2", "ncep_ncar", "nldas_mosaic", "nldas_noah"}
    assert by_key["merra2"].aggregated_var == "GWETTOP"
    assert by_key["ncep_ncar"].aggregated_var == "soilw_0_10cm"
    assert by_key["nldas_mosaic"].aggregated_var == "SoilM_0_10cm"
    assert by_key["nldas_noah"].aggregated_var == "SoilM_0_10cm"


# ---------------------------------------------------------------------------
# End-to-end build (monthly + annual variants)
# ---------------------------------------------------------------------------


def test_build_writes_both_monthly_and_annual_files(tmp_path: Path):
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(tmp_path, period="2000-01-01/2002-12-31")
    project = load(workdir)
    build(project)
    out = project.targets_dir()
    assert (out / "soil_moisture_targets_monthly.nc").exists()
    assert (out / "soil_moisture_targets_monthly_nn_filled.nc").exists()
    assert (out / "soil_moisture_targets_annual.nc").exists()
    assert (out / "soil_moisture_targets_annual_nn_filled.nc").exists()


def test_build_monthly_output_schema(tmp_path: Path):
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(tmp_path, period="2000-01-01/2002-12-31", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(
        project.targets_dir() / "soil_moisture_targets_monthly.nc"
    ) as ds:
        assert "lower_bound" in ds and "upper_bound" in ds
        assert "n_sources" in ds
        assert ds["lower_bound"].attrs["units"] == "1"
        assert ds.attrs["Conventions"] == "CF-1.6"
        assert ds.attrs["normalize_method"] == "per_calendar_month"
        # 36 monthly timesteps for 2000-2002
        assert ds.sizes["time"] == 36


def test_build_annual_output_schema(tmp_path: Path):
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(tmp_path, period="2000-01-01/2002-12-31", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(
        project.targets_dir() / "soil_moisture_targets_annual.nc"
    ) as ds:
        assert "lower_bound" in ds and "upper_bound" in ds
        assert "n_sources" in ds
        assert ds["lower_bound"].attrs["units"] == "1"
        assert ds.attrs["Conventions"] == "CF-1.6"
        assert ds.attrs["normalize_method"] == "whole_period"
        assert ds.attrs["annual_aggregation"] == "mean"
        # 3 annual timesteps for 2000-2002
        assert ds.sizes["time"] == 3


def test_build_monthly_bounds_in_0_1_range(tmp_path: Path):
    """Per-calendar-month normalization: 0 <= lower <= upper <= 1 everywhere."""
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(tmp_path, period="2000-01-01/2002-12-31", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(
        project.targets_dir() / "soil_moisture_targets_monthly.nc"
    ) as ds:
        finite_lo = ds["lower_bound"].values[np.isfinite(ds["lower_bound"].values)]
        finite_up = ds["upper_bound"].values[np.isfinite(ds["upper_bound"].values)]
        assert (finite_lo >= 0).all()
        assert (finite_lo <= 1).all()
        assert (finite_up >= 0).all()
        assert (finite_up <= 1).all()


def test_build_annual_bounds_in_0_1_range(tmp_path: Path):
    """Whole-period normalization: 0 <= lower <= upper <= 1 everywhere."""
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(tmp_path, period="2000-01-01/2002-12-31", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(
        project.targets_dir() / "soil_moisture_targets_annual.nc"
    ) as ds:
        finite_lo = ds["lower_bound"].values[np.isfinite(ds["lower_bound"].values)]
        finite_up = ds["upper_bound"].values[np.isfinite(ds["upper_bound"].values)]
        assert (finite_lo >= 0).all()
        assert (finite_lo <= 1).all()
        assert (finite_up >= 0).all()
        assert (finite_up <= 1).all()


def test_build_n_sources_full_coverage(tmp_path: Path):
    """All four sources present throughout → n_sources=4 everywhere (both outputs)."""
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(tmp_path, period="2000-01-01/2002-12-31", nn_fill=False)
    project = load(workdir)
    build(project)
    for fname in (
        "soil_moisture_targets_monthly.nc",
        "soil_moisture_targets_annual.nc",
    ):
        with xr.open_dataset(project.targets_dir() / fname) as ds:
            assert (ds["n_sources"].values == 4).all(), f"{fname} has gaps"


def test_build_monthly_normalize_per_calendar_month(tmp_path: Path):
    """First year is per-month min, last year is per-month max.

    With the synthetic year-over-year ramp (1.0 → 1.5x), every January
    across years should normalize to [0, 1] where 2000-01 is the min (0)
    and the last year's January is the max (1). Similarly for every other
    calendar month.
    """
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(tmp_path, period="2000-01-01/2002-12-31", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(
        project.targets_dir() / "soil_moisture_targets_monthly.nc"
    ) as ds:
        # Pick HRU 1's bounds; all sources scale together so lower == upper.
        lo = ds["lower_bound"].values[:, 0]
        # 36 months: indices 0,1,...,11 = year 2000 months 1..12
        #            indices 12,...,23 = year 2001 months 1..12
        #            indices 24,...,35 = year 2002 months 1..12
        # Within each calendar-month group (e.g. all Januaries at indices 0, 12, 24),
        # the ramp gives normalized values [0, 0.5, 1.0].
        for month_idx in range(12):
            jan_like = lo[month_idx::12]
            np.testing.assert_allclose(jan_like, [0.0, 0.5, 1.0], atol=1e-6)


def test_build_annual_aggregation_is_mean(tmp_path: Path):
    """Annual aggregation collapses each year's 12 monthly values to a mean.

    With the synthetic ramp scaling identically across calendar months
    within a year, the annual mean equals the year's scale × base.
    Normalization over 3 years gives [0, 0.5, 1.0].
    """
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(tmp_path, period="2000-01-01/2002-12-31", nn_fill=False)
    project = load(workdir)
    build(project)
    with xr.open_dataset(
        project.targets_dir() / "soil_moisture_targets_annual.nc"
    ) as ds:
        lo = ds["lower_bound"].values[:, 0]
        np.testing.assert_allclose(lo, [0.0, 0.5, 1.0], atol=1e-6)


def test_build_source_attr_reflects_active_sources(tmp_path: Path):
    """Dropping a source from config removes it from the output's source attr."""
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(
        tmp_path,
        period="2000-01-01/2002-12-31",
        sources=["merra2", "ncep_ncar"],
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(
        project.targets_dir() / "soil_moisture_targets_monthly.nc"
    ) as ds:
        src_attr = ds.attrs["source"]
        assert "MERRA-2" in src_attr
        assert "NCEP/NCAR" in src_attr
        assert "NLDAS" not in src_attr


def test_build_unknown_source_raises(tmp_path: Path):
    """soil_moisture.sources with an unknown key raises before any IO."""
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(
        tmp_path,
        period="2000-01-01/2002-12-31",
        sources=["merra2", "not_a_real_source"],
        nn_fill=False,
    )
    project = load(workdir)
    with pytest.raises(ValueError, match="unknown source 'not_a_real_source'"):
        build(project)


def test_build_hru_mismatch_raises(tmp_path: Path):
    """Source aggregated to a different HRU set than the fabric → raise."""
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(tmp_path, period="2000-01-01/2002-12-31", nn_fill=False)
    bad = workdir / "data" / "aggregated" / "merra2" / "merra2_2001_agg.nc"
    times = pd.DatetimeIndex([f"2001-{m:02d}-15" for m in range(1, 13)])
    hrus = [1, 2, 99]
    arr = np.full((12, 3), 0.5, dtype=np.float32)
    ds = xr.Dataset(
        {"GWETTOP": (("time", "nhm_id"), arr)},
        coords={"time": times, "nhm_id": hrus},
    )
    bad.unlink()
    ds.to_netcdf(bad)

    project = load(workdir)
    with pytest.raises(ValueError, match="HRU coords differ.*as sets"):
        build(project)


# ---------------------------------------------------------------------------
# normalize_period config knob
# ---------------------------------------------------------------------------


def test_build_normalize_period_defaults_to_period(tmp_path: Path):
    """When normalize_period is None, behavior matches the no-window primitive."""
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(
        tmp_path,
        period="2000-01-01/2002-12-31",
        normalize_period=None,
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(
        project.targets_dir() / "soil_moisture_targets_annual.nc"
    ) as ds:
        # Default behavior unchanged from PR base: ramp 2000→2002 normalizes to 0,0.5,1.
        np.testing.assert_allclose(
            ds["lower_bound"].values[:, 0], [0.0, 0.5, 1.0], atol=1e-6
        )
        assert ds.attrs["normalize_period"] == "2000-01-01/2002-12-31"


def test_build_normalize_period_narrower_than_period_extrapolates_above_1(
    tmp_path: Path,
):
    """normalize_period < period: years past the window normalize above 1.

    Use a 5-year period with normalize_period covering only the first 3 years.
    The ramp (1.0, 1.125, 1.25, 1.375, 1.5x) means the first 3 years' annual
    means normalize to [0, 0.5, 1.0]; the 4th year (1.375x) is above the
    window max so normalizes to 1.5 (visibly out-of-range, by design).
    """
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(
        tmp_path,
        period="2000-01-01/2004-12-31",
        normalize_period="2000-01-01/2002-12-31",
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(
        project.targets_dir() / "soil_moisture_targets_annual.nc"
    ) as ds:
        lo = ds["lower_bound"].values[:, 0]
        # Window years 2000, 2001, 2002 → 0, 0.5, 1.0
        np.testing.assert_allclose(lo[:3], [0.0, 0.5, 1.0], atol=1e-6)
        # Years 2003 and 2004 extrapolate above 1 (by design)
        assert lo[3] > 1.0
        assert lo[4] > lo[3]
        assert ds.attrs["normalize_period"] == "2000-01-01/2002-12-31"


def test_build_normalize_period_monthly_variant_records_attr(tmp_path: Path):
    """Monthly variant records the normalize_period in its global attrs."""
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(
        tmp_path,
        period="2000-01-01/2002-12-31",
        normalize_period="2000-01-01/2001-12-31",
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(
        project.targets_dir() / "soil_moisture_targets_monthly.nc"
    ) as ds:
        assert ds.attrs["normalize_period"] == "2000-01-01/2001-12-31"


# ---------------------------------------------------------------------------
# NN-fill effect
# ---------------------------------------------------------------------------


def test_build_nn_fill_actually_fills_nan_cells(tmp_path: Path):
    """End-to-end NN-fill for the SOM monthly variant: a NaN HRU is filled in
    the *_nn_filled.nc companion and the nn_filled flag is set."""
    from nhf_spatial_targets.targets.som import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_som_project(
        tmp_path,
        period="2000-01-01/2002-12-31",
        sources=["merra2"],  # single source so any NaN propagates to bound NaN
        nn_fill=True,
    )
    # Rewrite each merra2 NC so HRU 2 is NaN throughout.
    for year in (2000, 2001, 2002):
        path = workdir / "data" / "aggregated" / "merra2" / f"merra2_{year}_agg.nc"
        path.unlink()
        times = pd.DatetimeIndex([f"{year}-{m:02d}-15" for m in range(1, 13)])
        # Per-month base value × year-over-year ramp so the normalization
        # produces non-degenerate bounds for HRUs 1 and 3.
        scale = 1.0 + 0.5 * ((year - 2000) / 2)
        arr = np.full((12, 3), 0.5 * scale, dtype=np.float32)
        arr[:, 1] = np.nan
        ds = xr.Dataset(
            {"GWETTOP": (("time", "nhm_id"), arr)},
            coords={"time": times, "nhm_id": [1, 2, 3]},
        )
        ds.to_netcdf(path)

    project = load(workdir)
    build(project)

    monthly = project.targets_dir() / "soil_moisture_targets_monthly.nc"
    monthly_nn = project.targets_dir() / "soil_moisture_targets_monthly_nn_filled.nc"

    # Honest-NaN file: HRU 2 NaN throughout, n_sources=0 there.
    with xr.open_dataset(monthly) as raw:
        assert np.isnan(raw["lower_bound"].values[:, 1]).all()
        assert (raw["n_sources"].values[:, 1] == 0).all()

    # NN-filled file: HRU 2 now finite, nn_filled=1 there.
    assert monthly_nn.exists()
    with xr.open_dataset(monthly_nn) as filled:
        assert "nn_filled" in filled.data_vars
        assert np.isfinite(filled["lower_bound"].values[:, 1]).all()
        assert (filled["nn_filled"].values[:, 1] == 1).all()
        assert (filled["nn_filled"].values[:, 0] == 0).all()
