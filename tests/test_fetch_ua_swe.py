"""Tests for UA SWE (NSIDC-0719) fetch + consolidation (issue #238).

Synthetic NetCDF fixtures mirror the layout of the real per-WY files
(``time`` as float32 days since 1900-01-01 with no ``units`` attr,
``SWE`` / ``DEPTH`` float32 in mm, NaN fill, CRS embedded in an ``|S1``
``crs`` scalar with NAD83 WKT) so the consolidation logic is exercised
end-to-end without touching the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from nhf_spatial_targets.fetch.ua_swe import (
    _assign_worker_water_years,
    _calendar_years_to_water_years,
    _mask_url,
    _wy_url,
    consolidate_water_year_ua_swe,
    fetch_ua_swe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_project(tmp_path: Path) -> Path:
    """Materialize a minimal valid project directory.

    The project loader requires ``config.yml`` and ``fabric.json`` to
    exist. UA SWE downloads are full-CONUS regardless of the project's
    fabric, so a stub fabric.json suffices.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    datastore = tmp_path / "datastore"
    datastore.mkdir()
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {
                    "path": str(tmp_path / "fabric.gpkg"),
                    "id_col": "nhm_id",
                },
                "datastore": str(datastore),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))
    return tmp_path


def _make_synthetic_raw_nc(
    path: Path,
    wy: int,
    *,
    n_time: int = 3,
    n_lat: int = 5,
    n_lon: int = 6,
    fill_fraction: float = 0.2,
    inject_negative: bool = False,
) -> None:
    """Write a synthetic per-WY raw NC matching the real publisher layout.

    Layout matches the NSIDC-0719 v1 files inspected at planning time:

    - ``time`` is float32 days since 1900-01-01, **no** ``units`` attr.
      (The consolidator decodes the epoch explicitly.)
    - ``lat`` / ``lon`` are float32 degrees in NAD83.
    - ``crs`` is an ``|S1`` scalar with a NAD83 WKT in ``spatial_ref``.
    - ``SWE`` / ``DEPTH`` are float32 ``(time, lat, lon)`` arrays in mm,
      NaN fill outside CONUS.
    """
    # Water year `wy` runs Oct 1 (wy-1) -> Sep 30 (wy). The first
    # ``n_time`` days of that span are encoded.
    start = pd.Timestamp(f"{wy - 1}-10-01")
    epoch = pd.Timestamp("1900-01-01")
    days_since_epoch = np.array(
        [(start + pd.Timedelta(days=i) - epoch).days for i in range(n_time)],
        dtype="float32",
    )

    # Tiny CONUS sub-grid (the consolidator's reprojection still runs;
    # rio.reproject on a 5x6 grid is fast).
    lat = np.linspace(30.0, 35.0, n_lat, dtype="float32")
    lon = np.linspace(-110.0, -100.0, n_lon, dtype="float32")

    rng = np.random.default_rng(seed=int(wy))
    swe = rng.uniform(0.0, 500.0, size=(n_time, n_lat, n_lon)).astype("float32")
    depth = rng.uniform(0.0, 2000.0, size=(n_time, n_lat, n_lon)).astype("float32")
    # Stamp NaN fill on a deterministic pattern (matches the real
    # "outside CONUS" mask convention; NaN, not -9999).
    n_fill = int(fill_fraction * swe.size)
    flat_idx = rng.choice(swe.size, size=n_fill, replace=False)
    swe.ravel()[flat_idx] = np.nan
    depth.ravel()[flat_idx] = np.nan
    if inject_negative:
        swe[0, 0, 0] = -9999.0

    ds = xr.Dataset(
        data_vars={
            "SWE": (
                ("time", "lat", "lon"),
                swe,
                {
                    "long_name": "Snow Water Equivalent",
                    "grid_mapping": "crs",
                    "units": "millimeters h20",
                },
            ),
            "DEPTH": (
                ("time", "lat", "lon"),
                depth,
                {
                    "long_name": "Snow Depth",
                    "grid_mapping": "crs",
                    "units": "millimeters snow thickness",
                },
            ),
            "crs": (
                (),
                b" ",
                {
                    "grid_mapping_name": "latitude_longitude",
                    "long_name": "CRS definition",
                    "spatial_ref": (
                        'GEOGCS["NAD83",DATUM["North_American_Datum_1983",'
                        'SPHEROID["GRS 1980",6378137,298.257222101]],'
                        'PRIMEM["Greenwich",0],UNIT["degree",0.01745329251994328],'
                        'AUTHORITY["EPSG","4269"]]'
                    ),
                },
            ),
        },
        coords={
            "time": (("time",), days_since_epoch),
            "lat": (("lat",), lat),
            "lon": (("lon",), lon),
        },
    )
    ds.to_netcdf(path)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


class TestCalendarYearsToWaterYears:
    def test_single_year(self):
        # CY 1990 touches WY 1990 (Jan-Sep) and WY 1991 (Oct-Dec).
        assert _calendar_years_to_water_years([1990]) == [1990, 1991]

    def test_multi_year_dedupes(self):
        # CY 1990, 1991 touches WYs {1990, 1991, 1991, 1992} → {1990, 1991, 1992}.
        assert _calendar_years_to_water_years([1990, 1991]) == [1990, 1991, 1992]

    def test_empty(self):
        assert _calendar_years_to_water_years([]) == []

    def test_sorted_output(self):
        assert _calendar_years_to_water_years([1995, 1990]) == [
            1990,
            1991,
            1995,
            1996,
        ]


class TestAssignWorkerWaterYears:
    def test_single_worker(self):
        assert _assign_worker_water_years([1982, 1983, 1984], 0, 1) == [
            1982,
            1983,
            1984,
        ]

    def test_round_robin(self):
        wys = [1982, 1983, 1984, 1985, 1986]
        assert _assign_worker_water_years(wys, 0, 2) == [1982, 1984, 1986]
        assert _assign_worker_water_years(wys, 1, 2) == [1983, 1985]

    def test_worker_index_out_of_range(self):
        with pytest.raises(ValueError, match="worker_index"):
            _assign_worker_water_years([1982], 2, 2)

    def test_negative_n_workers(self):
        with pytest.raises(ValueError, match="n_workers"):
            _assign_worker_water_years([1982], 0, 0)


class TestUrlConstruction:
    def test_wy_url(self):
        url = _wy_url("https://example.org/data/", 1990)
        assert url == "https://example.org/data/4km_SWE_Depth_WY1990_v01.nc"

    def test_mask_url(self):
        # Trailing slash is normalised.
        url = _mask_url("https://example.org/data/")
        assert url == "https://example.org/data/SWE_Mask_v01.nc"


# ---------------------------------------------------------------------------
# consolidate_water_year_ua_swe
# ---------------------------------------------------------------------------


class TestConsolidateWaterYear:
    def test_happy_path(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        daily_dir = tmp_path / "daily"
        raw_path = raw_dir / "4km_SWE_Depth_WY1982_v01.nc"
        _make_synthetic_raw_nc(raw_path, wy=1982)

        out_path = consolidate_water_year_ua_swe(1982, raw_path, daily_dir)
        assert out_path.exists()
        assert out_path.name == "ua_swe_daily_WY1982.nc"

        with xr.open_dataset(out_path) as ds:
            # Variable rename
            assert "swe" in ds.data_vars
            assert "snow_depth" in ds.data_vars
            assert "SWE" not in ds.data_vars
            assert "DEPTH" not in ds.data_vars

            # CF-1.6 compliance
            assert ds.attrs.get("Conventions") == "CF-1.6"

            # Catalog-driven units
            assert ds["swe"].attrs["units"] == "kg m-2"
            assert ds["snow_depth"].attrs["units"] == "mm"
            assert ds["swe"].attrs["grid_mapping"] == "crs"

            # Time decoded as real timestamps, not floats
            assert np.issubdtype(ds["time"].dtype, np.datetime64)
            # The synthetic raw started at 1981-10-01 (WY 1982 day 1).
            assert pd.Timestamp(ds["time"].values[0]) == pd.Timestamp("1981-10-01")

            # Reprojected to EPSG:5070 — spatial dims are (y, x), not (lat, lon)
            assert "y" in ds.dims
            assert "x" in ds.dims
            assert "lat" not in ds.dims
            assert "lon" not in ds.dims

            # CRS grid_mapping_name reflects Albers
            assert (
                ds["crs"].attrs.get("grid_mapping_name") == "albers_conical_equal_area"
            )

    def test_mtime_idempotent_skip(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        daily_dir = tmp_path / "daily"
        raw_path = raw_dir / "4km_SWE_Depth_WY1982_v01.nc"
        _make_synthetic_raw_nc(raw_path, wy=1982)

        # First call: build.
        out_path = consolidate_water_year_ua_swe(1982, raw_path, daily_dir)
        first_mtime = out_path.stat().st_mtime

        # Second call: should skip (raw_path mtime <= out_path mtime).
        out_path2 = consolidate_water_year_ua_swe(1982, raw_path, daily_dir)
        assert out_path2 == out_path
        assert out_path2.stat().st_mtime == pytest.approx(first_mtime, abs=1e-3)

    def test_rebuild_on_newer_raw(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        daily_dir = tmp_path / "daily"
        raw_path = raw_dir / "4km_SWE_Depth_WY1982_v01.nc"
        _make_synthetic_raw_nc(raw_path, wy=1982)

        out_path = consolidate_water_year_ua_swe(1982, raw_path, daily_dir)
        first_mtime = out_path.stat().st_mtime

        # Touch raw to a strictly later mtime.
        new_mtime = first_mtime + 100
        import os

        os.utime(raw_path, (new_mtime, new_mtime))

        out_path2 = consolidate_water_year_ua_swe(1982, raw_path, daily_dir)
        assert out_path2.stat().st_mtime > first_mtime

    def test_negative_value_defense(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        daily_dir = tmp_path / "daily"
        raw_path = raw_dir / "4km_SWE_Depth_WY1982_v01.nc"
        _make_synthetic_raw_nc(raw_path, wy=1982, inject_negative=True)

        with pytest.raises(ValueError, match="integer sentinel"):
            consolidate_water_year_ua_swe(1982, raw_path, daily_dir)

    def test_missing_raw_path(self, tmp_path: Path):
        daily_dir = tmp_path / "daily"
        with pytest.raises(FileNotFoundError, match="raw NC not found"):
            consolidate_water_year_ua_swe(
                1982, tmp_path / "does-not-exist.nc", daily_dir
            )


# ---------------------------------------------------------------------------
# fetch_ua_swe entry-point gates
# ---------------------------------------------------------------------------


class TestFetchPeriodGate:
    def test_period_below_publisher_window_rejected(self, tmp_path: Path):
        workdir = _make_project(tmp_path)
        # Catalog window is 1981/2023; CY 1970 must be rejected.
        with pytest.raises(ValueError, match="outside the ua_swe publisher window"):
            fetch_ua_swe(workdir=workdir, period="1970/1971")

    def test_period_above_publisher_window_rejected(self, tmp_path: Path):
        workdir = _make_project(tmp_path)
        with pytest.raises(ValueError, match="outside the ua_swe publisher window"):
            fetch_ua_swe(workdir=workdir, period="2024/2025")

    def test_malformed_period_rejected(self, tmp_path: Path):
        workdir = _make_project(tmp_path)
        with pytest.raises(ValueError, match="period"):
            fetch_ua_swe(workdir=workdir, period="not-a-period")


# ---------------------------------------------------------------------------
# Integration (skipped by default; opt-in via -m integration)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_one_water_year_live(tmp_path: Path):
    """Live download of one WY from NSIDC and consolidate.

    Requires Earthdata Login credentials in ~/.netrc (run
    ``nhf-targets materialize-credentials`` first). Skipped by default;
    opt in via ``pixi run -e dev test-integration``.
    """
    workdir = _make_project(tmp_path)
    result = fetch_ua_swe(
        workdir=workdir,
        period="2010/2010",  # touches WY 2010 + WY 2011
        worker_index=0,
        n_workers=1,
    )
    assert result["source_key"] == "ua_swe"
    assert len(result["water_years"]) >= 1
    # At least one WY should have consolidated successfully.
    assert any("daily_path" in rec for rec in result["water_years"])
