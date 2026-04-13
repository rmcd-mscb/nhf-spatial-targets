from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.fetch.era5_land import daily_to_monthly, hourly_to_daily


def _make_hourly(start: str, hours: int, value_per_hour: float = 1.0) -> xr.DataArray:
    """Synthetic ERA5-Land accumulated field.

    ERA5-Land accumulates from 00 UTC: at hour H, value = H * per_hour
    (resetting to 0 at the next 00 UTC, where it then represents the
    accumulation step from 23->00 of the prior day).
    """
    times = pd.date_range(start, periods=hours, freq="1h")
    vals = np.zeros(hours)
    for i, t in enumerate(times):
        # value at time t = accumulation since 00 UTC of t's date
        hours_since_midnight = t.hour if t.hour != 0 else (24 if i > 0 else 0)
        vals[i] = hours_since_midnight * value_per_hour
    da = xr.DataArray(
        vals.reshape(-1, 1, 1),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": [40.0], "longitude": [-100.0]},
        name="ro",
    )
    return da


def _make_era5_midnight_reset(
    n_days: int = 2, value_per_hour: float = 1.0
) -> xr.DataArray:
    """Synthetic ERA5-Land accumulation with true midnight resets.

    ERA5-Land convention:
    - At 01:00 through 23:00 of day D: value = hours_since_midnight * per_hour
    - At 00:00 of day D+1: value = 24 * per_hour  (full-day-D accumulation;
      reset happens AFTER this timestamp)
    - At 01:00 of day D+1: value = 1 * per_hour   (new day, accumulation restarts)

    This means the raw .diff() at the 00->01 UTC boundary is negative, which
    is the sign that triggers the xr.where rescue branch in hourly_to_daily().
    """
    # Build timestamps from day 1 00:00 through day n_days 00:00 (inclusive of
    # the last "full-day" accumulation value)
    start = pd.Timestamp("2020-01-01 00:00")
    # We need n_days * 24 + 1 timestamps: hours 0..24 for day1, then 1..24 for day2, etc.
    # Simplest: build 25*n_days steps and let the accumulation logic fill them
    hours = n_days * 24 + 1
    times = pd.date_range(start, periods=hours, freq="1h")
    vals = np.zeros(hours)
    for i, t in enumerate(times):
        h = t.hour
        # At 00:00: value = 24 (full-day accumulation from prior day, not yet reset)
        # But only for timestamps after the very first 00:00
        if h == 0 and i > 0:
            vals[i] = 24.0 * value_per_hour
        else:
            vals[i] = h * value_per_hour
    da = xr.DataArray(
        vals.reshape(-1, 1, 1),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": [40.0], "longitude": [-100.0]},
        name="ro",
        attrs={"units": "m"},
    )
    return da


def test_midnight_reset_branch_is_exercised():
    """Verify xr.where substitution is triggered at midnight resets.

    Build a 2-day synthetic array where the 00 UTC value is the full prior-day
    accumulation (not yet reset). Confirm:
    1. The raw diff IS negative at the 00 UTC reset (the rescue branch fires).
    2. Each complete day sums to 24 * per_hour.
    """
    per_hour = 0.001  # 1 mm per step
    da = _make_era5_midnight_reset(n_days=2, value_per_hour=per_hour)

    # Confirm the raw diff is negative at the 00 UTC reset boundary (day2 00:00).
    # That is the hour where val goes from 23*per_hour (day1 23:00) up to
    # 24*per_hour (day2 00:00) — wait, that's positive. But from 24*per_hour
    # (day2 00:00) to 1*per_hour (day2 01:00), the diff IS negative.
    raw_diff = da.diff("time", label="upper")
    # Find the step where we go from 24 to 1 (day2 01:00)
    negative_mask = (raw_diff < 0).values.squeeze()
    assert negative_mask.any(), (
        "Expected at least one negative diff (midnight reset) but found none. "
        "The xr.where rescue branch would never be triggered."
    )

    daily = hourly_to_daily(da)

    # Each complete day should equal 24 * per_hour
    expected = 24.0 * per_hour
    for i in range(min(daily.sizes["time"], 2)):
        np.testing.assert_allclose(
            daily.isel(time=i).values,
            expected,
            rtol=1e-6,
            err_msg=f"Day {i} daily sum incorrect; midnight-reset logic may be broken",
        )


def test_hourly_to_daily_full_24h():
    # 48 hours starting 2020-01-01 00:00 → two full days of accumulation
    da = _make_hourly("2020-01-01 00:00", hours=49, value_per_hour=0.001)
    daily = hourly_to_daily(da)
    # Two complete days should each show 24 * 0.001 = 0.024 m
    assert daily.sizes["time"] == 2
    np.testing.assert_allclose(daily.isel(time=0).values, 0.024, rtol=1e-6)
    np.testing.assert_allclose(daily.isel(time=1).values, 0.024, rtol=1e-6)


def test_hourly_to_daily_preserves_units_attr():
    da = _make_hourly("2020-01-01 00:00", hours=49, value_per_hour=0.001)
    da.attrs["units"] = "m"
    daily = hourly_to_daily(da)
    assert daily.attrs["units"] == "m"


def test_daily_to_monthly_sum():
    times = pd.date_range("2020-01-01", periods=60, freq="1D")
    vals = np.full((60, 1, 1), 0.001)
    da = xr.DataArray(
        vals,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": [40.0], "longitude": [-100.0]},
        name="ro",
        attrs={"units": "m"},
    )
    monthly = daily_to_monthly(da)
    # January (31 days) and February 2020 (29 days, leap year)
    assert monthly.sizes["time"] == 2
    np.testing.assert_allclose(monthly.isel(time=0).values, 0.031, rtol=1e-6)
    np.testing.assert_allclose(monthly.isel(time=1).values, 0.029, rtol=1e-6)
    assert monthly.attrs["units"] == "m"


def test_download_month_variable_calls_cds_client(tmp_path, monkeypatch):
    """download_month_variable submits a single-month CDS request."""
    from nhf_spatial_targets.fetch.era5_land import download_month_variable

    out = tmp_path / "era5_land_ro_2020_03.nc"

    def fake_retrieve(dataset, request, path):
        Path(path).write_bytes(b"fake_nc_data")

    fake_client = MagicMock()
    fake_client.retrieve = MagicMock(side_effect=fake_retrieve)
    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.era5_land._cds_client", lambda: fake_client
    )

    result = download_month_variable(year=2020, month=3, variable="ro", output_path=out)

    fake_client.retrieve.assert_called_once()
    args, _ = fake_client.retrieve.call_args
    assert args[0] == "reanalysis-era5-land"
    request = args[1]
    assert request["variable"] == "runoff"
    assert request["year"] == "2020"
    assert request["month"] == "03"
    assert request["area"] == [53.0, -125.0, 24.7, -66.0]
    assert request["format"] == "netcdf"
    assert len(request["day"]) == 31
    assert len(request["time"]) == 24
    assert out.exists()
    assert not Path(str(out) + ".tmp").exists()
    assert result == out


def test_download_month_variable_skips_existing(tmp_path, monkeypatch):
    from nhf_spatial_targets.fetch.era5_land import download_month_variable

    fake_client = MagicMock()
    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.era5_land._cds_client", lambda: fake_client
    )
    out = tmp_path / "era5_land_ro_2020_03.nc"
    out.write_bytes(b"existing")
    download_month_variable(year=2020, month=3, variable="ro", output_path=out)
    fake_client.retrieve.assert_not_called()


def test_download_month_variable_atomic_cleanup_on_failure(tmp_path, monkeypatch):
    """If CDS retrieve raises, the .tmp file is cleaned up."""
    from nhf_spatial_targets.fetch.era5_land import download_month_variable

    out = tmp_path / "era5_land_ro_2020_03.nc"

    def fake_retrieve_fail(dataset, request, path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("CDS server error")

    fake_client = MagicMock()
    fake_client.retrieve = MagicMock(side_effect=fake_retrieve_fail)
    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.era5_land._cds_client", lambda: fake_client
    )

    import pytest

    with pytest.raises(RuntimeError, match="CDS server error"):
        download_month_variable(year=2020, month=3, variable="ro", output_path=out)

    assert not out.exists()
    assert not Path(str(out) + ".tmp").exists()


def test_download_year_calls_cds_client(tmp_path, monkeypatch):
    """download_year_variable submits 12 single-month CDS requests."""
    from nhf_spatial_targets.fetch.era5_land import download_year_variable

    out = tmp_path / "era5_land_ro_2020.nc"

    def fake_retrieve(dataset, request, path):
        # Write a minimal valid NetCDF so the concatenation step succeeds.
        month = int(request["month"])
        times = pd.date_range(f"2020-{month:02d}-01", periods=1, freq="1h")
        ds = xr.Dataset(
            {"ro": (("time", "latitude", "longitude"), np.zeros((1, 1, 1)))},
            coords={"time": times, "latitude": [40.0], "longitude": [-100.0]},
        )
        ds.to_netcdf(path)

    fake_client = MagicMock()
    fake_client.retrieve = MagicMock(side_effect=fake_retrieve)
    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.era5_land._cds_client", lambda: fake_client
    )

    result = download_year_variable(year=2020, variable="ro", output_path=out)

    assert fake_client.retrieve.call_count == 12
    for i, (args, _) in enumerate(fake_client.retrieve.call_args_list, start=1):
        assert args[0] == "reanalysis-era5-land"
        request = args[1]
        assert request["variable"] == "runoff"
        assert request["year"] == "2020"
        assert request["month"] == f"{i:02d}"
        assert request["area"] == [53.0, -125.0, 24.7, -66.0]
        assert request["format"] == "netcdf"
    assert out.exists(), "Year file should exist after all monthly chunks are assembled"
    assert not Path(str(out) + ".tmp").exists()
    assert result == out


def test_download_year_atomic_cleanup_on_failure(tmp_path, monkeypatch):
    """If a monthly CDS download fails, the .tmp is cleaned up and year file absent."""
    from nhf_spatial_targets.fetch.era5_land import download_year_variable

    out = tmp_path / "era5_land_ro_2020.nc"
    chunk_01_tmp = tmp_path / "era5_land_ro_2020_01.nc.tmp"

    def fake_retrieve_fail(dataset, request, path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("CDS server error")

    fake_client = MagicMock()
    fake_client.retrieve = MagicMock(side_effect=fake_retrieve_fail)
    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.era5_land._cds_client", lambda: fake_client
    )

    import pytest

    with pytest.raises(RuntimeError, match="CDS server error"):
        download_year_variable(year=2020, variable="ro", output_path=out)

    assert not out.exists(), "Year file must not exist after a failed monthly download"
    assert not chunk_01_tmp.exists(), ".tmp for the failed month must be cleaned up"


def test_download_year_skips_existing(tmp_path, monkeypatch):
    from nhf_spatial_targets.fetch.era5_land import download_year_variable

    fake_client = MagicMock()
    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.era5_land._cds_client", lambda: fake_client
    )
    out = tmp_path / "era5_land_ro_2020.nc"
    out.write_bytes(b"existing")
    download_year_variable(year=2020, variable="ro", output_path=out)
    fake_client.retrieve.assert_not_called()


def test_consolidate_year_writes_daily_and_updates_monthly(tmp_path, monkeypatch):
    """Given pre-existing hourly NCs, consolidation produces daily and monthly NCs."""
    from nhf_spatial_targets.fetch.era5_land import consolidate_year

    hourly_dir = tmp_path / "hourly"
    daily_dir = tmp_path / "daily"
    monthly_dir = tmp_path / "monthly"
    hourly_dir.mkdir()

    times = pd.date_range("2020-01-01", "2020-01-03 23:00", freq="1h")
    for var in ("ro", "sro", "ssro"):
        vals = np.tile(
            np.arange(24, dtype=float).reshape(24, 1, 1) * 0.001,
            (len(times) // 24, 1, 1),
        )
        ds = xr.Dataset(
            {var: (("time", "latitude", "longitude"), vals)},
            coords={
                "time": times[: vals.shape[0]],
                "latitude": [40.0],
                "longitude": [-100.0],
            },
        )
        ds[var].attrs["units"] = "m"
        ds.to_netcdf(hourly_dir / f"era5_land_{var}_2020.nc")

    daily_path, monthly_path = consolidate_year(
        year=2020,
        hourly_dir=hourly_dir,
        daily_dir=daily_dir,
        monthly_dir=monthly_dir,
    )

    assert daily_path.exists()
    daily = xr.open_dataset(daily_path)
    try:
        assert {"ro", "sro", "ssro"}.issubset(set(daily.data_vars))
        assert daily.sizes["time"] >= 2
    finally:
        daily.close()

    assert monthly_path.exists()


def _write_hourly_ncs(hourly_dir: Path, year: int) -> None:
    """Write synthetic hourly NCs for all three ERA5-Land runoff variables."""

    times = pd.date_range(f"{year}-01-01", f"{year}-01-03 23:00", freq="1h")
    for var in ("ro", "sro", "ssro"):
        vals = np.tile(
            np.arange(24, dtype=float).reshape(24, 1, 1) * 0.001,
            (len(times) // 24, 1, 1),
        )
        ds = xr.Dataset(
            {var: (("time", "latitude", "longitude"), vals)},
            coords={
                "time": times[: vals.shape[0]],
                "latitude": [40.0],
                "longitude": [-100.0],
            },
        )
        ds[var].attrs["units"] = "m"
        ds.to_netcdf(hourly_dir / f"era5_land_{var}_{year}.nc")


def test_consolidate_year_writes_global_attrs(tmp_path):
    """Daily and monthly NCs include source-level global attrs."""
    from nhf_spatial_targets.fetch.era5_land import consolidate_year

    hourly_dir = tmp_path / "hourly"
    daily_dir = tmp_path / "daily"
    monthly_dir = tmp_path / "monthly"
    hourly_dir.mkdir()

    _write_hourly_ncs(hourly_dir, 2020)

    daily_path, monthly_path = consolidate_year(
        year=2020,
        hourly_dir=hourly_dir,
        daily_dir=daily_dir,
        monthly_dir=monthly_dir,
    )

    daily = xr.open_dataset(daily_path)
    try:
        assert "title" in daily.attrs, "daily NC missing 'title' global attr"
        assert "institution" in daily.attrs, (
            "daily NC missing 'institution' global attr"
        )
        assert "references" in daily.attrs, "daily NC missing 'references' global attr"
        assert "frequency" in daily.attrs, "daily NC missing 'frequency' global attr"
        assert daily.attrs["institution"] == "ECMWF"
        assert "doi:10.5194/essd-13-4349-2021" in daily.attrs["references"]
        assert daily.attrs["frequency"] == "day"
        assert "source" in daily.attrs
        assert daily.attrs["source"] == "reanalysis-era5-land"
    finally:
        daily.close()

    monthly = xr.open_dataset(monthly_path)
    try:
        assert "title" in monthly.attrs, "monthly NC missing 'title' global attr"
        assert "institution" in monthly.attrs, (
            "monthly NC missing 'institution' global attr"
        )
        assert "references" in monthly.attrs, (
            "monthly NC missing 'references' global attr"
        )
        assert "frequency" in monthly.attrs, (
            "monthly NC missing 'frequency' global attr"
        )
        assert monthly.attrs["institution"] == "ECMWF"
        assert "doi:10.5194/essd-13-4349-2021" in monthly.attrs["references"]
        assert monthly.attrs["frequency"] == "month"
        assert "source" in monthly.attrs
        assert monthly.attrs["source"] == "reanalysis-era5-land"
    finally:
        monthly.close()


def test_consolidate_year_is_idempotent(tmp_path, monkeypatch):
    """Re-running consolidate_year on a year whose daily NC already exists is a no-op."""
    from unittest.mock import patch

    from nhf_spatial_targets.fetch.era5_land import consolidate_year

    hourly_dir = tmp_path / "hourly"
    daily_dir = tmp_path / "daily"
    monthly_dir = tmp_path / "monthly"
    hourly_dir.mkdir()

    _write_hourly_ncs(hourly_dir, 2020)

    # First call: produces the daily NC
    daily_path, monthly_path = consolidate_year(
        year=2020,
        hourly_dir=hourly_dir,
        daily_dir=daily_dir,
        monthly_dir=monthly_dir,
    )
    assert daily_path.exists()

    # Record the mtime of the daily file so we can confirm it wasn't rewritten
    mtime_before = daily_path.stat().st_mtime

    # Empty the hourly dir to simulate post-cleanup state
    for f in hourly_dir.iterdir():
        f.unlink()

    # Spy on hourly_to_daily to confirm it is NOT called on the second run
    with patch("nhf_spatial_targets.fetch.era5_land.hourly_to_daily") as mock_h2d:
        daily_path2, monthly_path2 = consolidate_year(
            year=2020,
            hourly_dir=hourly_dir,
            daily_dir=daily_dir,
            monthly_dir=monthly_dir,
        )

    mock_h2d.assert_not_called()
    assert daily_path2 == daily_path
    assert daily_path2.exists()
    # Daily file was NOT rewritten (mtime unchanged)
    assert daily_path2.stat().st_mtime == mtime_before


def test_consolidate_year_reaggregates_when_hourly_is_newer(tmp_path):
    """If a hourly NC is touched after the daily NC, re-aggregation happens."""
    import time
    from unittest.mock import patch

    from nhf_spatial_targets.fetch.era5_land import consolidate_year

    hourly_dir = tmp_path / "hourly"
    daily_dir = tmp_path / "daily"
    monthly_dir = tmp_path / "monthly"
    hourly_dir.mkdir()

    _write_hourly_ncs(hourly_dir, 2020)

    # First pass: produce the daily NC
    daily_path, _ = consolidate_year(
        year=2020,
        hourly_dir=hourly_dir,
        daily_dir=daily_dir,
        monthly_dir=monthly_dir,
    )
    assert daily_path.exists()

    # Force the daily NC to appear older by back-dating it 2 seconds
    old_mtime = daily_path.stat().st_mtime - 2.0
    import os

    os.utime(daily_path, (old_mtime, old_mtime))

    # Now touch one of the hourly files so it is newer than the daily NC
    hourly_ro = hourly_dir / "era5_land_ro_2020.nc"
    current = time.time()
    os.utime(hourly_ro, (current, current))

    # Second pass: should re-aggregate because the hourly file is newer
    with patch(
        "nhf_spatial_targets.fetch.era5_land.hourly_to_daily",
        wraps=__import__(
            "nhf_spatial_targets.fetch.era5_land", fromlist=["hourly_to_daily"]
        ).hourly_to_daily,
    ) as mock_h2d:
        consolidate_year(
            year=2020,
            hourly_dir=hourly_dir,
            daily_dir=daily_dir,
            monthly_dir=monthly_dir,
        )

    assert mock_h2d.called, (
        "hourly_to_daily should be called when a hourly NC is newer than the daily NC"
    )
