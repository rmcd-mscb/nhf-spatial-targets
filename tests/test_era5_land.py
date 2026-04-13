from __future__ import annotations

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


def test_download_year_calls_cds_client(tmp_path, monkeypatch):
    from nhf_spatial_targets.fetch.era5_land import download_year_variable

    fake_client = MagicMock()
    fake_client.retrieve = MagicMock()
    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.era5_land._cds_client", lambda: fake_client
    )

    out = tmp_path / "era5_land_ro_2020.nc"
    download_year_variable(year=2020, variable="ro", output_path=out)

    fake_client.retrieve.assert_called_once()
    args, kwargs = fake_client.retrieve.call_args
    assert args[0] == "reanalysis-era5-land"
    request = args[1]
    assert request["variable"] == "runoff"
    assert request["year"] == "2020"
    assert request["area"] == [53.0, -125.0, 24.7, -66.0]
    assert request["format"] == "netcdf"
    assert "month" in request and len(request["month"]) == 12
    assert args[2] == str(out)


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
