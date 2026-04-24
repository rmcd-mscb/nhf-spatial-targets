from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
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


def test_download_month_variable_extracts_zip(tmp_path, monkeypatch):
    """download_month_variable extracts the .nc from a CDS zip response."""
    import io
    import zipfile

    from nhf_spatial_targets.fetch.era5_land import download_month_variable

    out = tmp_path / "era5_land_ro_2020_03.nc"
    nc_content = b"fake_nc_data_inside_zip"

    def fake_retrieve(dataset, request, path):
        # Simulate CDS API returning a zip-wrapped NetCDF.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("data.nc", nc_content)
        Path(path).write_bytes(buf.getvalue())

    fake_client = MagicMock()
    fake_client.retrieve = MagicMock(side_effect=fake_retrieve)
    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.era5_land._cds_client", lambda: fake_client
    )

    result = download_month_variable(year=2020, month=3, variable="ro", output_path=out)

    assert result == out
    assert out.exists()
    assert out.read_bytes() == nc_content
    # Confirm the output file is not a zip.
    assert not zipfile.is_zipfile(out)
    # No .tmp file should remain.
    assert not Path(str(out) + ".tmp").exists()
    # Extracted intermediate file should not remain.
    assert not (tmp_path / "data.nc").exists()


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

    with pytest.raises(RuntimeError, match="CDS server error"):
        download_year_variable(year=2020, variable="ro", output_path=out)

    assert not out.exists(), "Year file must not exist after a failed monthly download"
    assert not chunk_01_tmp.exists(), ".tmp for the failed month must be cleaned up"


def test_download_year_skips_existing(tmp_path, monkeypatch):
    """Fast path: year file + all 12 chunks present, no newer chunks → skip."""
    from nhf_spatial_targets.fetch.era5_land import download_year_variable

    fake_client = MagicMock()
    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.era5_land._cds_client", lambda: fake_client
    )
    out = tmp_path / "era5_land_ro_2020.nc"
    # Pre-stage all 12 monthly chunks older than the year file.
    for month in range(1, 13):
        chunk = tmp_path / f"era5_land_ro_2020_{month:02d}.nc"
        chunk.write_bytes(b"chunk")
    out.write_bytes(b"existing")
    download_year_variable(year=2020, variable="ro", output_path=out)
    fake_client.retrieve.assert_not_called()


def test_download_year_rebuilds_when_chunks_incomplete(tmp_path, monkeypatch):
    """If year file exists but fewer than 12 chunks on disk, rebuild.

    Guards against the case where the year file was written from a
    partial set of chunks and later chunks arrive with an older mtime
    (e.g. rsync -a, restore-from-backup) that would otherwise bypass
    the mtime check.
    """
    from nhf_spatial_targets.fetch.era5_land import download_year_variable

    fake_client = MagicMock()

    def fake_retrieve(dataset, request, target_path):
        # Minimal valid NC so open_mfdataset can read back during concat.
        import numpy as np
        import pandas as pd
        import xarray as xr

        month = int(request["month"])
        times = pd.date_range(f"2020-{month:02d}-01", periods=1, freq="h")
        da = xr.DataArray(
            np.zeros((1, 1, 1)),
            dims=("time", "latitude", "longitude"),
            coords={"time": times, "latitude": [40.0], "longitude": [-100.0]},
            name="ro",
        )
        da.to_dataset().to_netcdf(target_path)

    fake_client.retrieve.side_effect = fake_retrieve
    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.era5_land._cds_client", lambda: fake_client
    )
    out = tmp_path / "era5_land_ro_2020.nc"
    # Pre-stage only 6 chunks; year file pretends to be up to date but isn't.
    for month in range(1, 7):
        chunk = tmp_path / f"era5_land_ro_2020_{month:02d}.nc"
        fake_retrieve(None, {"month": f"{month:02d}"}, chunk)
    out.write_bytes(b"partial")
    download_year_variable(year=2020, variable="ro", output_path=out)
    # Exactly the 6 missing months should have been fetched from CDS.
    assert fake_client.retrieve.call_count == 6
    # Final year file exists.
    assert out.exists()


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


def test_consolidate_year_valid_time_dim(tmp_path):
    """consolidate_year normalises CDS API v2 'valid_time' → 'time' dimension."""
    from nhf_spatial_targets.fetch.era5_land import consolidate_year

    hourly_dir = tmp_path / "hourly"
    daily_dir = tmp_path / "daily"
    monthly_dir = tmp_path / "monthly"
    hourly_dir.mkdir()

    # Write hourly NCs using 'valid_time' (CDS API ≥0.7 output convention)
    times = pd.date_range("2020-01-01", "2020-01-03 23:00", freq="1h")
    for var in ("ro", "sro", "ssro"):
        vals = np.tile(
            np.arange(24, dtype=float).reshape(24, 1, 1) * 0.001,
            (len(times) // 24, 1, 1),
        )
        ds = xr.Dataset(
            {var: (("valid_time", "latitude", "longitude"), vals)},
            coords={
                "valid_time": times[: vals.shape[0]],
                "latitude": [40.0],
                "longitude": [-100.0],
            },
        )
        ds[var].attrs["units"] = "m"
        ds.to_netcdf(hourly_dir / f"era5_land_{var}_2020.nc")

    # Should not raise "Dimensions {'time'} do not exist"
    daily_path, monthly_path = consolidate_year(
        year=2020,
        hourly_dir=hourly_dir,
        daily_dir=daily_dir,
        monthly_dir=monthly_dir,
    )

    assert daily_path.exists()
    with xr.open_dataset(daily_path) as ds:
        assert "time" in ds.dims, "daily NC must have 'time' dimension"
        assert {"ro", "sro", "ssro"}.issubset(set(ds.data_vars))

    assert monthly_path.exists()
    with xr.open_dataset(monthly_path) as ds:
        assert "time" in ds.dims, "monthly NC must have 'time' dimension"


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


# ---- Manifest merge --------------------------------------------------------


def _minimal_workdir(tmp_path: Path) -> Path:
    """Create a minimal project workdir for _update_manifest tests."""
    import json

    import yaml

    wd = tmp_path / "run"
    wd.mkdir()
    config = {
        "fabric": {"path": "", "id_col": "nhm_id"},
        "datastore": str(wd / "datastore"),
        "dir_mode": "2775",
    }
    (wd / "config.yml").write_text(yaml.dump(config))
    (wd / "fabric.json").write_text(json.dumps({"hru_count": 3, "id_col": "nhm_id"}))
    (wd / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    return wd


def test_update_manifest_accumulates_years(tmp_path):
    """Successive _update_manifest calls merge files rather than overwriting."""
    import json

    from nhf_spatial_targets.fetch.era5_land import _update_manifest

    wd = _minimal_workdir(tmp_path)
    meta = {
        "access": {"url": "https://cds.climate.copernicus.eu"},
        "variables": [{"name": "ro"}],
    }
    bbox = {"minx": -125.1, "miny": 23.9, "maxx": -65.9, "maxy": 50.1}
    license_str = "Copernicus license"

    files_1979 = [
        {
            "year": 1979,
            "daily_path": "/ds/era5_land/daily/era5_land_daily_1979.nc",
            "monthly_path": "/ds/era5_land/monthly/era5_land_monthly_1979.nc",
            "consolidated_utc": "2024-01-01T00:00:00+00:00",
        }
    ]
    files_1980 = [
        {
            "year": 1980,
            "daily_path": "/ds/era5_land/daily/era5_land_daily_1980.nc",
            "monthly_path": "/ds/era5_land/monthly/era5_land_monthly_1980.nc",
            "consolidated_utc": "2024-01-02T00:00:00+00:00",
        }
    ]

    # First run: fetch 1979
    _update_manifest(wd, "1979/1979", bbox, meta, license_str, files_1979)
    m1 = json.loads((wd / "manifest.json").read_text())
    entry1 = m1["sources"]["era5_land"]
    assert entry1["period"] == "1979/1979"
    assert len(entry1["files"]) == 1
    assert entry1["files"][0]["year"] == 1979

    # Second run: fetch 1980 — should not overwrite 1979
    _update_manifest(wd, "1980/1980", bbox, meta, license_str, files_1980)
    m2 = json.loads((wd / "manifest.json").read_text())
    entry2 = m2["sources"]["era5_land"]
    assert entry2["period"] == "1979/1980"
    assert len(entry2["files"]) == 2
    years = [f["year"] for f in entry2["files"]]
    assert years == [1979, 1980]


def test_update_manifest_updates_existing_year(tmp_path):
    """Re-running for the same year replaces that year's file record."""
    import json

    from nhf_spatial_targets.fetch.era5_land import _update_manifest

    wd = _minimal_workdir(tmp_path)
    meta = {
        "access": {"url": "https://cds.climate.copernicus.eu"},
        "variables": [{"name": "ro"}],
    }
    bbox = {"minx": -125.1, "miny": 23.9, "maxx": -65.9, "maxy": 50.1}
    license_str = "Copernicus license"

    files_v1 = [
        {
            "year": 1979,
            "daily_path": "/ds/era5_land/daily/era5_land_daily_1979.nc",
            "monthly_path": "/ds/era5_land/monthly/era5_land_monthly_1979.nc",
            "consolidated_utc": "2024-01-01T00:00:00+00:00",
        }
    ]
    files_v2 = [
        {
            "year": 1979,
            "daily_path": "/ds/era5_land/daily/era5_land_daily_1979.nc",
            "monthly_path": "/ds/era5_land/monthly/era5_land_monthly_1979.nc",
            "consolidated_utc": "2024-06-01T00:00:00+00:00",  # updated timestamp
        }
    ]

    _update_manifest(wd, "1979/1979", bbox, meta, license_str, files_v1)
    _update_manifest(wd, "1979/1979", bbox, meta, license_str, files_v2)

    m = json.loads((wd / "manifest.json").read_text())
    entry = m["sources"]["era5_land"]
    assert len(entry["files"]) == 1
    assert entry["files"][0]["consolidated_utc"] == "2024-06-01T00:00:00+00:00"


def test_update_manifest_handles_missing_year_key(tmp_path, caplog):
    """Prior manifest entries missing 'year' are skipped with a warning."""
    import json
    import logging

    from nhf_spatial_targets.fetch.era5_land import _update_manifest

    wd = _minimal_workdir(tmp_path)
    meta = {
        "access": {"url": "https://cds.climate.copernicus.eu"},
        "variables": [{"name": "ro"}],
    }
    bbox = {"minx": -125.0, "miny": 24.7, "maxx": -66.0, "maxy": 53.0}

    # Seed a manifest with a malformed file entry (no "year" key)
    manifest = {
        "sources": {
            "era5_land": {
                "files": [
                    {"daily_path": "/orphan.nc"},  # missing year
                    {
                        "year": 1979,
                        "daily_path": "/ds/daily_1979.nc",
                        "monthly_path": "/ds/monthly_1979.nc",
                    },
                ]
            }
        }
    }
    (wd / "manifest.json").write_text(json.dumps(manifest))

    files_new = [
        {
            "year": 1980,
            "daily_path": "/ds/daily_1980.nc",
            "monthly_path": "/ds/monthly_1980.nc",
            "consolidated_utc": "2024-01-01T00:00:00+00:00",
        }
    ]
    with caplog.at_level(logging.WARNING):
        _update_manifest(wd, "1980/1980", bbox, meta, "L", files_new)

    assert any("missing 'year' key" in rec.message for rec in caplog.records)
    result = json.loads((wd / "manifest.json").read_text())
    years = sorted(f["year"] for f in result["sources"]["era5_land"]["files"])
    assert years == [1979, 1980]


def test_update_manifest_skips_bad_year_with_warning(tmp_path, caplog):
    """Prior manifest entry with a non-numeric 'year' is skipped + warned.

    Consistency with the sibling missing-'year' handler: corrupt prior
    entries should not block recording of newly fetched years.
    """
    import json
    import logging

    from nhf_spatial_targets.fetch.era5_land import _update_manifest

    wd = _minimal_workdir(tmp_path)
    meta = {
        "access": {"url": "https://cds.climate.copernicus.eu"},
        "variables": [{"name": "ro"}],
    }
    bbox = {"minx": -125.0, "miny": 24.7, "maxx": -66.0, "maxy": 53.0}

    manifest = {
        "sources": {
            "era5_land": {
                "files": [
                    {"year": "twenty", "daily_path": "/bad.nc"},
                    {
                        "year": 1979,
                        "daily_path": "/good.nc",
                        "monthly_path": "/ds/monthly_1979.nc",
                    },
                ]
            }
        }
    }
    (wd / "manifest.json").write_text(json.dumps(manifest))

    with caplog.at_level(logging.WARNING):
        _update_manifest(
            wd,
            "1980/1980",
            bbox,
            meta,
            "L",
            [
                {
                    "year": 1980,
                    "daily_path": "/d.nc",
                    "monthly_path": "/ds/monthly_1980.nc",
                    "consolidated_utc": "2024-01-01",
                }
            ],
        )

    assert any(
        "invalid year" in rec.message and "twenty" in rec.message
        for rec in caplog.records
    )
    result = json.loads((wd / "manifest.json").read_text())
    years = sorted(f["year"] for f in result["sources"]["era5_land"]["files"])
    # Bad entry skipped; the good 1979 entry and new 1980 entry both kept.
    assert years == [1979, 1980]


def test_fetch_era5_land_writes_partial_manifest_on_failure(tmp_path, monkeypatch):
    """If a later year raises, the manifest records only the completed years.

    Guards the try/finally pattern in fetch_era5_land that persists
    partial-run state so operators can distinguish "completed" from
    "needs re-run" years after a SLURM crash.
    """
    import json

    import yaml

    from nhf_spatial_targets.fetch import era5_land
    from nhf_spatial_targets.fetch.era5_land import fetch_era5_land

    wd = tmp_path / "run"
    wd.mkdir()
    (wd / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": "", "id_col": "nhm_id"},
                "datastore": str(wd / "datastore"),
                "dir_mode": "2775",
            }
        )
    )
    (wd / "fabric.json").write_text(
        json.dumps(
            {
                "hru_count": 3,
                "id_col": "nhm_id",
                "bbox_buffered": {
                    "minx": -125.0,
                    "miny": 24.7,
                    "maxx": -66.0,
                    "maxy": 53.0,
                },
            }
        )
    )
    (wd / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))

    def fake_download(year, variable, output_path):
        pass

    def fake_consolidate(year, hourly_dir, daily_dir, monthly_dir):
        if year == 1981:
            raise RuntimeError("simulated failure at year 1981")
        daily_dir.mkdir(parents=True, exist_ok=True)
        monthly_dir.mkdir(parents=True, exist_ok=True)
        daily = daily_dir / f"era5_land_daily_{year}.nc"
        monthly = monthly_dir / f"era5_land_monthly_{year}.nc"
        daily.write_bytes(b"fake")
        monthly.write_bytes(b"fake")
        return daily, monthly

    monkeypatch.setattr(era5_land, "download_year_variable", fake_download)
    monkeypatch.setattr(era5_land, "consolidate_year", fake_consolidate)

    with pytest.raises(RuntimeError, match="simulated failure"):
        fetch_era5_land(workdir=wd, period="1979/1982")

    # Manifest should contain 1979 and 1980 (completed); not 1981 or 1982.
    manifest = json.loads((wd / "manifest.json").read_text())
    years = sorted(f["year"] for f in manifest["sources"]["era5_land"]["files"])
    assert years == [1979, 1980]


def test_variable_name_raises_on_unexpected_type():
    """_variable_name raises TypeError naming the bad entry."""
    from nhf_spatial_targets.fetch.modis import _variable_name

    with pytest.raises(TypeError, match="Unexpected variable entry type"):
        _variable_name(42)
    with pytest.raises(TypeError, match="Unexpected variable entry type"):
        _variable_name(["list_not_allowed"])


# ---- _completed_years_from_manifest ----------------------------------------


def test_completed_years_no_manifest(tmp_path):
    """Returns empty set when manifest.json does not exist."""
    from nhf_spatial_targets.fetch.era5_land import _completed_years_from_manifest

    result = _completed_years_from_manifest(tmp_path)
    assert result == set()


def test_completed_years_empty_sources(tmp_path):
    """Returns empty set when manifest has no era5_land entry."""
    import json

    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    from nhf_spatial_targets.fetch.era5_land import _completed_years_from_manifest

    assert _completed_years_from_manifest(tmp_path) == set()


def test_completed_years_files_exist(tmp_path):
    """Years whose daily + monthly files exist on disk are returned."""
    import json

    daily = tmp_path / "era5_land_daily_2000.nc"
    monthly = tmp_path / "era5_land_monthly_2000.nc"
    daily.write_bytes(b"d")
    monthly.write_bytes(b"m")

    manifest = {
        "sources": {
            "era5_land": {
                "files": [
                    {
                        "year": 2000,
                        "daily_path": str(daily),
                        "monthly_path": str(monthly),
                    }
                ]
            }
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    from nhf_spatial_targets.fetch.era5_land import _completed_years_from_manifest

    assert _completed_years_from_manifest(tmp_path) == {2000}


def test_completed_years_filters_missing_files(tmp_path):
    """Years whose output files are absent on disk are excluded."""
    import json

    # Only daily exists — monthly is missing → not complete
    daily = tmp_path / "era5_land_daily_1999.nc"
    daily.write_bytes(b"d")

    manifest = {
        "sources": {
            "era5_land": {
                "files": [
                    {
                        "year": 1999,
                        "daily_path": str(daily),
                        "monthly_path": str(tmp_path / "era5_land_monthly_1999.nc"),
                    }
                ]
            }
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    from nhf_spatial_targets.fetch.era5_land import _completed_years_from_manifest

    assert _completed_years_from_manifest(tmp_path) == set()


def test_completed_years_invalid_json(tmp_path, caplog):
    """Returns empty set (with a warning) when manifest.json is unparseable."""
    import logging

    (tmp_path / "manifest.json").write_text("{not valid json")

    from nhf_spatial_targets.fetch.era5_land import _completed_years_from_manifest

    with caplog.at_level(logging.WARNING):
        result = _completed_years_from_manifest(tmp_path)

    assert result == set()
    assert any("could not be parsed" in r.message for r in caplog.records)


# ---- fetch_era5_land worker partitioning -----------------------------------


def _make_fetch_project(tmp_path: Path) -> Path:
    """Minimal project dir for fetch_era5_land partitioning tests."""
    import json

    import yaml

    wd = tmp_path / "run"
    wd.mkdir()
    (wd / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": "", "id_col": "nhm_id"},
                "datastore": str(wd / "datastore"),
                "dir_mode": "2775",
            }
        )
    )
    (wd / "fabric.json").write_text(
        json.dumps(
            {
                "hru_count": 3,
                "id_col": "nhm_id",
                "bbox_buffered": {
                    "minx": -125.0,
                    "miny": 24.7,
                    "maxx": -66.0,
                    "maxy": 53.0,
                },
            }
        )
    )
    (wd / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    return wd


def _fake_download_noop(year, variable, output_path):
    """Fake download that touches the output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"fake")


def test_fetch_era5_land_worker_partitions_years(tmp_path, monkeypatch):
    """Each worker receives a non-overlapping, correctly assigned subset of years.

    With 3 workers over [2000, 2001, 2002]:
      worker 0 → [2000]
      worker 1 → [2001]
      worker 2 → [2002]

    Workers are simulated sequentially but the manifest is suppressed so each
    call sees the same empty starting state (matching true parallel execution
    where all SLURM tasks start before any worker finishes).
    """
    from nhf_spatial_targets.fetch import era5_land

    wd = _make_fetch_project(tmp_path)
    processed: dict[int, list[int]] = {0: [], 1: [], 2: []}

    def fake_consolidate(year, hourly_dir, daily_dir, monthly_dir):
        daily_dir.mkdir(parents=True, exist_ok=True)
        monthly_dir.mkdir(parents=True, exist_ok=True)
        daily = daily_dir / f"era5_land_daily_{year}.nc"
        monthly = monthly_dir / f"era5_land_monthly_{year}.nc"
        daily.write_bytes(b"fake")
        monthly.write_bytes(b"fake")
        return daily, monthly

    monkeypatch.setattr(era5_land, "download_year_variable", _fake_download_noop)
    monkeypatch.setattr(era5_land, "consolidate_year", fake_consolidate)
    # Suppress manifest writes so each sequential call sees the same empty state,
    # matching true parallel execution where all SLURM tasks start together.
    monkeypatch.setattr(era5_land, "_update_manifest", lambda *a, **kw: None)

    from nhf_spatial_targets.fetch.era5_land import fetch_era5_land

    for wi in range(3):
        result = fetch_era5_land(
            workdir=wd, period="2000/2002", worker_index=wi, n_workers=3
        )
        processed[wi] = [f["year"] for f in result["files"]]

    # Every year assigned to exactly one worker
    all_assigned = sorted(y for years in processed.values() for y in years)
    assert all_assigned == [2000, 2001, 2002]
    # No overlap
    for wi in range(3):
        for wj in range(wi + 1, 3):
            assert not set(processed[wi]) & set(processed[wj])


def test_fetch_era5_land_skips_manifest_completed(tmp_path, monkeypatch):
    """Years recorded in manifest with existing files are not re-processed."""
    import json

    from nhf_spatial_targets.fetch import era5_land

    wd = _make_fetch_project(tmp_path)

    # Pre-stage 2000 as complete in the manifest with real files on disk
    ws_datastore = wd / "datastore" / "era5_land"
    daily_2000 = ws_datastore / "daily" / "era5_land_daily_2000.nc"
    monthly_2000 = ws_datastore / "monthly" / "era5_land_monthly_2000.nc"
    daily_2000.parent.mkdir(parents=True, exist_ok=True)
    monthly_2000.parent.mkdir(parents=True, exist_ok=True)
    daily_2000.write_bytes(b"d")
    monthly_2000.write_bytes(b"m")

    manifest = {
        "sources": {
            "era5_land": {
                "files": [
                    {
                        "year": 2000,
                        "daily_path": str(daily_2000),
                        "monthly_path": str(monthly_2000),
                    }
                ]
            }
        },
        "steps": [],
    }
    (wd / "manifest.json").write_text(json.dumps(manifest))

    processed_years: list[int] = []

    def fake_consolidate(year, hourly_dir, daily_dir, monthly_dir):
        processed_years.append(year)
        daily_dir.mkdir(parents=True, exist_ok=True)
        monthly_dir.mkdir(parents=True, exist_ok=True)
        daily = daily_dir / f"era5_land_daily_{year}.nc"
        monthly = monthly_dir / f"era5_land_monthly_{year}.nc"
        daily.write_bytes(b"fake")
        monthly.write_bytes(b"fake")
        return daily, monthly

    monkeypatch.setattr(era5_land, "download_year_variable", _fake_download_noop)
    monkeypatch.setattr(era5_land, "consolidate_year", fake_consolidate)

    from nhf_spatial_targets.fetch.era5_land import fetch_era5_land

    result = fetch_era5_land(workdir=wd, period="2000/2002")

    # 2000 was in manifest with files on disk → skipped
    assert 2000 not in processed_years
    # 2001 and 2002 were not complete → processed
    assert sorted(processed_years) == [2001, 2002]
    result_years = [f["year"] for f in result["files"]]
    assert 2000 not in result_years
    assert sorted(result_years) == [2001, 2002]


def test_fetch_era5_land_invalid_worker_args(tmp_path):
    """Raises ValueError for invalid worker_index / n_workers combinations."""
    from nhf_spatial_targets.fetch.era5_land import fetch_era5_land

    wd = _make_fetch_project(tmp_path)

    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        fetch_era5_land(workdir=wd, period="2000/2000", n_workers=0)

    with pytest.raises(ValueError, match="worker_index must be in"):
        fetch_era5_land(workdir=wd, period="2000/2000", worker_index=3, n_workers=3)

    with pytest.raises(ValueError, match="worker_index must be in"):
        fetch_era5_land(workdir=wd, period="2000/2000", worker_index=-1, n_workers=3)


def test_fetch_era5_land_returns_empty_when_no_years_assigned(tmp_path, monkeypatch):
    """Returns an empty files list when this worker has no years to process."""
    from nhf_spatial_targets.fetch import era5_land
    from nhf_spatial_targets.fetch.era5_land import fetch_era5_land

    wd = _make_fetch_project(tmp_path)
    monkeypatch.setattr(era5_land, "download_year_variable", _fake_download_noop)

    # 1 year, 2 workers → worker 1 gets nothing
    result = fetch_era5_land(
        workdir=wd, period="2000/2000", worker_index=1, n_workers=2
    )
    assert result["files"] == []
    assert result["worker_index"] == 1
    assert result["n_workers"] == 2
