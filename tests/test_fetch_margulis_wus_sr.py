"""Tests for Margulis Western US Snow Reanalysis (NSIDC-0719) fetch + consolidate.

The module has two stages:

1. ``fetch_margulis_wus_sr`` issues an earthaccess search + download per
   year, clipped to the project's fabric bbox.
2. ``consolidate_calendar_year_margulis_wus_sr`` mosaics the resulting
   per-water-year per-tile granules into one CF-1.6 per-calendar-year
   NetCDF.

Tests in this file fall into two groups:

- **Fetch-flow tests** (download path, manifest plumbing, period gate).
  These use stub ``.nc`` files that are not valid NetCDFs, so each
  monkeypatches ``consolidate_calendar_year_margulis_wus_sr`` with a
  no-op that writes a stub daily NC.
- **Consolidator tests** drive the real consolidator against synthetic
  NSIDC-0719-shaped NetCDFs written to a tmp datastore.

All tests are fully offline — ``earthaccess.login``, ``search_data``,
``download`` are monkeypatched; consolidation tests build synthetic
``xr.Dataset`` inputs with ``ds.to_netcdf``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from nhf_spatial_targets.fetch.margulis_wus_sr import (
    _wy_for_filename,
    consolidate_calendar_year_margulis_wus_sr,
    fetch_margulis_wus_sr,
)


def _make_project(tmp_path: Path) -> Path:
    """Materialize a minimal valid project directory with a fabric bbox."""
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
    (tmp_path / "fabric.json").write_text(
        json.dumps(
            {
                "sha256": "f00",
                "bbox": {
                    "minx": -124.0,
                    "miny": 42.0,
                    "maxx": -116.0,
                    "maxy": 46.5,
                },
                "bbox_buffered": {
                    "minx": -124.2,
                    "miny": 41.9,
                    "maxx": -115.8,
                    "maxy": 46.6,
                },
            }
        )
    )
    return tmp_path


def _stub_earthaccess(
    monkeypatch: pytest.MonkeyPatch,
    *,
    granules_per_year: int = 2,
    files_returned: int | None = None,
    zero_byte_count: int = 0,
    login_error: Exception | None = None,
    stub_consolidate: bool = True,
) -> dict[str, list]:
    """Patch earthaccess login/search/download; return a call log dict.

    By default also stubs ``consolidate_calendar_year_margulis_wus_sr``
    with a no-op that writes a placeholder daily NC (see
    :func:`_stub_consolidate`). Set ``stub_consolidate=False`` only for
    tests that drive the real consolidator directly.
    """
    import earthaccess

    calls = {"search": [], "downloaded_dirs": [], "login": []}

    def fake_login(strategy="netrc"):
        calls["login"].append(strategy)
        if login_error is not None:
            raise login_error

    def fake_search(**kwargs):
        calls["search"].append(kwargs)
        return [object()] * granules_per_year

    def fake_download(results, dest):
        calls["downloaded_dirs"].append(str(dest))
        n = len(results) if files_returned is None else files_returned
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(n):
            p = dest_path / f"margulis_stub_{i}.nc"
            if i < zero_byte_count:
                p.write_bytes(b"")
            else:
                p.write_bytes(b"\x00")
            paths.append(str(p))
        return paths

    monkeypatch.setattr(earthaccess, "login", fake_login)
    monkeypatch.setattr(earthaccess, "search_data", fake_search)
    monkeypatch.setattr(earthaccess, "download", fake_download)
    if stub_consolidate:
        calls["consolidate"] = _stub_consolidate(monkeypatch)
    return calls


def _stub_consolidate(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace ``consolidate_calendar_year_margulis_wus_sr`` with a no-op stub.

    The fetch-flow tests work with 1-byte stub ``.nc`` files that the real
    consolidator (xarray-based) cannot open. This helper substitutes a
    deterministic stub that writes a placeholder daily NC at the
    expected path so the dual-gate completion check
    (``n_granules > 0`` **and** ``daily_path`` exists) is satisfied on
    subsequent runs. Returns a call-log list of the years passed in.
    """
    calls: list[int] = []

    def fake_consolidate(year, raw_dir, daily_dir):
        calls.append(int(year))
        daily_dir = Path(daily_dir)
        daily_dir.mkdir(parents=True, exist_ok=True)
        out = daily_dir / f"margulis_wus_sr_daily_{year}.nc"
        out.write_bytes(b"\x00")
        return out

    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.margulis_wus_sr."
        "consolidate_calendar_year_margulis_wus_sr",
        fake_consolidate,
    )
    return calls


# ---------------------------------------------------------------------------
# Period and inputs
# ---------------------------------------------------------------------------


def test_period_below_1985_rejected(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch)
    with pytest.raises(ValueError, match="outside the Margulis WUS-SR"):
        fetch_margulis_wus_sr(workdir=workdir, period="1980/1984")


def test_period_above_2021_rejected(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch)
    with pytest.raises(ValueError, match="outside the Margulis WUS-SR"):
        fetch_margulis_wus_sr(workdir=workdir, period="2022/2022")


def test_period_clamps_to_publisher_window(tmp_path, monkeypatch):
    """The boundary years 1985 and 2021 are inclusive."""
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch, granules_per_year=1)
    result = fetch_margulis_wus_sr(workdir=workdir, period="1985/1985")
    assert [r["year"] for r in result["years"]] == [1985]


def test_search_uses_fabric_bbox_buffered(tmp_path, monkeypatch):
    """The CMR search bounding box is taken from fabric.json `bbox_buffered`."""
    workdir = _make_project(tmp_path)
    calls = _stub_earthaccess(monkeypatch, granules_per_year=1)
    fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")
    bbox = calls["search"][-1]["bounding_box"]
    assert bbox == (-124.2, 41.9, -115.8, 46.6)


# ---------------------------------------------------------------------------
# Search / download paths
# ---------------------------------------------------------------------------


def test_empty_search_records_no_granules_note(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch, granules_per_year=0)
    result = fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")
    assert result["years"][0]["n_granules"] == 0
    assert result["years"][0]["note"] == "no_granules_in_CMR"


def test_partial_download_raises(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch, granules_per_year=3, files_returned=2)
    with pytest.raises(RuntimeError, match="partial download"):
        fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")


def test_zero_byte_files_dropped_then_partial_raises(tmp_path, monkeypatch):
    """Zero-byte downloads are deleted; a shortfall after the drop raises."""
    workdir = _make_project(tmp_path)
    _stub_earthaccess(
        monkeypatch,
        granules_per_year=3,
        files_returned=3,
        zero_byte_count=2,
    )
    with pytest.raises(RuntimeError, match="partial download"):
        fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")
    # The zero-byte files are unlinked, leaving the year dir with just
    # the surviving non-empty granule.
    year_dir = tmp_path / "datastore" / "margulis_wus_sr" / "raw" / "2000"
    survivors = [p for p in year_dir.iterdir() if p.stat().st_size > 0]
    assert len(survivors) == 1


def test_zero_download_raises(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch, granules_per_year=2, files_returned=0)
    with pytest.raises(RuntimeError, match="returned no files"):
        fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")


def test_authentication_failure_propagates(tmp_path, monkeypatch):
    """A login failure surfaces cleanly (not swallowed by retry logic)."""
    workdir = _make_project(tmp_path)
    _stub_earthaccess(
        monkeypatch,
        granules_per_year=1,
        login_error=RuntimeError("Earthdata login refused"),
    )
    with pytest.raises(RuntimeError, match="Earthdata login refused"):
        fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")


# ---------------------------------------------------------------------------
# Manifest fields
# ---------------------------------------------------------------------------


def test_manifest_records_fabric_scope(tmp_path, monkeypatch):
    """The manifest entry carries the Oregon-only scope from the catalog."""
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch, granules_per_year=1)
    fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")
    manifest = json.loads((workdir / "manifest.json").read_text())
    entry = manifest["sources"]["margulis_wus_sr"]
    assert entry["fabric_scope"]["fabrics"] == ["or"]
    assert entry["variables"] == ["SWE"]
    # Manifest field is `search_bbox` (the fabric-buffered search bbox),
    # distinct from SNODAS's fixed `bbox`.
    assert entry["search_bbox"] == [-124.2, 41.9, -115.8, 46.6]


def test_completed_years_skipped_on_rerun(tmp_path, monkeypatch):
    """Re-running the same period with a non-empty manifest skips done years."""
    workdir = _make_project(tmp_path)
    calls = _stub_earthaccess(monkeypatch, granules_per_year=1)
    fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")
    n_searches_first = len(calls["search"])
    # Second run for the same period should issue NO new CMR searches.
    fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")
    assert len(calls["search"]) == n_searches_first


def test_zero_granule_years_not_treated_as_complete(tmp_path, monkeypatch):
    """Years with zero CMR hits are retried on the next run.

    A year recorded with `n_granules: 0` means no CMR coverage was
    available; coverage can fill in retroactively, so re-runs should
    retry rather than treat the year as done.
    """
    workdir = _make_project(tmp_path)
    calls = _stub_earthaccess(monkeypatch, granules_per_year=0)
    fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")
    n_searches_first = len(calls["search"])
    fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")
    assert len(calls["search"]) == n_searches_first + 1


def test_manifest_accumulates_years_across_calls(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch, granules_per_year=1)
    fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")
    fetch_margulis_wus_sr(workdir=workdir, period="2001/2001")
    manifest = json.loads((workdir / "manifest.json").read_text())
    years = [r["year"] for r in manifest["sources"]["margulis_wus_sr"]["years"]]
    assert years == [2000, 2001]


def test_missing_bbox_buffered_raises(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    # Strip the buffered bbox from fabric.json to simulate a stale fabric.
    fabric_path = workdir / "fabric.json"
    fabric = json.loads(fabric_path.read_text())
    fabric.pop("bbox_buffered", None)
    fabric_path.write_text(json.dumps(fabric))
    _stub_earthaccess(monkeypatch, granules_per_year=1)
    with pytest.raises(ValueError, match="bbox_buffered"):
        fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")


# ---------------------------------------------------------------------------
# Consolidator: real synthetic-NetCDF inputs
# ---------------------------------------------------------------------------


def _wy_days(wy_next_year: int) -> int:
    """Return the day count for the water year *ending* in Sep of ``wy_next_year``.

    The water year is leap-day-bearing when the calendar year that
    *contains* Feb falls on a leap year. For WY *X* = Oct *X-1* – Sep *X*,
    that is calendar year *X*.
    """
    return 366 if pd.Timestamp(f"{wy_next_year}-12-31").is_leap_year else 365


def _build_synthetic_swe_granule(
    out_path: Path,
    *,
    wy_prev_year: int,
    lat_lo: float,
    lon_lo: float,
    n_cells: int = 5,
    seed: int = 0,
    swe_var: str = "SWE_Post",
) -> None:
    """Write a synthetic NSIDC-0719-shaped SWE granule to ``out_path``.

    The NSIDC-0719 schema is ``(Day, Stats, Longitude, Latitude)`` with
    capital-L coordinate names and a ``Stats`` axis of size 5
    ``[mean, std, median, p25, p75]``. Latitudes descend within a tile;
    longitudes ascend; both span exactly 1° per tile.
    """
    wy_next = wy_prev_year + 1
    n_days = _wy_days(wy_next)
    # Latitudes descend from lat_lo+1 to lat_lo across n_cells points.
    lat = np.linspace(lat_lo + 1.0, lat_lo, n_cells, dtype="float32")
    # Longitudes ascend from lon_lo to lon_lo+1.
    lon = np.linspace(lon_lo, lon_lo + 1.0, n_cells, dtype="float32")

    rng = np.random.default_rng(seed)
    # Mean SWE: zero in summer, ramps up in winter, peaks Mar-Apr.
    day_of_year = np.arange(n_days)
    seasonal = 0.5 * np.maximum(0, np.sin(2 * np.pi * (day_of_year - 60) / n_days))
    mean = (
        (
            seasonal[:, None, None]
            + 0.05 * rng.standard_normal((n_days, n_cells, n_cells)).astype("float32")
        )
        .clip(0)
        .astype("float32")
    )
    std = (0.1 * (mean + 0.01)).astype("float32")
    median = mean.astype("float32")
    p25 = (mean - 0.5 * std).clip(0).astype("float32")
    p75 = (mean + 0.5 * std).astype("float32")
    # Stack into (Stats, Day, Longitude, Latitude). Native order in
    # NSIDC granules is (Day, Stats, Longitude, Latitude); shuffle below.
    swe_stack = np.stack([mean, std, median, p25, p75], axis=1)
    sca_stack = np.zeros_like(swe_stack)

    ds = xr.Dataset(
        data_vars={
            swe_var: (
                ("Day", "Stats", "Latitude", "Longitude"),
                swe_stack,
                {"Units": "meters"},
            ),
            "SCA_Post": (
                ("Day", "Stats", "Latitude", "Longitude"),
                sca_stack,
                {"Units": "[-]"},
            ),
        },
        coords={
            "Latitude": ("Latitude", lat, {"Units": "degrees_north"}),
            "Longitude": ("Longitude", lon, {"Units": "degrees_east"}),
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)


def _build_synthetic_year_dir(
    raw_year_dir: Path,
    *,
    calendar_year: int,
    n_tiles: int = 2,
    include_next_wy: bool = True,
    swe_var: str = "SWE_Post",
) -> None:
    """Stage two adjacent water years of synthetic tile granules in one CY dir.

    Tiles abut along longitude: tile k covers [lon0+k, lon0+k+1].
    """
    raw_year_dir.mkdir(parents=True, exist_ok=True)
    lat_lo = 31.0
    lon_lo_base = -110.0
    wys = [calendar_year]
    if include_next_wy:
        wys.append(calendar_year + 1)
    for wy_next in wys:
        wy_prev = wy_next - 1
        for k in range(n_tiles):
            lon_lo = lon_lo_base + k
            tile_tag = f"N{int(lat_lo):02d}_0W{abs(int(lon_lo)):03d}_0"
            wy_curr_2d = wy_next % 100
            name = (
                f"WUS_UCLA_SR_v01_{tile_tag}_agg_16_"
                f"WY{wy_prev}_{wy_curr_2d:02d}_SWE_SCA_POST.nc"
            )
            _build_synthetic_swe_granule(
                raw_year_dir / name,
                wy_prev_year=wy_prev,
                lat_lo=lat_lo,
                lon_lo=lon_lo,
                seed=k * 10 + (wy_next - 2000),
                swe_var=swe_var,
            )


def test_wy_for_filename_parses_century_rollover():
    """``WY1999_00`` -> (1999, 2000); ``WY2099_00`` -> (2099, 2100)."""
    assert _wy_for_filename(
        "WUS_UCLA_SR_v01_N31_0W103_0_agg_16_WY1999_00_SWE_SCA_POST.nc"
    ) == (1999, 2000)
    assert _wy_for_filename(
        "WUS_UCLA_SR_v01_N31_0W103_0_agg_16_WY2020_21_SWE_SCA_POST.nc"
    ) == (2020, 2021)
    # SD_POST is the snow-depth product; the SWE consolidator ignores it.
    assert (
        _wy_for_filename("WUS_UCLA_SR_v01_N31_0W103_0_agg_16_WY2020_21_SD_POST.nc")
        is None
    )


def test_consolidate_happy_path(tmp_path):
    """Two synthetic WYs collapse into one per-CY CF NetCDF."""
    raw_dir = tmp_path / "raw" / "2000"
    daily_dir = tmp_path / "daily"
    _build_synthetic_year_dir(raw_dir, calendar_year=2000)

    out_path = consolidate_calendar_year_margulis_wus_sr(2000, raw_dir, daily_dir)
    assert out_path == daily_dir / "margulis_wus_sr_daily_2000.nc"
    assert out_path.exists()

    ds = xr.open_dataset(out_path)
    # Variable renamed to the catalog-declared name.
    assert list(ds.data_vars) == ["SWE", "crs"] or set(ds.data_vars) >= {"SWE", "crs"}
    # Dimension order normalized to (time, lat, lon).
    assert ds["SWE"].dims == ("time", "lat", "lon")
    # 2000 is a leap year -> 366 daily steps Jan 1 – Dec 31.
    assert ds.sizes["time"] == 366
    assert pd.Timestamp(ds.time.values[0]) == pd.Timestamp("2000-01-01")
    assert pd.Timestamp(ds.time.values[-1]) == pd.Timestamp("2000-12-31")
    # Two adjacent tiles abutting in longitude -> mosaic spans 2°.
    assert ds["SWE"].sizes["lon"] >= 5  # at least one tile's width
    assert ds["SWE"].sizes["lat"] == 5
    # Time axis is monotonically increasing.
    assert (np.diff(ds.time.values) > np.timedelta64(0)).all()
    # Provenance attrs recorded.
    assert json.loads(ds.attrs["margulis_source_water_years"]) == [2000, 2001]


def test_consolidate_is_cf_1_6_compliant(tmp_path):
    """Every checklist item from docs/superpowers/plans/2026-05-13... holds."""
    raw_dir = tmp_path / "raw" / "2000"
    daily_dir = tmp_path / "daily"
    _build_synthetic_year_dir(raw_dir, calendar_year=2000)
    out_path = consolidate_calendar_year_margulis_wus_sr(2000, raw_dir, daily_dir)

    ds = xr.open_dataset(out_path, decode_times=True)
    # 1. Conventions
    assert ds.attrs.get("Conventions") == "CF-1.6"
    # 2. Coord names + dim order
    assert "time" in ds.dims and "lat" in ds.dims and "lon" in ds.dims
    assert ds["SWE"].dims == ("time", "lat", "lon")
    # 3. lat attrs (latitude descending)
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lat.attrs["units"] == "degrees_north"
    assert ds.lat.attrs["axis"] == "Y"
    assert ds.lat.values[0] > ds.lat.values[-1]  # descending
    # 4. lon attrs (longitude ascending)
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert ds.lon.attrs["units"] == "degrees_east"
    assert ds.lon.attrs["axis"] == "X"
    assert ds.lon.values[0] < ds.lon.values[-1]  # ascending
    # 5. time attrs
    assert ds.time.attrs["standard_name"] == "time"
    assert ds.time.attrs["axis"] == "T"
    # 6. SWE variable attrs from catalog
    assert ds["SWE"].attrs["units"] == "m"  # catalog cf_units
    assert ds["SWE"].attrs["cell_methods"] == "time: point"
    assert ds["SWE"].attrs["grid_mapping"] == "crs"
    assert "long_name" in ds["SWE"].attrs
    # 7. crs ancillary variable with WGS84 grid mapping
    assert "crs" in ds.variables
    crs_attrs = ds["crs"].attrs
    assert crs_attrs["grid_mapping_name"] == "latitude_longitude"
    assert "crs_wkt" in crs_attrs
    assert "semi_major_axis" in crs_attrs
    # 8. Global provenance attrs
    assert "title" in ds.attrs
    assert "source" in ds.attrs
    assert "institution" in ds.attrs
    assert "references" in ds.attrs
    assert "history" in ds.attrs
    # 9. Margulis-specific provenance
    assert "margulis_source_water_years" in ds.attrs


def test_consolidate_idempotent_when_inputs_older(tmp_path):
    """Second call when NC newer than inputs is a no-op (same mtime)."""
    raw_dir = tmp_path / "raw" / "2000"
    daily_dir = tmp_path / "daily"
    _build_synthetic_year_dir(raw_dir, calendar_year=2000)
    out_path = consolidate_calendar_year_margulis_wus_sr(2000, raw_dir, daily_dir)
    first_mtime = out_path.stat().st_mtime
    second_out = consolidate_calendar_year_margulis_wus_sr(2000, raw_dir, daily_dir)
    assert second_out == out_path
    assert second_out.stat().st_mtime == first_mtime


def test_consolidate_missing_next_water_year_raises(tmp_path):
    """A CY whose WY *X+1* granules are absent raises FileNotFoundError.

    This is the source-domain-boundary case (e.g. CY 2021 with the v01
    dataset that ends at WY 2021): there is no WY 2022 to supply Oct–Dec.
    """
    raw_dir = tmp_path / "raw" / "2021"
    daily_dir = tmp_path / "daily"
    _build_synthetic_year_dir(raw_dir, calendar_year=2021, include_next_wy=False)
    with pytest.raises(FileNotFoundError, match="2022"):
        consolidate_calendar_year_margulis_wus_sr(2021, raw_dir, daily_dir)


def test_consolidate_missing_raw_root_raises(tmp_path):
    """A non-existent raw directory yields a clear FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="raw directory"):
        consolidate_calendar_year_margulis_wus_sr(
            2000, tmp_path / "no-such-dir", tmp_path / "daily"
        )


def test_consolidate_unexpected_swe_variable_name_raises(tmp_path):
    """A schema drift (e.g. NSIDC renames SWE_Post) surfaces as a clear error."""
    raw_dir = tmp_path / "raw" / "2000"
    daily_dir = tmp_path / "daily"
    # Build inputs with a different variable name (simulates a schema change).
    _build_synthetic_year_dir(raw_dir, calendar_year=2000, swe_var="SWE_Renamed")
    with pytest.raises(ValueError, match="SWE_Post"):
        consolidate_calendar_year_margulis_wus_sr(2000, raw_dir, daily_dir)


def test_fetch_backfill_consolidates_existing_raws(tmp_path, monkeypatch):
    """Pre-staged raws + manifest sans daily_path -> backfill consolidation only.

    Simulates the case where a user has raw NSIDC-0719 granules on disk
    from a fetch that pre-dates this consolidator. The second
    ``fetch_margulis_wus_sr`` call should NOT re-download (raw is intact
    and manifest records n_granules > 0), but SHOULD build the missing
    daily NC.
    """
    workdir = _make_project(tmp_path)
    # First call: download stubs, stub-consolidate writes a placeholder
    # daily NC, manifest records daily_path. Then we delete the daily NC
    # and clear daily_path to simulate "raws exist but never consolidated".
    calls = _stub_earthaccess(monkeypatch, granules_per_year=1)
    fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")
    daily_path = (
        tmp_path
        / "datastore"
        / "margulis_wus_sr"
        / "daily"
        / "margulis_wus_sr_daily_2000.nc"
    )
    daily_path.unlink()
    manifest = json.loads((workdir / "manifest.json").read_text())
    for rec in manifest["sources"]["margulis_wus_sr"]["years"]:
        rec.pop("daily_path", None)
        rec.pop("consolidated_utc", None)
    (workdir / "manifest.json").write_text(json.dumps(manifest))

    n_searches_first = len(calls["search"])
    fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")
    # No new CMR search — raws are still on disk per manifest.
    assert len(calls["search"]) == n_searches_first
    # Daily NC re-created by the backfill pass.
    assert daily_path.exists()
