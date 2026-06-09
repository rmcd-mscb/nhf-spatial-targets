"""Tests for UA SWE (NSIDC-0719) fetch + consolidation (issue #238).

Synthetic NetCDF fixtures mirror the layout of the real per-WY files
(``time`` as float32 days since 1900-01-01 with no ``units`` attr,
``SWE`` / ``DEPTH`` float32 in mm, NaN fill, CRS embedded in an ``|S1``
``crs`` scalar with NAD83 WKT) so the consolidation logic is exercised
end-to-end without touching the network.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from nhf_spatial_targets.fetch.ua_swe import (
    _assign_worker_water_years,
    _calendar_years_to_water_years,
    _download_file,
    _earthaccess_session,
    _mask_url,
    _reproject_wy_to_dataset,
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


def _make_synthetic_raw_nc_bytes(wy: int, **kwargs) -> bytes:
    """Build a synthetic raw NC and return its bytes (for fake-session bodies)."""
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _make_synthetic_raw_nc(tmp_path, wy, **kwargs)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


# Padding body for the CONUS mask download. The real mask is ~1.7 MB; for
# tests we just need enough bytes to clear the hard-coded 128 KiB floor
# that `fetch_ua_swe` passes as `min_bytes` for the mask call.
_MASK_PAD_BODY = b"\x00" * 200_000


def _synthetic_responder() -> Callable[[str], "_FakeResponse"]:
    """Responder that serves valid synthetic NCs for every WY URL + mask URL.

    Unrecognised URLs return 404. The same pattern as
    `tests/test_fetch_snodas.py::_synthetic_tar_responder` for the SNODAS
    tar URLs, adapted for ua_swe's per-WY NetCDF naming.
    """

    def responder(url: str) -> "_FakeResponse":
        m = re.search(r"4km_SWE_Depth_WY(\d{4})_v01\.nc$", url)
        if m:
            wy = int(m.group(1))
            body = _make_synthetic_raw_nc_bytes(wy)
            return _FakeResponse(200, body=body)
        if url.endswith("/SWE_Mask_v01.nc"):
            return _FakeResponse(200, body=_MASK_PAD_BODY)
        return _FakeResponse(404)

    return responder


class _FakeResponse:
    """Tiny stand-in for ``requests.Response`` covering what _download_file uses.

    `Content-Length` is auto-derived from the body length unless an
    explicit value is passed via ``content_length``. Pass
    ``content_length=None`` to simulate a server that omits the header.
    Pass an integer that *disagrees* with the body length to exercise
    the short-read integrity path. Ports `tests/test_fetch_snodas.py`'s
    `_FakeResponse`.
    """

    _UNSET = object()

    def __init__(
        self,
        status_code: int,
        body: bytes = b"",
        chunks: list[bytes] | None = None,
        content_length: object = _UNSET,
    ):
        self.status_code = status_code
        self._body = body
        self._chunks = chunks if chunks is not None else [body]
        self.headers: dict[str, str] = {}
        if content_length is _FakeResponse._UNSET:
            # Default: server sets Content-Length matching the body
            # (NSIDC's archive does this on per-WY NetCDF responses).
            self.headers["Content-Length"] = str(len(body))
        elif content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        # else: caller asked for no Content-Length header at all.

    def iter_content(self, chunk_size: int = 8192):
        for c in self._chunks:
            yield c

    def close(self):
        pass


def _stub_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    responder: Callable[[str], _FakeResponse] | None = None,
) -> dict[str, list]:
    """Patch ``_earthaccess_session`` with a session whose ``.get`` is stubbed.

    ``responder`` overrides the per-URL response. Otherwise the default
    `_synthetic_responder()` is used. Returns a call-log dict for
    assertions (`{"urls": [...], "session_built": N}`).
    """
    calls: dict[str, list] = {"urls": [], "session_built": 0}

    if responder is None:
        responder = _synthetic_responder()

    class _Session:
        def get(self, url, timeout=None, stream=None, allow_redirects=None):
            calls["urls"].append(url)
            return responder(url)

    def _fake_session():
        calls["session_built"] += 1
        return _Session()

    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.ua_swe._earthaccess_session", _fake_session
    )
    return calls


@pytest.fixture
def small_min_bytes(monkeypatch):
    """Lower `_MIN_VALID_NC_BYTES` so small synthetic test NCs pass.

    Production threshold is 10 MiB (~90 MB per real WY file); synthetic
    NCs are a few KiB. The mask's per-call `min_bytes=128*1024` is left
    alone — `_MASK_PAD_BODY` is sized to clear that floor.
    """
    monkeypatch.setattr("nhf_spatial_targets.fetch.ua_swe._MIN_VALID_NC_BYTES", 1024)


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

            # CRS ancillary carries the full Albers projection metadata
            # (CLAUDE.md requires the CF-1.6 attribute set on every
            # pipeline-written NC; assert the projection-specific bits
            # not covered by the generic `Conventions` check).
            crs = ds["crs"]
            assert crs.attrs.get("grid_mapping_name") == "albers_conical_equal_area"
            assert "crs_wkt" in crs.attrs
            assert crs.attrs.get("longitude_of_central_meridian") == pytest.approx(
                -96.0
            )
            assert crs.attrs.get("latitude_of_projection_origin") == pytest.approx(23.0)
            # standard_parallel for Albers is a 2-element array
            sp = np.asarray(crs.attrs.get("standard_parallel"))
            assert sp.shape == (2,)
            assert sp[0] == pytest.approx(29.5)
            assert sp[1] == pytest.approx(45.5)

            # Projected coordinate variables carry the CF-required attrs
            for coord_name, axis in (("x", "X"), ("y", "Y")):
                ca = ds[coord_name]
                assert (
                    ca.attrs.get("standard_name")
                    == f"projection_{coord_name}_coordinate"
                )
                assert ca.attrs.get("units") == "m"
                assert ca.attrs.get("axis") == axis

            # Time coord carries the CF-required attrs
            assert ds["time"].attrs.get("standard_name") == "time"
            assert ds["time"].attrs.get("axis") == "T"

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
# _reproject_wy_to_dataset
# ---------------------------------------------------------------------------


def test_reproject_wy_to_dataset_shape_and_vars(tmp_path: Path):
    """Characterization test for the extracted helper (issue #237).

    Verifies that :func:`_reproject_wy_to_dataset`:

    - Returns ``swe`` and ``snow_depth`` (renamed from native ``SWE`` /
      ``DEPTH``).
    - Reprojects spatial dims to ``(y, x)`` in EPSG:5070, dropping ``lat``
      / ``lon``.
    - Preserves the decoded time axis length and decodes the first
      timestamp correctly (Oct 1 of WY-1).
    - Does NOT apply CF metadata (no ``Conventions`` attr expected here).
    """
    raw = tmp_path / "4km_SWE_Depth_WY2000_v01.nc"
    # WY2000: Oct 1 1999 → Sep 30 2000 — use n_time=5 for speed.
    _make_synthetic_raw_nc(raw, wy=2000, n_time=5)

    ds = _reproject_wy_to_dataset(2000, raw)

    assert set(ds.data_vars) == {"swe", "snow_depth"}
    assert ds["swe"].dims == ("time", "y", "x")
    assert ds["snow_depth"].dims == ("time", "y", "x")
    assert "x" in ds.coords and "y" in ds.coords
    assert "lat" not in ds.dims and "lon" not in ds.dims
    assert ds.sizes["time"] == 5
    assert pd.Timestamp(ds["time"].values[0]) == pd.Timestamp("1999-10-01")
    # CF metadata NOT applied at this stage — no Conventions attr expected.
    assert "Conventions" not in ds.attrs


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
# _download_file — status code branches
# ---------------------------------------------------------------------------


class TestDownloadFile:
    """Exercise each status-code path in `_download_file`.

    Uses a stub session that returns a synthetic `_FakeResponse` per
    URL, so we can drive each branch (200, 404, short-read,
    no-Content-Length-too-small, session exception, already-present)
    without touching the network.
    """

    def _stub(self, monkeypatch, responder):
        """Return a callable session.get equivalent for direct testing."""

        class _Session:
            def get(self, url, timeout=None, stream=None, allow_redirects=None):
                return responder(url)

        return _Session()

    def test_200_downloaded(self, tmp_path: Path, monkeypatch):
        session = self._stub(
            monkeypatch, lambda url: _FakeResponse(200, body=b"X" * 4096)
        )
        out = tmp_path / "wy.nc"
        status = _download_file(
            session, "https://example.org/wy.nc", out, min_bytes=1024
        )
        assert status == "downloaded"
        assert out.exists()
        assert out.read_bytes() == b"X" * 4096
        # No .tmp left behind
        assert not (tmp_path / "wy.nc.tmp").exists()

    def test_404_returns_missing_404(self, tmp_path: Path, monkeypatch):
        session = self._stub(monkeypatch, lambda url: _FakeResponse(404))
        out = tmp_path / "wy.nc"
        status = _download_file(
            session, "https://example.org/missing.nc", out, min_bytes=1024
        )
        assert status == "missing_404"
        assert not out.exists()
        assert not (tmp_path / "wy.nc.tmp").exists()

    def test_already_present(self, tmp_path: Path, monkeypatch):
        out = tmp_path / "wy.nc"
        out.write_bytes(b"X" * 4096)
        # Session shouldn't even be called; we still hand one in.
        session = self._stub(
            monkeypatch,
            lambda url: pytest.fail(
                "session.get should not be called when file is present"
            ),
        )
        status = _download_file(
            session, "https://example.org/wy.nc", out, min_bytes=1024
        )
        assert status == "already_present"

    def test_short_read_content_length_mismatch(self, tmp_path: Path, monkeypatch):
        # Server claims 8192 bytes but only delivers 1024 — integrity check
        # should fail, .tmp should be cleaned up, status should be "error".
        def responder(url):
            return _FakeResponse(200, body=b"X" * 1024, content_length=8192)

        session = self._stub(monkeypatch, responder)
        out = tmp_path / "wy.nc"
        status = _download_file(
            session, "https://example.org/wy.nc", out, min_bytes=512
        )
        assert status == "error"
        assert not out.exists()
        assert not (tmp_path / "wy.nc.tmp").exists()

    def test_no_content_length_below_min_returns_error(
        self, tmp_path: Path, monkeypatch
    ):
        # Server omits Content-Length and the body is below min_bytes —
        # the fallback size check should reject and return "error".
        def responder(url):
            return _FakeResponse(200, body=b"X" * 128, content_length=None)

        session = self._stub(monkeypatch, responder)
        out = tmp_path / "wy.nc"
        status = _download_file(
            session, "https://example.org/wy.nc", out, min_bytes=1024
        )
        assert status == "error"
        assert not out.exists()
        assert not (tmp_path / "wy.nc.tmp").exists()

    def test_session_exception_returns_error(self, tmp_path: Path, monkeypatch):
        class _ExplodingSession:
            def get(self, *a, **kw):
                raise ConnectionError("simulated transport failure")

        out = tmp_path / "wy.nc"
        status = _download_file(
            _ExplodingSession(), "https://example.org/wy.nc", out, min_bytes=1024
        )
        assert status == "error"
        assert not out.exists()
        assert not (tmp_path / "wy.nc.tmp").exists()

    def test_non_200_non_404_returns_error(self, tmp_path: Path, monkeypatch):
        # E.g. 503 on transient outage. Without retries inside
        # _download_file (those live one layer up), the call returns
        # "error" and the caller decides whether to retry.
        session = self._stub(monkeypatch, lambda url: _FakeResponse(503))
        out = tmp_path / "wy.nc"
        status = _download_file(
            session, "https://example.org/wy.nc", out, min_bytes=1024
        )
        assert status == "error"
        assert not out.exists()


# ---------------------------------------------------------------------------
# _earthaccess_session — auth failure surface
# ---------------------------------------------------------------------------


class TestEarthaccessSession:
    def test_auth_failure_raises(self, monkeypatch):
        """`_earthaccess_session` must raise with an actionable message
        when earthaccess returns an unauthenticated Auth object."""

        class _UnauthAuth:
            authenticated = False

            def get_session(self):  # pragma: no cover - shouldn't be reached
                raise AssertionError("unauthenticated session should not be used")

        monkeypatch.setattr("earthaccess.login", lambda strategy="netrc": _UnauthAuth())
        with pytest.raises(RuntimeError, match="Earthdata login failed"):
            _earthaccess_session()


# ---------------------------------------------------------------------------
# fetch_ua_swe — mocked-session happy path
# ---------------------------------------------------------------------------


class TestFetchUaSweDriver:
    """End-to-end fetcher tests with a stubbed Earthdata session.

    These cover the integration of: catalog→WY mapping (CY→WY+1),
    publisher-window clamping, worker-0 mask gate, per-WY download +
    consolidate loop, and the manifest read-merge-write contract.
    """

    def test_period_2010_fetches_wy_2010_and_2011(
        self, tmp_path: Path, monkeypatch, small_min_bytes
    ):
        workdir = _make_project(tmp_path)
        calls = _stub_session(monkeypatch)

        result = fetch_ua_swe(
            workdir=workdir, period="2010/2010", worker_index=0, n_workers=1
        )

        # CY 2010 touches WY 2010 (Jan-Sep) and WY 2011 (Oct-Dec).
        wy_urls = [u for u in calls["urls"] if "4km_SWE_Depth_WY" in u]
        assert any("WY2010_v01.nc" in u for u in wy_urls)
        assert any("WY2011_v01.nc" in u for u in wy_urls)
        assert len(wy_urls) == 2

        # Worker 0 always fetches the mask.
        assert any("SWE_Mask_v01.nc" in u for u in calls["urls"])

        # Each WY downloaded successfully and was consolidated.
        assert len(result["water_years"]) == 2
        for rec in result["water_years"]:
            assert rec["status"] == "downloaded"
            assert "daily_path" in rec, f"WY {rec['water_year']} missing daily_path"

    def test_consolidated_ncs_land_on_disk(
        self, tmp_path: Path, monkeypatch, small_min_bytes
    ):
        workdir = _make_project(tmp_path)
        _stub_session(monkeypatch)

        fetch_ua_swe(workdir=workdir, period="2010/2010")

        daily_dir = workdir / "datastore" / "ua_swe" / "daily"
        assert (daily_dir / "ua_swe_daily_WY2010.nc").exists()
        assert (daily_dir / "ua_swe_daily_WY2011.nc").exists()


# ---------------------------------------------------------------------------
# Manifest read-merge-write + corrupt-JSON recovery
# ---------------------------------------------------------------------------


class TestFetchUaSweManifest:
    """Memory `feedback_manifest_writers_must_merge.md` documents an
    incident where a non-merging manifest writer wiped 13 sources of
    provenance. These tests pin the read-merge-write contract on the
    `_update_manifest` path."""

    def test_read_merge_write_preserves_unrelated_sources(
        self, tmp_path: Path, monkeypatch, small_min_bytes
    ):
        workdir = _make_project(tmp_path)
        # Pre-seed manifest with an unrelated source and an older ua_swe WY.
        (workdir / "manifest.json").write_text(
            json.dumps(
                {
                    "sources": {
                        "snodas": {
                            "source_key": "snodas",
                            "period": "2003/2010",
                            "years": [{"year": 2010, "n_granules": 365}],
                        },
                        "ua_swe": {
                            "source_key": "ua_swe",
                            "water_years": [
                                {"water_year": 1982, "status": "pre-existing"}
                            ],
                        },
                    },
                    "steps": [],
                }
            )
        )

        _stub_session(monkeypatch)
        fetch_ua_swe(workdir=workdir, period="2010/2010")

        manifest = json.loads((workdir / "manifest.json").read_text())

        # snodas entry must be untouched.
        assert "snodas" in manifest["sources"]
        assert manifest["sources"]["snodas"]["period"] == "2003/2010"
        assert manifest["sources"]["snodas"]["years"] == [
            {"year": 2010, "n_granules": 365}
        ]

        # ua_swe: WY 1982 preserved + new WY 2010 + WY 2011 added, sorted.
        ua_swe_wys = manifest["sources"]["ua_swe"]["water_years"]
        years_in_manifest = [r["water_year"] for r in ua_swe_wys]
        assert years_in_manifest == [1982, 2010, 2011]
        # The pre-existing 1982 entry retains its `status` field
        # (only WYs touched by this run get overwritten).
        wy1982 = next(r for r in ua_swe_wys if r["water_year"] == 1982)
        assert wy1982["status"] == "pre-existing"

    def test_corrupt_manifest_raises_valuerror(
        self, tmp_path: Path, monkeypatch, small_min_bytes
    ):
        workdir = _make_project(tmp_path)
        (workdir / "manifest.json").write_text("{not valid json")

        _stub_session(monkeypatch)
        with pytest.raises(ValueError, match="corrupt"):
            fetch_ua_swe(workdir=workdir, period="2010/2010")


# ---------------------------------------------------------------------------
# Worker-0 mask gate
# ---------------------------------------------------------------------------


class TestFetchUaSweWorkerGate:
    """The mask file is downloaded by worker 0 only. A future refactor
    that inverts the gate would silently break parallel runs."""

    def test_worker_zero_fetches_mask(
        self, tmp_path: Path, monkeypatch, small_min_bytes
    ):
        workdir = _make_project(tmp_path)
        calls = _stub_session(monkeypatch)

        result = fetch_ua_swe(
            workdir=workdir, period="2010/2010", worker_index=0, n_workers=2
        )

        assert any("SWE_Mask_v01.nc" in u for u in calls["urls"])
        assert result["mask"] is not None
        assert result["mask"]["status"] == "downloaded"

    def test_nonzero_worker_skips_mask(
        self, tmp_path: Path, monkeypatch, small_min_bytes
    ):
        workdir = _make_project(tmp_path)
        calls = _stub_session(monkeypatch)

        result = fetch_ua_swe(
            workdir=workdir, period="2010/2010", worker_index=1, n_workers=2
        )

        assert not any("SWE_Mask_v01.nc" in u for u in calls["urls"])
        assert result["mask"] is None


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
