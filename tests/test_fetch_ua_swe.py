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

import logging

from nhf_spatial_targets.fetch.ua_swe import (
    _CONSOLIDATED_FILENAME_TEMPLATE,
    _WY_FILENAME_TEMPLATE,
    _assign_worker_calendar_years,
    _download_file,
    _earthaccess_session,
    _mask_url,
    _reproject_wy_to_dataset,
    _wy_url,
    consolidate_calendar_year_ua_swe,
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


def _make_sparse_wy_raw_nc(
    path: Path,
    timestamps: list[pd.Timestamp],
    *,
    n_lat: int = 2,
    n_lon: int = 2,
    inject_negative_index: int | None = None,
) -> None:
    """Write a synthetic raw WY NC with exactly *timestamps* as the time axis.

    Unlike :func:`_make_synthetic_raw_nc` (which always starts from Oct 1
    of WY-1 and writes ``n_time`` sequential days), this helper accepts an
    arbitrary set of timestamps.  This lets calendar-year tests write only
    the minimal boundary dates needed to exercise the seam logic without
    triggering a full-WY (365–366 day) reproject loop.

    When *inject_negative_index* is given, an integer-sentinel value
    (-9999) is stamped at that time index of ``SWE`` so tests can place a
    bad value on a specific day (e.g. one outside the target calendar-year
    window) to exercise the windowed negative-value defense.

    The layout is identical to the real NSIDC-0719 files: float32 days
    since 1900-01-01, no ``units`` attr, NAD83 lat/lon CRS.
    """
    epoch = pd.Timestamp("1900-01-01")
    days_since_epoch = np.array(
        [(ts - epoch).days for ts in timestamps],
        dtype="float32",
    )
    n_time = len(timestamps)
    lat = np.linspace(30.0, 35.0, n_lat, dtype="float32")
    lon = np.linspace(-110.0, -100.0, n_lon, dtype="float32")
    rng = np.random.default_rng(seed=42)
    swe = rng.uniform(0.0, 500.0, size=(n_time, n_lat, n_lon)).astype("float32")
    depth = rng.uniform(0.0, 2000.0, size=(n_time, n_lat, n_lon)).astype("float32")
    if inject_negative_index is not None:
        swe[inject_negative_index, 0, 0] = -9999.0

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


class TestAssignWorkerCalendarYears:
    def test_single_worker(self):
        assert _assign_worker_calendar_years([1982, 1983, 1984], 0, 1) == [
            1982,
            1983,
            1984,
        ]

    def test_round_robin(self):
        cys = [1982, 1983, 1984, 1985, 1986]
        assert _assign_worker_calendar_years(cys, 0, 2) == [1982, 1984, 1986]
        assert _assign_worker_calendar_years(cys, 1, 2) == [1983, 1985]

    def test_worker_index_out_of_range(self):
        with pytest.raises(ValueError, match="worker_index"):
            _assign_worker_calendar_years([1982], 2, 2)

    def test_negative_n_workers(self):
        with pytest.raises(ValueError, match="n_workers"):
            _assign_worker_calendar_years([1982], 0, 0)


class TestUrlConstruction:
    def test_wy_url(self):
        url = _wy_url("https://example.org/data/", 1990)
        assert url == "https://example.org/data/4km_SWE_Depth_WY1990_v01.nc"

    def test_mask_url(self):
        # Trailing slash is normalised.
        url = _mask_url("https://example.org/data/")
        assert url == "https://example.org/data/SWE_Mask_v01.nc"


# ---------------------------------------------------------------------------
# _reproject_wy_to_dataset
# ---------------------------------------------------------------------------


class TestReprojWyToDataset:
    def test_negative_value_defense(self, tmp_path: Path):
        """Sentinel < -1.0 inside the CY window raises ValueError.

        The synthetic WY1982 raw starts Oct 1 1981 (the negative sentinel
        is injected at the first time step), so the defense only fires
        when the requested calendar-year window covers that day — here
        CY 1981. (Days outside the window are sliced off before the
        scan; they get validated when their own calendar year is built.)
        """
        raw_path = tmp_path / "4km_SWE_Depth_WY1982_v01.nc"
        _make_synthetic_raw_nc(raw_path, wy=1982, inject_negative=True)

        with pytest.raises(ValueError, match="integer sentinel"):
            _reproject_wy_to_dataset(1982, raw_path, 1981)

    def test_negative_outside_window_not_scanned(self, tmp_path: Path):
        """A sentinel outside the CY window is sliced off, so no raise.

        The WY1982 raw here carries a -9999 sentinel on Oct 1 1981 (CY
        1981) and a clean value on Jan 1 1982 (CY 1982). Requesting CY
        1982 windows the Oct 1981 day out *before* the scan, so the
        defense correctly does not fire — it would fire when CY 1981 is
        built. This pins the windowed-scan semantics (issue #298).
        """
        raw_path = tmp_path / "4km_SWE_Depth_WY1982_v01.nc"
        _make_sparse_wy_raw_nc(
            raw_path,
            [pd.Timestamp("1981-10-01"), pd.Timestamp("1982-01-01")],
            inject_negative_index=0,  # the Oct 1 1981 day (CY 1981)
        )

        # No raise: the only negative day (Oct 1 1981) is not in CY 1982.
        ds = _reproject_wy_to_dataset(1982, raw_path, 1982)
        assert "swe" in ds.data_vars
        t = pd.DatetimeIndex(ds["time"].values)
        assert list(t) == [pd.Timestamp("1982-01-01")]

    def test_missing_raw_path(self, tmp_path: Path):
        """FileNotFoundError is raised for a non-existent raw path."""
        with pytest.raises(FileNotFoundError, match="raw NC not found"):
            _reproject_wy_to_dataset(1982, tmp_path / "does-not-exist.nc", 1982)


def test_reproject_wy_to_dataset_shape_and_vars(tmp_path: Path):
    """Characterization test for the extracted helper (issues #237, #298).

    Verifies that :func:`_reproject_wy_to_dataset`:

    - Returns ``swe`` and ``snow_depth`` (renamed from native ``SWE`` /
      ``DEPTH``).
    - Reprojects spatial dims to ``(y, x)`` in EPSG:5070, dropping ``lat``
      / ``lon``.
    - Slices the raw time axis to the requested calendar-year window
      **before** reprojecting (issue #298): a WY2000 raw spanning Oct 1999
      – Sep 2000 yields only the Jan–Sep 2000 days when CY 2000 is asked
      for; the Oct–Dec 1999 days are dropped (never reprojected), so they
      neither incur reproject cost nor appear in the output.
    - Does NOT apply CF metadata (no ``Conventions`` attr expected here).
    """
    raw = tmp_path / "4km_SWE_Depth_WY2000_v01.nc"
    # WY2000 raw spans Oct 1999 → Sep 2000; mix out-of-window (1999) and
    # in-window (CY 2000) boundary days so the slice is observable.
    _make_sparse_wy_raw_nc(
        raw,
        [
            pd.Timestamp("1999-10-01"),  # out of CY 2000 window
            pd.Timestamp("1999-12-31"),  # out of CY 2000 window
            pd.Timestamp("2000-01-01"),  # in window
            pd.Timestamp("2000-09-30"),  # in window
        ],
    )

    ds = _reproject_wy_to_dataset(2000, raw, 2000)

    assert set(ds.data_vars) == {"swe", "snow_depth"}
    assert ds["swe"].dims == ("time", "y", "x")
    assert ds["snow_depth"].dims == ("time", "y", "x")
    assert "x" in ds.coords and "y" in ds.coords
    assert "lat" not in ds.dims and "lon" not in ds.dims
    # Only the in-window (CY 2000) days survive; the Oct–Dec 1999 days
    # were sliced off before the per-day reproject loop.
    t = pd.DatetimeIndex(ds["time"].values)
    assert list(t) == [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-09-30")]
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


def _cy_stub_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    calendar_years: list[int],
    min_bytes_patch: bool = True,
) -> dict[str, list]:
    """Stub session + optional min-bytes patch for calendar-year fetch tests.

    For each CY in *calendar_years*, writes sparse boundary-date raw NCs
    for both WY cy and WY cy+1 so ``consolidate_calendar_year_ua_swe``
    succeeds. The stub monkeypatches ``_download_file`` directly (rather
    than ``_earthaccess_session``) to write files at the expected paths
    and return ``"downloaded"``.  This approach avoids the
    ``_synthetic_responder``'s n_time=3-days-from-Oct-1 layout, which
    does not contain Jan–Sep data that the CY consolidator needs.

    Returns a call-log dict ``{"wy_paths": [...], "mask_downloaded": bool}``.
    """
    if min_bytes_patch:
        monkeypatch.setattr(
            "nhf_spatial_targets.fetch.ua_swe._MIN_VALID_NC_BYTES", 1024
        )

    log: dict[str, list | bool] = {"wy_paths": [], "mask_downloaded": False}

    # Build a map from WY → boundary timestamps (deferred write to download time).
    # Each CY needs WY=cy (Jan-Sep dates) and WY=cy+1 (Oct-Dec dates).
    wy_timestamps: dict[int, list] = {}
    for cy in calendar_years:
        # WY cy: needs Jan 1 and Sep 30 of cy (Jan-Sep portion of CY cy).
        wy_timestamps.setdefault(cy, [])
        wy_timestamps[cy].extend(
            [
                pd.Timestamp(f"{cy}-01-01"),
                pd.Timestamp(f"{cy}-09-30"),
            ]
        )
        # WY cy+1: needs Oct 1 and Dec 31 of cy (Oct-Dec portion of CY cy).
        wy_timestamps.setdefault(cy + 1, [])
        wy_timestamps[cy + 1].extend(
            [
                pd.Timestamp(f"{cy}-10-01"),
                pd.Timestamp(f"{cy}-12-31"),
            ]
        )
    # Deduplicate timestamps per WY.
    wy_timestamps = {wy: sorted(set(ts)) for wy, ts in wy_timestamps.items()}

    def _fake_download(session, url, out_path, **kwargs):
        m = re.search(r"4km_SWE_Depth_WY(\d{4})_v01\.nc$", url)
        if m:
            wy = int(m.group(1))
            log["wy_paths"].append(str(out_path))
            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                ts = wy_timestamps.get(wy)
                if ts:
                    _make_sparse_wy_raw_nc(out_path, ts)
                else:
                    _make_synthetic_raw_nc(out_path, wy, n_time=3)
            return "downloaded"
        if "SWE_Mask_v01.nc" in url:
            log["mask_downloaded"] = True
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(_MASK_PAD_BODY)
            return "downloaded"
        return "missing_404"

    # Also stub the session so _earthaccess_session() doesn't try earthaccess.
    class _DummySession:
        def get(self, url, **kwargs):
            raise AssertionError(
                "_DummySession.get should not be called; _download_file is patched"
            )

    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.ua_swe._earthaccess_session",
        lambda: _DummySession(),
    )
    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.ua_swe._download_file",
        _fake_download,
    )
    return log


class TestFetchUaSweDriver:
    """End-to-end fetcher tests with a stubbed Earthdata session.

    These cover the integration of: publisher-window clamping,
    worker-0 mask gate, per-CY download + consolidate loop, and the
    manifest read-merge-write contract.
    """

    def test_period_2010_fetches_wy_2010_and_2011(
        self, tmp_path: Path, monkeypatch, small_min_bytes
    ):
        workdir = _make_project(tmp_path)
        log = _cy_stub_session(
            monkeypatch, calendar_years=[2010], min_bytes_patch=False
        )

        result = fetch_ua_swe(
            workdir=workdir, period="2010/2010", worker_index=0, n_workers=1
        )

        # CY 2010 needs WY 2010 (Jan-Sep) and WY 2011 (Oct-Dec).
        downloaded_wys = {int(p.split("WY")[1].split("_")[0]) for p in log["wy_paths"]}
        assert 2010 in downloaded_wys
        assert 2011 in downloaded_wys

        # Worker 0 always fetches the mask.
        assert log["mask_downloaded"]

        # Result carries one CY record with a consolidated output.
        assert len(result["calendar_years"]) == 1
        rec = result["calendar_years"][0]
        assert rec["calendar_year"] == 2010
        assert "daily_path" in rec, "CY 2010 missing daily_path"

    def test_consolidated_ncs_land_on_disk(
        self, tmp_path: Path, monkeypatch, small_min_bytes
    ):
        workdir = _make_project(tmp_path)
        _cy_stub_session(monkeypatch, calendar_years=[2010], min_bytes_patch=False)

        fetch_ua_swe(workdir=workdir, period="2010/2010")

        daily_dir = workdir / "datastore" / "ua_swe" / "daily"
        assert (daily_dir / "ua_swe_daily_2010.nc").exists()


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
        # Pre-seed manifest with an unrelated source and an older ua_swe CY.
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
                            "calendar_years": [
                                {"calendar_year": 1982, "status": "pre-existing"}
                            ],
                        },
                    },
                    "steps": [],
                }
            )
        )

        _cy_stub_session(monkeypatch, calendar_years=[2010], min_bytes_patch=False)
        fetch_ua_swe(workdir=workdir, period="2010/2010")

        manifest = json.loads((workdir / "manifest.json").read_text())

        # snodas entry must be untouched.
        assert "snodas" in manifest["sources"]
        assert manifest["sources"]["snodas"]["period"] == "2003/2010"
        assert manifest["sources"]["snodas"]["years"] == [
            {"year": 2010, "n_granules": 365}
        ]

        # ua_swe: CY 1982 preserved + new CY 2010 added, sorted.
        ua_swe_cys = manifest["sources"]["ua_swe"]["calendar_years"]
        years_in_manifest = [r["calendar_year"] for r in ua_swe_cys]
        assert years_in_manifest == [1982, 2010]
        # The pre-existing 1982 entry retains its `status` field
        # (only CYs touched by this run get overwritten).
        cy1982 = next(r for r in ua_swe_cys if r["calendar_year"] == 1982)
        assert cy1982["status"] == "pre-existing"
        # No legacy water_years key on the ua_swe source entry.
        assert "water_years" not in manifest["sources"]["ua_swe"]

    def test_corrupt_manifest_raises_valuerror(
        self, tmp_path: Path, monkeypatch, small_min_bytes
    ):
        workdir = _make_project(tmp_path)
        (workdir / "manifest.json").write_text("{not valid json")

        _cy_stub_session(monkeypatch, calendar_years=[2010], min_bytes_patch=False)
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
        log = _cy_stub_session(
            monkeypatch, calendar_years=[2010, 2011], min_bytes_patch=False
        )

        result = fetch_ua_swe(
            workdir=workdir, period="2010/2011", worker_index=0, n_workers=2
        )

        assert log["mask_downloaded"]
        assert result["mask"] is not None
        assert result["mask"]["status"] == "downloaded"

    def test_nonzero_worker_skips_mask(
        self, tmp_path: Path, monkeypatch, small_min_bytes
    ):
        workdir = _make_project(tmp_path)
        log = _cy_stub_session(
            monkeypatch, calendar_years=[2010, 2011], min_bytes_patch=False
        )

        result = fetch_ua_swe(
            workdir=workdir, period="2010/2011", worker_index=1, n_workers=2
        )

        assert not log["mask_downloaded"]
        assert result["mask"] is None


# ---------------------------------------------------------------------------
# consolidate_calendar_year_ua_swe  (issue #237)
# ---------------------------------------------------------------------------


def _build_cy_raw_fixtures(raw_dir: Path, calendar_year: int) -> None:
    """Write the two adjacent WY raw NCs needed to assemble one calendar year.

    Uses :func:`_make_sparse_wy_raw_nc` with only boundary dates to keep
    per-test reproject runtime low (4 days × 2 files = 8 reprojections).

    Calendar year *X* is assembled from:
    - WY *X* (``4km_SWE_Depth_WY{X}_v01.nc``): Jan 1 and Sep 30 of CY X
    - WY *X+1* (``4km_SWE_Depth_WY{X+1}_v01.nc``): Oct 1 and Dec 31 of CY X
    """
    cy = calendar_year
    # WY X = the file named WY{cy}; it runs Oct 1 (cy-1) – Sep 30 (cy).
    # We only need the Jan-Sep portion that falls in CY cy.
    wy_x_path = raw_dir / _WY_FILENAME_TEMPLATE.format(wy=cy)
    _make_sparse_wy_raw_nc(
        wy_x_path,
        [
            pd.Timestamp(f"{cy}-01-01"),
            pd.Timestamp(f"{cy}-09-30"),
        ],
    )
    # WY X+1 = the file named WY{cy+1}; it runs Oct 1 (cy) – Sep 30 (cy+1).
    # We only need the Oct-Dec portion that falls in CY cy.
    wy_x1_path = raw_dir / _WY_FILENAME_TEMPLATE.format(wy=cy + 1)
    _make_sparse_wy_raw_nc(
        wy_x1_path,
        [
            pd.Timestamp(f"{cy}-10-01"),
            pd.Timestamp(f"{cy}-12-31"),
        ],
    )


def test_consolidate_calendar_year_spans_jan_to_dec(tmp_path: Path):
    """CY output covers Jan 1 – Dec 31, seamlessly stitched from WY X and WY X+1."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    daily = tmp_path / "daily"

    _build_cy_raw_fixtures(raw_dir, 2000)

    out = consolidate_calendar_year_ua_swe(2000, raw_dir, daily)

    assert out.name == _CONSOLIDATED_FILENAME_TEMPLATE.format(year=2000)

    with xr.open_dataset(out) as ds:
        t = pd.DatetimeIndex(ds["time"].values)

        # Full Jan-Dec span
        assert t.min() == pd.Timestamp("2000-01-01")
        assert t.max() == pd.Timestamp("2000-12-31")

        # Monotonically increasing, no duplicates
        assert t.is_monotonic_increasing
        assert t.is_unique
        # The fixture writes exactly 4 boundary dates: Jan 1, Sep 30, Oct 1, Dec 31.
        # The WY→WY seam (Sep 30 → Oct 1) must be exactly 1 day — no gap or dup.
        sep30 = pd.Timestamp("2000-09-30")
        oct01 = pd.Timestamp("2000-10-01")
        assert sep30 in t, "Sep 30 must be in the output (from WY X)"
        assert oct01 in t, "Oct 1 must be in the output (from WY X+1)"
        seam_gap = t[t.get_loc(oct01)] - t[t.get_loc(sep30)]
        assert seam_gap == pd.Timedelta(days=1), (
            f"Seam gap must be exactly 1 day (Sep 30 → Oct 1), got {seam_gap}"
        )

        # CF-1.6 compliance: Conventions attr + key variable attrs
        assert ds.attrs.get("Conventions") == "CF-1.6"
        assert ds["swe"].attrs["units"] == "kg m-2"
        assert ds["snow_depth"].attrs.get("units"), "snow_depth must have a units attr"

        # Projected coords survive concat
        assert "x" in ds.coords and "y" in ds.coords
        assert "lat" not in ds.dims and "lon" not in ds.dims


def test_consolidate_calendar_year_boundary_raises(tmp_path: Path):
    """Archive boundary: WY X+1 absent → FileNotFoundError (no partial CY)."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    daily = tmp_path / "daily"

    # Write only WY 2023 (the file named 4km_SWE_Depth_WY2023_v01.nc);
    # WY 2024 is absent, simulating the end-of-archive boundary.
    wy2023_path = raw_dir / _WY_FILENAME_TEMPLATE.format(wy=2023)
    _make_sparse_wy_raw_nc(
        wy2023_path,
        [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-09-30")],
    )

    with pytest.raises(FileNotFoundError):
        consolidate_calendar_year_ua_swe(2023, raw_dir, daily)


def test_consolidate_calendar_year_missing_wy_x_raises(tmp_path: Path):
    """Raise FileNotFoundError when WY X itself is missing."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    daily = tmp_path / "daily"

    # Write only WY 2001 (needed for Oct-Dec of CY 2000); WY 2000 absent.
    wy2001_path = raw_dir / _WY_FILENAME_TEMPLATE.format(wy=2001)
    _make_sparse_wy_raw_nc(
        wy2001_path,
        [pd.Timestamp("2000-10-01"), pd.Timestamp("2000-12-31")],
    )

    with pytest.raises(FileNotFoundError):
        consolidate_calendar_year_ua_swe(2000, raw_dir, daily)


def test_consolidate_calendar_year_idempotent(tmp_path: Path):
    """Calling twice returns the same path; second call skips (mtime-idempotent)."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    daily = tmp_path / "daily"

    _build_cy_raw_fixtures(raw_dir, 2000)

    out1 = consolidate_calendar_year_ua_swe(2000, raw_dir, daily)
    mtime1 = out1.stat().st_mtime

    out2 = consolidate_calendar_year_ua_swe(2000, raw_dir, daily)
    assert out2 == out1
    assert out2.stat().st_mtime == pytest.approx(mtime1, abs=1e-3)


def test_consolidate_calendar_year_rebuild_on_newer_raw(tmp_path: Path):
    """Staleness branch: output is rebuilt when either raw is newer than the output.

    Mirrors :meth:`TestConsolidateWaterYear.test_rebuild_on_newer_raw` for
    the calendar-year path.  The idempotency check compares ``out.st_mtime``
    against ``max(raw_x.st_mtime, raw_x1.st_mtime)``; touching one of the
    raws to a strictly later time than the output must trigger a rebuild.
    """
    import os

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    daily = tmp_path / "daily"

    _build_cy_raw_fixtures(raw_dir, 2000)

    # First call: build and record the output mtime.
    out = consolidate_calendar_year_ua_swe(2000, raw_dir, daily)
    first_mtime = out.stat().st_mtime

    # Touch WY 2001 (raw_x1) to a strictly later mtime.
    raw_x1 = raw_dir / _WY_FILENAME_TEMPLATE.format(wy=2001)
    t = first_mtime + 10
    os.utime(raw_x1, (t, t))

    # Second call: staleness detected → output rebuilt (mtime advanced).
    out2 = consolidate_calendar_year_ua_swe(2000, raw_dir, daily)
    assert out2.stat().st_mtime > first_mtime


# ---------------------------------------------------------------------------
# fetch_ua_swe calendar-year flow (Tasks 4 + 5 — issue #237)
# ---------------------------------------------------------------------------


def test_fetch_ua_swe_consolidates_by_calendar_year(
    tmp_path: Path, monkeypatch, small_min_bytes
):
    """fetch_ua_swe produces per-CY NCs keyed by calendar year (not WY)."""
    workdir = _make_project(tmp_path)
    _cy_stub_session(monkeypatch, calendar_years=[2000, 2001], min_bytes_patch=False)

    result = fetch_ua_swe(
        workdir=workdir, period="2000/2001", worker_index=0, n_workers=1
    )

    daily_dir = workdir / "datastore" / "ua_swe" / "daily"
    assert (daily_dir / "ua_swe_daily_2000.nc").exists(), "CY 2000 NC missing"
    assert (daily_dir / "ua_swe_daily_2001.nc").exists(), "CY 2001 NC missing"

    cy_years = {r["calendar_year"] for r in result["calendar_years"]}
    assert cy_years == {2000, 2001}
    assert (
        result["n_consolidated"] == 2
        and result["n_failed"] == 0
        and result["n_skipped"] == 0
    )


def test_fetch_ua_swe_drops_partial_edge_years(
    tmp_path: Path, monkeypatch, small_min_bytes
):
    """CY that needs WY X+1 beyond publisher_wy_hi is silently dropped.

    The catalog publishes WY 1982-2023 (catalog period "1981/2023").
    publisher_wy_hi = 2023.  CY 2023 would need WY 2024, which is
    outside the published range → dropped. CY 2022 (needs WY 2023) is
    kept.
    """
    workdir = _make_project(tmp_path)
    # Only CY 2022 is publishable; CY 2023 needs WY 2024 (outside window).
    _cy_stub_session(monkeypatch, calendar_years=[2022], min_bytes_patch=False)

    result = fetch_ua_swe(
        workdir=workdir, period="2022/2023", worker_index=0, n_workers=1
    )

    cy_years = {r["calendar_year"] for r in result["calendar_years"]}
    assert 2022 in cy_years, "CY 2022 should be produced"
    assert 2023 not in cy_years, "CY 2023 should be dropped (needs WY 2024)"


def test_manifest_records_calendar_years(tmp_path: Path, monkeypatch, small_min_bytes):
    """After fetch, manifest carries calendar_years (not water_years) + consolidate step."""
    workdir = _make_project(tmp_path)
    _cy_stub_session(monkeypatch, calendar_years=[2000], min_bytes_patch=False)

    fetch_ua_swe(workdir=workdir, period="2000/2000", worker_index=0, n_workers=1)

    manifest = json.loads((workdir / "manifest.json").read_text())
    ua = manifest["sources"]["ua_swe"]

    # calendar_years key present with the expected CY record.
    assert "calendar_years" in ua, "manifest must have calendar_years key"
    cy_entries = ua["calendar_years"]
    assert len(cy_entries) >= 1
    cy_keys = {r["calendar_year"] for r in cy_entries}
    assert 2000 in cy_keys

    # No legacy water_years key on the source entry.
    assert "water_years" not in ua, "manifest must NOT have legacy water_years key"

    # A consolidate step with calendar_years param and outputs pointing at CY NC.
    consolidate_steps = [
        s
        for s in manifest.get("steps", [])
        if s.get("kind") == "consolidate" and s.get("source_key") == "ua_swe"
    ]
    assert consolidate_steps, "at least one consolidate step for ua_swe"
    step = consolidate_steps[-1]
    assert "calendar_years" in step["params"], "step params must have calendar_years"
    assert 2000 in step["params"]["calendar_years"]
    # The step outputs include the CY 2000 NC path.
    output_paths = [o["path"] for o in step.get("outputs", [])]
    assert any("ua_swe_daily_2000.nc" in p for p in output_paths), (
        f"CY 2000 NC not in step outputs: {output_paths}"
    )


# ---------------------------------------------------------------------------
# Change D: short-year completeness warning
# ---------------------------------------------------------------------------


def test_consolidate_calendar_year_warns_on_short_year(tmp_path: Path, caplog):
    """Assembling a CY from sparse raws (fewer than 365/366 days) emits WARNING.

    The sparse fixtures produced by _build_cy_raw_fixtures contain only 4
    boundary dates (Jan 1, Sep 30, Oct 1, Dec 31), far fewer than 365/366
    days, so the assembled dataset is short.  The consolidator must emit a
    WARNING containing 'expected' and 'daily steps' — and the NC must still
    be written (warning, not raise).
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    daily = tmp_path / "daily"

    _build_cy_raw_fixtures(raw_dir, 2001)  # non-leap: expected 365

    with caplog.at_level(logging.WARNING, logger="nhf_spatial_targets.fetch.ua_swe"):
        out = consolidate_calendar_year_ua_swe(2001, raw_dir, daily)

    # NC must still be written
    assert out.exists(), (
        "consolidated NC must be written even when a warning is emitted"
    )

    # Warning must mention the day-count discrepancy
    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any(
        "expected" in str(m) and "daily steps" in str(m) for m in warning_messages
    ), f"Expected short-year warning not found; got: {warning_messages}"


# ---------------------------------------------------------------------------
# Change G: download-failed → CY-skipped branch
# ---------------------------------------------------------------------------


def test_fetch_ua_swe_skips_cy_when_wy_download_fails(
    tmp_path: Path, monkeypatch, small_min_bytes
):
    """A CY is skipped (no daily_path, no consolidate_error) when one of its
    WY downloads fails with 'missing_404'.

    Period 2010/2011 covers publishable CYs [2010, 2011].
    - CY 2010 needs WY 2010 and WY 2011.
    - CY 2011 needs WY 2011 and WY 2012.
    Injecting a missing_404 for WY 2011 means:
    - CY 2010 is skipped (wy_status has missing_404 for WY 2011).
    - CY 2011 is skipped (wy_status has missing_404 for WY 2011).
    So n_skipped >= 1 and no daily NC for any affected CY.
    """
    workdir = _make_project(tmp_path)

    monkeypatch.setattr("nhf_spatial_targets.fetch.ua_swe._MIN_VALID_NC_BYTES", 1024)

    # Build boundary timestamps for all WYs that will be attempted (except WY 2011).
    # CYs 2010 and 2011 both need WY 2011 which will be injected as missing_404.
    wy_timestamps: dict[int, list] = {
        2010: [pd.Timestamp("2010-01-01"), pd.Timestamp("2010-09-30")],
        2011: None,  # sentinel: this WY returns missing_404
        2012: [pd.Timestamp("2011-10-01"), pd.Timestamp("2011-12-31")],
    }

    def _fake_download(session, url, out_path, **kwargs):
        m = re.search(r"4km_SWE_Depth_WY(\d{4})_v01\.nc$", url)
        if m:
            wy = int(m.group(1))
            if wy_timestamps.get(wy) is None:
                # Simulate a 404 for WY 2011
                return "missing_404"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                _make_sparse_wy_raw_nc(out_path, wy_timestamps[wy])
            return "downloaded"
        if "SWE_Mask_v01.nc" in url:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(_MASK_PAD_BODY)
            return "downloaded"
        return "missing_404"

    class _DummySession:
        def get(self, url, **kwargs):
            raise AssertionError("_DummySession.get should not be called")

    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.ua_swe._earthaccess_session",
        lambda: _DummySession(),
    )
    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.ua_swe._download_file",
        _fake_download,
    )

    result = fetch_ua_swe(
        workdir=workdir, period="2010/2011", worker_index=0, n_workers=1
    )

    daily_dir = workdir / "datastore" / "ua_swe" / "daily"

    # Both CYs need WY 2011 → both skipped
    for rec in result["calendar_years"]:
        cy = rec["calendar_year"]
        assert "daily_path" not in rec, f"CY {cy} should be skipped, not consolidated"
        assert "consolidate_error" not in rec, (
            f"CY {cy} skip must be a download-skip (no consolidate_error), got: {rec}"
        )
        assert "missing_404" in str(rec["wy_status"].values()), (
            f"CY {cy} wy_status must show the missing_404: {rec['wy_status']}"
        )
        assert not (daily_dir / f"ua_swe_daily_{cy}.nc").exists(), (
            f"ua_swe_daily_{cy}.nc must not exist for a skipped CY"
        )

    assert result["n_skipped"] >= 1


# ---------------------------------------------------------------------------
# Change H: multi-worker partition end-to-end
# ---------------------------------------------------------------------------


def test_fetch_ua_swe_worker_sharding_partitions_calendar_years(
    tmp_path: Path, monkeypatch, small_min_bytes
):
    """Worker round-robin: with n_workers=2 and period 2010/2011, worker 0
    processes CY 2010 and worker 1 processes CY 2011 (publishable CYs in
    round-robin order [2010, 2011][i::2]).
    """
    workdir = _make_project(tmp_path)

    # Both CYs need WY 2010, 2011, 2012 — stub all of them.
    _cy_stub_session(monkeypatch, calendar_years=[2010, 2011], min_bytes_patch=False)

    result_w0 = fetch_ua_swe(
        workdir=workdir, period="2010/2011", worker_index=0, n_workers=2
    )
    result_w1 = fetch_ua_swe(
        workdir=workdir, period="2010/2011", worker_index=1, n_workers=2
    )

    cys_w0 = {r["calendar_year"] for r in result_w0["calendar_years"]}
    cys_w1 = {r["calendar_year"] for r in result_w1["calendar_years"]}

    assert cys_w0 == {2010}, f"Worker 0 should get CY 2010 only, got {cys_w0}"
    assert cys_w1 == {2011}, f"Worker 1 should get CY 2011 only, got {cys_w1}"


# ---------------------------------------------------------------------------
# Integration (skipped by default; opt-in via -m integration)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_one_water_year_live(tmp_path: Path):
    """Live download of one CY from NSIDC and consolidate.

    Requires Earthdata Login credentials in ~/.netrc (run
    ``nhf-targets materialize-credentials`` first). Skipped by default;
    opt in via ``pixi run -e dev test-integration``.
    """
    workdir = _make_project(tmp_path)
    result = fetch_ua_swe(
        workdir=workdir,
        period="2010/2010",  # CY 2010: needs WY 2010 + WY 2011
        worker_index=0,
        n_workers=1,
    )
    assert result["source_key"] == "ua_swe"
    assert len(result["calendar_years"]) >= 1
    # At least one CY should have consolidated successfully.
    assert any("daily_path" in rec for rec in result["calendar_years"])
