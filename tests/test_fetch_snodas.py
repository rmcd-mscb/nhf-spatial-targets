"""Tests for SNODAS daily fetch + consolidation (issues #107 and #109).

The original fetch (PR #108) used ``earthaccess.search_data`` against a
CMR collection that turned out to be a metadata-only stub with zero
granule records; the rewrite constructs URLs directly from dates and
streams each ``.tar`` via ``earthaccess.login().get_session()``. PR for
issue #109 added per-year consolidation: every year's ``.tars`` are
decoded into a CF NetCDF at ``<datastore>/snodas/daily/snodas_daily_<year>.nc``
after the download loop, mirroring ``era5_land.consolidate_year``.

These tests cover: the period gate, worker sharding, per-day status
accounting (downloaded / already-present / 404 / error), manifest
accumulation, the EDL auth-failure surface, decoder correctness against
a synthetic SNODAS .tar, ``consolidate_year_snodas`` happy path +
idempotency + within-year-grid-mismatch detection, the backfill
workflow (raw on disk, no daily NC → next run consolidates only), and
that the completion check now keys on ``daily_path`` existence rather
than ``n_granules > 0`` alone. No network is touched.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import xarray as xr
import yaml

from nhf_spatial_targets.fetch.snodas import fetch_snodas


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_project(tmp_path: Path) -> Path:
    """Materialize a minimal valid project directory.

    SNODAS no longer needs a fabric bbox at fetch time (download is a
    full-CONUS daily .tar regardless of project fabric), but the project
    loader still requires fabric.json to exist with a basic shape.
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


# Synthetic-tar helpers — used by the consolidation tests below. A real
# SNODAS .tar contains 8 product pairs and is 10-30 MB; for tests we
# only ship the SWE pair (product code 1034) plus a high-entropy
# padding member to push the .tar past the `_MIN_VALID_TAR_BYTES` floor.
# The notebook `notebooks/consolidated/inspect_consolidated_snodas.ipynb`
# documents the on-disk header format.

# Small but plausible CONUS grid: 4×4 cells at 30-arcsec resolution.
_TINY_ROWS = 4
_TINY_COLS = 4
_TINY_DX = 0.008333333333000  # 30 arcsec (matches real SNODAS headers)
_TINY_LON_MIN = -124.733750000000000
_TINY_LAT_MAX = 52.874583333333000


def _synthetic_snodas_header(rows: int, cols: int) -> str:
    """Return a minimal-but-valid SNODAS SWE .Hdr text.

    Mirrors a real header's keys exactly; ``consolidate_year_snodas``
    only reads the grid/shape/resolution keys.
    """
    lon_min = _TINY_LON_MIN
    lat_max = _TINY_LAT_MAX
    dx = _TINY_DX
    lon_max = lon_min + cols * dx
    lat_min = lat_max - rows * dx
    return (
        "Format version: NOHRSC GIS/RS raster file v1.1\n"
        "Description: Modeled snow water equivalent, total of snow layers\n"
        "Data units: Meters / 1000.000000\n"
        "Data type: integer\n"
        "Data bytes per pixel: 2\n"
        "Data intercept: 0.00000000000000\n"
        "Data slope: 1.000000000000000\n"
        "No data value: -9999.000000000000000\n"
        f"Number of columns: {cols}\n"
        f"Number of rows: {rows}\n"
        f"Minimum x-axis coordinate: {lon_min:.15f}\n"
        f"Maximum x-axis coordinate: {lon_max:.15f}\n"
        f"Minimum y-axis coordinate: {lat_min:.15f}\n"
        f"Maximum y-axis coordinate: {lat_max:.15f}\n"
        f"X-axis resolution: {dx:.15f}\n"
        f"Y-axis resolution: {dx:.15f}\n"
    )


def _build_synthetic_snodas_tar(
    yyyymmdd: str,
    *,
    rows: int = _TINY_ROWS,
    cols: int = _TINY_COLS,
    swe_int16: np.ndarray | None = None,
    header_overrides: dict[str, str] | None = None,
    pad_bytes: int = 1_200_000,  # > _MIN_VALID_TAR_BYTES (1 MiB)
) -> bytes:
    """Build raw bytes of a minimal valid SNODAS .tar for ``yyyymmdd``.

    Carries one SWE product pair (header + binary) plus a high-entropy
    padding member to push the .tar past `_MIN_VALID_TAR_BYTES` (1 MiB)
    so the size-floor check accepts it.
    """
    if swe_int16 is None:
        swe_int16 = np.full((rows, cols), 100, dtype=np.int16)
    if swe_int16.shape != (rows, cols):
        raise ValueError(f"swe_int16 shape {swe_int16.shape} != ({rows}, {cols})")
    header_text = _synthetic_snodas_header(rows, cols)
    if header_overrides:
        for k, v in header_overrides.items():
            header_text = re.sub(
                rf"^{re.escape(k)}: .*$",
                f"{k}: {v}",
                header_text,
                flags=re.MULTILINE,
            )

    base = f"us_ssmv11034tS__T0001TTNATS{yyyymmdd}05HP001"
    # mtime=0 makes the gzip header deterministic across calls so the
    # synthetic .tar bytes are reproducible (the backfill test compares
    # round-trip bytes byte-for-byte).
    hdr_gz = gzip.compress(header_text.encode("latin-1"), mtime=0)
    # SNODAS binary is big-endian int16; ``astype(">i2").tobytes()`` writes
    # in big-endian order regardless of host byte order.
    bin_gz = gzip.compress(swe_int16.astype(">i2").tobytes(), mtime=0)
    # High-entropy padding (defeats gzip compression so the tar grows
    # past 1 MiB without needing a huge SWE grid).
    pad = np.random.default_rng(seed=0).bytes(pad_bytes)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in (
            (f"{base}.txt.gz", hdr_gz),
            (f"{base}.dat.gz", bin_gz),
            ("padding.bin", pad),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _synthetic_tar_responder(
    rows: int = _TINY_ROWS, cols: int = _TINY_COLS
) -> Callable[[str], "_FakeResponse"]:
    """Return a `_stub_session` responder that serves valid synthetic tars."""

    def responder(url: str) -> "_FakeResponse":
        m = re.search(r"SNODAS_(\d{8})\.tar$", url)
        if not m:
            return _FakeResponse(404)
        body = _build_synthetic_snodas_tar(m.group(1), rows=rows, cols=cols)
        return _FakeResponse(200, body=body)

    return responder


class _FakeResponse:
    """Tiny stand-in for ``requests.Response`` covering the bits we use.

    `Content-Length` is auto-derived from the body length unless an
    explicit value is passed via ``content_length``. Pass
    ``content_length=None`` to simulate a server that omits the header.
    Pass an integer that *disagrees* with the body length to exercise
    the short-read integrity path.
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
            # (NSIDC's archive does this on tarball responses).
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
    body: bytes = b"FAKE_SNODAS_TAR",
) -> dict[str, list]:
    """Patch ``_earthaccess_session`` with a session whose ``.get`` is stubbed.

    ``responder`` overrides the per-URL response (e.g. to return 404
    for boundary dates). Otherwise every URL returns a 200 with
    ``body``. Returns a call-log dict for assertions.
    """
    calls: dict[str, list] = {"urls": [], "session_built": 0}

    if responder is None:

        def responder(url: str) -> _FakeResponse:  # type: ignore[misc]
            return _FakeResponse(200, body=body)

    class _Session:
        def get(self, url, timeout=None, stream=None, allow_redirects=None):
            calls["urls"].append(url)
            return responder(url)

    def _fake_session():
        calls["session_built"] += 1
        return _Session()

    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.snodas._earthaccess_session", _fake_session
    )
    return calls


# ---------------------------------------------------------------------------
# Period / catalog gate
# ---------------------------------------------------------------------------


def test_period_before_publisher_window_rejected(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_session(monkeypatch)
    with pytest.raises(ValueError, match="outside the SNODAS publisher window"):
        fetch_snodas(workdir=workdir, period="2000/2002")


def test_missing_archive_url_raises(tmp_path, monkeypatch):
    """If catalog access.archive_url is missing, fail fast with instructions.

    Exercises the catalog-validity guard by monkeypatching the catalog
    `archive_url` to empty.
    """
    workdir = _make_project(tmp_path)
    _stub_session(monkeypatch)

    from nhf_spatial_targets import catalog as _catalog

    real_source = _catalog.source

    def fake_source(name: str):
        s = real_source(name).copy()
        if name == "snodas":
            s["access"] = {**s["access"], "archive_url": ""}
        return s

    monkeypatch.setattr("nhf_spatial_targets.fetch.snodas._catalog.source", fake_source)
    with pytest.raises(ValueError, match="archive_url"):
        fetch_snodas(workdir=workdir, period="2020/2020")


# ---------------------------------------------------------------------------
# Worker partitioning
# ---------------------------------------------------------------------------


def test_worker_partitioning_3_years_3_workers(tmp_path, monkeypatch):
    """Three workers across three years each cover one distinct year.

    Each worker gets its own project directory so the partitioning logic
    is exercised against an empty manifest — exactly what parallel
    workers see at startup against the shared real manifest.
    """
    _stub_session(monkeypatch)
    seen_years: list[int] = []
    for wi in range(3):
        workdir = _make_project(tmp_path / f"worker_{wi}")
        result = fetch_snodas(
            workdir=workdir,
            period="2020/2022",
            worker_index=wi,
            n_workers=3,
        )
        for rec in result["years"]:
            seen_years.append(rec["year"])
    assert sorted(seen_years) == [2020, 2021, 2022]


def test_worker_with_no_assigned_years_returns_empty(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_session(monkeypatch)
    # 1 year, 2 workers → worker 1 gets nothing
    result = fetch_snodas(
        workdir=workdir,
        period="2020/2020",
        worker_index=1,
        n_workers=2,
    )
    assert result["years"] == []
    assert result["worker_index"] == 1


def test_worker_index_out_of_range_raises(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_session(monkeypatch)
    with pytest.raises(ValueError, match="worker_index must be in"):
        fetch_snodas(
            workdir=workdir,
            period="2020/2020",
            worker_index=3,
            n_workers=2,
        )


# ---------------------------------------------------------------------------
# URL construction and download accounting
# ---------------------------------------------------------------------------


def test_full_year_download_writes_tars_to_year_dir(tmp_path, monkeypatch):
    """Happy path: 200 OK for every day → 365 .tars on disk + recorded counts."""
    workdir = _make_project(tmp_path)
    _stub_session(monkeypatch, body=b"\x00\x01\x02")
    result = fetch_snodas(workdir=workdir, period="2020/2020")
    rec = result["years"][0]
    assert rec["year"] == 2020
    # 2020 is a leap year — 366 days.
    assert rec["n_downloaded_this_run"] == 366
    assert rec["n_already_present"] == 0
    assert rec["n_missing_404"] == 0
    assert rec["n_errors"] == 0
    assert rec["n_granules"] == 366
    year_dir = tmp_path / "datastore" / "snodas" / "raw" / "2020"
    tars = sorted(year_dir.glob("SNODAS_*.tar"))
    assert len(tars) == 366
    # First and last day filenames are exact.
    assert tars[0].name == "SNODAS_20200101.tar"
    assert tars[-1].name == "SNODAS_20201231.tar"


def test_url_layout_matches_nsidc_archive(tmp_path, monkeypatch):
    """URLs follow ``<archive>/<year>/MM_Mon/SNODAS_YYYYMMDD.tar``."""
    workdir = _make_project(tmp_path)
    calls = _stub_session(monkeypatch)
    fetch_snodas(workdir=workdir, period="2020/2020")
    urls = calls["urls"]
    # Sample a January 1 URL.
    jan1 = next(u for u in urls if u.endswith("SNODAS_20200101.tar"))
    assert "/2020/01_Jan/" in jan1
    assert jan1.startswith("https://noaadata.apps.nsidc.org/NOAA/G02158/masked/")
    # December 31 URL.
    dec31 = next(u for u in urls if u.endswith("SNODAS_20201231.tar"))
    assert "/2020/12_Dec/" in dec31


def test_404_days_recorded_not_raised(tmp_path, monkeypatch):
    """Per-day 404s are normal (partial-year boundaries); record, don't fail."""
    workdir = _make_project(tmp_path)

    def responder(url: str) -> _FakeResponse:
        # First 30 days of the year are missing (mimics a partial-year boundary).
        for d in range(1, 31):
            if url.endswith(f"SNODAS_202001{d:02d}.tar"):
                return _FakeResponse(404)
        return _FakeResponse(200, body=b"\x00")

    _stub_session(monkeypatch, responder=responder)
    result = fetch_snodas(workdir=workdir, period="2020/2020")
    rec = result["years"][0]
    assert rec["n_missing_404"] == 30
    assert rec["n_downloaded_this_run"] == 366 - 30
    assert rec["n_errors"] == 0


def test_already_present_files_are_skipped(tmp_path, monkeypatch):
    """Pre-existing realistically-sized .tar files are accounted as already_present.

    The fetch now requires existing files to be at least
    :data:`_MIN_VALID_TAR_BYTES` (1 MiB) to count as already-present,
    so a few-byte stub from a pre-fix corrupted run gets re-downloaded
    rather than silently treated as valid.
    """
    workdir = _make_project(tmp_path)
    year_dir = tmp_path / "datastore" / "snodas" / "raw" / "2020"
    year_dir.mkdir(parents=True)
    # Pre-stage 5 realistically-sized .tars (>= 1 MiB) on disk.
    payload = b"PRE_STAGED" * (200 * 1024)  # ~2 MiB
    for day in range(1, 6):
        (year_dir / f"SNODAS_202001{day:02d}.tar").write_bytes(payload)
    _stub_session(monkeypatch)
    result = fetch_snodas(workdir=workdir, period="2020/2020")
    rec = result["years"][0]
    assert rec["n_already_present"] == 5
    assert rec["n_downloaded_this_run"] == 366 - 5
    assert rec["n_granules"] == 366
    # Pre-staged files were NOT overwritten.
    assert (year_dir / "SNODAS_20200101.tar").read_bytes() == payload


def test_suspiciously_small_existing_file_is_redownloaded(tmp_path, monkeypatch):
    """A few-byte stub from a pre-fix corrupted run is redownloaded.

    Defends against the truncated-body bug class flagged in PR #108
    review: a previous, weaker `>0` predicate would lock such stubs
    in forever. The new predicate requires `>= _MIN_VALID_TAR_BYTES`.
    """
    workdir = _make_project(tmp_path)
    year_dir = tmp_path / "datastore" / "snodas" / "raw" / "2020"
    year_dir.mkdir(parents=True)
    stub_path = year_dir / "SNODAS_20200101.tar"
    stub_path.write_bytes(b"oops")  # 4 bytes, way below 1 MiB
    _stub_session(monkeypatch)
    fetch_snodas(workdir=workdir, period="2020/2020")
    # Stub was replaced by the (fake) full payload.
    assert stub_path.read_bytes() != b"oops"


def test_short_read_with_content_length_mismatch_recorded_as_error(
    tmp_path, monkeypatch
):
    """A server that drops mid-stream below Content-Length yields `n_errors`.

    Regression guard for the PR #108 review must-fix: `iter_content`
    does not raise on a short read, so without an explicit
    Content-Length check, a truncated body would be installed as
    `downloaded` and the next run would skip it forever. The check
    must catch the mismatch and unlink the .tar.tmp.
    """
    workdir = _make_project(tmp_path)
    # Server claims 1000 bytes but delivers 10.
    short_resp = _FakeResponse(200, body=b"\x00" * 10, content_length=1000)
    _stub_session(monkeypatch, responder=lambda url: short_resp)
    result = fetch_snodas(workdir=workdir, period="2020/2020")
    rec = result["years"][0]
    assert rec["n_errors"] == 366  # every day failed integrity
    assert rec["n_downloaded_this_run"] == 0
    # No partial .tar.tmp files left behind.
    year_dir = tmp_path / "datastore" / "snodas" / "raw" / "2020"
    leftovers = list(year_dir.glob("*.tmp"))
    assert leftovers == []


def test_no_content_length_small_body_recorded_as_error(tmp_path, monkeypatch):
    """Server omits Content-Length AND the body is below the size floor.

    Falls back to the `_MIN_VALID_TAR_BYTES` sanity check (1 MiB) and
    discards the response.
    """
    workdir = _make_project(tmp_path)
    no_cl_resp = _FakeResponse(200, body=b"\x00" * 10, content_length=None)
    _stub_session(monkeypatch, responder=lambda url: no_cl_resp)
    result = fetch_snodas(workdir=workdir, period="2020/2020")
    rec = result["years"][0]
    assert rec["n_errors"] == 366
    assert rec["n_downloaded_this_run"] == 0


def test_no_content_length_large_body_accepted(tmp_path, monkeypatch):
    """Without Content-Length, a body >= 1 MiB is accepted (defense fallback)."""
    workdir = _make_project(tmp_path)
    big = b"\x00" * (2 * 1024 * 1024)
    no_cl_resp = _FakeResponse(200, body=big, content_length=None)
    _stub_session(monkeypatch, responder=lambda url: no_cl_resp)
    result = fetch_snodas(workdir=workdir, period="2020/2020")
    rec = result["years"][0]
    assert rec["n_downloaded_this_run"] == 366
    assert rec["n_errors"] == 0


# ---------------------------------------------------------------------------
# Locale-safe URL construction (N1)
# ---------------------------------------------------------------------------


def test_month_names_are_locale_independent():
    """URL month names come from a hardcoded table, not strftime('%b').

    Regression guard: on a non-English LC_TIME (e.g. de_DE.UTF-8),
    `strftime('%b')` would emit 'Jän' / 'Mär' and every URL would 404.
    The hardcoded table makes the fetch portable across locales.
    """
    from nhf_spatial_targets.fetch.snodas import _MONTH_NAMES, _daily_urls

    assert _MONTH_NAMES[1] == "Jan"
    assert _MONTH_NAMES[12] == "Dec"

    urls = _daily_urls("https://example.test/archive", 2020)
    jan1 = next(u for _, u in urls if u.endswith("SNODAS_20200101.tar"))
    jul15 = next(u for _, u in urls if u.endswith("SNODAS_20200715.tar"))
    dec31 = next(u for _, u in urls if u.endswith("SNODAS_20201231.tar"))
    assert "/2020/01_Jan/" in jan1
    assert "/2020/07_Jul/" in jul15
    assert "/2020/12_Dec/" in dec31


# ---------------------------------------------------------------------------
# Retry adapter (C2)
# ---------------------------------------------------------------------------


def test_session_carries_retry_adapter_on_https(monkeypatch):
    """`_earthaccess_session` mounts a urllib3.Retry adapter on https://.

    Stubs earthaccess.login so we exercise the real adapter wiring
    without hitting the network. Doesn't try to exercise actual retry
    behaviour (that's urllib3's responsibility) — just confirms the
    adapter is attached with the expected config.
    """
    import earthaccess
    import requests
    from urllib3.util.retry import Retry

    from nhf_spatial_targets.fetch.snodas import _earthaccess_session

    class _StubAuth:
        authenticated = True

        def get_session(self):
            return requests.Session()

    monkeypatch.setattr(earthaccess, "login", lambda strategy=None: _StubAuth())

    session = _earthaccess_session()
    adapter = session.get_adapter("https://example.com")
    assert isinstance(adapter.max_retries, Retry)
    assert adapter.max_retries.total == 3
    assert 500 in adapter.max_retries.status_forcelist
    assert 503 in adapter.max_retries.status_forcelist


# ---------------------------------------------------------------------------
# Per-day error visibility (C6)
# ---------------------------------------------------------------------------


def test_errors_recorded_with_first_last_url_and_sidecar(tmp_path, monkeypatch):
    """When `n_errors > 0`, manifest carries first/last_error_url AND a sidecar.

    Sidecar `.failed_urls.txt` lives in the year dir and lists every
    failing URL line-separated. Lets operators triage with `curl -I`
    against the boundary URLs.
    """
    workdir = _make_project(tmp_path)
    # Every day returns 503 → 366 errors for 2020.
    _stub_session(monkeypatch, responder=lambda url: _FakeResponse(503))
    fetch_snodas(workdir=workdir, period="2020/2020")
    manifest = json.loads((workdir / "manifest.json").read_text())
    rec = manifest["sources"]["snodas"]["years"][0]
    assert rec["n_errors"] == 366
    assert rec["first_error_url"].endswith("SNODAS_20200101.tar")
    assert rec["last_error_url"].endswith("SNODAS_20201231.tar")

    sidecar = tmp_path / "datastore" / "snodas" / "raw" / "2020" / ".failed_urls.txt"
    lines = sidecar.read_text().strip().splitlines()
    assert len(lines) == 366
    assert lines[0].endswith("SNODAS_20200101.tar")
    assert lines[-1].endswith("SNODAS_20201231.tar")


def test_no_sidecar_when_no_errors(tmp_path, monkeypatch):
    """A clean year (no errors) does not write the diagnostic sidecar."""
    workdir = _make_project(tmp_path)
    _stub_session(monkeypatch)
    fetch_snodas(workdir=workdir, period="2020/2020")
    sidecar = tmp_path / "datastore" / "snodas" / "raw" / "2020" / ".failed_urls.txt"
    assert not sidecar.exists()
    manifest = json.loads((workdir / "manifest.json").read_text())
    rec = manifest["sources"]["snodas"]["years"][0]
    assert "first_error_url" not in rec
    assert "last_error_url" not in rec


def test_other_status_recorded_as_error(tmp_path, monkeypatch):
    """5xx / unexpected status surfaces as an `error`, not a crash."""
    workdir = _make_project(tmp_path)
    _stub_session(monkeypatch, responder=lambda url: _FakeResponse(503))
    result = fetch_snodas(workdir=workdir, period="2020/2020")
    rec = result["years"][0]
    assert rec["n_errors"] == 366
    assert rec["n_downloaded_this_run"] == 0


def test_partial_year_2003_boundary(tmp_path, monkeypatch):
    """SNODAS starts late September 2003; pre-Sep-30 URLs return 404.

    Regression guard: the year-boundary handling must not raise — partial
    coverage is part of normal SNODAS behaviour, and the manifest entry
    is expected to record `n_missing_404 > 0` for 2003 specifically.
    """
    workdir = _make_project(tmp_path)

    def responder(url: str) -> _FakeResponse:
        # Anything before 2003-09-30 → 404
        # Filename embeds the date as SNODAS_YYYYMMDD.tar
        date = url.rsplit("SNODAS_", 1)[-1].split(".tar")[0]
        if int(date) < 20030930:
            return _FakeResponse(404)
        return _FakeResponse(200, body=b"\x00")

    _stub_session(monkeypatch, responder=responder)
    result = fetch_snodas(workdir=workdir, period="2003/2003")
    rec = result["years"][0]
    assert rec["n_missing_404"] > 0
    assert rec["n_downloaded_this_run"] > 0
    assert rec["n_errors"] == 0


# ---------------------------------------------------------------------------
# Manifest accumulation + idempotency
# ---------------------------------------------------------------------------


def test_manifest_records_archive_url_and_metadata(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_session(monkeypatch)
    fetch_snodas(workdir=workdir, period="2020/2020")
    manifest = json.loads((workdir / "manifest.json").read_text())
    entry = manifest["sources"]["snodas"]
    assert entry["variables"] == ["swe"]
    assert entry["period"] == "2020/2020"
    assert entry["archive_url"].startswith(
        "https://noaadata.apps.nsidc.org/NOAA/G02158/"
    )


def test_manifest_accumulates_years_across_calls(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_session(monkeypatch)
    fetch_snodas(workdir=workdir, period="2020/2020")
    fetch_snodas(workdir=workdir, period="2021/2021")
    manifest = json.loads((workdir / "manifest.json").read_text())
    years = [r["year"] for r in manifest["sources"]["snodas"]["years"]]
    assert years == [2020, 2021]


def test_completed_years_skipped_on_rerun(tmp_path, monkeypatch):
    """Re-running the same period skips years already fully consolidated.

    The completion check now requires BOTH `n_granules > 0` AND a
    `daily_path` that exists on disk. Synthetic SWE tars make the
    consolidation path runnable end-to-end so the manifest carries
    `daily_path` after the first call.
    """
    workdir = _make_project(tmp_path)
    calls = _stub_session(monkeypatch, responder=_synthetic_tar_responder())
    fetch_snodas(workdir=workdir, period="2020/2020")
    urls_first = len(calls["urls"])
    # Manifest should now carry daily_path for 2020.
    manifest = json.loads((workdir / "manifest.json").read_text())
    rec = manifest["sources"]["snodas"]["years"][0]
    assert rec["daily_path"]
    assert Path(rec["daily_path"]).exists()

    fetch_snodas(workdir=workdir, period="2020/2020")
    # Second call issued zero new HTTP requests because daily_path exists.
    assert len(calls["urls"]) == urls_first


def test_raw_directory_created_under_datastore(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_session(monkeypatch)
    fetch_snodas(workdir=workdir, period="2020/2020")
    raw_dir = tmp_path / "datastore" / "snodas" / "raw" / "2020"
    assert raw_dir.exists()
    assert any(raw_dir.iterdir())


# ---------------------------------------------------------------------------
# Auth failure surface
# ---------------------------------------------------------------------------


def test_unauthenticated_login_raises(tmp_path, monkeypatch):
    """A failed earthaccess login surfaces cleanly, not as a NoneType crash."""
    workdir = _make_project(tmp_path)

    def fake_session():
        raise RuntimeError("Earthdata login failed")

    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.snodas._earthaccess_session", fake_session
    )
    with pytest.raises(RuntimeError, match="Earthdata login failed"):
        fetch_snodas(workdir=workdir, period="2020/2020")


# ---------------------------------------------------------------------------
# Decoder + consolidator (issue #109)
# ---------------------------------------------------------------------------


def test_decode_swe_tar_round_trip(tmp_path):
    """`_decode_snodas_swe_tar` reads back exactly what the synthetic helper writes."""
    from nhf_spatial_targets.fetch.snodas import _decode_snodas_swe_tar

    swe_in = np.array(
        [
            [0, 100, 200, -9999],
            [50, 150, 250, 32767],
            [25, 75, 125, 175],
            [-9999, 500, 1000, 2000],
        ],
        dtype=np.int16,
    )
    tar_bytes = _build_synthetic_snodas_tar("20200115", swe_int16=swe_in)
    tar_path = tmp_path / "SNODAS_20200115.tar"
    tar_path.write_bytes(tar_bytes)

    date, swe_out, header = _decode_snodas_swe_tar(tar_path)
    assert str(date.date()) == "2020-01-15"
    np.testing.assert_array_equal(swe_out, swe_in)
    assert header["Number of rows"] == "4"
    assert header["Number of columns"] == "4"


def test_decode_rejects_size_mismatch(tmp_path):
    """A SWE binary whose length disagrees with rows*cols*2 is rejected."""
    from nhf_spatial_targets.fetch.snodas import _decode_snodas_swe_tar

    # Build a tar by hand: claim 4x4 (32 bytes) in the header but ship 30 bytes.
    base = "us_ssmv11034tS__T0001TTNATS2020011505HP001"
    header = _synthetic_snodas_header(4, 4)
    hdr_gz = gzip.compress(header.encode("latin-1"))
    bin_gz = gzip.compress(b"\x00" * 30)  # wrong size
    pad = np.random.default_rng(0).bytes(1_200_000)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in (
            (f"{base}.txt.gz", hdr_gz),
            (f"{base}.dat.gz", bin_gz),
            ("padding.bin", pad),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    tar_path = tmp_path / "SNODAS_20200115.tar"
    tar_path.write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="SWE binary has 30 bytes"):
        _decode_snodas_swe_tar(tar_path)


def test_decode_rejects_missing_swe_member(tmp_path):
    """A tar without the 1034 SWE product raises ValueError."""
    from nhf_spatial_targets.fetch.snodas import _decode_snodas_swe_tar

    # Empty-ish tar (no 1034 member).
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo("us_ssmv11036tS__T0001TTNATS2020011505HP001.dat.gz")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"\x00"))
    tar_path = tmp_path / "SNODAS_20200115.tar"
    tar_path.write_bytes(buf.getvalue())
    # Header is read first in the dask-streaming path, so the missing-header
    # message can fire before the missing-binary one. Either is a valid
    # "no SWE (product code 1034) member" signal.
    with pytest.raises(
        ValueError, match=r"no SWE (product|header) \(code 1034\) member"
    ):
        _decode_snodas_swe_tar(tar_path)


def test_grids_match_tolerance():
    """Sub-µdeg text-representation noise compares equal; sub-pixel shift does not."""
    from nhf_spatial_targets.fetch.snodas import _grids_match

    h1 = {
        "Number of rows": "3351",
        "Number of columns": "6935",
        "Minimum x-axis coordinate": "-124.733333333328",
        "Maximum y-axis coordinate": "52.874999999997",
        "X-axis resolution": "0.008333333333000",
        "Y-axis resolution": "0.008333333333000",
    }
    # Same grid, lowest-bit text noise.
    h2 = dict(h1, **{"Minimum x-axis coordinate": "-124.733333333329"})
    assert _grids_match(h1, h2)
    # Real (sub-pixel but well above 1e-6) shift — the 2013/2014 regrid.
    h3 = dict(h1, **{"Minimum x-axis coordinate": "-124.7337500000000"})
    assert not _grids_match(h1, h3)


def test_coords_from_header_matches_expected_shape():
    """Lat is descending, lon is ascending, both at cell centers."""
    from nhf_spatial_targets.fetch.snodas import (
        _coords_from_snodas_header,
        _parse_snodas_header,
    )

    header = _parse_snodas_header(_synthetic_snodas_header(rows=4, cols=4))
    lat, lon = _coords_from_snodas_header(header)
    assert lat.shape == (4,)
    assert lon.shape == (4,)
    # Descending lat: lat[0] > lat[-1]
    assert lat[0] > lat[-1]
    # Ascending lon: lon[0] < lon[-1]
    assert lon[0] < lon[-1]
    # Cell-center offset: first cell center is dx/2 inside lon_min.
    expected_first_lon = _TINY_LON_MIN + 0.5 * _TINY_DX
    np.testing.assert_allclose(lon[0], expected_first_lon, rtol=0, atol=1e-12)


def test_consolidate_year_writes_cf_netcdf(tmp_path):
    """`consolidate_year_snodas` produces a CF NetCDF with the expected variable and metadata."""
    import xarray as xr

    from nhf_spatial_targets.fetch.snodas import consolidate_year_snodas

    raw_dir = tmp_path / "raw" / "2020"
    raw_dir.mkdir(parents=True)
    daily_dir = tmp_path / "daily"
    # Three synthetic days with distinct SWE values to detect ordering bugs.
    for day, value in zip(("20200101", "20200115", "20200131"), (10, 50, 90)):
        swe = np.full((4, 4), value, dtype=np.int16)
        (raw_dir / f"SNODAS_{day}.tar").write_bytes(
            _build_synthetic_snodas_tar(day, swe_int16=swe)
        )
    out_path = consolidate_year_snodas(2020, raw_dir, daily_dir)
    assert out_path == daily_dir / "snodas_daily_2020.nc"
    assert out_path.exists()

    with xr.open_dataset(out_path) as ds:
        assert ds["swe"].dims == ("time", "lat", "lon")
        assert ds["swe"].shape == (3, 4, 4)
        # Per-day SWE values preserved.
        day_means = ds["swe"].mean(("lat", "lon")).values
        np.testing.assert_array_equal(day_means, [10.0, 50.0, 90.0])
        # CF metadata from the catalog.
        assert ds["swe"].attrs["units"] == "kg m-2"
        assert ds["swe"].attrs["cell_methods"] == "time: point"
        assert ds["swe"].attrs["grid_mapping"] == "crs"
        # Time is sorted ascending.
        times = [str(t)[:10] for t in ds.time.values]
        assert times == ["2020-01-01", "2020-01-15", "2020-01-31"]
        # First-day header captured in global attrs for provenance.
        assert "snodas_first_day_header" in ds.attrs
        assert "Number of rows" in json.loads(ds.attrs["snodas_first_day_header"])


def test_daily_nc_is_cf_1_6_compliant(tmp_path):
    """Daily NCs carry the full CF-1.6 attribute set required by CLAUDE.md.

    Verifies the constraint "all NetCDFs the pipeline writes must be CF-1.6
    compliant" from CLAUDE.md "Data & Catalog Conventions". This is the
    light-weight in-test attribute audit; an external compliance-checker
    pass is a separate dev-tool concern.
    """
    import xarray as xr

    from nhf_spatial_targets.fetch.snodas import consolidate_year_snodas

    raw_dir = tmp_path / "raw" / "2020"
    raw_dir.mkdir(parents=True)
    daily_dir = tmp_path / "daily"
    for day in ("20200101", "20200102"):
        (raw_dir / f"SNODAS_{day}.tar").write_bytes(_build_synthetic_snodas_tar(day))
    out_path = consolidate_year_snodas(2020, raw_dir, daily_dir)

    with xr.open_dataset(out_path, decode_cf=False) as ds:
        # Global Conventions — CF section 2.6.1.
        assert ds.attrs.get("Conventions") == "CF-1.6"

        # Required ancillary CRS variable (CF section 5.6) with a
        # populated grid_mapping_name + crs_wkt.
        assert "crs" in ds.variables
        crs_attrs = ds["crs"].attrs
        assert crs_attrs.get("grid_mapping_name") == "latitude_longitude"
        assert "crs_wkt" in crs_attrs

        # Data variable attrs (CF section 3.1: units; 3.5: cell_methods;
        # 5.6: grid_mapping; 3.2: long_name).
        swe = ds["swe"]
        assert swe.attrs.get("units") == "kg m-2"
        assert swe.attrs.get("long_name") == "snow water equivalent"
        assert swe.attrs.get("cell_methods") == "time: point"
        assert swe.attrs.get("grid_mapping") == "crs"
        # int16 + _FillValue=-9999 (CF section 2.5.1 + 3.4). With
        # decode_cf=False the fill value lives in `attrs` as a CF
        # attribute; the on-disk dtype is reported via `encoding`.
        assert int(swe.attrs["_FillValue"]) == -9999
        assert swe.encoding.get("dtype") == "int16"
        assert str(swe.dtype) == "int16"

        # Latitude/longitude coords (CF section 4.1/4.2: standard_name +
        # units + axis are all required for proper CF detection).
        for coord, expected_units, expected_axis, expected_std in (
            ("lat", "degrees_north", "Y", "latitude"),
            ("lon", "degrees_east", "X", "longitude"),
        ):
            assert ds[coord].attrs.get("units") == expected_units
            assert ds[coord].attrs.get("axis") == expected_axis
            assert ds[coord].attrs.get("standard_name") == expected_std

        # Time coord (CF section 4.4: units required as "<interval> since <ref>";
        # calendar required for proleptic-Gregorian assumption).
        # `decode_cf=False` keeps the raw encoding attrs visible.
        time_attrs_or_encoding = {
            **ds["time"].attrs,
            **{
                k: v
                for k, v in ds["time"].encoding.items()
                if k in ("units", "calendar")
            },
        }
        assert "since" in time_attrs_or_encoding["units"]
        assert time_attrs_or_encoding.get("calendar") in (
            "standard",
            "proleptic_gregorian",
            "gregorian",
        )
        assert ds["time"].attrs.get("standard_name") == "time"
        assert ds["time"].attrs.get("axis") == "T"


def test_consolidate_year_idempotent_when_nc_newer(tmp_path):
    """If the daily NC is newer than every input .tar, the second call is a no-op."""
    from nhf_spatial_targets.fetch.snodas import consolidate_year_snodas

    raw_dir = tmp_path / "raw" / "2020"
    raw_dir.mkdir(parents=True)
    daily_dir = tmp_path / "daily"
    for day in ("20200101", "20200102"):
        (raw_dir / f"SNODAS_{day}.tar").write_bytes(_build_synthetic_snodas_tar(day))
    first = consolidate_year_snodas(2020, raw_dir, daily_dir)
    mtime_first = first.stat().st_mtime
    # No new inputs; mtime should NOT change on second call.
    consolidate_year_snodas(2020, raw_dir, daily_dir)
    assert first.stat().st_mtime == mtime_first


def test_consolidate_year_rebuilds_when_tar_newer(tmp_path):
    """If any input .tar is newer than the daily NC, the consolidator rebuilds."""
    import time

    from nhf_spatial_targets.fetch.snodas import consolidate_year_snodas

    raw_dir = tmp_path / "raw" / "2020"
    raw_dir.mkdir(parents=True)
    daily_dir = tmp_path / "daily"
    (raw_dir / "SNODAS_20200101.tar").write_bytes(
        _build_synthetic_snodas_tar("20200101", swe_int16=np.full((4, 4), 10, np.int16))
    )
    first = consolidate_year_snodas(2020, raw_dir, daily_dir)
    mtime_before = first.stat().st_mtime
    time.sleep(0.05)  # ensure new mtime is strictly greater (FS resolution ~1ms)
    # Write a NEW day with different values.
    new_tar = raw_dir / "SNODAS_20200102.tar"
    new_tar.write_bytes(
        _build_synthetic_snodas_tar("20200102", swe_int16=np.full((4, 4), 99, np.int16))
    )
    # Touch its mtime well past the NC mtime so the rebuild trigger fires.
    new_mtime = mtime_before + 60.0
    import os as _os

    _os.utime(new_tar, (new_mtime, new_mtime))
    second = consolidate_year_snodas(2020, raw_dir, daily_dir)
    assert second.stat().st_mtime > mtime_before
    import xarray as xr

    with xr.open_dataset(second) as ds:
        assert ds["swe"].sizes["time"] == 2


def test_consolidate_year_accepts_subpixel_grid_drift(tmp_path):
    """A sub-pixel mid-year origin drift is absorbed with provenance attrs.

    SNODAS has been re-georeferenced mid-year at least once (CY 2013
    shows ~4×10⁻⁴ ° origin drift between Jan and Oct). The
    consolidator accepts such drift up to half a pixel (~0.00417 °)
    and records the count + max magnitude as global attrs.
    """
    from nhf_spatial_targets.fetch.snodas import consolidate_year_snodas

    raw_dir = tmp_path / "raw" / "2020"
    raw_dir.mkdir(parents=True)
    daily_dir = tmp_path / "daily"
    (raw_dir / "SNODAS_20200101.tar").write_bytes(
        _build_synthetic_snodas_tar("20200101", rows=4, cols=4)
    )
    # Day 2 declares a lat_max ~1e-3 deg from day 1 — comparable to
    # the production CY 2013 drift (~4e-4 deg) and well within the
    # half-pixel drift tolerance (~4.17e-3 deg).
    (raw_dir / "SNODAS_20200102.tar").write_bytes(
        _build_synthetic_snodas_tar(
            "20200102",
            rows=4,
            cols=4,
            header_overrides={"Maximum y-axis coordinate": "52.875583333333000"},
        )
    )
    out = consolidate_year_snodas(2020, raw_dir, daily_dir)
    assert out.exists()
    ds = xr.open_dataset(out)
    assert ds.attrs["snodas_grid_drift_days_count"] == 1
    assert ds.attrs["snodas_grid_drift_max_deg"] > 0
    assert ds.attrs["snodas_grid_drift_max_deg"] <= 0.5 / 120
    assert "snodas_grid_drift_tol_deg" in ds.attrs


def test_consolidate_year_rejects_beyond_drift_tolerance(tmp_path):
    """Drift > half-pixel still raises rather than silently absorbing."""
    from nhf_spatial_targets.fetch.snodas import consolidate_year_snodas

    raw_dir = tmp_path / "raw" / "2020"
    raw_dir.mkdir(parents=True)
    daily_dir = tmp_path / "daily"
    (raw_dir / "SNODAS_20200101.tar").write_bytes(
        _build_synthetic_snodas_tar("20200101", rows=4, cols=4)
    )
    # Day 2 declares a lat_max ~0.1 deg from day 1 — well beyond the
    # half-pixel drift tolerance.
    (raw_dir / "SNODAS_20200102.tar").write_bytes(
        _build_synthetic_snodas_tar(
            "20200102",
            rows=4,
            cols=4,
            header_overrides={"Maximum y-axis coordinate": "53.000000000000000"},
        )
    )
    with pytest.raises(ValueError, match="within-year grid mismatch"):
        consolidate_year_snodas(2020, raw_dir, daily_dir)


def test_consolidate_year_rejects_row_count_change(tmp_path):
    """A row/col count change is a structural mismatch, never accepted as drift."""
    from nhf_spatial_targets.fetch.snodas import consolidate_year_snodas

    raw_dir = tmp_path / "raw" / "2020"
    raw_dir.mkdir(parents=True)
    daily_dir = tmp_path / "daily"
    (raw_dir / "SNODAS_20200101.tar").write_bytes(
        _build_synthetic_snodas_tar("20200101", rows=4, cols=4)
    )
    (raw_dir / "SNODAS_20200102.tar").write_bytes(
        _build_synthetic_snodas_tar("20200102", rows=5, cols=4)
    )
    with pytest.raises(ValueError, match="within-year grid mismatch"):
        consolidate_year_snodas(2020, raw_dir, daily_dir)


def test_consolidate_year_no_tars_raises(tmp_path):
    """An empty year dir raises FileNotFoundError with an actionable message."""
    from nhf_spatial_targets.fetch.snodas import consolidate_year_snodas

    raw_dir = tmp_path / "raw" / "2020"
    raw_dir.mkdir(parents=True)
    daily_dir = tmp_path / "daily"
    with pytest.raises(FileNotFoundError, match="No SNODAS_\\*\\.tar files"):
        consolidate_year_snodas(2020, raw_dir, daily_dir)


# ---------------------------------------------------------------------------
# Fetch + consolidate wiring (issue #109)
# ---------------------------------------------------------------------------


def test_fetch_runs_consolidation_after_download(tmp_path, monkeypatch):
    """`fetch_snodas` writes a daily NC and records `daily_path` in the manifest."""
    workdir = _make_project(tmp_path)
    _stub_session(monkeypatch, responder=_synthetic_tar_responder())
    fetch_snodas(workdir=workdir, period="2020/2020")
    daily_nc = tmp_path / "datastore" / "snodas" / "daily" / "snodas_daily_2020.nc"
    assert daily_nc.exists()
    manifest = json.loads((workdir / "manifest.json").read_text())
    rec = manifest["sources"]["snodas"]["years"][0]
    assert rec["daily_path"] == str(daily_nc)
    assert "consolidated_utc" in rec
    assert "consolidate_error" not in rec


def test_fetch_records_consolidation_error_but_does_not_abort(tmp_path, monkeypatch):
    """A consolidation failure surfaces as `consolidate_error`, not an exception.

    Non-tar bodies (this is the fast path used by older tests in this
    suite) make consolidation raise ValueError; the wiring catches that
    so the fetch run still completes with raw .tars preserved on disk.
    """
    workdir = _make_project(tmp_path)
    # Just-over-1-MiB junk body — passes the size floor without inflating
    # tmp_path I/O across 366 days of testing.
    junk = b"\x00" * (1024 * 1024 + 1)
    _stub_session(monkeypatch, responder=lambda url: _FakeResponse(200, body=junk))
    result = fetch_snodas(workdir=workdir, period="2020/2020")
    rec = result["years"][0]
    assert rec["n_granules"] == 366
    assert "daily_path" not in rec
    assert "consolidate_error" in rec
    # Raw .tars still on disk.
    raw_dir = tmp_path / "datastore" / "snodas" / "raw" / "2020"
    assert len(list(raw_dir.glob("SNODAS_*.tar"))) == 366


def test_backfill_consolidates_when_raw_present_no_daily_path(tmp_path, monkeypatch):
    """Pre-existing raw .tars + manifest without `daily_path` → consolidate-only rerun.

    This is the production backfill path. The already-downloaded
    2003–2024 corpus carries manifest entries with `n_granules > 0`
    but no `daily_path`; on the next run the new completion check
    sees that gap and re-enters the year. Per-file `_download_tar`
    finds each .tar already on disk and reports `already_present`,
    so no new HTTP GETs land for the bytes — only the consolidation
    step runs.
    """
    workdir = _make_project(tmp_path)
    raw_dir = tmp_path / "datastore" / "snodas" / "raw" / "2020"
    raw_dir.mkdir(parents=True)
    # Pre-stage real synthetic tars (2 days suffices for the assertion).
    for day in ("20200101", "20200102"):
        (raw_dir / f"SNODAS_{day}.tar").write_bytes(_build_synthetic_snodas_tar(day))
    # Manifest mimics the post-#108 state: granules counted, no daily_path.
    (workdir / "manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "snodas": {
                        "source_key": "snodas",
                        "period": "2020/2020",
                        "variables": ["swe"],
                        "years": [
                            {
                                "year": 2020,
                                "raw_dir": str(raw_dir),
                                "n_granules": 2,
                                "n_downloaded_this_run": 0,
                                "n_already_present": 2,
                                "n_missing_404": 0,
                                "n_errors": 0,
                            }
                        ],
                    }
                },
                "steps": [],
            }
        )
    )

    calls = _stub_session(monkeypatch, responder=_synthetic_tar_responder())
    fetch_snodas(workdir=workdir, period="2020/2020")
    # The fetch had to re-enter year 2020 because daily_path was missing.
    # But the .tars already on disk are size-floor-valid, so no new
    # successful downloads occurred — only `already_present` + consolidation.
    daily_nc = tmp_path / "datastore" / "snodas" / "daily" / "snodas_daily_2020.nc"
    assert daily_nc.exists()
    manifest = json.loads((workdir / "manifest.json").read_text())
    rec = manifest["sources"]["snodas"]["years"][0]
    assert rec["daily_path"] == str(daily_nc)
    # Per-day download attempts hit the `already_present` shortcut for the
    # 2 pre-staged days; the other 364 days of 2020 *will* issue HTTP
    # requests against the stubbed session (the responder serves them as
    # synthetic tars). The backfill semantics are about "no re-download
    # of bytes ALREADY on disk", not "zero HTTP for the whole year".
    # Assert the 2 pre-staged tars retained their original byte content.
    expected = _build_synthetic_snodas_tar("20200101")
    assert (raw_dir / "SNODAS_20200101.tar").read_bytes() == expected
    # And the call log shows our responder was invoked for the missing days.
    assert any(u.endswith("SNODAS_20200115.tar") for u in calls["urls"])


def test_year_with_zero_granules_skips_consolidation(tmp_path, monkeypatch):
    """If a year has 0 granules (all 404s), consolidation is skipped cleanly.

    No `daily_path` is recorded, but neither is `consolidate_error` —
    "no inputs" is normal at year boundaries, not a failure.
    """
    workdir = _make_project(tmp_path)
    _stub_session(monkeypatch, responder=lambda url: _FakeResponse(404))
    result = fetch_snodas(workdir=workdir, period="2020/2020")
    rec = result["years"][0]
    assert rec["n_granules"] == 0
    assert "daily_path" not in rec
    assert "consolidate_error" not in rec
    daily_dir = tmp_path / "datastore" / "snodas" / "daily"
    assert not list(daily_dir.glob("snodas_daily_*.nc"))
