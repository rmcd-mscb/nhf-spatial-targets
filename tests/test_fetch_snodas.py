"""Tests for SNODAS daily fetch via NSIDC's HTTPS archive (issue #107).

The original implementation used ``earthaccess.search_data`` against a
CMR collection that turned out to be a metadata-only stub with zero
granule records; the rewrite constructs URLs directly from dates and
streams each ``.tar`` via ``earthaccess.login().get_session()``. These
tests cover the period gate, worker sharding, per-day status accounting
(downloaded / already-present / 404 / error), manifest accumulation,
and the EDL auth-failure surface. No network is touched — the HTTPS
session is stubbed via ``_earthaccess_session`` monkeypatch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest
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


class _FakeResponse:
    """Tiny stand-in for ``requests.Response`` covering the bits we use."""

    def __init__(
        self, status_code: int, body: bytes = b"", chunks: list[bytes] | None = None
    ):
        self.status_code = status_code
        self._body = body
        self._chunks = chunks if chunks is not None else [body]

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
        # First 30 days of the year are missing (mimics 2003 partial year).
        for d in range(1, 31):
            if f"SNODAS_20200{d:02d}".replace("SNODAS_2020001", "X") and url.endswith(
                f"SNODAS_202001{d:02d}.tar"
            ):
                return _FakeResponse(404)
        return _FakeResponse(200, body=b"\x00")

    _stub_session(monkeypatch, responder=responder)
    result = fetch_snodas(workdir=workdir, period="2020/2020")
    rec = result["years"][0]
    assert rec["n_missing_404"] == 30
    assert rec["n_downloaded_this_run"] == 366 - 30
    assert rec["n_errors"] == 0


def test_already_present_files_are_skipped(tmp_path, monkeypatch):
    """Pre-existing non-empty .tar files are accounted as ``already_present``."""
    workdir = _make_project(tmp_path)
    year_dir = tmp_path / "datastore" / "snodas" / "raw" / "2020"
    year_dir.mkdir(parents=True)
    # Pre-stage 5 .tars on disk.
    for day in range(1, 6):
        (year_dir / f"SNODAS_202001{day:02d}.tar").write_bytes(b"PRE_STAGED")
    _stub_session(monkeypatch)
    result = fetch_snodas(workdir=workdir, period="2020/2020")
    rec = result["years"][0]
    assert rec["n_already_present"] == 5
    assert rec["n_downloaded_this_run"] == 366 - 5
    assert rec["n_granules"] == 366
    # Pre-staged files were NOT overwritten.
    assert (year_dir / "SNODAS_20200101.tar").read_bytes() == b"PRE_STAGED"


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
    """Re-running the same period skips years already recorded as complete."""
    workdir = _make_project(tmp_path)
    calls = _stub_session(monkeypatch)
    fetch_snodas(workdir=workdir, period="2020/2020")
    urls_first = len(calls["urls"])
    fetch_snodas(workdir=workdir, period="2020/2020")
    # Second call issued zero new HTTP requests.
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
