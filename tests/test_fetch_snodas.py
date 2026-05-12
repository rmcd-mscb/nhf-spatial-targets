"""Tests for SNODAS daily fetch via earthaccess.

`fetch_snodas` does an earthaccess search and download per year. The
consolidation step (raw tar/binary → CF NetCDF) is **deferred** to the
SNODAS aggregate follow-up issue and is not exercised here. These tests
cover the period gate, worker sharding, partial-download guard, the
empty-search fallback, and the per-year manifest accumulation.

All tests monkeypatch ``earthaccess.login``, ``search_data``, and
``download`` — no network access is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from nhf_spatial_targets.fetch import snodas as _snodas_module
from nhf_spatial_targets.fetch.snodas import fetch_snodas


def _make_project(tmp_path: Path) -> Path:
    """Materialize a minimal valid project directory."""
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


def _stub_earthaccess(
    monkeypatch: pytest.MonkeyPatch,
    *,
    granules_per_year: int = 2,
    files_returned: int | None = None,
) -> dict[str, list]:
    """Patch earthaccess login/search/download. Returns a call log dict."""
    import earthaccess

    calls = {"search": [], "downloaded_dirs": []}

    def fake_login(strategy="netrc"):
        calls.setdefault("login", []).append(strategy)
        return None

    def fake_search(**kwargs):
        calls["search"].append(kwargs)
        return [object()] * granules_per_year

    def fake_download(results, dest):
        calls["downloaded_dirs"].append(str(dest))
        n = len(results) if files_returned is None else files_returned
        # Write tiny placeholder files so the dest dir is non-empty.
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(n):
            p = dest_path / f"snodas_stub_{i}.dat"
            p.write_bytes(b"\x00")
            paths.append(str(p))
        return paths

    monkeypatch.setattr(earthaccess, "login", fake_login)
    monkeypatch.setattr(earthaccess, "search_data", fake_search)
    monkeypatch.setattr(earthaccess, "download", fake_download)
    return calls


# ---------------------------------------------------------------------------
# Period and worker gates
# ---------------------------------------------------------------------------


def test_period_before_2003_rejected(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch)
    with pytest.raises(ValueError, match="before the SNODAS publisher start"):
        fetch_snodas(workdir=workdir, period="2000/2002")


def test_worker_partitioning_3_years_3_workers(tmp_path, monkeypatch):
    """Three workers across three years each take one distinct year."""
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch, granules_per_year=1)
    seen_years: list[int] = []
    for wi in range(3):
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
    _stub_earthaccess(monkeypatch, granules_per_year=1)
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
    _stub_earthaccess(monkeypatch, granules_per_year=1)
    with pytest.raises(ValueError, match="worker_index must be in"):
        fetch_snodas(
            workdir=workdir,
            period="2020/2020",
            worker_index=3,
            n_workers=2,
        )


# ---------------------------------------------------------------------------
# Search and download paths
# ---------------------------------------------------------------------------


def test_empty_search_records_no_granules_note(tmp_path, monkeypatch):
    """Years with zero CMR hits are recorded with a note, not raised on."""
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch, granules_per_year=0)
    result = fetch_snodas(workdir=workdir, period="2020/2020")
    assert len(result["years"]) == 1
    assert result["years"][0]["n_granules"] == 0
    assert result["years"][0]["note"] == "no_granules_in_CMR"


def test_partial_download_raises(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch, granules_per_year=3, files_returned=2)
    with pytest.raises(RuntimeError, match="partial download"):
        fetch_snodas(workdir=workdir, period="2020/2020")


def test_zero_download_raises(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch, granules_per_year=2, files_returned=0)
    with pytest.raises(RuntimeError, match="returned no files"):
        fetch_snodas(workdir=workdir, period="2020/2020")


# ---------------------------------------------------------------------------
# Manifest accumulation
# ---------------------------------------------------------------------------


def test_manifest_accumulates_years_across_calls(tmp_path, monkeypatch):
    """Two successive single-year fetches yield both years in manifest."""
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch, granules_per_year=1)
    fetch_snodas(workdir=workdir, period="2020/2020")
    fetch_snodas(workdir=workdir, period="2021/2021")
    manifest = json.loads((workdir / "manifest.json").read_text())
    years = [r["year"] for r in manifest["sources"]["snodas"]["years"]]
    assert years == [2020, 2021]


def test_manifest_records_bbox_and_metadata(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch, granules_per_year=1)
    fetch_snodas(workdir=workdir, period="2020/2020")
    manifest = json.loads((workdir / "manifest.json").read_text())
    entry = manifest["sources"]["snodas"]
    assert entry["bbox"] == _snodas_module.BBOX_NWSE
    assert entry["variables"] == ["swe"]
    assert entry["period"] == "2020/2020"


def test_raw_directory_created_under_datastore(tmp_path, monkeypatch):
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch, granules_per_year=1)
    fetch_snodas(workdir=workdir, period="2020/2020")
    raw_dir = tmp_path / "datastore" / "snodas" / "raw" / "2020"
    assert raw_dir.exists()
    assert any(raw_dir.iterdir())  # at least one stub granule landed here
