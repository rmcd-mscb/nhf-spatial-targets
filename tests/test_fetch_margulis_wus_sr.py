"""Tests for Margulis Western US Snow Reanalysis (NSIDC-0719) fetch.

`fetch_margulis_wus_sr` does an earthaccess search and download per year,
clipped to the project's fabric bbox. Consolidation into CF NetCDFs is
deferred to the Margulis aggregate follow-up issue and is not exercised
here. These tests cover the period gate, the fabric-bbox plumbing, the
empty-search fallback, the zero-byte-drop / partial-download guards, the
fabric_scope manifest recording, and Earthdata auth-failure propagation.

All tests monkeypatch ``earthaccess.login``, ``search_data``, and
``download`` — no network access is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from nhf_spatial_targets.fetch.margulis_wus_sr import fetch_margulis_wus_sr


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
) -> dict[str, list]:
    """Patch earthaccess login/search/download; return a call log dict."""
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
