"""Tests for fetch.modis.reconcile_mod16a2 (issue #160)."""

from __future__ import annotations

from nhf_spatial_targets.fetch import modis
from nhf_spatial_targets.workspace import Project


def _project(tmp_path) -> Project:
    ds = tmp_path / "datastore"
    ds.mkdir()
    return Project(workdir=tmp_path, datastore=ds, config={}, fabric={}, dir_mode=None)


def _write_consolidated(project, year, content=b"nc"):
    root = project.raw_dir("mod16a2_v061")
    root.mkdir(parents=True, exist_ok=True)
    (root / f"mod16a2_v061_{year}_consolidated.nc").write_bytes(content)


def test_reconcile_empty_returns_empty(tmp_path):
    assert modis.reconcile_mod16a2(_project(tmp_path)) == []


def test_reconcile_returns_year_keyed_records(tmp_path):
    project = _project(tmp_path)
    _write_consolidated(project, 2018, content=b"abc")
    _write_consolidated(project, 2019, content=b"defgh")
    records = modis.reconcile_mod16a2(project)
    by_year = {r["year"]: r for r in records}
    assert set(by_year) == {2018, 2019}
    assert by_year[2018]["provenance"] == "reconciled"
    assert by_year[2018]["size_bytes"] == 3
    assert by_year[2018]["path"].endswith("mod16a2_v061_2018_consolidated.nc")
    assert by_year[2018]["downloaded_utc"].endswith("+00:00")
    assert "sha256" not in by_year[2018]


def test_reconcile_checksum_adds_sha256(tmp_path):
    project = _project(tmp_path)
    _write_consolidated(project, 2020)
    (rec,) = modis.reconcile_mod16a2(project, checksum=True)
    assert len(rec["sha256"]) == 64


def test_reconcile_ignores_unrelated_files(tmp_path):
    project = _project(tmp_path)
    root = project.raw_dir("mod16a2_v061")
    root.mkdir(parents=True, exist_ok=True)
    (root / "_tmp_2020_001.nc").write_bytes(b"junk")  # not a consolidated NC
    assert modis.reconcile_mod16a2(project) == []
