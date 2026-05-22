"""Tests for fetch.era5_land.reconcile (issue #160)."""

from __future__ import annotations

from nhf_spatial_targets.fetch import era5_land
from nhf_spatial_targets.workspace import Project


def _project(tmp_path) -> Project:
    ds = tmp_path / "datastore"
    ds.mkdir()
    return Project(workdir=tmp_path, datastore=ds, config={}, fabric={}, dir_mode=None)


def _write_year(project, year, *, daily=True, monthly=True):
    root = project.raw_dir("era5_land")
    if daily:
        d = root / "daily"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"era5_land_daily_{year}.nc").write_bytes(b"daily")
    if monthly:
        m = root / "monthly"
        m.mkdir(parents=True, exist_ok=True)
        (m / f"era5_land_monthly_{year}.nc").write_bytes(b"monthly")


def test_reconcile_empty_datastore_returns_empty(tmp_path):
    project = _project(tmp_path)
    assert era5_land.reconcile(project) == []


def test_reconcile_returns_one_record_per_complete_year(tmp_path):
    project = _project(tmp_path)
    _write_year(project, 2019)
    _write_year(project, 2020)
    records = era5_land.reconcile(project)
    years = sorted(r["year"] for r in records)
    assert years == [2019, 2020]
    r = records[0]
    assert r["provenance"] == "reconciled"
    assert r["daily_path"].endswith("era5_land_daily_2019.nc")
    assert r["monthly_path"].endswith("era5_land_monthly_2019.nc")
    # consolidated_utc is an ISO-8601 UTC string derived from mtime.
    assert r["consolidated_utc"].endswith("+00:00")
    assert "sha256_daily" not in r  # checksum off by default


def test_reconcile_skips_year_missing_its_monthly_pair(tmp_path):
    project = _project(tmp_path)
    _write_year(project, 2021, daily=True, monthly=False)
    assert era5_land.reconcile(project) == []


def test_reconcile_checksum_adds_hashes(tmp_path):
    project = _project(tmp_path)
    _write_year(project, 2022)
    (records,) = era5_land.reconcile(project, checksum=True)
    assert "sha256_daily" in records and "sha256_monthly" in records
    assert len(records["sha256_daily"]) == 64
