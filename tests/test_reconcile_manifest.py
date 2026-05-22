"""Tests for nhf_spatial_targets.reconcile (issue #160)."""

from __future__ import annotations

import json

from nhf_spatial_targets import reconcile
from nhf_spatial_targets.workspace import Project


def test_gap_fill_appends_only_new_year_records():
    existing = [{"year": 2020, "provenance": "fetch", "path": "a"}]
    new = [
        {"year": 2020, "provenance": "reconciled", "path": "a"},  # dup year
        {"year": 2021, "provenance": "reconciled", "path": "b"},  # new
    ]
    merged, added = reconcile._gap_fill(existing, new)
    # 2020 fetch record is untouched; only 2021 is appended.
    assert merged == [
        {"year": 2020, "provenance": "fetch", "path": "a"},
        {"year": 2021, "provenance": "reconciled", "path": "b"},
    ]
    assert added == [{"year": 2021, "provenance": "reconciled", "path": "b"}]


def test_gap_fill_keys_on_path_when_no_year():
    existing = [{"path": "x"}]
    new = [{"path": "x"}, {"path": "y"}]
    merged, added = reconcile._gap_fill(existing, new)
    assert added == [{"path": "y"}]
    assert len(merged) == 2


def test_gap_fill_tolerates_malformed_existing_records():
    existing = [{"note": "no id here"}]  # neither year nor path
    new = [{"year": 1999, "path": "z"}]
    merged, added = reconcile._gap_fill(existing, new)
    assert added == [{"year": 1999, "path": "z"}]
    assert merged == [{"note": "no id here"}, {"year": 1999, "path": "z"}]


def _make_project(tmp_path) -> Project:
    """A Project pointing workdir/datastore at tmp dirs; config minimal.

    reconcile only touches project.raw_dir(), project.manifest_path, and
    project.datastore, so config/fabric can be empty.
    """
    datastore = tmp_path / "datastore"
    datastore.mkdir()
    return Project(
        workdir=tmp_path,
        datastore=datastore,
        config={},
        fabric={},
        dir_mode=None,
    )


def test_apply_records_creates_minimal_entry_when_absent(tmp_path):
    project = _make_project(tmp_path)
    records = [
        {"year": 2020, "path": "p2020", "provenance": "reconciled"},
        {"year": 2021, "path": "p2021", "provenance": "reconciled"},
    ]
    result = reconcile._apply_records(project, "mod16a2_v061", records, dry_run=False)
    assert result.added == 2
    assert result.status == "reconciled"

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["sources"]["mod16a2_v061"]
    assert entry["source_key"] == "mod16a2_v061"
    assert entry["period"] == "2020/2021"  # derived from year span
    assert "reconciled_utc" in entry
    assert len(entry["files"]) == 2


def test_apply_records_gap_fills_without_touching_existing(tmp_path):
    project = _make_project(tmp_path)
    # Pre-existing manifest with a true fetch record for 2020.
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "mod16a2_v061": {
                        "source_key": "mod16a2_v061",
                        "period": "2020/2020",
                        "files": [
                            {"year": 2020, "path": "real", "downloaded_utc": "T"}
                        ],
                    }
                },
                "steps": [],
            }
        )
    )
    records = [
        {"year": 2020, "path": "ondisk", "provenance": "reconciled"},
        {"year": 2021, "path": "ondisk21", "provenance": "reconciled"},
    ]
    result = reconcile._apply_records(project, "mod16a2_v061", records, dry_run=False)
    assert result.added == 1
    assert result.already_recorded == 1

    entry = json.loads((tmp_path / "manifest.json").read_text())["sources"][
        "mod16a2_v061"
    ]
    files_by_year = {f["year"]: f for f in entry["files"]}
    # 2020 fetch record is byte-for-byte preserved (no provenance key added).
    assert files_by_year[2020] == {"year": 2020, "path": "real", "downloaded_utc": "T"}
    assert files_by_year[2021]["provenance"] == "reconciled"
    # Entry-level metadata of the pre-existing source is left alone.
    assert "reconciled_utc" not in entry


def test_apply_records_dry_run_does_not_write(tmp_path):
    project = _make_project(tmp_path)
    records = [{"year": 2020, "path": "p", "provenance": "reconciled"}]
    result = reconcile._apply_records(project, "mod16a2_v061", records, dry_run=True)
    assert result.added == 1
    assert not (tmp_path / "manifest.json").exists()  # nothing written


def test_apply_records_noop_when_all_present_does_not_rewrite(tmp_path):
    project = _make_project(tmp_path)
    records = [{"year": 2020, "path": "p", "provenance": "reconciled"}]
    reconcile._apply_records(project, "mod16a2_v061", records, dry_run=False)
    before = (tmp_path / "manifest.json").read_text()
    # Re-applying the same records adds nothing and must not rewrite the file.
    result = reconcile._apply_records(project, "mod16a2_v061", records, dry_run=False)
    assert result.status == "no-op"
    assert result.added == 0
    assert (tmp_path / "manifest.json").read_text() == before


def test_reconcile_manifest_dispatches_registered_hooks(tmp_path, monkeypatch):
    project = _make_project(tmp_path)

    def fake_hook(proj, *, checksum=False):
        return [{"year": 2020, "path": "p", "provenance": "reconciled"}]

    # Register a fake hook under a real catalog key.
    monkeypatch.setattr(
        reconcile,
        "_RECONCILERS",
        {"mod16a2_v061": "tests.test_reconcile_manifest:_HOOK"},
    )
    monkeypatch.setattr(
        reconcile, "_call_hook", lambda spec, proj, *, checksum: fake_hook(proj)
    )

    results = reconcile.reconcile_manifest(
        project, sources=["mod16a2_v061"], dry_run=False
    )
    assert len(results) == 1
    assert results[0].status == "reconciled"
    assert results[0].added == 1


def test_reconcile_manifest_reports_no_hook_for_unregistered_source(
    tmp_path, monkeypatch
):
    project = _make_project(tmp_path)
    monkeypatch.setattr(reconcile, "_RECONCILERS", {})
    results = reconcile.reconcile_manifest(
        project, sources=["era5_land"], dry_run=False
    )
    assert results[0].status == "no-hook"


def test_reconcile_manifest_reports_empty_when_hook_returns_nothing(
    tmp_path, monkeypatch
):
    project = _make_project(tmp_path)
    monkeypatch.setattr(reconcile, "_RECONCILERS", {"era5_land": "x:y"})
    monkeypatch.setattr(reconcile, "_call_hook", lambda spec, proj, *, checksum: [])
    results = reconcile.reconcile_manifest(
        project, sources=["era5_land"], dry_run=False
    )
    assert results[0].status == "empty"
    assert not (tmp_path / "manifest.json").exists()


# --- end-to-end through the real era5_land hook -------------------------


def _seed_era5(project, *years):
    root = project.raw_dir("era5_land")
    (root / "daily").mkdir(parents=True, exist_ok=True)
    (root / "monthly").mkdir(parents=True, exist_ok=True)
    for y in years:
        (root / "daily" / f"era5_land_daily_{y}.nc").write_bytes(b"d")
        (root / "monthly" / f"era5_land_monthly_{y}.nc").write_bytes(b"m")


def test_end_to_end_empty_datastore_is_noop(tmp_path):
    project = _make_project(tmp_path)
    results = reconcile.reconcile_manifest(project, sources=["era5_land"])
    assert results[0].status == "empty"
    assert not (tmp_path / "manifest.json").exists()


def test_end_to_end_full_backfill(tmp_path):
    project = _make_project(tmp_path)
    _seed_era5(project, 2019, 2020, 2021)
    results = reconcile.reconcile_manifest(project, sources=["era5_land"])
    assert results[0].added == 3

    entry = json.loads((tmp_path / "manifest.json").read_text())["sources"]["era5_land"]
    assert {f["year"] for f in entry["files"]} == {2019, 2020, 2021}
    assert all(f["provenance"] == "reconciled" for f in entry["files"])
    assert entry["period"] == "2019/2021"


def test_end_to_end_gap_fill_is_idempotent_and_preserves_fetch(tmp_path):
    project = _make_project(tmp_path)
    _seed_era5(project, 2019, 2020, 2021)
    # Pre-existing manifest with a true fetch record for 2020.
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "era5_land": {
                        "source_key": "era5_land",
                        "period": "2020/2020",
                        "files": [
                            {
                                "year": 2020,
                                "daily_path": "real_d",
                                "monthly_path": "real_m",
                                "consolidated_utc": "T",
                            }
                        ],
                    }
                },
                "steps": [],
            }
        )
    )
    first = reconcile.reconcile_manifest(project, sources=["era5_land"])
    assert first[0].added == 2  # 2019, 2021
    manifest_after_first = (tmp_path / "manifest.json").read_text()

    # Idempotent: a second run adds nothing and leaves the file identical.
    second = reconcile.reconcile_manifest(project, sources=["era5_land"])
    assert second[0].added == 0
    assert second[0].status == "no-op"
    assert (tmp_path / "manifest.json").read_text() == manifest_after_first

    files = json.loads(manifest_after_first)["sources"]["era5_land"]["files"]
    rec_2020 = next(f for f in files if f["year"] == 2020)
    assert rec_2020 == {
        "year": 2020,
        "daily_path": "real_d",
        "monthly_path": "real_m",
        "consolidated_utc": "T",
    }  # untouched fetch record, no provenance key


def test_end_to_end_dry_run_reports_without_writing(tmp_path):
    project = _make_project(tmp_path)
    _seed_era5(project, 2019, 2020)
    results = reconcile.reconcile_manifest(project, sources=["era5_land"], dry_run=True)
    assert results[0].added == 2
    assert not (tmp_path / "manifest.json").exists()


# --- CLI wiring ---------------------------------------------------------


def test_cli_command_is_registered():
    """The reconcile-manifest command is wired into the cyclopts app."""
    from nhf_spatial_targets import cli

    assert "reconcile-manifest" in cli.app
