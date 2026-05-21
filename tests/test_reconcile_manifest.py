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
