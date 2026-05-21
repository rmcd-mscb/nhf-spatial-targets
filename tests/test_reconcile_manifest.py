"""Tests for nhf_spatial_targets.reconcile (issue #160)."""

from __future__ import annotations

from nhf_spatial_targets import reconcile


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
