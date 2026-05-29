from __future__ import annotations

import json

import pytest

from nhf_spatial_targets.upgrade_manifest import check_manifest_schema


def _write(tmp_path, payload):
    (tmp_path / "manifest.json").write_text(json.dumps(payload))


def test_reports_behind_for_preversion(tmp_path):
    _write(tmp_path, {"sources": {}, "steps": []})  # no version key -> 0
    assert check_manifest_schema(tmp_path) == 0


def test_in_sync_returns_none(tmp_path):
    from nhf_spatial_targets.release.lineage import CURRENT_MANIFEST_SCHEMA_VERSION

    _write(
        tmp_path,
        {
            "manifest_schema_version": CURRENT_MANIFEST_SCHEMA_VERSION,
            "sources": {},
            "steps": [],
        },
    )
    assert check_manifest_schema(tmp_path) is None


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_manifest_schema(tmp_path)


def test_check_never_mutates(tmp_path):
    _write(tmp_path, {"sources": {}, "steps": []})
    p = tmp_path / "manifest.json"
    before = p.read_text()
    check_manifest_schema(tmp_path)
    assert p.read_text() == before  # report-only


def test_cli_exits_1_on_drift(tmp_path):
    from nhf_spatial_targets.cli.run import upgrade_manifest_cmd

    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    with pytest.raises(SystemExit) as e:
        upgrade_manifest_cmd(tmp_path)
    assert e.value.code == 1


def test_cli_exits_0_in_sync(tmp_path):
    from nhf_spatial_targets.cli.run import upgrade_manifest_cmd
    from nhf_spatial_targets.release.lineage import CURRENT_MANIFEST_SCHEMA_VERSION

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_schema_version": CURRENT_MANIFEST_SCHEMA_VERSION,
                "sources": {},
                "steps": [],
            }
        )
    )
    # No SystemExit on the in-sync path (returns normally / exit 0).
    upgrade_manifest_cmd(tmp_path)
