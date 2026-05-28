"""Tests for nhf_spatial_targets.release.config."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from nhf_spatial_targets.release.config import (
    load_release_config,
    load_release_yml,
    scaffold_release_yml,
    validate_release_config,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "release" / "fixture_project_config.yml"


def test_load_release_config_from_fixture(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    shutil.copy(_FIXTURE, project / "config.yml")

    block = load_release_config(project)
    assert block["ipds_number"] == "IP-123456"
    assert block["fabric_label"] == "gfv2"
    assert [a["family"] for a in block["authors"]] == ["Doe", "Roe"]


def test_default_block_when_release_absent(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "config.yml").write_text(
        "datastore: /tmp/ds\nfabric:\n  path: /tmp/f.gpkg\n  id_col: nhm_id\n"
    )
    block = load_release_config(project)
    assert block["authors"] == []
    assert block["ipds_number"] is None


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_release_config(tmp_path / "nope")


def test_validate_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown key.*ipds_no"):
        validate_release_config({"ipds_no": "IP-1"})


def test_validate_rejects_non_list_authors():
    with pytest.raises(ValueError, match="authors must be a list"):
        validate_release_config({"authors": "Jane Doe"})


def test_validate_rejects_author_missing_family():
    with pytest.raises(ValueError, match=r"authors\[0\].family is required"):
        validate_release_config({"authors": [{"given": "Jane"}]})


def test_validate_rejects_unknown_author_key():
    with pytest.raises(ValueError, match=r"authors\[0\] has unknown key"):
        validate_release_config(
            {"authors": [{"given": "J", "family": "D", "middle": "Q"}]}
        )


def test_validate_rejects_non_string_scalar():
    with pytest.raises(ValueError, match="release.ipds_number must be a string"):
        validate_release_config({"ipds_number": 123})


def test_validate_accepts_empty_block():
    validate_release_config({})  # no raise


def test_scaffold_round_trips(tmp_path):
    block = {
        "authors": [{"given": "Jane", "family": "Doe", "orcid": "0000"}],
        "ipds_number": "IP-9",
        "doi": None,
        "abstract_notes": "hello",
        "fabric_label": "gfv2",
    }
    path = scaffold_release_yml(tmp_path / "release", block=block)
    assert path == tmp_path / "release" / "release.yml"
    loaded = load_release_yml(path)
    assert loaded == block


def test_scaffold_empty_default_round_trips(tmp_path):
    path = scaffold_release_yml(tmp_path / "release")
    loaded = load_release_yml(path)
    assert loaded == {
        "authors": [],
        "ipds_number": None,
        "doi": None,
        "abstract_notes": None,
        "fabric_label": None,
    }


def test_scaffold_does_not_overwrite(tmp_path):
    release_dir = tmp_path / "release"
    first = scaffold_release_yml(release_dir, block={"ipds_number": "IP-1"})
    original = first.read_text()
    # Second call with different content must not clobber the operator's file.
    scaffold_release_yml(release_dir, block={"ipds_number": "IP-2"})
    assert first.read_text() == original


def test_scaffold_overwrite_flag(tmp_path):
    release_dir = tmp_path / "release"
    scaffold_release_yml(release_dir, block={"ipds_number": "IP-1"})
    path = scaffold_release_yml(
        release_dir, block={"ipds_number": "IP-2"}, overwrite=True
    )
    assert load_release_yml(path)["ipds_number"] == "IP-2"


def test_scaffold_validates_block(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        scaffold_release_yml(tmp_path / "release", block={"bogus": 1})
