"""Tests for nhf_spatial_targets.release.defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nhf_spatial_targets.release.defaults import (
    REQUIRED_SECTIONS,
    load_release_defaults,
    validate_release_defaults,
)

_FIXTURE = (
    Path(__file__).parent / "fixtures" / "release" / "fixture_release_defaults.yml"
)


def test_committed_catalog_release_defaults_load():
    """The committed catalog/release_defaults.yml is well-shaped."""
    data = load_release_defaults()
    assert set(data) == set(REQUIRED_SECTIONS)
    # Every section is normalized to a mapping (the scaffold ships empty).
    assert all(isinstance(v, dict) for v in data.values())


def test_load_fixture_returns_all_sections():
    data = load_release_defaults(_FIXTURE)
    assert set(data) == set(REQUIRED_SECTIONS)
    assert data["contacts"]["distributor"]["email"] == "sciencebase@usgs.gov"
    assert data["spatial_reference"]["epsg"] == 4326


def test_bare_none_section_normalized_to_empty_mapping(tmp_path):
    doc = {section: {} for section in REQUIRED_SECTIONS}
    doc["umbrella"] = None  # bare key in YAML parses to None
    path = tmp_path / "rd.yml"
    path.write_text(yaml.safe_dump(doc))
    data = load_release_defaults(path)
    assert data["umbrella"] == {}


def test_missing_section_fails_clearly():
    doc = {s: {} for s in REQUIRED_SECTIONS if s != "umbrella"}
    with pytest.raises(ValueError, match="missing required section.*umbrella"):
        validate_release_defaults(doc)


def test_unknown_section_fails():
    doc = {s: {} for s in REQUIRED_SECTIONS}
    doc["extra"] = {}
    with pytest.raises(ValueError, match="unknown top-level section.*extra"):
        validate_release_defaults(doc)


def test_non_mapping_section_fails():
    doc = {s: {} for s in REQUIRED_SECTIONS}
    doc["keywords"] = ["not", "a", "mapping"]
    with pytest.raises(ValueError, match="section 'keywords' must be a mapping"):
        validate_release_defaults(doc)


def test_non_mapping_top_level_fails():
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        validate_release_defaults(["a", "b"])


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_release_defaults(tmp_path / "does-not-exist.yml")
