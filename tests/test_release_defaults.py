"""Tests for nhf_spatial_targets.release.defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nhf_spatial_targets.release.defaults import (
    REQUIRED_POPULATED_PATHS,
    REQUIRED_SECTIONS,
    load_release_defaults,
    missing_release_defaults,
    require_populated_release_defaults,
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


# ---------------------------------------------------------------------------
# publish-time populated-check (separate from the shape-only loader)
# ---------------------------------------------------------------------------


def test_committed_defaults_are_fully_populated():
    """The committed catalog/release_defaults.yml passes the publish gate."""
    data = load_release_defaults()
    assert missing_release_defaults(data) == []
    require_populated_release_defaults(data)  # no raise


def test_empty_scaffold_reports_every_required_path():
    """A shape-valid but empty scaffold is loadable but not publishable."""
    empty = {s: {} for s in REQUIRED_SECTIONS}
    validate_release_defaults(empty)  # shape-only check passes
    missing = missing_release_defaults(empty)
    assert set(missing) == set(REQUIRED_POPULATED_PATHS)
    with pytest.raises(ValueError, match="not fully populated"):
        require_populated_release_defaults(empty)


def test_single_blank_leaf_is_reported(tmp_path):
    """A present-but-blank required field (folded YAML can leave whitespace)
    is reported by path, while the rest stay populated."""
    data = load_release_defaults()
    data["umbrella"] = dict(data["umbrella"])
    data["umbrella"]["title_template"] = "   \n  "  # whitespace only
    missing = missing_release_defaults(data)
    assert missing == ["umbrella.title_template"]
    with pytest.raises(ValueError, match="umbrella.title_template"):
        require_populated_release_defaults(data)
