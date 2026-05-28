"""Golden-file tests for nhf_spatial_targets.release.readme.

Each README is rendered from the MCF dict built by the same canned fixtures
as test_release_mcf.py, then byte-compared against a committed golden
Markdown file. Regenerate after an intentional change with::

    REGEN_GOLDENS=1 pixi run -e dev test -k test_release_readme
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from nhf_spatial_targets.defaults import apply_defaults
from nhf_spatial_targets.release import mcf, readme
from nhf_spatial_targets.release.defaults import load_release_defaults

FX = Path(__file__).parent / "fixtures" / "release"
NOW = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)

SRC_FILES = {
    "era5_land": [
        "monthly/era5_land_monthly_2003.nc",
        "daily/era5_land_daily_2003.nc",
    ],
    "snodas": ["daily/snodas_2003.nc"],
}
FABRIC_FILES = [
    "fabric.gpkg",
    "aggregated/era5_land/era5_land_2003.nc",
    "aggregated/snodas/snodas_2003.nc",
    "targets/runoff_targets.nc",
    "targets/swe_targets.nc",
    "manifest.json",
]


@pytest.fixture
def defaults() -> dict:
    return load_release_defaults(FX / "fixture_release_defaults.yml")


@pytest.fixture
def config() -> dict:
    return apply_defaults(
        yaml.safe_load((FX / "fixture_project_config.yml").read_text())
    )


@pytest.fixture
def fabric() -> dict:
    return json.loads((FX / "fixture_fabric.json").read_text())


@pytest.fixture
def manifest() -> dict:
    return json.loads((FX / "fixture_manifest.json").read_text())


def _check_golden(text: str, name: str) -> None:
    path = FX / name
    if os.environ.get("REGEN_GOLDENS"):
        path.write_text(text)
    assert text == path.read_text()


def _source(key: str, defaults: dict, manifest: dict) -> dict:
    return mcf.build_source_mcf(
        key,
        defaults=defaults,
        manifest=manifest,
        distribution_files=SRC_FILES.get(key),
        now=NOW,
    )


def _fabric(config: dict, fabric: dict, defaults: dict, manifest: dict) -> dict:
    return mcf.build_fabric_mcf(
        config=config,
        fabric=fabric,
        defaults=defaults,
        manifest=manifest,
        distribution_files=FABRIC_FILES,
        now=NOW,
    )


def test_source_era5_readme_golden(defaults, manifest):
    text = readme.render_readme(_source("era5_land", defaults, manifest), kind="source")
    _check_golden(text, "expected_source_era5_readme.md")


def test_source_snodas_readme_golden(defaults, manifest):
    text = readme.render_readme(_source("snodas", defaults, manifest), kind="source")
    _check_golden(text, "expected_source_snodas_readme.md")


def test_fabric_readme_golden(config, fabric, defaults, manifest):
    text = readme.render_readme(
        _fabric(config, fabric, defaults, manifest), kind="fabric"
    )
    _check_golden(text, "expected_fabric_gfv2_readme.md")


def test_umbrella_readme_golden(config, fabric, defaults, manifest):
    children = [
        _source("era5_land", defaults, manifest),
        _fabric(config, fabric, defaults, manifest),
    ]
    umb = mcf.build_umbrella_mcf(
        defaults=defaults,
        children=children,
        authors=config["release"]["authors"],
        version="1.0",
        now=NOW,
    )
    text = readme.render_readme(umb, kind="umbrella")
    _check_golden(text, "expected_umbrella_readme.md")


def test_unknown_kind_raises(defaults, manifest):
    with pytest.raises(ValueError, match="Unknown README kind"):
        readme.render_readme(_source("era5_land", defaults, manifest), kind="bogus")


def test_metadata_only_source_readme_has_no_files(defaults, manifest):
    """daymet README falls to the no-data-files branch (metadata_only)."""
    daymet = mcf.build_source_mcf(
        "daymet", defaults=defaults, manifest=manifest, now=NOW
    )
    text = readme.render_readme(daymet, kind="source")
    assert "metadata-only item" in text


def test_umbrella_readme_empty_authors_fallback(defaults):
    """No authors -> the 'to be supplied' citation fallback branch renders."""
    umb = mcf.build_umbrella_mcf(defaults=defaults, children=[], authors=[], now=NOW)
    text = readme.render_readme(umb, kind="umbrella")
    assert "Authors to be supplied" in text


def test_source_readme_empty_citation_fallback():
    """A minimal MCF with no citation/files hits both source else-branches.

    The renderer is pure over the MCF dict, so a near-empty dict exercises
    the empty-citation and no-files fallbacks without needing a catalog
    source that happens to lack citations.
    """
    text = readme.render_readme({}, kind="source")
    assert "See the upstream archive" in text
    assert "metadata-only item" in text
