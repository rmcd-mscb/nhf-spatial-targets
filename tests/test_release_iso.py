"""Golden-file tests for nhf_spatial_targets.release.iso.

ISO 19139 is emitted by pygeometa from the same MCF dict that feeds FGDC and
README. The render is deterministic for a pinned pygeometa version (every date
is baked into the MCF), so a byte-equal golden catches both MCF-shape changes
and pygeometa upgrades. Regenerate after an intentional change with::

    REGEN_GOLDENS=1 pixi run -e dev test -k test_release_iso

If a pygeometa version bump legitimately changes the output, regenerate and
review the diff. The per-source goldens (era5_land, snodas) are representative
examples; adding a new catalog source must not require touching them.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from lxml import etree

from nhf_spatial_targets.defaults import apply_defaults
from nhf_spatial_targets.release import iso, mcf
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


def _source_mcf(key: str, defaults: dict, manifest: dict) -> dict:
    return mcf.build_source_mcf(
        key,
        defaults=defaults,
        manifest=manifest,
        distribution_files=SRC_FILES.get(key),
        now=NOW,
    )


def _fabric_mcf(config: dict, fabric: dict, defaults: dict, manifest: dict) -> dict:
    return mcf.build_fabric_mcf(
        config=config,
        fabric=fabric,
        defaults=defaults,
        manifest=manifest,
        distribution_files=FABRIC_FILES,
        now=NOW,
    )


def _umbrella_mcf(config: dict, fabric: dict, defaults: dict, manifest: dict) -> dict:
    children = [
        _source_mcf("era5_land", defaults, manifest),
        _source_mcf("snodas", defaults, manifest),
        _fabric_mcf(config, fabric, defaults, manifest),
    ]
    return mcf.build_umbrella_mcf(
        defaults=defaults,
        children=children,
        authors=config["release"]["authors"],
        version="1.0",
        now=NOW,
    )


# ---------------------------------------------------------------------------
# Golden-file tests
# ---------------------------------------------------------------------------


def test_source_era5_iso_golden(defaults, manifest):
    _check_golden(
        iso.render_iso(_source_mcf("era5_land", defaults, manifest)),
        "expected_source_era5_iso.xml",
    )


def test_source_snodas_iso_golden(defaults, manifest):
    _check_golden(
        iso.render_iso(_source_mcf("snodas", defaults, manifest)),
        "expected_source_snodas_iso.xml",
    )


def test_fabric_iso_golden(config, fabric, defaults, manifest):
    _check_golden(
        iso.render_iso(_fabric_mcf(config, fabric, defaults, manifest)),
        "expected_fabric_gfv2_iso.xml",
    )


def test_umbrella_iso_golden(config, fabric, defaults, manifest):
    _check_golden(
        iso.render_iso(_umbrella_mcf(config, fabric, defaults, manifest)),
        "expected_umbrella_iso.xml",
    )


# ---------------------------------------------------------------------------
# Invariant spot-checks
# ---------------------------------------------------------------------------


def test_iso_is_wellformed_and_deterministic(defaults, manifest):
    m = _source_mcf("era5_land", defaults, manifest)
    xml1 = iso.render_iso(m)
    xml2 = iso.render_iso(m)
    assert xml1 == xml2
    root = etree.fromstring(xml1.encode("utf-8"))
    assert root.tag.endswith("MD_Metadata")
