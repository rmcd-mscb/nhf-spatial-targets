"""Golden-file tests for nhf_spatial_targets.release.fgdc.

Each item type is built from the same canned fixtures as test_release_mcf.py
(with an injected timestamp), rendered to FGDC CSDGM 2.0 XML, then byte-equal
compared against a committed golden. Regenerate after an intentional change
with::

    REGEN_GOLDENS=1 pixi run -e dev test -k test_release_fgdc

The per-source XML goldens (era5_land, snodas) are representative examples,
not an exhaustive registry: adding a new catalog source must not require
touching them. Spot-check tests assert the load-bearing CSDGM invariants
(strict <procstep> element order, srcused/srcprod -> <srcinfo> referential
integrity, bbox -> bounding mapping) so intent is visible without diffing the
large golden XML.
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
from nhf_spatial_targets.release import fgdc, mcf
from nhf_spatial_targets.release.defaults import load_release_defaults

FX = Path(__file__).parent / "fixtures" / "release"

# Injected so metadata.datestamp / dates are byte-stable in the golden.
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


def test_source_era5_fgdc_golden(defaults, manifest):
    xml = fgdc.render_fgdc(_source_mcf("era5_land", defaults, manifest), kind="source")
    _check_golden(xml, "expected_source_era5_fgdc.xml")


def test_source_snodas_fgdc_golden(defaults, manifest):
    xml = fgdc.render_fgdc(_source_mcf("snodas", defaults, manifest), kind="source")
    _check_golden(xml, "expected_source_snodas_fgdc.xml")


def test_fabric_fgdc_golden(config, fabric, defaults, manifest):
    xml = fgdc.render_fgdc(
        _fabric_mcf(config, fabric, defaults, manifest), kind="fabric"
    )
    _check_golden(xml, "expected_fabric_gfv2_fgdc.xml")


def test_umbrella_fgdc_golden(config, fabric, defaults, manifest):
    xml = fgdc.render_fgdc(
        _umbrella_mcf(config, fabric, defaults, manifest), kind="umbrella"
    )
    _check_golden(xml, "expected_umbrella_fgdc.xml")


# ---------------------------------------------------------------------------
# Invariant spot-checks (intent visible without diffing the big goldens)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["source", "umbrella", "fabric"])
def test_renders_wellformed_xml(kind, config, fabric, defaults, manifest):
    if kind == "fabric":
        m = _fabric_mcf(config, fabric, defaults, manifest)
    elif kind == "umbrella":
        m = _umbrella_mcf(config, fabric, defaults, manifest)
    else:
        m = _source_mcf("era5_land", defaults, manifest)
    root = etree.fromstring(fgdc.render_fgdc(m, kind=kind).encode("utf-8"))
    assert root.tag == "metadata"


def test_unknown_kind_raises(defaults, manifest):
    with pytest.raises(ValueError, match="Unknown FGDC kind"):
        fgdc.render_fgdc(_source_mcf("era5_land", defaults, manifest), kind="bogus")


def test_procstep_element_order_is_csdgm(defaults, manifest):
    """CSDGM <procstep> order: procdesc -> srcused* -> procdate -> proctime ->
    srcprod*. mp rejects any other order."""
    root = etree.fromstring(
        fgdc.render_fgdc(
            _source_mcf("era5_land", defaults, manifest), kind="source"
        ).encode("utf-8")
    )
    steps = root.findall(".//procstep")
    assert steps
    for step in steps:
        tags = [child.tag for child in step]
        assert tags[0] == "procdesc"
        procdate_i = tags.index("procdate")
        assert all(tags.index(t) < procdate_i for t in tags if t == "srcused")
        assert all(tags.index(t) > procdate_i for t in tags if t == "srcprod")


def test_srcused_srcprod_resolve_to_srcinfo(config, fabric, defaults, manifest):
    """Every <srcused>/<srcprod> abbreviation has a matching <srccitea>.

    Dangling Source Citation Abbreviations are an mp error; this guards the
    basename -> srcinfo mapping in fgdc._srcinfo_entries.
    """
    root = etree.fromstring(
        fgdc.render_fgdc(
            _fabric_mcf(config, fabric, defaults, manifest), kind="fabric"
        ).encode("utf-8")
    )
    defined = {e.text for e in root.findall(".//srccitea")}
    referenced = {e.text for e in root.findall(".//srcused")} | {
        e.text for e in root.findall(".//srcprod")
    }
    assert referenced <= defined


def test_bbox_maps_to_bounding(defaults, manifest):
    """MCF bbox [W, S, E, N] -> westbc/southbc/eastbc/northbc."""
    root = etree.fromstring(
        fgdc.render_fgdc(
            _source_mcf("era5_land", defaults, manifest), kind="source"
        ).encode("utf-8")
    )
    bounding = root.find(".//bounding")
    assert bounding.findtext("westbc") == "-125.0"
    assert bounding.findtext("eastbc") == "-66.0"
    assert bounding.findtext("northbc") == "53.0"
    assert bounding.findtext("southbc") == "24.7"


def test_snodas_has_bounding(defaults, manifest):
    """The inline catalog bbox fix gives SNODAS a mandatory FGDC <bounding>."""
    root = etree.fromstring(
        fgdc.render_fgdc(
            _source_mcf("snodas", defaults, manifest), kind="source"
        ).encode("utf-8")
    )
    assert root.find(".//bounding") is not None


def test_umbrella_has_no_eainfo_or_dataqual(config, fabric, defaults, manifest):
    """The umbrella carries no files and no per-step lineage of its own."""
    root = etree.fromstring(
        fgdc.render_fgdc(
            _umbrella_mcf(config, fabric, defaults, manifest), kind="umbrella"
        ).encode("utf-8")
    )
    assert root.find("eainfo") is None
    assert root.find("dataqual") is None
    # The lineage prose still surfaces in supplemental information.
    assert root.findtext(".//supplinf")


def test_special_characters_are_escaped(config, fabric, defaults, manifest):
    """Range notes contain '>=' / '&' which must become XML entities."""
    xml = fgdc.render_fgdc(
        _fabric_mcf(config, fabric, defaults, manifest), kind="fabric"
    )
    assert "&gt;=1" in xml
    # Round-trips: the escaped text parses back to the literal.
    root = etree.fromstring(xml.encode("utf-8"))
    udoms = " ".join(e.text or "" for e in root.findall(".//udom"))
    assert ">=1 source is finite" in udoms
