"""Tests for nhf_spatial_targets.release.validate_xml.

``mp`` (the USGS Metadata Parser) is operator-installed and absent in CI, so
the deterministic tests here cover the report parser (with a synthetic ``mp``
report) and the ``lxml`` well-formedness fallback (forced via monkeypatch).
The actual ``mp`` run against the committed FGDC goldens is integration-gated
and skipped unless ``mp`` is on PATH.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nhf_spatial_targets.release import validate_xml

FX = Path(__file__).parent / "fixtures" / "release"

FGDC_GOLDENS = [
    "expected_source_era5_fgdc.xml",
    "expected_source_snodas_fgdc.xml",
    "expected_fabric_gfv2_fgdc.xml",
    "expected_umbrella_fgdc.xml",
]


def _force_no_mp(monkeypatch) -> None:
    monkeypatch.setattr(validate_xml.shutil, "which", lambda *a, **k: None)


def test_mp_available_returns_bool():
    assert isinstance(validate_xml.mp_available(), bool)


def test_parse_mp_report_splits_warnings_and_errors():
    report = (
        "mp: metadata.xml\n"
        "Warnings:\n"
        "  3: optional element missing\n"
        "Errors:\n"
        "  10: element out of order\n"
        "  11: undefined source citation abbreviation\n"
    )
    warnings, errors = validate_xml._parse_mp_report(report)
    assert warnings == ["3: optional element missing"]
    assert errors == [
        "10: element out of order",
        "11: undefined source citation abbreviation",
    ]


def test_wellformed_fallback_passes_valid_xml(monkeypatch):
    _force_no_mp(monkeypatch)
    good = (FX / "expected_source_era5_fgdc.xml").read_text()
    result = validate_xml.validate_xml_string(good)
    assert result.ok is True
    assert result.errors == []
    assert any("well-formed" in w for w in result.warnings)


def test_wellformed_fallback_flags_malformed_xml(monkeypatch):
    _force_no_mp(monkeypatch)
    result = validate_xml.validate_xml_string("<a><b></a>")
    assert result.ok is False
    assert result.errors


@pytest.mark.parametrize("name", FGDC_GOLDENS)
def test_golden_fgdc_is_wellformed(name):
    """A committed FGDC golden must at least be well-formed (mp-independent)."""
    from lxml import etree

    etree.parse(str(FX / name))


@pytest.mark.integration
@pytest.mark.skipif(
    not validate_xml.mp_available(), reason="mp (USGS Metadata Parser) not on PATH"
)
@pytest.mark.parametrize("name", FGDC_GOLDENS)
def test_mp_reports_no_errors_on_goldens(name):
    result = validate_xml.validate_xml_file(FX / name)
    assert not result.errors, result.errors
