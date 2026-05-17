"""Tests for the snow water equivalent calibration target (catalog-level).

End-to-end builder behaviour lives in ``test_targets_swe.py``; this file
keeps only the catalog-level invariants that should hold regardless of
builder implementation status.
"""

from __future__ import annotations


def test_swe_variable_lists_four_sources():
    """SWE catalog variable carries all four SWE sources."""
    from nhf_spatial_targets import catalog

    v = catalog.variable("snow_water_equivalent")
    assert set(v["sources"]) == {
        "daymet",
        "snodas",
        "era5_land",
        "margulis_wus_sr",
    }
