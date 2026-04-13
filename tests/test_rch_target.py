"""Tests for the recharge calibration target configuration."""

from __future__ import annotations


def test_recharge_target_lists_three_sources():
    from nhf_spatial_targets import catalog

    v = catalog.variable("recharge")
    assert set(v["sources"]) == {"reitz2017", "watergap22d", "era5_land"}
