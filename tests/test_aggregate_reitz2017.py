"""Tests for Reitz 2017 aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets.aggregate.reitz2017 import ADAPTER


def test_adapter_declares_recharge_variables():
    assert ADAPTER.source_key == "reitz2017"
    assert ADAPTER.output_name == "reitz2017_agg.nc"
    assert ADAPTER.variables == ("total_recharge", "eff_recharge")
    assert ADAPTER.source_crs == "EPSG:4269"
