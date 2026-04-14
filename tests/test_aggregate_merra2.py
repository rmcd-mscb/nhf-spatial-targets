"""Tests for MERRA-2 aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets.aggregate.merra2 import ADAPTER


def test_adapter_declares_soil_wetness_vars():
    assert ADAPTER.source_key == "merra2"
    assert ADAPTER.output_name == "merra2_agg.nc"
    assert set(ADAPTER.variables) == {"GWETTOP", "GWETROOT", "GWETPROF"}
