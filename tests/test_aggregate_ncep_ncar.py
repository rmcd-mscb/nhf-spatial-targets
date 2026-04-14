"""Tests for NCEP/NCAR aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets.aggregate.ncep_ncar import ADAPTER


def test_adapter_declares_soil_moisture_vars():
    assert ADAPTER.source_key == "ncep_ncar"
    assert ADAPTER.output_name == "ncep_ncar_agg.nc"
    assert set(ADAPTER.variables) == {"soilw_0_10cm", "soilw_10_200cm"}
