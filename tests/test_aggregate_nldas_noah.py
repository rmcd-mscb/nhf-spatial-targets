"""Tests for NLDAS-NOAH aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets.aggregate.nldas_noah import ADAPTER


def test_adapter_declares_four_layers():
    assert ADAPTER.source_key == "nldas_noah"
    assert ADAPTER.output_name == "nldas_noah_agg.nc"
    assert set(ADAPTER.variables) == {
        "SoilM_0_10cm",
        "SoilM_10_40cm",
        "SoilM_40_100cm",
        "SoilM_100_200cm",
    }
