"""Tests for NLDAS-MOSAIC aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets.aggregate.nldas_mosaic import ADAPTER


def test_adapter_declares_three_layers():
    assert ADAPTER.source_key == "nldas_mosaic"
    assert ADAPTER.output_name == "nldas_mosaic_agg.nc"
    assert set(ADAPTER.variables) == {
        "SoilM_0_10cm",
        "SoilM_10_40cm",
        "SoilM_40_200cm",
    }
