"""Tests for WaterGAP 2.2d aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets.aggregate.watergap22d import ADAPTER


def test_adapter_declares_qrdif():
    assert ADAPTER.source_key == "watergap22d"
    assert ADAPTER.output_name == "watergap22d_agg.nc"
    assert ADAPTER.variables == ("qrdif",)
