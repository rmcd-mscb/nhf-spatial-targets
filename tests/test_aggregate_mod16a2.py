"""Tests for MOD16A2 aggregation adapter (sinusoidal CRS)."""

from __future__ import annotations

from nhf_spatial_targets.aggregate.mod16a2 import ADAPTER, MODIS_SINUSOIDAL_PROJ


def test_adapter_declares_sinusoidal_crs():
    assert ADAPTER.source_key == "mod16a2_v061"
    assert ADAPTER.output_name == "mod16a2_agg.nc"
    assert ADAPTER.variables == ("ET_500m",)
    assert ADAPTER.source_crs == MODIS_SINUSOIDAL_PROJ
    assert "+proj=sinu" in ADAPTER.source_crs
    # x_coord / y_coord are None — resolved at runtime via CF attrs on the
    # consolidated NC (projection_x_coordinate / projection_y_coordinate).
    assert ADAPTER.x_coord is None
    assert ADAPTER.y_coord is None
