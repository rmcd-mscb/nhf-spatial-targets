"""Tests for MOD16A2 aggregation adapter."""

from __future__ import annotations

import numpy as np
import xarray as xr

from nhf_spatial_targets.aggregate.mod16a2 import ADAPTER, MODIS_SINUSOIDAL_PROJ


def test_adapter_declares_wgs84_crs():
    # consolidate_mod16a2 reprojects tiles to EPSG:4326; the adapter must
    # match so gdptools slices lon/lat correctly (not sinusoidal metres).
    assert ADAPTER.source_key == "mod16a2_v061"
    assert ADAPTER.output_name == "mod16a2_agg.nc"
    assert ADAPTER.variables == ("ET_500m",)
    assert ADAPTER.source_crs == "EPSG:4326"
    # MODIS_SINUSOIDAL_PROJ is retained for reference but NOT used as source_crs.
    assert "+proj=sinu" in MODIS_SINUSOIDAL_PROJ
    # x_coord / y_coord are None — resolved at runtime via CF attrs on the
    # consolidated NC.
    assert ADAPTER.x_coord is None
    assert ADAPTER.y_coord is None


def test_adapter_has_fill_value_hook():
    assert ADAPTER.pre_aggregate_hook is not None


def test_mask_et_fill_masks_special_codes():
    """_mask_et_fill replaces MODIS special codes (>3270) with NaN."""
    from nhf_spatial_targets.aggregate.mod16a2 import _mask_et_fill

    # Simulate a tiny dataset: one valid pixel, one fill-value pixel.
    # MODIS ET_500m valid_range_max = 32700 raw = 3270.0 scaled.
    # Fill codes (water, barren, cloudy, etc.) range from 32761–32767 raw
    # = 3276.1–3276.7 scaled.
    et = xr.DataArray(
        np.array([[10.0, 50.0], [3270.0, 3276.6]], dtype=np.float32),
        dims=["time", "space"],
    )
    ds = xr.Dataset({"ET_500m": et})

    result = _mask_et_fill(ds)

    # Values at or below 3270 are preserved; above are NaN.
    assert float(result["ET_500m"].values[0, 0]) == 10.0
    assert float(result["ET_500m"].values[0, 1]) == 50.0
    assert float(result["ET_500m"].values[1, 0]) == 3270.0  # boundary kept
    assert np.isnan(result["ET_500m"].values[1, 1])  # fill → NaN
