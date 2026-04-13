from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.fetch.gldas import (
    BBOX_NWSE,
    clip_to_bbox,
    derive_runoff_total,
)


def _global_grid(value_qs=2.0, value_qsb=3.0):
    lat = np.arange(-89.875, 90.0, 0.25)
    lon = np.arange(-179.875, 180.0, 0.25)
    times = pd.date_range("2020-01-01", periods=2, freq="1MS")
    shape = (len(times), len(lat), len(lon))
    return xr.Dataset(
        {
            "Qs_acc": (("time", "lat", "lon"), np.full(shape, value_qs)),
            "Qsb_acc": (("time", "lat", "lon"), np.full(shape, value_qsb)),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )


def test_derive_runoff_total_sums_qs_and_qsb():
    ds = _global_grid()
    out = derive_runoff_total(ds)
    assert "runoff_total" in out.data_vars
    np.testing.assert_allclose(out.runoff_total.values, 5.0)
    assert (
        out.runoff_total.attrs["long_name"]
        == "total runoff (Qs_acc + Qsb_acc, derived)"
    )
    assert out.runoff_total.attrs["units"] == "kg m-2"


def test_clip_to_bbox_reduces_extent():
    ds = _global_grid()
    clipped = clip_to_bbox(ds, BBOX_NWSE)
    # bbox is N=53.0, W=-125.0, S=24.7, E=-66.0
    assert clipped.lat.min() >= 24.7 - 0.25
    assert clipped.lat.max() <= 53.0 + 0.25
    assert clipped.lon.min() >= -125.0 - 0.25
    assert clipped.lon.max() <= -66.0 + 0.25
    assert clipped.lat.size < ds.lat.size
    assert clipped.lon.size < ds.lon.size
