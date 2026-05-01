"""Tests for ERA5-Land aggregation adapter."""

from __future__ import annotations

import numpy as np
import xarray as xr

from nhf_spatial_targets.aggregate._coords import detect_coords
from nhf_spatial_targets.aggregate.era5_land import ADAPTER


def test_adapter_declares_runoff_vars():
    assert ADAPTER.source_key == "era5_land"
    assert ADAPTER.output_name == "era5_land_agg.nc"
    assert set(ADAPTER.variables) == {"ro", "sro", "ssro"}


def test_adapter_uses_monthly_files_glob():
    # Glob carries the `monthly/` subdir component because the era5_land
    # fetch script writes consolidated monthly NCs into <datastore>/
    # era5_land/monthly/ alongside daily/ and hourly/ subdirs.
    assert ADAPTER.files_glob == "monthly/era5_land_monthly_*.nc"


def test_adapter_relies_on_cf_coord_detection():
    # The on-disk monthly NCs use `lat`/`lon` dim names (not
    # `latitude`/`longitude`). Setting x_coord/y_coord overrides to the
    # CF standard names raised "override is not a dim of 'ro'". The
    # adapter should leave overrides unset and let detect_coords resolve
    # axes from CF metadata.
    assert ADAPTER.x_coord is None
    assert ADAPTER.y_coord is None


def test_detect_coords_resolves_era5_land_shape():
    # Mirrors the on-disk monthly NC: dims (time, lat, lon) with CF
    # axis/standard_name attrs on lat and lon. detect_coords must return
    # the dim names ("lon", "lat", "time"), not the standard names.
    ds = xr.Dataset(
        {"ro": (["time", "lat", "lon"], np.zeros((1, 2, 2)))},
        coords={
            "time": ("time", [0], {"standard_name": "time", "axis": "T"}),
            "lat": (
                "lat",
                [0.0, 1.0],
                {"standard_name": "latitude", "axis": "Y", "units": "degrees_north"},
            ),
            "lon": (
                "lon",
                [0.0, 1.0],
                {"standard_name": "longitude", "axis": "X", "units": "degrees_east"},
            ),
        },
    )
    x, y, t = detect_coords(
        ds,
        ADAPTER.grid_variable,
        x_override=ADAPTER.x_coord,
        y_override=ADAPTER.y_coord,
        time_override=ADAPTER.time_coord,
    )
    assert (x, y, t) == ("lon", "lat", "time")
