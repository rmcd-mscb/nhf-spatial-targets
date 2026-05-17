"""Tests for ERA5-Land aggregation adapters (monthly runoff + daily sd)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from nhf_spatial_targets.aggregate._coords import detect_coords
from nhf_spatial_targets.aggregate.era5_land import ADAPTER, ADAPTER_SD


def test_adapter_declares_runoff_vars():
    assert ADAPTER.source_key == "era5_land"
    assert ADAPTER.output_name == "era5_land_agg.nc"
    assert set(ADAPTER.variables) == {"ro", "sro", "ssro"}


def test_adapter_uses_monthly_files_glob():
    # Glob carries the `monthly/` subdir component because the era5_land
    # fetch script writes consolidated monthly NCs into <datastore>/
    # era5_land/monthly/ alongside daily/ and hourly/ subdirs.
    assert ADAPTER.files_glob == "monthly/era5_land_monthly_*.nc"


def test_sd_adapter_separate_storage_keys():
    """Daily sd adapter uses synthetic source_key to keep its outputs in
    their own per-source dir (era5_land_sd/), while inheriting catalog
    metadata and raw NCs from the real era5_land entry."""
    assert ADAPTER_SD.source_key == "era5_land_sd"
    assert ADAPTER_SD.catalog_key == "era5_land"
    assert ADAPTER_SD.raw_dir_key == "era5_land"


def test_sd_adapter_declares_only_sd():
    assert ADAPTER_SD.variables == ("sd",)


def test_sd_adapter_uses_daily_files_glob():
    """sd is instantaneous; SWE target needs daily snapshots, not monthly
    means. The daily aggregator reads from <datastore>/era5_land/daily/.
    """
    assert ADAPTER_SD.files_glob == "daily/era5_land_daily_*.nc"


def test_sd_adapter_default_mean():
    """sd arrives without per-pixel masking; default ``mean`` propagates
    NaN honestly at HRU-edge cells (same rationale as SNODAS)."""
    assert ADAPTER_SD.stat_method == "mean"


def test_monthly_and_sd_have_distinct_source_keys():
    """The two ERA5-Land adapters must keep separate aggregated dirs so
    read_aggregated_source's `<source_key>_*_agg.nc` glob can't cross
    them. Bug class: PR #133 review consider — runoff target picking up
    daily sd files would broadcast NaN on ro/sro/ssro for any non-monthly
    time."""
    assert ADAPTER.source_key != ADAPTER_SD.source_key


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
