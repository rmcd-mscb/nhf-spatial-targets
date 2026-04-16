"""MOD16A2 v061 AET adapter.

consolidate_mod16a2 reprojects the raw sinusoidal HDF tiles to WGS84 geographic
(EPSG:4326) and renames the spatial coords to ``lon`` / ``lat`` before writing
the consolidated NC.  The adapter therefore declares EPSG:4326 — using the
original sinusoidal CRS here would cause gdptools to reproject the fabric bounds
into sinusoidal metres and attempt to slice the lon/lat coordinate with those
values, producing an empty spatial subset.

MODIS_SINUSOIDAL_PROJ is retained for reference / tests but is no longer used
in the adapter.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source

# MODIS Land Tile gridding uses an authalic sphere (not WGS84 ellipsoid).
# Retained for reference; the consolidated NC is reprojected to EPSG:4326.
MODIS_SINUSOIDAL_PROJ = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
)


def _mask_et_fill(ds: xr.Dataset) -> xr.Dataset:
    """Mask MODIS ET_500m special/fill values before area-weighted aggregation.

    The consolidated NC stores scaled float32 values (raw × 0.1).  MOD16A2
    valid_range for ET_500m is 0–32700 raw (0–3270.0 scaled).  Values above
    3270 are special codes (water=32761, barren=32762, snow/ice=32763,
    cloudy=32764, no-data=32766, not-processed=32767 in raw units) that must
    be set to NaN so they do not contaminate the weighted mean.
    """
    ds["ET_500m"] = ds["ET_500m"].where(ds["ET_500m"] <= 3270.0)
    return ds


ADAPTER = SourceAdapter(
    source_key="mod16a2_v061",
    output_name="mod16a2_agg.nc",
    variables=["ET_500m"],
    source_crs="EPSG:4326",  # consolidate_mod16a2 reprojects tiles to WGS84
    pre_aggregate_hook=_mask_et_fill,
)


def aggregate_mod16a2(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    """Aggregate MOD16A2 v061 8-day AET to HRU polygons.

    gdptools reprojects the declared sinusoidal source onto the driver's
    equal-area weight CRS for area-weighted intersection.
    """
    return aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
