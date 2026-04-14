"""MOD16A2 v061 AET adapter (sinusoidal MODIS projection)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source

# MODIS sinusoidal PROJ4 string — Earth as a sphere of radius 6371007.181 m.
MODIS_SINUSOIDAL_PROJ = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
)


ADAPTER = SourceAdapter(
    source_key="mod16a2_v061",
    output_name="mod16a2_agg.nc",
    variables=["ET_500m"],
    x_coord="x",
    y_coord="y",
    source_crs=MODIS_SINUSOIDAL_PROJ,
)


def aggregate_mod16a2(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    """Aggregate MOD16A2 v061 8-day AET to HRU polygons.

    gdptools reprojects the declared sinusoidal source onto EPSG:5070 for
    area-weighted intersection.
    """
    return aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
