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
    """Safety-net mask for MODIS ET_500m special/fill values.

    Primary fill-masking now happens at consolidation time, *before* the
    sinusoidal-to-EPSG:4326 reprojection in
    ``fetch.consolidate._mosaic_and_reproject_timestep``. That is the only
    place where masking is correct: the consolidation reprojection uses
    ``Resampling.average``, and post-reprojection masking cannot undo the
    fill-contaminated averages it produces (an HRU with 50/50 valid/fill
    blend reads as ~1660 mm/8day in scaled units, well below the 3270
    threshold). See PR #88 / lessons-learned.md § MOD16A2 v061
    flat-on-CONUS+ for the worked example.

    This hook remains as belt-and-suspenders against (a) consolidated NCs
    produced by older pipeline versions and (b) any pure-fill cells that
    happen to slip through (which would only occur if the consolidation
    mask is bypassed). On a freshly-consolidated NC it is a no-op.
    """
    ds["ET_500m"] = ds["ET_500m"].where(ds["ET_500m"] <= 3270.0)
    return ds


ADAPTER = SourceAdapter(
    source_key="mod16a2_v061",
    output_name="mod16a2_agg.nc",
    variables=["ET_500m"],
    source_crs="EPSG:4326",  # consolidate_mod16a2 reprojects tiles to WGS84
    pre_aggregate_hook=_mask_et_fill,
    # Per-pixel fill mask runs in pre_aggregate_hook AND inside
    # consolidate before reprojection (PR #88). With stat_method="mean"
    # those NaN pixels would propagate to the HRU, marking every HRU
    # that touches a coastline / water body / snow boundary as NaN.
    # Use masked_mean so the HRU value reflects the area-weighted mean
    # of pixels that survived the fill mask; HRUs whose contributing
    # pixels are all fills still come out NaN, which is the honest
    # "no MOD16A2 here" signal. See
    # docs/architecture/transformation-pipeline.md.
    stat_method="masked_mean",
)


def aggregate_mod16a2(
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
    *,
    worker_index: int = 0,
    n_workers: int = 1,
) -> None:
    """Aggregate MOD16A2 v061 8-day AET to HRU polygons.

    gdptools reprojects the declared sinusoidal source onto the driver's
    equal-area weight CRS for area-weighted intersection.
    """
    aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
        worker_index=worker_index,
        n_workers=n_workers,
    )
