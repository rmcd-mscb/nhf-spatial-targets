"""SNODAS daily SWE aggregator (CONUS).

Reads per-year consolidated NCs at
``<datastore>/snodas/daily/snodas_daily_<year>.nc`` (written by
``fetch.snodas.consolidate_year_snodas``) and emits area-weighted HRU
means via the shared aggregation driver.

The consolidated NCs are **pre-projected to EPSG:5070 at consolidate
time** (issue #121): SWE arrives here on a 1000 m NAD83 / CONUS Albers
grid with projected ``y``/``x`` coords in metres. Declaring
``source_crs="EPSG:5070"`` here matches the driver's ``WEIGHT_GEN_CRS``,
so gdptools' WeightGen skips the reproject-source-to-target step
entirely (~5-8× speedup per batch on CONUS fabrics).

``stat_method="masked_mean"`` (issue #151): SNODAS is the
**CONUS-masked** NSIDC product, where roughly half of the native bbox
carries ``_FillValue=-9999`` by design (oceans, the Canadian and
Mexican portions of the bbox, the Great Lakes). xarray's default
``mask_and_scale=True`` decodes those fills to NaN at open time, and
the ``pre_aggregate_hook`` below re-asserts that mask explicitly so
the design intent is visible at the adapter level. Under the default
``mean``, one NaN pixel anywhere in an HRU footprint propagates a NaN
HRU value — measured against gfv2 in 2010-03-01 that meant 40,230 of
361,471 HRUs (11.13%) were SNODAS-NaN at peak winter despite their
footprint being almost entirely inside CONUS. Same class of deliberate
per-pixel mask as ``aggregate/mod16a2.py`` and ``aggregate/mod10c1.py``,
which already use ``masked_mean``; see CLAUDE.md's "Aggregation
Transformation Policy" → ``stat_method`` choice.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source

# Pixel-level fill mask: SNODAS reserves -9999 as the only documented
# fill code; xarray's mask_and_scale decodes that to NaN on read, so we
# gate on `> -9990` (strict, with a slop margin) to catch any encoded
# values that survive a non-standard reader. Per CLAUDE.md's
# transformation policy, the explicit hook is what justifies the
# masked_mean stat_method below.
_SNODAS_FILL_THRESHOLD = -9990


def _mask_snodas_fill(ds: xr.Dataset) -> xr.Dataset:
    """Re-assert NaN on SNODAS fill-coded pixels before aggregation."""
    return ds.assign(swe=ds["swe"].where(ds["swe"] > _SNODAS_FILL_THRESHOLD))


ADAPTER = SourceAdapter(
    source_key="snodas",
    output_name="snodas_agg.nc",
    variables=("swe",),
    # Pre-projected at consolidate time (issue #121); matches the
    # driver's WEIGHT_GEN_CRS so gdptools does not reproject.
    source_crs="EPSG:5070",
    files_glob="daily/snodas_daily_*.nc",
    pre_aggregate_hook=_mask_snodas_fill,
    # Deliberate per-pixel CONUS mask -> masked_mean (issue #151).
    # Without this override, the per-pixel -9999 fill poisons every HRU
    # that intersects even one fill pixel — measured as ~11% NaN HRUs at
    # peak winter on gfv2. masked_mean computes the weighted mean of the
    # finite survivors; the HRU is NaN only when its entire footprint is
    # fill.
    stat_method="masked_mean",
)


def aggregate_snodas(
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
    period: str | None = None,
) -> None:
    """Aggregate SNODAS daily SWE to HRU polygons; emit per-year NCs."""
    aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
        period=period,
    )
