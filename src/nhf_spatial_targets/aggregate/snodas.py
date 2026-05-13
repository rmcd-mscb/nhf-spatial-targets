"""SNODAS daily SWE aggregator (CONUS).

Reads per-year consolidated NCs at
``<datastore>/snodas/daily/snodas_daily_<year>.nc`` (written by
``fetch.snodas.consolidate_year_snodas``) and emits area-weighted HRU
means via the shared aggregation driver.

The consolidated NCs store ``swe`` as ``int16`` with ``_FillValue=-9999``
and ``scale_factor=1``; xarray's default ``mask_and_scale=True`` decodes
fills to NaN at open time, so the driver sees a plain float field with
NaN over oceans and unmasked regions. No ``pre_aggregate_hook`` is
needed, and ``stat_method="mean"`` is the right choice: any HRU that
straddles the CONUS edge becomes an honest NaN (geometric partial
coverage), which the SWE target builder handles at multi-source
combination time.
"""

from __future__ import annotations

from pathlib import Path

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="snodas",
    output_name="snodas_agg.nc",
    variables=("swe",),
    source_crs="EPSG:4326",
    files_glob="daily/snodas_daily_*.nc",
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
