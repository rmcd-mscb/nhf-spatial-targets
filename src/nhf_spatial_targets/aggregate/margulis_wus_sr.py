"""Margulis Western US Snow Reanalysis daily SWE aggregator.

Reads per-year consolidated NCs at
``<datastore>/margulis_wus_sr/daily/margulis_wus_sr_daily_<year>.nc``
(written by ``fetch.margulis_wus_sr`` / its consolidator) and emits
area-weighted HRU means via the shared aggregation driver.

The consolidated NCs store ``SWE`` as float32 in **metres water
equivalent** on a regular WGS84 lat/lon grid (the consolidator
reprojects from the native ~90 m EASE-Grid). Per-pixel NaN over
unmodelled cells decodes naturally via ``mask_and_scale=True``; the
default ``stat_method="mean"`` is the right choice since the source
does not apply any per-pixel quality gate in a ``pre_aggregate_hook``
— HRUs that straddle the WUS domain edge become honest NaN, which the
SWE target builder handles at multi-source combination time.

The source covers only the Western US. The aggregation driver's
geometry-driven coverage guard (#309) skips fabric batches outside the
grid and emits honest NaN rows for them, so on a CONUS fabric the
aggregated NC carries data only at WUS HRUs; the SWE target's NaN-aware
combine uses it wherever it is finite.
"""

from __future__ import annotations

from pathlib import Path

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="margulis_wus_sr",
    output_cadence="daily",
    output_name="margulis_wus_sr_agg.nc",
    variables=("SWE",),
    source_crs="EPSG:4326",
    files_glob="daily/margulis_wus_sr_daily_*.nc",
)


def aggregate_margulis_wus_sr(
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
    period: str | None = None,
    *,
    worker_index: int = 0,
    n_workers: int = 1,
) -> None:
    """Aggregate Margulis WUS-SR daily SWE to HRU polygons; emit per-year NCs."""
    aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
        period=period,
        worker_index=worker_index,
        n_workers=n_workers,
    )
