"""Margulis Western US Snow Reanalysis daily SWE aggregator (Oregon-only).

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

The catalog declares ``fabric_scope.fabrics: [or]`` for this source;
``fetch_margulis_wus_sr`` records that scope in ``manifest.json`` but
does not enforce it (raw downloads stay reusable across projects). The
SWE target builder is responsible for skipping this source on non-OR
fabrics — see ``targets/swe.py``.
"""

from __future__ import annotations

from pathlib import Path

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="margulis_wus_sr",
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
) -> None:
    """Aggregate Margulis WUS-SR daily SWE to HRU polygons; emit per-year NCs."""
    aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
        period=period,
    )
