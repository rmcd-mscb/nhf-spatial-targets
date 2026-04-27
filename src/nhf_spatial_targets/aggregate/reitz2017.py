"""Reitz 2017 annual recharge adapter (total_recharge + eff_recharge)."""

from __future__ import annotations

from pathlib import Path

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="reitz2017",
    output_name="reitz2017_agg.nc",
    variables=("total_recharge", "eff_recharge"),
    source_crs="EPSG:4269",  # NAD83 geographic — Reitz GeoTIFFs preserve this
)


def aggregate_reitz2017(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> None:
    aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
