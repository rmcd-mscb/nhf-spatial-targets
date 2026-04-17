"""WaterGAP 2.2d monthly diffuse groundwater recharge adapter (qrdif)."""

from __future__ import annotations

from pathlib import Path

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="watergap22d",
    output_name="watergap22d_agg.nc",
    variables=["qrdif"],
    files_glob="*_cf.nc",
)


def aggregate_watergap22d(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> None:
    aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
