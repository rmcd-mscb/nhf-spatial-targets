"""USGS MWBM (ClimGrid-forced) monthly aggregator: runoff, aet, soilstorage, swe."""

from __future__ import annotations

from pathlib import Path

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="mwbm_climgrid",
    output_name="mwbm_climgrid_agg.nc",
    variables=("runoff", "aet", "soilstorage", "swe"),
    files_glob="ClimGrid_WBM.nc",
)


def aggregate_mwbm_climgrid(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> None:
    aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
