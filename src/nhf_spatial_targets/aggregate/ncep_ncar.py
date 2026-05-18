"""NCEP/NCAR Reanalysis monthly soil moisture adapter."""

from __future__ import annotations

from pathlib import Path

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="ncep_ncar",
    output_name="ncep_ncar_agg.nc",
    variables=["soilw_0_10cm", "soilw_10_200cm"],
)


def aggregate_ncep_ncar(
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
    *,
    worker_index: int = 0,
    n_workers: int = 1,
) -> None:
    aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
        worker_index=worker_index,
        n_workers=n_workers,
    )
