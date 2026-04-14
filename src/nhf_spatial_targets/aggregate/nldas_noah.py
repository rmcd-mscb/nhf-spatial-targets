"""NLDAS-2 NOAH monthly soil moisture adapter (four layers)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="nldas_noah",
    output_name="nldas_noah_agg.nc",
    variables=[
        "SoilM_0_10cm",
        "SoilM_10_40cm",
        "SoilM_40_100cm",
        "SoilM_100_200cm",
    ],
)


def aggregate_nldas_noah(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    return aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
