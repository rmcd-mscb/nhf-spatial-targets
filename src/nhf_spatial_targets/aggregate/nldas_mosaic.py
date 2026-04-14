"""NLDAS-2 MOSAIC monthly soil moisture adapter (three layers)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="nldas_mosaic",
    output_name="nldas_mosaic_agg.nc",
    variables=["SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_200cm"],
)


def aggregate_nldas_mosaic(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    """Aggregate NLDAS-2 MOSAIC monthly soil moisture to HRU polygons."""
    return aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
