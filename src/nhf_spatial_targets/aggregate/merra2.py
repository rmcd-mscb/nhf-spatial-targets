"""MERRA-2 M2TMNXLND monthly soil wetness adapter (GWETTOP/GWETROOT/GWETPROF)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="merra2",
    output_name="merra2_agg.nc",
    variables=["GWETTOP", "GWETROOT", "GWETPROF"],
)


def aggregate_merra2(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    """Aggregate MERRA-2 monthly soil wetness to HRU polygons."""
    return aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
