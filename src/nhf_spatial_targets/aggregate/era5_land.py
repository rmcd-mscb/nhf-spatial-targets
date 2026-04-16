"""ERA5-Land aggregation adapter (runoff: ro, sro, ssro)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="era5_land",
    output_name="era5_land_agg.nc",
    variables=("ro", "sro", "ssro"),
    x_coord="longitude",
    y_coord="latitude",
    files_glob="*monthly*.nc",
)


def aggregate_era5_land(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    return aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
