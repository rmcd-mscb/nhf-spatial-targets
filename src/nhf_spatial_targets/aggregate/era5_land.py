"""ERA5-Land aggregation adapter (runoff: ro, sro, ssro)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source
from nhf_spatial_targets.workspace import Project


def _open_monthly(project: Project) -> xr.Dataset:
    """Open the monthly consolidated ERA5-Land NC in the datastore.

    The ERA5-Land fetch stores both a daily and monthly NC; pick the monthly
    one by filename token.
    """
    raw_dir = project.raw_dir("era5_land")
    monthly_ncs = sorted(Path(raw_dir).glob("*monthly*.nc"))
    if not monthly_ncs:
        raise FileNotFoundError(
            f"No monthly ERA5-Land NC found in {raw_dir}. "
            "Run 'nhf-targets fetch era5-land' first."
        )
    ds = xr.open_dataset(monthly_ncs[0])
    loaded = ds.load()
    ds.close()
    return loaded


ADAPTER = SourceAdapter(
    source_key="era5_land",
    output_name="era5_land_agg.nc",
    variables=["ro", "sro", "ssro"],
    x_coord="longitude",
    y_coord="latitude",
    open_hook=_open_monthly,
)


def aggregate_era5_land(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    """Aggregate ERA5-Land monthly runoff (ro, sro, ssro) to HRU polygons."""
    return aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
