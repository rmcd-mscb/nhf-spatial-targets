"""ERA5-Land aggregation adapter (runoff: ro, sro, ssro)."""

from __future__ import annotations

from pathlib import Path

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="era5_land",
    output_name="era5_land_agg.nc",
    variables=("ro", "sro", "ssro"),
    # No x_coord/y_coord overrides: the monthly NCs carry CF axis attrs
    # (axis: X/Y, standard_name: longitude/latitude) on `lon`/`lat` dims,
    # which detect_coords resolves automatically. Earlier overrides set
    # "longitude"/"latitude" but the dim names are "lon"/"lat", which
    # raised "x override 'longitude' is not a dim of 'ro'".
    # The ERA5-Land fetch script writes hourly/, daily/, and monthly/
    # subdirs under <datastore>/era5_land/. Aggregation reads the monthly
    # consolidated NCs only, so the glob carries the subdir component.
    files_glob="monthly/era5_land_monthly_*.nc",
)


def aggregate_era5_land(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> None:
    aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
