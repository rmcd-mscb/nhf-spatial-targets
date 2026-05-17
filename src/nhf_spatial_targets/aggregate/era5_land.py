"""ERA5-Land aggregation adapters.

Two adapters for one upstream source:

- ``ADAPTER`` reads the **monthly** consolidated NCs and emits the
  runoff variables ``ro``, ``sro``, and ``ssro`` (used by the runoff and
  recharge targets). Output dir:
  ``<project>/data/aggregated/era5_land/era5_land_<year>_agg.nc``.

- ``ADAPTER_SD`` reads the **daily** consolidated NCs and emits the
  snow depth water equivalent ``sd`` (used by the SWE target). Output
  dir: ``<project>/data/aggregated/era5_land_sd/era5_land_sd_<year>_agg.nc``.

The two outputs live in separate per-source directories so the loose
``{source_key}_*_agg.nc`` glob in
``targets._common.read_aggregated_source`` cannot pick up both cadences
when a target asks for one. The daily adapter sets ``catalog_key`` and
``raw_dir_key`` to ``"era5_land"`` so it inherits the real source's CF
metadata and reads from ``<datastore>/era5_land/daily/`` while still
keeping its own weight cache, manifest entry, and aggregated subdir
keyed on the synthetic ``"era5_land_sd"``.
"""

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


# SWE target consumes daily snapshots; the monthly adapter above only
# emits the runoff/recharge accumulation variables, so `sd` gets its own
# adapter reading from <datastore>/era5_land/daily/*.nc.
ADAPTER_SD = SourceAdapter(
    source_key="era5_land_sd",
    catalog_key="era5_land",
    raw_dir_key="era5_land",
    output_name="era5_land_sd_agg.nc",
    variables=("sd",),
    files_glob="daily/era5_land_daily_*.nc",
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


def aggregate_era5_land_sd(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> None:
    """Aggregate ERA5-Land daily snow depth (`sd`) to HRU polygons."""
    aggregate_source(
        ADAPTER_SD,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
