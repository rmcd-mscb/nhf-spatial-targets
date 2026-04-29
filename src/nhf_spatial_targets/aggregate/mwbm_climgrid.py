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
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
    period: str | None = None,
) -> None:
    """Aggregate the MWBM ClimGrid source to HRU polygons.

    The publisher distributes a single 1895-2020 NetCDF; pass ``period``
    (e.g. ``"1979/2020"``) to clip the per-year output to a specific
    window, otherwise every year in the file is aggregated. The
    publisher's 1895-1899 spinup is not auto-skipped — clip it via
    ``period`` if you don't want it.
    """
    aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
        period=period,
    )
