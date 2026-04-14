"""Declarative adapter for tier-1 gridded sources aggregated via gdptools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import xarray as xr

from nhf_spatial_targets.workspace import Project


@dataclass(frozen=True)
class SourceAdapter:
    """Declarative description of a tier-1 source for the aggregation driver.

    ``open_hook`` receives the resolved :class:`Project` and must return an
    :class:`xarray.Dataset` with CRS set and all ``variables`` present
    (including any derived variables). When ``None``, the driver opens the
    single consolidated NetCDF under ``project.raw_dir(source_key)``.
    """

    source_key: str
    output_name: str
    variables: list[str]
    x_coord: str = "lon"
    y_coord: str = "lat"
    time_coord: str = "time"
    source_crs: str = "EPSG:4326"
    open_hook: Callable[[Project], xr.Dataset] | None = field(default=None)
