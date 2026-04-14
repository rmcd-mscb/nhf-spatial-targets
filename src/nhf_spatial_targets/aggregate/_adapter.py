"""Declarative adapter for tier-1 gridded sources aggregated via gdptools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import xarray as xr

from nhf_spatial_targets.workspace import Project


@dataclass(frozen=True)
class SourceAdapter:
    """Declarative description of a source for the shared aggregation driver.

    Use for sources whose consolidated NetCDF can be opened directly (optionally
    via an ``open_hook`` for derived variables or file selection) and handed to
    ``aggregate_source`` without per-source batching/weighting logic. Sources
    requiring pre-aggregation masking or post-aggregation rename (e.g. MOD10C1)
    call the driver helpers directly and do not use this adapter.

    ``open_hook`` receives the resolved :class:`Project` and must return an
    :class:`xarray.Dataset` with CRS set and all ``variables`` present
    (including any derived variables). When ``None``, the driver opens the
    single consolidated NetCDF under ``project.raw_dir(source_key)``.
    """

    source_key: str
    output_name: str
    variables: tuple[str, ...]
    x_coord: str = "lon"
    y_coord: str = "lat"
    time_coord: str = "time"
    source_crs: str = "EPSG:4326"
    grid_variable: str | None = None
    open_hook: Callable[[Project], xr.Dataset] | None = field(default=None)

    def __post_init__(self) -> None:
        # Coerce list → tuple so callers can pass list literals.
        object.__setattr__(self, "variables", tuple(self.variables))
        if len(self.variables) == 0:
            raise ValueError("SourceAdapter.variables must be non-empty")
        if "/" in self.output_name or "\\" in self.output_name:
            raise ValueError(
                f"SourceAdapter.output_name must be a bare filename, got {self.output_name!r}"
            )
        # Named invariant: which declared variable is used to infer the source
        # grid for WeightGen. Defaults to the first variable; drivers/tests that
        # care about a specific one can override.
        if self.grid_variable is None:
            object.__setattr__(self, "grid_variable", self.variables[0])
        elif self.grid_variable not in self.variables:
            raise ValueError(
                f"SourceAdapter.grid_variable {self.grid_variable!r} must be one of "
                f"variables={self.variables}"
            )
        # Catalog-typo check. If the catalog is unavailable at construction
        # time (e.g. test harness, repackaged install), defer the check until
        # the driver runs. Do NOT swallow KeyError — that's the typo case we
        # specifically want to surface.
        try:
            from nhf_spatial_targets.catalog import source as _catalog_source

            _catalog_source(self.source_key)
        except KeyError as exc:
            raise ValueError(
                f"SourceAdapter.source_key {self.source_key!r} not found in "
                f"catalog/sources.yml"
            ) from exc
        except Exception:
            # Catalog file missing/unreadable/YAML broken — let the aggregator
            # surface this at run time with richer context.
            pass
