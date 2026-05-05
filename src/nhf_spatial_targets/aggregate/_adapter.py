"""Declarative adapter for tier-1 gridded sources aggregated via gdptools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import xarray as xr


@dataclass(frozen=True)
class SourceAdapter:
    """Declarative description of a source for the shared aggregation driver.

    Use for sources whose NetCDFs can be opened directly and handed to
    ``aggregate_source`` without per-source batching/weighting logic. Sources
    requiring pre-aggregation masking or post-aggregation rename (e.g. MOD10C1)
    call the driver helpers directly and do not use this adapter.

    ``files_glob`` controls which files are enumerated from the datastore raw
    directory.  The default ``"*_consolidated.nc"`` matches the standard fetch
    output; sources with non-standard naming (e.g. ERA5-Land monthly NCs) set
    this to a tighter pattern.  ``pre_aggregate_hook``, when set, receives each
    lazily-opened per-year Dataset and may add derived variables before the
    aggregation loop runs.

    ``stat_method`` selects the gdptools area-weighted reduction. Default
    ``"mean"`` is the right choice when source pixels arrive at the
    aggregator without per-pixel masking — any NaN source pixel propagates
    to a NaN HRU value, which is an honest "no useful data here" signal.
    Override to ``"masked_mean"`` (skips NaN pixels and computes the
    weighted mean of the survivors) when the source **deliberately** masks
    pixels in ``pre_aggregate_hook`` — fill-value masks, quality gates,
    etc. Without this override, the per-pixel mask poisons every HRU that
    touches even one masked pixel; with it, the HRU mean honestly reports
    the area-weighted mean of pixels that survived the pre-aggregate gate
    and the HRU is NaN only when *every* contributing pixel was masked.
    Currently used by ``aggregate/mod16a2.py`` (PR #88 fill mask) and
    ``aggregate/mod10c1.py`` (CI > 70 quality gate). See
    ``docs/architecture/transformation-pipeline.md`` for the full rule.
    """

    source_key: str
    output_name: str
    variables: tuple[str, ...]
    x_coord: str | None = None
    y_coord: str | None = None
    time_coord: str | None = None
    source_crs: str = "EPSG:4326"
    grid_variable: str | None = None
    raw_grid_variable: str | None = None
    files_glob: str = "*_consolidated.nc"
    pre_aggregate_hook: Callable[[xr.Dataset], xr.Dataset] | None = field(default=None)
    post_aggregate_hook: Callable[[xr.Dataset], xr.Dataset] | None = field(default=None)
    stat_method: str = "mean"

    def __post_init__(self) -> None:
        # Coerce list → tuple so callers can pass list literals.
        object.__setattr__(self, "variables", tuple(self.variables))
        if len(self.variables) == 0:
            raise ValueError("SourceAdapter.variables must be non-empty")
        if "/" in self.output_name or "\\" in self.output_name:
            raise ValueError(
                f"SourceAdapter.output_name must be a bare filename, got {self.output_name!r}"
            )
        # Validate against gdptools' allowed stat methods. Listing the
        # commonly-used pair explicitly here so a typo is caught at adapter
        # construction rather than deep inside the aggregation loop.
        _ALLOWED_STAT_METHODS = {
            "mean",
            "masked_mean",
            "median",
            "masked_median",
            "std",
            "masked_std",
            "min",
            "masked_min",
            "max",
            "masked_max",
            "sum",
            "masked_sum",
            "count",
            "masked_count",
        }
        if self.stat_method not in _ALLOWED_STAT_METHODS:
            raise ValueError(
                f"SourceAdapter.stat_method={self.stat_method!r} is not a "
                f"gdptools STATSMETHODS value; expected one of "
                f"{sorted(_ALLOWED_STAT_METHODS)}"
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
        # raw_grid_variable names the pre-hook raw-NC variable used to detect
        # grid-shape invariance across source files. For non-hooked adapters it
        # defaults to grid_variable (which is itself a raw var). For adapters
        # whose pre_aggregate_hook synthesizes all declared variables from
        # raw inputs (e.g. MOD10C1), raw_grid_variable must be set explicitly
        # to a variable that exists in the raw NC; otherwise the driver cannot
        # enforce the cross-year grid invariant.
        if self.raw_grid_variable is None:
            object.__setattr__(self, "raw_grid_variable", self.grid_variable)
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
        # Validate source_crs parses cleanly so typos fail at construction time
        # rather than deep inside gdptools.
        try:
            from pyproj import CRS as _CRS

            _CRS.from_user_input(self.source_crs)
        except Exception as exc:
            raise ValueError(
                f"SourceAdapter.source_crs {self.source_crs!r} is not a valid "
                f"PROJ / EPSG input: {exc}"
            ) from exc
