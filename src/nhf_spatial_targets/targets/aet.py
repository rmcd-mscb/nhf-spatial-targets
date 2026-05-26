"""Build AET calibration targets from MOD16A2 v061 + SSEBop + MWBM.

Three monthly-cadence sources contribute to per-HRU per-month bounds in
``inches/day`` (per ``catalog/variables.yml`` → ``aet.units``):

  - SSEBop ``et``                  (mm/month, native)
  - MWBM ClimGrid ``aet``          (mm/month, native)
  - MOD16A2 v061 ``ET_500m``       (kg m⁻² per 8-day composite, decoded)

MOD16A2 is the only source that arrives at non-monthly cadence: 46 8-day
composites per year, capped at the year boundary. ``_mod16a2_to_monthly_mm``
resamples to calendar months via overlap-weighted summation — the same
recipe used in ``notebooks/aggregated/inspect_aggregated_aet.ipynb`` and
documented in ``docs/references/calibration-target-recipes.md`` §2.

After per-source unit harmonization to mm/month and conversion to
inches/day, sources are stacked on a ``source`` dim and reduced with
NaN-aware min/max so a bound is defined whenever ≥1 source is finite at
that (HRU, month). An int8 ``n_sources`` diagnostic is also written.

If ``aet.nn_fill`` is True (default), a second file
``<output>_nn_filled.nc`` is written with bound NaNs filled by the
nearest finite HRU's value at the same time step (cKDTree donor walk in
``project.area_crs``).

The ``aet.sources`` config key controls which sources contribute: this
exists so the open question about MOD16A2 v061's flat-on-CONUS+
seasonality (recipes §2) can be resolved by config change rather than
code edit.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.targets._adapter import (
    SourceLoaderResult,
    TargetAdapter,
)
from nhf_spatial_targets.targets._combine import multi_source_nanminmax
from nhf_spatial_targets.targets._io import (
    check_hru_coords,
    read_aggregated_source,
    reindex_to_month_start,
)
from nhf_spatial_targets.targets._shims import (
    SourceShim,
    shims_by_key,
    validate_source_units,
)
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


# 1 inch = 25.4 mm.
_MM_PER_INCH = 25.4


# ---------------------------------------------------------------------------
# Per-source unit shims (mm/month is the common intermediate)
# ---------------------------------------------------------------------------


def ssebop_to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """SSEBop ``et`` is already mm/month — pass through with attr cleanup."""
    out = da.copy()
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


def mwbm_to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """MWBM ClimGrid ``aet`` is already mm/month — pass through."""
    out = da.copy()
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


def mod16a2_to_mm_per_month(
    da: xr.DataArray, *, composite_days: int = 8
) -> xr.DataArray:
    """Resample MOD16A2 8-day kg m⁻² composites to calendar-month mm totals.

    For each calendar month spanned by ``da.time``, sum overlap-weighted
    contributions of every composite that intersects the month::

        month_mm = Σ_c (composite_c × overlap_days_c / composite_length_c)

    The standard composite covers 8 days; the year-end composite (DOY 361)
    is capped at the next calendar year's Jan 1 by LP DAAC and covers 5-6
    days — the cap prevents day-1-of-Jan double counting between Dec 26's
    nominal 8-day window and Jan 1's actual composite.

    The aggregated NCs store raw int-like values with ``scale_factor=0.1``
    in attrs; xarray's default ``decode_cf=True`` applies the scale on read,
    so the values arriving here are already in scaled kg m⁻² (= mm) per
    composite. **Do not multiply by 0.1 again** — earlier versions of this
    helper did, which produced figures 10× too low (PR #88).

    Output ``time`` coord is month-start (``freq='MS'``) for the months
    fully spanned by the input composites.
    """
    if "time" not in da.dims:
        raise ValueError(
            f"mod16a2_to_mm_per_month: expected 'time' dim, got {tuple(da.dims)!r}."
        )
    if int(da.sizes["time"]) == 0:
        raise ValueError(
            "mod16a2_to_mm_per_month: input has no time steps; "
            "aggregator output may be empty."
        )

    starts = pd.DatetimeIndex(da.time.values)
    year_ends = pd.DatetimeIndex(
        [pd.Timestamp(year=t.year + 1, month=1, day=1) for t in starts]
    )
    nominal_ends = starts + pd.Timedelta(days=composite_days)
    ends = pd.DatetimeIndex(np.minimum(nominal_ends.values, year_ends.values))
    composite_lengths = (ends - starts).days.to_numpy().astype(float)

    first_month = pd.Timestamp(year=starts[0].year, month=starts[0].month, day=1)
    last_dt = ends[-1] - pd.Timedelta(seconds=1)
    last_month = pd.Timestamp(year=last_dt.year, month=last_dt.month, day=1)
    candidate_starts = pd.date_range(first_month, last_month, freq="MS")
    candidate_ends = candidate_starts + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)

    full = (candidate_starts >= starts[0]) & (candidate_ends <= ends[-1])
    month_starts = candidate_starts[full]
    month_ends = candidate_ends[full]
    if len(month_starts) == 0:
        raise ValueError(
            "mod16a2_to_mm_per_month: no calendar month is fully covered by "
            "the input composites — aggregator output is too short to resample."
        )

    overlap_starts = np.maximum(starts.values[None, :], month_starts.values[:, None])
    overlap_ends = np.minimum(ends.values[None, :], month_ends.values[:, None])
    overlap_days = np.clip(
        (overlap_ends - overlap_starts) / np.timedelta64(1, "D"),
        0,
        None,
    )
    weights = overlap_days / composite_lengths[None, :]

    weight_da = xr.DataArray(
        weights,
        coords={"month": month_starts, "time": starts},
        dims=["month", "time"],
    )
    monthly = xr.dot(weight_da, da, dim="time").rename({"month": "time"})
    monthly.attrs = dict(da.attrs)
    monthly.attrs["units"] = "mm"
    return monthly


# Per-source registry: (source_key, aggregated_var, description, to_mm shim).
# `shims_by_key(SHIMS)` is used at build time for O(1) lookup.
SHIMS: tuple[SourceShim, ...] = (
    SourceShim(
        source_key="mod16a2_v061",
        aggregated_var="ET_500m",
        description=(
            "MOD16A2 v061 ET_500m (8-day kg m-2; overlap-weighted to mm/month)"
        ),
        to_common_units=mod16a2_to_mm_per_month,
        expected_cf_units="kg m-2",
    ),
    SourceShim(
        source_key="ssebop",
        aggregated_var="et",
        description="SSEBop et (mm/month, native)",
        to_common_units=ssebop_to_mm_per_month,
        expected_cf_units="mm",
    ),
    SourceShim(
        source_key="mwbm_climgrid",
        aggregated_var="aet",
        description="MWBM ClimGrid aet (mm/month, native)",
        to_common_units=mwbm_to_mm_per_month,
        expected_cf_units="mm",
    ),
)


def mm_per_month_to_inches_per_day(da: xr.DataArray) -> xr.DataArray:
    """Convert mm/month → inches/day using each timestamp's days_in_month."""
    days = da["time"].dt.days_in_month
    out = (da / _MM_PER_INCH) / days
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "inches/day"
    return out


# ---------------------------------------------------------------------------
# Source loader
# ---------------------------------------------------------------------------


def _load(
    *,
    project: Project,
    adapter: TargetAdapter,
    period: tuple[str, str],
    hru_meta,
    fabric_hru_ids,
    id_col: str,
    year_context=None,
) -> SourceLoaderResult:
    aet_cfg = project.target(adapter.config_key)
    sources = list(aet_cfg["sources"])
    chunk_months = int(aet_cfg["chunk_months"])

    validate_source_units(SHIMS, sources)

    logger.info(
        "Building AET target: %d sources (%s), period %s .. %s, fabric=%s",
        len(sources),
        ",".join(sources),
        period[0],
        period[1],
        project.config["fabric"]["path"],
    )

    master_idx = pd.date_range(period[0], period[1], freq="MS")
    if len(master_idx) == 0:
        raise ValueError(
            f"aet.period {aet_cfg['period']} produces no months at freq='MS'. "
            "Check the date range."
        )

    shims = shims_by_key(SHIMS)
    sources_in_day: dict[str, xr.DataArray] = {}
    for src in sources:
        shim = shims[src]
        # For MOD16A2 the time dim is 8-day; slicing by the requested
        # monthly period is still correct because xr.sel(time=slice(...))
        # is a half-open inclusive-of-both-endpoints date range.
        da_native = read_aggregated_source(
            project,
            shim.source_key,
            shim.aggregated_var,
            period,
            chunks={"time": chunk_months, id_col: -1},
        )
        check_hru_coords(da_native, fabric_hru_ids, id_col, src)
        da_mm = shim.to_common_units(da_native)
        da_in_day = mm_per_month_to_inches_per_day(da_mm)
        sources_in_day[src] = reindex_to_month_start(da_in_day, master_idx)

    lower, upper, n_sources = multi_source_nanminmax(sources_in_day)

    extra_attrs = {
        "source": "; ".join(shims[s].description for s in sources),
    }

    return SourceLoaderResult(
        lower=lower,
        upper=upper,
        n_sources=n_sources,
        n_sources_count=len(sources),
        time_index=master_idx,
        time_offset_unit=pd.offsets.MonthBegin(1),
        extra_attrs=extra_attrs,
    )


# ---------------------------------------------------------------------------
# Adapter declaration
# ---------------------------------------------------------------------------


ADAPTER = TargetAdapter(
    target_key="aet",
    config_key="aet",
    cadence="monthly",
    bounds_units="inches/day",
    bounds_long_name_kind="monthly AET",
    cell_methods="time: mean",
    title="NHM AET calibration target (lower/upper bounds in inches/day)",
    nn_title="NHM AET calibration target (NN-filled, inches/day)",
    source_loader=_load,
)


def build(project: Project) -> None:
    """Build the AET calibration target.

    Thin wrapper around :func:`targets._driver.build` for the AET
    :data:`ADAPTER`. The driver runs the read → unit-convert → NaN-aware
    min/max → write pipeline; this module owns the per-source unit shims
    (the MOD16A2 8-day → mm/month resampler, then mm/month → inches/day).
    """
    from nhf_spatial_targets.targets._driver import build as run_driver

    run_driver(ADAPTER, project)
