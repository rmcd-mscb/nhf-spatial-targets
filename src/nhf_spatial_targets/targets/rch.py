"""Build recharge calibration targets from Reitz 2017 + WaterGAP 2.2d + ERA5-Land.

Three annual-cadence sources contribute to per-HRU per-year bounds in
dimensionless [0, 1] (per ``catalog/variables.yml`` → ``recharge.units``):

  - Reitz 2017 ``total_recharge``  (m/year, native annual; mid-year timestamp)
  - WaterGAP 2.2d ``qrdif``         (kg m-2 s-1, native monthly mean rate)
  - ERA5-Land ``ssro``              (m water-eq, native monthly accumulation;
                                    end-of-month timestamp)

Per-source pipeline (Option D from PR #132 review consider-2 — the shim
absorbs unit conversion AND any monthly→annual aggregation; rch's
``to_common_units`` is mm/year at year-start):

  - reitz2017: × 1000 → mm/year, time canonicalized to year-start
  - watergap22d: × (days_in_month × 86400) per month → mm/month, sum to mm/year
  - era5_land ssro: × 1000 per month → mm/month, sum to mm/year

Each source is then independently normalized to [0, 1] using its own
min/max over the configured ``normalize_period`` (TM 6-B10 §3 default:
2000-01-01 / 2009-12-31). The multi-source NaN-aware min/max combines
the three normalized series per HRU per year; the bound is NaN only when
every source is NaN at that (HRU, year), which is exactly when
``n_sources == 0``.

If ``recharge.nn_fill`` is True (default), a second file
``<output>_nn_filled.nc`` is written with bound NaNs filled by the
nearest finite HRU's value at the same time step.
"""

from __future__ import annotations

import logging

import pandas as pd
import xarray as xr

from nhf_spatial_targets.normalize.methods import normalize_0_1_over_window
from nhf_spatial_targets.targets._adapter import (
    SourceLoaderResult,
    TargetAdapter,
)
from nhf_spatial_targets.targets._combine import multi_source_nanminmax
from nhf_spatial_targets.targets._io import (
    check_hru_coords,
    parse_period,
    read_aggregated_source,
)
from nhf_spatial_targets.targets._shims import (
    SourceShim,
    shims_by_key,
)
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


_SECONDS_PER_DAY = 86400.0


# ---------------------------------------------------------------------------
# Per-source shims (mm/year at year-start is the common intermediate)
# ---------------------------------------------------------------------------


def reitz_to_mm_per_year(da: xr.DataArray) -> xr.DataArray:
    """Reitz 2017 ``total_recharge`` (m/year, mid-year timestamp) → mm/year.

    Time coord is canonicalized to year-start via a year-resampled mean
    (the mean is a no-op for native annual data with one timestamp per
    year; the resample shifts the mid-year label to year-start).
    """
    mm = da * 1000.0
    annual = mm.resample(time="YS").mean()
    annual.attrs = dict(da.attrs)
    annual.attrs["units"] = "mm"
    return annual


def watergap22d_to_mm_per_year(da: xr.DataArray) -> xr.DataArray:
    """WaterGAP 2.2d ``qrdif`` (kg m-2 s-1, monthly mean rate) → mm/year.

    Per month: × (days_in_month × 86400) → mm/month (kg/m² ≡ mm). Then
    sum 12 months per year → mm/year at year-start.
    """
    seconds_in_month = da["time"].dt.days_in_month * _SECONDS_PER_DAY
    monthly_mm = da * seconds_in_month
    annual = monthly_mm.resample(time="YS").sum(skipna=True)
    annual.attrs = dict(da.attrs)
    annual.attrs["units"] = "mm"
    return annual


def era5_ssro_to_mm_per_year(da: xr.DataArray) -> xr.DataArray:
    """ERA5-Land ``ssro`` (m water-eq, monthly accumulation) → mm/year.

    × 1000 per month → mm/month, then sum 12 months per year → mm/year
    at year-start. End-of-month timestamps from ERA5 land in the correct
    calendar year (Dec 31 ∈ year Y, not Y+1), so the resample groups
    them correctly.
    """
    monthly_mm = da * 1000.0
    annual = monthly_mm.resample(time="YS").sum(skipna=True)
    annual.attrs = dict(da.attrs)
    annual.attrs["units"] = "mm"
    return annual


SHIMS: tuple[SourceShim, ...] = (
    SourceShim(
        source_key="reitz2017",
        aggregated_var="total_recharge",
        description="Reitz 2017 total_recharge (m/year → mm/year)",
        to_common_units=reitz_to_mm_per_year,
    ),
    SourceShim(
        source_key="watergap22d",
        aggregated_var="qrdif",
        description=("WaterGAP 2.2d qrdif (kg/m²/s monthly rate, summed to mm/year)"),
        to_common_units=watergap22d_to_mm_per_year,
    ),
    SourceShim(
        source_key="era5_land",
        aggregated_var="ssro",
        description="ERA5-Land ssro (m/month, summed to mm/year)",
        to_common_units=era5_ssro_to_mm_per_year,
    ),
)


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
    rch_cfg = project.target(adapter.config_key)
    normalize_period = parse_period(rch_cfg["normalize_period"])
    sources = list(rch_cfg["sources"])

    logger.info(
        "Building recharge target: %d sources (%s), period %s..%s, "
        "normalize_period %s..%s, fabric=%s",
        len(sources),
        ",".join(sources),
        period[0],
        period[1],
        normalize_period[0],
        normalize_period[1],
        project.config["fabric"]["path"],
    )

    master_idx = pd.date_range(period[0], period[1], freq="YS")
    if len(master_idx) == 0:
        raise ValueError(
            f"recharge.period {rch_cfg['period']} produces no years at "
            "freq='YS'. Check the date range."
        )

    shims = shims_by_key(SHIMS)
    sources_normalized: dict[str, xr.DataArray] = {}
    for src in sources:
        if src not in shims:
            raise ValueError(
                f"recharge.sources includes unknown source '{src}'. "
                f"Known: {sorted(shims)}"
            )
        shim = shims[src]
        # Period for read: union of output period and normalize_period so we
        # have enough data to (a) emit on the output period and (b) compute
        # min/max over the normalize_period.
        read_start = min(period[0], normalize_period[0])
        read_end = max(period[1], normalize_period[1])
        da_native = read_aggregated_source(
            project,
            shim.source_key,
            shim.aggregated_var,
            (read_start, read_end),
            chunks={"time": 12, id_col: -1},
        )
        check_hru_coords(da_native, fabric_hru_ids, id_col, src)
        da_annual_mm = shim.to_common_units(da_native)
        window = da_annual_mm.sel(time=slice(normalize_period[0], normalize_period[1]))
        if window.sizes.get("time", 0) == 0:
            raise ValueError(
                f"recharge.normalize_period {rch_cfg['normalize_period']} "
                f"yields no annual timesteps for source '{src}'. "
                f"Source covers {da_annual_mm.time.values[0]} .. "
                f"{da_annual_mm.time.values[-1]}."
            )
        da_normalized = normalize_0_1_over_window(da_annual_mm, window)
        sources_normalized[src] = da_normalized.reindex(time=master_idx)

    lower, upper, n_sources = multi_source_nanminmax(sources_normalized)

    extra_attrs = {
        "source": "; ".join(shims[s].description for s in sources),
        "normalize_period": rch_cfg["normalize_period"],
    }

    return SourceLoaderResult(
        lower=lower,
        upper=upper,
        n_sources=n_sources,
        n_sources_count=len(sources),
        time_index=master_idx,
        time_offset_unit=pd.offsets.YearBegin(1),
        extra_attrs=extra_attrs,
    )


# ---------------------------------------------------------------------------
# Adapter declaration
# ---------------------------------------------------------------------------


ADAPTER = TargetAdapter(
    target_key="recharge",
    config_key="recharge",
    cadence="annual",
    bounds_units="1",
    bounds_long_name_kind="annual recharge",
    cell_methods="time: sum",
    title=("NHM recharge calibration target (lower/upper bounds, dimensionless 0-1)"),
    nn_title="NHM recharge calibration target (NN-filled, dimensionless 0-1)",
    source_loader=_load,
)


def build(project: Project) -> None:
    """Build the recharge calibration target.

    Thin wrapper around :func:`targets._driver.build` for the recharge
    :data:`ADAPTER`. The driver runs the read → unit-convert → normalize →
    NaN-aware min/max → write pipeline; this module owns the per-source
    unit shims (each ending at mm/year at year-start) and the
    per-source normalization step (each source independently normalized
    to [0, 1] over the configured ``recharge.normalize_period``).
    """
    from nhf_spatial_targets.targets._driver import build as run_driver

    run_driver(ADAPTER, project)
