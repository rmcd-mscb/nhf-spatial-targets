"""Build soil moisture calibration targets from MERRA-2 + NCEP/NCAR + NLDAS-2.

Four monthly-cadence sources contribute to per-HRU per-time bounds in
dimensionless [0, 1]. Emits TWO target NCs:

  - Monthly: per-calendar-month normalization (all Januaries pooled,
    all Februaries pooled, etc.); time axis at month-start.
  - Annual: monthly → annual mean (state variable) per source, then
    single-period normalization over the full target period; time axis
    at year-start.

Per TM 6-B10 §4 Appendix 1: the per-calendar-month normalization for
the monthly variant removes seasonality so the bound reflects relative
wet/dry within each month rather than absolute differences between
January and July.

Sources (all native monthly; native units differ but the per-source
0-1 normalization cancels the offsets, so the SHIMS pass through values
unchanged — only time canonicalization happens via
``reindex_to_month_start``):

  - MERRA-2 ``GWETTOP`` (dimensionless 0-1 plant-available wetness;
    mid-month timestamp)
  - NCEP/NCAR ``soilw_0_10cm`` (m³/m³ VWC; end-of-month timestamp)
  - NLDAS-2 MOSAIC ``SoilM_0_10cm`` (kg/m² in 0-10 cm; month-start)
  - NLDAS-2 NOAH ``SoilM_0_10cm`` (kg/m² in 0-10 cm; month-start)

If ``soil_moisture.nn_fill`` is True (default), NN-filled companion
files are written for both the monthly and annual outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from nhf_spatial_targets.normalize.methods import (
    complete_years_window,
    normalize_0_1_by_calendar_month_over_window,
    normalize_0_1_over_window,
)
from nhf_spatial_targets.targets._adapter import (
    SourceLoaderResult,
    TargetAdapter,
)
from nhf_spatial_targets.targets._combine import multi_source_nanminmax
from nhf_spatial_targets.targets._io import (
    PER_SOURCE_POR,
    check_hru_coords,
    parse_period,
    read_aggregated_source,
    reindex_to_month_start,
)
from nhf_spatial_targets.targets._shims import (
    SourceShim,
    label_members,
    shims_by_key,
)
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


def som_passthrough(da: xr.DataArray) -> xr.DataArray:
    """SOM sources arrive at native monthly cadence; no unit conversion needed.

    Per recipes §4: the four sources carry incompatible native units
    (dimensionless plant-available wetness vs m³/m³ VWC vs kg/m² mass)
    but the per-source 0-1 normalization downstream cancels the constant
    offset. Cross-source unit harmonization is therefore cosmetic and
    omitted here. Time canonicalization to month-start is handled by
    ``reindex_to_month_start`` in the build loop, after this shim.
    """
    return da


SHIMS: tuple[SourceShim, ...] = (
    SourceShim(
        source_key="merra2",
        aggregated_var="GWETTOP",
        description=("MERRA-2 GWETTOP (dimensionless plant-available wetness, 0-5 cm)"),
        to_common_units=som_passthrough,
    ),
    SourceShim(
        source_key="ncep_ncar",
        aggregated_var="soilw_0_10cm",
        description="NCEP/NCAR soilw_0_10cm (m³/m³ VWC, 0-10 cm)",
        to_common_units=som_passthrough,
    ),
    SourceShim(
        source_key="nldas_mosaic",
        aggregated_var="SoilM_0_10cm",
        description="NLDAS-2 MOSAIC SoilM_0_10cm (kg/m², 0-10 cm)",
        to_common_units=som_passthrough,
    ),
    SourceShim(
        source_key="nldas_noah",
        aggregated_var="SoilM_0_10cm",
        description="NLDAS-2 NOAH SoilM_0_10cm (kg/m², 0-10 cm)",
        to_common_units=som_passthrough,
    ),
)


def _derive_variant_path(base_path: Path, variant: str) -> Path:
    """Insert '_<variant>' before the suffix of ``base_path``.

    Example: ``soil_moisture_targets.nc`` + ``monthly`` →
    ``soil_moisture_targets_monthly.nc``.
    """
    return base_path.with_name(base_path.stem + f"_{variant}" + base_path.suffix)


def _read_monthly_sources(
    *,
    project: Project,
    period: tuple[str, str],
    sources: list[str],
    fabric_hru_ids,
    id_col: str,
    master_monthly: pd.DatetimeIndex,
    read_period: tuple[str, str] | None = None,
) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]:
    """Read each source's monthly series.

    Returns ``(sources_monthly, sources_raw)``. ``sources_monthly`` is
    reindexed onto ``master_monthly`` (the output axis); ``sources_raw`` is
    the pre-reindex series exactly as read.

    ``read_period`` defaults to ``period``. Under ``per_source_por`` the
    caller passes a wide range so each source's whole record is present in
    ``sources_raw`` — the reindexed series cannot be used to judge record
    completeness, because it pads uncovered months with NaN at real
    timestamps and a distinct-month count then reads them as covered.
    """
    shims = shims_by_key(SHIMS)
    sources_monthly: dict[str, xr.DataArray] = {}
    sources_raw: dict[str, xr.DataArray] = {}
    for src in sources:
        if src not in shims:
            raise ValueError(
                f"soil_moisture.sources includes unknown source '{src}'. "
                f"Known: {sorted(shims)}"
            )
        shim = shims[src]
        da_native = read_aggregated_source(
            project,
            shim.source_key,
            shim.aggregated_var,
            read_period or period,
            chunks={"time": 12, id_col: -1},
        )
        check_hru_coords(da_native, fabric_hru_ids, id_col, src)
        da_monthly_native = shim.to_common_units(da_native)
        sources_raw[src] = da_monthly_native
        sources_monthly[src] = reindex_to_month_start(da_monthly_native, master_monthly)
    return sources_monthly, sources_raw


def _load_monthly(
    *,
    project: Project,
    adapter: TargetAdapter,
    period: tuple[str, str],
    hru_meta,
    fabric_hru_ids,
    id_col: str,
    year_context=None,
) -> SourceLoaderResult:
    """Monthly-variant loader: per-calendar-month normalize over window."""
    som_cfg = project.target(adapter.config_key)
    raw_norm_period = som_cfg.get("normalize_period") or som_cfg["period"]
    per_source_por = raw_norm_period == PER_SOURCE_POR
    normalize_period = None if per_source_por else parse_period(raw_norm_period)
    read_period = ("1900-01-01", "2200-12-31") if per_source_por else None
    sources = list(som_cfg["sources"])

    logger.info(
        "Building SOM monthly target: %d sources (%s), period %s..%s, "
        "normalize_period %s, fabric=%s",
        len(sources),
        ",".join(sources),
        period[0],
        period[1],
        raw_norm_period
        if per_source_por
        else f"{normalize_period[0]}..{normalize_period[1]}",
        project.config["fabric"]["path"],
    )

    master_monthly = pd.date_range(period[0], period[1], freq="MS")
    if len(master_monthly) == 0:
        raise ValueError(
            f"soil_moisture.period {som_cfg['period']} produces no months at "
            "freq='MS'. Check the date range."
        )

    _, sources_raw = _read_monthly_sources(
        project=project,
        period=period,
        sources=sources,
        fabric_hru_ids=fabric_hru_ids,
        id_col=id_col,
        master_monthly=master_monthly,
        read_period=read_period,
    )

    sources_monthly_norm: dict[str, xr.DataArray] = {}
    normalize_windows: dict[str, str] = {}
    for src, da_raw in sources_raw.items():
        if per_source_por:
            win_start, win_end = complete_years_window(da_raw, "monthly")
            normalize_windows[src] = f"{win_start}/{win_end}"
        else:
            win_start, win_end = normalize_period
        window = da_raw.sel(time=slice(win_start, win_end))
        if window.sizes.get("time", 0) == 0:
            raise ValueError(
                f"soil_moisture: source '{src}' has no monthly timesteps in "
                f"its normalization window {win_start}..{win_end}."
            )
        normed = normalize_0_1_by_calendar_month_over_window(da_raw, window)
        sources_monthly_norm[src] = reindex_to_month_start(normed, master_monthly)
    shims = shims_by_key(SHIMS)
    label_members(sources_monthly_norm, shims)
    lo_m, up_m, ns_m = multi_source_nanminmax(sources_monthly_norm)

    extra_attrs = {
        "source": "; ".join(shims[s].description for s in sources),
        "normalize_period": raw_norm_period,
        "normalize_method": "per_calendar_month",
    }
    for src, window_str in normalize_windows.items():
        extra_attrs[f"normalize_window_{src}"] = window_str
    return SourceLoaderResult(
        lower=lo_m,
        upper=up_m,
        n_sources=ns_m,
        n_sources_count=len(sources),
        time_index=master_monthly,
        time_offset_unit=pd.offsets.MonthBegin(1),
        extra_attrs=extra_attrs,
        members=sources_monthly_norm,
    )


def _load_annual(
    *,
    project: Project,
    adapter: TargetAdapter,
    period: tuple[str, str],
    hru_meta,
    fabric_hru_ids,
    id_col: str,
    year_context=None,
) -> SourceLoaderResult:
    """Annual-variant loader: monthly → annual mean, then whole-period normalize."""
    som_cfg = project.target(adapter.config_key)
    raw_norm_period = som_cfg.get("normalize_period") or som_cfg["period"]
    per_source_por = raw_norm_period == PER_SOURCE_POR
    normalize_period = None if per_source_por else parse_period(raw_norm_period)
    read_period = ("1900-01-01", "2200-12-31") if per_source_por else None
    sources = list(som_cfg["sources"])

    master_monthly = pd.date_range(period[0], period[1], freq="MS")
    master_annual = pd.date_range(period[0], period[1], freq="YS")
    _, sources_raw = _read_monthly_sources(
        project=project,
        period=period,
        sources=sources,
        fabric_hru_ids=fabric_hru_ids,
        id_col=id_col,
        master_monthly=master_monthly,
        read_period=read_period,
    )

    sources_annual_norm: dict[str, xr.DataArray] = {}
    normalize_windows: dict[str, str] = {}
    for src, da_raw in sources_raw.items():
        # The completeness window is derived from the RAW MONTHLY series,
        # not the resampled annual series: resample(time="YS") emits one
        # step for a partial year exactly as for a complete one, so
        # complete_years_window at "annual" cadence structurally cannot
        # detect an incomplete trailing/leading year (see its docstring).
        if per_source_por:
            win_start, win_end = complete_years_window(da_raw, "monthly")
            normalize_windows[src] = f"{win_start}/{win_end}"
        else:
            win_start, win_end = normalize_period
        annual = da_raw.resample(time="YS").mean(skipna=True)
        window = annual.sel(time=slice(win_start, win_end))
        if window.sizes.get("time", 0) == 0:
            raise ValueError(
                f"soil_moisture: source '{src}' has no annual timesteps in "
                f"its normalization window {win_start}..{win_end}."
            )
        normed = normalize_0_1_over_window(annual, window)
        sources_annual_norm[src] = normed.reindex(time=master_annual)
    shims = shims_by_key(SHIMS)
    label_members(sources_annual_norm, shims)
    lo_a, up_a, ns_a = multi_source_nanminmax(sources_annual_norm)

    extra_attrs = {
        "source": "; ".join(shims[s].description for s in sources),
        "normalize_period": raw_norm_period,
        "normalize_method": "whole_period",
        "annual_aggregation": "mean",
    }
    for src, window_str in normalize_windows.items():
        extra_attrs[f"normalize_window_{src}"] = window_str
    return SourceLoaderResult(
        lower=lo_a,
        upper=up_a,
        n_sources=ns_a,
        n_sources_count=len(sources),
        time_index=master_annual,
        time_offset_unit=pd.offsets.YearBegin(1),
        extra_attrs=extra_attrs,
        members=sources_annual_norm,
    )


# ---------------------------------------------------------------------------
# Two adapters (monthly + annual variants)
# ---------------------------------------------------------------------------


ADAPTER_MONTHLY = TargetAdapter(
    target_key="soil_moisture_monthly",
    config_key="soil_moisture",
    cadence="monthly",
    bounds_units="1",
    bounds_long_name_kind="monthly soil moisture",
    cell_methods="time: mean",
    title="NHM soil moisture monthly calibration target (dimensionless 0-1)",
    nn_title=(
        "NHM soil moisture monthly calibration target (NN-filled, dimensionless 0-1)"
    ),
    source_loader=_load_monthly,
)


ADAPTER_ANNUAL = TargetAdapter(
    target_key="soil_moisture_annual",
    config_key="soil_moisture",
    cadence="annual",
    bounds_units="1",
    bounds_long_name_kind="annual soil moisture",
    cell_methods="time: mean",
    title="NHM soil moisture annual calibration target (dimensionless 0-1)",
    nn_title=(
        "NHM soil moisture annual calibration target (NN-filled, dimensionless 0-1)"
    ),
    source_loader=_load_annual,
)


def build(project: Project) -> None:
    """Build the soil moisture calibration target (monthly + annual variants).

    Two outputs are written by running the generic driver twice — once
    per variant adapter (:data:`ADAPTER_MONTHLY`, :data:`ADAPTER_ANNUAL`).
    Each driver call materialises one ``soil_moisture_targets_<variant>.nc``
    file under the project targets dir; the variant suffix is inserted
    by overriding the driver's resolved output path via
    :func:`_derive_variant_path`. Multi-variant outputs are otherwise out
    of scope for the driver — declaring two adapters keeps the driver's
    "one adapter, one file" contract intact.
    """
    from nhf_spatial_targets.targets._driver import build_single_shot
    from nhf_spatial_targets.targets._io import (
        compute_hru_centroids,
        parse_period as _parse,
    )

    som_cfg = project.target("soil_moisture")
    period = _parse(som_cfg["period"])
    period_str = som_cfg["period"]
    hru_meta = compute_hru_centroids(project)
    id_col = project.id_col
    base_output = project.targets_dir() / som_cfg["output_file"]

    # Drive each variant manually so we can rewrite the output_file via
    # _derive_variant_path. The driver itself owns no notion of variants.
    for adapter, variant in (
        (ADAPTER_MONTHLY, "monthly"),
        (ADAPTER_ANNUAL, "annual"),
    ):
        variant_cfg = dict(som_cfg)
        variant_cfg["output_file"] = _derive_variant_path(base_output, variant).name
        build_single_shot(
            adapter=adapter,
            project=project,
            target_cfg=variant_cfg,
            period=period,
            period_str=period_str,
            hru_meta=hru_meta,
            id_col=id_col,
        )
