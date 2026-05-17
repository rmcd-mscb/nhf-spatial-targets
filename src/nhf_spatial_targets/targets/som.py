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

import pandas as pd
import xarray as xr

from nhf_spatial_targets.normalize.methods import (
    normalize_0_1_by_calendar_month_over_window,
    normalize_0_1_over_window,
)
from nhf_spatial_targets.targets._common import (
    SourceShim,
    check_hru_coords,
    compute_hru_centroids,
    multi_source_nanminmax,
    parse_period,
    read_aggregated_source,
    reindex_to_month_start,
    shims_by_key,
    write_bounds_target,
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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _derive_variant_path(base_path, variant: str):
    """Insert '_<variant>' before the suffix of ``base_path``.

    Example: ``soil_moisture_targets.nc`` + ``monthly`` →
    ``soil_moisture_targets_monthly.nc``.
    """
    return base_path.with_name(base_path.stem + f"_{variant}" + base_path.suffix)


def build(project: Project) -> None:
    """Build the soil moisture calibration target (monthly + annual variants).

    Reads each enabled source's per-year aggregated NCs, harmonizes onto
    a monthly master index over ``soil_moisture.period``, then emits
    two CF-1.6 NetCDFs:

      - ``<output>_monthly.nc`` — per-calendar-month 0-1 normalization,
        multi-source min/max bound, monthly cadence.
      - ``<output>_annual.nc`` — monthly → annual mean per source,
        whole-period 0-1 normalization, year-start cadence.

    NN-filled companions are written for both when
    ``soil_moisture.nn_fill`` is True.
    """
    som_cfg = project.target("soil_moisture")
    period = parse_period(som_cfg["period"])
    # normalize_period defaults to the output period (whole-period
    # normalization). Set independently to extend the output past the
    # calibration-period climatology; values outside the window may then
    # produce normalized values < 0 or > 1, by design.
    raw_norm_period = som_cfg.get("normalize_period") or som_cfg["period"]
    normalize_period = parse_period(raw_norm_period)
    sources = list(som_cfg["sources"])

    logger.info(
        "Building SOM target: %d sources (%s), period %s..%s, "
        "normalize_period %s..%s, fabric=%s",
        len(sources),
        ",".join(sources),
        period[0],
        period[1],
        normalize_period[0],
        normalize_period[1],
        project.config["fabric"]["path"],
    )

    hru_meta = compute_hru_centroids(project)
    id_col = project.id_col

    master_monthly = pd.date_range(period[0], period[1], freq="MS")
    if len(master_monthly) == 0:
        raise ValueError(
            f"soil_moisture.period {som_cfg['period']} produces no months at "
            "freq='MS'. Check the date range."
        )

    # Read + canonicalize each source to monthly cadence at month-start.
    shims = shims_by_key(SHIMS)
    fabric_hru_ids = hru_meta.index.values
    sources_monthly: dict[str, xr.DataArray] = {}
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
            period,
            chunks={"time": 12, id_col: -1},
        )
        check_hru_coords(da_native, fabric_hru_ids, id_col, src)
        da_monthly_native = shim.to_common_units(da_native)
        sources_monthly[src] = reindex_to_month_start(da_monthly_native, master_monthly)

    # --- Monthly variant: per-calendar-month normalize over the window,
    # then combine. Window equals the full output period when
    # normalize_period == period (default).
    sources_monthly_norm: dict[str, xr.DataArray] = {}
    for src, da in sources_monthly.items():
        window = da.sel(time=slice(normalize_period[0], normalize_period[1]))
        if window.sizes.get("time", 0) == 0:
            raise ValueError(
                f"soil_moisture.normalize_period {raw_norm_period} yields no "
                f"monthly timesteps for source '{src}' (period intersection "
                f"with source's monthly index is empty)."
            )
        sources_monthly_norm[src] = normalize_0_1_by_calendar_month_over_window(
            da, window
        )
    lo_m, up_m, ns_m = multi_source_nanminmax(sources_monthly_norm)

    base_output = project.targets_dir() / som_cfg["output_file"]
    extra_attrs_monthly = {
        "source": "; ".join(shims[s].description for s in sources),
        "references": "Hay et al. 2022, doi:10.3133/tm6B10",
        "fabric": project.config["fabric"]["path"],
        "fabric_sha256": project.fabric.get("sha256", ""),
        "period": som_cfg["period"],
        "normalize_period": raw_norm_period,
        "normalize_method": "per_calendar_month",
        "area_crs": project.area_crs,
    }
    write_bounds_target(
        project=project,
        lower=lo_m,
        upper=up_m,
        n_sources=ns_m,
        n_sources_count=len(sources),
        time_index=master_monthly,
        time_offset_unit=pd.offsets.MonthBegin(1),
        bounds_units="1",
        bounds_long_name_kind="monthly soil moisture",
        cell_methods="time: mean",
        output_path=_derive_variant_path(base_output, "monthly"),
        title="NHM soil moisture monthly calibration target (dimensionless 0-1)",
        nn_title=(
            "NHM soil moisture monthly calibration target (NN-filled, "
            "dimensionless 0-1)"
        ),
        extra_global_attrs=extra_attrs_monthly,
        hru_meta=hru_meta,
        nn_fill=som_cfg["nn_fill"],
        nn_max_candidates=int(som_cfg["nn_max_candidates"]),
        id_col=id_col,
    )

    # --- Annual variant: monthly → annual mean per source, then normalize
    # over the annual slice of the normalize window.
    sources_annual = {
        src: da.resample(time="YS").mean(skipna=True)
        for src, da in sources_monthly.items()
    }
    sources_annual_norm: dict[str, xr.DataArray] = {}
    for src, da in sources_annual.items():
        window = da.sel(time=slice(normalize_period[0], normalize_period[1]))
        if window.sizes.get("time", 0) == 0:
            raise ValueError(
                f"soil_moisture.normalize_period {raw_norm_period} yields no "
                f"annual timesteps for source '{src}' after annual aggregation."
            )
        sources_annual_norm[src] = normalize_0_1_over_window(da, window)
    lo_a, up_a, ns_a = multi_source_nanminmax(sources_annual_norm)
    master_annual = pd.date_range(period[0], period[1], freq="YS")
    extra_attrs_annual = {
        "source": "; ".join(shims[s].description for s in sources),
        "references": "Hay et al. 2022, doi:10.3133/tm6B10",
        "fabric": project.config["fabric"]["path"],
        "fabric_sha256": project.fabric.get("sha256", ""),
        "period": som_cfg["period"],
        "normalize_period": raw_norm_period,
        "normalize_method": "whole_period",
        "annual_aggregation": "mean",
        "area_crs": project.area_crs,
    }
    write_bounds_target(
        project=project,
        lower=lo_a,
        upper=up_a,
        n_sources=ns_a,
        n_sources_count=len(sources),
        time_index=master_annual,
        time_offset_unit=pd.offsets.YearBegin(1),
        bounds_units="1",
        bounds_long_name_kind="annual soil moisture",
        cell_methods="time: mean",
        output_path=_derive_variant_path(base_output, "annual"),
        title="NHM soil moisture annual calibration target (dimensionless 0-1)",
        nn_title=(
            "NHM soil moisture annual calibration target (NN-filled, dimensionless 0-1)"
        ),
        extra_global_attrs=extra_attrs_annual,
        hru_meta=hru_meta,
        nn_fill=som_cfg["nn_fill"],
        nn_max_candidates=int(som_cfg["nn_max_candidates"]),
        id_col=id_col,
    )
