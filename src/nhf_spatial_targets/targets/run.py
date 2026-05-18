"""Build runoff calibration targets from ERA5-Land + GLDAS-2.1 NOAH + MWBM.

Three monthly sources contribute to per-HRU per-month bounds in cfs:

  - ERA5-Land ``ro``               (m water-equivalent / month)
  - GLDAS-2.1 NOAH ``runoff_total`` (kg/m², mean of 3-hourly accumulations;
                                    multiply by 8 × days_in_month for mm/month)
  - MWBM ClimGrid ``runoff``       (mm/month, native)

Per-source unit shims convert each to mm/month, ``mm_per_month_to_cfs`` then
converts to cfs using the per-HRU equal-area area and days-in-month. Sources
are stacked on a ``source`` dim and reduced with NaN-aware min/max so a bound
is defined whenever >=1 source is finite at that (HRU, time). An int8
``n_sources`` diagnostic is also written.

If ``runoff.nn_fill`` is True (default), a second file
``<output>_nn_filled.nc`` is written with bound NaNs filled by the nearest
finite HRU's value at the same time step (cKDTree donor walk in
``project.area_crs``).
"""

from __future__ import annotations

import logging

import pandas as pd
import xarray as xr

from nhf_spatial_targets.targets._common import (
    SourceShim,
    check_hru_coords,
    compute_hru_area_and_centroids,
    multi_source_nanminmax,
    parse_period,
    read_aggregated_source,
    reindex_to_month_start,
    shims_by_key,
    validate_source_units,
    write_bounds_target,
)
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


# 1 m³/day = (1/86400) m³/s = (35.3147/86400) ft³/s
_M3_PER_DAY_TO_CFS = 35.3146667 / 86400.0


# ---------------------------------------------------------------------------
# Per-source unit shims (mm/month is the common intermediate)
# ---------------------------------------------------------------------------


def era5_to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """ERA5-Land runoff (m water-eq / month) -> mm/month."""
    out = da * 1000.0
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


def gldas_to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """GLDAS ``runoff_total`` (kg m⁻²) → mm/month.

    ``runoff_total`` is the consolidation-time sum of ``Qs_acc + Qsb_acc``;
    this function does NOT perform the sum, it only converts the already-
    summed variable's units.

    GLDAS-2.1 ``_acc`` monthly values are the *mean* of 3-hourly
    accumulations (NOT a monthly sum). Per the NASA GES DISC GLDAS-2.1
    README, the monthly total is recovered via × 8 × days_in_month.
    1 kg/m² ≡ 1 mm depth.
    """
    days = da["time"].dt.days_in_month
    out = da * 8.0 * days
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


def mwbm_to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """MWBM ClimGrid runoff is already mm/month -- pass through."""
    out = da.copy()
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


# Per-source registry: (source_key, aggregated_var, description, to_mm shim).
# `shims_by_key(SHIMS)` is used at build time for O(1) lookup.
SHIMS: tuple[SourceShim, ...] = (
    SourceShim(
        source_key="era5_land",
        aggregated_var="ro",
        description="ERA5-Land ro",
        to_common_units=era5_to_mm_per_month,
        expected_cf_units="m",
    ),
    SourceShim(
        source_key="gldas_noah_v21_monthly",
        aggregated_var="runoff_total",
        description=(
            "GLDAS-2.1 NOAH runoff_total (Qs_acc + Qsb_acc, summed at consolidation)"
        ),
        to_common_units=gldas_to_mm_per_month,
        expected_cf_units="kg m-2",
    ),
    SourceShim(
        source_key="mwbm_climgrid",
        aggregated_var="runoff",
        description="MWBM ClimGrid runoff",
        to_common_units=mwbm_to_mm_per_month,
        expected_cf_units="mm",
    ),
)


def mm_per_month_to_cfs(da: xr.DataArray, hru_area_m2: xr.DataArray) -> xr.DataArray:
    """Convert mm/month -> cfs given per-HRU area and the month length."""
    days = da["time"].dt.days_in_month
    m_per_day = (da * 1e-3) / days
    m3_per_day = m_per_day * hru_area_m2
    cfs = m3_per_day * _M3_PER_DAY_TO_CFS
    cfs.attrs = dict(da.attrs)
    cfs.attrs["units"] = "cfs"
    return cfs


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build(project: Project) -> None:
    """Build the runoff calibration target.

    Reads each enabled source's per-year aggregated NCs, harmonizes time
    coords onto a master month-start index over ``runoff.period``,
    converts each to cfs using per-HRU area, combines via NaN-aware
    min/max, and writes a CF-1.6 NetCDF. If ``runoff.nn_fill`` is True,
    additionally writes ``runoff_targets_nn_filled.nc``.
    """
    runoff_cfg = project.target("runoff")
    period = parse_period(runoff_cfg["period"])
    sources = list(runoff_cfg["sources"])
    chunk_months = int(runoff_cfg["chunk_months"])

    # Fail loud at startup if a catalog cf_units string has drifted from
    # the per-source shim's hardcoded conversion factor (issue #130).
    validate_source_units(SHIMS, sources)

    logger.info(
        "Building runoff target: %d sources (%s), period %s .. %s, fabric=%s",
        len(sources),
        ",".join(sources),
        period[0],
        period[1],
        project.config["fabric"]["path"],
    )

    # 1. Per-HRU area + centroids (computed once from fabric).
    hru_meta = compute_hru_area_and_centroids(project)
    id_col = project.id_col
    hru_area_da = xr.DataArray(
        hru_meta["area_m2"].values,
        dims=(id_col,),
        coords={id_col: hru_meta.index.values},
        name="area_m2",
    )

    # 2. Master month-start index over the requested period.
    master_idx = pd.date_range(period[0], period[1], freq="MS")
    if len(master_idx) == 0:
        raise ValueError(
            f"runoff.period {runoff_cfg['period']} produces no months at "
            f"freq='MS'. Check the date range."
        )

    # 3. Read, convert, reindex each source.
    # Fabric HRU IDs as a numpy array for coord-agreement checks.
    fabric_hru_ids = hru_meta.index.values

    shims = shims_by_key(SHIMS)
    sources_cfs: dict[str, xr.DataArray] = {}
    for src in sources:
        shim = shims[src]
        da_native = read_aggregated_source(
            project,
            shim.source_key,
            shim.aggregated_var,
            period,
            chunks={"time": chunk_months, id_col: -1},
        )
        check_hru_coords(da_native, fabric_hru_ids, id_col, src)
        da_mm = shim.to_common_units(da_native)
        da_cfs = mm_per_month_to_cfs(da_mm, hru_area_da)
        sources_cfs[src] = reindex_to_month_start(da_cfs, master_idx)

    # 4. NaN-aware combination across sources.
    lower, upper, n_sources = multi_source_nanminmax(sources_cfs)

    # 5. Assemble + write (with optional NN-fill companion).
    extra_attrs = {
        "source": "; ".join(shims[s].description for s in sources),
        "references": "Hay et al. 2022, doi:10.3133/tm6B10",
        "fabric": project.config["fabric"]["path"],
        "fabric_sha256": project.fabric.get("sha256", ""),
        "period": runoff_cfg["period"],
        "area_crs": project.area_crs,
    }
    output_path = project.targets_dir() / runoff_cfg["output_file"]
    write_bounds_target(
        project=project,
        lower=lower,
        upper=upper,
        n_sources=n_sources,
        n_sources_count=len(sources),
        time_index=master_idx,
        time_offset_unit=pd.offsets.MonthBegin(1),
        bounds_units="cfs",
        bounds_long_name_kind="monthly runoff",
        cell_methods="time: sum",
        output_path=output_path,
        title="NHM runoff calibration target (lower/upper bounds in cfs)",
        nn_title="NHM runoff calibration target (NN-filled, cfs)",
        extra_global_attrs=extra_attrs,
        hru_meta=hru_meta,
        nn_fill=runoff_cfg["nn_fill"],
        nn_max_candidates=int(runoff_cfg["nn_max_candidates"]),
        id_col=id_col,
    )
