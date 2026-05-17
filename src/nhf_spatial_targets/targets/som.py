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

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.normalize.methods import (
    nn_fill_bounds,
    normalize_0_1,
    normalize_0_1_by_calendar_month,
)
from nhf_spatial_targets.targets._common import (
    SourceShim,
    compute_hru_centroids,
    multi_source_nanminmax,
    read_aggregated_source,
    reindex_to_month_start,
    shims_by_key,
    write_target_nc,
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


def _parse_period(period_str: str) -> tuple[str, str]:
    """Parse 'YYYY-MM-DD/YYYY-MM-DD' (or 'YYYY/YYYY') into (start, end)."""
    if "/" not in period_str:
        raise ValueError(
            f"Invalid period {period_str!r}. Expected 'YYYY-MM-DD/YYYY-MM-DD'."
        )
    start, end = period_str.split("/", 1)
    return start.strip(), end.strip()


def _derive_variant_path(base_path, variant: str):
    """Insert '_<variant>' before the suffix of ``base_path``.

    Example: ``soil_moisture_targets.nc`` + ``monthly`` →
    ``soil_moisture_targets_monthly.nc``.
    """
    return base_path.with_name(base_path.stem + f"_{variant}" + base_path.suffix)


def _hru_coord_check(da_native, fabric_hru_ids, id_col: str, src: str) -> None:
    """Same canonical-sort invariant as runoff/aet/rch."""
    src_hru_ids = da_native[id_col].values
    if np.array_equal(src_hru_ids, fabric_hru_ids):
        return
    same_set = len(src_hru_ids) == len(fabric_hru_ids) and np.array_equal(
        np.sort(src_hru_ids), np.sort(fabric_hru_ids)
    )
    if same_set:
        raise ValueError(
            f"HRU coords for source '{src}' have the same set as the "
            f"fabric ({len(fabric_hru_ids)} HRUs) but a different "
            f"order. Both sides are expected to be sorted ascending by "
            f"id_col='{id_col}' — this indicates a regression in the "
            f"canonical-sort invariant in targets/_common.py."
        )
    raise ValueError(
        f"HRU coords differ between fabric and source '{src}' as "
        f"sets. Fabric has {len(fabric_hru_ids)} HRUs "
        f"(first={fabric_hru_ids[0]}, last={fabric_hru_ids[-1]}); "
        f"source has {len(src_hru_ids)} "
        f"(first={src_hru_ids[0]}, last={src_hru_ids[-1]}). "
        f"Re-aggregate '{src}' against the current fabric."
    )


def _build_diag_attrs(n_sources: int, ancillary_coords: str) -> dict:
    return {
        "units": "1",
        "long_name": "number of finite source contributions",
        "flag_values": list(range(0, n_sources + 1)),
        "flag_meanings": " ".join(
            ["none", "one", "two", "three", "four", "five"][: n_sources + 1]
        ),
        "coordinates": ancillary_coords,
    }


def _assemble_and_write(
    *,
    project: Project,
    lower: xr.DataArray,
    upper: xr.DataArray,
    n_sources: xr.DataArray,
    n_sources_count: int,
    time_index: pd.DatetimeIndex,
    time_offset_unit: pd.offsets.BaseOffset,
    long_name_suffix: str,
    cell_methods: str,
    output_path,
    title: str,
    extra_global_attrs: dict,
    hru_meta,
    nn_fill: bool,
    nn_max_candidates: int,
    nn_title: str,
    id_col: str,
) -> None:
    """Wrap up a (lower, upper, n_sources) trio into a NC file + optional NN-fill.

    Common code for both the monthly and annual SOM variants — the only
    things that differ are the time index, the variable long_names, the
    cell_methods string, the output paths, and the dataset titles.
    """
    lower.name = "lower_bound"
    upper.name = "upper_bound"
    n_sources.name = "n_sources"

    time_bnds = xr.DataArray(
        list(zip(time_index.values, (time_index + time_offset_unit).values)),
        dims=("time", "nv"),
        coords={"time": time_index.values},
        name="time_bnds",
    )
    centroid_lat = xr.DataArray(
        hru_meta["centroid_lat"].values,
        dims=(id_col,),
        coords={id_col: hru_meta.index.values},
        attrs={
            "units": "degrees_north",
            "standard_name": "latitude",
            "long_name": "HRU centroid latitude",
        },
    )
    centroid_lon = xr.DataArray(
        hru_meta["centroid_lon"].values,
        dims=(id_col,),
        coords={id_col: hru_meta.index.values},
        attrs={
            "units": "degrees_east",
            "standard_name": "longitude",
            "long_name": "HRU centroid longitude",
        },
    )

    lower.attrs.update(
        {
            "units": "1",
            "long_name": (
                f"lower bound of {long_name_suffix} "
                f"(NaN-aware min across normalized sources)"
            ),
            "cell_methods": cell_methods,
            "coordinates": "centroid_lat centroid_lon",
        }
    )
    upper.attrs.update(
        {
            "units": "1",
            "long_name": (
                f"upper bound of {long_name_suffix} "
                f"(NaN-aware max across normalized sources)"
            ),
            "cell_methods": cell_methods,
            "coordinates": "centroid_lat centroid_lon",
        }
    )
    n_sources.attrs.update(
        _build_diag_attrs(
            n_sources=n_sources_count,
            ancillary_coords="centroid_lat centroid_lon",
        )
    )

    ds = xr.Dataset(
        {
            "lower_bound": lower,
            "upper_bound": upper,
            "n_sources": n_sources,
        },
        coords={
            "time": time_index,
            id_col: lower[id_col],
            "time_bnds": time_bnds,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
        },
    )
    ds["time"].attrs["bounds"] = "time_bnds"
    ds["time"].attrs["axis"] = "T"
    ds["time"].attrs["standard_name"] = "time"
    ds[id_col].attrs["long_name"] = "HRU identifier"
    ds[id_col].attrs["cf_role"] = "timeseries_id"

    ds_loaded = ds.compute()

    write_target_nc(
        ds_loaded,
        output_path,
        title=title,
        extra_global_attrs=extra_global_attrs,
        sort_dim=id_col,
    )

    n = ds_loaded["n_sources"].values
    total = n.size
    none = int((n == 0).sum())
    logger.info(
        "%s coverage: %d/%d cells have >=1 finite source (%.2f%% all-NaN)",
        long_name_suffix,
        total - none,
        total,
        100.0 * none / total if total else 0.0,
    )

    if nn_fill:
        centroids_xy = hru_meta[["centroid_x", "centroid_y"]].values
        filled_ds, nn_diag = nn_fill_bounds(
            ds_loaded, centroids_xy, max_candidates=nn_max_candidates
        )
        nn_diag.attrs.update(
            {
                "units": "1",
                "long_name": "nearest-neighbor fill flag",
                "flag_values": [0, 1],
                "flag_meanings": "not_filled filled",
                "coordinates": "centroid_lat centroid_lon",
            }
        )
        filled_ds["nn_filled"] = nn_diag
        filled_attrs = dict(extra_global_attrs)
        filled_attrs["nn_fill_max_candidates"] = nn_max_candidates
        filled_attrs["nn_fill_distance_crs"] = project.area_crs
        nn_path = output_path.with_name(
            output_path.stem + "_nn_filled" + output_path.suffix
        )
        write_target_nc(
            filled_ds,
            nn_path,
            title=nn_title,
            extra_global_attrs=filled_attrs,
            sort_dim=id_col,
        )


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
    period = _parse_period(som_cfg["period"])
    sources = list(som_cfg["sources"])

    logger.info(
        "Building SOM target: %d sources (%s), period %s..%s, fabric=%s",
        len(sources),
        ",".join(sources),
        period[0],
        period[1],
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
        _hru_coord_check(da_native, fabric_hru_ids, id_col, src)
        da_monthly_native = shim.to_common_units(da_native)
        sources_monthly[src] = reindex_to_month_start(da_monthly_native, master_monthly)

    # --- Monthly variant: per-calendar-month normalize, then combine.
    sources_monthly_norm = {
        src: normalize_0_1_by_calendar_month(da) for src, da in sources_monthly.items()
    }
    lo_m, up_m, ns_m = multi_source_nanminmax(sources_monthly_norm)

    base_output = project.targets_dir() / som_cfg["output_file"]
    extra_attrs_monthly = {
        "source": "; ".join(shims[s].description for s in sources),
        "references": "Hay et al. 2022, doi:10.3133/tm6B10",
        "fabric": project.config["fabric"]["path"],
        "fabric_sha256": project.fabric.get("sha256", ""),
        "period": som_cfg["period"],
        "normalize_method": "per_calendar_month",
        "area_crs": project.area_crs,
    }
    _assemble_and_write(
        project=project,
        lower=lo_m,
        upper=up_m,
        n_sources=ns_m,
        n_sources_count=len(sources),
        time_index=master_monthly,
        time_offset_unit=pd.offsets.MonthBegin(1),
        long_name_suffix="monthly soil moisture",
        cell_methods="time: mean",
        output_path=_derive_variant_path(base_output, "monthly"),
        title="NHM soil moisture monthly calibration target (dimensionless 0-1)",
        extra_global_attrs=extra_attrs_monthly,
        hru_meta=hru_meta,
        nn_fill=som_cfg["nn_fill"],
        nn_max_candidates=int(som_cfg["nn_max_candidates"]),
        nn_title=(
            "NHM soil moisture monthly calibration target (NN-filled, "
            "dimensionless 0-1)"
        ),
        id_col=id_col,
    )

    # --- Annual variant: monthly → annual mean per source, then normalize.
    sources_annual = {
        src: da.resample(time="YS").mean(skipna=True)
        for src, da in sources_monthly.items()
    }
    sources_annual_norm = {
        src: normalize_0_1(da, dim="time") for src, da in sources_annual.items()
    }
    lo_a, up_a, ns_a = multi_source_nanminmax(sources_annual_norm)
    master_annual = pd.date_range(period[0], period[1], freq="YS")
    extra_attrs_annual = {
        "source": "; ".join(shims[s].description for s in sources),
        "references": "Hay et al. 2022, doi:10.3133/tm6B10",
        "fabric": project.config["fabric"]["path"],
        "fabric_sha256": project.fabric.get("sha256", ""),
        "period": som_cfg["period"],
        "normalize_method": "whole_period",
        "annual_aggregation": "mean",
        "area_crs": project.area_crs,
    }
    _assemble_and_write(
        project=project,
        lower=lo_a,
        upper=up_a,
        n_sources=ns_a,
        n_sources_count=len(sources),
        time_index=master_annual,
        time_offset_unit=pd.offsets.YearBegin(1),
        long_name_suffix="annual soil moisture",
        cell_methods="time: mean",
        output_path=_derive_variant_path(base_output, "annual"),
        title="NHM soil moisture annual calibration target (dimensionless 0-1)",
        extra_global_attrs=extra_attrs_annual,
        hru_meta=hru_meta,
        nn_fill=som_cfg["nn_fill"],
        nn_max_candidates=int(som_cfg["nn_max_candidates"]),
        nn_title=(
            "NHM soil moisture annual calibration target (NN-filled, dimensionless 0-1)"
        ),
        id_col=id_col,
    )
