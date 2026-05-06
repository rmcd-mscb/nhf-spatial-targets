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
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.normalize.methods import nn_fill_bounds
from nhf_spatial_targets.targets._common import (
    compute_hru_area_and_centroids,
    multi_source_nanminmax,
    read_aggregated_source,
    reindex_to_month_start,
    write_target_nc,
)
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


# 1 m³/day = (1/86400) m³/s = (35.3147/86400) ft³/s
_M3_PER_DAY_TO_CFS = 35.3146667 / 86400.0

# Per-source variable name in the aggregated NC.
_SOURCE_VAR: dict[str, str] = {
    "era5_land": "ro",
    "gldas_noah_v21_monthly": "runoff_total",
    "mwbm_climgrid": "runoff",
}


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
    """GLDAS Qs_acc + Qsb_acc -> mm/month.

    GLDAS-2.1 ``_acc`` monthly values are the *mean* of 3-hourly
    accumulations (NOT a monthly sum). Per the NASA GES DISC GLDAS-2.1
    README, the monthly total is recovered via x 8 x days_in_month.
    1 kg/m^2 = 1 mm depth.
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


_TO_MM: dict[str, Callable[[xr.DataArray], xr.DataArray]] = {
    "era5_land": era5_to_mm_per_month,
    "gldas_noah_v21_monthly": gldas_to_mm_per_month,
    "mwbm_climgrid": mwbm_to_mm_per_month,
}


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


def _parse_period(period_str: str) -> tuple[str, str]:
    """Parse 'YYYY-MM-DD/YYYY-MM-DD' (or 'YYYY/YYYY') into (start, end)."""
    if "/" not in period_str:
        raise ValueError(
            f"Invalid period {period_str!r}. Expected 'YYYY-MM-DD/YYYY-MM-DD'."
        )
    start, end = period_str.split("/", 1)
    return start.strip(), end.strip()


def build(project: Project) -> None:
    """Build the runoff calibration target.

    Reads each enabled source's per-year aggregated NCs, harmonizes time
    coords onto a master month-start index over ``runoff.period``,
    converts each to cfs using per-HRU area, combines via NaN-aware
    min/max, and writes a CF-1.6 NetCDF. If ``runoff.nn_fill`` is True,
    additionally writes ``runoff_targets_nn_filled.nc``.
    """
    runoff_cfg = project.target("runoff")
    period = _parse_period(runoff_cfg["period"])
    sources = list(runoff_cfg["sources"])
    chunk_months = int(runoff_cfg["chunk_months"])

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

    sources_cfs: dict[str, xr.DataArray] = {}
    for src in sources:
        if src not in _SOURCE_VAR:
            raise ValueError(
                f"runoff.sources includes unknown source '{src}'. "
                f"Known: {sorted(_SOURCE_VAR.keys())}"
            )
        var = _SOURCE_VAR[src]
        da_native = read_aggregated_source(
            project, src, var, period, chunks={"time": chunk_months, id_col: -1}
        )
        # Validate that the source's HRU IDs match the fabric before any
        # arithmetic that would silently broadcast/intersect mismatched coords.
        src_hru_ids = da_native[id_col].values
        if not np.array_equal(src_hru_ids, fabric_hru_ids):
            raise ValueError(
                f"HRU coords differ between fabric and source '{src}'. "
                f"Fabric has {len(fabric_hru_ids)} HRUs "
                f"({fabric_hru_ids[0]}..{fabric_hru_ids[-1]}); source has "
                f"{len(src_hru_ids)}. Re-aggregate '{src}' against the "
                f"current fabric."
            )
        da_mm = _TO_MM[src](da_native)
        da_cfs = mm_per_month_to_cfs(da_mm, hru_area_da)
        sources_cfs[src] = reindex_to_month_start(da_cfs, master_idx)

    # 4. NaN-aware combination.
    lower, upper, n_sources = multi_source_nanminmax(sources_cfs)
    lower.name = "lower_bound"
    upper.name = "upper_bound"
    n_sources.name = "n_sources"

    # 5. Assemble Dataset with CF metadata + ancillary coords.
    time_bnds = xr.DataArray(
        list(zip(master_idx.values, (master_idx + pd.offsets.MonthBegin(1)).values)),
        dims=("time", "nv"),
        coords={"time": master_idx.values},
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
            "units": "cfs",
            "long_name": (
                "lower bound of monthly runoff (NaN-aware min across sources)"
            ),
            "cell_methods": "time: sum",
            "coordinates": "centroid_lat centroid_lon",
        }
    )
    upper.attrs.update(
        {
            "units": "cfs",
            "long_name": (
                "upper bound of monthly runoff (NaN-aware max across sources)"
            ),
            "cell_methods": "time: sum",
            "coordinates": "centroid_lat centroid_lon",
        }
    )
    n_sources.attrs.update(
        {
            "units": "1",
            "long_name": "number of finite source contributions",
            "flag_values": list(range(0, len(sources) + 1)),
            "flag_meanings": " ".join(
                ["none", "one", "two", "three", "four", "five"][: len(sources) + 1]
            ),
            "coordinates": "centroid_lat centroid_lon",
        }
    )

    ds = xr.Dataset(
        {
            "lower_bound": lower,
            "upper_bound": upper,
            "n_sources": n_sources,
        },
        coords={
            "time": master_idx,
            id_col: lower[id_col],
            "time_bnds": time_bnds,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
        },
    )
    ds["time"].attrs["bounds"] = "time_bnds"
    ds["time"].attrs["axis"] = "T"
    ds["time"].attrs["standard_name"] = "time"
    ds["time"].attrs["long_name"] = "time at month start"
    ds[id_col].attrs["long_name"] = "HRU identifier"
    ds[id_col].attrs["cf_role"] = "timeseries_id"

    extra_attrs = {
        "source": ("ERA5-Land ro; GLDAS-2.1 NOAH Qs_acc+Qsb_acc; MWBM ClimGrid runoff"),
        "references": "Hay et al. 2022, doi:10.3133/tm6B10",
        "fabric": project.config["fabric"]["path"],
        "fabric_sha256": project.fabric.get("sha256", ""),
        "period": runoff_cfg["period"],
        "area_crs": project.area_crs,
    }

    output_path = project.targets_dir() / runoff_cfg["output_file"]
    write_target_nc(
        ds,
        output_path,
        title="NHM runoff calibration target (lower/upper bounds in cfs)",
        extra_global_attrs=extra_attrs,
    )

    # Coverage summary log line.
    n = ds["n_sources"].values
    total = n.size
    none = int((n == 0).sum())
    logger.info(
        "Coverage: %d/%d cells have >=1 finite source (%.2f%% all-NaN)",
        total - none,
        total,
        100.0 * none / total if total else 0.0,
    )

    # 6. NN-fill (optional).
    if runoff_cfg["nn_fill"]:
        # Materialize the bounds for the in-memory NN walk.
        ds_loaded = ds.compute()
        centroids_xy = hru_meta[["centroid_x", "centroid_y"]].values
        filled_ds, nn_diag = nn_fill_bounds(
            ds_loaded,
            centroids_xy,
            max_candidates=int(runoff_cfg["nn_max_candidates"]),
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
        filled_attrs = dict(extra_attrs)
        filled_attrs["nn_fill_max_candidates"] = int(runoff_cfg["nn_max_candidates"])
        filled_attrs["nn_fill_distance_crs"] = project.area_crs

        nn_path = output_path.with_name(
            output_path.stem + "_nn_filled" + output_path.suffix
        )
        write_target_nc(
            filled_ds,
            nn_path,
            title="NHM runoff calibration target (NN-filled, cfs)",
            extra_global_attrs=filled_attrs,
        )
