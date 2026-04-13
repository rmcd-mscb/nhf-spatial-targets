"""Build runoff calibration targets from ERA5-Land + GLDAS-2.1 NOAH.

Two reanalysis sources are used to produce per-HRU per-month bounds:
  - ERA5-Land 'ro' (m water equivalent / month, ECMWF)
  - GLDAS-2.1 NOAH 'runoff_total' = Qs_acc + Qsb_acc (kg m-2 / month, NASA)

Unit chain: source native units → mm/month → m³/day → cfs, using HRU area
and days-in-month.  Per-HRU per-month:
  lower_bound = min(era5_cfs, gldas_cfs)
  upper_bound = max(era5_cfs, gldas_cfs)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# 1 m³/day = (1/86400) m³/s = (35.3147/86400) ft³/s
# i.e. multiply m³/day by (35.3147 / 86400) to get cfs
_M3_PER_DAY_TO_CFS = 35.3146667 / 86400.0


def era5_to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """ERA5-Land runoff (m water-eq / month) → mm/month."""
    out = da * 1000.0
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


def gldas_to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """GLDAS Qs_acc + Qsb_acc (kg m-2) ≡ mm/month directly.

    Equivalence holds because 1 kg m-2 = 1 mm depth (assuming ρ_water = 1000 kg/m³).
    """
    out = da.copy()
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


def mm_per_month_to_cfs(
    da: xr.DataArray, hru_area_m2: float | xr.DataArray
) -> xr.DataArray:
    """Convert mm/month → cfs given HRU area and the month length.

    mm/month × 1e-3 m/mm × area_m2 / days_in_month → m³/day → cfs.
    Days-in-month is read from ``da.time.dt.days_in_month``.
    """
    days = da["time"].dt.days_in_month
    m_per_day = (da * 1e-3) / days
    m3_per_day = m_per_day * hru_area_m2
    cfs = m3_per_day * _M3_PER_DAY_TO_CFS
    cfs.attrs = dict(da.attrs)
    cfs.attrs["units"] = "cfs"
    return cfs


def multi_source_runoff_bounds(
    sources: list[xr.DataArray],
) -> tuple[xr.DataArray, xr.DataArray]:
    """Per-coord minimum and maximum across input sources.

    All inputs must share dimensions and coords. Returns (lower, upper).
    """
    stacked = xr.concat(sources, dim="source")
    return stacked.min("source"), stacked.max("source")


def _validate_alignment(
    era5: xr.DataArray, gldas: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Validate that two DataArrays are compatible and return overlap-sliced copies.

    Checks:
    - Both have the same dimension names.
    - HRU coordinates are identical.
    - Time ranges overlap; raises ValueError if disjoint.

    Returns the two arrays sliced to the overlapping time range.
    """
    if set(era5.dims) != set(gldas.dims):
        raise ValueError(
            f"ERA5-Land and GLDAS have different dimension names: "
            f"{list(era5.dims)} vs {list(gldas.dims)}"
        )

    era5_hrus = era5.coords.get("hru")
    gldas_hrus = gldas.coords.get("hru")
    if era5_hrus is not None and gldas_hrus is not None:
        if not era5_hrus.equals(gldas_hrus):
            raise ValueError(
                "ERA5-Land and GLDAS HRU coordinates differ. "
                "Ensure both were aggregated to the same fabric."
            )

    era5_start = pd.Timestamp(era5.time.min().values)
    era5_end = pd.Timestamp(era5.time.max().values)
    gldas_start = pd.Timestamp(gldas.time.min().values)
    gldas_end = pd.Timestamp(gldas.time.max().values)

    overlap_start = max(era5_start, gldas_start)
    overlap_end = min(era5_end, gldas_end)

    if overlap_start > overlap_end:
        raise ValueError(
            f"ERA5-Land and GLDAS time ranges do not overlap. "
            f"ERA5-Land: {era5_start} – {era5_end}. "
            f"GLDAS: {gldas_start} – {gldas_end}."
        )

    logger.info("Runoff target overlap window: %s – %s", overlap_start, overlap_end)
    era5 = era5.sel(time=slice(overlap_start, overlap_end))
    gldas = gldas.sel(time=slice(overlap_start, overlap_end))
    return era5, gldas


def build(config: dict, output_path: str) -> None:
    """Build runoff target dataset.

    Reads HRU-aggregated monthly runoff for ERA5-Land (``ro``) and GLDAS
    (``runoff_total``), validates alignment (same dims, HRU coords, overlapping
    time ranges), harmonizes units to cfs, computes per-HRU per-month
    min/max bounds, and writes a CF-compliant NetCDF atomically (via ``.tmp``
    rename) with ``lower_bound`` and ``upper_bound`` variables, dims
    ``(time, hru)``.

    Aggregated input convention: ``<aggregated_dir>/<source_key>/<var>.nc``

    config keys:
      aggregated_dir : str | Path
          Directory containing per-source per-variable aggregated NCs at
          ``<aggregated_dir>/<source_key>/<var>.nc``.
      hru_area_m2 : xr.DataArray
          Per-HRU area in m², dims=('hru',), coord aligned with the
          aggregated outputs.
    """
    agg_dir = Path(config["aggregated_dir"])
    hru_area = config["hru_area_m2"]

    with xr.open_dataset(agg_dir / "era5_land" / "ro.nc") as ds:
        era5 = ds["ro"].load()
    with xr.open_dataset(agg_dir / "gldas_noah_v21_monthly" / "runoff_total.nc") as ds:
        gldas = ds["runoff_total"].load()

    era5, gldas = _validate_alignment(era5, gldas)

    era5_cfs = mm_per_month_to_cfs(era5_to_mm_per_month(era5), hru_area)
    gldas_cfs = mm_per_month_to_cfs(gldas_to_mm_per_month(gldas), hru_area)

    lower, upper = multi_source_runoff_bounds([era5_cfs, gldas_cfs])
    lower.name = "lower_bound"
    upper.name = "upper_bound"
    lower.attrs["units"] = "cfs"
    lower.attrs["cell_methods"] = "time: sum"
    upper.attrs["units"] = "cfs"
    upper.attrs["cell_methods"] = "time: sum"
    out_ds = xr.Dataset(
        {"lower_bound": lower, "upper_bound": upper},
        attrs={
            "title": "NHM runoff calibration target (ERA5-Land + GLDAS-2.1)",
            "Conventions": "CF-1.6",
        },
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".nc.tmp")
    try:
        out_ds.to_netcdf(tmp, format="NETCDF4")
        tmp.rename(output_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    logger.info("Wrote runoff target: %s", output_path)
