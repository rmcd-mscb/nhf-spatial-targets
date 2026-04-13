"""Build runoff calibration targets from ERA5-Land + GLDAS-2.1 NOAH."""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)

_M3_PER_DAY_TO_CFS = 35.3146667 / 86400.0  # cubic feet per second per m³/day


def era5_to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """ERA5-Land runoff (m water-eq / month) → mm/month."""
    out = da * 1000.0
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


def gldas_to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """GLDAS Qs_acc + Qsb_acc (kg m-2) ≡ mm/month directly."""
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


def build(config: dict, fabric_path: str, output_path: str) -> None:
    """Build runoff target dataset.

    Reads HRU-aggregated monthly runoff for ERA5-Land (``ro``) and GLDAS
    (``runoff_total``), harmonizes units to cfs, computes per-HRU per-month
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

    era5_cfs = mm_per_month_to_cfs(era5_to_mm_per_month(era5), hru_area)
    gldas_cfs = mm_per_month_to_cfs(gldas_to_mm_per_month(gldas), hru_area)

    lower, upper = multi_source_runoff_bounds([era5_cfs, gldas_cfs])
    lower.name = "lower_bound"
    upper.name = "upper_bound"
    out_ds = xr.Dataset(
        {"lower_bound": lower, "upper_bound": upper},
        attrs={
            "title": "NHM runoff calibration target (ERA5-Land + GLDAS-2.1)",
            "Conventions": "CF-1.6",
            "cell_methods": "time: sum",
            "units": "cfs",
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
