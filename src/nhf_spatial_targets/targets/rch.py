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

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.normalize.methods import nn_fill_bounds
from nhf_spatial_targets.targets._common import (
    SourceShim,
    compute_hru_centroids,
    multi_source_nanminmax,
    read_aggregated_source,
    shims_by_key,
    write_target_nc,
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


def _normalize_0_1_over_window(
    da: xr.DataArray, window: xr.DataArray, dim: str = "time"
) -> xr.DataArray:
    """Normalize ``da`` to [0, 1] using ``window``'s min/max along ``dim``.

    Like :func:`normalize_0_1` but the min/max are computed over a subset
    of ``da`` (the normalization period). Values in ``da`` outside the
    window may produce values < 0 or > 1 — by design: TM 6-B10 §3 says
    the bound reflects the calibration-period climatology, and a future
    year outside that climatology should be visibly out-of-range, not
    silently re-normalized.
    """
    mn = window.min(dim=dim, skipna=True)
    mx = window.max(dim=dim, skipna=True)
    rng = mx - mn
    safe_rng = rng.where(rng > 0)
    norm = (da - mn) / safe_rng
    norm.attrs = dict(da.attrs)
    norm.attrs["units"] = "1"
    return norm


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
    """Build the recharge calibration target.

    Reads each enabled source's per-year aggregated NCs, harmonizes onto
    a master year-start index over ``recharge.period``, normalizes each
    independently to [0, 1] using its min/max over
    ``recharge.normalize_period``, combines via NaN-aware min/max, and
    writes a CF-1.6 NetCDF. If ``recharge.nn_fill`` is True, also writes
    ``recharge_targets_nn_filled.nc``.
    """
    rch_cfg = project.target("recharge")
    period = _parse_period(rch_cfg["period"])
    normalize_period = _parse_period(rch_cfg["normalize_period"])
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

    # 1. Per-HRU centroids for NN-fill (no area conversion needed).
    hru_meta = compute_hru_centroids(project)
    id_col = project.id_col

    # 2. Master year-start index over the requested period.
    master_idx = pd.date_range(period[0], period[1], freq="YS")
    if len(master_idx) == 0:
        raise ValueError(
            f"recharge.period {rch_cfg['period']} produces no years at "
            "freq='YS'. Check the date range."
        )

    # 3. Read, harmonize, normalize each source.
    shims = shims_by_key(SHIMS)
    fabric_hru_ids = hru_meta.index.values
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
        # HRU coord check (canonical-sort invariant, same as runoff/aet).
        src_hru_ids = da_native[id_col].values
        if not np.array_equal(src_hru_ids, fabric_hru_ids):
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
        da_annual_mm = shim.to_common_units(da_native)
        # Normalize using normalize_period's min/max; project onto master_idx.
        window = da_annual_mm.sel(time=slice(normalize_period[0], normalize_period[1]))
        if window.sizes.get("time", 0) == 0:
            raise ValueError(
                f"recharge.normalize_period {rch_cfg['normalize_period']} "
                f"yields no annual timesteps for source '{src}'. "
                f"Source covers {da_annual_mm.time.values[0]} .. "
                f"{da_annual_mm.time.values[-1]}."
            )
        da_normalized = _normalize_0_1_over_window(da_annual_mm, window)
        # Reindex to master year-start index (NaN-pads years the source
        # doesn't cover; the multi-source min/max handles partial coverage).
        sources_normalized[src] = da_normalized.reindex(time=master_idx)

    # 4. NaN-aware combination across normalized sources.
    lower, upper, n_sources = multi_source_nanminmax(sources_normalized)
    lower.name = "lower_bound"
    upper.name = "upper_bound"
    n_sources.name = "n_sources"

    # 5. Assemble Dataset with CF metadata + ancillary coords.
    time_bnds = xr.DataArray(
        list(
            zip(
                master_idx.values,
                (master_idx + pd.offsets.YearBegin(1)).values,
            )
        ),
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
            "units": "1",
            "long_name": (
                "lower bound of annual recharge "
                "(NaN-aware min across normalized sources)"
            ),
            "cell_methods": "time: sum",
            "coordinates": "centroid_lat centroid_lon",
        }
    )
    upper.attrs.update(
        {
            "units": "1",
            "long_name": (
                "upper bound of annual recharge "
                "(NaN-aware max across normalized sources)"
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
    ds["time"].attrs["long_name"] = "time at year start"
    ds[id_col].attrs["long_name"] = "HRU identifier"
    ds[id_col].attrs["cf_role"] = "timeseries_id"

    extra_attrs = {
        "source": "; ".join(shims[s].description for s in sources),
        "references": "Hay et al. 2022, doi:10.3133/tm6B10",
        "fabric": project.config["fabric"]["path"],
        "fabric_sha256": project.fabric.get("sha256", ""),
        "period": rch_cfg["period"],
        "normalize_period": rch_cfg["normalize_period"],
        "area_crs": project.area_crs,
    }

    ds_loaded = ds.compute()

    output_path = project.targets_dir() / rch_cfg["output_file"]
    write_target_nc(
        ds_loaded,
        output_path,
        title=(
            "NHM recharge calibration target (lower/upper bounds, dimensionless 0-1)"
        ),
        extra_global_attrs=extra_attrs,
        sort_dim=id_col,
    )

    n = ds_loaded["n_sources"].values
    total = n.size
    none = int((n == 0).sum())
    logger.info(
        "Coverage: %d/%d cells have >=1 finite source (%.2f%% all-NaN)",
        total - none,
        total,
        100.0 * none / total if total else 0.0,
    )

    # 6. NN-fill (optional).
    if rch_cfg["nn_fill"]:
        centroids_xy = hru_meta[["centroid_x", "centroid_y"]].values
        filled_ds, nn_diag = nn_fill_bounds(
            ds_loaded,
            centroids_xy,
            max_candidates=int(rch_cfg["nn_max_candidates"]),
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
        filled_attrs["nn_fill_max_candidates"] = int(rch_cfg["nn_max_candidates"])
        filled_attrs["nn_fill_distance_crs"] = project.area_crs

        nn_path = output_path.with_name(
            output_path.stem + "_nn_filled" + output_path.suffix
        )
        write_target_nc(
            filled_ds,
            nn_path,
            title=("NHM recharge calibration target (NN-filled, dimensionless 0-1)"),
            extra_global_attrs=filled_attrs,
            sort_dim=id_col,
        )
