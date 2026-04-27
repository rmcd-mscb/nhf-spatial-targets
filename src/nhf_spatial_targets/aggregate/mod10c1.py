"""MOD10C1 v061 daily SCA aggregator (CI-masked) — adapter + hooks."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import (
    _find_time_coord_name,
    aggregate_source,
)

logger = logging.getLogger(__name__)

_SOURCE_KEY = "mod10c1_v061"
_CI_THRESHOLD = 0.70  # TM 6-B10: keep cells where CI > 0.70
_OUTPUT_NAME = "mod10c1_agg.nc"
_LOW_COVERAGE_WARN_THRESHOLD = 0.10


def build_masked_source(ds: xr.Dataset) -> xr.Dataset:
    """Derive ``sca``, ``ci``, ``valid_mask`` from raw MOD10C1 v061 variables.

    The CI field for MOD10C1 is ``Day_CMG_Clear_Index`` (renamed from
    ``Day_CMG_Confidence_Index`` in v006); ``Snow_Spatial_QA`` in v061 is a
    different field — a 0-4 quality code with flag values 237/239/250/etc —
    and must not be used as a CI percentage.

    Both ``Day_CMG_Snow_Cover`` and ``Day_CMG_Clear_Index`` carry flag values
    above 100 (107=lake ice, 111=night, 237=inland water, 239=ocean,
    250=cloud-obscured water, 253=not mapped, 255=fill); these are masked to
    NaN before scaling so they don't contaminate the weighted mean.

    - ``sca``        = Day_CMG_Snow_Cover / 100, NaN where ci <= 0.70.
    - ``ci``         = Day_CMG_Clear_Index / 100 (NaN at flag values).
    - ``valid_mask`` = 1.0 where ci > 0.70, 0.0 otherwise.
    """
    snow_raw = ds["Day_CMG_Snow_Cover"].where(ds["Day_CMG_Snow_Cover"] <= 100)
    ci_raw = ds["Day_CMG_Clear_Index"].where(ds["Day_CMG_Clear_Index"] <= 100)

    ci = ci_raw / 100.0
    pass_mask = ci > _CI_THRESHOLD
    sca_raw = snow_raw / 100.0
    sca = sca_raw.where(pass_mask)
    valid_mask = pass_mask.astype("float64")

    out = xr.Dataset(
        {"sca": sca, "ci": ci, "valid_mask": valid_mask},
        coords=ds.coords,
    )
    out["sca"].attrs = {"long_name": "fractional snow-covered area", "units": "1"}
    out["ci"].attrs = {
        "long_name": "confidence interval (Snow_Spatial_QA/100)",
        "units": "1",
    }
    out["valid_mask"].attrs = {
        "long_name": "per-cell CI-pass indicator",
        "units": "1",
        "ci_threshold": _CI_THRESHOLD,
    }
    return out


def _log_low_valid_coverage(year_ds: xr.Dataset, *, year: int) -> None:
    """Warn if > 10% of (HRU, time) cells for this year have zero valid area."""
    vaf = year_ds["valid_area_fraction"]
    n_total = int(vaf.notnull().sum())
    if n_total == 0:
        return
    n_zero = int(((vaf == 0) & vaf.notnull()).sum())
    zero_frac = n_zero / n_total
    if zero_frac > _LOW_COVERAGE_WARN_THRESHOLD:
        logger.warning(
            "mod10c1 year=%d: %.1f%% of (HRU, time) cells had zero "
            "valid-area after CI>%.2f filter (n=%d of %d finite). "
            "Downstream sca values are NaN for these cells.",
            year,
            zero_frac * 100,
            _CI_THRESHOLD,
            n_zero,
            n_total,
        )


def _rename_and_warn(year_ds: xr.Dataset) -> xr.Dataset:
    year_ds = year_ds.rename({"valid_mask": "valid_area_fraction"})
    year_ds["valid_area_fraction"].attrs = {
        "long_name": "fraction of HRU area that passed CI filter",
        "units": "1",
        "ci_threshold": _CI_THRESHOLD,
    }
    # post_aggregate_hook runs inside aggregate_year, so year_ds covers
    # exactly one calendar year. Use CF time-coord detection rather than
    # a hardcoded "time" name so a future source-coord rename doesn't
    # break the warning silently.
    time_name = _find_time_coord_name(year_ds)
    if time_name is None:
        logger.warning(
            "mod10c1: post-aggregate year_ds has no CF time coord; "
            "skipping low-valid-coverage warning."
        )
        return year_ds
    time_vals = year_ds[time_name].values
    if len(time_vals) == 0:
        logger.warning(
            "mod10c1: post-aggregate year_ds has zero timesteps; "
            "skipping low-valid-coverage warning."
        )
        return year_ds
    year = int(pd.DatetimeIndex(time_vals).year[0])
    _log_low_valid_coverage(year_ds, year=year)
    return year_ds


ADAPTER = SourceAdapter(
    source_key=_SOURCE_KEY,
    output_name=_OUTPUT_NAME,
    variables=("sca", "ci", "valid_mask"),
    grid_variable="sca",
    raw_grid_variable="Day_CMG_Snow_Cover",
    source_crs="EPSG:4326",
    pre_aggregate_hook=build_masked_source,
    post_aggregate_hook=_rename_and_warn,
)


def aggregate_mod10c1(
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
) -> None:
    """Aggregate MOD10C1 v061 daily SCA to HRU polygons with CI masking.

    The CI>0.70 filter is applied per-year inside the driver's per-year loop
    via ``pre_aggregate_hook``. Output is one NC per year at
    ``data/aggregated/mod10c1_v061/mod10c1_v061_<year>_agg.nc``; each carries
    ``sca``, ``ci``, and ``valid_area_fraction`` keyed on (time, HRU).
    """
    aggregate_source(ADAPTER, fabric_path, id_col, workdir, batch_size)
