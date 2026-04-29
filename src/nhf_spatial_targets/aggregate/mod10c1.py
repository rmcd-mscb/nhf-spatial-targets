"""MOD10C1 v061 daily SCA aggregator (CI-masked) — adapter + hooks.

Follows the repo's aggregation transformation policy
(``docs/architecture/transformation-pipeline.md``):

- Pixel-defined operations live in the pre-aggregate hook: flag-value
  masks and the CI > 70 quality gate. These cannot move downstream
  without changing the result, because area-weighting low-confidence
  pixels into the HRU mean and gating after the fact gives a different
  answer than per-pixel gating before aggregation.
- Native variable names and units pass through untouched. The aggregated
  NC carries ``Day_CMG_Snow_Cover`` and ``Day_CMG_Clear_Index`` in their
  source-native 0–100 integer scale; the ``÷ 100`` rescale is a linear
  conversion that lives downstream in ``targets/sca.py`` /
  ``normalize/methods.py``.
- The diagnostic ``valid_area_fraction`` is aggregation metadata, not a
  source-data transform, so it is allowed as a derived output.
"""

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
# CI threshold expressed in the native 0–100 integer scale (TM 6-B10: keep
# pixels where CI > 70). Strict >, not >=: a pixel with CI exactly 70 fails.
_CI_THRESHOLD_NATIVE = 70
_OUTPUT_NAME = "mod10c1_agg.nc"
_LOW_COVERAGE_WARN_THRESHOLD = 0.10


def build_masked_source(ds: xr.Dataset) -> xr.Dataset:
    """Apply pixel-level flag and CI quality masks to MOD10C1 raw vars.

    Both ``Day_CMG_Snow_Cover`` and ``Day_CMG_Clear_Index`` carry flag
    values above 100 (107=lake ice, 111=night, 237=inland water,
    239=ocean, 250=cloud-obscured water, 253=not mapped, 255=fill);
    these are masked to NaN so they do not contaminate the area-weighted
    mean.

    The CI gate is applied per pixel before aggregation. Per-pixel gating
    is methodologically different from gating the area-weighted-mean CI
    after aggregation, and matches TM 6-B10's recipe: a pixel
    contributes to the HRU mean only if its own CI > 70.

    The ``Day_CMG_Clear_Index`` field itself is only flag-masked (not
    CI-gated) so the post-aggregation HRU-mean CI carries information
    about the broader confidence distribution.

    Output (all variables on the source's native grid; aggregator will
    area-weight to the HRU fabric):

    - ``Day_CMG_Snow_Cover`` — native 0–100 integer scale, NaN at flag
      values *or* where pixel CI ≤ 70.
    - ``Day_CMG_Clear_Index`` — native 0–100 integer scale, NaN at flag
      values only.
    - ``valid_mask`` — 1.0 where pixel CI > 70, else 0.0 (including
      flag-coded and fill cells, since ``NaN > 70`` is False). After
      area-weighted aggregation this becomes ``valid_area_fraction``:
      the fraction of the HRU's source-grid area whose pixels passed
      the CI gate, counting unobserved (flag/fill/ocean) cells as
      failing. Downstream NN-fill in ``normalize/methods.py`` is the
      intended path for handling HRUs where this is too low.
    """
    for var in ("Day_CMG_Snow_Cover", "Day_CMG_Clear_Index"):
        if var not in ds.data_vars:
            raise KeyError(
                f"{_SOURCE_KEY}: pre_aggregate_hook expected raw variable "
                f"{var!r}, found {list(ds.data_vars)}. The v006 → v061 "
                f"rename is the most likely cause; check the consolidated "
                f"NC layer in <datastore>/{_SOURCE_KEY}/."
            )
    snow_masked = ds["Day_CMG_Snow_Cover"].where(ds["Day_CMG_Snow_Cover"] <= 100)
    ci_masked = ds["Day_CMG_Clear_Index"].where(ds["Day_CMG_Clear_Index"] <= 100)
    pass_mask = ci_masked > _CI_THRESHOLD_NATIVE

    out = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": snow_masked.where(pass_mask),
            "Day_CMG_Clear_Index": ci_masked,
            "valid_mask": pass_mask.astype("float64"),
        },
        coords=ds.coords,
    )
    # Preserve native source attrs; tag the gate threshold for downstream
    # consumers that need to know how the data was filtered.
    out["Day_CMG_Snow_Cover"].attrs = dict(ds["Day_CMG_Snow_Cover"].attrs)
    out["Day_CMG_Snow_Cover"].attrs["ci_threshold_native_scale"] = _CI_THRESHOLD_NATIVE
    out["Day_CMG_Clear_Index"].attrs = dict(ds["Day_CMG_Clear_Index"].attrs)
    out["valid_mask"].attrs = {
        "long_name": "per-cell CI-pass indicator",
        "units": "1",
        "ci_threshold_native_scale": _CI_THRESHOLD_NATIVE,
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
            "valid-area after CI>%d filter (n=%d of %d finite). "
            "Downstream Day_CMG_Snow_Cover values are NaN for these cells.",
            year,
            zero_frac * 100,
            _CI_THRESHOLD_NATIVE,
            n_zero,
            n_total,
        )


def _rename_valid_mask(year_ds: xr.Dataset) -> xr.Dataset:
    """Post-aggregation: rename ``valid_mask`` → ``valid_area_fraction``.

    After area-weighted aggregation, the per-pixel 0/1 ``valid_mask``
    becomes a per-HRU fraction in [0, 1] — the share of HRU area whose
    pixels passed the CI filter. Unobserved cells (flag/fill/ocean) are
    counted as failing, so a partly-ocean HRU and a fully-cloudy HRU
    can present the same low ``valid_area_fraction``; the downstream
    NN-fill step in ``normalize/methods.py`` is the path that
    distinguishes them. The rename makes the post-aggregation semantic
    explicit.
    """
    year_ds = year_ds.rename({"valid_mask": "valid_area_fraction"})
    year_ds["valid_area_fraction"].attrs = {
        "long_name": "fraction of HRU area that passed CI filter",
        "units": "1",
        "ci_threshold_native_scale": _CI_THRESHOLD_NATIVE,
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
    variables=("Day_CMG_Snow_Cover", "Day_CMG_Clear_Index", "valid_mask"),
    grid_variable="Day_CMG_Snow_Cover",
    raw_grid_variable="Day_CMG_Snow_Cover",
    source_crs="EPSG:4326",
    pre_aggregate_hook=build_masked_source,
    post_aggregate_hook=_rename_valid_mask,
)


def aggregate_mod10c1(
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
) -> None:
    """Aggregate MOD10C1 v061 daily SCA to HRU polygons with CI masking.

    Pixel-level flag-masking and the CI > 70 gate are applied per-year
    inside the driver's per-year loop via ``pre_aggregate_hook``.
    Output is one NC per year at
    ``data/aggregated/mod10c1_v061/mod10c1_v061_<year>_agg.nc``; each
    carries ``Day_CMG_Snow_Cover`` (native 0–100 scale, CI-filtered),
    ``Day_CMG_Clear_Index`` (native 0–100 scale, flag-masked), and
    ``valid_area_fraction`` keyed on (time, HRU). The ``÷ 100`` rescale
    to fractional [0, 1] happens downstream in target builders /
    notebooks.
    """
    aggregate_source(ADAPTER, fabric_path, id_col, workdir, batch_size)
