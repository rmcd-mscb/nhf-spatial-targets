"""Build snow-covered area calibration targets from MOD10C1 v061.

Single-source CI-bounded build. The bound formula is taken verbatim from
``PRMSobjfun.f90:calcSCA`` (`docs/references/PRMSobjfun.f90` lines
1052-1061) — the original PRMS calibration objective function this
pipeline's targets are designed to feed. Both inputs are first scaled
from MOD10C1's native 0-100 integer scale to fractional [0, 1]:

    sca_obs = Day_CMG_Snow_Cover / 100
    ci      = Day_CMG_Clear_Index / 100

For each (HRU, day) where the HRU-mean clear index passes the configured
threshold (``ci >= snow_covered_area.ci_threshold``, default 0.70 — the
fractional equivalent of the Fortran ``70.0`` on the native scale)::

    lower = ci * sca_obs
    upper = lower + (1 - ci)

``upper`` is algebraically capped at 1.0:
``ci*sca + (1-ci) ≤ ci*1 + (1-ci) = 1`` for any ``sca`` in ``[0, 1]``.
Cells whose HRU-mean CI is below threshold produce NaN bounds.

July and August are forced to ``(lower, upper) = (0, 0)`` for every CI-
passing HRU, again mirroring calcSCA. The driver applies the forced-zero
mask after this module's loader returns via
:attr:`TargetAdapter.forced_zero_months` /
:attr:`TargetAdapter.forced_zero_validity_var`, so the rule stays
declarative on the adapter.

If ``snow_covered_area.nn_fill`` is True (default), a second file
``<output>_nn_filled.nc`` is written with NaN bounds filled by the
nearest finite HRU's bound at the same day.

**Cache invalidation.** Per-year intermediates under
``<project>/targets/.sca_intermediates/`` carry two fingerprint global
attrs (``config_fingerprint``, ``code_version``) that the skip branch
compares against the active values.

**Period-shrink defense.** Downward changes to ``period`` are handled
automatically by ``prune_orphan_year_intermediates`` plus a
``iter_period_years``-derived stitch input list (#211).

The year loop, skip-cache, orphan pruning, and stitch are owned by the
generic year-chunked driver (``targets/_driver.py``). This module only
declares the SCA adapter, the per-year source loader, and the
on-year-skip hook that re-emits the low-valid-coverage WARNING when an
operator re-runs against a healthy cache.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.targets._adapter import SourceLoaderResult, TargetAdapter
from nhf_spatial_targets.targets._io import (
    check_hru_coords,
    read_aggregated_source,
    reindex_to_day_start,
)
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)

_MOD10C1_KEY = "mod10c1_v061"
_SCA_VAR = "Day_CMG_Snow_Cover"
_CI_VAR = "Day_CMG_Clear_Index"
_SUMMER_ZERO_MONTHS = (7, 8)
_FRAC_VALID_BOUND_ATTR = "frac_valid_bound"
_LOW_VALID_WARN_FRACTION = 0.01


def _warn_low_valid_coverage(year: int, frac_valid: float, ci_threshold: float) -> None:
    """Emit the per-year low-valid-coverage WARNING if below the floor."""
    if frac_valid < _LOW_VALID_WARN_FRACTION:
        logger.warning(
            "sca year %d: only %.4f%% of (HRU, day) cells produced a "
            "valid CI-passing bound (threshold for warning: %.2f%%). "
            "Either the aggregated mod10c1 NC is degenerate for this year "
            "or ci_threshold=%.2f is too strict.",
            year,
            frac_valid * 100,
            _LOW_VALID_WARN_FRACTION * 100,
            ci_threshold,
        )


# ---------------------------------------------------------------------------
# Per-year source loader (the driver calls this once per year)
# ---------------------------------------------------------------------------


def _load_sca_year(
    *,
    project: Project,
    adapter: TargetAdapter,
    period: tuple[str, str],
    hru_meta: pd.DataFrame,
    fabric_hru_ids: np.ndarray,
    id_col: str,
    year_context: tuple[int, str, str] | None,
) -> SourceLoaderResult:
    """Compute CI-bounded SCA bounds for one calendar year.

    Returns the result the generic year-chunked driver hands to
    :func:`_writers.write_bounds_target`. The July/August forced-zero
    policy is applied **after** this loader returns by the driver via
    :func:`_driver._apply_forced_zero`, reading the ``valid`` mask from
    ``extras`` — keeping the seasonal rule declarative on the adapter
    rather than baked into the loader.

    Per-year ``frac_valid_bound`` is surfaced via
    ``extras["per_year_attrs"]`` so it lands on each year's intermediate
    NC and survives the skip-branch re-read (which the
    :func:`_on_sca_year_skip` hook uses to re-emit the WARNING).

    ``adapter`` is used to re-read ``ci_threshold`` from the project
    config so per-build threshold changes invalidate cached
    intermediates via the config-fingerprint path. ``year_context`` is
    always supplied by the driver for SCA (``year_chunked=True``);
    ``None`` is treated as a programmer error and raises.
    """
    if year_context is None:
        raise RuntimeError(
            "targets.sca._load_sca_year requires year_context — SCA is "
            "year_chunked=True and the driver always supplies it."
        )
    year, _year_start, _year_end = year_context

    year_master_idx = pd.date_range(period[0], period[1], freq="D")
    if len(year_master_idx) == 0:
        raise ValueError(f"Year {year}: empty master index from period {period!r}.")

    ci_threshold = float(project.target(adapter.config_key)["ci_threshold"])

    snow_native = read_aggregated_source(
        project,
        _MOD10C1_KEY,
        _SCA_VAR,
        period,
        chunks={"time": 366, id_col: -1},
    )
    ci_native = read_aggregated_source(
        project,
        _MOD10C1_KEY,
        _CI_VAR,
        period,
        chunks={"time": 366, id_col: -1},
    )
    check_hru_coords(snow_native, fabric_hru_ids, id_col, _MOD10C1_KEY)
    check_hru_coords(ci_native, fabric_hru_ids, id_col, _MOD10C1_KEY)

    sca_obs = reindex_to_day_start(snow_native / 100.0, year_master_idx)
    ci = reindex_to_day_start(ci_native / 100.0, year_master_idx)

    valid = ci >= ci_threshold
    lower = xr.where(valid, ci * sca_obs, np.nan)
    upper = xr.where(valid, lower + (1.0 - ci), np.nan)
    # Note: July/August forced-zero is applied post-loader by the driver
    # from the `valid` mask in `extras`; see TargetAdapter.forced_zero_*.

    valid_bound = valid & sca_obs.notnull()
    n_sources = valid_bound.astype(np.int8)

    frac_valid = float(valid_bound.mean().compute())
    _warn_low_valid_coverage(year, frac_valid, ci_threshold)

    extra_attrs = {
        "source": (
            "MOD10C1 v061 Day_CMG_Snow_Cover + Day_CMG_Clear_Index "
            "(CI-bounded per TM 6-B10 / PRMSobjfun.f90:calcSCA)"
        ),
        "ci_threshold": ci_threshold,
        "summer_zero_months": ",".join(str(m) for m in _SUMMER_ZERO_MONTHS),
    }

    return SourceLoaderResult(
        lower=lower,
        upper=upper,
        n_sources=n_sources,
        n_sources_count=1,
        time_index=year_master_idx,
        time_offset_unit=pd.offsets.Day(1),
        extra_attrs=extra_attrs,
        extras={
            "valid": valid,
            "per_year_attrs": {_FRAC_VALID_BOUND_ATTR: frac_valid},
        },
    )


def _on_sca_year_skip(
    *,
    project: Project,
    adapter: TargetAdapter,
    year: int,
    cached_attrs: dict,
) -> None:
    """Re-emit the low-valid-coverage WARNING from the cached attr.

    Fired by the driver when a per-year intermediate is healthy and the
    build is skipped. Without this an operator re-running against a
    healthy cache would miss the alert that fired on the original build
    (#213).

    If the cached intermediate predates the per-year
    ``frac_valid_bound`` attr, this hook is a no-op — staleness is
    :func:`should_skip_year_build`'s concern, not this hook's. In
    practice that branch is unreachable for caches the current pipeline
    wrote: any cache missing the fingerprint attrs would have been
    unlinked by ``should_skip_year_build`` before this hook fires.
    """
    cached_frac = cached_attrs.get(_FRAC_VALID_BOUND_ATTR)
    if cached_frac is None:
        return
    ci_threshold = float(project.target(adapter.config_key)["ci_threshold"])
    _warn_low_valid_coverage(year, float(cached_frac), ci_threshold)


# ---------------------------------------------------------------------------
# Adapter declaration
# ---------------------------------------------------------------------------


ADAPTER = TargetAdapter(
    target_key="sca",
    config_key="snow_covered_area",
    cadence="daily",
    bounds_units="1",
    bounds_long_name_kind="daily fractional snow-covered area",
    cell_methods="time: point",
    title=(
        "NHM SCA calibration target (CI-bounded fractional 0-1; "
        "lower/upper from MOD10C1 v061)"
    ),
    nn_title="NHM SCA calibration target (NN-filled, fractional 0-1)",
    references=(
        "Hay et al. 2022, doi:10.3133/tm6B10; "
        "PRMSobjfun.f90 calcSCA (docs/references/PRMSobjfun.f90)"
    ),
    year_chunked=True,
    intermediates_subdir=".sca_intermediates",
    intermediate_base="sca_targets",
    forced_zero_months=_SUMMER_ZERO_MONTHS,
    forced_zero_validity_var="valid",
    per_year_title_template=("NHM SCA calibration target year {year} (intermediate)"),
    per_year_nn_title_template=(
        "NHM SCA calibration target year {year} (NN-filled intermediate)"
    ),
    source_loader=_load_sca_year,
    on_year_skip=_on_sca_year_skip,
)


def build(project: Project) -> None:
    """Build the SCA calibration target.

    Validates the SCA-specific config invariants (single source =
    mod10c1_v061; ci_threshold ∈ [0, 1]) and delegates the year loop,
    cache management, orphan pruning, and stitching to the generic
    year-chunked driver via :func:`_driver.build`.
    """
    from nhf_spatial_targets.targets._driver import build as run_driver

    sca_cfg = project.target(ADAPTER.config_key)
    if list(sca_cfg["sources"]) != [_MOD10C1_KEY]:
        raise ValueError(
            f"snow_covered_area.sources={sca_cfg['sources']!r}; the "
            f"modis_ci range method requires exactly one source: "
            f"[{_MOD10C1_KEY!r}]. Adjust the project config."
        )
    ci_threshold = float(sca_cfg["ci_threshold"])
    if not (0.0 <= ci_threshold <= 1.0):
        raise ValueError(
            f"snow_covered_area.ci_threshold={ci_threshold!r} must be in "
            f"[0.0, 1.0] (fractional). The native-scale Fortran constant "
            f"70 corresponds to 0.70 here."
        )

    logger.info(
        "Building SCA target: source=%s, period=%s, ci_threshold=%.2f, fabric=%s",
        _MOD10C1_KEY,
        sca_cfg["period"],
        ci_threshold,
        project.config["fabric"]["path"],
    )

    run_driver(ADAPTER, project)
