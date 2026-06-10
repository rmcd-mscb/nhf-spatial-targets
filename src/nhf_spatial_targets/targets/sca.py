"""Build snow-covered area calibration targets (multi-source bound).

Each configured source produces a per-(HRU, day) ``[lower, upper]``
interval; the intervals are combined NaN-aware (``lower = nanmin``,
``upper = nanmax``) into the calibration bound, with an int8 ``n_sources``
finite-contribution diagnostic. Two sources are registered today:

- **mod10c1_v061** — the CI-bounded interval taken verbatim from
  ``PRMSobjfun.f90:calcSCA`` (`docs/references/PRMSobjfun.f90` lines
  1052-1061), the original PRMS calibration objective function this
  pipeline's targets feed. Inputs are scaled from MOD10C1's native 0-100
  integer scale to fractional [0, 1]::

      sca_obs = Day_CMG_Snow_Cover / 100
      ci      = Day_CMG_Clear_Index / 100

  For each (HRU, day) where the HRU-mean clear index passes
  ``snow_covered_area.ci_threshold`` (default 0.70 — the fractional
  equivalent of the Fortran ``70.0``)::

      lower = ci * sca_obs
      upper = lower + (1 - ci)          # algebraically capped at 1.0

  Sub-threshold cells produce a NaN interval.

- **ua_swe** — a degenerate ``[v, v]`` interval, ``v =
  snow_covered_fraction`` (an already-[0, 1] per-HRU fraction from the
  ua_swe aggregated NC, the depth-thresholded binary area-weighted to the
  fabric). No CI gate — ua_swe's fraction is physical, not confidence-
  weighted. ua_swe reaches back to WY 1982, widening the pre-2003 bound
  where MOD10C1 (2000+) is the only other source.

**July/August forced-zero (calcSCA).** Summer is forced to
``(lower, upper) = (0, 0)``, gated by ``forced_zero_combined`` (default
True):

- ``True`` — the driver zeros the *combined* bound in Jul/Aug for every
  cell any source observes (the validity mask the loader returns).
- ``False`` — only the mod10c1 contribution is zeroed (pre-combine, in the
  loader) and the driver-level forced-zero is suppressed, so ua_swe's
  alpine summer fraction survives into the combined ``upper`` bound.

**Zero-width bounds.** ``min_sources_for_bound`` (default 1) controls
degenerate intervals: ``1`` keeps a bound wherever ≥1 source is finite
(a single source's ``[v, v]`` is allowed); ``2`` masks the combined bound
to NaN wherever fewer than two sources are finite, guaranteeing width.

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
from collections.abc import Callable
from dataclasses import dataclass

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
_UA_SWE_KEY = "ua_swe"
_SCA_VAR = "Day_CMG_Snow_Cover"
_CI_VAR = "Day_CMG_Clear_Index"
_SCF_VAR = "snow_covered_fraction"
_SUMMER_ZERO_MONTHS = (7, 8)
_FRAC_VALID_BOUND_ATTR = "frac_valid_bound"
_LOW_VALID_WARN_FRACTION = 0.01
#: Per-year time chunking shared by every source interval loader.
_DAY_CHUNKS = {"time": 366}


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
# Per-source interval registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SourceInterval:
    """One source's per-(HRU, day) bound interval for a calendar year.

    ``lower`` / ``upper`` are the source's ``[lower, upper]`` bound (NaN
    where the source contributes nothing at that cell). ``summer_valid`` is
    the per-cell mask of where this source triggers the calcSCA July/August
    forced-zero (mod10c1: CI-passing cells; ua_swe: every observed cell).
    The source is *finite* — and counts toward ``n_sources`` — exactly where
    ``lower`` is non-NaN.
    """

    lower: xr.DataArray
    upper: xr.DataArray
    summer_valid: xr.DataArray
    label: str


def _load_mod10c1_interval(
    *,
    project: Project,
    period: tuple[str, str],
    year_master_idx: pd.DatetimeIndex,
    fabric_hru_ids: np.ndarray,
    id_col: str,
    sca_cfg: dict,
) -> _SourceInterval:
    """CI-bounded MOD10C1 interval (the calcSCA formula, unchanged)."""
    ci_threshold = float(sca_cfg["ci_threshold"])
    chunks = {**_DAY_CHUNKS, id_col: -1}
    snow_native = read_aggregated_source(
        project, _MOD10C1_KEY, _SCA_VAR, period, chunks
    )
    ci_native = read_aggregated_source(project, _MOD10C1_KEY, _CI_VAR, period, chunks)
    check_hru_coords(snow_native, fabric_hru_ids, id_col, _MOD10C1_KEY)
    check_hru_coords(ci_native, fabric_hru_ids, id_col, _MOD10C1_KEY)

    sca_obs = reindex_to_day_start(snow_native / 100.0, year_master_idx)
    ci = reindex_to_day_start(ci_native / 100.0, year_master_idx)

    ci_passing = ci >= ci_threshold
    lower = xr.where(ci_passing, ci * sca_obs, np.nan)
    upper = xr.where(ci_passing, lower + (1.0 - ci), np.nan)
    return _SourceInterval(
        lower=lower, upper=upper, summer_valid=ci_passing, label=_MOD10C1_KEY
    )


def _load_ua_swe_interval(
    *,
    project: Project,
    period: tuple[str, str],
    year_master_idx: pd.DatetimeIndex,
    fabric_hru_ids: np.ndarray,
    id_col: str,
    sca_cfg: dict,
) -> _SourceInterval:
    """Degenerate ``[v, v]`` interval from ua_swe ``snow_covered_fraction``.

    ``v`` is the already-[0, 1] per-HRU snow-covered fraction (the depth-
    thresholded binary area-weighted to the fabric in ``aggregate/ua_swe``);
    no scaling and no CI gate — ua_swe is physical. ``summer_valid`` is
    every finite cell, so under ``forced_zero_combined=True`` ua_swe's
    summer cells are forced to zero alongside mod10c1's.
    """
    scf = read_aggregated_source(
        project, _UA_SWE_KEY, _SCF_VAR, period, {**_DAY_CHUNKS, id_col: -1}
    )
    check_hru_coords(scf, fabric_hru_ids, id_col, _UA_SWE_KEY)
    v = reindex_to_day_start(scf, year_master_idx)
    return _SourceInterval(
        lower=v, upper=v, summer_valid=v.notnull(), label=_UA_SWE_KEY
    )


#: source key -> interval loader. ``build`` validates every configured
#: ``snow_covered_area.sources`` key against this registry, so an unknown
#: source fails loudly at build time rather than KeyError-ing here.
_INTERVAL_LOADERS: dict[str, Callable[..., _SourceInterval]] = {
    _MOD10C1_KEY: _load_mod10c1_interval,
    _UA_SWE_KEY: _load_ua_swe_interval,
}


def _combine_intervals(
    intervals: dict[str, _SourceInterval], *, min_sources_for_bound: int
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """NaN-aware union of per-source intervals.

    Returns ``(lower, upper, n_sources, summer_valid)`` where ``lower`` is
    the per-cell min of source lowers, ``upper`` the max of source uppers,
    ``n_sources`` the int8 count of finite contributions, and
    ``summer_valid`` the OR of the sources' forced-zero validity masks.

    When ``min_sources_for_bound >= 2`` the combined bound is masked to NaN
    wherever ``n_sources < min_sources_for_bound`` (guaranteeing a
    non-degenerate interval); the mod10c1-only / ``min=1`` path leaves the
    bound untouched, preserving the single-source result exactly.
    """
    keys = list(intervals)
    lowers = xr.concat(
        [intervals[k].lower for k in keys], dim=xr.Variable("source", keys)
    )
    uppers = xr.concat(
        [intervals[k].upper for k in keys], dim=xr.Variable("source", keys)
    )
    summer = xr.concat(
        [intervals[k].summer_valid for k in keys], dim=xr.Variable("source", keys)
    )
    lower = lowers.min(dim="source", skipna=True)
    upper = uppers.max(dim="source", skipna=True)
    n_sources = lowers.notnull().sum(dim="source").astype(np.int8)
    summer_valid = summer.any(dim="source")

    if min_sources_for_bound >= 2:
        has_width = n_sources >= min_sources_for_bound
        lower = lower.where(has_width)
        upper = upper.where(has_width)
    return lower, upper, n_sources, summer_valid


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
    """Combine the configured sources into one calendar year's SCA bound.

    Each ``snow_covered_area.sources`` key is loaded via
    :data:`_INTERVAL_LOADERS` into a per-source ``[lower, upper]`` interval,
    then combined NaN-aware (:func:`_combine_intervals`). The July/August
    forced-zero is applied **after** this loader returns by the driver via
    :func:`_driver._apply_forced_zero`, reading the ``valid`` mask from
    ``extras``.

    **Forced-zero / combine coupling.** ``forced_zero_combined`` (default
    True) gates how summer is zeroed *without* any driver or adapter change:

    - ``True`` — the returned ``valid`` mask is ``summer_valid`` (every cell
      any source observes, AND ≥ ``min_sources_for_bound`` finite), so the
      driver zeros the *combined* bound in Jul/Aug. For a mod10c1-only,
      ``min=1`` config this is ``ci >= ci_threshold`` — byte-identical to the
      single-source build.
    - ``False`` — the mod10c1 interval is zeroed in Jul/Aug *before* the
      combine (here, in the loader), and the returned ``valid`` mask is
      forced False in Jul/Aug so the driver-level forced-zero no-ops, letting
      ua_swe's summer fraction survive into the combined ``upper``.

    Per-year ``frac_valid_bound`` is surfaced via ``extras["per_year_attrs"]``
    so it lands on each year's intermediate NC and survives the skip-branch
    re-read (which :func:`_on_sca_year_skip` uses to re-emit the WARNING).
    ``year_context`` is always supplied by the driver for SCA
    (``year_chunked=True``); ``None`` is a programmer error and raises.
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

    sca_cfg = project.target(adapter.config_key)
    sources = list(sca_cfg["sources"])
    ci_threshold = float(sca_cfg["ci_threshold"])
    forced_zero_combined = bool(sca_cfg.get("forced_zero_combined", True))
    min_sources = int(sca_cfg.get("min_sources_for_bound", 1))

    intervals = {
        key: _INTERVAL_LOADERS[key](
            project=project,
            period=period,
            year_master_idx=year_master_idx,
            fabric_hru_ids=fabric_hru_ids,
            id_col=id_col,
            sca_cfg=sca_cfg,
        )
        for key in sources
    }

    # forced_zero_combined=False: zero the mod10c1 contribution in summer
    # BEFORE the combine, so ua_swe's (un-zeroed) fraction survives into the
    # combined upper bound. The driver-level forced-zero is suppressed below.
    in_summer = year_master_idx.month.isin(_SUMMER_ZERO_MONTHS)
    in_summer = xr.DataArray(in_summer, coords={"time": year_master_idx}, dims="time")
    if not forced_zero_combined and _MOD10C1_KEY in intervals:
        iv = intervals[_MOD10C1_KEY]
        zero_here = in_summer & iv.summer_valid
        intervals[_MOD10C1_KEY] = _SourceInterval(
            lower=xr.where(zero_here, 0.0, iv.lower),
            upper=xr.where(zero_here, 0.0, iv.upper),
            summer_valid=iv.summer_valid,
            label=iv.label,
        )

    lower, upper, n_sources, summer_valid = _combine_intervals(
        intervals, min_sources_for_bound=min_sources
    )

    # The driver forces (0, 0) where `valid & month in {7, 8}`. Gate by the
    # min-sources width mask too (min>=2) so a NaN'd narrow cell is never
    # un-masked back to zero. For forced_zero_combined=False, drive `valid`
    # False in summer so the driver no-ops there (mod10c1 already pre-zeroed).
    valid = summer_valid
    if min_sources >= 2:
        valid = valid & (n_sources >= min_sources)
    if not forced_zero_combined:
        valid = valid & ~in_summer

    bound_finite = lower.notnull()
    frac_valid = float(bound_finite.mean().compute())
    _warn_low_valid_coverage(year, frac_valid, ci_threshold)

    # `source_keys` is stamped by the driver's _common_global_attrs from the
    # merged target config; we add only the human-readable `source` and the
    # SCA-specific resolved knobs here.
    extra_attrs = {
        "source": _sca_source_description(sources),
        "ci_threshold": ci_threshold,
        "forced_zero_combined": str(forced_zero_combined),
        "min_sources_for_bound": min_sources,
        "summer_zero_months": ",".join(str(m) for m in _SUMMER_ZERO_MONTHS),
    }

    return SourceLoaderResult(
        lower=lower,
        upper=upper,
        n_sources=n_sources,
        n_sources_count=len(sources),
        time_index=year_master_idx,
        time_offset_unit=pd.offsets.Day(1),
        extra_attrs=extra_attrs,
        extras={
            "valid": valid,
            "per_year_attrs": {_FRAC_VALID_BOUND_ATTR: frac_valid},
        },
    )


def _sca_source_description(sources: list[str]) -> str:
    """Human-readable ``source`` global attr listing the combined sources."""
    parts = []
    if _MOD10C1_KEY in sources:
        parts.append(
            "MOD10C1 v061 Day_CMG_Snow_Cover + Day_CMG_Clear_Index "
            "(CI-bounded per TM 6-B10 / PRMSobjfun.f90:calcSCA)"
        )
    if _UA_SWE_KEY in sources:
        parts.append(
            "UA NSIDC-0719 snow_covered_fraction (depth-thresholded binary, "
            "area-weighted to HRU; degenerate [v, v] interval)"
        )
    for key in sources:
        if key not in (_MOD10C1_KEY, _UA_SWE_KEY):
            parts.append(key)
    return "; ".join(parts)


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
        "NHM SCA calibration target (fractional 0-1; NaN-aware bound over "
        "MOD10C1 v061 CI-interval + ua_swe depth-derived fraction)"
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

    Validates the SCA config invariants — every ``sources`` key has a
    registered interval loader; ``ci_threshold ∈ [0, 1]``;
    ``min_sources_for_bound ∈ {1, 2}``; ``depth_threshold_mm > 0`` (consumed
    at the ua_swe aggregate stage, validated here so a bad value fails before
    a long build) — and delegates the year loop, cache management, orphan
    pruning, and stitching to the generic year-chunked driver via
    :func:`_driver.build`.
    """
    from nhf_spatial_targets.targets._driver import build as run_driver

    sca_cfg = project.target(ADAPTER.config_key)
    sources = list(sca_cfg["sources"])
    if not sources:
        raise ValueError(
            "snow_covered_area.sources is empty; configure at least one of "
            f"{sorted(_INTERVAL_LOADERS)}."
        )
    unknown = [s for s in sources if s not in _INTERVAL_LOADERS]
    if unknown:
        raise ValueError(
            f"snow_covered_area.sources has no registered interval loader for "
            f"{unknown}; known sources are {sorted(_INTERVAL_LOADERS)}."
        )

    ci_threshold = float(sca_cfg["ci_threshold"])
    if not (0.0 <= ci_threshold <= 1.0):
        raise ValueError(
            f"snow_covered_area.ci_threshold={ci_threshold!r} must be in "
            f"[0.0, 1.0] (fractional). The native-scale Fortran constant "
            f"70 corresponds to 0.70 here."
        )

    min_sources = int(sca_cfg.get("min_sources_for_bound", 1))
    if min_sources not in (1, 2):
        raise ValueError(
            f"snow_covered_area.min_sources_for_bound={min_sources!r} must be "
            f"1 (degenerate single-source bound allowed) or 2 (require width)."
        )

    depth_threshold_mm = float(sca_cfg.get("depth_threshold_mm", 1.0))
    if depth_threshold_mm <= 0:
        raise ValueError(
            f"snow_covered_area.depth_threshold_mm={depth_threshold_mm!r} must "
            f"be > 0 (it is the pixel snow-depth cutoff for ua_swe's "
            f"snow_covered_fraction binary, consumed at the aggregate stage)."
        )

    logger.info(
        "Building SCA target: sources=%s, period=%s, ci_threshold=%.2f, "
        "forced_zero_combined=%s, min_sources_for_bound=%d, fabric=%s",
        sources,
        sca_cfg["period"],
        ci_threshold,
        bool(sca_cfg.get("forced_zero_combined", True)),
        min_sources,
        project.config["fabric"]["path"],
    )

    run_driver(ADAPTER, project)
