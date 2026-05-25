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
passing HRU, again mirroring calcSCA. This is honest in the PNW where
the snowpack is gone by July; it is questionable in deep-snowpack HRUs
(Cascades high-elevation) where late-summer snow persists.

**Pre-pipeline gate vs. post-aggregation gate.** ``aggregate/mod10c1.py``
already applies a *per-pixel* ``CI > 70`` gate inside
``pre_aggregate_hook`` — pixels with CI ≤ 70 are NaN'd before the
area-weighted mean, then ``stat_method="masked_mean"`` averages only
the survivors. The target-stage gate here is in addition: even when
some pixels passed pre-aggregation, the resulting HRU-mean CI can drop
below ``ci_threshold`` (e.g. an HRU where only the highest-CI fragment
of a partly-cloudy day survived). Re-gating at HRU scale matches what
``calcSCA`` does with the per-HRU input it sees.

**Single-source ``n_sources`` semantics.** With one source the
``n_sources`` diagnostic is binary: 1 where a finite bound was produced
(CI gate passed AND ``sca_obs`` is finite), 0 elsewhere. The shared
write helper writes int8 with flag_values ``[0, 1]`` and flag_meanings
``"none one"``.

If ``snow_covered_area.nn_fill`` is True (default), a second file
``<output>_nn_filled.nc`` is written with NaN bounds filled by the
nearest finite HRU's bound at the same day.

**Cache invalidation.** Per-year intermediates under
``<project>/targets/.sca_intermediates/`` carry two fingerprint global
attrs (``config_fingerprint``, ``code_version``) that the skip branch
compares against the active values. A mismatch — caused by a config
edit (``ci_threshold``, sources, period), a fabric swap, or a
``__version__`` bump — logs a WARNING, deletes the stale intermediate,
and rebuilds the year. Operators do not need to manually ``rm`` after
config or version-bump changes; any commit that modifies builder logic
WITHOUT a ``__version__`` bump still requires a manual ``rm`` because
the ``code_version`` tag is tied to the package version string.
The per-year low-valid-coverage WARNING is also persisted via the
``frac_valid_bound`` attr and re-emitted on the skip branch so it
fires on every run, not just the first.

**Period-shrink defense.** Downward changes to ``period`` are
handled automatically by ``prune_orphan_year_intermediates`` plus a
``iter_period_years``-derived stitch input list — see the helper
docstring in ``_common.py`` (#211).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.targets._common import (
    INTERMEDIATE_CODE_VERSION_ATTR,
    INTERMEDIATE_CONFIG_FINGERPRINT_ATTR,
    check_hru_coords,
    code_version_fingerprint,
    compute_hru_centroids,
    iter_period_years,
    parse_period,
    prune_orphan_year_intermediates,
    read_aggregated_source,
    reindex_to_day_start,
    should_skip_year_build,
    stitch_year_chunks_to_target,
    target_config_fingerprint,
    write_bounds_target,
)
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)

_MOD10C1_KEY = "mod10c1_v061"
_SCA_VAR = "Day_CMG_Snow_Cover"
_CI_VAR = "Day_CMG_Clear_Index"

# Captured as a constant so the output NC's global ``summer_zero_months``
# attr can name the months explicitly. Matches PRMSobjfun.f90:calcSCA.
_SUMMER_ZERO_MONTHS = (7, 8)

# Global attr name written to every per-year intermediate that records
# the actual fraction of (HRU, day) cells with a valid bound. Read on
# the skip branch so the low-coverage warning can be re-emitted from a
# cached intermediate (#213).
_FRAC_VALID_BOUND_ATTR = "frac_valid_bound"

# Operator-visible warning threshold: if fewer than this fraction of
# (HRU, day) cells in a year produce a valid bound, the build emits a
# WARNING with the actual fraction. Catches all-NaN CI, fabric mismatch,
# and fully-cloudy-season cases that would otherwise silently produce an
# all-zero n_sources year.
#
# 1% (not the 10% threshold at ``aggregate/mod10c1.py:_LOW_COVERAGE_WARN_THRESHOLD``)
# because the two count different things: the aggregator's 10% counts
# zero-valid-area cells (a per-pixel HRU-coverage statistic), while this
# 1% counts surviving finite bounds (the downstream target's usable
# product). A healthy MOD10C1 year clears 1% trivially even in heavily
# clouded regions where the aggregator's HRU-coverage already sits near
# the 10% warn line — so the two thresholds are intentionally calibrated
# to fire on different failure classes, not redundantly on the same one.
_LOW_VALID_WARN_FRACTION = 0.01


def build(project: Project) -> None:
    """Build the SCA calibration target.

    Year-chunked: each year reads its own slice of the aggregated NCs,
    computes per-day bounds, and writes a per-year intermediate. The
    intermediates are then stitched into the canonical
    ``sca_targets.nc``. Year chunking bounds peak memory regardless of
    period length — for a multi-decade CONUS-scale daily build the
    single-shot alternative would materialise O(10s of GB).
    """
    sca_cfg = project.target("snow_covered_area")
    if list(sca_cfg["sources"]) != [_MOD10C1_KEY]:
        raise ValueError(
            f"snow_covered_area.sources={sca_cfg['sources']!r}; the "
            f"modis_ci range method requires exactly one source: "
            f"[{_MOD10C1_KEY!r}]. Adjust the project config."
        )
    period = parse_period(sca_cfg["period"])
    ci_threshold = float(sca_cfg["ci_threshold"])
    if not (0.0 <= ci_threshold <= 1.0):
        raise ValueError(
            f"snow_covered_area.ci_threshold={ci_threshold!r} must be in "
            f"[0.0, 1.0] (fractional). The native-scale Fortran constant "
            f"70 corresponds to 0.70 here."
        )

    logger.info(
        "Building SCA target: source=%s, period=%s..%s, ci_threshold=%.2f, fabric=%s",
        _MOD10C1_KEY,
        period[0],
        period[1],
        ci_threshold,
        project.config["fabric"]["path"],
    )

    hru_meta = compute_hru_centroids(project)
    id_col = project.id_col
    fabric_hru_ids = hru_meta.index.values

    # Cache fingerprints — see should_skip_year_build (#213).
    config_fp = target_config_fingerprint(project, "snow_covered_area")
    code_ver = code_version_fingerprint()

    extra_attrs = {
        "source": (
            "MOD10C1 v061 Day_CMG_Snow_Cover + Day_CMG_Clear_Index "
            "(CI-bounded per TM 6-B10 / PRMSobjfun.f90:calcSCA)"
        ),
        "references": (
            "Hay et al. 2022, doi:10.3133/tm6B10; "
            "PRMSobjfun.f90 calcSCA (docs/references/PRMSobjfun.f90)"
        ),
        "fabric": project.config["fabric"]["path"],
        "fabric_sha256": project.fabric.get("sha256", ""),
        "period": sca_cfg["period"],
        "area_crs": project.area_crs,
        "ci_threshold": ci_threshold,
        "summer_zero_months": ",".join(str(m) for m in _SUMMER_ZERO_MONTHS),
        INTERMEDIATE_CONFIG_FINGERPRINT_ATTR: config_fp,
        INTERMEDIATE_CODE_VERSION_ATTR: code_ver,
    }

    year_specs = iter_period_years(period[0], period[1])
    if not year_specs:
        raise ValueError(
            f"snow_covered_area.period {sca_cfg['period']} produces no years to build."
        )

    intermediates_dir = project.targets_dir() / ".sca_intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Year-chunked build: %d years, intermediates -> %s "
        "(retained after stitch for forensic value; rm to reclaim disk)",
        len(year_specs),
        intermediates_dir,
    )

    nn_fill = bool(sca_cfg["nn_fill"])
    nn_max_candidates = int(sca_cfg["nn_max_candidates"])

    for year, year_start, year_end in year_specs:
        _build_year(
            project=project,
            year=year,
            year_period=(year_start, year_end),
            ci_threshold=ci_threshold,
            hru_meta=hru_meta,
            fabric_hru_ids=fabric_hru_ids,
            id_col=id_col,
            extra_attrs=extra_attrs,
            intermediates_dir=intermediates_dir,
            nn_fill=nn_fill,
            nn_max_candidates=nn_max_candidates,
            config_fingerprint=config_fp,
            code_version=code_ver,
        )

    # Prune orphans + compute stitch input from year_specs (#211).
    in_period_years = {year for year, _, _ in year_specs}
    prune_orphan_year_intermediates(
        intermediates_dir,
        "sca_targets",
        in_period_years,
        target_label="sca",
        logger=logger,
    )

    output_path = project.targets_dir() / sca_cfg["output_file"]
    unfilled_files = [
        intermediates_dir / f"sca_targets_{year}.nc" for year, _, _ in year_specs
    ]
    stitch_year_chunks_to_target(
        unfilled_files,
        output_path,
        title=(
            "NHM SCA calibration target (CI-bounded fractional 0-1; "
            "lower/upper from MOD10C1 v061)"
        ),
        extra_global_attrs=extra_attrs,
        sort_dim=id_col,
    )

    if nn_fill:
        nn_files = [
            intermediates_dir / f"sca_targets_{year}_nn_filled.nc"
            for year, _, _ in year_specs
        ]
        nn_path = output_path.with_name(
            output_path.stem + "_nn_filled" + output_path.suffix
        )
        nn_attrs = dict(extra_attrs)
        nn_attrs["nn_fill_max_candidates"] = nn_max_candidates
        nn_attrs["nn_fill_distance_crs"] = project.area_crs
        stitch_year_chunks_to_target(
            nn_files,
            nn_path,
            title=("NHM SCA calibration target (NN-filled, fractional 0-1)"),
            extra_global_attrs=nn_attrs,
            sort_dim=id_col,
        )


def _warn_low_valid_coverage(year: int, frac_valid: float, ci_threshold: float) -> None:
    """Emit the per-year low-valid-coverage WARNING if below the floor.

    Called from ``_build_year`` after the live computation, and again
    from the skip branch when ``should_skip_year_build`` returns a
    cached ``frac_valid_bound``. Re-emitting from cache means an
    operator re-running against a healthy-fingerprint cache continues
    to see the alert that fired on the original build (#213).
    """
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


def _build_year(
    *,
    project: Project,
    year: int,
    year_period: tuple[str, str],
    ci_threshold: float,
    hru_meta: pd.DataFrame,
    fabric_hru_ids: np.ndarray,
    id_col: str,
    extra_attrs: dict,
    intermediates_dir: Path,
    nn_fill: bool,
    nn_max_candidates: int,
    config_fingerprint: str,
    code_version: str,
) -> None:
    """Build SCA bounds for one calendar year and write per-year NCs.

    Idempotent: if both expected per-year NCs already exist AND their
    fingerprint global attrs match the active config + code version,
    the build is skipped. On a fingerprint mismatch the stale
    intermediate(s) are deleted automatically and rebuilt — see
    :func:`nhf_spatial_targets.targets._common.should_skip_year_build`.

    On a healthy skip, the cached ``frac_valid_bound`` global attr is
    read and the low-valid-coverage WARNING is re-emitted if the
    cached fraction is below threshold, so an operator re-running
    against a valid cache continues to see the alert.
    """
    year_unfilled = intermediates_dir / f"sca_targets_{year}.nc"
    year_nn = intermediates_dir / f"sca_targets_{year}_nn_filled.nc"
    expected_paths = [year_unfilled] + ([year_nn] if nn_fill else [])
    skip, cached_attrs = should_skip_year_build(
        expected_paths,
        active_config_fingerprint=config_fingerprint,
        active_code_version=code_version,
        target_label="sca",
        year=year,
        logger=logger,
    )
    if skip:
        logger.info(
            "Year %d intermediates valid (config=%s code=%s); skipping.",
            year,
            config_fingerprint,
            code_version,
        )
        cached_frac = (cached_attrs or {}).get(_FRAC_VALID_BOUND_ATTR)
        if cached_frac is not None:
            _warn_low_valid_coverage(year, float(cached_frac), ci_threshold)
        return

    year_master_idx = pd.date_range(year_period[0], year_period[1], freq="D")
    if len(year_master_idx) == 0:
        raise ValueError(
            f"Year {year}: empty master index from period {year_period!r}."
        )

    # Read both vars from the same aggregated source. Chunk the time dim
    # to the year length so a single read per file fills one dask chunk.
    snow_native = read_aggregated_source(
        project,
        _MOD10C1_KEY,
        _SCA_VAR,
        year_period,
        chunks={"time": 366, id_col: -1},
    )
    ci_native = read_aggregated_source(
        project,
        _MOD10C1_KEY,
        _CI_VAR,
        year_period,
        chunks={"time": 366, id_col: -1},
    )
    check_hru_coords(snow_native, fabric_hru_ids, id_col, _MOD10C1_KEY)
    check_hru_coords(ci_native, fabric_hru_ids, id_col, _MOD10C1_KEY)

    # Scale to fractional [0, 1] (catalog/variables.yml:snow_covered_area
    # units = "fraction (0-1)") and reindex onto the canonical day-start
    # master index so days missing from the source contribute NaN.
    sca_obs = reindex_to_day_start(snow_native / 100.0, year_master_idx)
    ci = reindex_to_day_start(ci_native / 100.0, year_master_idx)

    # HRU-mean CI gate. NaN ci compares False so already-pre-gated NaN
    # days correctly fall through to NaN bounds.
    valid = ci >= ci_threshold
    lower = xr.where(valid, ci * sca_obs, np.nan)
    upper = xr.where(valid, lower + (1.0 - ci), np.nan)

    # July/August force to (0, 0) per calcSCA. Applied only where the
    # CI gate passed — invalid days stay NaN, not zero, so downstream
    # consumers can still distinguish "no observation" from "observed
    # bare ground."
    summer = lower["time"].dt.month.isin(list(_SUMMER_ZERO_MONTHS))
    lower = xr.where(summer & valid, 0.0, lower)
    upper = xr.where(summer & valid, 0.0, upper)

    # Single-source binary diagnostic: 1 where a FINITE bound was
    # produced (CI gate passed AND sca_obs is finite), 0 elsewhere.
    # ``valid`` alone would mis-flag the case where a small set of
    # high-CI pixels lifts the area-weighted CI mean above threshold
    # while the snow variable's per-pixel mask (``aggregate/mod10c1.py``
    # ``stat_method="masked_mean"`` after the strict ``CI > 70`` gate)
    # had already annihilated coverage — there the bound is NaN even
    # though ``valid`` is True. Guarded by
    # ``test_n_sources_contract_when_ci_passes_but_snow_nan``.
    # build_n_sources_attrs(1) → flag_values=[0, 1], flag_meanings="none one".
    valid_bound = valid & sca_obs.notnull()
    n_sources = valid_bound.astype(np.int8)

    # Operator-visible warning when an upstream regression or
    # genuinely fully-cloudy season produces near-zero coverage for
    # the year. Persisted to global attrs so the skip branch can re-emit
    # it on a cached re-run (#213).
    frac_valid = float(valid_bound.mean().compute())
    _warn_low_valid_coverage(year, frac_valid, ci_threshold)

    write_bounds_target(
        project=project,
        lower=lower,
        upper=upper,
        n_sources=n_sources,
        n_sources_count=1,
        time_index=year_master_idx,
        time_offset_unit=pd.offsets.Day(1),
        bounds_units="1",
        bounds_long_name_kind="daily fractional snow-covered area",
        cell_methods="time: point",
        output_path=year_unfilled,
        title=f"NHM SCA calibration target year {year} (intermediate)",
        nn_title=(f"NHM SCA calibration target year {year} (NN-filled intermediate)"),
        extra_global_attrs={
            **extra_attrs,
            "year_chunk": year,
            _FRAC_VALID_BOUND_ATTR: frac_valid,
        },
        hru_meta=hru_meta,
        nn_fill=nn_fill,
        nn_max_candidates=nn_max_candidates,
        id_col=id_col,
    )
