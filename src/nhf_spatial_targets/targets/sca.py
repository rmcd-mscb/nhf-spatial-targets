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

.. warning::

   Per-year intermediates under ``<project>/targets/.sca_intermediates/``
   are keyed by year only — they are NOT fingerprinted by
   ``ci_threshold`` or by the code version that wrote them. After
   changing ``ci_threshold`` or after any change to this module that
   alters the formula, ``rm -rf <project>/targets/.sca_intermediates/``
   before re-running, otherwise stale intermediates will be silently
   reused. The active ``ci_threshold`` is logged whenever the per-year
   skip branch fires so operators can cross-check. Tracked for a
   shared-with-SWE automatic fix in issue #213.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.targets._common import (
    check_hru_coords,
    compute_hru_centroids,
    iter_period_years,
    parse_period,
    read_aggregated_source,
    reindex_to_day_start,
    stitch_year_chunks_to_target,
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

# Operator-visible warning threshold: if fewer than this fraction of
# (HRU, day) cells in a year produce a valid bound, the build emits a
# WARNING with the actual fraction. Mirrors the per-year warning shape
# in aggregate/mod10c1.py:_log_low_valid_coverage so an upstream
# regression (all-NaN CI, fabric mismatch, fully-cloudy season) is
# loud rather than silently producing an all-zero n_sources year.
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
        )

    output_path = project.targets_dir() / sca_cfg["output_file"]
    unfilled_files = sorted(
        intermediates_dir.glob("sca_targets_[0-9][0-9][0-9][0-9].nc")
    )
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
        nn_files = sorted(
            intermediates_dir.glob("sca_targets_[0-9][0-9][0-9][0-9]_nn_filled.nc")
        )
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
) -> None:
    """Build SCA bounds for one calendar year and write per-year NCs.

    Idempotent: if both expected per-year NCs already exist
    (``sca_targets_<year>.nc`` and, when ``nn_fill``,
    ``sca_targets_<year>_nn_filled.nc``), the build is skipped — useful
    when re-running after a partial OOM mid-period.
    """
    year_unfilled = intermediates_dir / f"sca_targets_{year}.nc"
    year_nn = intermediates_dir / f"sca_targets_{year}_nn_filled.nc"
    if year_unfilled.exists() and ((not nn_fill) or year_nn.exists()):
        # Intermediates are path-keyed only — not fingerprinted by
        # ci_threshold or code version. Log the active threshold so an
        # operator re-reading the run output can cross-check it against
        # the file's global attrs (see module docstring warning).
        logger.info(
            "Year %d intermediates exist; skipping. Active ci_threshold=%.2f "
            "— if it has changed since the intermediates were written, "
            "rm %s/sca_targets_%d*.nc and re-run.",
            year,
            ci_threshold,
            intermediates_dir,
            year,
        )
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
    # ``valid`` alone would mis-flag the rare case where CI passed at
    # HRU scale but the pre-aggregation per-pixel CI gate dropped every
    # snow pixel — there the bound is NaN even though ``valid`` is True.
    # build_n_sources_attrs(1) → flag_values=[0, 1], flag_meanings="none one".
    valid_bound = valid & sca_obs.notnull()
    n_sources = valid_bound.astype(np.int8)

    # Operator-visible warning when an upstream regression or
    # genuinely fully-cloudy season produces near-zero coverage for
    # the year. Cheap single-scalar reduction; mirrors the per-year
    # warning shape in aggregate/mod10c1.py:_log_low_valid_coverage.
    frac_valid = float(valid_bound.mean().compute())
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
        extra_global_attrs={**extra_attrs, "year_chunk": year},
        hru_meta=hru_meta,
        nn_fill=nn_fill,
        nn_max_candidates=nn_max_candidates,
        id_col=id_col,
    )
