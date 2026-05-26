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
passing HRU, again mirroring calcSCA.

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

Because SCA's per-year action is a single ``_build_year`` function that
both computes and writes, the year loop is owned by ``build`` here
rather than by the generic driver — this keeps the unit of monkeypatch
at the function the test fixture replaces. The shared helpers
(skip-cache, write_bounds_target, orphan-prune, stitch) are still
imported from the driver-side modules so SCA participates in the
adapter pattern's metadata contract.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.targets._adapter import TargetAdapter
from nhf_spatial_targets.targets._intermediates import (
    INTERMEDIATE_CODE_VERSION_ATTR,
    INTERMEDIATE_CONFIG_FINGERPRINT_ATTR,
    code_version_fingerprint,
    iter_period_years,
    prune_orphan_year_intermediates,
    should_skip_year_build,
    stitch_year_chunks_to_target,
    target_config_fingerprint,
)
from nhf_spatial_targets.targets._io import (
    check_hru_coords,
    compute_hru_centroids,
    parse_period,
    read_aggregated_source,
    reindex_to_day_start,
)
from nhf_spatial_targets.targets._writers import write_bounds_target
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
# Adapter declaration (declarative target metadata; the driver-style adapter
# pattern means a future seventh target can mirror this block almost
# verbatim, replacing only the loader/per-year action).
# ---------------------------------------------------------------------------


def _noop_loader(**_kwargs):
    """SCA owns its year loop directly (see ``build``); this loader is unused.

    The adapter ``source_loader`` field is required by ``TargetAdapter``
    construction; SCA's adapter sets it to this no-op so the dataclass
    invariants are satisfied while ``build`` bypasses the generic
    year-chunked driver path. SWE uses the generic path; SCA does not
    because the test fixture monkeypatches ``_build_year`` and the unit
    of patching must be a module-level function, not an adapter field.
    """
    raise NotImplementedError(
        "targets.sca._noop_loader is a placeholder; SCA owns its year loop "
        "directly in build() and does not invoke the generic year-chunked driver."
    )


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
    source_loader=_noop_loader,
)


# ---------------------------------------------------------------------------
# Per-year build (the monkeypatch target for tests)
# ---------------------------------------------------------------------------


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
    the build is skipped via :func:`should_skip_year_build`. On a
    healthy skip, the cached ``frac_valid_bound`` global attr is read
    and the low-valid-coverage WARNING is re-emitted.
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

    sca_obs = reindex_to_day_start(snow_native / 100.0, year_master_idx)
    ci = reindex_to_day_start(ci_native / 100.0, year_master_idx)

    valid = ci >= ci_threshold
    lower = xr.where(valid, ci * sca_obs, np.nan)
    upper = xr.where(valid, lower + (1.0 - ci), np.nan)

    # July/August forced-zero only where the CI gate passed.
    summer = lower["time"].dt.month.isin(list(_SUMMER_ZERO_MONTHS))
    lower = xr.where(summer & valid, 0.0, lower)
    upper = xr.where(summer & valid, 0.0, upper)

    valid_bound = valid & sca_obs.notnull()
    n_sources = valid_bound.astype(np.int8)

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
        bounds_units=ADAPTER.bounds_units,
        bounds_long_name_kind=ADAPTER.bounds_long_name_kind,
        cell_methods=ADAPTER.cell_methods,
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


def build(project: Project) -> None:
    """Build the SCA calibration target.

    Year-chunked: each year reads its own slice of the aggregated NCs,
    computes per-day bounds, and writes a per-year intermediate. The
    intermediates are then stitched into the canonical
    ``sca_targets.nc``.

    Year-loop is owned directly by this module (rather than delegated
    to the generic year-chunked driver) so that test fixtures can
    ``monkeypatch.setattr(sca, "_build_year", no_op)`` to exercise
    the stitch-only path. Single-shot builders (runoff/aet/rch) and
    SWE delegate to the generic ``targets._driver.build``.
    """
    sca_cfg = project.target(ADAPTER.config_key)
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

    config_fp = target_config_fingerprint(project, ADAPTER.config_key)
    code_ver = code_version_fingerprint()

    extra_attrs = {
        "source": (
            "MOD10C1 v061 Day_CMG_Snow_Cover + Day_CMG_Clear_Index "
            "(CI-bounded per TM 6-B10 / PRMSobjfun.f90:calcSCA)"
        ),
        "references": ADAPTER.references,
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

    intermediates_dir = project.targets_dir() / ADAPTER.intermediates_subdir
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

    in_period_years = {year for year, _, _ in year_specs}
    prune_orphan_year_intermediates(
        intermediates_dir,
        ADAPTER.intermediate_base,
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
        title=ADAPTER.title,
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
            title="NHM SCA calibration target (NN-filled, fractional 0-1)",
            extra_global_attrs=nn_attrs,
            sort_dim=id_col,
        )
