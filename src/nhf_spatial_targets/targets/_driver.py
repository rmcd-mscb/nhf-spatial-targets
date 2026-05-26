"""Generic target driver: consumes a TargetAdapter, runs the shared pipeline.

Two execution paths, gated by :attr:`TargetAdapter.year_chunked`:

- **Single-shot** (runoff, AET, recharge, soil moisture): compute the
  fabric geometry, call the adapter's ``source_loader`` once for the
  whole period, hand the result to :func:`_writers.write_bounds_target`.
- **Year-chunked** (SCA, SWE): compute the fabric geometry once, iterate
  over years from :func:`iter_period_years`, call the loader per year
  with a fingerprint-cached skip, then prune orphans and stitch
  intermediates into the canonical output via
  :func:`stitch_year_chunks_to_target`.

Source-loader callable shape (see :class:`SourceLoaderResult`)::

    loader(project, adapter, *, year_context=None) -> SourceLoaderResult

``year_context`` is ``None`` for single-shot builds; for year-chunked
builds it is ``(year, year_start_iso, year_end_iso)``. The loader is
the one place builder-specific code lives — unit shims, normalization,
multi-source NaN-aware min/max, fabric-scope filtering, etc. The driver
applies adapter-level concerns (forced-zero months, cache invalidation,
orphan pruning, stitching) uniformly.

The driver does NOT own SOM's two-variant output (monthly + annual) —
SOM builds its adapter twice and calls the driver twice, once per
variant. This keeps the driver's contract simple ("one adapter, one
output file") and leaves multi-variant orchestration in the only
builder that needs it.
"""

from __future__ import annotations

import logging

import xarray as xr

from nhf_spatial_targets.targets._adapter import SourceLoaderResult, TargetAdapter
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
    compute_hru_area_and_centroids,
    compute_hru_centroids,
    parse_period,
)
from nhf_spatial_targets.targets._writers import write_bounds_target
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


def _common_global_attrs(
    project: Project,
    adapter: TargetAdapter,
    period_str: str,
) -> dict:
    """Adapter-level extra attrs every builder emits.

    The per-target extras (``source``, ``normalize_period``, etc.)
    are layered on top by the loader's ``extra_attrs`` dict at write
    time.
    """
    return {
        "references": adapter.references,
        "fabric": project.config["fabric"]["path"],
        "fabric_sha256": project.fabric.get("sha256", ""),
        "period": period_str,
        "area_crs": project.area_crs,
    }


def _apply_forced_zero(
    adapter: TargetAdapter,
    result: SourceLoaderResult,
) -> SourceLoaderResult:
    """Force ``(lower, upper) = (0, 0)`` for cells in ``forced_zero_months``.

    Uses ``forced_zero_validity_var`` (when set) from ``result.extras``
    as the validity mask: only cells where the mask is True AND month
    matches are zeroed. SCA's calcSCA rule: only the CI-passing cells
    are forced to zero in July/August; sub-threshold cells stay NaN so
    downstream consumers can still distinguish "no observation" from
    "observed bare ground."
    """
    if not adapter.forced_zero_months:
        return result
    months = list(adapter.forced_zero_months)
    in_months = result.lower["time"].dt.month.isin(months)

    if adapter.forced_zero_validity_var is not None:
        valid = result.extras.get(adapter.forced_zero_validity_var)
        if valid is None:
            raise KeyError(
                f"TargetAdapter({adapter.target_key!r}): "
                f"forced_zero_validity_var={adapter.forced_zero_validity_var!r} "
                f"declared but absent from source_loader extras "
                f"(have: {sorted(result.extras)})."
            )
        mask = in_months & valid
    else:
        mask = in_months

    new_lower = xr.where(mask, 0.0, result.lower)
    new_upper = xr.where(mask, 0.0, result.upper)
    new_lower.attrs = dict(result.lower.attrs)
    new_upper.attrs = dict(result.upper.attrs)
    return SourceLoaderResult(
        lower=new_lower,
        upper=new_upper,
        n_sources=result.n_sources,
        n_sources_count=result.n_sources_count,
        time_index=result.time_index,
        time_offset_unit=result.time_offset_unit,
        extra_attrs=result.extra_attrs,
        extras=result.extras,
    )


def _adapter_logger(adapter: TargetAdapter) -> logging.Logger:
    """Per-adapter logger so driver-emitted lines route under the target.

    A per-target logger ("nhf_spatial_targets.targets.sca", etc.) means
    the driver's "skipping" / "Year-chunked build" messages appear under
    the target's own logger namespace — tests that pin caplog to that
    logger (e.g. SCA's `caplog.at_level(..., logger="nhf_spatial_targets.targets.sca")`)
    continue to see those lines without the driver being aware of who
    is watching.
    """
    return logging.getLogger(f"nhf_spatial_targets.targets.{adapter.target_key}")


def build(adapter: TargetAdapter, project: Project) -> None:
    """Build a target's canonical NC according to the adapter.

    Dispatches to :func:`build_single_shot` for unchunked targets and
    :func:`_build_year_chunked` otherwise.
    """
    target_cfg = project.target(adapter.config_key)
    period_str = target_cfg["period"]
    period = parse_period(period_str)

    if adapter.needs_hru_area:
        hru_meta = compute_hru_area_and_centroids(project)
    else:
        hru_meta = compute_hru_centroids(project)
    id_col = project.id_col

    if adapter.year_chunked:
        _build_year_chunked(
            adapter=adapter,
            project=project,
            target_cfg=target_cfg,
            period=period,
            period_str=period_str,
            hru_meta=hru_meta,
            id_col=id_col,
        )
    else:
        build_single_shot(
            adapter=adapter,
            project=project,
            target_cfg=target_cfg,
            period=period,
            period_str=period_str,
            hru_meta=hru_meta,
            id_col=id_col,
        )


def build_single_shot(
    *,
    adapter: TargetAdapter,
    project: Project,
    target_cfg: dict,
    period: tuple[str, str],
    period_str: str,
    hru_meta,
    id_col: str,
) -> None:
    """Single-shot driver path: load the period, combine, write one NC.

    Public seam for **multi-variant targets** — targets that emit more than
    one NC under a single ``config_key`` and therefore can't be driven by
    :func:`build` (whose contract is one adapter, one output file). Such
    targets declare one :class:`TargetAdapter` per variant and call this
    function directly in a loop, overriding ``target_cfg["output_file"]``
    per variant. See :func:`nhf_spatial_targets.targets.som.build` for the
    canonical example (monthly + annual soil-moisture targets).

    Single-variant targets should call :func:`build` instead and let it
    dispatch here on the adapter's ``year_chunked`` flag.
    """
    fabric_hru_ids = hru_meta.index.values

    result = adapter.source_loader(
        project=project,
        adapter=adapter,
        period=period,
        hru_meta=hru_meta,
        fabric_hru_ids=fabric_hru_ids,
        id_col=id_col,
        year_context=None,
    )
    result = _apply_forced_zero(adapter, result)

    extra_attrs = dict(_common_global_attrs(project, adapter, period_str))
    extra_attrs.update(result.extra_attrs)

    output_path = project.targets_dir() / target_cfg["output_file"]
    write_bounds_target(
        project=project,
        lower=result.lower,
        upper=result.upper,
        n_sources=result.n_sources,
        n_sources_count=result.n_sources_count,
        time_index=result.time_index,
        time_offset_unit=result.time_offset_unit,
        bounds_units=adapter.bounds_units,
        bounds_long_name_kind=adapter.bounds_long_name_kind,
        cell_methods=adapter.cell_methods,
        output_path=output_path,
        title=adapter.title,
        nn_title=adapter.nn_title,
        extra_global_attrs=extra_attrs,
        hru_meta=hru_meta,
        nn_fill=bool(target_cfg["nn_fill"]),
        nn_max_candidates=int(target_cfg["nn_max_candidates"]),
        id_col=id_col,
    )


def _build_year_chunked(
    *,
    adapter: TargetAdapter,
    project: Project,
    target_cfg: dict,
    period: tuple[str, str],
    period_str: str,
    hru_meta,
    id_col: str,
) -> None:
    """Year-chunked path: per-year build + cache skip + stitch."""
    fabric_hru_ids = hru_meta.index.values

    config_fp = target_config_fingerprint(project, adapter.config_key)
    code_ver = code_version_fingerprint()

    year_specs = iter_period_years(period[0], period[1])
    if not year_specs:
        raise ValueError(
            f"{adapter.config_key}.period {period_str} produces no years to build."
        )

    intermediates_dir = project.targets_dir() / adapter.intermediates_subdir
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    adapter_log = _adapter_logger(adapter)
    adapter_log.info(
        "Year-chunked build: %d years, intermediates -> %s "
        "(retained after stitch for forensic value; rm to reclaim disk)",
        len(year_specs),
        intermediates_dir,
    )

    nn_fill = bool(target_cfg["nn_fill"])
    nn_max_candidates = int(target_cfg["nn_max_candidates"])

    base_extra_attrs = dict(_common_global_attrs(project, adapter, period_str))
    base_extra_attrs.update(
        {
            INTERMEDIATE_CONFIG_FINGERPRINT_ATTR: config_fp,
            INTERMEDIATE_CODE_VERSION_ATTR: code_ver,
        }
    )

    last_loader_attrs: dict = {}
    for year, year_start, year_end in year_specs:
        year_period = (year_start, year_end)
        year_unfilled = intermediates_dir / f"{adapter.intermediate_base}_{year}.nc"
        year_nn = intermediates_dir / f"{adapter.intermediate_base}_{year}_nn_filled.nc"
        expected_paths = [year_unfilled] + ([year_nn] if nn_fill else [])
        skip, cached_attrs = should_skip_year_build(
            expected_paths,
            active_config_fingerprint=config_fp,
            active_code_version=code_ver,
            target_label=adapter.target_key,
            year=year,
            logger=adapter_log,
        )
        if skip:
            adapter_log.info(
                "Year %d intermediates valid (config=%s code=%s); skipping.",
                year,
                config_fp,
                code_ver,
            )
            if adapter.on_year_skip is not None:
                adapter.on_year_skip(
                    project=project,
                    adapter=adapter,
                    year=year,
                    cached_attrs=cached_attrs or {},
                )
            continue

        result = adapter.source_loader(
            project=project,
            adapter=adapter,
            period=year_period,
            hru_meta=hru_meta,
            fabric_hru_ids=fabric_hru_ids,
            id_col=id_col,
            year_context=(year, year_start, year_end),
        )
        result = _apply_forced_zero(adapter, result)
        last_loader_attrs = dict(result.extra_attrs)

        per_year_attrs = dict(base_extra_attrs)
        per_year_attrs.update(result.extra_attrs)
        per_year_attrs["year_chunk"] = year
        # Per-year diagnostic the loader may want persisted (e.g. SCA's
        # frac_valid_bound). Surfaced via result.extras["per_year_attrs"]
        # so it sits on the year's NC and survives a skip-branch re-read.
        loader_per_year = result.extras.get("per_year_attrs") or {}
        per_year_attrs.update(loader_per_year)

        per_year_title = (
            adapter.per_year_title_template.format(year=year)
            if adapter.per_year_title_template is not None
            else f"{adapter.title} year {year} (intermediate)"
        )
        per_year_nn_title = (
            adapter.per_year_nn_title_template.format(year=year)
            if adapter.per_year_nn_title_template is not None
            else f"{adapter.nn_title} year {year} (intermediate)"
        )
        write_bounds_target(
            project=project,
            lower=result.lower,
            upper=result.upper,
            n_sources=result.n_sources,
            n_sources_count=result.n_sources_count,
            time_index=result.time_index,
            time_offset_unit=result.time_offset_unit,
            bounds_units=adapter.bounds_units,
            bounds_long_name_kind=adapter.bounds_long_name_kind,
            cell_methods=adapter.cell_methods,
            output_path=year_unfilled,
            title=per_year_title,
            nn_title=per_year_nn_title,
            extra_global_attrs=per_year_attrs,
            hru_meta=hru_meta,
            nn_fill=nn_fill,
            nn_max_candidates=nn_max_candidates,
            id_col=id_col,
        )

    # Prune orphans + compute stitch input from year_specs (#211).
    in_period_years = {year for year, _, _ in year_specs}
    prune_orphan_year_intermediates(
        intermediates_dir,
        adapter.intermediate_base,
        in_period_years,
        target_label=adapter.target_key,
        logger=adapter_log,
    )

    output_path = project.targets_dir() / target_cfg["output_file"]
    unfilled_files = [
        intermediates_dir / f"{adapter.intermediate_base}_{year}.nc"
        for year, _, _ in year_specs
    ]
    # Stitch global attrs: adapter common + last loader's per-target extras
    # (which carry e.g. SCA's ci_threshold / summer_zero_months and SWE's
    # fabric_token / source list). Since every per-year file carries the
    # same set of these attrs (they're config-derived, not data-derived),
    # using the last loader's contribution is consistent with the existing
    # behaviour where the original builders built one extra_attrs dict at
    # the top of build() and used it for both per-year and stitch passes.
    stitch_attrs = dict(base_extra_attrs)
    stitch_attrs.update(last_loader_attrs)
    stitch_year_chunks_to_target(
        unfilled_files,
        output_path,
        title=adapter.title,
        extra_global_attrs=stitch_attrs,
        sort_dim=id_col,
    )

    if nn_fill:
        nn_files = [
            intermediates_dir / f"{adapter.intermediate_base}_{year}_nn_filled.nc"
            for year, _, _ in year_specs
        ]
        nn_path = output_path.with_name(
            output_path.stem + "_nn_filled" + output_path.suffix
        )
        nn_attrs = dict(stitch_attrs)
        nn_attrs["nn_fill_max_candidates"] = nn_max_candidates
        nn_attrs["nn_fill_distance_crs"] = project.area_crs
        stitch_year_chunks_to_target(
            nn_files,
            nn_path,
            title=adapter.nn_title,
            extra_global_attrs=nn_attrs,
            sort_dim=id_col,
        )


# Re-export for convenience: builders import this module's `build` symbol
# along with the adapter.
__all__ = ["build", "TargetAdapter", "SourceLoaderResult"]
