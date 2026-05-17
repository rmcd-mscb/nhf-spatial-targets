"""Build SWE calibration targets from Daymet + SNODAS + ERA5-Land sd + Margulis.

Up to four daily-cadence sources contribute to per-HRU per-day bounds in
``inches`` (per ``catalog/variables.yml`` → ``snow_water_equivalent.units``,
matching the PRMS ``pkwater_equiv`` PUNIT):

  - Daymet V4 R1 ``swe``       (kg m⁻² ≡ mm water-eq, daily)
  - SNODAS ``swe``             (kg m⁻² ≡ mm water-eq, daily)
  - ERA5-Land ``sd``           (m water-eq, daily — instantaneous snapshot)
  - Margulis WUS-SR ``SWE``    (m water-eq, daily — Oregon fabric only)

Per-source pipeline (every shim ends in mm; the target then converts
mm → inches in a single linear step):

  - daymet: identity (already mm)
  - snodas: identity (already mm)
  - era5_land sd: × 1000 (m → mm)
  - margulis_wus_sr SWE: × 1000 (m → mm)

After mm conversion, sources are stacked on a ``source`` dim and reduced
with NaN-aware min/max so a bound is defined whenever ≥1 source is finite
at that (HRU, day). An int8 ``n_sources`` diagnostic is written
alongside. ``snow_water_equivalent.sources`` controls which sources
contribute, so dropping a source from the bound is a one-line config
change rather than a code edit.

**Fabric scope enforcement.** Margulis WUS-SR carries
``fabric_scope.fabrics: ["or"]`` in ``catalog/sources.yml`` — it should
only contribute on the Oregon fabric. The builder reads the project's
``fabric.token`` (e.g. ``"or"``) and silently skips any requested source
whose ``fabric_scope`` does not include that token (with a clear log
line). Projects whose ``fabric.token`` is unset behave as if the token
were "(unset)" — which means any source carrying ``fabric_scope`` is
skipped. This is the safe default for non-Oregon fabrics that
accidentally inherit the four-source SWE default from
``defaults.py``.

If ``snow_water_equivalent.nn_fill`` is True (default), a second file
``<output>_nn_filled.nc`` is written with bound NaNs filled by the
nearest finite HRU's value at the same day (cKDTree donor walk in
``project.area_crs``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets import catalog
from nhf_spatial_targets.targets._common import (
    SourceShim,
    check_hru_coords,
    compute_hru_centroids,
    iter_period_years,
    multi_source_nanminmax,
    parse_period,
    read_aggregated_source,
    reindex_to_day_start,
    shims_by_config_label,
    stitch_year_chunks_to_target,
    write_bounds_target,
)
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


# 1 inch = 25.4 mm.
_MM_PER_INCH = 25.4


# ---------------------------------------------------------------------------
# Per-source unit shims (mm is the common intermediate)
# ---------------------------------------------------------------------------


def daymet_to_mm(da: xr.DataArray) -> xr.DataArray:
    """Daymet ``swe`` is already mm (kg m⁻² ≡ mm water-eq) — pass through."""
    out = da.copy()
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


def snodas_to_mm(da: xr.DataArray) -> xr.DataArray:
    """SNODAS ``swe`` is already mm (kg m⁻² ≡ mm water-eq) — pass through."""
    out = da.copy()
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


def era5_sd_to_mm(da: xr.DataArray) -> xr.DataArray:
    """ERA5-Land ``sd`` (m water-eq) → mm via × 1000."""
    out = da * 1000.0
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


def margulis_to_mm(da: xr.DataArray) -> xr.DataArray:
    """Margulis ``SWE`` (m water-eq) → mm via × 1000."""
    out = da * 1000.0
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


# Per-source registry. The ERA5-Land sd shim has ``source_key="era5_land_sd"``
# (the on-disk storage key, matching aggregate/era5_land.py:ADAPTER_SD's
# output dir under <project>/data/aggregated/era5_land_sd/) but
# ``config_label="era5_land"`` so the project config can keep a single
# logical "era5_land" entry in ``snow_water_equivalent.sources``. The
# label→shim map is derived from this tuple at module load via
# :func:`shims_by_config_label` — there is no parallel dict to keep in
# sync.
SHIMS: tuple[SourceShim, ...] = (
    SourceShim(
        source_key="daymet",
        aggregated_var="swe",
        description="Daymet V4 R1 swe (kg/m² ≡ mm, daily)",
        to_common_units=daymet_to_mm,
    ),
    SourceShim(
        source_key="snodas",
        aggregated_var="swe",
        description="SNODAS swe (kg/m² ≡ mm, daily)",
        to_common_units=snodas_to_mm,
    ),
    SourceShim(
        source_key="era5_land_sd",
        aggregated_var="sd",
        description="ERA5-Land sd (m → mm, daily snapshot)",
        to_common_units=era5_sd_to_mm,
        config_label="era5_land",
    ),
    SourceShim(
        source_key="margulis_wus_sr",
        aggregated_var="SWE",
        description="Margulis WUS-SR SWE (m → mm, daily; OR fabric only)",
        to_common_units=margulis_to_mm,
    ),
)


def mm_to_inches(da: xr.DataArray) -> xr.DataArray:
    """Convert mm → inches (linear, ÷ 25.4)."""
    out = da / _MM_PER_INCH
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "inches"
    return out


# ---------------------------------------------------------------------------
# Fabric scope filter
# ---------------------------------------------------------------------------


def _filter_sources_by_fabric_scope(
    requested: list[str], fabric_token: str | None
) -> list[str]:
    """Drop sources whose ``fabric_scope`` excludes this project's fabric.

    Reads each requested source's ``catalog.source(...)['fabric_scope']``
    (validated via :func:`catalog.validate_fabric_scope`) and keeps only
    those whose ``fabrics`` list contains ``fabric_token``. Sources with
    no ``fabric_scope`` block pass through unconditionally. A ``None``
    fabric token causes every fabric-scoped source to be dropped, which
    is the safe default for projects that haven't explicitly identified
    their fabric.
    """
    kept: list[str] = []
    for src in requested:
        meta = catalog.source(src)
        scope = meta.get("fabric_scope")
        catalog.validate_fabric_scope(src, scope)
        if scope is None:
            kept.append(src)
            continue
        scope_fabrics = list(scope.get("fabrics") or [])
        if fabric_token is not None and fabric_token in scope_fabrics:
            kept.append(src)
        else:
            logger.info(
                "swe: skipping source '%s' — fabric_scope=%s, project fabric token=%r",
                src,
                scope_fabrics,
                fabric_token,
            )
    return kept


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build(project: Project) -> None:
    """Build the SWE calibration target.

    Reads each enabled source's per-year aggregated NCs, harmonizes time
    coords onto a master day-start index over
    ``snow_water_equivalent.period``, converts each to inches, combines
    via NaN-aware min/max, and writes a CF-1.6 NetCDF. If
    ``snow_water_equivalent.nn_fill`` is True, additionally writes
    ``<output>_nn_filled.nc``.
    """
    swe_cfg = project.target("snow_water_equivalent")
    period = parse_period(swe_cfg["period"])
    requested_sources = list(swe_cfg["sources"])
    fabric_cfg = project.config.get("fabric") or {}
    fabric_token = fabric_cfg.get("token")
    shims = shims_by_config_label(SHIMS)

    # Validate every requested source against the SHIMS registry BEFORE
    # any catalog or fabric_scope lookups, so an unknown source name
    # surfaces a "Known: ..." message rather than a confusing KeyError
    # from the catalog or a "zero sources after filtering" message
    # after silent drops.
    for src in requested_sources:
        if src not in shims:
            raise ValueError(
                f"snow_water_equivalent.sources includes unknown source "
                f"'{src}'. Known: {sorted(shims)}"
            )

    # Validate the project's fabric token if set, to catch typos like
    # "oregon" instead of "or" before silently filtering every fabric-
    # scoped source out.
    if fabric_token is not None and fabric_token not in catalog.FABRIC_SCOPE_TOKENS:
        raise ValueError(
            f"fabric.token={fabric_token!r} is not a recognised fabric token. "
            f"Allowed: {sorted(catalog.FABRIC_SCOPE_TOKENS)}. To add a new "
            f"fabric, extend catalog.FABRIC_SCOPE_TOKENS."
        )

    sources = _filter_sources_by_fabric_scope(requested_sources, fabric_token)
    if not sources:
        raise ValueError(
            f"snow_water_equivalent.sources={requested_sources!r} resolved to "
            f"zero sources after fabric_scope filtering (fabric token="
            f"{fabric_token!r}). Add a non-scoped source or set fabric.token."
        )

    logger.info(
        "Building SWE target: %d sources (%s) [requested %d (%s)], "
        "period %s..%s, fabric=%s (token=%r)",
        len(sources),
        ",".join(sources),
        len(requested_sources),
        ",".join(requested_sources),
        period[0],
        period[1],
        project.config["fabric"]["path"],
        fabric_token,
    )

    # 1. Per-HRU centroids (only centroids needed for NN-fill — SWE has
    # no per-HRU-area unit conversion). Computed once and reused across
    # every per-year build to amortise the fabric load (~361k polygons
    # on gfv2).
    hru_meta = compute_hru_centroids(project)
    id_col = project.id_col
    fabric_hru_ids = hru_meta.index.values

    # 2. Common global attrs (same on per-year intermediates and the
    # stitched final NC).
    extra_attrs = {
        "source": "; ".join(shims[s].description for s in sources),
        "references": (
            "Hay et al. 2022, doi:10.3133/tm6B10; Markstrom et al. 2015, TM 6-B7"
        ),
        "fabric": project.config["fabric"]["path"],
        "fabric_sha256": project.fabric.get("sha256", ""),
        "fabric_token": fabric_token or "",
        "period": swe_cfg["period"],
        "area_crs": project.area_crs,
    }

    # 3. Year-chunked build. SWE is the only daily-cadence multi-source
    # target in the pipeline; materialising the full assembled Dataset
    # in one shot blows past the large-mem partition (~575 GB peak for
    # 46 yrs × gfv2 × 3 sources). The year-chunked path keeps peak
    # bounded by one year's worth of data regardless of period length.
    year_specs = iter_period_years(period[0], period[1])
    if not year_specs:
        raise ValueError(
            f"snow_water_equivalent.period {swe_cfg['period']} produces "
            f"no years to build."
        )

    intermediates_dir = project.targets_dir() / ".swe_intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Year-chunked build: %d years, intermediates -> %s "
        "(retained after stitch for forensic value; rm to reclaim disk)",
        len(year_specs),
        intermediates_dir,
    )

    nn_fill = bool(swe_cfg["nn_fill"])
    nn_max_candidates = int(swe_cfg["nn_max_candidates"])

    for year, year_start, year_end in year_specs:
        _build_year(
            project=project,
            year=year,
            year_period=(year_start, year_end),
            sources=sources,
            shims=shims,
            hru_meta=hru_meta,
            fabric_hru_ids=fabric_hru_ids,
            id_col=id_col,
            n_sources_count=len(sources),
            extra_attrs=extra_attrs,
            intermediates_dir=intermediates_dir,
            nn_fill=nn_fill,
            nn_max_candidates=nn_max_candidates,
        )

    # 4. Stitch per-year intermediates into the canonical single-file
    # outputs. Dask-streamed so peak memory stays bounded.
    output_path = project.targets_dir() / swe_cfg["output_file"]
    unfilled_files = sorted(
        intermediates_dir.glob("swe_targets_[0-9][0-9][0-9][0-9].nc")
    )
    stitch_year_chunks_to_target(
        unfilled_files,
        output_path,
        title="NHM SWE calibration target (lower/upper bounds in inches)",
        extra_global_attrs=extra_attrs,
        sort_dim=id_col,
    )

    if nn_fill:
        nn_files = sorted(
            intermediates_dir.glob("swe_targets_[0-9][0-9][0-9][0-9]_nn_filled.nc")
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
            title="NHM SWE calibration target (NN-filled, inches)",
            extra_global_attrs=nn_attrs,
            sort_dim=id_col,
        )


def _build_year(
    *,
    project: Project,
    year: int,
    year_period: tuple[str, str],
    sources: list[str],
    shims: dict[str, SourceShim],
    hru_meta: pd.DataFrame,
    fabric_hru_ids: np.ndarray,
    id_col: str,
    n_sources_count: int,
    extra_attrs: dict,
    intermediates_dir: Path,
    nn_fill: bool,
    nn_max_candidates: int,
) -> None:
    """Build SWE bounds for one calendar year and write per-year NCs.

    Per-year contributions follow the period-union semantics — sources
    whose coverage doesn't include this year are silently skipped (with
    a log line) and contribute NaN to that year's bound. The ``n_sources``
    diagnostic carries the per-cell count so downstream consumers can
    filter, e.g., ``ds.where(ds.n_sources >= 2)`` to require 2-source
    agreement, or ``>= 3`` to require SNODAS-era completeness.

    Idempotent: if both expected per-year NCs already exist
    (``swe_targets_<year>.nc`` and, when ``nn_fill``,
    ``swe_targets_<year>_nn_filled.nc``), the build is skipped — useful
    when re-running after a partial OOM mid-period.
    """
    year_unfilled = intermediates_dir / f"swe_targets_{year}.nc"
    year_nn = intermediates_dir / f"swe_targets_{year}_nn_filled.nc"
    if year_unfilled.exists() and ((not nn_fill) or year_nn.exists()):
        logger.info("Year %d intermediates exist; skipping", year)
        return

    year_master_idx = pd.date_range(year_period[0], year_period[1], freq="D")
    if len(year_master_idx) == 0:
        raise ValueError(
            f"Year {year}: empty master index from period {year_period!r}."
        )

    year_sources: dict[str, xr.DataArray] = {}
    for src_label in sources:
        shim = shims[src_label]
        try:
            da_native = read_aggregated_source(
                project,
                shim.source_key,
                shim.aggregated_var,
                year_period,
                chunks={"time": 365, id_col: -1},
            )
        except ValueError as exc:
            if "entirely outside source coverage" in str(exc):
                logger.info(
                    "swe year %d: source '%s' has no data; contributes NaN",
                    year,
                    src_label,
                )
                continue
            raise
        check_hru_coords(da_native, fabric_hru_ids, id_col, src_label)
        da_mm = shim.to_common_units(da_native)
        da_in = mm_to_inches(da_mm)
        year_sources[src_label] = reindex_to_day_start(da_in, year_master_idx)

    if not year_sources:
        raise ValueError(
            f"swe year {year}: no source contributed any data for the year. "
            f"Either the period is set outside every source's coverage or "
            f"every aggregated NC is missing for this year."
        )

    lower, upper, n_sources = multi_source_nanminmax(year_sources)

    write_bounds_target(
        project=project,
        lower=lower,
        upper=upper,
        n_sources=n_sources,
        n_sources_count=n_sources_count,
        time_index=year_master_idx,
        time_offset_unit=pd.offsets.Day(1),
        bounds_units="inches",
        bounds_long_name_kind="daily SWE",
        cell_methods="time: point",
        output_path=year_unfilled,
        title=f"NHM SWE calibration target year {year} (intermediate)",
        nn_title=f"NHM SWE calibration target year {year} (NN-filled intermediate)",
        extra_global_attrs={**extra_attrs, "year_chunk": year},
        hru_meta=hru_meta,
        nn_fill=nn_fill,
        nn_max_candidates=nn_max_candidates,
        id_col=id_col,
    )
