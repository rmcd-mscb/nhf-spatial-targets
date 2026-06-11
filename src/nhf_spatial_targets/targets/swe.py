"""Build SWE targets from Daymet + SNODAS + ERA5-Land sd + Margulis + UA SWE.

Up to five daily-cadence sources contribute to per-HRU per-day bounds in
``inches`` (per ``catalog/variables.yml`` → ``snow_water_equivalent.units``,
matching the PRMS ``pkwater_equiv`` PUNIT):

  - Daymet V4 R1 ``swe``       (kg m⁻² ≡ mm water-eq, daily)
  - SNODAS ``swe``             (kg m⁻² ≡ mm water-eq, daily)
  - ERA5-Land ``sd``           (m water-eq, daily — instantaneous snapshot)
  - Margulis WUS-SR ``SWE``    (m water-eq, daily — Western US coverage)
  - UA SWE (NSIDC-0719) ``swe``  (kg m⁻² ≡ mm water-eq, daily; CONUS 1982–2022)

UA SWE (NSIDC-0719, the University of Arizona daily 4-km product) reaches
back to **1982** (calendar years 1982–2022, re-windowed at consolidate
from water years 1982–2023), far earlier than SNODAS (2003+), so it widens
the pre-2003 SWE bound where the other CONUS sources are thin — the
original motivation for adding it (#237). Because the combine is NaN-aware
min/max over the *union* of sources (see below), a source contributes only
in the years its coverage spans; ``snow_water_equivalent.period`` may run
earlier than 2003 without making the bound all-NaN.

Per-source pipeline (every shim ends in mm; the target then converts
mm → inches in a single linear step):

  - daymet: identity (already mm)
  - snodas: identity (already mm)
  - era5_land sd: × 1000 (m → mm)
  - margulis_wus_sr SWE: × 1000 (m → mm)
  - ua_swe swe: identity (already mm, kg m⁻² ≡ mm)

After mm conversion, sources are stacked on a ``source`` dim and reduced
with NaN-aware min/max so a bound is defined whenever ≥1 source is finite
at that (HRU, day). An int8 ``n_sources`` diagnostic is written
alongside. ``snow_water_equivalent.sources`` controls which sources
contribute, so dropping a source from the bound is a one-line config
change rather than a code edit.

**Partial spatial coverage.** Margulis WUS-SR covers only the Western
US; the aggregation driver (#309) emits its aggregated NCs reindexed to
the full fabric with honest NaN at uncovered HRUs. Because the combine
is NaN-aware, the source contributes wherever it is finite and drops
out elsewhere — no configuration needed. The former catalog
``fabric_scope`` / config ``fabric.token`` gate was removed in #309.

If ``snow_water_equivalent.nn_fill`` is True (default), a second file
``<output>_nn_filled.nc`` is written with bound NaNs filled by the
nearest finite HRU's value at the same day (cKDTree donor walk in
``project.area_crs``).

**Cache invalidation.** Per-year intermediates under
``<project>/targets/.swe_intermediates/`` carry two fingerprint global
attrs (``config_fingerprint``, ``code_version``) that the skip branch
compares against the active values. A mismatch — caused by a config
edit (``sources``, ``period``, ``nn_max_candidates``), a fabric swap,
or a ``__version__`` bump — logs a WARNING, deletes the stale
intermediate, and rebuilds the year. Operators do not need to manually
``rm`` after config or version-bump changes; any commit that modifies
builder logic WITHOUT a ``__version__`` bump still requires a manual
``rm`` because the ``code_version`` tag is tied to the package
version string.

**Period-shrink defense.** Downward changes to ``period`` are
handled automatically by ``prune_orphan_year_intermediates`` plus a
``iter_period_years``-derived stitch input list — see the helper
docstring in ``targets/_intermediates.py`` (#211).
"""

from __future__ import annotations

import logging

import pandas as pd
import xarray as xr

from nhf_spatial_targets.targets._adapter import (
    SourceLoaderResult,
    TargetAdapter,
)
from nhf_spatial_targets.targets._combine import multi_source_nanminmax
from nhf_spatial_targets.targets._io import (
    check_hru_coords,
    read_aggregated_source,
    reindex_to_day_start,
)
from nhf_spatial_targets.targets._shims import (
    SourceShim,
    shims_by_config_label,
    validate_source_units,
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


def ua_swe_to_mm(da: xr.DataArray) -> xr.DataArray:
    """UA SWE ``swe`` is already mm (kg m⁻² ≡ mm water-eq) — pass through."""
    out = da.copy()
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


# Per-source registry. The ERA5-Land sd shim has ``source_key="era5_land_sd"``
# (the on-disk storage key, matching aggregate/era5_land.py:ADAPTER_SD's
# output dir under <project>/data/aggregated/era5_land_sd/) but
# ``config_label="era5_land"`` so the project config can keep a single
# logical "era5_land" entry in ``snow_water_equivalent.sources``.
SHIMS: tuple[SourceShim, ...] = (
    SourceShim(
        source_key="daymet",
        aggregated_var="swe",
        description="Daymet V4 R1 swe (kg/m² ≡ mm, daily)",
        to_common_units=daymet_to_mm,
        expected_cf_units="kg m-2",
    ),
    SourceShim(
        source_key="snodas",
        aggregated_var="swe",
        description="SNODAS swe (kg/m² ≡ mm, daily)",
        to_common_units=snodas_to_mm,
        expected_cf_units="kg m-2",
    ),
    SourceShim(
        source_key="era5_land_sd",
        aggregated_var="sd",
        description="ERA5-Land sd (m → mm, daily snapshot)",
        to_common_units=era5_sd_to_mm,
        config_label="era5_land",
        expected_cf_units="m",
    ),
    SourceShim(
        source_key="margulis_wus_sr",
        aggregated_var="SWE",
        description="Margulis WUS-SR SWE (m → mm, daily; WUS coverage)",
        to_common_units=margulis_to_mm,
        expected_cf_units="m",
    ),
    SourceShim(
        source_key="ua_swe",
        aggregated_var="swe",
        description="UA SWE NSIDC-0719 (kg/m² ≡ mm, daily; CONUS 1982–2022)",
        to_common_units=ua_swe_to_mm,
        expected_cf_units="kg m-2",
    ),
)


def mm_to_inches(da: xr.DataArray) -> xr.DataArray:
    """Convert mm → inches (linear, ÷ 25.4)."""
    out = da / _MM_PER_INCH
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "inches"
    return out


# ---------------------------------------------------------------------------
# Availability filter (single-pass before per-year loop)
# ---------------------------------------------------------------------------


def _filter_sources_by_availability(
    project: Project, requested: list[str], shims: dict[str, SourceShim]
) -> list[str]:
    """Drop sources whose aggregated NC directory is empty or missing."""
    kept: list[str] = []
    for src in requested:
        shim = shims[src]
        agg_dir = project.aggregated_dir() / shim.source_key
        pattern = f"{shim.source_key}_*_agg.nc"
        if not any(agg_dir.glob(pattern)):
            logger.warning(
                "swe: skipping source '%s' — no aggregated NCs found under %s "
                "(pattern: %s). Run "
                "'pixi run nhf-targets agg %s --project-dir %s' to include it. "
                "If that command reports no spatial overlap with this fabric, "
                "the source cannot contribute here.",
                src,
                agg_dir,
                pattern,
                shim.source_key.replace("_", "-"),
                project.workdir,
            )
            continue
        kept.append(src)
    return kept


def _resolve_sources(project: Project) -> tuple[list[str], list[str]]:
    """Resolve the effective source list for the current build.

    Returns ``(effective_sources, requested_sources)``. Called once by
    ``build`` for logging and once per year by the loader — the
    re-resolution is a cheap directory scan, and the inputs are
    config-derived and constant across years.
    """
    swe_cfg = project.target("snow_water_equivalent")
    requested = list(swe_cfg["sources"])
    shims = shims_by_config_label(SHIMS)

    validate_source_units(SHIMS, requested)

    sources = _filter_sources_by_availability(project, requested, shims)
    if not sources:
        raise ValueError(
            f"snow_water_equivalent.sources={requested!r} resolved to "
            f"zero sources after dropping unaggregated sources. Run "
            f"'pixi run nhf-targets agg <source> --project-dir {project.workdir}' "
            f"for at least one requested source before building the SWE target."
        )

    return sources, requested


def _load_year(
    *,
    project: Project,
    adapter: TargetAdapter,
    period: tuple[str, str],
    hru_meta,
    fabric_hru_ids,
    id_col: str,
    year_context,
) -> SourceLoaderResult:
    """SWE per-year loader.

    Year-scoped read of every enabled source, NaN-aware min/max combine,
    mm → inches conversion. Per-year source-coverage gaps surface as INFO
    logs and contribute NaN to that year's bound.
    """
    sources, _ = _resolve_sources(project)
    shims = shims_by_config_label(SHIMS)
    year, year_start, year_end = year_context

    year_master_idx = pd.date_range(year_start, year_end, freq="D")
    if len(year_master_idx) == 0:
        raise ValueError(
            f"Year {year}: empty master index from period ({year_start}, {year_end})."
        )

    year_sources: dict[str, xr.DataArray] = {}
    for src_label in sources:
        shim = shims[src_label]
        try:
            da_native = read_aggregated_source(
                project,
                shim.source_key,
                shim.aggregated_var,
                (year_start, year_end),
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

    extra_attrs = {
        "source": "; ".join(shims[s].description for s in sources),
    }

    return SourceLoaderResult(
        lower=lower,
        upper=upper,
        n_sources=n_sources,
        n_sources_count=len(sources),
        time_index=year_master_idx,
        time_offset_unit=pd.offsets.Day(1),
        extra_attrs=extra_attrs,
    )


# ---------------------------------------------------------------------------
# Adapter declaration
# ---------------------------------------------------------------------------


ADAPTER = TargetAdapter(
    target_key="swe",
    config_key="snow_water_equivalent",
    cadence="daily",
    bounds_units="inches",
    bounds_long_name_kind="daily SWE",
    cell_methods="time: point",
    title="NHM SWE calibration target (lower/upper bounds in inches)",
    nn_title="NHM SWE calibration target (NN-filled, inches)",
    references=("Hay et al. 2022, doi:10.3133/tm6B10; Markstrom et al. 2015, TM 6-B7"),
    year_chunked=True,
    intermediates_subdir=".swe_intermediates",
    intermediate_base="swe_targets",
    source_loader=_load_year,
    per_year_title_template="NHM SWE calibration target year {year} (intermediate)",
    per_year_nn_title_template=(
        "NHM SWE calibration target year {year} (NN-filled intermediate)"
    ),
)


def build(project: Project) -> None:
    """Build the SWE calibration target.

    Reads each enabled source's per-year aggregated NCs, harmonizes time
    coords onto a master day-start index over
    ``snow_water_equivalent.period``, converts each to inches, combines
    via NaN-aware min/max, and writes a CF-1.6 NetCDF. If
    ``snow_water_equivalent.nn_fill`` is True, additionally writes
    ``<output>_nn_filled.nc``.

    Thin wrapper around :func:`targets._driver.build`. Source filtering
    (availability) runs inside the loader so it is
    visible in test fixtures that drive the loader directly without the
    full driver. Per-year contributions follow the period-union
    semantics — sources whose coverage doesn't include a given year are
    silently skipped (with a log line) and contribute NaN to that
    year's bound.
    """
    from nhf_spatial_targets.targets._driver import build as run_driver

    sources, requested = _resolve_sources(project)
    swe_cfg = project.target("snow_water_equivalent")
    logger.info(
        "Building SWE target: %d sources (%s) [requested %d (%s)], "
        "period %s, fabric=%s",
        len(sources),
        ",".join(sources),
        len(requested),
        ",".join(requested),
        swe_cfg["period"],
        project.config["fabric"]["path"],
    )
    run_driver(ADAPTER, project)
