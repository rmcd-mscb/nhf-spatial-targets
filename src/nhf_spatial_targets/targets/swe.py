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

import pandas as pd
import xarray as xr

from nhf_spatial_targets import catalog
from nhf_spatial_targets.targets._common import (
    SourceShim,
    check_hru_coords,
    compute_hru_centroids,
    multi_source_nanminmax,
    parse_period,
    read_aggregated_source,
    reindex_to_day_start,
    shims_by_config_label,
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
    # no per-HRU-area unit conversion).
    hru_meta = compute_hru_centroids(project)
    id_col = project.id_col

    # 2. Master day-start index over the requested period.
    master_idx = pd.date_range(period[0], period[1], freq="D")
    if len(master_idx) == 0:
        raise ValueError(
            f"snow_water_equivalent.period {swe_cfg['period']} produces no "
            f"days at freq='D'. Check the date range."
        )

    # 3. Read, convert, reindex each source.
    fabric_hru_ids = hru_meta.index.values
    sources_in_inches: dict[str, xr.DataArray] = {}
    for src_label in sources:
        shim = shims[src_label]
        da_native = read_aggregated_source(
            project,
            shim.source_key,
            shim.aggregated_var,
            period,
            # Daily cadence; one year per chunk keeps the working set
            # bounded without exploding the chunk count.
            chunks={"time": 365, id_col: -1},
        )
        check_hru_coords(da_native, fabric_hru_ids, id_col, src_label)
        da_mm = shim.to_common_units(da_native)
        da_in = mm_to_inches(da_mm)
        sources_in_inches[src_label] = reindex_to_day_start(da_in, master_idx)

    # 4. NaN-aware combination across sources.
    lower, upper, n_sources = multi_source_nanminmax(sources_in_inches)

    # 5. Assemble + write (with optional NN-fill companion).
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
    output_path = project.targets_dir() / swe_cfg["output_file"]
    write_bounds_target(
        project=project,
        lower=lower,
        upper=upper,
        n_sources=n_sources,
        n_sources_count=len(sources),
        time_index=master_idx,
        time_offset_unit=pd.offsets.Day(1),
        bounds_units="inches",
        bounds_long_name_kind="daily SWE",
        cell_methods="time: point",
        output_path=output_path,
        title="NHM SWE calibration target (lower/upper bounds in inches)",
        nn_title="NHM SWE calibration target (NN-filled, inches)",
        extra_global_attrs=extra_attrs,
        hru_meta=hru_meta,
        nn_fill=swe_cfg["nn_fill"],
        nn_max_candidates=int(swe_cfg["nn_max_candidates"]),
        id_col=id_col,
    )
