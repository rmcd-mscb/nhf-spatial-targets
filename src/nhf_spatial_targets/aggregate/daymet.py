"""Aggregate Daymet V4 R1 daily SWE from a local zarr to HRU fabric polygons.

Daymet is distributed as a single multi-year zarr store per region
(``daymet_na.zarr``, ``daymet_hi.zarr``, ``daymet_pr.zarr``); the path
is recorded in ``manifest.json`` by ``fetch.daymet``. The grid is
Lambert Conformal Conic with projected ``x``, ``y`` coords in metres,
so the standard ``SourceAdapter`` / ``aggregate_source`` driver
(which globs per-year NCs with CF lat/lon coords) does not apply. We
follow the ssebop precedent: a custom year loop that reuses the
driver's weight-cache, gdptools, atomic-write, and manifest helpers.

This PR supports the NA region only. ``--region hi`` / ``--region pr``
raise ``NotImplementedError`` until a fabric that needs them lands
(tracked under issue #101).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import xarray as xr
from pyproj import CRS

from nhf_spatial_targets import catalog
from nhf_spatial_targets.aggregate._driver import (
    _atomic_write_netcdf,
    _attach_cf_global_attrs,
    _migrate_legacy_layout,
    _verify_year_coverage,
    aggregate_variables_for_batch,
    compute_or_load_weights,
    load_and_batch_fabric,
    update_manifest,
)
from nhf_spatial_targets.fetch._period import parse_period
from nhf_spatial_targets.workspace import Project
from nhf_spatial_targets.workspace import load as _load_project

logger = logging.getLogger(__name__)

_SOURCE_KEY = "daymet"
_SOURCE_VAR = "swe"
_GRID_MAPPING_VAR = "lambert_conformal_conic"
_SUPPORTED_REGIONS = frozenset({"na"})


def _region_source_key(region: str) -> str:
    """Synthetic source_key used for region-segregated weight caches + globs.

    Per-region weight caches and per-region year-coverage checks must
    not collide across regions. We pass ``"daymet_<region>"`` to the
    weight-cache and ``_verify_year_coverage`` helpers so caches land
    at ``weights/daymet_na_batch<i>.csv`` and the coverage glob picks
    up only that region's files. Manifest keying stays ``"daymet"``
    (matches catalog source_key) so provenance reads naturally.
    """
    return f"{_SOURCE_KEY}_{region}"


def _region_year_path(project: Project, region: str, year: int) -> Path:
    """Per-(region, year) output path: ``<project>/data/aggregated/daymet/daymet_<region>_<year>_agg.nc``.

    Region is encoded in the filename (not a subdirectory) so HI/PR
    can land alongside NA without disturbing existing artifacts.
    """
    return (
        project.workdir
        / "data"
        / "aggregated"
        / _SOURCE_KEY
        / f"{_SOURCE_KEY}_{region}_{year}_agg.nc"
    )


def _resolve_zarr_path(project: Project, region: str) -> Path:
    """Resolve the zarr path for ``region`` from ``manifest.json``.

    Reads ``manifest["sources"]["daymet"]["regions"][region]["path"]``,
    populated by ``fetch.daymet._update_manifest``. The aggregator does
    not re-resolve ``daymet_root`` from ``config.yml`` — the manifest
    is the post-fetch source of truth and decouples this module from
    the fetch-side path-resolution rules.
    """
    manifest_path = project.manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"daymet: manifest.json not found at {manifest_path}. "
            f"Run 'nhf-targets fetch daymet --project-dir {project.workdir}' first."
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"daymet: manifest.json at {manifest_path} is corrupt: {exc}"
        ) from exc
    sources = manifest.get("sources", {})
    daymet_entry = sources.get(_SOURCE_KEY)
    if daymet_entry is None:
        raise FileNotFoundError(
            f"daymet: no provenance entry in {manifest_path}. Run "
            f"'nhf-targets fetch daymet --project-dir {project.workdir}' first."
        )
    regions = daymet_entry.get("regions") or {}
    region_record = regions.get(region)
    if region_record is None or "path" not in region_record:
        known = sorted(regions.keys())
        raise FileNotFoundError(
            f"daymet: no manifest entry for region {region!r} "
            f"(known regions: {known}). Run "
            f"'nhf-targets fetch daymet --region {region}' first."
        )
    zarr_path = Path(region_record["path"])
    if not zarr_path.exists():
        raise FileNotFoundError(
            f"daymet: zarr at {zarr_path} (from manifest) does not exist. "
            f"Re-run 'nhf-targets fetch daymet --region {region}'."
        )
    return zarr_path


def _validate_period_in_zarr_range(
    ds: xr.Dataset, start_year: int, end_year: int
) -> None:
    """Reject ``--period`` ranges that extend past the zarr's time coord.

    Without this check, a user passing ``--period 1900/1979`` (before
    Daymet starts) would iterate years, slice empty time ranges, and
    surface an opaque gdptools error from the first batch. Doing the
    check here — with the already-open zarr's actual time coord
    (authoritative; the catalog's ``period`` may lag staging) —
    surfaces the misuse with a clear remediation.

    Hard-fails on any out-of-range year rather than silently clipping
    the requested window: silently skipping years would produce
    aggregator outputs that look complete but cover less than the
    operator asked for.
    """
    times = ds["time"].values
    if len(times) == 0:
        raise ValueError("daymet: zarr time coord is empty; cannot aggregate.")
    zarr_min_year = pd.Timestamp(times[0]).year
    zarr_max_year = pd.Timestamp(times[-1]).year
    if start_year < zarr_min_year or end_year > zarr_max_year:
        raise ValueError(
            f"daymet: --period {start_year}/{end_year} extends past the "
            f"zarr's time coverage ({zarr_min_year}/{zarr_max_year}). "
            f"Restrict --period to within the zarr's available years."
        )


def _crs_from_grid_mapping(ds: xr.Dataset) -> str:
    """Decode the ``lambert_conformal_conic`` grid mapping to a WKT CRS string.

    Uses ``pyproj.CRS.from_cf`` so the projection parameters travel
    with the zarr rather than being hardcoded in Python. Falls back
    with a precise error if the grid mapping variable is missing.
    """
    if _GRID_MAPPING_VAR not in ds.variables:
        raise ValueError(
            f"daymet: zarr is missing the {_GRID_MAPPING_VAR!r} grid "
            f"mapping variable required to derive the source CRS. "
            f"Available variables: {sorted(ds.variables)}."
        )
    attrs = dict(ds[_GRID_MAPPING_VAR].attrs)
    try:
        crs = CRS.from_cf(attrs)
    except Exception as exc:
        raise ValueError(
            f"daymet: could not derive a pyproj CRS from {_GRID_MAPPING_VAR} "
            f"attrs {attrs!r}: {exc}"
        ) from exc
    return crs.to_wkt()


def aggregate_daymet(
    fabric_path: str | Path,
    id_col: str,
    period: str,
    workdir: str | Path,
    batch_size: int = 500,
    region: str = "na",
    *,
    worker_index: int = 0,
    n_workers: int = 1,
) -> None:
    """Aggregate Daymet daily SWE (one region) to fabric HRU polygons.

    Writes one NC per year at
    ``data/aggregated/daymet/daymet_<region>_<year>_agg.nc``.
    Idempotent: existing per-year NCs are preserved on re-run; weight
    caches are reused across years (Daymet's per-region grid is
    invariant across the archive) and segregated per region.

    SLURM-array support (issue #156): ``worker_index`` / ``n_workers``
    select a round-robin slice of years for this worker. Default
    ``(0, 1)`` is serial; with ``n_workers > 1`` the contiguous-year
    check is skipped (other workers are still producing their slice)
    and the manifest update merges file lists across workers.

    Parameters
    ----------
    fabric_path : str | Path
        HRU fabric GeoPackage or GeoParquet path.
    id_col : str
        Fabric column carrying the HRU identifier.
    period : str
        Temporal range as ``'YYYY/YYYY'`` (inclusive). Daymet covers
        1980/2024; periods outside that range are clamped by the
        in-zarr time-coord bounds and may yield zero-timestep slices
        for unsupported years.
    workdir : str | Path
        Project directory created by ``nhf-targets init``.
    batch_size : int
        Target HRUs per spatial batch.
    region : str
        Daymet region; currently only ``"na"`` is supported.
    worker_index, n_workers : int
        SLURM-array round-robin sharding (per :func:`_assign_worker_years`
        in ``_driver.py``).
    """
    from nhf_spatial_targets.aggregate._driver import _assign_worker_years

    if region not in _SUPPORTED_REGIONS:
        raise NotImplementedError(
            f"daymet: region {region!r} is not yet supported. Only "
            f"{sorted(_SUPPORTED_REGIONS)} are wired up in this build; "
            f"HI/PR are tracked under issue #101. Drop the --region "
            f"flag or pass --region na."
        )

    parse_period(period)
    start_year, end_year = (int(p) for p in period.split("/"))

    workdir = Path(workdir)
    fabric_path = Path(fabric_path)
    project = _load_project(workdir)
    meta = catalog.source(_SOURCE_KEY)
    source_key_region = _region_source_key(region)

    # Legacy migration is keyed on the catalog source_key; safe no-op
    # on first run.
    _migrate_legacy_layout(project, _SOURCE_KEY)

    zarr_path = _resolve_zarr_path(project, region)
    logger.info("daymet: opening %s", zarr_path)
    # consolidated=False matches the fetch-side open in
    # fetch/daymet.py:175. Real Daymet zarrs are v2 without
    # consolidated metadata.
    src_ds = xr.open_zarr(zarr_path, consolidated=False)
    try:
        if _SOURCE_VAR not in src_ds.data_vars:
            raise ValueError(
                f"daymet: zarr at {zarr_path} is missing variable "
                f"{_SOURCE_VAR!r} (have {list(src_ds.data_vars)})."
            )
        source_crs = _crs_from_grid_mapping(src_ds)
        _validate_period_in_zarr_range(src_ds, start_year, end_year)
        # Keep only the swe variable (plus its coords and the grid
        # mapping reference) so gdptools doesn't chunk-walk prcp/
        # tmax/tmin/srad/vp.
        keep_vars = {_SOURCE_VAR}
        if _GRID_MAPPING_VAR in src_ds.variables:
            keep_vars.add(_GRID_MAPPING_VAR)
        drop = [v for v in src_ds.data_vars if v not in keep_vars]
        ds = src_ds.drop_vars(drop)

        logger.info("daymet: loading fabric %s", fabric_path)
        fabric_batched = load_and_batch_fabric(fabric_path, batch_size=batch_size)
        n_batches = int(fabric_batched["batch_id"].nunique())
        # SLURM-array year slice. The full year range comes from --period;
        # each worker gets a round-robin subset (default n_workers=1 -> all).
        all_years = list(range(start_year, end_year + 1))
        all_year_files = [(y, zarr_path) for y in all_years]
        assigned_year_files = _assign_worker_years(
            all_year_files, worker_index, n_workers
        )
        if not assigned_year_files:
            logger.info(
                "daymet: worker %d/%d has no years to process",
                worker_index,
                n_workers,
            )
            return
        assigned_years = [y for y, _ in assigned_year_files]
        if n_workers == 1:
            logger.info(
                "daymet: region=%s years=%d-%d batches=%d",
                region,
                start_year,
                end_year,
                n_batches,
            )
        else:
            logger.info(
                "daymet: worker %d/%d region=%s aggregating %d of %d years "
                "across %d batches",
                worker_index,
                n_workers,
                region,
                len(assigned_years),
                len(all_years),
                n_batches,
            )

        per_year_paths: list[Path] = []
        for year in assigned_years:
            out_path = _region_year_path(project, region, year)
            if out_path.exists():
                if out_path.stat().st_size == 0:
                    logger.warning(
                        "daymet: year %d: removing zero-byte stub at %s "
                        "and re-aggregating",
                        year,
                        out_path,
                    )
                    out_path.unlink()
                else:
                    logger.info(
                        "daymet: year %d: per-year NC exists, skipping (%s)",
                        year,
                        out_path,
                    )
                    per_year_paths.append(out_path)
                    continue

            time_period = (f"{year}-01-01", f"{year}-12-31")
            batch_datasets: list[xr.Dataset] = []
            for bid in sorted(fabric_batched["batch_id"].unique()):
                batch_gdf = fabric_batched[fabric_batched["batch_id"] == bid].drop(
                    columns=["batch_id"]
                )
                try:
                    weights = compute_or_load_weights(
                        batch_gdf=batch_gdf,
                        source_ds=ds,
                        source_var=_SOURCE_VAR,
                        source_crs=source_crs,
                        x_coord="x",
                        y_coord="y",
                        time_coord="time",
                        id_col=id_col,
                        source_key=source_key_region,
                        batch_id=int(bid),
                        workdir=project.workdir,
                        period=time_period,
                    )
                    batch_ds = aggregate_variables_for_batch(
                        batch_gdf=batch_gdf,
                        source_ds=ds,
                        variables=[_SOURCE_VAR],
                        source_crs=source_crs,
                        x_coord="x",
                        y_coord="y",
                        time_coord="time",
                        id_col=id_col,
                        weights=weights,
                        period=time_period,
                        stat_method="mean",
                    )
                except Exception as exc:
                    exc.add_note(
                        f"daymet: aggregation failed for region={region} "
                        f"year={year} batch={int(bid)}"
                    )
                    raise
                batch_datasets.append(batch_ds)

            year_ds = xr.concat(batch_datasets, dim=id_col)
            _attach_cf_global_attrs(year_ds, _SOURCE_KEY, meta)
            year_ds.attrs["daymet_region"] = region
            # Canonical id_col-ascending row order on emission (issue #93).
            year_ds = year_ds.sortby(id_col)
            _atomic_write_netcdf(year_ds, out_path)
            logger.info("daymet: year %d: wrote %s", year, out_path)
            per_year_paths.append(out_path)
    finally:
        src_ds.close()

    per_source_dir = project.aggregated_dir() / _SOURCE_KEY
    # Scope the contiguity check to the region's filename pattern so
    # parallel runs of other regions don't trip the check. Skip under
    # SLURM-array: other workers may still be writing their year slice.
    if n_workers == 1:
        _verify_year_coverage(
            per_source_dir, source_key_region, period=f"{start_year}/{end_year}"
        )

    rel_outputs = [str(p.relative_to(project.workdir)) for p in per_year_paths]
    weight_files = [
        str(Path("weights") / f"{source_key_region}_batch{i}.csv")
        for i in range(n_batches)
    ]
    access = meta["access"]
    access_with_doi = {**access}
    if meta.get("doi"):
        access_with_doi["doi"] = meta["doi"]
    update_manifest(
        project=project,
        source_key=_SOURCE_KEY,
        access=access_with_doi,
        period=f"{start_year}-01-01/{end_year}-12-31",
        output_files=rel_outputs,
        weight_files=weight_files,
        batch_size=batch_size,
        n_workers=n_workers,
    )
    if n_workers == 1:
        logger.info(
            "daymet: region=%s wrote %d per-year NCs to %s",
            region,
            len(per_year_paths),
            per_source_dir,
        )
    else:
        logger.info(
            "daymet: worker %d/%d region=%s wrote %d per-year NCs to %s",
            worker_index,
            n_workers,
            region,
            len(per_year_paths),
            per_source_dir,
        )
