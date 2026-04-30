"""Aggregate SSEBop AET from NHGF STAC catalog to HRU fabric polygons.

SSEBop is read on the fly from the USGS NHGF STAC Zarr store rather than
from a local consolidated NC, so it follows a different access path than
the file-based sources orchestrated by ``_driver.aggregate_source``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
from gdptools import AggGen, NHGFStacZarrData, WeightGen
from gdptools.helpers import get_stac_collection

from nhf_spatial_targets import catalog
from nhf_spatial_targets.aggregate._driver import (
    WEIGHT_GEN_CRS,
    _atomic_write_netcdf,
    _attach_cf_global_attrs,
    _migrate_legacy_layout,
    _verify_year_coverage,
    load_and_batch_fabric,
    per_year_output_path,
    update_manifest,
)
from nhf_spatial_targets.workspace import load as _load_project

logger = logging.getLogger(__name__)

_SOURCE_KEY = "ssebop"
_SOURCE_VAR = "et"


def _weight_path(workdir: Path, batch_id: int) -> Path:
    return workdir / "weights" / f"ssebop_batch{batch_id}.csv"


def _process_batch(
    batch_gdf: gpd.GeoDataFrame,
    batch_id: int,
    collection,
    id_col: str,
    time_period: list[str],
    workdir: Path,
) -> xr.Dataset:
    """Compute weights (or load cache) and aggregate one batch for one year.

    Weights are cached on disk per batch and reused across years because
    the SSEBop source grid is invariant across the archive.
    """
    wp = _weight_path(workdir, batch_id)

    if wp.exists():
        logger.info("Batch %d: loading cached weights from %s", batch_id, wp)
        weights = pd.read_csv(wp)
    else:
        logger.info("Batch %d: computing weights (%d HRUs)", batch_id, len(batch_gdf))
        stac_data = NHGFStacZarrData(
            source_collection=collection,
            source_var=_SOURCE_VAR,
            target_gdf=batch_gdf,
            target_id=id_col,
            source_time_period=time_period,
        )
        wg = WeightGen(
            user_data=stac_data,
            method="serial",
            weight_gen_crs=WEIGHT_GEN_CRS,
        )
        weights = wg.calculate_weights()
        wp.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tempfile in same dir + replace, so a SIGKILL between
        # mkstemp and replace can't leave a truncated CSV that pd.read_csv
        # would silently load on the next run.
        tmp_fd, tmp_path = tempfile.mkstemp(dir=wp.parent, suffix=".csv.tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                weights.to_csv(f, index=False)
            Path(tmp_path).replace(wp)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        logger.info("Batch %d: weights saved to %s", batch_id, wp)

    stac_data = NHGFStacZarrData(
        source_collection=collection,
        source_var=_SOURCE_VAR,
        target_gdf=batch_gdf,
        target_id=id_col,
        source_time_period=time_period,
    )
    agg = AggGen(
        user_data=stac_data,
        stat_method="mean",
        agg_engine="serial",
        agg_writer="none",
        weights=weights,
    )
    _gdf, ds = agg.calculate_agg()
    return ds


def aggregate_ssebop(
    fabric_path: str | Path,
    id_col: str,
    period: str,
    workdir: str | Path,
    batch_size: int = 500,
) -> None:
    """Aggregate SSEBop monthly AET to fabric HRU polygons, one NC per year.

    Writes ``data/aggregated/ssebop/ssebop_<year>_agg.nc`` for each year
    in ``period``. Idempotent: existing per-year NCs are left in place
    on re-run (zero-byte stubs from a prior crash are unlinked and
    re-written); weight CSVs are cached per batch and reused.

    On a year-N failure, years <N already written are preserved on
    disk; re-running resumes at year N. If every year writes
    successfully but the manifest update then fails, the next run's
    idempotency reuses the on-disk NCs and re-attempts the manifest
    update, healing the divergence.

    Parameters
    ----------
    fabric_path : str | Path
        Path to the HRU fabric GeoPackage or GeoParquet.
    id_col : str
        Column name for HRU identifiers in the fabric.
    period : str
        Temporal range as 'YYYY/YYYY' (inclusive on both ends).
    workdir : str | Path
        Project directory.
    batch_size : int
        Target number of HRUs per spatial batch.
    """
    # Validate period before any network or fabric work — a malformed
    # `--period` should fail in milliseconds, not after a STAC round-trip
    # and a fabric load.
    parts = period.split("/")
    if len(parts) != 2 or not all(p.isdigit() and len(p) == 4 for p in parts):
        raise ValueError(f"ssebop: --period must be 'YYYY/YYYY', got {period!r}")
    start_year, end_year = int(parts[0]), int(parts[1])
    if end_year < start_year:
        raise ValueError(
            f"ssebop: --period end year ({end_year}) precedes start ({start_year})"
        )

    workdir = Path(workdir)
    fabric_path = Path(fabric_path)
    project = _load_project(workdir)
    meta = catalog.source(_SOURCE_KEY)

    # Legacy migration: prior pipeline emitted one consolidated NC at
    # data/aggregated/ssebop_agg_aet.nc. _migrate_legacy_layout only
    # handles the <source_key>_agg.nc convention, so we have to unlink
    # this bespoke filename ourselves. missing_ok defends against a
    # parallel agg_all.slurm task already having removed it.
    legacy_consolidated = project.aggregated_dir() / "ssebop_agg_aet.nc"
    legacy_consolidated.unlink(missing_ok=True)
    _migrate_legacy_layout(project, _SOURCE_KEY)

    collection_id = meta["access"]["collection_id"]
    logger.info("Resolving STAC collection: %s", collection_id)
    try:
        collection = get_stac_collection(collection_id)
    except Exception as exc:
        exc.add_note(
            f"ssebop: failed to resolve STAC collection {collection_id!r} "
            f"(network or NHGF STAC endpoint may be unavailable)"
        )
        raise

    logger.info("Loading fabric: %s", fabric_path)
    fabric_batched = load_and_batch_fabric(fabric_path, batch_size=batch_size)
    n_batches = int(fabric_batched["batch_id"].nunique())
    logger.info("Fabric split into %d spatial batches", n_batches)
    years = list(range(start_year, end_year + 1))
    logger.info(
        "ssebop: %d year(s) to aggregate (%d-%d) across %d batches",
        len(years),
        start_year,
        end_year,
        n_batches,
    )

    per_year_paths: list[Path] = []
    for year in years:
        out_path = per_year_output_path(project, _SOURCE_KEY, year)
        if out_path.exists():
            # A zero-byte stub is left if anything truncated the file
            # outside the atomic-write path (e.g. a manual touch, a
            # filesystem hiccup, or a process killed during a non-atomic
            # write before _atomic_write_netcdf was introduced). Treat
            # it as missing instead of skipping forever and failing
            # later in _verify_year_coverage with no remediation.
            if out_path.stat().st_size == 0:
                logger.warning(
                    "ssebop: year %d: removing zero-byte stub at %s and re-aggregating",
                    year,
                    out_path,
                )
                out_path.unlink()
            else:
                logger.info(
                    "ssebop: year %d: per-year NC exists, skipping (%s)",
                    year,
                    out_path,
                )
                per_year_paths.append(out_path)
                continue

        time_period = [f"{year}-01-01", f"{year}-12-31"]
        batch_datasets: list[xr.Dataset] = []
        for bid in sorted(fabric_batched["batch_id"].unique()):
            batch_gdf = fabric_batched[fabric_batched["batch_id"] == bid].drop(
                columns=["batch_id"]
            )
            try:
                ds = _process_batch(
                    batch_gdf,
                    int(bid),
                    collection,
                    id_col,
                    time_period,
                    project.workdir,
                )
            except Exception as exc:
                exc.add_note(
                    f"ssebop: aggregation failed for year={year} batch={int(bid)}"
                )
                raise
            batch_datasets.append(ds)

        year_ds = xr.concat(batch_datasets, dim=id_col)
        _attach_cf_global_attrs(year_ds, _SOURCE_KEY, meta)
        _atomic_write_netcdf(year_ds, out_path)
        logger.info("ssebop: year %d: wrote %s", year, out_path)
        per_year_paths.append(out_path)

    per_source_dir = project.aggregated_dir() / _SOURCE_KEY
    _verify_year_coverage(per_source_dir, _SOURCE_KEY)

    rel_outputs = [str(p.relative_to(project.workdir)) for p in per_year_paths]
    weight_files = [
        str(Path("weights") / f"ssebop_batch{i}.csv") for i in range(n_batches)
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
    )
    logger.info(
        "ssebop: %d per-year NCs written to %s",
        len(per_year_paths),
        per_source_dir,
    )
