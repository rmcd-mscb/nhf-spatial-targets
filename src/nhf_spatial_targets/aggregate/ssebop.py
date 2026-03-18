"""Aggregate SSEBop AET from NHGF STAC catalog to HRU fabric polygons."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
from gdptools import AggGen, NHGFStacZarrData, WeightGen
from gdptools.helpers import get_stac_collection

from nhf_spatial_targets import catalog
from nhf_spatial_targets.aggregate.batching import spatial_batch
from nhf_spatial_targets.workspace import load as _load_workspace

logger = logging.getLogger(__name__)

_SOURCE_KEY = "ssebop"
_SOURCE_VAR = "et"
_WEIGHT_GEN_CRS = 5070  # NAD83 / CONUS Albers


def _parse_period(period: str) -> list[str]:
    """Convert 'YYYY/YYYY' to ['YYYY-01-01', 'YYYY-12-31']."""
    parts = period.split("/")
    return [f"{parts[0]}-01-01", f"{parts[-1]}-12-31"]


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
    """Process a single spatial batch: compute/load weights, aggregate."""
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
            weight_gen_crs=_WEIGHT_GEN_CRS,
        )
        weights = wg.calculate_weights()
        wp.parent.mkdir(parents=True, exist_ok=True)
        weights.to_csv(wp, index=False)
        logger.info("Batch %d: weights saved to %s", batch_id, wp)

    # Fresh NHGFStacZarrData for aggregation
    stac_data = NHGFStacZarrData(
        source_collection=collection,
        source_var=_SOURCE_VAR,
        target_gdf=batch_gdf,
        target_id=id_col,
        source_time_period=time_period,
    )
    agg = AggGen(
        user_data=stac_data,
        stat_method="masked_mean",
        agg_engine="serial",
        agg_writer="none",
        weights=weights,
    )
    _gdf, ds = agg.calculate_agg()
    logger.info("Batch %d: aggregation complete", batch_id)
    return ds


def aggregate_ssebop(
    fabric_path: str | Path,
    id_col: str,
    period: str,
    workdir: str | Path,
    batch_size: int = 500,
) -> xr.Dataset:
    """Aggregate SSEBop monthly AET to fabric HRU polygons.

    Parameters
    ----------
    fabric_path : str | Path
        Path to the HRU fabric GeoPackage.
    id_col : str
        Column name for HRU identifiers in the fabric.
    period : str
        Temporal range as 'YYYY/YYYY'.
    workdir : str | Path
        Workspace directory.
    batch_size : int
        Target number of HRUs per spatial batch.

    Returns
    -------
    xr.Dataset
        Aggregated SSEBop AET with dimensions (time, <id_col>).
    """
    workdir = Path(workdir)
    fabric_path = Path(fabric_path)
    ws = _load_workspace(workdir)

    # 1. Load source metadata and STAC collection
    meta = catalog.source(_SOURCE_KEY)
    collection_id = meta["access"]["collection_id"]
    logger.info("Resolving STAC collection: %s", collection_id)
    collection = get_stac_collection(collection_id)

    # 2. Load and batch the fabric
    logger.info("Loading fabric: %s", fabric_path)
    gdf = gpd.read_file(fabric_path)
    batched = spatial_batch(gdf, batch_size=batch_size)
    n_batches = batched["batch_id"].nunique()
    logger.info("Fabric split into %d spatial batches", n_batches)

    # 3. Process each batch
    time_period = _parse_period(period)
    datasets = []
    for bid in sorted(batched["batch_id"].unique()):
        batch_gdf = batched[batched["batch_id"] == bid].copy()
        batch_gdf = batch_gdf.drop(columns=["batch_id"])
        ds = _process_batch(batch_gdf, bid, collection, id_col, time_period, workdir)
        datasets.append(ds)

    # 4. Concatenate batches
    combined = xr.concat(datasets, dim=id_col)
    logger.info(
        "Combined dataset: %s time steps x %s HRUs",
        combined.sizes.get("time", "?"),
        combined.sizes.get(id_col, "?"),
    )

    # 5. Write output
    output_dir = ws.aggregated_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ssebop_aet.nc"
    combined.to_netcdf(output_path)
    logger.info("Output written to %s", output_path)

    # 6. Update manifest
    _update_manifest(ws, period, meta, n_batches)

    return combined


def _update_manifest(
    ws,
    period: str,
    meta: dict,
    n_batches: int,
) -> None:
    """Merge SSEBop aggregation provenance into manifest.json."""
    manifest_path = ws.manifest_path
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {ws.workdir} is corrupt: {exc}"
            ) from exc
    else:
        manifest = {"sources": {}, "steps": []}

    if "sources" not in manifest:
        manifest["sources"] = {}

    fabric_json = ws.workdir / "fabric.json"
    fabric_sha = ""
    if fabric_json.exists():
        fabric_meta = json.loads(fabric_json.read_text())
        fabric_sha = fabric_meta.get("sha256", "")

    access = meta["access"]
    time_period = _parse_period(period)
    weight_files = [
        str(Path("weights") / f"ssebop_batch{i}.csv") for i in range(n_batches)
    ]

    manifest["sources"][_SOURCE_KEY] = {
        "source_key": _SOURCE_KEY,
        "access_type": access["type"],
        "collection_id": access["collection_id"],
        "doi": meta.get("doi", ""),
        "period": f"{time_period[0]}/{time_period[1]}",
        "fabric_sha256": fabric_sha,
        "output_file": str(Path("data") / "aggregated" / "ssebop_aet.nc"),
        "weight_files": weight_files,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    tmp_fd, tmp_path = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp_path).replace(manifest_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    logger.info("Updated manifest.json with SSEBop aggregation provenance")
