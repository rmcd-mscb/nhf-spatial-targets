"""Shared aggregation driver: manifest helper, weight cache, and tier-1 engine."""

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

from nhf_spatial_targets.aggregate.batching import spatial_batch
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)

# Lazy imports for gdptools (exposed as module-level names for test patching)
AggGen = None
UserCatData = None
WeightGen = None


def update_manifest(
    project: Project,
    source_key: str,
    access: dict,
    period: str,
    output_file: str,
    weight_files: list[str],
) -> None:
    """Merge an aggregation provenance entry into ``manifest.json`` atomically.

    The manifest is keyed as ``sources[source_key]``; existing entries for
    other sources are preserved. ``period`` is stored as-is for provenance;
    ``fabric_sha256`` is read from ``fabric.json``.
    """
    manifest_path = project.manifest_path
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {project.workdir} is corrupt: {exc}"
            ) from exc
    else:
        manifest = {"sources": {}, "steps": []}

    manifest.setdefault("sources", {})

    fabric_json = project.workdir / "fabric.json"
    fabric_sha = ""
    if fabric_json.exists():
        fabric_meta = json.loads(fabric_json.read_text())
        fabric_sha = fabric_meta.get("sha256", "")

    entry: dict = {
        "source_key": source_key,
        "access_type": access.get("type", ""),
        "period": period,
        "fabric_sha256": fabric_sha,
        "output_file": output_file,
        "weight_files": list(weight_files),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Carry a few optional access identifiers through for provenance parity
    # with ssebop.py's existing behaviour.
    for extra_key in ("collection_id", "short_name", "version", "doi"):
        if extra_key in access:
            entry[extra_key] = access[extra_key]

    manifest["sources"][source_key] = entry

    tmp_fd, tmp_path = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp_path).replace(manifest_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    logger.info("Updated manifest.json with '%s' aggregation provenance", source_key)


_WEIGHT_GEN_CRS = 5070  # NAD83 / CONUS Albers (equal-area)


def load_and_batch_fabric(fabric_path: Path, batch_size: int = 500) -> gpd.GeoDataFrame:
    """Load the fabric GeoPackage (or GeoParquet) and attach ``batch_id``.

    Parameters
    ----------
    fabric_path : Path
        Path to the HRU fabric file (GeoPackage, GeoParquet, or other OGR-supported format).
    batch_size : int
        Target number of features per spatial batch.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of the fabric with an added ``batch_id`` column.
    """
    fabric_path = Path(fabric_path)
    if fabric_path.suffix.lower() in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(fabric_path)
    else:
        gdf = gpd.read_file(fabric_path)
    return spatial_batch(gdf, batch_size=batch_size)


def weight_cache_path(workdir: Path, source_key: str, batch_id: int) -> Path:
    """Return the per-batch weight CSV path.

    Parameters
    ----------
    workdir : Path
        Project working directory.
    source_key : str
        Data source identifier (e.g., 'ssebop', 'merra2').
    batch_id : int
        Batch number.

    Returns
    -------
    Path
        Path to the weight cache CSV for this batch.
    """
    return Path(workdir) / "weights" / f"{source_key}_batch{batch_id}.csv"


def compute_or_load_weights(
    batch_gdf: gpd.GeoDataFrame,
    source_ds: xr.Dataset,
    source_var: str,
    source_crs: str,
    x_coord: str,
    y_coord: str,
    time_coord: str,
    id_col: str,
    source_key: str,
    batch_id: int,
    workdir: Path,
) -> pd.DataFrame:
    """Compute (or load from cache) the per-batch weight table.

    Uses gdptools WeightGen to compute area-weighted aggregation weights
    from source grid to HRU fabric. Results are cached per batch to avoid
    recomputation.

    Parameters
    ----------
    batch_gdf : gpd.GeoDataFrame
        Subset of the fabric for this batch (one row per HRU).
    source_ds : xr.Dataset
        Source gridded dataset (must include x_coord, y_coord, time_coord).
    source_var : str
        Variable name in source_ds used to infer grid structure.
    source_crs : str
        CRS of the source grid (e.g., 'EPSG:4326').
    x_coord : str
        Name of the x-coordinate dimension in source_ds.
    y_coord : str
        Name of the y-coordinate dimension in source_ds.
    time_coord : str
        Name of the time-coordinate dimension in source_ds.
    id_col : str
        Name of the HRU identifier column in batch_gdf.
    source_key : str
        Data source identifier.
    batch_id : int
        Batch number for caching.
    workdir : Path
        Project working directory.

    Returns
    -------
    pd.DataFrame
        Weight table with columns (at minimum) for grid indices and HRU ID.
    """
    global UserCatData, WeightGen
    from gdptools import UserCatData as _UserCatData
    from gdptools import WeightGen as _WeightGen

    UserCatData = _UserCatData
    WeightGen = _WeightGen

    wp = weight_cache_path(workdir, source_key, batch_id)
    if wp.exists():
        logger.info("Batch %d: loading cached weights from %s", batch_id, wp)
        return pd.read_csv(wp)

    logger.info(
        "Batch %d: computing weights (%d HRUs, source_var=%s)",
        batch_id,
        len(batch_gdf),
        source_var,
    )
    user_data = UserCatData(
        ds=source_ds,
        proj_ds=source_crs,
        x_coord=x_coord,
        y_coord=y_coord,
        t_coord=time_coord,
        var=[source_var],
        f_feature=batch_gdf,
        proj_feature=batch_gdf.crs.to_string(),
        id_feature=id_col,
        period=[
            str(source_ds[time_coord].values[0]),
            str(source_ds[time_coord].values[-1]),
        ],
    )
    wg = WeightGen(
        user_data=user_data,
        method="serial",
        weight_gen_crs=_WEIGHT_GEN_CRS,
    )
    weights = wg.calculate_weights()
    wp.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(wp, index=False)
    logger.info("Batch %d: weights saved to %s", batch_id, wp)
    return weights


def aggregate_variables_for_batch(
    batch_gdf: gpd.GeoDataFrame,
    source_ds: xr.Dataset,
    variables: list[str],
    source_crs: str,
    x_coord: str,
    y_coord: str,
    time_coord: str,
    id_col: str,
    weights: pd.DataFrame,
) -> xr.Dataset:
    """Run gdptools AggGen once per variable, merge results on HRU ID.

    Parameters
    ----------
    batch_gdf : gpd.GeoDataFrame
        Subset of the fabric for this batch (one row per HRU).
    source_ds : xr.Dataset
        Source gridded dataset.
    variables : list[str]
        List of variable names to aggregate from source_ds.
    source_crs : str
        CRS of the source grid.
    x_coord : str
        Name of the x-coordinate dimension.
    y_coord : str
        Name of the y-coordinate dimension.
    time_coord : str
        Name of the time-coordinate dimension.
    id_col : str
        Name of the HRU identifier column in batch_gdf.
    weights : pd.DataFrame
        Pre-computed weight table from compute_or_load_weights.

    Returns
    -------
    xr.Dataset
        Merged Dataset with all ``variables`` aggregated to HRU dimensions.
    """
    global AggGen, UserCatData
    from gdptools import AggGen as _AggGen
    from gdptools import UserCatData as _UserCatData

    AggGen = _AggGen
    UserCatData = _UserCatData

    per_var: list[xr.Dataset] = []
    for var in variables:
        user_data = UserCatData(
            ds=source_ds,
            proj_ds=source_crs,
            x_coord=x_coord,
            y_coord=y_coord,
            t_coord=time_coord,
            var=[var],
            f_feature=batch_gdf,
            proj_feature=batch_gdf.crs.to_string(),
            id_feature=id_col,
            period=[
                str(source_ds[time_coord].values[0]),
                str(source_ds[time_coord].values[-1]),
            ],
        )
        agg = AggGen(
            user_data=user_data,
            stat_method="masked_mean",
            agg_engine="serial",
            agg_writer="none",
            weights=weights,
        )
        _gdf, ds = agg.calculate_agg()
        per_var.append(ds)
    return xr.merge(per_var)
