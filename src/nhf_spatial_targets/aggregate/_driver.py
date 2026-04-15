"""Shared aggregation driver: manifest helper, weight cache, and tier-1 engine."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
from gdptools import AggGen, UserCatData, WeightGen

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._coords import detect_coords
from nhf_spatial_targets.aggregate.batching import spatial_batch
from nhf_spatial_targets.catalog import source as catalog_source
from nhf_spatial_targets.workspace import Project, load as load_project

logger = logging.getLogger(__name__)


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
    # Forward optional catalog access keys so downstream provenance consumers
    # see DOI, collection_id, short_name, and version when catalogued.
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


WEIGHT_GEN_CRS = 5070  # NAD83 / CONUS Albers (equal-area)


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
    """Return the per-batch weight CSV path."""
    return Path(workdir) / "weights" / f"{source_key}_batch{batch_id}.csv"


def _find_time_coord_name(ds: xr.Dataset) -> str | None:
    """Return the name of the time coord via CF attrs, or None."""
    for name in ds.coords:
        attrs = ds.coords[name].attrs
        if attrs.get("axis") == "T" or attrs.get("standard_name") == "time":
            return name
    # Fall back: any coord literally named 'time' (non-CF legacy NCs).
    return "time" if "time" in ds.coords else None


def enumerate_years(files: list[Path]) -> list[tuple[int, Path]]:
    """Map each year covered by the files to its source file.

    Opens each NC lazily, reads its time coord, and expands to one
    ``(year, file)`` tuple per distinct year. Returns results sorted by year.

    Raises:
        ValueError: two files cover the same year (stale fetch), or a file has
            no resolvable time coord.
    """
    year_to_file: dict[int, Path] = {}
    for path in files:
        with xr.open_dataset(path) as ds:
            time_name = _find_time_coord_name(ds)
            if time_name is None:
                attrs_by_coord = {n: dict(ds.coords[n].attrs) for n in ds.coords}
                raise ValueError(
                    f"No time coord found in {path.name}. "
                    f"dims={list(ds.dims)}, coord attrs={attrs_by_coord}"
                )
            years = pd.DatetimeIndex(ds[time_name].values).year.unique().tolist()
        for y in years:
            if y in year_to_file:
                raise ValueError(
                    f"Year {y} overlaps between {year_to_file[y].name} "
                    f"and {path.name}. Check the datastore for a stale "
                    f"fetch artifact."
                )
            year_to_file[int(y)] = path
    return sorted(year_to_file.items())


def per_year_output_path(project: Project, source_key: str, year: int) -> Path:
    """Return the per-year intermediate NC path."""
    return (
        project.workdir
        / "data"
        / "aggregated"
        / "_by_year"
        / f"{source_key}_{year}_agg.nc"
    )


def _atomic_write_netcdf(ds: xr.Dataset, path: Path) -> None:
    """Atomically write a Dataset to disk via tempfile + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".nc.tmp")
    os.close(tmp_fd)
    try:
        ds.to_netcdf(tmp_path)
        Path(tmp_path).replace(path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def aggregate_year(
    adapter: SourceAdapter,
    project: Project,
    year: int,
    source_file: Path,
    fabric_batched: gpd.GeoDataFrame,
    id_col: str,
) -> Path:
    """Aggregate one year to HRU polygons; idempotent on the intermediate NC.

    Returns the path of the per-year intermediate. If that path already
    exists, returns immediately without opening the source file. Otherwise
    opens the source file lazily, applies ``adapter.pre_aggregate_hook`` if
    set, detects coords via CF attrs (respecting adapter overrides), runs
    the batch loop with ``period=(YYYY-01-01, YYYY-12-31)``, concatenates
    batches on ``id_col``, and writes the intermediate atomically.
    """
    out_path = per_year_output_path(project, adapter.source_key, year)
    if out_path.exists():
        logger.info(
            "%s: year %d: intermediate exists, skipping (%s)",
            adapter.source_key,
            year,
            out_path,
        )
        return out_path

    logger.info(
        "%s: year %d: aggregating from %s",
        adapter.source_key,
        year,
        source_file.name,
    )
    period = (f"{year}-01-01", f"{year}-12-31")

    with xr.open_dataset(source_file) as raw:
        ds = raw
        if adapter.pre_aggregate_hook is not None:
            ds = adapter.pre_aggregate_hook(ds)

        grid_var = adapter.grid_variable or adapter.variables[0]
        x_coord, y_coord, time_coord = detect_coords(
            ds,
            grid_var,
            x_override=adapter.x_coord,
            y_override=adapter.y_coord,
            time_override=adapter.time_coord,
        )

        datasets: list[xr.Dataset] = []
        for bid in sorted(fabric_batched["batch_id"].unique()):
            batch_gdf = fabric_batched[fabric_batched["batch_id"] == bid].drop(
                columns=["batch_id"]
            )
            weights = compute_or_load_weights(
                batch_gdf=batch_gdf,
                source_ds=ds,
                source_var=grid_var,
                source_crs=adapter.source_crs,
                x_coord=x_coord,
                y_coord=y_coord,
                time_coord=time_coord,
                id_col=id_col,
                source_key=adapter.source_key,
                batch_id=int(bid),
                workdir=project.workdir,
                period=period,
            )
            batch_ds = aggregate_variables_for_batch(
                batch_gdf=batch_gdf,
                source_ds=ds,
                variables=list(adapter.variables),
                source_crs=adapter.source_crs,
                x_coord=x_coord,
                y_coord=y_coord,
                time_coord=time_coord,
                id_col=id_col,
                weights=weights,
                period=period,
            )
            datasets.append(batch_ds)

        year_ds = xr.concat(datasets, dim=id_col)

    _atomic_write_netcdf(year_ds, out_path)
    logger.info("%s: year %d: wrote %s", adapter.source_key, year, out_path)
    return out_path


def concat_years(paths: list[Path], time_coord: str) -> xr.Dataset:
    """Open per-year intermediates, concat on time, validate monotonic + unique.

    Loads each intermediate into memory and closes the on-disk handle so the
    returned Dataset is detached from the filesystem.
    """
    if not paths:
        raise ValueError("concat_years called with empty paths list")
    loaded: list[xr.Dataset] = []
    for p in paths:
        with xr.open_dataset(p) as ds:
            loaded.append(ds.load())
    combined = xr.concat(loaded, dim=time_coord).sortby(time_coord)
    t = combined[time_coord].values
    if len(np.unique(t)) != len(t):
        raise ValueError(
            f"Duplicate time coords across per-year intermediates: "
            f"{[p.name for p in paths]}"
        )
    return combined


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
    *,
    period: tuple[str, str],
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
        period=[period[0], period[1]],
    )
    wg = WeightGen(
        user_data=user_data,
        method="serial",
        weight_gen_crs=WEIGHT_GEN_CRS,
    )
    weights = wg.calculate_weights()
    wp.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=wp.parent, suffix=".csv.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            weights.to_csv(f, index=False)
        Path(tmp_path).replace(wp)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
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
    *,
    period: tuple[str, str],
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
            period=[period[0], period[1]],
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


def _attach_cf_global_attrs(ds: xr.Dataset, source_key: str, meta: dict) -> None:
    """Attach CF-1.6 global attrs in place (non-destructive for var attrs)."""
    access = meta.get("access", {})
    history = (
        f"{datetime.now(timezone.utc).isoformat()}: aggregated to HRU fabric "
        f"by nhf_spatial_targets.aggregate._driver"
    )
    ds.attrs.setdefault("Conventions", "CF-1.6")
    ds.attrs["history"] = history
    ds.attrs["source"] = source_key
    if "doi" in access:
        ds.attrs["source_doi"] = access["doi"]
    ds.attrs.setdefault("institution", "USGS")


def aggregate_source(
    adapter: SourceAdapter,
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
) -> xr.Dataset:
    """Aggregate a source to fabric HRU polygons via the per-year pipeline.

    Enumerates years from ``*_consolidated.nc`` in the datastore, aggregates
    each year to ``data/aggregated/_by_year/<source_key>_<year>_agg.nc``
    (idempotent; existing intermediates are reused on restart), then concats
    on time to ``data/aggregated/<output_name>``. Per-year intermediates
    are preserved for audit/restart.

    Variables declared by ``adapter.variables`` that are missing from the
    source NC cause ValueError before any year is aggregated — unless the
    adapter defines a ``pre_aggregate_hook`` (in which case the declared
    variables are constructed by the hook, not read from the raw NC).
    """
    workdir = Path(workdir)
    project = load_project(workdir)
    meta = catalog_source(adapter.source_key)

    raw_dir = project.raw_dir(adapter.source_key)
    files = sorted(raw_dir.glob("*_consolidated.nc"))
    if not files:
        raise FileNotFoundError(
            f"No consolidated NC found in {raw_dir}. "
            f"Run 'nhf-targets fetch {adapter.source_key}' first."
        )

    # Fail fast on missing declared variables — but only if there is no
    # pre_aggregate_hook. Hooked adapters derive their declared variables.
    if adapter.pre_aggregate_hook is None:
        with xr.open_dataset(files[0]) as peek:
            missing = [v for v in adapter.variables if v not in peek.data_vars]
            if missing:
                raise ValueError(
                    f"{adapter.source_key}: variables {missing} missing from "
                    f"source dataset (have {list(peek.data_vars)})"
                )

    year_files = enumerate_years(files)
    fabric_batched = load_and_batch_fabric(fabric_path, batch_size=batch_size)
    n_batches = int(fabric_batched["batch_id"].nunique())
    logger.info(
        "%s: %d years to aggregate across %d spatial batches",
        adapter.source_key,
        len(year_files),
        n_batches,
    )

    per_year_paths = [
        aggregate_year(adapter, project, year, path, fabric_batched, id_col)
        for year, path in year_files
    ]

    with xr.open_dataset(per_year_paths[0]) as probe:
        time_coord = _find_time_coord_name(probe) or "time"
    combined = concat_years(per_year_paths, time_coord=time_coord)

    if adapter.post_aggregate_hook is not None:
        combined = adapter.post_aggregate_hook(combined)

    _attach_cf_global_attrs(combined, adapter.source_key, meta)

    output_path = project.aggregated_dir() / adapter.output_name
    _atomic_write_netcdf(combined, output_path)
    logger.info("%s: output written to %s", adapter.source_key, output_path)

    t0 = str(combined[time_coord].values[0])[:10]
    t1 = str(combined[time_coord].values[-1])[:10]
    update_manifest(
        project=project,
        source_key=adapter.source_key,
        access=meta.get("access", {}),
        period=f"{t0}/{t1}",
        output_file=str(Path("data") / "aggregated" / adapter.output_name),
        weight_files=[
            str(Path("weights") / f"{adapter.source_key}_batch{i}.csv")
            for i in range(n_batches)
        ],
    )
    return combined
