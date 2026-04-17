"""Shared aggregation driver: manifest helper, weight cache, and tier-1 engine."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

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
    output_files: list[str],
    weight_files: list[str],
) -> None:
    """Merge an aggregation provenance entry into ``manifest.json`` atomically.

    The manifest is keyed as ``sources[source_key]``; existing entries for
    other sources are preserved. ``period`` is stored as-is for provenance;
    ``fabric_sha256`` is read from ``fabric.json``. ``output_files`` lists
    each per-year NC relative to ``project.workdir``.
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
        "output_files": list(output_files),
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
    except Exception:
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
    """Return the name of the time coord via CF ``axis=T`` / ``standard_name=time``.

    Returns ``None`` only if no dim carries either CF attribute. Callers
    should raise with context rather than guessing a literal name — a coord
    literally named ``"time"`` without CF attrs could be a non-time axis.
    """
    for name in ds.dims:
        if name not in ds.coords:
            continue
        attrs = ds.coords[name].attrs
        if attrs.get("axis") == "T" or attrs.get("standard_name") == "time":
            return name
    return None


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
    """Return the per-year aggregated NC path (canonical output)."""
    return (
        project.workdir
        / "data"
        / "aggregated"
        / source_key
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
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _migrate_legacy_layout(project: Project, source_key: str) -> None:
    """Migrate legacy aggregated layout into per-source subdirs.

    Idempotent. Moves ``data/aggregated/_by_year/<source_key>_*.nc`` into
    ``data/aggregated/<source_key>/`` and unlinks any stale
    ``data/aggregated/<source_key>_agg.nc``. If a target path already
    exists for a given year (collision), the legacy file is left in
    place — the new-path file is canonical.
    """
    agg_dir = project.aggregated_dir()
    legacy_dir = agg_dir / "_by_year"
    new_dir = agg_dir / source_key

    if legacy_dir.is_dir():
        new_dir.mkdir(parents=True, exist_ok=True)
        for legacy_file in sorted(legacy_dir.glob(f"{source_key}_*_agg.nc")):
            target = new_dir / legacy_file.name
            if target.exists():
                logger.info(
                    "%s: legacy %s collides with existing %s; "
                    "leaving both in place (new path is canonical)",
                    source_key,
                    legacy_file,
                    target,
                )
                continue
            legacy_file.rename(target)
            logger.info("%s: migrated %s -> %s", source_key, legacy_file, target)

    stale_consolidated = agg_dir / f"{source_key}_agg.nc"
    if stale_consolidated.is_file():
        stale_consolidated.unlink()
        logger.info(
            "%s: removed stale consolidated file %s",
            source_key,
            stale_consolidated,
        )


_YEAR_FNAME_RE = re.compile(r"^(?P<key>.+)_(?P<year>\d{4})_agg\.nc$")


def _parse_year_from_filename(path: Path, source_key: str) -> int | None:
    """Return the year parsed from ``<source_key>_<YYYY>_agg.nc``, else None."""
    m = _YEAR_FNAME_RE.match(path.name)
    if m is None or m.group("key") != source_key:
        return None
    return int(m.group("year"))


def _verify_year_coverage(per_source_dir: Path, source_key: str) -> None:
    """Scan the per-source dir and verify contiguous year coverage.

    Filename-level check: parses ``<source_key>_<YYYY>_agg.nc`` matches.
    Raises ``ValueError`` if no matching files exist or if there is an
    interior gap between ``min_year`` and ``max_year``. Filesystem
    uniqueness prevents two files with the same name in one directory,
    so the duplicate-year case that the old ``concat_years`` caught via
    time-coord inspection is unreachable here.
    """
    years: list[int] = []
    for p in sorted(per_source_dir.glob(f"{source_key}_*_agg.nc")):
        y = _parse_year_from_filename(p, source_key)
        if y is not None:
            years.append(y)
    if not years:
        raise ValueError(
            f"{source_key}: no per-year aggregated files found in {per_source_dir}"
        )
    expected = set(range(min(years), max(years) + 1))
    missing = sorted(expected - set(years))
    if missing:
        raise ValueError(
            f"{source_key}: year gap(s) in per-year aggregated files: "
            f"missing={missing}, covered={sorted(set(years))}"
        )


def aggregate_year(
    adapter: SourceAdapter,
    project: Project,
    year: int,
    source_file: Path,
    fabric_batched: gpd.GeoDataFrame,
    id_col: str,
    *,
    catalog_meta: dict | None = None,
) -> Path:
    """Aggregate one year to HRU polygons; idempotent on the per-year NC.

    Returns the path of the per-year aggregated NC. If that path already
    exists, returns immediately without opening the source file. Otherwise
    opens the source file lazily, applies ``adapter.pre_aggregate_hook`` if
    set, detects coords via CF attrs (respecting adapter overrides), runs
    the batch loop with ``period=(YYYY-01-01, YYYY-12-31)``, concatenates
    batches on ``id_col``, applies ``adapter.post_aggregate_hook`` if set,
    attaches CF-1.6 global attrs (using ``catalog_meta`` when provided;
    otherwise looked up from the catalog), and writes the per-year NC
    atomically.
    """
    out_path = per_year_output_path(project, adapter.source_key, year)
    if out_path.exists():
        logger.info(
            "%s: year %d: per-year NC exists, skipping (%s)",
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
            try:
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
            except Exception as exc:
                # Preserve the original exception type (FileNotFoundError,
                # PermissionError, gdptools errors, etc.) so callers can still
                # except on concrete types; attach provenance via a note.
                exc.add_note(
                    f"{adapter.source_key}: aggregation failed for "
                    f"year={year} batch={int(bid)} "
                    f"source_file={source_file.name}"
                )
                raise
            datasets.append(batch_ds)

        year_ds = xr.concat(datasets, dim=id_col)

    if adapter.post_aggregate_hook is not None:
        year_ds = adapter.post_aggregate_hook(year_ds)

    meta = (
        catalog_meta if catalog_meta is not None else catalog_source(adapter.source_key)
    )
    _attach_cf_global_attrs(year_ds, adapter.source_key, meta)

    _atomic_write_netcdf(year_ds, out_path)
    logger.info("%s: year %d: wrote %s", adapter.source_key, year, out_path)
    return out_path


def _derive_period(per_year_paths: list[Path], time_coord: str) -> str:
    """Return ``'YYYY-MM-DD/YYYY-MM-DD'`` from the first/last per-year files.

    Opens only the first and last files (lazy, closed immediately) and reads
    the first/last ``time_coord`` values. Avoids opening intermediate files.
    """
    first, last = per_year_paths[0], per_year_paths[-1]
    with xr.open_dataset(first) as ds_first:
        t0 = str(ds_first[time_coord].values[0])[:10]
    with xr.open_dataset(last) as ds_last:
        t1 = str(ds_last[time_coord].values[-1])[:10]
    return f"{t0}/{t1}"


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
        proj_feature=WEIGHT_GEN_CRS,
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
    except Exception:
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
            proj_feature=WEIGHT_GEN_CRS,
            id_feature=id_col,
            period=[period[0], period[1]],
        )
        agg = AggGen(
            user_data=user_data,
            stat_method="mean",
            agg_engine="serial",
            agg_writer="none",
            weights=weights,
        )
        _gdf, ds = agg.calculate_agg()
        per_var.append(ds)
    return xr.merge(per_var)


def _attach_cf_global_attrs(ds: xr.Dataset, source_key: str, meta: dict) -> None:
    """Attach CF-1.6 global attrs in place (non-destructive for var attrs).

    Appends to any existing ``history`` rather than overwriting, preserving
    provenance carried on the consolidated source NC.
    """
    access = meta.get("access", {})
    entry = (
        f"{datetime.now(timezone.utc).isoformat()}: aggregated to HRU fabric "
        f"by nhf_spatial_targets.aggregate._driver"
    )
    ds.attrs.setdefault("Conventions", "CF-1.6")
    existing = ds.attrs.get("history", "")
    ds.attrs["history"] = f"{existing}\n{entry}" if existing else entry
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
) -> None:
    """Aggregate a source to fabric HRU polygons; emit per-year NCs.

    Writes one NC per year to
    ``data/aggregated/<source_key>/<source_key>_<year>_agg.nc``. No
    consolidated single-file output is produced; per-year files are the
    canonical aggregated output. Idempotent: existing per-year files are
    preserved on restart. Legacy ``_by_year/`` files and stale
    ``<source_key>_agg.nc`` consolidated files are migrated via
    ``_migrate_legacy_layout`` at the top of the function.

    Variables declared by ``adapter.variables`` that are missing from the
    source NC cause ValueError before any year is aggregated — unless the
    adapter defines a ``pre_aggregate_hook`` (in which case the declared
    variables are constructed by the hook, not read from the raw NC).
    """
    workdir = Path(workdir)
    project = load_project(workdir)
    meta = catalog_source(adapter.source_key)

    _migrate_legacy_layout(project, adapter.source_key)

    raw_dir = project.raw_dir(adapter.source_key)
    files = sorted(raw_dir.glob(adapter.files_glob))
    if not files:
        raise FileNotFoundError(
            f"No NC matching '{adapter.files_glob}' found in {raw_dir}. "
            f"Run 'nhf-targets fetch {adapter.source_key}' first."
        )

    # Fail fast on missing declared variables and on source-grid drift across
    # files. Hooked adapters derive their declared variables, so the missing-var
    # check is skipped in that case. The grid-shape check always runs: weight
    # caches are keyed per (source_key, batch_id) — not per year — and reused
    # across years, so every file must share the same source grid.
    raw_grid_var = adapter.raw_grid_variable
    reference_grid: tuple | None = None
    for f in files:
        with xr.open_dataset(f) as peek:
            if adapter.pre_aggregate_hook is None:
                missing = [v for v in adapter.variables if v not in peek.data_vars]
                if missing:
                    raise ValueError(
                        f"{adapter.source_key}: variables {missing} missing from "
                        f"{f.name} (have {list(peek.data_vars)})"
                    )
            if raw_grid_var not in peek.data_vars:
                raise ValueError(
                    f"{adapter.source_key}: raw_grid_variable "
                    f"{raw_grid_var!r} missing from {f.name} "
                    f"(have {list(peek.data_vars)}). For adapters whose "
                    f"pre_aggregate_hook synthesizes declared variables, set "
                    f"SourceAdapter.raw_grid_variable to a variable that exists "
                    f"in the raw NC so the cross-year grid invariant can be "
                    f"enforced."
                )
            # Exclude the CF-detected time dim from the grid shape so that
            # per-file timestep differences (leap years, partial-year
            # fetches) don't masquerade as grid drift. Name-based filtering
            # would false-positive on sources whose time dim is not named
            # literally "time" (e.g. ERA5-Land "valid_time").
            time_name = _find_time_coord_name(peek)
            shape = tuple(
                peek[raw_grid_var].sizes[d]
                for d in peek[raw_grid_var].dims
                if d != time_name
            )
            if reference_grid is None:
                reference_grid = shape
            elif shape != reference_grid:
                raise ValueError(
                    f"{adapter.source_key}: grid shape drift across source "
                    f"files — {f.name} has {shape}, expected {reference_grid}. "
                    f"Weight caches are reused across years and require a "
                    f"stable source grid."
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
        aggregate_year(
            adapter,
            project,
            year,
            path,
            fabric_batched,
            id_col,
            catalog_meta=meta,
        )
        for year, path in year_files
    ]

    per_source_dir = project.aggregated_dir() / adapter.source_key
    _verify_year_coverage(per_source_dir, adapter.source_key)

    with xr.open_dataset(per_year_paths[0]) as probe:
        time_coord = _find_time_coord_name(probe)
    if time_coord is None:
        raise ValueError(
            f"{adapter.source_key}: could not detect CF time coord on "
            f"per-year NC {per_year_paths[0].name}. Expected a "
            f"coord with axis='T' or standard_name='time'."
        )
    period = _derive_period(per_year_paths, time_coord)

    rel_output_files = [str(p.relative_to(project.workdir)) for p in per_year_paths]
    update_manifest(
        project=project,
        source_key=adapter.source_key,
        access=meta.get("access", {}),
        period=period,
        output_files=rel_output_files,
        weight_files=[
            str(Path("weights") / f"{adapter.source_key}_batch{i}.csv")
            for i in range(n_batches)
        ],
    )
    logger.info(
        "%s: %d per-year NCs written to %s",
        adapter.source_key,
        len(per_year_paths),
        per_source_dir,
    )
