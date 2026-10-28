"""Shared helpers for the Margulis perennial-snow mask checks (#363).

Used by ``inspect_aggregated_swe.ipynb`` (checks 1-5) and
``inspect_target_swe.ipynb`` (check 6). Kept out of ``_helpers.py`` because
both notebook directories have their own ``_helpers`` module.

The one thing to get right here is the weight-cache index convention. The
rows of ``<project>/weights/margulis_wus_sr_batch<N>.csv`` hold
``(hru_id, i, j, wght)`` where ``i`` / ``j`` index the per-batch gdptools
**subset** of the source grid (cropped to the batch bbox), not the full
Margulis grid. :func:`weight_cells` resolves them to ``(lat, lon)`` with the
driver's own ``_weight_cell_coords`` on the same ``UserCatData`` the
aggregation driver builds, so the result is exact for any batch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from gdptools import UserCatData

from nhf_spatial_targets.aggregate._coords import detect_coords
from nhf_spatial_targets.aggregate._driver import (
    WEIGHT_GEN_CRS,
    _batch_fingerprint,
    _weight_cache_fingerprint_path,
    _weight_cell_coords,
    load_and_batch_fabric,
    weight_cache_path,
)

SOURCE_KEY = "margulis_wus_sr"
SOURCE_CRS = "EPSG:4326"
VAR = "SWE"


def mask_path(datastore_dir: Path) -> Path:
    """Datastore location of the perennial-snow mask NC."""
    return Path(datastore_dir) / SOURCE_KEY / "derived" / "perennial_snow_mask.nc"


def archive_dir(project_dir: Path) -> Path:
    """Pre-mask archive written by the #363 rollout."""
    return Path(project_dir) / "archive" / "margulis_premask_2026-10-28"


def weight_cells(
    project_dir: Path, datastore_dir: Path, fabric_cfg: dict, year: int
) -> pd.DataFrame:
    """Return every cached weight row with its source cell's ``(lat, lon)``.

    Columns: ``hru`` (id as ``str``), ``wght``, ``cell_lat``, ``cell_lon``,
    ``batch_id``. Batches with no cache CSV (not covered by Margulis) are
    skipped. Raises if a cache's batch fingerprint sidecar does not match the
    batch rebuilt from the fabric (stale cache: ``i`` / ``j`` would point at
    the wrong cells).
    """
    project_dir = Path(project_dir)
    id_col = fabric_cfg["id_col"]
    cfg = yaml.safe_load((project_dir / "config.yml").read_text())
    batch_size = int((cfg.get("fabric") or {}).get("batch_size", 500))
    fabric_batched = load_and_batch_fabric(Path(fabric_cfg["path"]), batch_size)

    src = Path(datastore_dir) / SOURCE_KEY / "daily" / f"{SOURCE_KEY}_daily_{year}.nc"
    period = (f"{year}-01-01", f"{year}-12-31")
    frames = []
    with xr.open_dataset(src) as ds:
        x_coord, y_coord, time_coord = detect_coords(ds, VAR)
        for bid in sorted(int(b) for b in fabric_batched["batch_id"].unique()):
            csv = weight_cache_path(project_dir, SOURCE_KEY, bid)
            if not csv.exists():
                continue
            batch_gdf = fabric_batched[fabric_batched["batch_id"] == bid].drop(
                columns=["batch_id"]
            )
            fp_path = _weight_cache_fingerprint_path(csv)
            expected = _batch_fingerprint(batch_gdf, id_col, source_crs=SOURCE_CRS)
            if not fp_path.exists() or fp_path.read_text().strip() != expected:
                raise RuntimeError(
                    f"weight cache {csv} is stale (fingerprint sidecar missing or "
                    "different); re-run `agg margulis-wus-sr` before this check"
                )
            weights = pd.read_csv(csv)
            user_data = UserCatData(
                source_ds=ds,
                source_crs=SOURCE_CRS,
                source_x_coord=x_coord,
                source_y_coord=y_coord,
                source_t_coord=time_coord,
                source_var=[VAR],
                target_gdf=batch_gdf,
                target_crs=WEIGHT_GEN_CRS,
                target_id=id_col,
                source_time_period=[period[0], period[1]],
            )
            cell_y, cell_x = _weight_cell_coords(
                user_data, VAR, weights, x_coord=x_coord, y_coord=y_coord
            )
            hru_col = id_col if id_col in weights.columns else weights.columns[0]
            frames.append(
                pd.DataFrame(
                    {
                        # the cached CSV stores ids as int, fresh weights as str
                        "hru": weights[hru_col].astype(str).to_numpy(),
                        "wght": weights["wght"].to_numpy(dtype="float64"),
                        "cell_lat": np.asarray(cell_y),
                        "cell_lon": np.asarray(cell_x),
                        "batch_id": bid,
                    }
                )
            )
    if not frames:
        raise FileNotFoundError(f"no cached Margulis weight CSVs under {project_dir}")
    return pd.concat(frames, ignore_index=True)


def flag_perennial(cells: pd.DataFrame, mask_nc: Path) -> pd.DataFrame:
    """Add a boolean ``perennial`` column by exact ``(lat, lon)`` mask lookup.

    Same lookup as ``make_perennial_weight_filter``; raises if any row's cell
    is absent from the mask grid.
    """
    with xr.open_dataset(mask_nc) as m:
        perennial = m["perennial"].transpose("lat", "lon").values.astype(bool)
        lat_index = pd.Index(m["lat"].values)
        lon_index = pd.Index(m["lon"].values)
    iy = lat_index.get_indexer(cells["cell_lat"].to_numpy())
    ix = lon_index.get_indexer(cells["cell_lon"].to_numpy())
    if ((iy < 0) | (ix < 0)).any():
        raise ValueError("weight rows reference cells absent from the mask grid")
    return cells.assign(perennial=perennial[iy, ix])


def hru_mask_fractions(cells: pd.DataFrame) -> pd.DataFrame:
    """Per-HRU masked-area fraction from rows carrying ``perennial``.

    Columns: ``w_total``, ``w_masked``, ``frac_masked``, ``n_cells``,
    ``n_unmasked``. Indexed by ``hru`` (str).
    """
    w = cells["wght"].to_numpy()
    p = cells["perennial"].to_numpy()
    df = pd.DataFrame(
        {
            "hru": cells["hru"].to_numpy(),
            "w_total": w,
            "w_masked": np.where(p, w, 0.0),
            "n_unmasked": (~p).astype(int),
        }
    )
    out = df.groupby("hru").agg(
        w_total=("w_total", "sum"),
        w_masked=("w_masked", "sum"),
        n_cells=("w_total", "size"),
        n_unmasked=("n_unmasked", "sum"),
    )
    out["frac_masked"] = out["w_masked"] / out["w_total"]
    return out
