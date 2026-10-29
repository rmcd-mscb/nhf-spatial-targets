"""Unit tests for notebooks/aggregated/_margulis_mask.py (#368).

The helper module lives outside the package, so we load it via importlib
rather than a regular import.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from shapely.geometry import box

from nhf_spatial_targets.aggregate._driver import (
    _batch_fingerprint,
    _weight_cache_fingerprint_path,
    weight_cache_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MASK_HELPER_PATH = REPO_ROOT / "notebooks" / "aggregated" / "_margulis_mask.py"
YEAR = 2010
AGG_BATCH_SIZE = 4  # what "agg" ran with: all four HRUs in batch 0
CONFIG_BATCH_SIZE = 2  # what config.yml says: splits them into two batches


@pytest.fixture(scope="module")
def mm():
    spec = importlib.util.spec_from_file_location("margulis_mask", MASK_HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def project(tmp_path, mm):
    """Four-HRU project whose weight cache was built at AGG_BATCH_SIZE."""
    project_dir = tmp_path / "project"
    datastore_dir = tmp_path / "datastore"
    (project_dir / "weights").mkdir(parents=True)

    fabric = gpd.GeoDataFrame(
        {"hru_id": [1, 2, 3, 4]},
        geometry=[
            box(x, 40.0, x + 0.5, 40.5) for x in (-120.0, -119.5, -119.0, -118.5)
        ],
        crs="EPSG:4326",
    )
    fabric_path = tmp_path / "fabric.gpkg"
    fabric.to_file(fabric_path, driver="GPKG")
    (project_dir / "config.yml").write_text(
        yaml.safe_dump({"fabric": {"batch_size": CONFIG_BATCH_SIZE}})
    )

    src = datastore_dir / mm.SOURCE_KEY / "daily"
    src.mkdir(parents=True)
    xr.Dataset(
        {mm.VAR: (("time", "lat", "lon"), np.zeros((1, 2, 4), dtype="float32"))},
        coords={
            "time": ("time", pd.to_datetime([f"{YEAR}-03-01"]), {"axis": "T"}),
            "lat": ("lat", [40.125, 40.375], {"axis": "Y"}),
            "lon": ("lon", [-119.75, -119.25, -118.75, -118.25], {"axis": "X"}),
        },
    ).to_netcdf(src / f"{mm.SOURCE_KEY}_daily_{YEAR}.nc")

    csv = weight_cache_path(project_dir, mm.SOURCE_KEY, 0)
    pd.DataFrame(
        {"hru_id": [1, 2, 3, 4], "i": [0] * 4, "j": [0, 1, 2, 3], "wght": [1.0] * 4}
    ).to_csv(csv, index=False)
    _weight_cache_fingerprint_path(csv).write_text(
        _batch_fingerprint(fabric, "hru_id", source_crs=mm.SOURCE_CRS)
    )
    fabric_cfg = {"path": str(fabric_path), "id_col": "hru_id"}
    return project_dir, datastore_dir, fabric_cfg, csv


def test_config_batch_size_mismatch_names_batch_size(mm, project):
    project_dir, datastore_dir, fabric_cfg, _ = project
    with pytest.raises(RuntimeError, match="batch_size=2.*--batch-size"):
        mm.weight_cells(project_dir, datastore_dir, fabric_cfg, YEAR)


def test_missing_sidecar_raises(mm, project):
    project_dir, datastore_dir, fabric_cfg, csv = project
    _weight_cache_fingerprint_path(csv).unlink()
    with pytest.raises(RuntimeError, match="no fingerprint sidecar"):
        mm.weight_cells(
            project_dir, datastore_dir, fabric_cfg, YEAR, batch_size=AGG_BATCH_SIZE
        )


def test_explicit_batch_size_overrides_config(mm, project, monkeypatch):
    project_dir, datastore_dir, fabric_cfg, _ = project
    # Cell resolution is the driver's job and covered there; stub it so this
    # test only exercises the batching / fingerprint path.
    monkeypatch.setattr(mm, "UserCatData", lambda **kwargs: None)
    monkeypatch.setattr(
        mm,
        "_weight_cell_coords",
        lambda user_data, var, weights, x_coord, y_coord: (
            np.full(len(weights), 40.125),
            np.array([-119.75, -119.25, -118.75, -118.25])[weights["j"].to_numpy()],
        ),
    )
    cells = mm.weight_cells(
        project_dir, datastore_dir, fabric_cfg, YEAR, batch_size=AGG_BATCH_SIZE
    )
    assert list(cells["hru"]) == ["1", "2", "3", "4"]
    assert (cells["batch_id"] == 0).all()
    np.testing.assert_allclose(cells["cell_lon"], [-119.75, -119.25, -118.75, -118.25])
