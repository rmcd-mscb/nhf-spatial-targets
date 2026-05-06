"""Tests for shared multi-source-minmax target machinery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from nhf_spatial_targets.workspace import load


def _write_year_nc(path: Path, year: int, var: str, id_col: str = "nhm_id"):
    """Write a synthetic per-year aggregated NC at <path>/<source_key>_<year>_agg.nc."""
    times = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    hrus = [1, 2, 3]
    data = np.arange(len(times) * len(hrus), dtype=np.float32).reshape(
        len(times), len(hrus)
    )
    ds = xr.Dataset(
        {var: ((("time", id_col)), data)},
        coords={"time": times, id_col: hrus},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _make_project(tmp_path: Path, source_keys: list[str]) -> Path:
    workdir = tmp_path / "proj"
    workdir.mkdir()
    (workdir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "datastore": str(tmp_path / "store"),
                "fabric": {"path": str(tmp_path / "f.gpkg"), "id_col": "nhm_id"},
            }
        )
    )
    (workdir / "fabric.json").write_text(json.dumps({"id_col": "nhm_id"}))
    return workdir


def test_read_aggregated_source_concats_per_year_nc(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = _make_project(tmp_path, ["era5_land"])
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    _write_year_nc(src_dir / f"{src}_2000_agg.nc", 2000, var)
    _write_year_nc(src_dir / f"{src}_2001_agg.nc", 2001, var)

    project = load(workdir)
    da = read_aggregated_source(
        project, src, var, period=("2000-01-01", "2001-12-31"), chunks={"time": 12}
    )
    assert da.dims == ("time", "nhm_id")
    assert len(da.time) == 24
    assert da.time.values[0] == np.datetime64("2000-01-01")
    assert da.time.values[-1] == np.datetime64("2001-12-01")


def test_read_aggregated_source_slices_to_period(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = _make_project(tmp_path, ["era5_land"])
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    for y in (1999, 2000, 2001, 2002):
        _write_year_nc(src_dir / f"{src}_{y}_agg.nc", y, var)

    project = load(workdir)
    da = read_aggregated_source(
        project, src, var, period=("2000-06-01", "2001-06-30"), chunks={"time": 12}
    )
    # months 2000-06 .. 2001-06 inclusive -> 13 months
    assert len(da.time) == 13


def test_read_aggregated_source_raises_when_dir_empty(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = _make_project(tmp_path, ["era5_land"])
    project = load(workdir)
    with pytest.raises(FileNotFoundError, match="No aggregated NC files found"):
        read_aggregated_source(
            project, "era5_land", "ro", period=("2000-01-01", "2001-12-31")
        )


def test_read_aggregated_source_raises_when_period_outside_coverage(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = _make_project(tmp_path, ["era5_land"])
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    _write_year_nc(src_dir / f"{src}_2000_agg.nc", 2000, var)

    project = load(workdir)
    with pytest.raises(ValueError, match="entirely outside source coverage"):
        read_aggregated_source(project, src, var, period=("2010-01-01", "2010-12-31"))
