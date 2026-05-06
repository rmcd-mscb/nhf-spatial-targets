"""Tests for shared multi-source-minmax target machinery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nhf_spatial_targets.workspace import load
from tests.conftest import make_minimal_project, write_year_nc


def test_read_aggregated_source_concats_per_year_nc(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    write_year_nc(src_dir / f"{src}_2000_agg.nc", 2000, var)
    write_year_nc(src_dir / f"{src}_2001_agg.nc", 2001, var)

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

    workdir = make_minimal_project(tmp_path)
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    for y in (1999, 2000, 2001, 2002):
        write_year_nc(src_dir / f"{src}_{y}_agg.nc", y, var)

    project = load(workdir)
    da = read_aggregated_source(
        project, src, var, period=("2000-06-01", "2001-06-30"), chunks={"time": 12}
    )
    # months 2000-06 .. 2001-06 inclusive -> 13 months
    assert len(da.time) == 13


def test_read_aggregated_source_raises_when_dir_empty(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    project = load(workdir)
    with pytest.raises(FileNotFoundError, match="No aggregated NC files found"):
        read_aggregated_source(
            project, "era5_land", "ro", period=("2000-01-01", "2001-12-31")
        )


def test_read_aggregated_source_raises_when_period_outside_coverage(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    write_year_nc(src_dir / f"{src}_2000_agg.nc", 2000, var)

    project = load(workdir)
    with pytest.raises(ValueError, match="entirely outside source coverage"):
        read_aggregated_source(project, src, var, period=("2010-01-01", "2010-12-31"))


def test_read_aggregated_source_raises_diagnostic_on_missing_var(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = make_minimal_project(tmp_path)
    src = "era5_land"
    src_dir = workdir / "data" / "aggregated" / src
    write_year_nc(src_dir / f"{src}_2000_agg.nc", 2000, "ro")

    project = load(workdir)
    with pytest.raises(KeyError, match="Available variables"):
        read_aggregated_source(
            project, src, "not_a_var", period=("2000-01-01", "2000-12-31")
        )
