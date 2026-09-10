"""Unit tests for the id_col relabel migration (issue #353).

The Oregon project keyed every artifact on ``nhm_id`` (sparse national,
1-41195) and moves to ``hru_id`` (dense local, 1-16814). Geometry never
changes, so values are identical and only labels move -- which is
exactly the kind of migration that corrupts silently when it goes
wrong, because nothing about the file's shape or dtype changes.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from nhf_spatial_targets.relabel_id import (
    build_id_map,
    relabel_dataset,
    relabel_weight_frame,
)


def _fabric(from_ids, to_ids) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "nhm_id": list(from_ids),
            "hru_id": list(to_ids),
            "geometry": [box(i, 0, i + 1, 1) for i in range(len(from_ids))],
        },
        crs="EPSG:5070",
    )


# --- build_id_map ---------------------------------------------------------


def test_build_id_map_pairs_the_columns_rowwise():
    fab = _fabric([10, 3566, 41195], [1, 2, 3])
    assert build_id_map(fab, "nhm_id", "hru_id") == {10: 1, 3566: 2, 41195: 3}


def test_build_id_map_rejects_a_missing_column():
    fab = _fabric([1, 2], [1, 2])
    with pytest.raises(KeyError, match="or_id"):
        build_id_map(fab, "nhm_id", "or_id")


def test_build_id_map_rejects_a_non_unique_source():
    fab = _fabric([7, 7], [1, 2])
    with pytest.raises(ValueError, match="not unique"):
        build_id_map(fab, "nhm_id", "hru_id")


def test_build_id_map_rejects_a_non_unique_destination():
    """A collapsing map would silently merge two HRUs into one."""
    fab = _fabric([1, 2], [5, 5])
    with pytest.raises(ValueError, match="not unique"):
        build_id_map(fab, "nhm_id", "hru_id")


# --- relabel_dataset ------------------------------------------------------


def _dataset(ids, dim="nhm_id") -> xr.Dataset:
    """Values encode their own id, so misalignment is detectable."""
    ids = np.asarray(ids, dtype="int32")
    time = pd.date_range("2005-01-01", periods=3, freq="MS")
    data = np.tile(ids.astype("float32") * 100.0, (len(time), 1))
    return xr.Dataset(
        {"value": (("time", dim), data)},
        coords={"time": time, dim: ids},
    )


def test_relabel_dataset_renames_the_dim_and_maps_the_values():
    ds = _dataset([10, 3566, 41195])
    out = relabel_dataset(ds, "nhm_id", "hru_id", {10: 1, 3566: 2, 41195: 3})
    assert "hru_id" in out.dims
    assert "nhm_id" not in out.dims
    assert out["hru_id"].values.tolist() == [1, 2, 3]
    assert out["value"].dims == ("time", "hru_id")


def test_relabel_dataset_keeps_data_attached_to_its_own_row():
    """The whole point: value 3566*100 must still sit at the row that was 3566."""
    ds = _dataset([10, 3566, 41195])
    out = relabel_dataset(ds, "nhm_id", "hru_id", {10: 1, 3566: 2, 41195: 3})
    got = out["value"].sel(hru_id=2).values
    assert np.allclose(got, 3566 * 100.0)


def test_relabel_dataset_sorts_when_the_new_ids_are_out_of_order():
    """Canonical row order is id_col ascending (issue #93)."""
    ds = _dataset([10, 3566, 41195])
    # Deliberately order-reversing map.
    out = relabel_dataset(ds, "nhm_id", "hru_id", {10: 3, 3566: 2, 41195: 1})
    assert out["hru_id"].values.tolist() == [1, 2, 3]
    # ...and the data followed its row, rather than staying put.
    assert np.allclose(out["value"].sel(hru_id=3).values, 10 * 100.0)
    assert np.allclose(out["value"].sel(hru_id=1).values, 41195 * 100.0)


def test_relabel_dataset_rejects_an_id_missing_from_the_map():
    """Abort loudly rather than invent a label."""
    ds = _dataset([10, 999])
    with pytest.raises(ValueError, match="not in the id map"):
        relabel_dataset(ds, "nhm_id", "hru_id", {10: 1})


def test_relabel_dataset_is_idempotent_when_already_relabelled():
    ds = _dataset([1, 2, 3], dim="hru_id")
    out = relabel_dataset(ds, "nhm_id", "hru_id", {10: 1})
    assert out is ds


def test_relabel_dataset_rejects_a_dataset_with_neither_dim():
    ds = _dataset([1, 2], dim="something_else")
    with pytest.raises(ValueError, match="neither"):
        relabel_dataset(ds, "nhm_id", "hru_id", {1: 1, 2: 2})


def test_relabel_dataset_preserves_variable_attrs():
    ds = _dataset([10, 3566])
    ds["value"].attrs["units"] = "mm"
    out = relabel_dataset(ds, "nhm_id", "hru_id", {10: 1, 3566: 2})
    assert out["value"].attrs["units"] == "mm"


# --- relabel_weight_frame -------------------------------------------------


def test_relabel_weight_frame_renames_and_maps():
    df = pd.DataFrame(
        {
            "nhm_id": [10, 10, 3566],
            "i": [1, 2, 3],
            "j": [4, 5, 6],
            "wght": [0.1, 0.2, 0.7],
        }
    )
    out = relabel_weight_frame(df, "nhm_id", "hru_id", {10: 1, 3566: 2})
    assert list(out.columns) == ["hru_id", "i", "j", "wght"]
    assert out["hru_id"].tolist() == [1, 1, 2]
    assert out["wght"].tolist() == [0.1, 0.2, 0.7]


def test_relabel_weight_frame_rejects_an_unmapped_id():
    df = pd.DataFrame(
        {"nhm_id": [10, 999], "i": [1, 2], "j": [3, 4], "wght": [0.5, 0.5]}
    )
    with pytest.raises(ValueError, match="not in the id map"):
        relabel_weight_frame(df, "nhm_id", "hru_id", {10: 1})


def test_relabel_weight_frame_is_idempotent_when_already_relabelled():
    df = pd.DataFrame({"hru_id": [1, 2], "i": [1, 2], "j": [3, 4], "wght": [0.5, 0.5]})
    out = relabel_weight_frame(df, "nhm_id", "hru_id", {10: 1})
    assert out is df


# --- relabel_project (the file walker) ------------------------------------


@pytest.fixture
def relabel_project_dir(tmp_path):
    """A minimal project: two aggregated NCs and one weight CSV."""
    from nhf_spatial_targets.io_nc import atomic_to_netcdf

    agg = tmp_path / "data" / "aggregated" / "era5_land"
    agg.mkdir(parents=True)
    for year in (1980, 1981):
        atomic_to_netcdf(_dataset([10, 3566, 41195]), agg / f"era5_{year}_agg.nc")

    weights = tmp_path / "weights"
    weights.mkdir()
    pd.DataFrame(
        {"nhm_id": [10, 3566, 41195], "i": [1, 2, 3], "j": [4, 5, 6], "wght": [0.5] * 3}
    ).to_csv(weights / "era5_land_batch0.csv", index=False)
    return tmp_path


def test_relabel_project_dry_run_writes_nothing(relabel_project_dir):
    from nhf_spatial_targets.relabel_id import relabel_project

    before = {
        p: p.stat().st_mtime_ns for p in relabel_project_dir.rglob("*") if p.is_file()
    }
    results = relabel_project(
        relabel_project_dir,
        "nhm_id",
        "hru_id",
        {10: 1, 3566: 2, 41195: 3},
        dry_run=True,
    )
    after = {
        p: p.stat().st_mtime_ns for p in relabel_project_dir.rglob("*") if p.is_file()
    }
    assert before == after
    assert all(r["status"] == "would-relabel" for r in results)
    assert len(results) == 3  # 2 NCs + 1 CSV


def test_relabel_project_relabels_ncs_and_csvs(relabel_project_dir):
    from nhf_spatial_targets.relabel_id import relabel_project

    results = relabel_project(
        relabel_project_dir, "nhm_id", "hru_id", {10: 1, 3566: 2, 41195: 3}
    )
    assert {r["status"] for r in results} == {"relabelled"}

    nc = relabel_project_dir / "data/aggregated/era5_land/era5_1980_agg.nc"
    with xr.open_dataset(nc) as ds:
        assert "hru_id" in ds.dims and "nhm_id" not in ds.dims
        assert ds["hru_id"].values.tolist() == [1, 2, 3]
        # data still attached to its own row
        assert np.allclose(ds["value"].sel(hru_id=2).values, 3566 * 100.0)

    csv = pd.read_csv(relabel_project_dir / "weights/era5_land_batch0.csv")
    assert list(csv.columns)[0] == "hru_id"
    assert csv["hru_id"].tolist() == [1, 2, 3]


def test_relabel_project_is_idempotent(relabel_project_dir):
    """Re-running a partially migrated project must not double-map."""
    from nhf_spatial_targets.relabel_id import relabel_project

    relabel_project(relabel_project_dir, "nhm_id", "hru_id", {10: 1, 3566: 2, 41195: 3})
    second = relabel_project(
        relabel_project_dir, "nhm_id", "hru_id", {10: 1, 3566: 2, 41195: 3}
    )
    assert {r["status"] for r in second} == {"skipped-already-relabelled"}

    nc = relabel_project_dir / "data/aggregated/era5_land/era5_1980_agg.nc"
    with xr.open_dataset(nc) as ds:
        assert ds["hru_id"].values.tolist() == [1, 2, 3]


def test_relabel_project_aborts_the_file_on_an_unmapped_id(relabel_project_dir):
    """A bad map must not leave a half-migrated project silently."""
    from nhf_spatial_targets.relabel_id import relabel_project

    results = relabel_project(
        relabel_project_dir,
        "nhm_id",
        "hru_id",
        {10: 1},  # 3566 / 41195 missing
    )
    assert all(r["status"] == "failed" for r in results)
    # originals untouched
    nc = relabel_project_dir / "data/aggregated/era5_land/era5_1980_agg.nc"
    with xr.open_dataset(nc) as ds:
        assert "nhm_id" in ds.dims


# --- encoding preservation ------------------------------------------------


def test_relabel_preserves_on_disk_chunking_and_compression(tmp_path):
    """A relabel must change labels and nothing else.

    CLAUDE.md leaves the daymet/ssebop aggregated outputs deliberately
    unchunked (rechunk._SKIP_SOURCES), so imposing the aggregated-layer
    encoding formula here would silently re-encode files that policy
    says to leave alone. Asserted against the on-disk filters/chunking,
    not the encoding dict -- netCDF4 defaults shuffle=True under zlib,
    so the dict can disagree with the file.
    """
    import netCDF4

    from nhf_spatial_targets.relabel_id import _relabel_nc

    path = tmp_path / "daymet_1980_agg.nc"
    ds = _dataset([10, 3566, 41195])
    # Deliberately contiguous + uncompressed, like the daymet outputs.
    ds.to_netcdf(path, encoding={"value": {"zlib": False, "contiguous": True}})

    with netCDF4.Dataset(path) as nc:
        before = (nc["value"].chunking(), nc["value"].filters())

    _relabel_nc(path, "nhm_id", "hru_id", {10: 1, 3566: 2, 41195: 3})

    with netCDF4.Dataset(path) as nc:
        after = (nc["value"].chunking(), nc["value"].filters())
        assert "hru_id" in nc.dimensions
    assert after == before, f"encoding changed: {before} -> {after}"


def test_relabel_preserves_a_chunked_compressed_layout(tmp_path):
    import netCDF4

    from nhf_spatial_targets.relabel_id import _relabel_nc

    path = tmp_path / "era5_1980_agg.nc"
    ds = _dataset([10, 3566, 41195])
    ds.to_netcdf(
        path,
        encoding={"value": {"zlib": True, "complevel": 4, "chunksizes": (3, 2)}},
    )
    with netCDF4.Dataset(path) as nc:
        before = (nc["value"].chunking(), nc["value"].filters())

    _relabel_nc(path, "nhm_id", "hru_id", {10: 1, 3566: 2, 41195: 3})

    with netCDF4.Dataset(path) as nc:
        after = (nc["value"].chunking(), nc["value"].filters())
    assert after == before, f"encoding changed: {before} -> {after}"
