"""Unit tests for per-year streaming aggregation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from shapely.geometry import box

from nhf_spatial_targets.workspace import load as load_project


def _write_nc(path: Path, times: pd.DatetimeIndex) -> None:
    xr.Dataset(
        {"v": (["time", "lat", "lon"], np.ones((len(times), 2, 2)))},
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "lat": ("lat", [0.0, 1.0], {"standard_name": "latitude"}),
            "lon": ("lon", [0.0, 1.0], {"standard_name": "longitude"}),
        },
    ).to_netcdf(path)


def test_enumerate_years_per_year_files(tmp_path):
    from nhf_spatial_targets.aggregate._driver import enumerate_years

    f2000 = tmp_path / "src_2000_consolidated.nc"
    f2001 = tmp_path / "src_2001_consolidated.nc"
    _write_nc(f2000, pd.date_range("2000-01-01", periods=12, freq="MS"))
    _write_nc(f2001, pd.date_range("2001-01-01", periods=12, freq="MS"))

    year_files = enumerate_years([f2000, f2001])
    assert year_files == [(2000, f2000), (2001, f2001)]


def test_enumerate_years_single_multi_year_file(tmp_path):
    from nhf_spatial_targets.aggregate._driver import enumerate_years

    f = tmp_path / "src_consolidated.nc"
    _write_nc(f, pd.date_range("2000-01-01", periods=36, freq="MS"))

    year_files = enumerate_years([f])
    assert year_files == [(2000, f), (2001, f), (2002, f)]


def test_enumerate_years_raises_on_year_overlap(tmp_path):
    from nhf_spatial_targets.aggregate._driver import enumerate_years

    fa = tmp_path / "src_a_consolidated.nc"
    fb = tmp_path / "src_b_consolidated.nc"
    _write_nc(fa, pd.date_range("2001-01-01", periods=12, freq="MS"))
    _write_nc(fb, pd.date_range("2001-06-01", periods=12, freq="MS"))

    with pytest.raises(ValueError, match="overlap"):
        enumerate_years([fa, fb])


def test_enumerate_years_raises_when_no_time_coord(tmp_path):
    from nhf_spatial_targets.aggregate._driver import enumerate_years

    f = tmp_path / "no_time_consolidated.nc"
    xr.Dataset({"v": (["y", "x"], np.zeros((2, 2)))}).to_netcdf(f)
    with pytest.raises(ValueError, match="time"):
        enumerate_years([f])


@pytest.fixture()
def project(tmp_path):
    datastore = tmp_path / "datastore"
    datastore.mkdir()
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": "", "id_col": "hru_id"},
                "datastore": str(datastore),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "abc"}))
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    (tmp_path / "data" / "aggregated").mkdir(parents=True)
    (tmp_path / "weights").mkdir()
    return load_project(tmp_path)


@pytest.fixture()
def tiny_batched_fabric():
    polys = [box(i, 0, i + 1, 1) for i in range(2)]
    gdf = gpd.GeoDataFrame(
        {"hru_id": range(2), "batch_id": [0, 0]}, geometry=polys, crs="EPSG:4326"
    )
    return gdf


def test_per_year_output_path(project):
    from nhf_spatial_targets.aggregate._driver import per_year_output_path

    p = per_year_output_path(project, "foo", 2005)
    assert p == (project.workdir / "data" / "aggregated" / "foo" / "foo_2005_agg.nc")


def test_aggregate_year_skips_when_intermediate_exists(project, tiny_batched_fabric):
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import (
        aggregate_year,
        per_year_output_path,
    )

    out_path = per_year_output_path(project, "merra2", 2005)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2005-01-01", periods=1, freq="MS")
    xr.Dataset(
        {"GWETTOP": (["time", "hru_id"], np.ones((1, 2)))},
        coords={"time": times, "hru_id": [0, 1]},
    ).to_netcdf(out_path)

    src_dir = project.raw_dir("merra2")
    src_dir.mkdir(parents=True, exist_ok=True)
    src_file = src_dir / "src_2005_consolidated.nc"
    _write_nc(src_file, pd.date_range("2005-01-01", periods=12, freq="MS"))

    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=["v"]
    )
    with patch(
        "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch"
    ) as mock_batch:
        path = aggregate_year(
            adapter, project, 2005, src_file, tiny_batched_fabric, "hru_id"
        )
    assert path == out_path
    assert not mock_batch.called, "existing intermediate must skip aggregation"


def test_aggregate_year_writes_intermediate_when_missing(project, tiny_batched_fabric):
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import (
        aggregate_year,
        per_year_output_path,
    )

    src_dir = project.raw_dir("merra2")
    src_dir.mkdir(parents=True, exist_ok=True)
    src_file = src_dir / "src_2005_consolidated.nc"
    _write_nc(src_file, pd.date_range("2005-01-01", periods=12, freq="MS"))

    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=["v"]
    )

    fake_weights = pd.DataFrame({"i": [0], "j": [0], "wght": [1.0], "hru_id": [0]})
    fake_year_ds = xr.Dataset(
        {"v": (["time", "hru_id"], np.ones((1, 2)))},
        coords={
            "time": (
                "time",
                pd.date_range("2005-01-01", periods=1, freq="MS"),
                {"standard_name": "time"},
            ),
            "hru_id": [0, 1],
        },
    )
    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=fake_weights,
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            return_value=fake_year_ds,
        ) as mock_batch,
    ):
        path = aggregate_year(
            adapter, project, 2005, src_file, tiny_batched_fabric, "hru_id"
        )

    assert path == per_year_output_path(project, "merra2", 2005)
    assert path.exists()
    call = mock_batch.call_args
    assert call.kwargs["period"] == ("2005-01-01", "2005-12-31")


def _write_year_intermediate(path: Path, year: int) -> None:
    times = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
    xr.Dataset(
        {"v": (["time", "hru_id"], np.ones((12, 2)) * year)},
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "hru_id": [0, 1],
        },
    ).to_netcdf(path)


def test_concat_years_orders_by_time(tmp_path):
    from nhf_spatial_targets.aggregate._driver import concat_years

    p2001 = tmp_path / "src_2001_agg.nc"
    p2000 = tmp_path / "src_2000_agg.nc"
    _write_year_intermediate(p2001, 2001)
    _write_year_intermediate(p2000, 2000)

    combined = concat_years([p2001, p2000], time_coord="time")
    years = pd.DatetimeIndex(combined["time"].values).year
    assert list(years[:12]) == [2000] * 12
    assert list(years[-12:]) == [2001] * 12


def test_concat_years_raises_on_duplicate_time(tmp_path):
    from nhf_spatial_targets.aggregate._driver import concat_years

    p1 = tmp_path / "src_a_agg.nc"
    p2 = tmp_path / "src_b_agg.nc"
    _write_year_intermediate(p1, 2001)
    _write_year_intermediate(p2, 2001)
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        concat_years([p1, p2], time_coord="time")


def test_concat_years_raises_on_year_gap(tmp_path):
    from nhf_spatial_targets.aggregate._driver import concat_years

    p2000 = tmp_path / "src_2000_agg.nc"
    p2002 = tmp_path / "src_2002_agg.nc"
    _write_year_intermediate(p2000, 2000)
    _write_year_intermediate(p2002, 2002)
    with pytest.raises(ValueError, match="year gap"):
        concat_years([p2000, p2002], time_coord="time")


def test_concat_years_handles_cftime_calendar(tmp_path):
    """Year-gap detection must not crash when time coord is cftime (noleap etc.)."""
    from nhf_spatial_targets.aggregate._driver import concat_years

    def _write_cftime_year(path: Path, year: int) -> None:
        times = xr.date_range(
            f"{year}-01-01",
            periods=12,
            freq="MS",
            calendar="noleap",
            use_cftime=True,
        )
        xr.Dataset(
            {"v": (["time", "hru_id"], np.ones((12, 2)) * year)},
            coords={
                "time": ("time", times, {"standard_name": "time"}),
                "hru_id": [0, 1],
            },
        ).to_netcdf(path)

    p2000 = tmp_path / "src_2000_agg.nc"
    p2001 = tmp_path / "src_2001_agg.nc"
    _write_cftime_year(p2000, 2000)
    _write_cftime_year(p2001, 2001)

    combined = concat_years([p2000, p2001], time_coord="time")
    assert combined["v"].values.shape == (24, 2)


def test_concat_years_cftime_year_gap_still_detected(tmp_path):
    """Even on cftime calendars, interior year gaps must raise."""
    from nhf_spatial_targets.aggregate._driver import concat_years

    def _write_cftime_year(path: Path, year: int) -> None:
        times = xr.date_range(
            f"{year}-01-01",
            periods=12,
            freq="MS",
            calendar="noleap",
            use_cftime=True,
        )
        xr.Dataset(
            {"v": (["time", "hru_id"], np.ones((12, 2)))},
            coords={
                "time": ("time", times, {"standard_name": "time"}),
                "hru_id": [0, 1],
            },
        ).to_netcdf(path)

    p2000 = tmp_path / "src_2000_agg.nc"
    p2002 = tmp_path / "src_2002_agg.nc"
    _write_cftime_year(p2000, 2000)
    _write_cftime_year(p2002, 2002)
    with pytest.raises(ValueError, match="year gap"):
        concat_years([p2000, p2002], time_coord="time")


def test_concat_years_detaches_from_disk(tmp_path):
    """Returned Dataset must remain usable after source intermediates deleted."""
    from nhf_spatial_targets.aggregate._driver import concat_years

    p2000 = tmp_path / "src_2000_agg.nc"
    p2001 = tmp_path / "src_2001_agg.nc"
    _write_year_intermediate(p2000, 2000)
    _write_year_intermediate(p2001, 2001)

    combined = concat_years([p2000, p2001], time_coord="time")
    p2000.unlink()
    p2001.unlink()
    assert combined["v"].values.shape == (24, 2)


def test_find_time_coord_returns_none_without_cf_attrs(tmp_path):
    """Legacy literal-name fallback is removed — return None for non-CF coord."""
    from nhf_spatial_targets.aggregate._driver import _find_time_coord_name

    ds = xr.Dataset(
        {"v": (["time", "x"], np.zeros((3, 2)))},
        coords={
            "time": ("time", pd.date_range("2000-01-01", periods=3, freq="D")),
            "x": ("x", [0.0, 1.0]),
        },
    )
    assert _find_time_coord_name(ds) is None


def test_aggregate_year_preserves_exception_type_and_adds_context_note(
    project, tiny_batched_fabric, tmp_path
):
    """Batch failures must surface with original exception type and a note
    carrying year/batch/source_file provenance."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_year

    src = tmp_path / "datastore" / "era5_land" / "era5_land_monthly_2005.nc"
    src.parent.mkdir(parents=True, exist_ok=True)
    _write_nc(src, pd.date_range("2005-01-01", periods=12, freq="MS"))

    adapter = SourceAdapter(
        source_key="era5_land",
        output_name="era5_land_agg.nc",
        variables=("v",),
    )

    class GdptoolsSentinel(RuntimeError):
        pass

    def boom(**_kwargs):
        raise GdptoolsSentinel("gdptools exploded")

    with patch(
        "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
        side_effect=boom,
    ):
        with pytest.raises(GdptoolsSentinel) as excinfo:
            aggregate_year(adapter, project, 2005, src, tiny_batched_fabric, "hru_id")

    notes = getattr(excinfo.value, "__notes__", [])
    assert any(
        "year=2005" in n and "batch=0" in n and "era5_land_monthly_2005.nc" in n
        for n in notes
    ), f"Expected provenance note; got notes={notes}"


def test_aggregate_year_attaches_cf_global_attrs(project, tiny_batched_fabric):
    """Each per-year file must carry Conventions/history/source independent
    of any consolidation step."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import (
        aggregate_year,
        per_year_output_path,
    )

    src_dir = project.raw_dir("merra2")
    src_dir.mkdir(parents=True, exist_ok=True)
    src_file = src_dir / "src_2005_consolidated.nc"
    _write_nc(src_file, pd.date_range("2005-01-01", periods=12, freq="MS"))

    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=["v"]
    )
    fake_weights = pd.DataFrame({"i": [0], "j": [0], "wght": [1.0], "hru_id": [0]})
    fake_year_ds = xr.Dataset(
        {"v": (["time", "hru_id"], np.ones((1, 2)))},
        coords={
            "time": (
                "time",
                pd.date_range("2005-01-01", periods=1, freq="MS"),
                {"standard_name": "time"},
            ),
            "hru_id": [0, 1],
        },
    )
    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.catalog_source",
            return_value={"access": {"doi": "10.0/TEST"}},
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=fake_weights,
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            return_value=fake_year_ds,
        ),
    ):
        aggregate_year(adapter, project, 2005, src_file, tiny_batched_fabric, "hru_id")

    with xr.open_dataset(per_year_output_path(project, "merra2", 2005)) as written:
        assert written.attrs["Conventions"] == "CF-1.6"
        assert written.attrs["source"] == "merra2"
        assert "aggregated to HRU fabric" in written.attrs["history"]
        assert written.attrs["source_doi"] == "10.0/TEST"


def test_aggregate_year_runs_post_aggregate_hook(project, tiny_batched_fabric):
    """post_aggregate_hook must run inside aggregate_year on the per-year
    dataset (before the atomic write)."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import (
        aggregate_year,
        per_year_output_path,
    )

    src_dir = project.raw_dir("merra2")
    src_dir.mkdir(parents=True, exist_ok=True)
    src_file = src_dir / "src_2005_consolidated.nc"
    _write_nc(src_file, pd.date_range("2005-01-01", periods=12, freq="MS"))

    calls: list[int] = []

    def post_hook(ds):
        calls.append(1)
        return ds.rename({"v": "v_renamed"})

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["v"],
        post_aggregate_hook=post_hook,
    )
    fake_weights = pd.DataFrame({"i": [0], "j": [0], "wght": [1.0], "hru_id": [0]})
    fake_year_ds = xr.Dataset(
        {"v": (["time", "hru_id"], np.ones((1, 2)))},
        coords={
            "time": (
                "time",
                pd.date_range("2005-01-01", periods=1, freq="MS"),
                {"standard_name": "time"},
            ),
            "hru_id": [0, 1],
        },
    )
    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.catalog_source",
            return_value={"access": {"type": "local_nc"}},
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=fake_weights,
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            return_value=fake_year_ds,
        ),
    ):
        aggregate_year(adapter, project, 2005, src_file, tiny_batched_fabric, "hru_id")

    assert calls, "post_aggregate_hook was not invoked"
    with xr.open_dataset(per_year_output_path(project, "merra2", 2005)) as written:
        assert "v_renamed" in written.data_vars
        assert "v" not in written.data_vars


def test_migrate_legacy_layout_moves_by_year_files(project):
    from nhf_spatial_targets.aggregate._driver import _migrate_legacy_layout

    legacy_dir = project.workdir / "data" / "aggregated" / "_by_year"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    f2000 = legacy_dir / "mod10c1_v061_2000_agg.nc"
    f2001 = legacy_dir / "mod10c1_v061_2001_agg.nc"
    _write_year_intermediate(f2000, 2000)
    _write_year_intermediate(f2001, 2001)

    _migrate_legacy_layout(project, "mod10c1_v061")

    new_dir = project.workdir / "data" / "aggregated" / "mod10c1_v061"
    assert (new_dir / "mod10c1_v061_2000_agg.nc").exists()
    assert (new_dir / "mod10c1_v061_2001_agg.nc").exists()
    assert not f2000.exists()
    assert not f2001.exists()


def test_migrate_legacy_layout_removes_stale_consolidated(project):
    from nhf_spatial_targets.aggregate._driver import _migrate_legacy_layout

    agg_dir = project.workdir / "data" / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    stale = agg_dir / "mod10c1_v061_agg.nc"
    stale.write_bytes(b"placeholder")

    _migrate_legacy_layout(project, "mod10c1_v061")

    assert not stale.exists()


def test_migrate_legacy_layout_idempotent(project):
    from nhf_spatial_targets.aggregate._driver import _migrate_legacy_layout

    legacy_dir = project.workdir / "data" / "aggregated" / "_by_year"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    _write_year_intermediate(legacy_dir / "foo_2000_agg.nc", 2000)

    _migrate_legacy_layout(project, "foo")
    _migrate_legacy_layout(project, "foo")  # must not raise

    new_file = project.workdir / "data" / "aggregated" / "foo" / "foo_2000_agg.nc"
    assert new_file.exists()


def test_migrate_legacy_layout_collision_leaves_both(project):
    from nhf_spatial_targets.aggregate._driver import _migrate_legacy_layout

    legacy_dir = project.workdir / "data" / "aggregated" / "_by_year"
    new_dir = project.workdir / "data" / "aggregated" / "foo"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)
    legacy_file = legacy_dir / "foo_2000_agg.nc"
    new_file = new_dir / "foo_2000_agg.nc"
    _write_year_intermediate(legacy_file, 2000)
    _write_year_intermediate(new_file, 2000)

    _migrate_legacy_layout(project, "foo")

    # New-path file is canonical and untouched; legacy file is left in place.
    assert legacy_file.exists()
    assert new_file.exists()


def test_verify_year_coverage_ok_on_contiguous(tmp_path):
    from nhf_spatial_targets.aggregate._driver import _verify_year_coverage

    d = tmp_path / "foo"
    d.mkdir()
    for y in (2000, 2001, 2002):
        (d / f"foo_{y}_agg.nc").write_bytes(b"")

    # Must not raise.
    _verify_year_coverage(d, "foo")


def test_verify_year_coverage_raises_on_interior_gap(tmp_path):
    from nhf_spatial_targets.aggregate._driver import _verify_year_coverage

    d = tmp_path / "foo"
    d.mkdir()
    for y in (2000, 2002, 2003):
        (d / f"foo_{y}_agg.nc").write_bytes(b"")

    with pytest.raises(ValueError, match=r"missing=\[2001\]"):
        _verify_year_coverage(d, "foo")


def test_verify_year_coverage_raises_on_empty(tmp_path):
    from nhf_spatial_targets.aggregate._driver import _verify_year_coverage

    d = tmp_path / "foo"
    d.mkdir()
    with pytest.raises(ValueError, match="no per-year"):
        _verify_year_coverage(d, "foo")
