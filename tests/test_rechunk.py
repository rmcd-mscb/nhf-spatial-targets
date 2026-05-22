"""Tests for the rechunk backfill CLI (issue #165 ST5)."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from tests.conftest import make_minimal_project, write_year_nc


def _write_contiguous_target(path: Path, id_col: str = "nhm_id") -> None:
    """A target-style NC written plain (contiguous, uncompressed)."""
    n_time, n_hru = 12, 200
    times = pd.date_range("2000-01-01", periods=n_time, freq="MS")
    lower = np.random.default_rng(0).random((n_time, n_hru)).astype("float32")
    ds = xr.Dataset(
        {
            "lower_bound": (("time", id_col), lower),
            "upper_bound": (("time", id_col), lower + 1.0),
            "n_sources": (("time", id_col), np.ones((n_time, n_hru), dtype="int8")),
        },
        coords={"time": times, id_col: np.arange(1, n_hru + 1)},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _data_var_chunked(path: Path) -> bool:
    with netCDF4.Dataset(path) as nc:
        for v in nc.variables.values():
            if v.ndim >= 2:
                return v.chunking() != "contiguous" and bool(v.filters().get("zlib"))
    return False


# --- core rechunk behavior ----------------------------------------------


def test_rechunk_converts_contiguous_aggregated_to_chunked(tmp_path: Path):
    from nhf_spatial_targets.rechunk import rechunk_project
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    f = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    write_year_nc(f, 2000, "ro")  # contiguous, uncompressed
    assert not _data_var_chunked(f)

    project = load(workdir)
    results = rechunk_project(project, layer="aggregated")

    assert _data_var_chunked(f)  # now chunked + compressed
    statuses = {r["path"].name: r["status"] for r in results}
    assert statuses[f.name] == "rechunked"


def test_rechunk_is_bit_identical(tmp_path: Path):
    from nhf_spatial_targets.rechunk import rechunk_project
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    f = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    write_year_nc(f, 2000, "ro")
    before = xr.open_dataset(f).load()

    project = load(workdir)
    rechunk_project(project, layer="aggregated")

    after = xr.open_dataset(f).load()
    np.testing.assert_array_equal(before["ro"].values, after["ro"].values)


def test_rechunk_is_idempotent(tmp_path: Path):
    from nhf_spatial_targets.rechunk import rechunk_project
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    f = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    write_year_nc(f, 2000, "ro")
    project = load(workdir)

    rechunk_project(project, layer="aggregated")
    results2 = rechunk_project(project, layer="aggregated")  # second pass

    assert all(r["status"] == "skipped" for r in results2)


def test_rechunk_dry_run_does_not_write(tmp_path: Path):
    from nhf_spatial_targets.rechunk import rechunk_project
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    f = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    write_year_nc(f, 2000, "ro")
    mtime_before = f.stat().st_mtime_ns

    project = load(workdir)
    results = rechunk_project(project, layer="aggregated", dry_run=True)

    assert f.stat().st_mtime_ns == mtime_before  # untouched
    assert not _data_var_chunked(f)
    assert all(r["status"] == "would-rechunk" for r in results)


def test_rechunk_skips_daymet_and_ssebop(tmp_path: Path):
    """daymet/ssebop outputs are intentionally left as-is (#165 ST3a)."""
    from nhf_spatial_targets.rechunk import rechunk_project
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    dm = workdir / "data" / "aggregated" / "daymet" / "daymet_2000_agg.nc"
    write_year_nc(dm, 2000, "swe")
    project = load(workdir)

    results = rechunk_project(project, layer="aggregated")

    assert not _data_var_chunked(dm)  # untouched
    assert dm.name not in {r["path"].name for r in results}


def test_rechunk_target_layer(tmp_path: Path):
    from nhf_spatial_targets.rechunk import rechunk_project
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    t = workdir / "targets" / "runoff_targets.nc"
    _write_contiguous_target(t)
    assert not _data_var_chunked(t)

    project = load(workdir)
    rechunk_project(project, layer="target")

    assert _data_var_chunked(t)


def test_cli_rechunk_command_runs(tmp_path: Path, capsys):
    """The CLI command wires through to rechunk_project (dry-run smoke)."""
    from nhf_spatial_targets import cli

    workdir = make_minimal_project(tmp_path)
    f = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    write_year_nc(f, 2000, "ro")

    cli.rechunk(workdir, dry_run=True)

    out = capsys.readouterr().out
    assert "dry-run" in out
    assert not _data_var_chunked(f)  # dry-run leaves the file untouched


def test_rechunk_source_filter(tmp_path: Path):
    from nhf_spatial_targets.rechunk import rechunk_project
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    a = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    b = workdir / "data" / "aggregated" / "gldas" / "gldas_2000_agg.nc"
    write_year_nc(a, 2000, "ro")
    write_year_nc(b, 2000, "ro")

    project = load(workdir)
    rechunk_project(project, layer="aggregated", source="era5_land")

    assert _data_var_chunked(a)
    assert not _data_var_chunked(b)  # filtered out
