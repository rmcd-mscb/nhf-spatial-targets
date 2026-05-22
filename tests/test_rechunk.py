"""Tests for the rechunk backfill CLI (issue #165 ST5)."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.conftest import make_minimal_project, write_year_nc


def _write_realistic_aggregated(path: Path, id_col: str = "nhm_id") -> None:
    """Aggregated-style NC: float64 field (+NaN), int8 diagnostic, crs container,
    lat/lon coords, and a re-encoded ``time`` axis — the real on-disk shape."""
    n_time, n_hru = 12, 300
    times = pd.date_range("2000-01-01", periods=n_time, freq="MS")
    rng = np.random.default_rng(1)
    ro = rng.random((n_time, n_hru))  # float64
    ro[0, 0] = np.nan
    ds = xr.Dataset(
        {
            "ro": (("time", id_col), ro),
            "n_sources": (("time", id_col), np.ones((n_time, n_hru), dtype="int8")),
            "crs": ((), np.int32(0)),
        },
        coords={
            "time": times,
            id_col: np.arange(1, n_hru + 1),
            "lat": ((id_col,), np.linspace(25.0, 49.0, n_hru)),
            "lon": ((id_col,), np.linspace(-124.0, -67.0, n_hru)),
        },
    )
    ds["crs"].attrs["grid_mapping_name"] = "latitude_longitude"
    ds["ro"].attrs["grid_mapping"] = "crs"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


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
    """daymet/ssebop outputs are intentionally left as-is (#165 ST3)."""
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


# --- safety paths surfaced by PR #170 review ----------------------------


def test_is_rechunked_detects_mixed_chunk_state(tmp_path: Path):
    """A file with one chunked + one contiguous field var is NOT 'rechunked'."""
    from nhf_spatial_targets.rechunk import is_rechunked

    p = tmp_path / "mixed.nc"
    n_time, n_hru = 12, 50
    ds = xr.Dataset(
        {
            "a": (("time", "nhm_id"), np.ones((n_time, n_hru))),
            "b": (("time", "nhm_id"), np.ones((n_time, n_hru))),
        },
        coords={
            "time": pd.date_range("2000-01-01", periods=n_time, freq="MS"),
            "nhm_id": np.arange(n_hru),
        },
    )
    # Chunk+compress only 'a'; leave 'b' contiguous → mixed state.
    ds.to_netcdf(
        p, encoding={"a": {"zlib": True, "complevel": 1, "chunksizes": (n_time, n_hru)}}
    )
    assert not is_rechunked(p)  # would have been True under first-var-only check


def test_rechunk_file_aborts_on_value_mismatch_leaving_original(
    tmp_path: Path, monkeypatch
):
    from nhf_spatial_targets import rechunk as R

    workdir = make_minimal_project(tmp_path)
    f = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    write_year_nc(f, 2000, "ro")
    before = f.read_bytes()
    tmp = f.with_suffix(f.suffix + ".rechunk.tmp")

    monkeypatch.setattr(R, "_arrays_equal", lambda a, b: False)
    with pytest.raises(RuntimeError, match="aborting"):
        R.rechunk_file(f, "aggregated", "nhm_id")

    assert f.read_bytes() == before  # original untouched
    assert not tmp.exists()  # tmp cleaned


def test_rechunk_file_cleans_tmp_on_write_failure(tmp_path: Path, monkeypatch):
    from nhf_spatial_targets import rechunk as R

    workdir = make_minimal_project(tmp_path)
    f = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    write_year_nc(f, 2000, "ro")
    before = f.read_bytes()
    tmp = f.with_suffix(f.suffix + ".rechunk.tmp")

    def _boom(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _boom)
    with pytest.raises(OSError, match="disk full"):
        R.rechunk_file(f, "aggregated", "nhm_id")

    assert f.read_bytes() == before
    assert not tmp.exists()


def test_rechunk_preserves_coords_crs_and_int8(tmp_path: Path):
    """Coords (lat/lon/time), the crs container, and int8 diagnostics survive."""
    from nhf_spatial_targets.rechunk import rechunk_project
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    f = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    _write_realistic_aggregated(f)
    before = xr.open_dataset(f).load()

    rechunk_project(load(workdir), layer="aggregated")

    after = xr.open_dataset(f).load()
    assert np.array_equal(before["ro"].values, after["ro"].values, equal_nan=True)
    np.testing.assert_array_equal(before["n_sources"].values, after["n_sources"].values)
    assert after["n_sources"].dtype == np.int8
    np.testing.assert_array_equal(before["lat"].values, after["lat"].values)
    np.testing.assert_array_equal(before["lon"].values, after["lon"].values)
    np.testing.assert_array_equal(before["time"].values, after["time"].values)
    assert after["crs"].attrs.get("grid_mapping_name") == "latitude_longitude"
    with netCDF4.Dataset(f) as nc:
        assert nc.variables["crs"].filters()["zlib"] is False  # container untouched
        assert nc.variables["ro"].filters()["zlib"] is True  # field compressed


def test_rechunk_project_isolates_failure(tmp_path: Path, monkeypatch):
    """One file's failure is recorded as 'failed'; others still rechunk."""
    from nhf_spatial_targets import rechunk as R
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    a = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    b = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2001_agg.nc"
    write_year_nc(a, 2000, "ro")
    write_year_nc(b, 2001, "ro")

    real = R.rechunk_file

    def _flaky(path, layer, id_col, *, dry_run=False):
        if path.name.endswith("2000_agg.nc"):
            raise RuntimeError("boom")
        return real(path, layer, id_col, dry_run=dry_run)

    monkeypatch.setattr(R, "rechunk_file", _flaky)
    results = R.rechunk_project(load(workdir), layer="aggregated")
    statuses = {r["path"].name: r["status"] for r in results}
    assert statuses["era5_land_2000_agg.nc"] == "failed"
    assert statuses["era5_land_2001_agg.nc"] == "rechunked"


def test_rechunk_skips_stray_non_fabric_nc(tmp_path: Path):
    """A targets-dir NC without the HRU dim is skipped, not crashed on."""
    from nhf_spatial_targets.rechunk import rechunk_project
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    stray = workdir / "targets" / "weird.nc"
    stray.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"x": (("a", "b"), np.ones((2, 3)))},
        coords={"a": [0, 1], "b": [0, 1, 2]},
    ).to_netcdf(stray)

    results = rechunk_project(load(workdir), layer="target")
    assert any(
        r["status"] == "skipped-no-hru" and r["path"].name == "weird.nc"
        for r in results
    )


def test_rechunk_layer_none_does_both_layers(tmp_path: Path):
    from nhf_spatial_targets.rechunk import rechunk_project
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    a = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    t = workdir / "targets" / "runoff_targets.nc"
    write_year_nc(a, 2000, "ro")
    _write_contiguous_target(t)

    rechunk_project(load(workdir), layer=None)

    assert _data_var_chunked(a)
    assert _data_var_chunked(t)


def test_rechunk_invalid_layer_raises(tmp_path: Path):
    from nhf_spatial_targets.rechunk import rechunk_project
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    with pytest.raises(ValueError, match="layer"):
        rechunk_project(load(workdir), layer="bogus")


# --- CLI error/summary paths --------------------------------------------


def test_cli_rechunk_bad_layer_exits_2(tmp_path: Path):
    from nhf_spatial_targets import cli

    workdir = make_minimal_project(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.rechunk(workdir, layer="bogus")
    assert exc.value.code == 2


def test_cli_rechunk_bad_source_exits_2(tmp_path: Path):
    from nhf_spatial_targets import cli

    workdir = make_minimal_project(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.rechunk(workdir, source="does_not_exist", dry_run=True)
    assert exc.value.code == 2


def test_cli_rechunk_real_run_reports_and_chunks(tmp_path: Path, capsys):
    from nhf_spatial_targets import cli

    workdir = make_minimal_project(tmp_path)
    f = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    write_year_nc(f, 2000, "ro")

    cli.rechunk(workdir, layer="aggregated")

    out = capsys.readouterr().out
    assert "Rechunked" in out
    assert _data_var_chunked(f)
