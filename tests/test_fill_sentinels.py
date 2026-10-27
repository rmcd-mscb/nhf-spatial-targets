"""Tests for the default-fill sentinel scan / backfill (issue #361)."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from netCDF4 import default_fillvals

from tests.conftest import make_minimal_project

SENTINEL_F8 = default_fillvals["f8"]  # NC_FILL_DOUBLE == 9.969209968386869e+36
SENTINEL_F4 = default_fillvals["f4"]


def _write_agg(
    path: Path,
    *,
    id_col: str = "nhm_id",
    sentinel_cells: int = 0,
    declare_sentinel_as_fill: bool = False,
    dtype: str = "float64",
) -> None:
    """Aggregated-style NC (float field, crs container, lat/lon coords).

    ``sentinel_cells`` > 0 plants the netCDF default fill as *data* in the
    first N HRU columns — the pre-#205 gdptools state. By default the file
    declares ``_FillValue=NaN`` (the io_nc policy), so the sentinel survives
    ``decode_cf``; ``declare_sentinel_as_fill=True`` instead declares the
    sentinel itself as ``_FillValue`` (the daymet/ssebop state), so xarray
    masks it on read and the file is clean by construction.
    """
    n_time, n_hru = 12, 50
    times = pd.date_range("2000-01-01", periods=n_time, freq="MS")
    rng = np.random.default_rng(1)
    ro = rng.random((n_time, n_hru)).astype(dtype)
    ro[0, -1] = np.nan  # a genuine NaN must survive untouched
    sentinel = default_fillvals["f8" if dtype == "float64" else "f4"]
    if sentinel_cells:
        ro[:, :sentinel_cells] = sentinel
    ds = xr.Dataset(
        {
            "ro": (("time", id_col), ro),
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
    fill = sentinel if declare_sentinel_as_fill else np.nan
    ds.to_netcdf(path, encoding={"ro": {"_FillValue": fill}})


def _agg_path(workdir: Path, key: str = "era5_land", year: int = 2000) -> Path:
    return workdir / "data" / "aggregated" / key / f"{key}_{year}_agg.nc"


# --- scan_file -------------------------------------------------------------


def test_scan_file_counts_sentinel_cells_per_var(tmp_path: Path):
    from nhf_spatial_targets.fill_sentinels import scan_file

    f = tmp_path / "x.nc"
    _write_agg(f, sentinel_cells=3)

    assert scan_file(f) == {"ro": 3 * 12}


def test_scan_file_is_empty_for_a_clean_file(tmp_path: Path):
    from nhf_spatial_targets.fill_sentinels import scan_file

    f = tmp_path / "x.nc"
    _write_agg(f)

    assert scan_file(f) == {}


def test_scan_file_treats_declared_fill_as_clean(tmp_path: Path):
    """A file whose _FillValue IS the sentinel decodes to NaN: nothing to fix."""
    from nhf_spatial_targets.fill_sentinels import scan_file

    f = tmp_path / "x.nc"
    _write_agg(f, sentinel_cells=3, declare_sentinel_as_fill=True)

    assert scan_file(f) == {}


def test_scan_file_handles_float32_sentinel(tmp_path: Path):
    from nhf_spatial_targets.fill_sentinels import scan_file

    f = tmp_path / "x.nc"
    _write_agg(f, sentinel_cells=2, dtype="float32")

    assert scan_file(f) == {"ro": 2 * 12}


# --- mask_file -------------------------------------------------------------


def test_mask_file_replaces_sentinel_with_nan_and_preserves_the_rest(
    tmp_path: Path,
):
    from nhf_spatial_targets.fill_sentinels import mask_file, scan_file

    f = tmp_path / "x.nc"
    _write_agg(f, sentinel_cells=3)
    before = xr.open_dataset(f).load()

    rec = mask_file(f, "aggregated", "nhm_id")

    assert rec["status"] == "masked"
    assert rec["counts"] == {"ro": 36}
    after = xr.open_dataset(f).load()
    assert np.isnan(after["ro"].values[:, :3]).all()
    np.testing.assert_array_equal(before["ro"].values[:, 3:], after["ro"].values[:, 3:])
    assert np.isnan(after["ro"].values[0, -1])  # the genuine NaN is still NaN
    np.testing.assert_array_equal(before["time"].values, after["time"].values)
    np.testing.assert_array_equal(before["lat"].values, after["lat"].values)
    assert after["crs"].attrs["grid_mapping_name"] == "latitude_longitude"
    assert scan_file(f) == {}


def test_mask_file_lands_in_canonical_encoding(tmp_path: Path):
    """The rewrite goes through io_nc.build_encoding, so the file comes out
    chunked + zlib with _FillValue=NaN — the same layout a fresh agg write gets."""
    from nhf_spatial_targets.fill_sentinels import mask_file

    f = tmp_path / "x.nc"
    _write_agg(f, sentinel_cells=1)
    mask_file(f, "aggregated", "nhm_id")

    with netCDF4.Dataset(f) as nc:
        v = nc.variables["ro"]
        assert v.chunking() != "contiguous"
        assert bool(v.filters().get("zlib"))
        assert np.isnan(v.getncattr("_FillValue"))


def test_mask_file_is_idempotent(tmp_path: Path):
    from nhf_spatial_targets.fill_sentinels import mask_file

    f = tmp_path / "x.nc"
    _write_agg(f, sentinel_cells=3)
    mask_file(f, "aggregated", "nhm_id")
    mtime = f.stat().st_mtime_ns

    rec = mask_file(f, "aggregated", "nhm_id")

    assert rec["status"] == "clean"
    assert f.stat().st_mtime_ns == mtime  # not rewritten


def test_mask_file_dry_run_reports_without_writing(tmp_path: Path):
    from nhf_spatial_targets.fill_sentinels import mask_file, scan_file

    f = tmp_path / "x.nc"
    _write_agg(f, sentinel_cells=3)
    mtime = f.stat().st_mtime_ns

    rec = mask_file(f, "aggregated", "nhm_id", dry_run=True)

    assert rec["status"] == "would-mask"
    assert rec["counts"] == {"ro": 36}
    assert f.stat().st_mtime_ns == mtime
    assert scan_file(f) == {"ro": 36}
    assert not list(tmp_path.glob("*.tmp"))


def test_mask_file_skips_nc_without_hru_dim(tmp_path: Path):
    from nhf_spatial_targets.fill_sentinels import mask_file

    f = tmp_path / "stray.nc"
    xr.Dataset({"x": (("t",), np.full(4, SENTINEL_F8))}).to_netcdf(
        f, encoding={"x": {"_FillValue": np.nan}}
    )

    rec = mask_file(f, "aggregated", "nhm_id")

    assert rec["status"] == "skipped-no-hru"


def test_mask_file_leaves_no_tempfile_on_failure(tmp_path: Path, monkeypatch):
    from nhf_spatial_targets import fill_sentinels

    f = tmp_path / "x.nc"
    _write_agg(f, sentinel_cells=3)
    before = f.read_bytes()

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        fill_sentinels.mask_file(f, "aggregated", "nhm_id")

    assert f.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


# --- project-level ---------------------------------------------------------


def test_scan_project_walks_both_layers_and_does_not_skip_daymet(tmp_path: Path):
    """Unlike rechunk, the sentinel scan covers daymet/ssebop too: a sentinel is
    wrong wherever it is, and those files are clean by construction anyway."""
    from nhf_spatial_targets.fill_sentinels import scan_project
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    _write_agg(_agg_path(workdir, "era5_land"), sentinel_cells=2)
    _write_agg(_agg_path(workdir, "daymet"), sentinel_cells=2)
    _write_agg(workdir / "targets" / "aet_targets.nc", sentinel_cells=1)

    recs = scan_project(load(workdir))

    by_name = {r["path"].name: r for r in recs}
    assert set(by_name) == {
        "era5_land_2000_agg.nc",
        "daymet_2000_agg.nc",
        "aet_targets.nc",
    }
    assert by_name["era5_land_2000_agg.nc"]["counts"] == {"ro": 24}
    assert by_name["daymet_2000_agg.nc"]["counts"] == {"ro": 24}
    assert by_name["aet_targets.nc"]["layer"] == "target"
    assert by_name["aet_targets.nc"]["counts"] == {"ro": 12}


def test_scan_project_records_an_unreadable_file_instead_of_aborting(
    tmp_path: Path,
):
    from nhf_spatial_targets.fill_sentinels import scan_project
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    _write_agg(_agg_path(workdir, "era5_land", 2000), sentinel_cells=2)
    bad = _agg_path(workdir, "era5_land", 2001)
    bad.write_bytes(b"not a netcdf")

    recs = scan_project(load(workdir))

    by_name = {r["path"].name: r for r in recs}
    assert by_name["era5_land_2000_agg.nc"]["counts"] == {"ro": 24}
    assert by_name["era5_land_2001_agg.nc"]["error"]
    assert by_name["era5_land_2001_agg.nc"]["counts"] == {}


def test_mask_project_rewrites_only_affected_files_and_isolates_failures(
    tmp_path: Path,
):
    from nhf_spatial_targets.fill_sentinels import mask_project, scan_file
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    dirty = _agg_path(workdir, "era5_land", 2000)
    clean = _agg_path(workdir, "era5_land", 2001)
    bad = _agg_path(workdir, "era5_land", 2002)
    _write_agg(dirty, sentinel_cells=2)
    _write_agg(clean)
    bad.write_bytes(b"not a netcdf")
    clean_mtime = clean.stat().st_mtime_ns

    recs = mask_project(load(workdir))

    status = {r["path"].name: r["status"] for r in recs}
    assert status == {
        "era5_land_2000_agg.nc": "masked",
        "era5_land_2001_agg.nc": "clean",
        "era5_land_2002_agg.nc": "failed",
    }
    assert scan_file(dirty) == {}
    assert clean.stat().st_mtime_ns == clean_mtime
    assert bad.read_bytes() == b"not a netcdf"


def test_mask_project_source_and_layer_filters(tmp_path: Path):
    from nhf_spatial_targets.fill_sentinels import mask_project, scan_file
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    a = _agg_path(workdir, "era5_land")
    b = _agg_path(workdir, "nldas_noah")
    t = workdir / "targets" / "aet_targets.nc"
    for p in (a, b, t):
        _write_agg(p, sentinel_cells=1)

    mask_project(load(workdir), layer="aggregated", source="era5_land")

    assert scan_file(a) == {}
    assert scan_file(b) == {"ro": 12}
    assert scan_file(t) == {"ro": 12}


def test_mask_project_rejects_unknown_layer(tmp_path: Path):
    from nhf_spatial_targets.fill_sentinels import mask_project
    from nhf_spatial_targets.workspace import load

    workdir = make_minimal_project(tmp_path)
    with pytest.raises(ValueError, match="invalid layer"):
        mask_project(load(workdir), layer="consolidated")


# --- CLI -------------------------------------------------------------------


def test_cli_check_fills_exits_nonzero_when_sentinels_found(tmp_path: Path, capsys):
    from nhf_spatial_targets.cli.maintenance import check_fills_cmd

    workdir = make_minimal_project(tmp_path)
    _write_agg(_agg_path(workdir), sentinel_cells=2)
    before = _agg_path(workdir).read_bytes()

    with pytest.raises(SystemExit) as exc:
        check_fills_cmd(workdir)

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "era5_land_2000_agg.nc" in out and "mask-fills" in out
    assert _agg_path(workdir).read_bytes() == before  # report-only


def test_cli_check_fills_exits_zero_when_clean(tmp_path: Path, capsys):
    from nhf_spatial_targets.cli.maintenance import check_fills_cmd

    workdir = make_minimal_project(tmp_path)
    _write_agg(_agg_path(workdir))

    check_fills_cmd(workdir)  # no SystemExit

    assert "No netCDF default-fill sentinels" in capsys.readouterr().out


def test_cli_mask_fills_dry_run_and_apply(tmp_path: Path, capsys):
    from nhf_spatial_targets.cli.maintenance import mask_fills_cmd
    from nhf_spatial_targets.fill_sentinels import scan_file

    workdir = make_minimal_project(tmp_path)
    f = _agg_path(workdir)
    _write_agg(f, sentinel_cells=2)

    mask_fills_cmd(workdir, dry_run=True)
    assert "dry-run" in capsys.readouterr().out
    assert scan_file(f) == {"ro": 24}

    mask_fills_cmd(workdir)
    assert "Masked 1 file(s)" in capsys.readouterr().out
    assert scan_file(f) == {}


def test_cli_mask_fills_exits_nonzero_on_a_failed_file(tmp_path: Path, capsys):
    from nhf_spatial_targets.cli.maintenance import mask_fills_cmd

    workdir = make_minimal_project(tmp_path)
    _agg_path(workdir).parent.mkdir(parents=True)
    _agg_path(workdir).write_bytes(b"not a netcdf")

    with pytest.raises(SystemExit) as exc:
        mask_fills_cmd(workdir)

    assert exc.value.code == 1
    assert "failed" in capsys.readouterr().err
