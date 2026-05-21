"""Tests for the shared NetCDF encoding/atomic-write helper (issue #165 ST1)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _make_aggregated_ds(n_time: int = 365, n_hru: int = 50_000) -> xr.Dataset:
    """A minimal (time, hru) aggregated-style Dataset on id_col ``nhm_id``."""
    times = pd.date_range("2000-01-01", periods=n_time, freq="D")
    hru = np.arange(1, n_hru + 1, dtype="int64")
    data = np.ones((n_time, n_hru), dtype="float32")
    ds = xr.Dataset(
        {"ro": (("time", "nhm_id"), data)},
        coords={"time": times, "nhm_id": hru},
    )
    return ds


# --- chunk shape --------------------------------------------------------


def test_aggregated_float_var_gets_chunks_and_compression():
    from nhf_spatial_targets.io_nc import build_encoding

    ds = _make_aggregated_ds(n_time=365, n_hru=50_000)
    enc = build_encoding(
        ds, layer="aggregated", hru_dim="nhm_id", timesteps_per_file=365
    )

    expected_hru_chunk = math.ceil(1_048_576 / (365 * 4))  # 719
    assert enc["ro"]["chunksizes"] == (365, expected_hru_chunk)
    assert enc["ro"]["zlib"] is True
    assert enc["ro"]["complevel"] == 4
    assert enc["ro"]["dtype"] == "float32"
    # floats do not get the byte-shuffle filter
    assert enc["ro"].get("shuffle", False) is False
    assert np.isnan(enc["ro"]["_FillValue"])


def test_target_layer_uses_same_formula():
    from nhf_spatial_targets.io_nc import build_encoding

    ds = _make_aggregated_ds(n_time=240, n_hru=50_000)
    enc = build_encoding(ds, layer="target", hru_dim="nhm_id", timesteps_per_file=240)
    expected_hru_chunk = math.ceil(1_048_576 / (240 * 4))  # 1093
    assert enc["ro"]["chunksizes"] == (240, expected_hru_chunk)


def test_hru_chunk_capped_at_n_hrus():
    from nhf_spatial_targets.io_nc import build_encoding

    # Few timesteps + small fabric → formula wants more HRUs than exist.
    ds = _make_aggregated_ds(n_time=12, n_hru=5_000)
    enc = build_encoding(
        ds, layer="aggregated", hru_dim="nhm_id", timesteps_per_file=12
    )
    # ceil(1MiB/(12*4)) == 21846 > 5000 → capped at the fabric size.
    assert enc["ro"]["chunksizes"] == (12, 5_000)


def test_timesteps_per_file_defaults_to_full_time_axis():
    from nhf_spatial_targets.io_nc import build_encoding

    ds = _make_aggregated_ds(n_time=365, n_hru=50_000)
    enc = build_encoding(ds, layer="aggregated", hru_dim="nhm_id")
    assert enc["ro"]["chunksizes"][0] == 365


# --- time encoding ------------------------------------------------------


@pytest.mark.parametrize("layer", ["aggregated", "target"])
def test_time_encoding_pinned_for_any_layer(layer: str):
    from nhf_spatial_targets.io_nc import build_encoding

    ds = _make_aggregated_ds()
    enc = build_encoding(ds, layer=layer, hru_dim="nhm_id", timesteps_per_file=365)
    assert enc["time"]["dtype"] == "float64"
    assert enc["time"]["units"] == "days since 1970-01-01 00:00:00"
    assert enc["time"]["calendar"] == "proleptic_gregorian"


def test_time_bnds_encoding_pinned_when_present():
    from nhf_spatial_targets.io_nc import build_encoding

    ds = _make_aggregated_ds(n_time=12)
    # attach a CF time_bnds variable
    bnds = np.stack([ds.time.values, ds.time.values + np.timedelta64(1, "D")], axis=1)
    ds["time_bnds"] = (("time", "nv"), bnds)
    enc = build_encoding(ds, layer="target", hru_dim="nhm_id", timesteps_per_file=12)
    assert enc["time_bnds"]["calendar"] == "proleptic_gregorian"
    assert enc["time_bnds"]["dtype"] == "float64"


# --- dtype policy -------------------------------------------------------


def test_integer_dtype_gets_shuffle_and_no_nan_fill():
    from nhf_spatial_targets.io_nc import build_encoding

    ds = _make_aggregated_ds()
    ds["n_sources"] = (("time", "nhm_id"), np.ones(ds["ro"].shape, dtype="int8"))
    enc = build_encoding(ds, layer="target", hru_dim="nhm_id", timesteps_per_file=365)
    assert enc["n_sources"]["dtype"] == "int8"
    assert enc["n_sources"]["shuffle"] is True
    # int8 diagnostic carries no fill value
    assert enc["n_sources"]["_FillValue"] is None


def test_per_var_dtype_override():
    from nhf_spatial_targets.io_nc import build_encoding

    ds = _make_aggregated_ds()
    # ``ro`` is float32 in memory; force it to encode as int16.
    enc = build_encoding(
        ds,
        layer="aggregated",
        hru_dim="nhm_id",
        timesteps_per_file=365,
        var_dtype={"ro": "int16"},
    )
    assert enc["ro"]["dtype"] == "int16"
    assert enc["ro"]["shuffle"] is True
    assert enc["ro"]["_FillValue"] == -9999
    # chunk byte budget now uses the 2-byte dtype
    expected_hru_chunk = math.ceil(1_048_576 / (365 * 2))
    assert enc["ro"]["chunksizes"] == (365, expected_hru_chunk)


def test_compression_level_override():
    from nhf_spatial_targets.io_nc import build_encoding

    ds = _make_aggregated_ds()
    enc = build_encoding(
        ds,
        layer="aggregated",
        hru_dim="nhm_id",
        timesteps_per_file=365,
        compression_level=1,
    )
    assert enc["ro"]["complevel"] == 1


# --- consolidated seam (#158 owns the body) -----------------------------


def test_consolidated_layer_is_a_seam_for_issue_158():
    from nhf_spatial_targets.io_nc import build_encoding

    ds = _make_aggregated_ds()
    with pytest.raises(NotImplementedError, match="158"):
        build_encoding(ds, layer="consolidated", spatial_dims=("y", "x"))


def test_unknown_layer_raises():
    from nhf_spatial_targets.io_nc import build_encoding

    ds = _make_aggregated_ds()
    with pytest.raises(ValueError, match="layer"):
        build_encoding(ds, layer="bogus", hru_dim="nhm_id")


def test_none_hru_dim_raises():
    from nhf_spatial_targets.io_nc import build_encoding

    ds = _make_aggregated_ds()
    with pytest.raises(ValueError, match="hru_dim"):
        build_encoding(ds, layer="aggregated", hru_dim=None)


def test_absent_hru_dim_raises_loudly():
    """A drifted/mistyped hru_dim must fail loudly, not silently skip chunking."""
    from nhf_spatial_targets.io_nc import build_encoding

    ds = _make_aggregated_ds()  # dim is ``nhm_id``
    with pytest.raises(ValueError, match="nhm_X"):
        build_encoding(ds, layer="aggregated", hru_dim="nhm_X", timesteps_per_file=365)


def test_static_hru_only_var_fills_byte_budget():
    """A var with hru_dim but no time dim chunks on HRU alone (no phantom time factor)."""
    from nhf_spatial_targets.io_nc import build_encoding

    ds = _make_aggregated_ds(n_time=365, n_hru=5_000)
    # A static per-HRU diagnostic riding alongside the (time, hru) data var.
    ds["hru_area"] = (("nhm_id",), np.ones(5_000, dtype="float32"))
    enc = build_encoding(
        ds, layer="aggregated", hru_dim="nhm_id", timesteps_per_file=365
    )
    # No time factor -> ceil(1 MiB / 4) = 262144, capped at the 5000-HRU fabric.
    assert enc["hru_area"]["chunksizes"] == (5_000,)
    # The (time, hru) var is unaffected by the static var's presence.
    assert enc["ro"]["chunksizes"] == (365, math.ceil(1_048_576 / (365 * 4)))


# --- atomic_to_netcdf ---------------------------------------------------


def test_atomic_to_netcdf_roundtrips_nan_aware(tmp_path: Path):
    from nhf_spatial_targets.io_nc import atomic_to_netcdf, build_encoding

    ds = _make_aggregated_ds(n_time=12, n_hru=1_000)
    ds["ro"].values[0, 0] = np.nan  # ensure NaN survives the roundtrip
    enc = build_encoding(
        ds, layer="aggregated", hru_dim="nhm_id", timesteps_per_file=12
    )
    out = tmp_path / "agg.nc"
    atomic_to_netcdf(ds, out, encoding=enc)

    assert out.exists()
    with xr.open_dataset(out) as got:
        np.testing.assert_array_equal(
            got["ro"].values, ds["ro"].values
        )  # NaN == NaN positionally


def test_atomic_to_netcdf_no_tempfile_left_behind(tmp_path: Path):
    from nhf_spatial_targets.io_nc import atomic_to_netcdf

    ds = _make_aggregated_ds(n_time=4, n_hru=100)
    out = tmp_path / "agg.nc"
    atomic_to_netcdf(ds, out)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != out.name]
    assert leftovers == []


def test_atomic_to_netcdf_cleans_tempfile_on_failure(tmp_path: Path):
    from nhf_spatial_targets.io_nc import atomic_to_netcdf

    ds = _make_aggregated_ds(n_time=4, n_hru=100)
    out = tmp_path / "agg.nc"
    # An impossible encoding dtype makes to_netcdf raise mid-write.
    bad_encoding = {"ro": {"dtype": "not-a-dtype"}}
    with pytest.raises(Exception):
        atomic_to_netcdf(ds, out, encoding=bad_encoding)
    assert not out.exists()
    leftovers = list(tmp_path.iterdir())
    assert leftovers == [], f"tempfile left behind: {leftovers}"


def test_on_disk_chunking_matches_encoding(tmp_path: Path):
    """The chunksizes we request actually land on the HDF5 variable."""
    import netCDF4

    from nhf_spatial_targets.io_nc import atomic_to_netcdf, build_encoding

    ds = _make_aggregated_ds(n_time=365, n_hru=50_000)
    enc = build_encoding(
        ds, layer="aggregated", hru_dim="nhm_id", timesteps_per_file=365
    )
    out = tmp_path / "agg.nc"
    atomic_to_netcdf(ds, out, encoding=enc)

    with netCDF4.Dataset(out) as nc:
        assert tuple(nc.variables["ro"].chunking()) == enc["ro"]["chunksizes"]
