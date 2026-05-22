"""CF-1.6 regression tests for the aggregation + target NetCDF writers.

Covers the defects a ``cfchecks`` shakedown found at the shared write-sites
(issues #178 / #182). Each test reproduces the pre-fix structure and asserts
the writer now emits CF-clean metadata:

- **A** scalar (0-dim) ``crs`` grid-mapping anchor — minted on targets,
  collapsed back from the per-batch-merge broadcast on aggregated NCs.
- **B** ``flag_values`` dtype matches its int8 parent variable (CF §3.5).
- **C** ``time_bnds`` carries no ``_FillValue`` (CF §7.1).
- **E** ``time_bnds`` reconstructed for monthly/annual aggregates; dangling
  ``time:bounds`` stripped for instantaneous sources.
- **F** the HRU index coordinate is labeled and carries no units.
"""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

ID_COL = "nat_hru_id"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _make_target_ds() -> xr.Dataset:
    """A minimal multi-source-minmax target Dataset as the builders assemble it.

    Bound vars carry ``grid_mapping="crs"`` inherited from the aggregated
    sources (the dangling reference, defect A) but no ``crs`` container.
    """
    from nhf_spatial_targets.targets._common import build_n_sources_attrs

    ids = np.array([10, 20, 30], dtype="int64")
    time = pd.date_range("2000-01-01", periods=3, freq="MS")
    tb = xr.DataArray(
        list(zip(time.values, (time + pd.offsets.MonthBegin(1)).values)),
        dims=("time", "nv"),
        coords={"time": time.values},
        name="time_bnds",
    )

    def bound(name: str) -> xr.DataArray:
        da = xr.DataArray(
            np.ones((3, 3), dtype="float32"),
            dims=("time", ID_COL),
            coords={"time": time.values, ID_COL: ids},
            attrs={"units": "mm", "grid_mapping": "crs"},
        )
        da.name = name
        return da

    n_sources = xr.DataArray(
        np.array([[1, 2, 3]] * 3, dtype="int8"),
        dims=("time", ID_COL),
        coords={"time": time.values, ID_COL: ids},
    )
    n_sources.attrs.update(build_n_sources_attrs(3))
    n_sources.name = "n_sources"
    clat = xr.DataArray(
        np.array([40.0, 41.0, 42.0]),
        dims=(ID_COL,),
        coords={ID_COL: ids},
        attrs={"units": "degrees_north", "standard_name": "latitude"},
    )
    clon = xr.DataArray(
        np.array([-120.0, -121.0, -122.0]),
        dims=(ID_COL,),
        coords={ID_COL: ids},
        attrs={"units": "degrees_east", "standard_name": "longitude"},
    )
    ds = xr.Dataset(
        {
            "lower_bound": bound("lower_bound"),
            "upper_bound": bound("upper_bound"),
            "n_sources": n_sources,
        },
        coords={
            "time": time,
            ID_COL: ids,
            "time_bnds": tb,
            "centroid_lat": clat,
            "centroid_lon": clon,
        },
    )
    ds["time"].attrs["bounds"] = "time_bnds"
    return ds


def _make_aggregated_ds(*, n_time: int, broadcast_crs: bool = True) -> xr.Dataset:
    """A minimal aggregated Dataset reproducing the gdptools + merge artifacts.

    ``time`` carries an inherited ``bounds="time_bnds"`` with no ``time_bnds``
    variable (defect E), and ``crs`` is broadcast along the HRU dim by the
    per-batch ``xr.concat`` merge (defect A).
    """
    from pyproj import CRS

    ids = np.array([10, 20, 30], dtype="int64")
    time = pd.date_range("2000-01-01", periods=n_time, freq="MS")
    data = np.ones((n_time, 3), dtype="float64")
    crs_attrs = dict(CRS.from_epsg(4326).to_cf())
    crs = (
        xr.DataArray(
            np.zeros(3, dtype="int32"),
            dims=(ID_COL,),
            coords={ID_COL: ids},
            attrs=crs_attrs,
        )
        if broadcast_crs
        else xr.DataArray(np.int32(0), attrs=crs_attrs)
    )
    ds = xr.Dataset(
        {
            "runoff_total": xr.DataArray(
                data,
                dims=("time", ID_COL),
                coords={"time": time.values, ID_COL: ids},
                attrs={"units": "mm", "grid_mapping": "crs"},
            ),
            "crs": crs,
        },
        coords={"time": time, ID_COL: ids},
    )
    ds["time"].attrs.update({"standard_name": "time", "bounds": "time_bnds"})
    ds[ID_COL].attrs["feature_id"] = ID_COL
    return ds


# --------------------------------------------------------------------------
# io_nc — C
# --------------------------------------------------------------------------


def test_build_encoding_pins_time_bnds_fill_value_none():
    from nhf_spatial_targets.io_nc import build_encoding

    ids = np.array([1, 2, 3], dtype="int64")
    time = pd.date_range("2000-01-01", periods=2, freq="MS")
    tb = xr.DataArray(
        np.zeros((2, 2)), dims=("time", "nv"), coords={"time": time.values}
    )
    ds = xr.Dataset(
        {"ro": (("time", ID_COL), np.ones((2, 3), dtype="float32"))},
        coords={"time": time, ID_COL: ids, "time_bnds": tb},
    )
    enc = build_encoding(ds, layer="aggregated", hru_dim=ID_COL)
    assert "time_bnds" in enc
    assert enc["time_bnds"]["_FillValue"] is None


# --------------------------------------------------------------------------
# target writer — A, B, C, F
# --------------------------------------------------------------------------


def test_target_writer_anchors_grid_mapping_with_scalar_crs(tmp_path: Path):
    from nhf_spatial_targets.targets._common import write_target_nc

    out = tmp_path / "recharge_targets.nc"
    write_target_nc(_make_target_ds(), out, title="t", sort_dim=ID_COL)
    ds = xr.open_dataset(out, decode_times=False)
    assert "crs" in ds.variables
    assert ds["crs"].ndim == 0  # CF §5.6: grid mapping var is 0-dim
    assert ds["crs"].attrs.get("grid_mapping_name") == "latitude_longitude"
    for v in ("lower_bound", "upper_bound", "n_sources"):
        assert ds[v].attrs.get("grid_mapping") == "crs"


def test_target_n_sources_flag_values_are_int8(tmp_path: Path):
    from nhf_spatial_targets.targets._common import (
        build_n_sources_attrs,
        write_target_nc,
    )

    assert build_n_sources_attrs(3)["flag_values"].dtype == np.dtype("int8")
    out = tmp_path / "recharge_targets.nc"
    write_target_nc(_make_target_ds(), out, title="t", sort_dim=ID_COL)
    with nc.Dataset(out) as ncf:
        # CF §3.5: flag_values dtype must match its parent variable (int8).
        assert ncf["n_sources"].flag_values.dtype == np.dtype("int8")


def test_target_time_bnds_has_no_fill_value(tmp_path: Path):
    from nhf_spatial_targets.targets._common import write_target_nc

    out = tmp_path / "recharge_targets.nc"
    write_target_nc(_make_target_ds(), out, title="t", sort_dim=ID_COL)
    with nc.Dataset(out) as ncf:
        assert "_FillValue" not in ncf["time_bnds"].ncattrs()


def test_target_hru_index_labeled_without_units(tmp_path: Path):
    from nhf_spatial_targets.targets._common import write_target_nc

    out = tmp_path / "recharge_targets.nc"
    write_target_nc(_make_target_ds(), out, title="t", sort_dim=ID_COL)
    ds = xr.open_dataset(out, decode_times=False)
    assert ds[ID_COL].attrs.get("long_name") == "HRU Index"
    assert "units" not in ds[ID_COL].attrs


# --------------------------------------------------------------------------
# aggregated writer — A, E, F
# --------------------------------------------------------------------------


def test_aggregated_writer_collapses_broadcast_crs(tmp_path: Path):
    from nhf_spatial_targets.aggregate._driver import _atomic_write_netcdf

    out = tmp_path / "agg.nc"
    _atomic_write_netcdf(
        _make_aggregated_ds(n_time=12),
        out,
        id_col=ID_COL,
        time_bounds_cadence="monthly",
    )
    ds = xr.open_dataset(out, decode_times=False)
    assert ds["crs"].ndim == 0  # collapsed from the (id_col,) broadcast
    # gdptools' correct attrs are preserved through the collapse.
    assert ds["crs"].attrs.get("grid_mapping_name") == "latitude_longitude"


def test_aggregated_writer_reconstructs_monthly_time_bnds(tmp_path: Path):
    from nhf_spatial_targets.aggregate._driver import _atomic_write_netcdf

    out = tmp_path / "agg.nc"
    _atomic_write_netcdf(
        _make_aggregated_ds(n_time=12),
        out,
        id_col=ID_COL,
        time_bounds_cadence="monthly",
    )
    ds = xr.open_dataset(out)
    assert "time_bnds" in ds.variables
    assert ds["time"].attrs.get("bounds") == "time_bnds"
    # Jan 2000 bound brackets the calendar month.
    lo, hi = ds["time_bnds"].values[0]
    assert pd.Timestamp(lo) == pd.Timestamp("2000-01-01")
    assert pd.Timestamp(hi) == pd.Timestamp("2000-02-01")


def test_aggregated_writer_reconstructs_annual_time_bnds(tmp_path: Path):
    from nhf_spatial_targets.aggregate._driver import _atomic_write_netcdf

    # Annual source: a single timestamp per file (spacing can't infer freq).
    ds_in = _make_aggregated_ds(n_time=1)
    ds_in["time"].attrs.pop("bounds")  # reitz-style: no inherited bounds attr
    out = tmp_path / "agg.nc"
    _atomic_write_netcdf(ds_in, out, id_col=ID_COL, time_bounds_cadence="annual")
    ds = xr.open_dataset(out)
    assert "time_bnds" in ds.variables
    lo, hi = ds["time_bnds"].values[0]
    assert pd.Timestamp(lo) == pd.Timestamp("2000-01-01")
    assert pd.Timestamp(hi) == pd.Timestamp("2001-01-01")


def test_aggregated_writer_strips_dangling_bounds_for_point_source(tmp_path: Path):
    from nhf_spatial_targets.aggregate._driver import _atomic_write_netcdf

    out = tmp_path / "agg.nc"
    # Daily/instantaneous: no cadence -> no averaging interval.
    _atomic_write_netcdf(
        _make_aggregated_ds(n_time=3),
        out,
        id_col=ID_COL,
        time_bounds_cadence="daily",
    )
    ds = xr.open_dataset(out, decode_times=False)
    assert "time_bnds" not in ds.variables
    assert "bounds" not in ds["time"].attrs  # dangling reference stripped


def test_unchunked_path_pins_time_bnds_encoding(tmp_path: Path, recwarn):
    """ssebop/daymet (id_col=None): reconstructed time_bnds shares time's units.

    Without pinned time encoding on this path, xarray derives time and
    time_bnds units independently and warns (CF §7.1). Assert the write is
    clean and time_bnds carries no _FillValue.
    """
    from nhf_spatial_targets.aggregate._driver import _atomic_write_netcdf

    out = tmp_path / "ssebop_agg.nc"
    _atomic_write_netcdf(
        _make_aggregated_ds(n_time=12), out, time_bounds_cadence="monthly"
    )
    msgs = [str(w.message) for w in recwarn]
    assert not any("units encodings for time" in m for m in msgs), msgs
    with nc.Dataset(out) as ncf:
        assert "time_bnds" in ncf.variables
        assert "_FillValue" not in ncf["time_bnds"].ncattrs()


def test_aggregated_writer_labels_hru_index(tmp_path: Path):
    from nhf_spatial_targets.aggregate._driver import _atomic_write_netcdf

    out = tmp_path / "agg.nc"
    _atomic_write_netcdf(
        _make_aggregated_ds(n_time=12),
        out,
        id_col=ID_COL,
        time_bounds_cadence="monthly",
    )
    ds = xr.open_dataset(out, decode_times=False)
    assert ds[ID_COL].attrs.get("long_name") == "HRU Index"
    assert "units" not in ds[ID_COL].attrs
