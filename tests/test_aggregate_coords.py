"""Tests for CF-based coordinate detection."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from nhf_spatial_targets.aggregate._coords import detect_coords


def _ds_with_axis_attrs() -> xr.Dataset:
    ds = xr.Dataset(
        {"v": (["t", "y", "x"], np.zeros((1, 2, 2)))},
        coords={
            "t": ("t", [0], {"axis": "T", "standard_name": "time"}),
            "y": ("y", [0.0, 1.0], {"axis": "Y", "standard_name": "latitude"}),
            "x": ("x", [0.0, 1.0], {"axis": "X", "standard_name": "longitude"}),
        },
    )
    return ds


def _ds_with_standard_names_only() -> xr.Dataset:
    ds = xr.Dataset(
        {"v": (["time", "lat", "lon"], np.zeros((1, 2, 2)))},
        coords={
            "time": ("time", [0], {"standard_name": "time"}),
            "lat": ("lat", [0.0, 1.0], {"standard_name": "latitude"}),
            "lon": ("lon", [0.0, 1.0], {"standard_name": "longitude"}),
        },
    )
    return ds


def _ds_projected() -> xr.Dataset:
    ds = xr.Dataset(
        {"v": (["time", "y", "x"], np.zeros((1, 2, 2)))},
        coords={
            "time": ("time", [0], {"standard_name": "time"}),
            "y": ("y", [0.0, 1.0], {"standard_name": "projection_y_coordinate"}),
            "x": ("x", [0.0, 1.0], {"standard_name": "projection_x_coordinate"}),
        },
    )
    return ds


def test_detects_via_axis_attrs():
    x, y, t = detect_coords(_ds_with_axis_attrs(), "v")
    assert (x, y, t) == ("x", "y", "t")


def test_detects_via_standard_name_when_axis_missing():
    x, y, t = detect_coords(_ds_with_standard_names_only(), "v")
    assert (x, y, t) == ("lon", "lat", "time")


def test_detects_projected_xy():
    x, y, t = detect_coords(_ds_projected(), "v")
    assert (x, y, t) == ("x", "y", "time")


def test_override_takes_precedence():
    ds = _ds_with_axis_attrs()
    x, y, t = detect_coords(ds, "v", x_override="x", y_override="y", time_override="t")
    assert (x, y, t) == ("x", "y", "t")


def test_override_must_be_in_var_dims():
    ds = _ds_with_axis_attrs()
    with pytest.raises(ValueError, match="override"):
        detect_coords(ds, "v", x_override="bogus")


def test_raises_when_axis_unresolvable():
    ds = xr.Dataset(
        {"v": (["time", "lat", "lon"], np.zeros((1, 2, 2)))},
        coords={
            "time": ("time", [0], {"standard_name": "time"}),
            "lat": ("lat", [0.0, 1.0]),  # no attrs
            "lon": ("lon", [0.0, 1.0]),  # no attrs
        },
    )
    with pytest.raises(ValueError, match=r"(x|X)"):
        detect_coords(ds, "v")


def test_raises_when_var_missing():
    ds = _ds_with_axis_attrs()
    with pytest.raises(KeyError):
        detect_coords(ds, "not_a_var")
