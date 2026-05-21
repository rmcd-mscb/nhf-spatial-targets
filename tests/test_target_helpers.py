"""Unit tests for notebooks/targets/_helpers.py.

The helper module lives outside the package (it is a sibling of the
``inspect_target_*.ipynb`` notebooks), so we load it via importlib
rather than a regular import — same approach as
``tests/test_aggregated_helpers.py``.

Focus here is ``open_target_nc``'s on-disk time subsetting (issue #163):
the daily SWE target is ~11 GB, so the notebook subsets a single water
year before ``.load()`` rather than materialising the whole file.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPERS_PATH = REPO_ROOT / "notebooks" / "targets" / "_helpers.py"


@pytest.fixture(scope="session")
def helpers():
    spec = importlib.util.spec_from_file_location("target_helpers", HELPERS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def daily_target_nc(tmp_path: Path) -> Path:
    """A small synthetic daily target NC mirroring the SWE target schema."""
    time = pd.date_range("2009-10-01", "2011-09-30", freq="D")
    hru = np.arange(5, dtype="int64")
    shape = (time.size, hru.size)
    lower = np.zeros(shape, dtype="float32")
    upper = np.ones(shape, dtype="float32")
    n_sources = np.full(shape, 3, dtype="int8")
    ds = xr.Dataset(
        {
            "lower_bound": (("time", "nat_hru_id"), lower),
            "upper_bound": (("time", "nat_hru_id"), upper),
            "n_sources": (("time", "nat_hru_id"), n_sources),
        },
        coords={"time": time, "nat_hru_id": hru},
    )
    path = tmp_path / "swe_targets.nc"
    ds.to_netcdf(path)
    ds.close()
    return path


def test_open_target_nc_default_loads_full_range(helpers, daily_target_nc):
    """time=None is backwards-compatible: the whole file is loaded."""
    ds = helpers.open_target_nc(daily_target_nc)
    assert ds.sizes["time"] == pd.date_range("2009-10-01", "2011-09-30", freq="D").size
    # Detached from the handle: data is in memory, not a lazy/dask array.
    assert isinstance(ds["lower_bound"].data, np.ndarray)


def test_open_target_nc_tuple_window_is_inclusive(helpers, daily_target_nc):
    """A 2-tuple is treated as inclusive slice endpoints (label-based)."""
    ds = helpers.open_target_nc(daily_target_nc, time=("2009-10-01", "2010-09-30"))
    times = pd.DatetimeIndex(ds["time"].values)
    assert times.min() == pd.Timestamp("2009-10-01")
    assert times.max() == pd.Timestamp("2010-09-30")
    # WY2010 (Oct 1 2009 – Sep 30 2010) is 365 days.
    assert ds.sizes["time"] == 365
    # The windowed path is also eagerly materialised (handle detached).
    assert isinstance(ds["lower_bound"].data, np.ndarray)


def test_open_target_nc_slice_window_matches_tuple(helpers, daily_target_nc):
    """An explicit slice gives the same result as the tuple form."""
    window = ("2010-10-01", "2011-09-30")
    ds_tuple = helpers.open_target_nc(daily_target_nc, time=window)
    ds_slice = helpers.open_target_nc(daily_target_nc, time=slice(*window))
    assert ds_tuple.sizes["time"] == ds_slice.sizes["time"]
    assert (ds_tuple["time"].values == ds_slice["time"].values).all()


def test_open_target_nc_window_contains_target_date(helpers, daily_target_nc):
    """The notebook's derived water-year window contains TARGET_DATE."""
    target_date = pd.Timestamp("2010-03-01")
    wy_start = target_date.year - 1 if target_date.month < 10 else target_date.year
    window = (f"{wy_start}-10-01", f"{wy_start + 1}-09-30")
    ds = helpers.open_target_nc(daily_target_nc, time=window)
    # The at-date choropleth panels do ds.sel(time=TARGET_DATE) — must hit.
    sel = ds.sel(time=target_date)
    assert sel.sizes == {"nat_hru_id": 5}


def test_open_target_nc_subsets_before_load(helpers, daily_target_nc, monkeypatch):
    """The window is applied on-disk *before* .load() — the point of #163.

    Shape assertions alone can't catch a load-first regression
    (``ds.load().sel(...)`` returns the same window), so spy on
    ``xr.Dataset.load`` and assert it ran against the already-subset
    dataset (365 days) rather than the full file (730 days).
    """
    seen: dict[str, int] = {}
    orig_load = xr.Dataset.load

    def spy_load(self, *args, **kwargs):
        seen["time_size"] = self.sizes.get("time", 0)
        return orig_load(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "load", spy_load)
    helpers.open_target_nc(daily_target_nc, time=("2009-10-01", "2010-09-30"))
    assert seen["time_size"] == 365  # not the full 730-day file


def test_open_target_nc_window_outside_range_is_empty(helpers, daily_target_nc):
    """A window entirely outside the file range clips to an empty time dim.

    Pins the silent-empty contract: callers (e.g. an out-of-range
    TARGET_DATE) get a 0-length time axis rather than an exception here
    — the KeyError surfaces later at the at-date ``sel``, not in the
    loader.
    """
    ds = helpers.open_target_nc(daily_target_nc, time=("2030-01-01", "2030-12-31"))
    assert ds.sizes["time"] == 0


def test_open_target_nc_window_clips_to_available_range(helpers, daily_target_nc):
    """A window straddling the file's lower edge clips, not raises."""
    # File starts 2009-10-01; ask from before that.
    ds = helpers.open_target_nc(daily_target_nc, time=("2009-01-01", "2009-10-31"))
    times = pd.DatetimeIndex(ds["time"].values)
    assert times.min() == pd.Timestamp("2009-10-01")
    assert times.max() == pd.Timestamp("2009-10-31")
