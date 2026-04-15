# Per-Year Streaming Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `aggregate/_driver.py` so every adapter streams aggregation year-by-year with CF-based coord detection and restartable per-year intermediates, fixing MOD16A2/MOD10C1 OOM and the `['time','lat','lon']`-vs-`time,y,x` coord mismatch.

**Architecture:** One uniform pipeline — enumerate years from `*_consolidated.nc` time coords → aggregate each year via gdptools with explicit `period=(YYYY-01-01, YYYY-12-31)` → write per-year NC atomically to `data/aggregated/_by_year/` (idempotent; skip if exists) → final concat on time → atomic write to `data/aggregated/<src>_agg.nc`. MOD10C1 becomes an adapter with `pre_aggregate_hook` (CI mask) and `post_aggregate_hook` (rename + warn). Coord names come from CF attrs (`axis`, `standard_name`) with adapter fields as optional overrides.

**Tech Stack:** Python 3.11+, xarray, gdptools (`UserCatData`, `WeightGen`, `AggGen`), geopandas, pandas, pytest. All commands go through `pixi run -e dev …`.

**Spec:** [`docs/superpowers/specs/2026-04-15-per-year-streaming-aggregation-design.md`](../specs/2026-04-15-per-year-streaming-aggregation-design.md)

---

## File structure

Files created / modified across tasks:

| Path | Action | Purpose |
|---|---|---|
| `src/nhf_spatial_targets/aggregate/_coords.py` | **Create** | CF coord detection. |
| `src/nhf_spatial_targets/aggregate/_adapter.py` | **Modify** | Optional coord overrides; `pre/post_aggregate_hook` fields. |
| `src/nhf_spatial_targets/aggregate/_driver.py` | **Modify** | Per-year enumeration, `aggregate_year`, `concat_years`, new orchestration in `aggregate_source`; delete `_default_open_hook` concat branch. |
| `src/nhf_spatial_targets/aggregate/mod10c1.py` | **Modify** | Collapse to adapter + hooks. |
| `src/nhf_spatial_targets/aggregate/mod16a2.py` | **Modify** | Drop hard-coded `x_coord`/`y_coord` (fall through to CF detection). |
| `tests/test_aggregate_coords.py` | **Create** | Unit tests for `detect_coords`. |
| `tests/test_aggregate_driver_per_year.py` | **Create** | Unit tests for `enumerate_years`, `aggregate_year` (idempotency), `concat_years`. |
| `tests/test_aggregate_driver.py` | **Modify** | Delete tests for the removed concat-in-`_default_open_hook` path; update `test_aggregate_source_writes_multi_var_nc_and_manifest` to the new pipeline. |
| `tests/test_aggregate_mod10c1.py` | **Modify** | Delete `_open` tests; add adapter-wiring test. |

---

## Task 1: Coordinate detection module

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/_coords.py`
- Test: `tests/test_aggregate_coords.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_aggregate_coords.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pixi run -e dev test -- tests/test_aggregate_coords.py
```

Expected: ImportError / ModuleNotFoundError on `nhf_spatial_targets.aggregate._coords`.

- [ ] **Step 3: Implement `_coords.py`**

Create `src/nhf_spatial_targets/aggregate/_coords.py`:

```python
"""CF-based coordinate detection for source datasets."""

from __future__ import annotations

import xarray as xr

_X_STANDARD_NAMES = frozenset({"longitude", "projection_x_coordinate"})
_Y_STANDARD_NAMES = frozenset({"latitude", "projection_y_coordinate"})
_T_STANDARD_NAMES = frozenset({"time"})


def _find_axis(
    ds: xr.Dataset,
    dims: tuple[str, ...],
    axis_letter: str,
    standard_names: frozenset[str],
) -> str | None:
    # First pass: CF axis attribute.
    for name in dims:
        if name in ds.coords and ds.coords[name].attrs.get("axis") == axis_letter:
            return name
    # Second pass: CF standard_name.
    for name in dims:
        if name in ds.coords and ds.coords[name].attrs.get("standard_name") in standard_names:
            return name
    return None


def detect_coords(
    ds: xr.Dataset,
    var: str,
    x_override: str | None = None,
    y_override: str | None = None,
    time_override: str | None = None,
) -> tuple[str, str, str]:
    """Return (x_coord, y_coord, time_coord) for ``ds[var]``.

    Resolution order per axis:
      1. Explicit override (must be one of ``ds[var].dims``).
      2. Coordinate whose ``axis`` attr is 'X' / 'Y' / 'T'.
      3. Coordinate whose ``standard_name`` attr matches a CF name for that axis.

    Raises KeyError if ``var`` is not in ``ds``.
    Raises ValueError if an override is not in ``ds[var].dims`` or if any axis
    cannot be resolved after overrides + CF passes.
    """
    if var not in ds.data_vars:
        raise KeyError(f"Variable {var!r} not in dataset (have {list(ds.data_vars)})")
    dims = tuple(ds[var].dims)

    def _resolve(
        override: str | None,
        axis_letter: str,
        standard_names: frozenset[str],
        label: str,
    ) -> str:
        if override is not None:
            if override not in dims:
                raise ValueError(
                    f"{label} override {override!r} is not a dim of {var!r} "
                    f"(dims={dims})"
                )
            return override
        found = _find_axis(ds, dims, axis_letter, standard_names)
        if found is None:
            attrs_by_dim = {d: dict(ds.coords[d].attrs) for d in dims if d in ds.coords}
            raise ValueError(
                f"Could not detect {label} coord for {var!r}. "
                f"No dim has axis={axis_letter!r} or standard_name in "
                f"{sorted(standard_names)}. "
                f"dims={dims}, coord attrs={attrs_by_dim}"
            )
        return found

    x = _resolve(x_override, "X", _X_STANDARD_NAMES, "x")
    y = _resolve(y_override, "Y", _Y_STANDARD_NAMES, "y")
    t = _resolve(time_override, "T", _T_STANDARD_NAMES, "time")
    return x, y, t
```

- [ ] **Step 4: Run tests to verify they pass**

```
pixi run -e dev test -- tests/test_aggregate_coords.py
```

Expected: all pass.

- [ ] **Step 5: Lint, format, and commit**

```
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/aggregate/_coords.py tests/test_aggregate_coords.py
git commit -m "feat: CF-based coord detection for aggregate driver"
```

---

## Task 2: Extend `SourceAdapter` with optional coord overrides and hooks

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_adapter.py`
- Modify: `tests/test_aggregate_driver.py` (adapter default assertions)

- [ ] **Step 1: Update adapter default-field tests**

In `tests/test_aggregate_driver.py`, replace `test_source_adapter_defaults`:

```python
def test_source_adapter_defaults():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP", "GWETROOT"],
    )
    assert adapter.source_crs == "EPSG:4326"
    assert adapter.x_coord is None
    assert adapter.y_coord is None
    assert adapter.time_coord is None
    assert adapter.open_hook is None
    assert adapter.pre_aggregate_hook is None
    assert adapter.post_aggregate_hook is None
```

Add a new test:

```python
def test_source_adapter_accepts_hooks():
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter

    def _pre(ds):
        return ds

    def _post(ds):
        return ds

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["GWETTOP"],
        pre_aggregate_hook=_pre,
        post_aggregate_hook=_post,
    )
    assert adapter.pre_aggregate_hook is _pre
    assert adapter.post_aggregate_hook is _post
```

- [ ] **Step 2: Run tests to verify failures**

```
pixi run -e dev test -- tests/test_aggregate_driver.py::test_source_adapter_defaults tests/test_aggregate_driver.py::test_source_adapter_accepts_hooks
```

Expected: FAIL (fields don't yet exist / have wrong defaults).

- [ ] **Step 3: Implement adapter changes**

Edit `src/nhf_spatial_targets/aggregate/_adapter.py` — change the three coord fields to optional and add the two hook fields. Replace the dataclass body with:

```python
@dataclass(frozen=True)
class SourceAdapter:
    """..."""

    source_key: str
    output_name: str
    variables: tuple[str, ...]
    x_coord: str | None = None
    y_coord: str | None = None
    time_coord: str | None = None
    source_crs: str = "EPSG:4326"
    grid_variable: str | None = None
    open_hook: Callable[[Project], xr.Dataset] | None = field(default=None)
    pre_aggregate_hook: Callable[[xr.Dataset], xr.Dataset] | None = field(default=None)
    post_aggregate_hook: Callable[[xr.Dataset], xr.Dataset] | None = field(default=None)
```

Leave `__post_init__` unchanged; coord validation is now deferred to `detect_coords` at run time.

- [ ] **Step 4: Verify tests pass**

```
pixi run -e dev test -- tests/test_aggregate_driver.py
```

Expected: all pass (note `test_aggregate_variables_for_batch_merges_variables` still passes because it bypasses the adapter).

- [ ] **Step 5: Commit**

```
git add src/nhf_spatial_targets/aggregate/_adapter.py tests/test_aggregate_driver.py
git commit -m "refactor: make SourceAdapter coord fields optional and add pre/post aggregate hooks"
```

---

## Task 3: Add explicit `period` parameter to batch helpers

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py` (`compute_or_load_weights`, `aggregate_variables_for_batch`)
- Modify: `tests/test_aggregate_driver.py`

The per-year pipeline needs each batch call to target a specific year's period. Today both helpers derive `period` from `source_ds[time_coord].values[0]` / `[-1]`. Take `period: tuple[str, str]` as a required parameter instead.

- [ ] **Step 1: Update existing tests to pass `period`**

In `tests/test_aggregate_driver.py`, add `period=("2000-01-01", "2000-02-01")` to the `compute_or_load_weights` and `aggregate_variables_for_batch` calls in:

- `test_aggregate_variables_for_batch_merges_variables`
- `test_compute_or_load_weights_writes_cache_on_miss`
- `test_compute_or_load_weights_uses_cache_on_hit`
- `test_compute_or_load_weights_ignores_stray_tmp_from_crashed_run`

- [ ] **Step 2: Add a test that verifies `period` is passed through to `UserCatData`**

Append to `tests/test_aggregate_driver.py`:

```python
def test_aggregate_variables_for_batch_passes_period_through(tiny_fabric):
    from nhf_spatial_targets.aggregate._driver import aggregate_variables_for_batch

    batch_gdf = gpd.read_file(tiny_fabric)
    batch_gdf["batch_id"] = 0
    times = pd.date_range("2005-01-01", periods=12, freq="MS")
    source_ds = xr.Dataset(
        {"a": (["time", "lat", "lon"], np.ones((12, 2, 2)))},
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )

    with (
        patch("nhf_spatial_targets.aggregate._driver.AggGen") as mock_agg,
        patch("nhf_spatial_targets.aggregate._driver.UserCatData") as mock_ucd,
    ):
        mock_ucd.return_value = _fake_user_data()
        inst = MagicMock()
        mock_agg.return_value = inst
        inst.calculate_agg.side_effect = [_fake_agg_result("a", [0, 1, 2, 3])]

        aggregate_variables_for_batch(
            batch_gdf=batch_gdf,
            source_ds=source_ds,
            variables=["a"],
            source_crs="EPSG:4326",
            x_coord="lon",
            y_coord="lat",
            time_coord="time",
            id_col="hru_id",
            weights=_fake_weights(),
            period=("2005-01-01", "2005-12-01"),
        )

    call_kwargs = mock_ucd.call_args.kwargs
    assert call_kwargs["period"] == ["2005-01-01", "2005-12-01"]
```

- [ ] **Step 3: Run tests to verify failures**

```
pixi run -e dev test -- tests/test_aggregate_driver.py
```

Expected: the updated tests fail with TypeError ("unexpected keyword argument 'period'").

- [ ] **Step 4: Modify the helpers**

Edit `src/nhf_spatial_targets/aggregate/_driver.py`. Change the signatures and the body of both helpers to use an explicit period.

`compute_or_load_weights` — add `period: tuple[str, str]` as a keyword-only argument before `source_key`:

```python
def compute_or_load_weights(
    batch_gdf: gpd.GeoDataFrame,
    source_ds: xr.Dataset,
    source_var: str,
    source_crs: str,
    x_coord: str,
    y_coord: str,
    time_coord: str,
    id_col: str,
    source_key: str,
    batch_id: int,
    workdir: Path,
    period: tuple[str, str],
) -> pd.DataFrame:
    ...
    user_data = UserCatData(
        ds=source_ds,
        proj_ds=source_crs,
        x_coord=x_coord,
        y_coord=y_coord,
        t_coord=time_coord,
        var=[source_var],
        f_feature=batch_gdf,
        proj_feature=batch_gdf.crs.to_string(),
        id_feature=id_col,
        period=[period[0], period[1]],
    )
    ...
```

`aggregate_variables_for_batch` — same treatment; accept `period` and pass it through:

```python
def aggregate_variables_for_batch(
    batch_gdf: gpd.GeoDataFrame,
    source_ds: xr.Dataset,
    variables: list[str],
    source_crs: str,
    x_coord: str,
    y_coord: str,
    time_coord: str,
    id_col: str,
    weights: pd.DataFrame,
    period: tuple[str, str],
) -> xr.Dataset:
    per_var: list[xr.Dataset] = []
    for var in variables:
        user_data = UserCatData(
            ds=source_ds,
            proj_ds=source_crs,
            x_coord=x_coord,
            y_coord=y_coord,
            t_coord=time_coord,
            var=[var],
            f_feature=batch_gdf,
            proj_feature=batch_gdf.crs.to_string(),
            id_feature=id_col,
            period=[period[0], period[1]],
        )
        agg = AggGen(
            user_data=user_data,
            stat_method="masked_mean",
            agg_engine="serial",
            agg_writer="none",
            weights=weights,
        )
        _gdf, ds = agg.calculate_agg()
        per_var.append(ds)
    return xr.merge(per_var)
```

No other callers exist today (they're inside `aggregate_source` which we rewrite in Task 7).

- [ ] **Step 5: Run tests**

```
pixi run -e dev test -- tests/test_aggregate_driver.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```
git add src/nhf_spatial_targets/aggregate/_driver.py tests/test_aggregate_driver.py
git commit -m "refactor: require explicit period in compute_or_load_weights / aggregate_variables_for_batch"
```

---

## Task 4: `enumerate_years` — map consolidated NCs to years

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py`
- Create: `tests/test_aggregate_driver_per_year.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_aggregate_driver_per_year.py`:

```python
"""Unit tests for per-year streaming aggregation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


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
```

- [ ] **Step 2: Run tests — expect ImportError**

```
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py
```

- [ ] **Step 3: Implement `enumerate_years`**

Add to `src/nhf_spatial_targets/aggregate/_driver.py` (alongside the existing helpers). Include the `detect_coords` import at the top of the module.

```python
from nhf_spatial_targets.aggregate._coords import detect_coords


def enumerate_years(files: list[Path]) -> list[tuple[int, Path]]:
    """Map each year covered by the files to its source file.

    Opens each NC lazily, reads its time coord, and expands to one
    ``(year, file)`` tuple per distinct year. Returns results sorted by year.

    Raises:
        ValueError: two files cover the same year (stale fetch), or a file has
            no resolvable time coord.
    """
    year_to_file: dict[int, Path] = {}
    for path in files:
        with xr.open_dataset(path) as ds:
            time_name = _find_time_coord_name(ds)
            if time_name is None:
                raise ValueError(
                    f"No time coord found in {path.name}. "
                    f"dims={list(ds.dims)}, coord attrs="
                    f"{ {n: dict(ds.coords[n].attrs) for n in ds.coords} }"
                )
            years = pd.DatetimeIndex(ds[time_name].values).year.unique().tolist()
        for y in years:
            if y in year_to_file:
                raise ValueError(
                    f"Year {y} overlaps between {year_to_file[y].name} "
                    f"and {path.name}. Check the datastore for a stale "
                    f"fetch artifact."
                )
            year_to_file[y] = path
    return sorted(year_to_file.items())


def _find_time_coord_name(ds: xr.Dataset) -> str | None:
    """Return the name of the time coord via CF attrs, or None."""
    for name in ds.coords:
        attrs = ds.coords[name].attrs
        if attrs.get("axis") == "T" or attrs.get("standard_name") == "time":
            return name
    # Fall back: any coord literally named 'time' (non-CF legacy NCs).
    return "time" if "time" in ds.coords else None
```

- [ ] **Step 4: Run tests — expect pass**

```
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py
```

- [ ] **Step 5: Commit**

```
git add src/nhf_spatial_targets/aggregate/_driver.py tests/test_aggregate_driver_per_year.py
git commit -m "feat: enumerate_years helper for per-year streaming aggregation"
```

---

## Task 5: `per_year_output_path` + idempotent `aggregate_year`

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py`
- Modify: `tests/test_aggregate_driver_per_year.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_aggregate_driver_per_year.py`:

```python
import json
import yaml
import geopandas as gpd
from unittest.mock import MagicMock, patch
from shapely.geometry import box

from nhf_spatial_targets.workspace import load as load_project


@pytest.fixture()
def project(tmp_path):
    datastore = tmp_path / "datastore"
    datastore.mkdir()
    (tmp_path / "config.yml").write_text(
        yaml.dump({"fabric": {"path": "", "id_col": "hru_id"}, "datastore": str(datastore)})
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
    assert p == project.workdir / "data" / "aggregated" / "_by_year" / "foo_2005_agg.nc"


def test_aggregate_year_skips_when_intermediate_exists(project, tiny_batched_fabric):
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_year, per_year_output_path

    out_path = per_year_output_path(project, "merra2", 2005)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write a stub intermediate so aggregate_year short-circuits.
    times = pd.date_range("2005-01-01", periods=1, freq="MS")
    xr.Dataset(
        {"GWETTOP": (["time", "hru_id"], np.ones((1, 2)))},
        coords={"time": times, "hru_id": [0, 1]},
    ).to_netcdf(out_path)

    src_file = project.raw_dir("merra2")
    src_file.mkdir(parents=True, exist_ok=True)
    src_file = src_file / "src_2005_consolidated.nc"
    _write_nc(src_file, pd.date_range("2005-01-01", periods=12, freq="MS"))

    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=["v"]
    )
    with patch(
        "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch"
    ) as mock_batch:
        path = aggregate_year(adapter, project, 2005, src_file, tiny_batched_fabric, "hru_id")
    assert path == out_path
    assert not mock_batch.called, "existing intermediate must skip aggregation"


def test_aggregate_year_writes_intermediate_when_missing(project, tiny_batched_fabric):
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_year, per_year_output_path

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
            "time": ("time", pd.date_range("2005-01-01", periods=1, freq="MS"),
                     {"standard_name": "time"}),
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
    # Period was passed through with the year bounds.
    call = mock_batch.call_args
    assert call.kwargs["period"] == ("2005-01-01", "2005-12-31")
```

- [ ] **Step 2: Run tests — expect failures**

```
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py
```

- [ ] **Step 3: Implement the two functions**

Add to `src/nhf_spatial_targets/aggregate/_driver.py`:

```python
def per_year_output_path(project: Project, source_key: str, year: int) -> Path:
    """Return the per-year intermediate NC path."""
    return (
        project.workdir
        / "data"
        / "aggregated"
        / "_by_year"
        / f"{source_key}_{year}_agg.nc"
    )


def _atomic_write_netcdf(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".nc.tmp")
    os.close(tmp_fd)
    try:
        ds.to_netcdf(tmp_path)
        Path(tmp_path).replace(path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def aggregate_year(
    adapter: "SourceAdapter",
    project: Project,
    year: int,
    source_file: Path,
    fabric_batched: gpd.GeoDataFrame,
    id_col: str,
) -> Path:
    """Aggregate one year to HRU polygons; idempotent on the intermediate NC.

    Returns the path of the per-year intermediate. If that path already
    exists, returns immediately without opening the source file.
    """
    out_path = per_year_output_path(project, adapter.source_key, year)
    if out_path.exists():
        logger.info("%s: year %d: intermediate exists, skipping (%s)",
                    adapter.source_key, year, out_path)
        return out_path

    logger.info("%s: year %d: aggregating from %s",
                adapter.source_key, year, source_file.name)
    period = (f"{year}-01-01", f"{year}-12-31")

    with xr.open_dataset(source_file) as raw:
        ds = raw
        if adapter.pre_aggregate_hook is not None:
            ds = adapter.pre_aggregate_hook(ds)

        grid_var = adapter.grid_variable or adapter.variables[0]
        x_coord, y_coord, time_coord = detect_coords(
            ds,
            grid_var,
            x_override=adapter.x_coord,
            y_override=adapter.y_coord,
            time_override=adapter.time_coord,
        )

        datasets: list[xr.Dataset] = []
        for bid in sorted(fabric_batched["batch_id"].unique()):
            batch_gdf = fabric_batched[fabric_batched["batch_id"] == bid].drop(
                columns=["batch_id"]
            )
            weights = compute_or_load_weights(
                batch_gdf=batch_gdf,
                source_ds=ds,
                source_var=grid_var,
                source_crs=adapter.source_crs,
                x_coord=x_coord,
                y_coord=y_coord,
                time_coord=time_coord,
                id_col=id_col,
                source_key=adapter.source_key,
                batch_id=int(bid),
                workdir=project.workdir,
                period=period,
            )
            batch_ds = aggregate_variables_for_batch(
                batch_gdf=batch_gdf,
                source_ds=ds,
                variables=list(adapter.variables),
                source_crs=adapter.source_crs,
                x_coord=x_coord,
                y_coord=y_coord,
                time_coord=time_coord,
                id_col=id_col,
                weights=weights,
                period=period,
            )
            datasets.append(batch_ds)

        year_ds = xr.concat(datasets, dim=id_col)

    _atomic_write_netcdf(year_ds, out_path)
    logger.info("%s: year %d: wrote %s", adapter.source_key, year, out_path)
    return out_path
```

- [ ] **Step 4: Run tests — expect pass**

```
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py
```

- [ ] **Step 5: Commit**

```
git add src/nhf_spatial_targets/aggregate/_driver.py tests/test_aggregate_driver_per_year.py
git commit -m "feat: per-year aggregation with idempotent intermediate"
```

---

## Task 6: `concat_years` — final concat with validation

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py`
- Modify: `tests/test_aggregate_driver_per_year.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_aggregate_driver_per_year.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect failures**

```
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py
```

- [ ] **Step 3: Implement `concat_years`**

Add to `src/nhf_spatial_targets/aggregate/_driver.py`:

```python
def concat_years(paths: list[Path], time_coord: str) -> xr.Dataset:
    """Open per-year intermediates, concat on time, validate monotonic + unique.

    Loads each intermediate into memory and closes the on-disk handle so the
    returned Dataset is detached from the filesystem (per the project's
    rioxarray/close convention).
    """
    if not paths:
        raise ValueError("concat_years called with empty paths list")
    loaded: list[xr.Dataset] = []
    for p in paths:
        with xr.open_dataset(p) as ds:
            loaded.append(ds.load())
    combined = xr.concat(loaded, dim=time_coord).sortby(time_coord)
    t = combined[time_coord].values
    if len(np.unique(t)) != len(t):
        raise ValueError(
            f"Duplicate time coords across per-year intermediates: "
            f"{[p.name for p in paths]}"
        )
    return combined
```

- [ ] **Step 4: Run tests — expect pass**

```
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py
```

- [ ] **Step 5: Commit**

```
git add src/nhf_spatial_targets/aggregate/_driver.py tests/test_aggregate_driver_per_year.py
git commit -m "feat: concat_years helper for per-year intermediates"
```

---

## Task 7: Rewrite `aggregate_source` to orchestrate the per-year pipeline

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py`
- Modify: `tests/test_aggregate_driver.py`

- [ ] **Step 1: Update the integration test for the new pipeline**

In `tests/test_aggregate_driver.py`, replace `test_aggregate_source_writes_multi_var_nc_and_manifest` with:

```python
def test_aggregate_source_writes_multi_var_nc_and_manifest(tmp_path, tiny_fabric):
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    datastore = tmp_path / "datastore"
    (datastore / "merra2").mkdir(parents=True)
    src_nc = datastore / "merra2" / "merra2_2000_consolidated.nc"
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    xr.Dataset(
        {
            "a": (["time", "lat", "lon"], np.ones((12, 2, 2))),
            "b": (["time", "lat", "lon"], np.ones((12, 2, 2)) * 2.0),
        },
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "lat": ("lat", [0.25, 0.75], {"standard_name": "latitude"}),
            "lon": ("lon", [0.5, 1.5], {"standard_name": "longitude"}),
        },
    ).to_netcdf(src_nc)

    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": str(tiny_fabric), "id_col": "hru_id"},
                "datastore": str(datastore),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    (tmp_path / "data" / "aggregated").mkdir(parents=True)
    (tmp_path / "weights").mkdir()

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["a", "b"],
    )

    fake_meta = {"access": {"type": "local_nc"}}
    fake_year_ds = xr.Dataset(
        {
            "a": (["time", "hru_id"], np.ones((12, 4))),
            "b": (["time", "hru_id"], np.ones((12, 4)) * 2.0),
        },
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "hru_id": [0, 1, 2, 3],
        },
    )

    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.catalog_source",
            return_value=fake_meta,
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=_fake_weights(),
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            return_value=fake_year_ds,
        ),
    ):
        out = aggregate_source(
            adapter,
            fabric_path=tiny_fabric,
            id_col="hru_id",
            workdir=tmp_path,
            batch_size=500,
        )

    assert set(out.data_vars) == {"a", "b"}
    output_nc = tmp_path / "data" / "aggregated" / "merra2_agg.nc"
    assert output_nc.exists()
    # Per-year intermediate preserved.
    assert (tmp_path / "data" / "aggregated" / "_by_year" / "merra2_2000_agg.nc").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "merra2" in manifest["sources"]
    assert manifest["sources"]["merra2"]["output_file"] == (
        "data/aggregated/merra2_agg.nc"
    )
```

Also update `test_aggregate_source_raises_when_variable_missing` (same file): the failure should still surface, but now happens inside `aggregate_year` after opening the first year's file. Amend the assertion so it passes with the new pipeline — change the NC filename to `merra2_2000_consolidated.nc` and keep the `pytest.raises(ValueError, match="BOGUS_VAR")` check. The `aggregate_year` body must raise when a declared variable is absent; add that guard in Step 3.

Delete the now-obsolete `_default_open_hook` tests (entire functions):

- `test_default_open_hook_single_consolidated_nc`
- `test_default_open_hook_concatenates_multiple_consolidated_ncs`
- `test_default_open_hook_concat_sorts_when_filename_order_disagrees_with_time`
- `test_default_open_hook_returned_dataset_detached_from_disk`
- `test_default_open_hook_single_non_consolidated_nc`
- `test_default_open_hook_raises_when_multiple_ncs_none_consolidated`
- `test_default_open_hook_raises_when_no_nc_at_all`
- `test_default_open_hook_raises_on_duplicate_time_across_consolidated_ncs`

Their semantics are replaced by `enumerate_years` + `aggregate_year` tests from Tasks 4-5.

- [ ] **Step 2: Run tests — expect failures**

```
pixi run -e dev test -- tests/test_aggregate_driver.py
```

- [ ] **Step 3: Rewrite `aggregate_source` and delete `_default_open_hook`**

In `src/nhf_spatial_targets/aggregate/_driver.py`:

1. **Delete** `_default_open_hook` entirely.
2. **Add** the CF global-attr helper:

```python
def _attach_cf_global_attrs(ds: xr.Dataset, source_key: str, meta: dict) -> None:
    """Attach CF-1.6 global attrs in place (non-destructive for var attrs)."""
    access = meta.get("access", {})
    history = (
        f"{datetime.now(timezone.utc).isoformat()}: aggregated to HRU fabric "
        f"by nhf_spatial_targets.aggregate._driver"
    )
    ds.attrs.setdefault("Conventions", "CF-1.6")
    ds.attrs["history"] = history
    ds.attrs["source"] = source_key
    if "doi" in access:
        ds.attrs["source_doi"] = access["doi"]
    ds.attrs.setdefault("institution", "USGS")
```

3. **Rewrite** `aggregate_source`:

```python
def aggregate_source(
    adapter: SourceAdapter,
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
) -> xr.Dataset:
    """Aggregate a source to fabric HRU polygons via the per-year pipeline.

    Enumerates years from ``*_consolidated.nc`` in the datastore, aggregates
    each year to ``data/aggregated/_by_year/<source_key>_<year>_agg.nc``
    (idempotent; existing intermediates are reused on restart), then concats
    on time to ``data/aggregated/<output_name>``. Per-year intermediates
    are preserved for audit/restart.

    Variables declared by ``adapter.variables`` that are missing from the
    source NC cause ValueError before any year is aggregated.
    """
    workdir = Path(workdir)
    project = load_project(workdir)
    meta = catalog_source(adapter.source_key)

    raw_dir = project.raw_dir(adapter.source_key)
    files = sorted(raw_dir.glob("*_consolidated.nc"))
    if not files:
        raise FileNotFoundError(
            f"No consolidated NC found in {raw_dir}. "
            f"Run 'nhf-targets fetch {adapter.source_key}' first."
        )

    # Fail fast on missing declared variables: peek the first file.
    with xr.open_dataset(files[0]) as peek:
        missing = [v for v in adapter.variables if v not in peek.data_vars]
        if missing:
            raise ValueError(
                f"{adapter.source_key}: variables {missing} missing from source "
                f"dataset (have {list(peek.data_vars)})"
            )

    year_files = enumerate_years(files)
    fabric_batched = load_and_batch_fabric(fabric_path, batch_size=batch_size)
    n_batches = int(fabric_batched["batch_id"].nunique())
    logger.info(
        "%s: %d years to aggregate across %d spatial batches",
        adapter.source_key,
        len(year_files),
        n_batches,
    )

    per_year_paths = [
        aggregate_year(adapter, project, year, path, fabric_batched, id_col)
        for year, path in year_files
    ]

    # Resolve time coord name from the first intermediate for the concat.
    with xr.open_dataset(per_year_paths[0]) as probe:
        time_coord = _find_time_coord_name(probe) or "time"
    combined = concat_years(per_year_paths, time_coord=time_coord)

    if adapter.post_aggregate_hook is not None:
        combined = adapter.post_aggregate_hook(combined)

    _attach_cf_global_attrs(combined, adapter.source_key, meta)

    output_path = project.aggregated_dir() / adapter.output_name
    _atomic_write_netcdf(combined, output_path)
    logger.info("%s: output written to %s", adapter.source_key, output_path)

    t0 = str(combined[time_coord].values[0])[:10]
    t1 = str(combined[time_coord].values[-1])[:10]
    update_manifest(
        project=project,
        source_key=adapter.source_key,
        access=meta.get("access", {}),
        period=f"{t0}/{t1}",
        output_file=str(Path("data") / "aggregated" / adapter.output_name),
        weight_files=[
            str(Path("weights") / f"{adapter.source_key}_batch{i}.csv")
            for i in range(n_batches)
        ],
    )
    return combined
```

- [ ] **Step 4: Run the full driver test file**

```
pixi run -e dev test -- tests/test_aggregate_driver.py tests/test_aggregate_driver_per_year.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```
git add src/nhf_spatial_targets/aggregate/_driver.py tests/test_aggregate_driver.py
git commit -m "refactor: aggregate_source uses per-year streaming pipeline"
```

---

## Task 8: Collapse `mod10c1.py` onto the adapter + hooks path

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/mod10c1.py`
- Modify: `tests/test_aggregate_mod10c1.py`

- [ ] **Step 1: Update tests**

In `tests/test_aggregate_mod10c1.py`:

- **Delete** the import of `_open` and every `_open` test (all six `test_open_*` functions).
- Keep the `build_masked_source` and `_log_low_valid_coverage` tests.
- Add a new test verifying the adapter wiring:

```python
def test_mod10c1_adapter_wires_hooks_and_variables():
    from nhf_spatial_targets.aggregate.mod10c1 import (
        ADAPTER,
        build_masked_source,
    )

    assert ADAPTER.source_key == "mod10c1_v061"
    assert ADAPTER.output_name == "mod10c1_agg.nc"
    assert set(ADAPTER.variables) == {"sca", "ci", "valid_mask"}
    assert ADAPTER.grid_variable == "sca"
    assert ADAPTER.pre_aggregate_hook is build_masked_source
    assert ADAPTER.post_aggregate_hook is not None
```

- [ ] **Step 2: Run tests — expect failures**

```
pixi run -e dev test -- tests/test_aggregate_mod10c1.py
```

- [ ] **Step 3: Rewrite `src/nhf_spatial_targets/aggregate/mod10c1.py`**

Replace the entire file with:

```python
"""MOD10C1 v061 daily SCA aggregator (CI-masked) — adapter + hooks."""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source

logger = logging.getLogger(__name__)

_SOURCE_KEY = "mod10c1_v061"
_CI_THRESHOLD = 0.70  # TM 6-B10: keep cells where CI > 0.70
_OUTPUT_NAME = "mod10c1_agg.nc"
_LOW_COVERAGE_WARN_THRESHOLD = 0.10


def build_masked_source(ds: xr.Dataset) -> xr.Dataset:
    """Derive ``sca``, ``ci``, ``valid_mask`` from raw MOD10C1 variables.

    - ``sca``        = Day_CMG_Snow_Cover / 100, NaN where CI <= 0.70.
    - ``ci``         = Snow_Spatial_QA / 100 (passed through, unmasked).
    - ``valid_mask`` = 1.0 where CI > 0.70, 0.0 otherwise.
    """
    ci = ds["Snow_Spatial_QA"] / 100.0
    pass_mask = ci > _CI_THRESHOLD
    sca_raw = ds["Day_CMG_Snow_Cover"] / 100.0
    sca = sca_raw.where(pass_mask)
    valid_mask = pass_mask.astype("float64")

    out = xr.Dataset(
        {"sca": sca, "ci": ci, "valid_mask": valid_mask},
        coords=ds.coords,
    )
    out["sca"].attrs = {"long_name": "fractional snow-covered area", "units": "1"}
    out["ci"].attrs = {
        "long_name": "confidence interval (Snow_Spatial_QA/100)",
        "units": "1",
    }
    out["valid_mask"].attrs = {
        "long_name": "per-cell CI-pass indicator",
        "units": "1",
        "ci_threshold": _CI_THRESHOLD,
    }
    return out


def _log_low_valid_coverage(combined: xr.Dataset) -> None:
    """Warn if > 10% of (HRU, time) cells have zero valid area."""
    vaf = combined["valid_area_fraction"]
    n_total = int(vaf.notnull().sum())
    if n_total == 0:
        return
    n_zero = int(((vaf == 0) & vaf.notnull()).sum())
    zero_frac = n_zero / n_total
    if zero_frac > _LOW_COVERAGE_WARN_THRESHOLD:
        logger.warning(
            "mod10c1: %.1f%% of (HRU, time) cells had zero valid-area "
            "after CI>%.2f filter (n=%d of %d finite). Downstream sca "
            "values are NaN for these cells.",
            zero_frac * 100,
            _CI_THRESHOLD,
            n_zero,
            n_total,
        )


def _rename_and_warn(combined: xr.Dataset) -> xr.Dataset:
    combined = combined.rename({"valid_mask": "valid_area_fraction"})
    combined["valid_area_fraction"].attrs = {
        "long_name": "fraction of HRU area that passed CI filter",
        "units": "1",
        "ci_threshold": _CI_THRESHOLD,
    }
    _log_low_valid_coverage(combined)
    return combined


ADAPTER = SourceAdapter(
    source_key=_SOURCE_KEY,
    output_name=_OUTPUT_NAME,
    variables=("sca", "ci", "valid_mask"),
    grid_variable="sca",
    source_crs="EPSG:4326",
    pre_aggregate_hook=build_masked_source,
    post_aggregate_hook=_rename_and_warn,
)


def aggregate_mod10c1(
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
) -> xr.Dataset:
    """Aggregate MOD10C1 v061 daily SCA to HRU polygons with CI masking.

    The CI>0.70 filter is applied per-year inside the driver's per-year loop
    via ``pre_aggregate_hook``. Final output at
    ``data/aggregated/mod10c1_agg.nc`` carries ``sca``, ``ci``, and
    ``valid_area_fraction`` keyed on (time, HRU).
    """
    return aggregate_source(ADAPTER, fabric_path, id_col, workdir, batch_size)
```

**Note:** `ADAPTER.variables` is declared as `("sca", "ci", "valid_mask")`. These variables exist on the output of `build_masked_source`, not on the raw NC. The driver's "fail fast on missing variables" check in `aggregate_source` peeks the raw NC for `adapter.variables` — which would spuriously fail here. **Add a hook-aware check**: if `adapter.pre_aggregate_hook is not None`, skip the peek (the hook constructs the variables). Update Step 3 of Task 7 to wrap the peek:

```python
if adapter.pre_aggregate_hook is None:
    with xr.open_dataset(files[0]) as peek:
        missing = [v for v in adapter.variables if v not in peek.data_vars]
        if missing:
            raise ValueError(...)
```

(Fold this adjustment into `_driver.py` as part of this task.)

- [ ] **Step 4: Run tests**

```
pixi run -e dev test -- tests/test_aggregate_mod10c1.py tests/test_aggregate_driver.py tests/test_aggregate_driver_per_year.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```
git add src/nhf_spatial_targets/aggregate/mod10c1.py tests/test_aggregate_mod10c1.py src/nhf_spatial_targets/aggregate/_driver.py
git commit -m "refactor: MOD10C1 uses SourceAdapter + pre/post aggregate hooks"
```

---

## Task 9: Let MOD16A2 rely on CF coord detection

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/mod16a2.py`

MOD16A2 currently sets `x_coord="x", y_coord="y"` explicitly. With Task 2 the adapter accepts `None`, and Task 1's detector resolves `projection_x_coordinate` / `projection_y_coordinate` automatically. Drop the explicit overrides so the detector is exercised in production; keep `source_crs` (the sinusoidal PROJ string) since it's not inferable.

- [ ] **Step 1: Edit `src/nhf_spatial_targets/aggregate/mod16a2.py`**

Remove `x_coord="x"` and `y_coord="y"` from the `ADAPTER` definition:

```python
ADAPTER = SourceAdapter(
    source_key="mod16a2_v061",
    output_name="mod16a2_agg.nc",
    variables=["ET_500m"],
    source_crs=MODIS_SINUSOIDAL_PROJ,
)
```

- [ ] **Step 2: Run all aggregate tests**

```
pixi run -e dev test -- tests/test_aggregate_driver.py tests/test_aggregate_driver_per_year.py tests/test_aggregate_mod10c1.py
```

Expected: all pass (no MOD16A2-specific test exists; this change is covered by the shared driver tests).

- [ ] **Step 3: Commit**

```
git add src/nhf_spatial_targets/aggregate/mod16a2.py
git commit -m "refactor: MOD16A2 adapter drops hard-coded x/y coord overrides"
```

---

## Task 10: Full test sweep, lint/format, and PR update

**Files:**
- No code changes; verification + housekeeping.

- [ ] **Step 1: Run the full dev test suite**

```
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
```

Expected: format clean, lint clean, all unit tests pass. Fix any issues in place and stage them.

- [ ] **Step 2: Smoke-test on a real project (operator step; skip if no project available)**

If `$NHF_PROJECT_DIR` points to a real project with MOD10C1 or MOD16A2 fetched to the datastore:

```
pixi run nhf-targets agg mod10c1 --project-dir "$NHF_PROJECT_DIR"
```

Confirm:

- `data/aggregated/_by_year/mod10c1_v061_<year>_agg.nc` exists for every year covered.
- `data/aggregated/mod10c1_agg.nc` exists and opens with xarray; has `sca`, `ci`, `valid_area_fraction`.
- A re-run prints "intermediate exists, skipping" for every year and does not re-invoke `WeightGen` / `AggGen`.

If no project is available, record "smoke test deferred — operator to run before merge" in the PR description.

- [ ] **Step 3: Push and update PR**

```
git push
gh pr view --web  # or: gh pr edit <num> --body "..."
```

Update the PR description with:

- Link to the design spec (`docs/superpowers/specs/2026-04-15-per-year-streaming-aggregation-design.md`).
- Summary of the refactor (per-year streaming, CF coord detection, restartable intermediates).
- Note on which smoke tests were run (or deferred).

- [ ] **Step 4: Final commit for any fmt/lint fixups**

```
git status
# If anything is staged:
git commit -m "chore: format / lint cleanup"
git push
```

---

## Notes for the implementer

- **Hook-aware variable check** (Task 8 note): if this is missed, the MOD10C1 aggregation raises before ever opening a year because `sca` / `ci` / `valid_mask` are not in the raw NC. Test `test_aggregate_source_raises_when_variable_missing` covers the non-hook path; the MOD10C1 adapter-wiring test exercises the hook path implicitly by not raising at construction.
- **CF attribute expectations**: the project's consolidated NCs are written CF-1.6 with `axis` and `standard_name` attrs on every coord (see `feedback_cf_netcdf_patterns.md`). The detector fails loudly if those are missing — that's intentional; don't add a bare-name fallback.
- **Period string format**: `aggregate_year` uses `f"{year}-01-01"` and `f"{year}-12-31"`. gdptools `UserCatData.period` accepts date strings; the year range is inclusive on both ends per gdptools convention. If a year is partially populated (e.g. source ends mid-year), gdptools clips to available data — no special casing needed.
- **No changes to `fetch/`**: this refactor is purely on the aggregation side.
- **Manifest schema unchanged**: existing downstream consumers (target builders, audit scripts) keep working.
