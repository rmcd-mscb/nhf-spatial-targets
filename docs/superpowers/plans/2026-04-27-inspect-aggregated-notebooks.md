# Inspect-Aggregated Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build five HRU-aggregated inspection notebooks (RUN/AET/RCH/SOM/SCA) under `notebooks/inspect_aggregated/`, plus a sibling `_helpers.py` module and unit tests, mirroring the existing `inspect_consolidated_*` notebooks at the HRU level.

**Architecture:** Five fully-parallel notebooks share a sibling `_helpers.py` (path-discovery, fabric I/O, choropleth plotting, area-weighted means, time-window slicing, point-to-HRU lookup, catalog-units lookup, figure save). Each notebook follows the same 15-cell template with target-specific unit conversions and source lists. Helpers are pure-Python and unit-tested; notebooks are validated end-to-end against a real project (`gfv2-spatial-targets` on caldera).

**Tech Stack:** Python 3.11+, `xarray`, `geopandas`, `matplotlib`, `pandas`, `numpy`, `pyyaml`, `pytest`. Jupyter notebooks via VSCode. Pixi for environment management.

**Spec:** `docs/superpowers/specs/2026-04-27-inspect-aggregated-notebooks-design.md`

---

## Task 1: Set up feature branch and folder layout

**Files:**
- Create: `notebooks/inspect_aggregated/` (directory)
- Modify: `.gitignore`

- [ ] **Step 1: Create feature branch from main**

```bash
git checkout main
git pull
git checkout -b feature/70-inspect-aggregated-notebooks
```

- [ ] **Step 2: Create the new folder**

```bash
mkdir -p notebooks/inspect_aggregated
```

- [ ] **Step 3: Add docs/figures/ to .gitignore**

Append this line to `.gitignore`:

```
# Inspection notebook figures (saved when SAVE_FIGURES=True)
docs/figures/
```

- [ ] **Step 4: Verify .gitignore change**

Run: `git diff .gitignore`
Expected: one new line under a comment for inspection figures.

- [ ] **Step 5: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore docs/figures/ for inspection notebook output (#70)"
```

---

## Task 2: Scaffold `_helpers.py` with module-level configuration

**Files:**
- Create: `notebooks/inspect_aggregated/_helpers.py`
- Create: `tests/test_inspect_helpers.py`

- [ ] **Step 1: Write failing test for module import + SAVE_FIGURES default**

Create `tests/test_inspect_helpers.py`:

```python
"""Unit tests for notebooks/inspect_aggregated/_helpers.py.

The helper module lives outside the package, so we load it via importlib
rather than a regular import.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPERS_PATH = REPO_ROOT / "notebooks" / "inspect_aggregated" / "_helpers.py"


@pytest.fixture(scope="session")
def helpers():
    spec = importlib.util.spec_from_file_location(
        "inspect_aggregated_helpers", HELPERS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_loads_with_default_save_figures_off(helpers):
    assert helpers.SAVE_FIGURES is False
    assert helpers.FIGURES_DIR == Path("docs/figures/inspect_aggregated/")
```

- [ ] **Step 2: Run test to verify failure**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v`
Expected: FAIL — `_helpers.py` does not exist.

- [ ] **Step 3: Create the helpers module skeleton**

Create `notebooks/inspect_aggregated/_helpers.py`:

```python
"""Shared helpers for the inspect_aggregated_*.ipynb notebooks.

This module is a sibling of the notebooks (not packaged into
nhf_spatial_targets). It holds path discovery, fabric I/O, HRU
choropleth plotting, area-weighted means, time-window slicing,
representative-point lookup, catalog-units lookup, and a save-figure
helper used to populate docs/figures/inspect_aggregated/ for downstream
slide / documentation work.

Notebooks import via:

    import sys
    sys.path.insert(0, str(Path.cwd()))   # if needed
    from _helpers import (
        load_project_paths, load_fabric, discover_aggregated, ...
    )

Or, since Jupyter puts the notebook's directory on sys.path, simply:

    from _helpers import load_project_paths, ...
"""

from __future__ import annotations

from pathlib import Path

SAVE_FIGURES: bool = False
FIGURES_DIR: Path = Path("docs/figures/inspect_aggregated/")
```

- [ ] **Step 4: Run test to verify pass**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/inspect_aggregated/_helpers.py tests/test_inspect_helpers.py
git commit -m "feat(notebooks): scaffold inspect_aggregated _helpers module (#70)"
```

---

## Task 3: `unit_from_catalog`

**Files:**
- Modify: `notebooks/inspect_aggregated/_helpers.py`
- Modify: `tests/test_inspect_helpers.py`

The catalog has two shapes for `variables` blocks:

1. **Dict list:** `variables: [{name: ro, cf_units: "m", ...}, ...]` (most sources).
2. **Flat list:** `variables: [actual_et]` with the unit at the source level (`units: mm/month`).

`unit_from_catalog(source_key, var)` returns:

- The matched dict's `cf_units` or `units` when the variable block is a list of dicts and the var name matches.
- The source-level `units` field when the variable block is a flat list and the var name is in it.
- Raises `KeyError` otherwise.

- [ ] **Step 1: Add failing tests**

Append to `tests/test_inspect_helpers.py`:

```python
def test_unit_from_catalog_dict_variables(helpers):
    # gldas_noah_v21_monthly has dict-form variables with cf_units
    units = helpers.unit_from_catalog("gldas_noah_v21_monthly", "Qs_acc")
    assert units == "kg m-2"


def test_unit_from_catalog_flat_variables(helpers):
    # ssebop has a flat variables list and a source-level units field
    units = helpers.unit_from_catalog("ssebop", "actual_et")
    assert units == "mm/month"


def test_unit_from_catalog_unknown_variable_raises(helpers):
    with pytest.raises(KeyError):
        helpers.unit_from_catalog("ssebop", "nonexistent_var")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py::test_unit_from_catalog_dict_variables tests/test_inspect_helpers.py::test_unit_from_catalog_flat_variables tests/test_inspect_helpers.py::test_unit_from_catalog_unknown_variable_raises -v`
Expected: FAIL — `unit_from_catalog` not defined.

- [ ] **Step 3: Implement `unit_from_catalog`**

Append to `notebooks/inspect_aggregated/_helpers.py`:

```python
from nhf_spatial_targets import catalog as cat


def unit_from_catalog(source_key: str, var: str) -> str:
    """Return units for ``var`` from ``catalog/sources.yml``.

    The catalog has two shapes for ``variables``: a list of dicts (each
    with ``name`` and ``cf_units`` / ``units``), or a flat list of names
    with the unit at the source level. Both are handled. Reading units
    from the catalog (rather than from on-disk attrs) is the convention
    enforced by ``docs/references/calibration-target-recipes.md`` lesson 9.
    """
    src = cat.source(source_key)
    variables = src.get("variables", [])
    for entry in variables:
        if isinstance(entry, dict):
            if entry.get("name") == var or entry.get("file_variable") == var:
                return entry.get("cf_units") or entry.get("units") or src["units"]
        elif entry == var:
            return src["units"]
    raise KeyError(
        f"Variable {var!r} not found in catalog entry for {source_key!r} "
        f"(available: {variables})"
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/inspect_aggregated/_helpers.py tests/test_inspect_helpers.py
git commit -m "feat(notebooks): unit_from_catalog helper for inspect_aggregated (#70)"
```

---

## Task 4: `select_month`

**Files:**
- Modify: `notebooks/inspect_aggregated/_helpers.py`
- Modify: `tests/test_inspect_helpers.py`

Cross-source time alignment: month-window slice, returns the first hit. Defends against the SOM lesson — `time.sel(method="nearest")` against a mid-month target silently picks the wrong calendar month for end-of-month-stamped sources (NCEP/NCAR).

- [ ] **Step 1: Add failing tests**

Append to `tests/test_inspect_helpers.py`:

```python
import numpy as np
import pandas as pd
import xarray as xr


def test_select_month_start_of_month(helpers):
    # GLDAS/NLDAS convention: start-of-month
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    da = xr.DataArray(np.arange(12), coords={"time": times}, dims=["time"])
    result = helpers.select_month(da, 2000, 3)
    assert pd.Timestamp(result.time.values) == pd.Timestamp("2000-03-01")
    assert int(result.values) == 2  # March = index 2


def test_select_month_end_of_month(helpers):
    # NCEP/NCAR convention: end-of-month — "nearest" to mid-month silently
    # picks the wrong calendar month here; select_month gets it right.
    times = pd.date_range("2000-01-31", periods=12, freq="ME")
    da = xr.DataArray(np.arange(12), coords={"time": times}, dims=["time"])
    result = helpers.select_month(da, 2000, 3)
    assert pd.Timestamp(result.time.values) == pd.Timestamp("2000-03-31")
    assert int(result.values) == 2


def test_select_month_mid_month(helpers):
    # MERRA-2 convention: mid-month
    times = pd.DatetimeIndex(
        [pd.Timestamp(year=2000, month=m, day=15) for m in range(1, 13)]
    )
    da = xr.DataArray(np.arange(12), coords={"time": times}, dims=["time"])
    result = helpers.select_month(da, 2000, 3)
    assert pd.Timestamp(result.time.values) == pd.Timestamp("2000-03-15")


def test_select_month_no_data_raises(helpers):
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    da = xr.DataArray(np.arange(12), coords={"time": times}, dims=["time"])
    with pytest.raises(IndexError):
        helpers.select_month(da, 2099, 1)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v -k select_month`
Expected: FAIL — `select_month` not defined.

- [ ] **Step 3: Implement `select_month`**

Append to `notebooks/inspect_aggregated/_helpers.py`:

```python
import pandas as pd
import xarray as xr


def select_month(da: xr.DataArray, year: int, month: int) -> xr.DataArray:
    """Select the first timestep in the given calendar month.

    Slices ``da`` to the calendar-month window ``[YYYY-MM-01, YYYY-MM-end]``
    and returns the first hit. Robust to start-of-month / end-of-month /
    mid-month timestamping conventions; the canonical SOM pattern from
    ``docs/references/calibration-target-recipes.md`` lesson 2.

    Raises ``IndexError`` if the window contains no timesteps.
    """
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    sliced = da.sel(time=slice(start, end))
    if sliced.sizes.get("time", 0) == 0:
        raise IndexError(
            f"No timesteps in {da.name or 'array'} between {start.date()} "
            f"and {end.date()}"
        )
    return sliced.isel(time=0)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/inspect_aggregated/_helpers.py tests/test_inspect_helpers.py
git commit -m "feat(notebooks): select_month helper for inspect_aggregated (#70)"
```

---

## Task 5: `discover_aggregated`

**Files:**
- Modify: `notebooks/inspect_aggregated/_helpers.py`
- Modify: `tests/test_inspect_helpers.py`

Glob `<project>/data/aggregated/<source_key>/<source_key>_*_agg.nc` and return sorted paths, or `None` if absent / empty.

- [ ] **Step 1: Add failing tests**

Append to `tests/test_inspect_helpers.py`:

```python
def test_discover_aggregated_returns_sorted_paths(helpers, tmp_path):
    src = "era5_land"
    agg_dir = tmp_path / "data" / "aggregated" / src
    agg_dir.mkdir(parents=True)
    (agg_dir / f"{src}_2002_agg.nc").touch()
    (agg_dir / f"{src}_2000_agg.nc").touch()
    (agg_dir / f"{src}_2001_agg.nc").touch()

    paths = helpers.discover_aggregated(tmp_path, src)
    assert paths is not None
    assert [p.name for p in paths] == [
        f"{src}_2000_agg.nc",
        f"{src}_2001_agg.nc",
        f"{src}_2002_agg.nc",
    ]


def test_discover_aggregated_returns_none_when_dir_missing(helpers, tmp_path):
    assert helpers.discover_aggregated(tmp_path, "no_such_source") is None


def test_discover_aggregated_returns_none_when_dir_empty(helpers, tmp_path):
    (tmp_path / "data" / "aggregated" / "src1").mkdir(parents=True)
    assert helpers.discover_aggregated(tmp_path, "src1") is None
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v -k discover_aggregated`
Expected: FAIL.

- [ ] **Step 3: Implement `discover_aggregated`**

Append to `notebooks/inspect_aggregated/_helpers.py`:

```python
def discover_aggregated(project_dir: Path, source_key: str) -> list[Path] | None:
    """Return sorted per-year aggregated NC paths, or ``None`` if absent.

    Globs ``<project>/data/aggregated/<source_key>/<source_key>_*_agg.nc``.
    Returns ``None`` when the directory is missing or empty so callers
    can print a single "skip with reason" line and continue.
    """
    agg_dir = Path(project_dir) / "data" / "aggregated" / source_key
    if not agg_dir.is_dir():
        return None
    paths = sorted(agg_dir.glob(f"{source_key}_*_agg.nc"))
    return paths if paths else None
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/inspect_aggregated/_helpers.py tests/test_inspect_helpers.py
git commit -m "feat(notebooks): discover_aggregated helper for inspect_aggregated (#70)"
```

---

## Task 6: `load_project_paths` and `load_fabric`

**Files:**
- Modify: `notebooks/inspect_aggregated/_helpers.py`
- Modify: `tests/test_inspect_helpers.py`

`load_project_paths` reads `<project>/config.yml`, returning a (project_dir, datastore_dir, fabric_cfg) triple. `load_fabric` is integration-y (needs a real GeoPackage); we test only `load_project_paths` and exercise `load_fabric` end-to-end during notebook validation.

- [ ] **Step 1: Add failing tests**

Append to `tests/test_inspect_helpers.py`:

```python
import yaml


def test_load_project_paths_reads_config_yml(helpers, tmp_path):
    cfg = {
        "datastore": "/mnt/d/nhf-datastore",
        "fabric": {
            "path": "/mnt/d/fabric/gfv2.gpkg",
            "id_col": "nhm_id",
            "crs": "EPSG:4326",
        },
    }
    (tmp_path / "config.yml").write_text(yaml.safe_dump(cfg))

    project_dir, datastore_dir, fabric_cfg = helpers.load_project_paths(tmp_path)
    assert project_dir == tmp_path
    assert datastore_dir == Path("/mnt/d/nhf-datastore")
    assert fabric_cfg["path"] == "/mnt/d/fabric/gfv2.gpkg"
    assert fabric_cfg["id_col"] == "nhm_id"


def test_load_project_paths_missing_config_raises(helpers, tmp_path):
    with pytest.raises(FileNotFoundError):
        helpers.load_project_paths(tmp_path)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v -k load_project_paths`
Expected: FAIL.

- [ ] **Step 3: Implement `load_project_paths` and `load_fabric`**

Append to `notebooks/inspect_aggregated/_helpers.py`:

```python
import geopandas as gpd
import yaml

DEFAULT_CALDERA_PROJECT = Path(
    "/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets"
)


def load_project_paths(
    project_dir: Path | None = None,
) -> tuple[Path, Path, dict]:
    """Read ``<project>/config.yml`` and return ``(project_dir, datastore_dir, fabric_cfg)``.

    ``fabric_cfg`` is the ``fabric`` sub-block from ``config.yml`` (keys
    typically include ``path``, ``id_col``, ``crs``, ``buffer_deg``).
    Defaults ``project_dir`` to the caldera ``gfv2-spatial-targets`` project
    when called with ``None``.
    """
    project_dir = Path(project_dir) if project_dir is not None else DEFAULT_CALDERA_PROJECT
    cfg_path = project_dir / "config.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"config.yml not found at {cfg_path}. "
            f"Edit PROJECT_DIR at the top of the notebook to point at "
            f"a real project directory."
        )
    cfg = yaml.safe_load(cfg_path.read_text())
    datastore_dir = Path(cfg["datastore"])
    fabric_cfg = dict(cfg["fabric"])
    return project_dir, datastore_dir, fabric_cfg


def load_fabric(fabric_cfg: dict) -> gpd.GeoDataFrame:
    """Read the HRU fabric file and index by ``fabric_cfg['id_col']``.

    Kept in EPSG:4326 for plotting; downstream area calculations
    re-project to EPSG:5070 (CONUS Albers) so we don't pay that cost
    on every map.
    """
    gdf = gpd.read_file(fabric_cfg["path"])
    gdf = gdf.set_index(fabric_cfg["id_col"])
    return gdf
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/inspect_aggregated/_helpers.py tests/test_inspect_helpers.py
git commit -m "feat(notebooks): load_project_paths and load_fabric (#70)"
```

---

## Task 7: `area_weighted_mean`, `nan_hru_count`

**Files:**
- Modify: `notebooks/inspect_aggregated/_helpers.py`
- Modify: `tests/test_inspect_helpers.py`

`area_weighted_mean` reprojects fabric to EPSG:5070 to get areas in m², then Σ(v · A) / Σ(A) over non-NaN HRUs. `nan_hru_count` is a one-liner.

- [ ] **Step 1: Add failing tests**

Append to `tests/test_inspect_helpers.py`:

```python
from shapely.geometry import box


def _three_hru_fabric():
    """Three square HRUs in EPSG:4326 with known relative areas."""
    geoms = [
        box(-100, 40, -99, 41),  # ~111 x 85 km
        box(-99, 40, -98, 41),
        box(-98, 40, -97, 41),
    ]
    return gpd.GeoDataFrame(
        {"hru_id": [1, 2, 3]},
        geometry=geoms,
        crs="EPSG:4326",
    ).set_index("hru_id")


def test_area_weighted_mean_equal_areas(helpers):
    import geopandas as gpd

    fabric = _three_hru_fabric()
    values = pd.Series([10.0, 20.0, 30.0], index=[1, 2, 3])
    result = helpers.area_weighted_mean(values, fabric)
    # Areas in EPSG:5070 are very nearly equal here; expect ~20.0
    assert abs(result - 20.0) < 0.5


def test_area_weighted_mean_skips_nan(helpers):
    import geopandas as gpd

    fabric = _three_hru_fabric()
    values = pd.Series([10.0, np.nan, 30.0], index=[1, 2, 3])
    result = helpers.area_weighted_mean(values, fabric)
    # Should average HRUs 1 and 3 only
    assert abs(result - 20.0) < 0.5


def test_nan_hru_count(helpers):
    values = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
    assert helpers.nan_hru_count(values) == 2
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v -k "area_weighted_mean or nan_hru_count"`
Expected: FAIL.

- [ ] **Step 3: Implement both helpers**

Append to `notebooks/inspect_aggregated/_helpers.py`:

```python
import numpy as np

ALBERS_CRS = "EPSG:5070"  # matches the aggregator's WEIGHT_GEN_CRS


def area_weighted_mean(values: pd.Series, fabric_gdf: gpd.GeoDataFrame) -> float:
    """Compute Σ(v · A) / Σ(A) using fabric area in EPSG:5070.

    Skips NaN values (and their corresponding areas). Aligns on the
    fabric's index — ``values`` must be indexed by HRU id.
    """
    aligned = values.reindex(fabric_gdf.index)
    areas = fabric_gdf.to_crs(ALBERS_CRS).area
    mask = ~aligned.isna()
    if not mask.any():
        return float("nan")
    return float((aligned[mask] * areas[mask]).sum() / areas[mask].sum())


def nan_hru_count(values: pd.Series) -> int:
    """Number of NaN HRUs in ``values``."""
    return int(values.isna().sum())
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/inspect_aggregated/_helpers.py tests/test_inspect_helpers.py
git commit -m "feat(notebooks): area_weighted_mean and nan_hru_count (#70)"
```

---

## Task 8: `lookup_hrus_by_points`

**Files:**
- Modify: `notebooks/inspect_aggregated/_helpers.py`
- Modify: `tests/test_inspect_helpers.py`

Spatial-join points to containing HRU. Each notebook declares `REPRESENTATIVE_POINTS = {regime_name: (lon, lat), ...}` near the top; this helper resolves regime names to HRU ids.

- [ ] **Step 1: Add failing tests**

Append to `tests/test_inspect_helpers.py`:

```python
def test_lookup_hrus_by_points_resolves_inside(helpers):
    fabric = _three_hru_fabric()
    points = {
        "left": (-99.5, 40.5),    # inside HRU 1
        "middle": (-98.5, 40.5),  # inside HRU 2
        "right": (-97.5, 40.5),   # inside HRU 3
    }
    result = helpers.lookup_hrus_by_points(fabric, points)
    assert result == {"left": 1, "middle": 2, "right": 3}


def test_lookup_hrus_by_points_outside_raises(helpers):
    fabric = _three_hru_fabric()
    points = {"in_ocean": (-50.0, 40.5)}
    with pytest.raises(ValueError):
        helpers.lookup_hrus_by_points(fabric, points)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v -k lookup_hrus_by_points`
Expected: FAIL.

- [ ] **Step 3: Implement `lookup_hrus_by_points`**

Append to `notebooks/inspect_aggregated/_helpers.py`:

```python
from shapely.geometry import Point


def lookup_hrus_by_points(
    fabric_gdf: gpd.GeoDataFrame,
    points: dict[str, tuple[float, float]],
) -> dict[str, object]:
    """Resolve ``{label: (lon, lat)}`` to ``{label: hru_id}`` via sjoin.

    Raises ``ValueError`` if any point falls outside the fabric — better
    to fail early than silently drop a regime from the time-series cell.
    """
    pts = gpd.GeoDataFrame(
        {"label": list(points.keys())},
        geometry=[Point(lon, lat) for lon, lat in points.values()],
        crs="EPSG:4326",
    )
    fabric_for_join = fabric_gdf.reset_index()
    id_col = fabric_gdf.index.name
    joined = gpd.sjoin(pts, fabric_for_join, predicate="within", how="left")
    missing = joined[joined[id_col].isna()]["label"].tolist()
    if missing:
        raise ValueError(
            f"REPRESENTATIVE_POINTS lie outside the fabric: {missing}. "
            f"Pick coordinates inside the fabric's CONUS extent."
        )
    return dict(zip(joined["label"], joined[id_col].tolist()))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/inspect_aggregated/_helpers.py tests/test_inspect_helpers.py
git commit -m "feat(notebooks): lookup_hrus_by_points helper (#70)"
```

---

## Task 9: `save_figure`

**Files:**
- Modify: `notebooks/inspect_aggregated/_helpers.py`
- Modify: `tests/test_inspect_helpers.py`

Module-level toggle. No-op by default; writes PNG to `FIGURES_DIR / f"{name}.png"` when `SAVE_FIGURES=True`.

- [ ] **Step 1: Add failing tests**

Append to `tests/test_inspect_helpers.py`:

```python
def test_save_figure_no_op_when_disabled(helpers, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(helpers, "SAVE_FIGURES", False)
    monkeypatch.setattr(helpers, "FIGURES_DIR", tmp_path / "figures")
    fig = plt.figure()
    helpers.save_figure(fig, "test_disabled")
    plt.close(fig)
    assert not (tmp_path / "figures").exists()


def test_save_figure_writes_png_when_enabled(helpers, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(helpers, "SAVE_FIGURES", True)
    monkeypatch.setattr(helpers, "FIGURES_DIR", tmp_path / "figures")
    fig = plt.figure()
    helpers.save_figure(fig, "test_enabled")
    plt.close(fig)
    assert (tmp_path / "figures" / "test_enabled.png").exists()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v -k save_figure`
Expected: FAIL.

- [ ] **Step 3: Implement `save_figure`**

Append to `notebooks/inspect_aggregated/_helpers.py`:

```python
def save_figure(fig, name: str) -> None:
    """Write ``fig`` to ``FIGURES_DIR/<name>.png`` iff ``SAVE_FIGURES``.

    No-op when ``SAVE_FIGURES`` is ``False`` (the default). Notebooks
    enable saving by setting ``_helpers.SAVE_FIGURES = True`` near the
    top before any plotting cell runs.
    """
    if not SAVE_FIGURES:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/inspect_aggregated/_helpers.py tests/test_inspect_helpers.py
git commit -m "feat(notebooks): save_figure helper for inspect_aggregated (#70)"
```

---

## Task 10: Plotting helpers `plot_hru_choropleth`, `plot_nan_hrus`, `open_year`, `open_year_range`

**Files:**
- Modify: `notebooks/inspect_aggregated/_helpers.py`

These are not unit-tested (plotting visual correctness is covered by manual end-to-end notebook validation; `open_year` / `open_year_range` are thin xarray wrappers exercised at notebook load).

- [ ] **Step 1: Implement the four helpers**

Append to `notebooks/inspect_aggregated/_helpers.py`:

```python
import matplotlib.pyplot as plt
import xarray as xr


def open_year(project_dir: Path, source_key: str, year: int) -> xr.Dataset:
    """Open a single per-year aggregated NC and detach from the file.

    The Dataset is loaded into memory and the underlying file handle is
    closed before return — the SOM/MERRA-2 family of xarray bugs around
    file handles staying open is covered by ``feedback_rioxarray_close.md``.
    """
    paths = discover_aggregated(project_dir, source_key)
    if paths is None:
        raise FileNotFoundError(
            f"No aggregated NCs for {source_key} in {project_dir}"
        )
    target = next((p for p in paths if f"_{year}_agg.nc" in p.name), None)
    if target is None:
        raise FileNotFoundError(
            f"No {source_key}_{year}_agg.nc in {paths[0].parent}"
        )
    with xr.open_dataset(target) as ds:
        loaded = ds.load()
    return loaded


def open_year_range(
    project_dir: Path, source_key: str, years: range
) -> xr.Dataset:
    """Open a contiguous range of per-year aggregated NCs (lazy).

    Caller is responsible for ``.sel(hru=[...])`` and ``.load()`` to bound
    memory, then ``.close()`` afterwards.
    """
    paths = discover_aggregated(project_dir, source_key)
    if paths is None:
        raise FileNotFoundError(
            f"No aggregated NCs for {source_key} in {project_dir}"
        )
    wanted = [
        p for p in paths
        if any(f"_{y}_agg.nc" in p.name for y in years)
    ]
    if not wanted:
        raise FileNotFoundError(
            f"None of years {list(years)} present in {paths[0].parent}"
        )
    return xr.open_mfdataset(wanted, combine="by_coords")


def plot_hru_choropleth(
    ax,
    fabric_gdf: gpd.GeoDataFrame,
    values: pd.Series,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "YlGnBu",
    title: str = "",
    units: str = "",
    nan_color: str = "lightgrey",
) -> None:
    """Render an HRU-level choropleth with NaN HRUs in ``nan_color``.

    Joins ``values`` (indexed by HRU id) onto ``fabric_gdf``. NaN HRUs
    are plotted first in ``nan_color`` so coverage gaps are visually
    obvious; finite-value HRUs are plotted on top.
    """
    plot_gdf = fabric_gdf.copy()
    plot_gdf["value"] = values.reindex(plot_gdf.index)

    nan_mask = plot_gdf["value"].isna()
    if nan_mask.any():
        plot_gdf[nan_mask].plot(ax=ax, color=nan_color, edgecolor="none")

    plot_gdf[~nan_mask].plot(
        ax=ax,
        column="value",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        legend=True,
        legend_kwds={"label": units, "shrink": 0.6},
        edgecolor="none",
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def plot_nan_hrus(
    ax,
    fabric_gdf: gpd.GeoDataFrame,
    values: pd.Series,
    *,
    title: str = "",
) -> None:
    """Boolean coverage map: NaN HRUs in red, finite HRUs in light grey."""
    plot_gdf = fabric_gdf.copy()
    plot_gdf["is_nan"] = values.reindex(plot_gdf.index).isna()
    plot_gdf[~plot_gdf["is_nan"]].plot(ax=ax, color="lightgrey", edgecolor="none")
    plot_gdf[plot_gdf["is_nan"]].plot(ax=ax, color="crimson", edgecolor="none")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
```

- [ ] **Step 2: Run all helper tests to confirm nothing regressed**

Run: `pixi run -e dev pytest tests/test_inspect_helpers.py -v`
Expected: PASS (all tests added in Tasks 2-9 still pass).

- [ ] **Step 3: Lint and format**

Run: `pixi run -e dev fmt && pixi run -e dev lint`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/inspect_aggregated/_helpers.py
git commit -m "feat(notebooks): plotting and xarray-open helpers (#70)"
```

---

## Task 11: Build `inspect_aggregated_runoff.ipynb` (the template)

**Files:**
- Create: `notebooks/inspect_aggregated/inspect_aggregated_runoff.ipynb`

This is the canonical notebook. Build it cell-by-cell using `jupyter notebook` or by hand-editing the JSON. The cell contents below are the source of truth; after authoring, regenerate cell IDs as UUIDs (via `nbstripout` or by re-saving in Jupyter — both produce fresh ids).

The notebook has 15 cells. Use this exact ordering and content.

- [ ] **Step 1: Create the notebook with these cells**

**Cell 1 — markdown — Intro:**

```markdown
# Inspect Aggregated Runoff Datasets

HRU-level inspection of the two runoff sources, aggregated to the fabric. Mirrors `inspect_consolidated_runoff.ipynb` at the HRU level.

Sources (see `catalog/variables.yml` → `runoff`):

- ERA5-Land total runoff (`ro`, m/month)
- GLDAS-2.1 NOAH total runoff (`runoff_total = Qs_acc + Qsb_acc`, kg m⁻² stored as the mean of 3-hourly accumulations; multiply by `8 × days_in_month` to get mm/month)

See `docs/references/calibration-target-recipes.md` §1 for the canonical-unit conversions and combination rule.
```

**Cell 2 — markdown — Per-target conventions in this notebook:**

```markdown
## Per-target conventions in this notebook

- ERA5-Land `ro` × 1000 → mm/month before normalised comparison.
- GLDAS NOAH `runoff_total` × 8 × `days_in_month` → mm/month (recipes §1, lesson 3). The aggregated NC still carries `kg m⁻²` units; the conversion happens in this notebook before plotting.
- Reference source for the colour scale and footprint-clip: ERA5-Land.
- ERA5-Land timestamps are end-of-month; GLDAS at start-of-month. We use `select_month` rather than `time.sel(method="nearest")` for cross-source alignment (recipes lesson 2).
- Units are read from `catalog/sources.yml` via `unit_from_catalog`, not from the on-disk NC `attrs` (recipes lesson 9).
```

**Cell 3 — code — Setup:**

```python
import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from _helpers import (
    area_weighted_mean,
    discover_aggregated,
    load_fabric,
    load_project_paths,
    lookup_hrus_by_points,
    nan_hru_count,
    open_year,
    open_year_range,
    plot_hru_choropleth,
    plot_nan_hrus,
    save_figure,
    select_month,
    unit_from_catalog,
)

# Edit me to point at a real project directory:
PROJECT_DIR = Path(
    "/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets"
)

# Set True (and re-run) to populate docs/figures/inspect_aggregated/<run>_*.png
# import _helpers
# _helpers.SAVE_FIGURES = True

project_dir, datastore_dir, fabric_cfg = load_project_paths(PROJECT_DIR)
fabric = load_fabric(fabric_cfg)

TARGET = "runoff"
TARGET_TIME = "2000-01-15"
TARGET_YEAR = 2000
TARGET_MONTH = 1
TIME_SERIES_YEARS = range(2000, 2011)
REPRESENTATIVE_POINTS = {
    "Olympic Peninsula (PNW mountains)": (-123.5, 47.8),
    "Iowa cropland (Midwest)": (-93.6, 42.0),
    "Phoenix metro (arid SW)": (-112.1, 33.4),
    "Southern Appalachians (Eastern forest)": (-83.5, 35.5),
}

SOURCES = {
    "era5_land": {"label": "ERA5-Land (ro)", "var": "ro"},
    "gldas_noah_v21_monthly": {"label": "GLDAS-2.1 NOAH (runoff_total)", "var": "runoff_total"},
}

print(f"Project: {project_dir}")
print(f"Datastore: {datastore_dir}")
print(f"Fabric: {fabric_cfg['path']} ({len(fabric)} HRUs)")
```

**Cell 4 — markdown — Load:**

```markdown
## Load aggregated datasets

Each source is opened from `<project>/data/aggregated/<source_key>/<source_key>_<TARGET_YEAR>_agg.nc`. Sources whose aggregation has not yet been produced are skipped with a clear message; downstream cells iterate over the loaded set so missing sources drop out naturally.
```

**Cell 5 — code — Load per source:**

```python
opened = {}
for source_key, info in SOURCES.items():
    paths = discover_aggregated(project_dir, source_key)
    if paths is None:
        print(f"SKIP {info['label']}: no aggregated NCs at "
              f"{project_dir}/data/aggregated/{source_key}/")
        continue
    ds = open_year(project_dir, source_key, TARGET_YEAR)
    info["units"] = unit_from_catalog(source_key, info["var"])
    opened[source_key] = (ds, info)
    values = ds[info["var"]].isel(time=0).to_pandas()
    print(
        f"{info['label']}: vars={list(ds.data_vars)} | "
        f"time={ds.time.values[0]}..{ds.time.values[-1]} | "
        f"HRUs={ds.sizes.get('nhm_id', ds.sizes.get('hru_id', 'unknown'))} | "
        f"NaN HRUs (t=0): {nan_hru_count(values)} | "
        f"catalog units: {info['units']}"
    )
```

**Cell 6 — code — Dataset reps:**

```python
for source_key, (ds, info) in opened.items():
    print(f"{'=' * 60}\n{info['label']}\n{'=' * 60}")
    display(ds)
```

**Cell 7 — code — Native-unit map at TARGET_TIME:**

```python
if not opened:
    print("No sources available; skipping native-unit map.")
else:
    fig, axes = plt.subplots(1, len(opened), figsize=(8 * len(opened), 6), squeeze=False)
    for idx, (source_key, (ds, info)) in enumerate(opened.items()):
        da = select_month(ds[info["var"]], TARGET_YEAR, TARGET_MONTH)
        values = da.to_pandas()
        plot_hru_choropleth(
            axes[0, idx],
            fabric,
            values,
            cmap="YlGnBu",
            title=f"{info['label']}\n{TARGET_TIME} | {info['units']}",
            units=info["units"],
        )
    fig.suptitle(f"Runoff — native units, {TARGET_TIME}", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, f"{TARGET}_native_units_map")
    plt.show()
```

**Cell 8 — markdown — Normalized comparison:**

```markdown
## Normalized comparison map (mm/month)

Convert both sources to **mm/month** and plot side by side on a shared colour scale derived from ERA5-Land (the reference source).

- ERA5-Land `ro` (m/month) × 1000 → mm/month.
- GLDAS NOAH `runoff_total` (kg m⁻², mean of 3-hourly accumulations) × `8 × days_in_month` → mm/month.

Both panels are on the HRU fabric, so the geographic footprint is identical; the colour scale is anchored on ERA5-Land's distribution to avoid GLDAS's tropical maxima dominating it.
```

**Cell 9 — code — Normalized comparison map:**

```python
def _to_mm_per_month(da: xr.DataArray, source_key: str) -> xr.DataArray:
    if source_key == "era5_land":
        return da * 1000.0
    if source_key == "gldas_noah_v21_monthly":
        ts = pd.Timestamp(da.time.values)
        days = calendar.monthrange(ts.year, ts.month)[1]
        return da * 8.0 * days
    raise ValueError(f"No mm/month conversion for {source_key}")


if opened:
    converted = {}
    for source_key, (ds, info) in opened.items():
        da = select_month(ds[info["var"]], TARGET_YEAR, TARGET_MONTH)
        converted[source_key] = _to_mm_per_month(da, source_key).to_pandas()

    ref_key = "era5_land"
    if ref_key in converted:
        ref_vals = converted[ref_key].dropna().values
        vmin, vmax = float(np.percentile(ref_vals, 2)), float(np.percentile(ref_vals, 98))
    else:
        all_vals = np.concatenate([s.dropna().values for s in converted.values()])
        vmin, vmax = float(np.percentile(all_vals, 2)), float(np.percentile(all_vals, 98))

    fig, axes = plt.subplots(1, len(converted), figsize=(8 * len(converted), 6), squeeze=False)
    for idx, (source_key, values) in enumerate(converted.items()):
        plot_hru_choropleth(
            axes[0, idx],
            fabric,
            values,
            vmin=vmin,
            vmax=vmax,
            cmap="YlGnBu",
            title=f"{SOURCES[source_key]['label']}\n{TARGET_TIME} | mm/month",
            units="mm/month",
        )
    fig.suptitle(
        f"Runoff — mm/month, colour scale from ERA5-Land — {TARGET_TIME}",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, f"{TARGET}_normalized_comparison")
    plt.show()
```

**Cell 10 — code — Cross-source HRU-level histograms:**

```python
if opened:
    fig, ax = plt.subplots(figsize=(10, 5))
    for source_key, values in converted.items():
        ax.hist(
            values.dropna(),
            bins=60,
            histtype="step",
            label=SOURCES[source_key]["label"],
            density=True,
            linewidth=2,
        )
    ax.set_xlabel("Runoff (mm/month)")
    ax.set_ylabel("Density")
    ax.set_title(f"Cross-source HRU distribution — {TARGET_TIME}")
    ax.legend()
    save_figure(fig, f"{TARGET}_histogram")
    plt.show()
```

**Cell 11 — code — Time series at representative HRUs:**

```python
if opened:
    rep_hrus = lookup_hrus_by_points(fabric, REPRESENTATIVE_POINTS)
    print("Representative HRUs:", rep_hrus)

    series_data = {}
    for source_key, info in SOURCES.items():
        if source_key not in opened:
            continue
        ds_range = open_year_range(project_dir, source_key, TIME_SERIES_YEARS)
        try:
            id_dim = "nhm_id" if "nhm_id" in ds_range.dims else "hru_id"
            sel = ds_range[info["var"]].sel({id_dim: list(rep_hrus.values())}).load()
        finally:
            ds_range.close()
        series_data[source_key] = sel

    def _convert_series(da, source_key):
        if source_key == "era5_land":
            return da * 1000.0
        if source_key == "gldas_noah_v21_monthly":
            days = pd.DatetimeIndex(da.time.values).days_in_month
            return da * 8.0 * xr.DataArray(days.values, coords={"time": da.time}, dims=["time"])
        return da

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    id_dim = "nhm_id" if "nhm_id" in next(iter(series_data.values())).dims else "hru_id"
    for ax, (label, hru_id) in zip(axes.flat, rep_hrus.items()):
        for source_key, da in series_data.items():
            da_hru = _convert_series(da.sel({id_dim: hru_id}), source_key)
            ax.plot(da_hru.time, da_hru.values, label=SOURCES[source_key]["label"])
        ax.set_title(f"{label} (HRU {hru_id})")
        ax.set_ylabel("mm/month")
        ax.legend(fontsize=8)
    fig.suptitle(f"Runoff at representative HRUs — {min(TIME_SERIES_YEARS)}–{max(TIME_SERIES_YEARS)}")
    plt.tight_layout()
    save_figure(fig, f"{TARGET}_time_series")
    plt.show()
```

**Cell 12 — code — Coverage diagnostic:**

```python
if opened:
    fig, axes = plt.subplots(1, len(opened), figsize=(8 * len(opened), 6), squeeze=False)
    for idx, (source_key, (ds, info)) in enumerate(opened.items()):
        da = select_month(ds[info["var"]], TARGET_YEAR, TARGET_MONTH)
        values = da.to_pandas()
        n_nan = nan_hru_count(values)
        print(f"{info['label']}: {n_nan} NaN HRUs ({100 * n_nan / len(fabric):.2f}%)")
        plot_nan_hrus(
            axes[0, idx],
            fabric,
            values,
            title=f"{info['label']}\nNaN HRUs (red) — {n_nan} of {len(fabric)}",
        )
    fig.suptitle(
        "Coverage gaps — these will be nearest-neighbor-filled in normalize/ before target combination",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, f"{TARGET}_coverage")
    plt.show()
```

**Cell 13 — code — Validation:**

```python
def _gridded_mean_runoff(source_key, info):
    """Compute the gridded CONUS-footprint mean for the validation cell."""
    if source_key == "era5_land":
        path = datastore_dir / "era5_land" / "monthly" / f"era5_land_monthly_{TARGET_YEAR}.nc"
    elif source_key == "gldas_noah_v21_monthly":
        path = datastore_dir / "gldas_noah_v21_monthly" / "gldas_noah_v21_monthly.nc"
    else:
        return None, f"unknown gridded path for {source_key}"
    if not path.exists():
        return None, f"missing consolidated NC at {path}"
    with xr.open_dataset(path) as ds:
        da = select_month(ds[info["var"]], TARGET_YEAR, TARGET_MONTH).load()
    return _to_mm_per_month(da, source_key).mean(skipna=True).item(), None


print(f"{'Source':<35} {'Aggregated':>12} {'Gridded':>12} {'Δ':>12} {'% diff':>8}")
print("-" * 85)
for source_key, (ds, info) in opened.items():
    da_agg = select_month(ds[info["var"]], TARGET_YEAR, TARGET_MONTH)
    converted_agg = _to_mm_per_month(da_agg, source_key).to_pandas()
    agg_mean = area_weighted_mean(converted_agg, fabric)
    gridded_mean, reason = _gridded_mean_runoff(source_key, info)
    if gridded_mean is None:
        print(f"{info['label']:<35} {agg_mean:>12.3f}  {'SKIP':>12} ({reason})")
        continue
    diff = agg_mean - gridded_mean
    pct = 100 * diff / gridded_mean if gridded_mean else float("nan")
    print(f"{info['label']:<35} {agg_mean:>12.3f} {gridded_mean:>12.3f} {diff:>12.3f} {pct:>7.2f}%")
```

**Cell 14 — markdown — Explanation:**

```markdown
## Why HRU-level patterns differ across sources

After area-weighted aggregation to HRUs, the HRU-level magnitudes track the gridded means within a few percent (validation cell above). Differences that remain are physical:

- **Different LSM physics.** ERA5-Land's H-TESSEL and GLDAS-2.1's NOAH-2.7 partition precipitation between infiltration, surface runoff, and sub-surface drainage with different parameterisations. Two LSMs forced with the same precipitation will still produce different runoff.
- **Different forcing.** ERA5-Land uses ERA5 precipitation; GLDAS-2.1 blends NOAA CPC gauge-corrected precipitation with model fields. Disagreement is largest in winter and over mountains.
- **Aggregation behaviour at coarse-grid sources.** GLDAS at ~25 km will yield more HRUs whose polygon does not have full source-grid coverage than ERA5-Land at ~9 km, so the NaN-HRU count is generally higher for GLDAS — visible in the coverage diagnostic above.

**Calibration target implication.** The runoff target uses both sources as a per-HRU per-month `min/max` range. Both must be in mm/month before the cfs conversion (`run.py`) — the GLDAS unit fix is critical, since omitting it makes GLDAS values 224–248× too small and degenerates the multi-source range.
```

**Cell 15 — code — Cleanup:**

```python
for source_key, (ds, _) in opened.items():
    ds.close()
opened.clear()
```

- [ ] **Step 2: Regenerate cell IDs as UUIDs**

In Jupyter (or via `nbstripout`), re-save the notebook to refresh cell IDs. Confirm: `python -c "import json; ids=[c.get('id') for c in json.load(open('notebooks/inspect_aggregated/inspect_aggregated_runoff.ipynb'))['cells']]; assert len(set(ids)) == len(ids), 'duplicate cell ids'"`
Expected: no error.

- [ ] **Step 3: Render the notebook in VSCode and confirm**

Manual checks:

1. All 15 cells visible — nothing silently dropped.
2. Cell 5 prints `NaN HRUs` per source.
3. Cell 7 native-unit map shows two panels with NaN HRUs in light grey.
4. Cell 9 normalized comparison shows two panels on a shared colour scale.
5. Cell 11 time series shows 2×2 grid with both sources overlaid per panel and a sensible seasonal cycle.
6. Cell 12 coverage diagnostic shows NaN HRUs concentrated near coastlines / source-grid edges.
7. Cell 13 validation reports `% diff` within a few percent for both sources.

- [ ] **Step 4: Commit**

```bash
git add notebooks/inspect_aggregated/inspect_aggregated_runoff.ipynb
git commit -m "feat(notebooks): inspect_aggregated_runoff template (#70)"
```

---

## Task 12: Build `inspect_aggregated_aet.ipynb`

**Files:**
- Create: `notebooks/inspect_aggregated/inspect_aggregated_aet.ipynb`

Copy `inspect_aggregated_runoff.ipynb` and apply the substitutions below. Cells not listed are unchanged from the runoff notebook except for trivial label/title text replacements (`Runoff` → `AET`).

- [ ] **Step 1: Copy the runoff notebook as a starting point**

```bash
cp notebooks/inspect_aggregated/inspect_aggregated_runoff.ipynb \
   notebooks/inspect_aggregated/inspect_aggregated_aet.ipynb
```

- [ ] **Step 2: Replace cell 1 (intro) with this content**

```markdown
# Inspect Aggregated AET Datasets

HRU-level inspection of the two AET sources. Mirrors `inspect_consolidated_aet.ipynb`.

Sources (see `catalog/variables.yml` → `aet`):

- SSEBop (`et`, mm/month) — accessed via USGS NHGF STAC; consolidated and aggregated.
- MOD16A2 v061 (`ET_500m`, kg m⁻² per 8-day composite — `scale_factor=0.1`) — global tiles consolidated to CONUS+ then aggregated.

See `docs/references/calibration-target-recipes.md` §2 for canonical-unit conversions and the open `scale_factor` question.
```

- [ ] **Step 3: Replace cell 2 (per-target conventions) with this content**

```markdown
## Per-target conventions in this notebook

- SSEBop `et` is native mm/month. No conversion before plotting or comparison.
- MOD16A2 `ET_500m` is kg m⁻² per 8-day composite, with `scale_factor=0.1` per the HDF spec. We apply the scale factor explicitly in cell 9 to defend against the missing-`decode_cf` case (recipes §2 — flagged 500× domain-mean offset in the consolidated notebook).
- Per-month aggregation of MOD16A2 8-day composites: weight each composite by `(days of overlap with target month) / 8`, then sum weighted contributions for all composites that intersect the month. **For the inspection notebook we use the simpler approximation: take the composite whose start lies in the target month and scale to mm/month.** This is good enough for spot-checking; the target builder will do the proper overlap weighting.
- Reference source: SSEBop.
```

- [ ] **Step 4: Replace the `SOURCES` declaration in cell 3 with**

```python
TARGET = "aet"
TARGET_TIME = "2000-01-15"
TARGET_YEAR = 2000
TARGET_MONTH = 1
TIME_SERIES_YEARS = range(2000, 2011)
REPRESENTATIVE_POINTS = {
    "Olympic Peninsula (PNW mountains)": (-123.5, 47.8),
    "Iowa cropland (Midwest)": (-93.6, 42.0),
    "Phoenix metro (arid SW)": (-112.1, 33.4),
    "Southern Appalachians (Eastern forest)": (-83.5, 35.5),
}
SOURCES = {
    "ssebop": {"label": "SSEBop (et)", "var": "et"},
    "mod16a2_v061": {"label": "MOD16A2 v061 (ET_500m)", "var": "ET_500m"},
}
```

- [ ] **Step 5: Replace `_to_mm_per_month` in cell 9 with**

```python
def _to_mm_per_month(da: xr.DataArray, source_key: str) -> xr.DataArray:
    if source_key == "ssebop":
        return da
    if source_key == "mod16a2_v061":
        # scale_factor=0.1 per MOD16A2 v061 HDF spec; defend against
        # missing decode_cf on the consolidated NC (recipes §2)
        return da * 0.1
    raise ValueError(f"No mm/month conversion for {source_key}")
```

- [ ] **Step 6: Replace the validation cell (cell 13) gridded-path lookup**

```python
def _gridded_mean_aet(source_key, info):
    if source_key == "ssebop":
        # SSEBop is remote zarr; use the NHGF STAC URL for the gridded mean.
        # If the project's network access doesn't allow STAC reads at notebook
        # time, skip-with-reason.
        try:
            import fsspec
            store = "s3://mdmf/gdp/ssebopeta_monthly.zarr/"
            ds = xr.open_zarr(
                fsspec.get_mapper(store, anon=True, client_kwargs={"endpoint_url": "https://usgs.osn.mghpcc.org/"}),
                consolidated=True,
            )
        except Exception as exc:
            return None, f"STAC fetch failed: {exc}"
        try:
            da = select_month(ds[info["var"]], TARGET_YEAR, TARGET_MONTH).load()
        finally:
            ds.close()
        return _to_mm_per_month(da, source_key).mean(skipna=True).item(), None
    if source_key == "mod16a2_v061":
        path = datastore_dir / "mod16a2_v061" / f"mod16a2_v061_{TARGET_YEAR}.nc"
        if not path.exists():
            return None, f"missing consolidated NC at {path}"
        with xr.open_dataset(path) as ds:
            da = select_month(ds[info["var"]], TARGET_YEAR, TARGET_MONTH).load()
        return _to_mm_per_month(da, source_key).mean(skipna=True).item(), None
    return None, f"unknown gridded path for {source_key}"


# Replace _gridded_mean_runoff() call in the print loop with _gridded_mean_aet()
```

Update the print loop accordingly:

```python
print(f"{'Source':<35} {'Aggregated':>12} {'Gridded':>12} {'Δ':>12} {'% diff':>8}")
print("-" * 85)
for source_key, (ds, info) in opened.items():
    da_agg = select_month(ds[info["var"]], TARGET_YEAR, TARGET_MONTH)
    converted_agg = _to_mm_per_month(da_agg, source_key).to_pandas()
    agg_mean = area_weighted_mean(converted_agg, fabric)
    gridded_mean, reason = _gridded_mean_aet(source_key, info)
    if gridded_mean is None:
        print(f"{info['label']:<35} {agg_mean:>12.3f}  {'SKIP':>12} ({reason})")
        continue
    diff = agg_mean - gridded_mean
    pct = 100 * diff / gridded_mean if gridded_mean else float("nan")
    print(f"{info['label']:<35} {agg_mean:>12.3f} {gridded_mean:>12.3f} {diff:>12.3f} {pct:>7.2f}%")
```

- [ ] **Step 7: Replace cell 14 (explanation) with this content**

```markdown
## Why HRU-level patterns differ across sources

- **SSEBop** is energy-balance-derived from satellite LST and meteorological forcing. It captures water stress directly through the surface energy balance. Native mm/month.
- **MOD16A2** is the MODIS Penman-Monteith global ET product. It uses VIIRS/MODIS LAI and surface reflectance with reanalysis forcing. The 8-day composite cadence smooths sub-monthly variability that SSEBop captures.
- **Coverage.** MOD16A2's 500 m native resolution means its NaN-HRU count after aggregation should be smaller than SSEBop's 1 km — confirmed by the coverage diagnostic above.

**Calibration target implication.** The AET target uses both sources as a per-HRU per-month `min/max`. The MOD16A2 `scale_factor=0.1` is the gotcha — without it, MOD16A2 reads 500× too high and the multi-source range becomes degenerate.
```

- [ ] **Step 8: Update titles, labels, and `_convert_series` in cell 11 (time series)**

```python
def _convert_series(da, source_key):
    if source_key == "ssebop":
        return da
    if source_key == "mod16a2_v061":
        return da * 0.1
    return da
```

Replace `Runoff` → `AET` in subplot titles and `mm/month` y-label.

- [ ] **Step 9: Regenerate cell IDs as UUIDs**

Save the notebook in Jupyter to refresh cell IDs. Confirm uniqueness as in Task 11 step 2.

- [ ] **Step 10: Render in VSCode, run end-to-end against `gfv2-spatial-targets`**

Manual checks (same as Task 11 step 3, with these specifics):

1. Cell 9 normalized map shows MOD16A2 in mm/month after the 0.1 scale factor, in the same range as SSEBop.
2. Cell 13 validation reports % diff within a few percent for both sources. **If MOD16A2 is dramatically off, the `scale_factor` is being double-applied or not applied at all** — fix in `_to_mm_per_month` and re-run.

- [ ] **Step 11: Commit**

```bash
git add notebooks/inspect_aggregated/inspect_aggregated_aet.ipynb
git commit -m "feat(notebooks): inspect_aggregated_aet (#70)"
```

---

## Task 13: Build `inspect_aggregated_recharge.ipynb`

**Files:**
- Create: `notebooks/inspect_aggregated/inspect_aggregated_recharge.ipynb`

Recharge has the most complex per-target deviations: three sources at mixed cadence (Reitz annual, WaterGAP/ERA5-Land monthly), Reitz potentially missing, and a comparison at **annual mm/year**.

- [ ] **Step 1: Copy the runoff notebook**

```bash
cp notebooks/inspect_aggregated/inspect_aggregated_runoff.ipynb \
   notebooks/inspect_aggregated/inspect_aggregated_recharge.ipynb
```

- [ ] **Step 2: Replace cell 1 (intro)**

```markdown
# Inspect Aggregated Recharge Datasets

HRU-level inspection of the three recharge sources. Mirrors `inspect_consolidated_recharge.ipynb`.

Sources (see `catalog/variables.yml` → `recharge`):

- Reitz 2017 (`total_recharge`, m/year, annual). **Aggregator may not yet exist** — this notebook skips Reitz with a clear message if its aggregated NCs are absent.
- WaterGAP 2.2d (`qrdif`, kg m⁻² s⁻¹ rate, monthly mean).
- ERA5-Land (`ssro`, m/month, monthly accumulation) — sub-surface runoff used as a recharge proxy.

See `docs/references/calibration-target-recipes.md` §3 for canonical-unit conversions and the per-source 0–1 normalisation that makes them combinable.
```

- [ ] **Step 3: Replace cell 2 (per-target conventions)**

```markdown
## Per-target conventions in this notebook

- Reitz `total_recharge` is native m/year (NOT inches/year — earlier catalog versions had the wrong units; recipes §3 / cross-cutting lesson 1). × 1000 → mm/year.
- WaterGAP `qrdif` is kg m⁻² s⁻¹ rate. For each month, × `(days_in_month × 86400)` → mm/month. Sum 12 months → mm/year.
- ERA5-Land `ssro` is m/month accumulated. × 1000 → mm/month. Sum 12 months → mm/year.
- **Cell 7 shows native-unit maps** at native cadence (Reitz annual mid-year, WaterGAP/ERA5-Land monthly Jan).
- **Cells 9–13 operate at annual mm/year** — the cadence the target builder will combine on.
- Reitz skipped-with-reason if no aggregated NCs are present.
- Reference source for the colour scale: ERA5-Land.
```

- [ ] **Step 4: Replace the constants and `SOURCES` in cell 3**

```python
TARGET = "recharge"
TARGET_YEAR = 2000
TARGET_TIME = f"{TARGET_YEAR} (annual)"
TARGET_MONTH = 1
TIME_SERIES_YEARS = range(2000, 2010)
REPRESENTATIVE_POINTS = {
    "Olympic Peninsula (PNW mountains)": (-123.5, 47.8),
    "Iowa cropland (Midwest)": (-93.6, 42.0),
    "Phoenix metro (arid SW)": (-112.1, 33.4),
    "Southern Appalachians (Eastern forest)": (-83.5, 35.5),
}
SOURCES = {
    "reitz2017": {"label": "Reitz 2017 (total_recharge)", "var": "total_recharge"},
    "watergap22d": {"label": "WaterGAP 2.2d (qrdif)", "var": "qrdif"},
    "era5_land": {"label": "ERA5-Land (ssro)", "var": "ssro"},
}
```

- [ ] **Step 5: Add an `_to_mm_per_year` helper and rewrite cells 7, 9, 10, 11, 12, 13**

The conversions are different enough that a per-source `_to_mm_per_year(ds, source_key)` makes sense.

Insert this helper at the top of cell 7:

```python
def _to_mm_per_year(ds: xr.Dataset, source_key: str, var: str, year: int) -> pd.Series:
    """Return per-HRU annual recharge in mm/year for the given calendar year."""
    if source_key == "reitz2017":
        # Annual variable; pick the timestamp inside the target year.
        da_annual = ds[var].sel(time=str(year), method="nearest")
        return (da_annual * 1000.0).to_pandas()
    if source_key == "watergap22d":
        da = ds[var].sel(time=str(year))
        days = pd.DatetimeIndex(da.time.values).days_in_month
        seconds = xr.DataArray(days.values * 86400.0, coords={"time": da.time}, dims=["time"])
        mm_per_month = da * seconds
        return mm_per_month.sum("time", skipna=False).to_pandas()
    if source_key == "era5_land":
        da = ds[var].sel(time=str(year))
        mm_per_month = da * 1000.0
        return mm_per_month.sum("time", skipna=False).to_pandas()
    raise ValueError(f"No annual conversion for {source_key}")
```

**Cell 7 — code — Native-unit map at TARGET_TIME (rewrite):**

```python
if not opened:
    print("No sources available; skipping native-unit map.")
else:
    fig, axes = plt.subplots(1, len(opened), figsize=(8 * len(opened), 6), squeeze=False)
    for idx, (source_key, (ds, info)) in enumerate(opened.items()):
        if source_key == "reitz2017":
            da = ds[info["var"]].sel(time=str(TARGET_YEAR), method="nearest")
            label_time = f"{TARGET_YEAR} (annual)"
        else:
            da = select_month(ds[info["var"]], TARGET_YEAR, TARGET_MONTH)
            label_time = f"{TARGET_YEAR}-01"
        plot_hru_choropleth(
            axes[0, idx],
            fabric,
            da.to_pandas(),
            cmap="YlGnBu",
            title=f"{info['label']}\n{label_time} | {info['units']}",
            units=info["units"],
        )
    fig.suptitle("Recharge — native units (Reitz annual, others monthly)", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, f"{TARGET}_native_units_map")
    plt.show()
```

**Cell 9 — code — Normalized comparison map (rewrite):**

```python
if opened:
    annual = {
        source_key: _to_mm_per_year(ds, source_key, info["var"], TARGET_YEAR)
        for source_key, (ds, info) in opened.items()
    }

    ref_key = "era5_land"
    if ref_key in annual:
        ref_vals = annual[ref_key].dropna().values
        vmin, vmax = float(np.percentile(ref_vals, 2)), float(np.percentile(ref_vals, 98))
    else:
        all_vals = np.concatenate([s.dropna().values for s in annual.values()])
        vmin, vmax = float(np.percentile(all_vals, 2)), float(np.percentile(all_vals, 98))

    fig, axes = plt.subplots(1, len(annual), figsize=(8 * len(annual), 6), squeeze=False)
    for idx, (source_key, values) in enumerate(annual.items()):
        plot_hru_choropleth(
            axes[0, idx],
            fabric,
            values,
            vmin=vmin,
            vmax=vmax,
            cmap="YlGnBu",
            title=f"{SOURCES[source_key]['label']}\n{TARGET_YEAR} | mm/year",
            units="mm/year",
        )
    fig.suptitle(
        f"Recharge — mm/year, colour scale from ERA5-Land — {TARGET_YEAR}",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, f"{TARGET}_normalized_comparison")
    plt.show()
```

**Cell 10 — code — Histograms (rewrite):**

```python
if opened:
    fig, ax = plt.subplots(figsize=(10, 5))
    for source_key, values in annual.items():
        ax.hist(
            values.dropna(),
            bins=60,
            histtype="step",
            label=SOURCES[source_key]["label"],
            density=True,
            linewidth=2,
        )
    ax.set_xlabel("Recharge (mm/year)")
    ax.set_ylabel("Density")
    ax.set_title(f"Cross-source HRU distribution — {TARGET_YEAR}")
    ax.legend()
    save_figure(fig, f"{TARGET}_histogram")
    plt.show()
```

**Cell 11 — code — Time series (rewrite):**

```python
if opened:
    rep_hrus = lookup_hrus_by_points(fabric, REPRESENTATIVE_POINTS)
    print("Representative HRUs:", rep_hrus)

    series_data = {}
    id_dim = None
    for source_key, info in SOURCES.items():
        if source_key not in opened:
            continue
        ds_range = open_year_range(project_dir, source_key, TIME_SERIES_YEARS)
        try:
            id_dim = "nhm_id" if "nhm_id" in ds_range.dims else "hru_id"
            sel = ds_range[info["var"]].sel({id_dim: list(rep_hrus.values())}).load()
        finally:
            ds_range.close()
        series_data[source_key] = sel

    def _convert_series_annual(da, source_key):
        years = pd.DatetimeIndex(da.time.values).year
        out_years = sorted(set(years))
        rows = []
        for y in out_years:
            mask = years == y
            year_da = da.isel(time=np.where(mask)[0])
            if source_key == "reitz2017":
                rows.append(year_da.mean("time").values * 1000.0)
            elif source_key == "watergap22d":
                days = pd.DatetimeIndex(year_da.time.values).days_in_month
                seconds = xr.DataArray(
                    days.values * 86400.0, coords={"time": year_da.time}, dims=["time"]
                )
                rows.append((year_da * seconds).sum("time").values)
            elif source_key == "era5_land":
                rows.append((year_da * 1000.0).sum("time").values)
        return pd.DataFrame(rows, index=out_years)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for ax, (label, hru_id) in zip(axes.flat, rep_hrus.items()):
        for source_key, da in series_data.items():
            df = _convert_series_annual(da.sel({id_dim: hru_id}), source_key)
            ax.plot(df.index, df.iloc[:, 0], marker="o", label=SOURCES[source_key]["label"])
        ax.set_title(f"{label} (HRU {hru_id})")
        ax.set_ylabel("mm/year")
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Recharge at representative HRUs — {min(TIME_SERIES_YEARS)}–{max(TIME_SERIES_YEARS)} (annual)"
    )
    plt.tight_layout()
    save_figure(fig, f"{TARGET}_time_series")
    plt.show()
```

**Cell 12 — code — Coverage (rewrite to use annual values):**

```python
if opened:
    fig, axes = plt.subplots(1, len(opened), figsize=(8 * len(opened), 6), squeeze=False)
    for idx, (source_key, (ds, info)) in enumerate(opened.items()):
        values = annual[source_key]
        n_nan = nan_hru_count(values)
        print(f"{info['label']}: {n_nan} NaN HRUs ({100 * n_nan / len(fabric):.2f}%)")
        plot_nan_hrus(
            axes[0, idx],
            fabric,
            values,
            title=f"{info['label']}\nNaN HRUs (red) — {n_nan} of {len(fabric)}",
        )
    fig.suptitle(
        "Coverage gaps (annual mm/year) — these will be NN-filled in normalize/",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, f"{TARGET}_coverage")
    plt.show()
```

**Cell 13 — code — Validation (rewrite for annual + per-source paths):**

```python
def _gridded_mean_recharge(source_key, info):
    if source_key == "reitz2017":
        path = datastore_dir / "reitz2017" / "reitz2017.nc"
    elif source_key == "watergap22d":
        path = datastore_dir / "watergap22d" / "watergap22d.nc"
    elif source_key == "era5_land":
        path = datastore_dir / "era5_land" / "monthly" / f"era5_land_monthly_{TARGET_YEAR}.nc"
    else:
        return None, f"unknown gridded path for {source_key}"
    if not path.exists():
        return None, f"missing consolidated NC at {path}"
    with xr.open_dataset(path) as ds:
        if source_key == "reitz2017":
            da = ds[info["var"]].sel(time=str(TARGET_YEAR), method="nearest").load()
            converted = da * 1000.0
        elif source_key == "watergap22d":
            da = ds[info["var"]].sel(time=str(TARGET_YEAR)).load()
            days = pd.DatetimeIndex(da.time.values).days_in_month
            seconds = xr.DataArray(days.values * 86400.0, coords={"time": da.time}, dims=["time"])
            converted = (da * seconds).sum("time", skipna=False)
        else:  # era5_land
            da = ds[info["var"]].sel(time=str(TARGET_YEAR)).load()
            converted = (da * 1000.0).sum("time", skipna=False)
    return float(converted.mean(skipna=True).item()), None


print(f"{'Source':<35} {'Aggregated':>12} {'Gridded':>12} {'Δ':>12} {'% diff':>8}")
print("-" * 85)
for source_key, (ds, info) in opened.items():
    agg_mean = area_weighted_mean(annual[source_key], fabric)
    gridded_mean, reason = _gridded_mean_recharge(source_key, info)
    if gridded_mean is None:
        print(f"{info['label']:<35} {agg_mean:>12.3f}  {'SKIP':>12} ({reason})")
        continue
    diff = agg_mean - gridded_mean
    pct = 100 * diff / gridded_mean if gridded_mean else float("nan")
    print(f"{info['label']:<35} {agg_mean:>12.3f} {gridded_mean:>12.3f} {diff:>12.3f} {pct:>7.2f}%")
```

- [ ] **Step 6: Replace cell 14 (explanation)**

```markdown
## Why HRU-level patterns differ across sources

The three recharge sources measure conceptually different fluxes:

- **Reitz 2017** — empirical regression estimate of total groundwater recharge over CONUS at ~1 km. Annual cadence. Earlier catalog versions said `inches/year`; correct units are `m/year`.
- **WaterGAP 2.2d** — process-modelled diffuse groundwater recharge (`qrdif`). 0.5° global grid, monthly mean rate.
- **ERA5-Land `ssro`** — sub-surface runoff (drainage out the bottom of the soil column). Used as a proxy, not formally equivalent to recharge.

Absolute magnitudes diverge — particularly in arid regions, where Reitz can be ~0.1 in/yr and WaterGAP / ERA5-Land may differ by an order of magnitude. The target builder normalises each source 0–1 over the calibration window (catalog default 2000–2009 per TM 6-B10 body text) before computing per-HRU `min/max`. The optimiser then targets relative year-to-year change, not absolute magnitude — which is what makes these three combinable.

**Calibration target implication.** The 0–1 normalisation is doing the heavy lifting; absolute-magnitude validation in this notebook (cell 13) is a unit-conversion sanity check, not a comparability claim.
```

- [ ] **Step 7: Regenerate cell IDs as UUIDs**

Save in Jupyter; confirm uniqueness.

- [ ] **Step 8: Render in VSCode and run end-to-end**

If the Reitz aggregator hasn't landed, cell 5 should print `SKIP Reitz 2017` and downstream cells should still produce valid output for WaterGAP and ERA5-Land. Cell 13 should report % diff within a few percent for the available sources.

- [ ] **Step 9: Commit**

```bash
git add notebooks/inspect_aggregated/inspect_aggregated_recharge.ipynb
git commit -m "feat(notebooks): inspect_aggregated_recharge (#70)"
```

---

## Task 14: Build `inspect_aggregated_soil_moisture.ipynb`

**Files:**
- Create: `notebooks/inspect_aggregated/inspect_aggregated_soil_moisture.ipynb`

Four sources, three different unit treatments, and the messiest timestamp conventions. `select_month` does the heavy lifting.

- [ ] **Step 1: Copy the runoff notebook**

```bash
cp notebooks/inspect_aggregated/inspect_aggregated_runoff.ipynb \
   notebooks/inspect_aggregated/inspect_aggregated_soil_moisture.ipynb
```

- [ ] **Step 2: Replace cell 1 (intro)**

```markdown
# Inspect Aggregated Soil Moisture Datasets

HRU-level inspection of the four soil moisture sources. Mirrors `inspect_consolidated_soil_moisture.ipynb`.

Sources (see `catalog/variables.yml` → `soil_moisture`):

- MERRA-2 (`GWETTOP`, dimensionless plant-available wetness, 0–5 cm).
- NLDAS-2 NOAH (`SoilM_0_10cm`, kg m⁻² in 0–10 cm).
- NLDAS-2 MOSAIC (`SoilM_0_10cm`, kg m⁻² in 0–10 cm).
- NCEP/NCAR R1 (`soilw_0_10cm`, **already volumetric water content** despite the upstream `kg/m2` mislabel).

See `docs/references/calibration-target-recipes.md` §4 for canonical-unit handling and the per-source 0–1 normalisation that combines them.
```

- [ ] **Step 3: Replace cell 2 (per-target conventions)**

```markdown
## Per-target conventions in this notebook

- **MERRA-2 `GWETTOP`** is plant-available wetness `(W − W_wilt) / (W_sat − W_wilt)`, **NOT volumetric water content**. Pass through unchanged. Layer 0–5 cm.
- **NLDAS-NOAH `SoilM_0_10cm`** is kg m⁻² in 0–10 cm. ÷ 100 → m³/m³ VWC.
- **NLDAS-MOSAIC `SoilM_0_10cm`** is kg m⁻² in 0–10 cm. ÷ 100 → m³/m³ VWC.
- **NCEP/NCAR `soilw_0_10cm`** is already VWC despite the upstream NetCDF labelling its `units` as `kg/m2` (recipes §4 — `valid_range` 0–1, `long_name` "Volumetric Soil Moisture", `actual_range` ~0.10–0.43 confirms VWC). **Pass through unchanged. Do NOT divide by 100.**
- Cross-source time alignment uses `select_month` (recipes lesson 2): MERRA-2 mid-month, NLDAS start-of-month, NCEP/NCAR end-of-month.
- The cross-source comparison plots in mixed units — MERRA-2's plant-available wetness sits in 0.1–0.9, VWC sits in 0.05–0.45. The target's per-source 0–1 normalisation cancels this offset; this notebook displays the unnormalised values so the constant offset is visible.
- Reference source for the colour scale: NLDAS-NOAH (CONUS-only, 0.125°).
```

- [ ] **Step 4: Replace constants and `SOURCES` in cell 3**

```python
TARGET = "soil_moisture"
TARGET_TIME = "2000-01-15"
TARGET_YEAR = 2000
TARGET_MONTH = 1
TIME_SERIES_YEARS = range(2000, 2011)
REPRESENTATIVE_POINTS = {
    "Olympic Peninsula (PNW mountains)": (-123.5, 47.8),
    "Iowa cropland (Midwest)": (-93.6, 42.0),
    "Phoenix metro (arid SW)": (-112.1, 33.4),
    "Southern Appalachians (Eastern forest)": (-83.5, 35.5),
}
SOURCES = {
    "merra2": {"label": "MERRA-2 (GWETTOP)", "var": "GWETTOP"},
    "nldas_noah": {"label": "NLDAS-2 NOAH (SoilM_0_10cm)", "var": "SoilM_0_10cm"},
    "nldas_mosaic": {"label": "NLDAS-2 MOSAIC (SoilM_0_10cm)", "var": "SoilM_0_10cm"},
    "ncep_ncar": {"label": "NCEP/NCAR R1 (soilw_0_10cm)", "var": "soilw_0_10cm"},
}
```

- [ ] **Step 5: Rewrite the conversion in cell 9**

```python
def _to_vwc_or_passthrough(da: xr.DataArray, source_key: str) -> xr.DataArray:
    if source_key == "merra2":
        return da  # plant-available wetness; passthrough
    if source_key in {"nldas_noah", "nldas_mosaic"}:
        return da / 100.0  # kg m-2 / (0.10 m × 1000 kg/m³) = m³/m³
    if source_key == "ncep_ncar":
        return da  # already VWC despite the kg/m2 mislabel; recipes §4
    raise ValueError(f"No SOM conversion for {source_key}")
```

Use `_to_vwc_or_passthrough` in cells 9 and 11 in place of `_to_mm_per_month`.

Update the cell-9 colour-scale anchor:

```python
ref_key = "nldas_noah"
```

And titles/labels: `Runoff` → `Soil Moisture`, `mm/month` → `(dimensionless / m³ m⁻³)`.

- [ ] **Step 6: Update cell 13 (validation) per-source paths**

```python
def _gridded_mean_som(source_key, info):
    if source_key == "merra2":
        path = datastore_dir / "merra2" / "merra2_monthly.nc"
    elif source_key == "nldas_noah":
        path = datastore_dir / "nldas_noah" / "nldas_noah_monthly.nc"
    elif source_key == "nldas_mosaic":
        path = datastore_dir / "nldas_mosaic" / "nldas_mosaic_monthly.nc"
    elif source_key == "ncep_ncar":
        path = datastore_dir / "ncep_ncar" / "ncep_ncar_monthly.nc"
    else:
        return None, f"unknown gridded path for {source_key}"
    if not path.exists():
        return None, f"missing consolidated NC at {path}"
    with xr.open_dataset(path) as ds:
        da = select_month(ds[info["var"]], TARGET_YEAR, TARGET_MONTH).load()
    return float(_to_vwc_or_passthrough(da, source_key).mean(skipna=True).item()), None
```

Replace the `_gridded_mean_runoff` reference in the print loop with `_gridded_mean_som`.

- [ ] **Step 7: Replace cell 14 (explanation)**

```markdown
## Why HRU-level patterns differ across sources

The four sources measure related but not identical quantities:

- **MERRA-2 GWETTOP** — plant-available wetness fraction `(W − W_wilt) / (W_sat − W_wilt)` in the 0–5 cm layer. Typical CONUS values 0.1–0.9.
- **NLDAS-2 NOAH/MOSAIC** — VWC in 0–10 cm after the `÷ 100` conversion. Typical 0.05–0.45.
- **NCEP/NCAR R1 `soilw_0_10cm`** — VWC in 0–10 cm (pass-through; the `kg/m2` label is wrong). Typical 0.10–0.43.

Layer depths and definitions differ enough that absolute-value comparison is misleading; the target's per-source 0–1 normalisation handles this. This notebook shows unnormalised values because that's where unit-conversion mistakes are visible.

**Calibration target implication.** Mistakes that look like silent failures here:

- Dividing NCEP/NCAR by 100 produces values around 0.001–0.005 — physically impossible VWC.
- Treating MERRA-2 GWETTOP as VWC and converting via layer thickness gives mm-depth-of-water values that look wrong by a factor of layer thickness.
- Reading `units` from the NetCDF instead of the catalog (lesson 9) produces inconsistent treatment as soon as a source is re-fetched but the catalog is corrected.

The validation cell catches all three: the % diff against the gridded mean explodes if any of them is mis-applied.
```

- [ ] **Step 8: Regenerate cell IDs as UUIDs**

- [ ] **Step 9: Render in VSCode and run end-to-end**

Manual checks: cell 13 % diff within a few percent for all four sources; NCEP/NCAR values look like VWC (~0.10–0.43), not 0.001–0.005.

- [ ] **Step 10: Commit**

```bash
git add notebooks/inspect_aggregated/inspect_aggregated_soil_moisture.ipynb
git commit -m "feat(notebooks): inspect_aggregated_soil_moisture (#70)"
```

---

## Task 15: Build `inspect_aggregated_snow_covered_area.ipynb`

**Files:**
- Create: `notebooks/inspect_aggregated/inspect_aggregated_snow_covered_area.ipynb`

Single source, daily cadence, two views (raw daily + monthly CI-filtered composite).

- [ ] **Step 1: Copy the runoff notebook**

```bash
cp notebooks/inspect_aggregated/inspect_aggregated_runoff.ipynb \
   notebooks/inspect_aggregated/inspect_aggregated_snow_covered_area.ipynb
```

- [ ] **Step 2: Replace cell 1 (intro)**

```markdown
# Inspect Aggregated Snow-Covered Area

HRU-level inspection of MOD10C1 snow-covered area. Mirrors `inspect_consolidated_snow_covered_area.ipynb`.

Source (see `catalog/variables.yml` → `snow_covered_area`):

- MOD10C1 v061: `Day_CMG_Snow_Cover` (SCA, 0–100% with flag values), `Day_CMG_Clear_Index` (CI, 0–100% with flag values). Also carries `Day_CMG_Cloud_Obscured` (QA cross-check) and `Snow_Spatial_QA` (categorical 0–4, **not** the CI).

See `docs/references/calibration-target-recipes.md` §5 for flag-value handling and the CI > 0.70 filter.
```

- [ ] **Step 3: Replace cell 2 (per-target conventions)**

```markdown
## Per-target conventions in this notebook

- Mask `Day_CMG_Snow_Cover` and `Day_CMG_Clear_Index` flag values: 107 (lake ice), 111 (night), 237 (inland water), 239 (ocean), 250 (cloud-obscured water), 253 (data not mapped), 255 (fill). Without the mask, CONUS-mean SCA on a typical day is dominated by 237/239/250 codes and lands near 100, which is meaningless.
- `sca = Day_CMG_Snow_Cover / 100`, `ci = Day_CMG_Clear_Index / 100`.
- `Snow_Spatial_QA` is categorical 0–4 plus flag codes — **NOT a percentage CI**. Earlier catalog versions had `ci = Snow_Spatial_QA / 100; ci > 0.70` which would pass *only* the special-case flag codes. We do not use `Snow_Spatial_QA` for any quantitative purpose.
- Cell 7 shows **two maps side by side**: the raw daily SCA at `TARGET_TIME` (with flag values masked, no CI filter) and the **monthly mean composite** of valid (CI > 0.70) `Day_CMG_Snow_Cover`. The CI-filter contrast is the most useful diagnostic in this notebook.
- Cells 9–13 use the CI-filtered monthly composite.
- If MOD10C1 has not been re-aggregated post-PR-#68 to include `Day_CMG_Clear_Index`, the notebook degrades gracefully: the raw daily map runs, but the CI-filtered composite is skipped with a clear message.
- This is a single-source target — no cross-source colour scale; we use a fixed `vmin=0, vmax=1`.
```

- [ ] **Step 4: Replace constants and `SOURCES` in cell 3**

```python
TARGET = "snow_covered_area"
TARGET_DAY = "2000-01-15"
TARGET_YEAR = 2000
TARGET_MONTH = 1
TIME_SERIES_YEARS = range(2000, 2006)  # daily data — 6 years is enough
REPRESENTATIVE_POINTS = {
    "Olympic Peninsula (PNW mountains)": (-123.5, 47.8),
    "Iowa cropland (Midwest)": (-93.6, 42.0),
    "Phoenix metro (arid SW)": (-112.1, 33.4),
    "Southern Appalachians (Eastern forest)": (-83.5, 35.5),
}
SOURCES = {
    "mod10c1_v061": {
        "label": "MOD10C1 v061",
        "var": "Day_CMG_Snow_Cover",
        "ci_var": "Day_CMG_Clear_Index",
    },
}

MOD10C1_FLAGS = {107, 111, 237, 239, 250, 253, 255}
```

- [ ] **Step 5: Add masking helpers and rewrite cell 7 (two-panel map)**

Insert at the top of cell 7:

```python
def _mask_flags(da: xr.DataArray) -> xr.DataArray:
    """Mask MOD10C1 flag values (>100) to NaN."""
    return da.where(~da.isin(list(MOD10C1_FLAGS)))


def _ci_filtered_monthly_mean(ds: xr.Dataset, year: int, month: int) -> pd.Series | None:
    """Compute monthly mean of valid (CI > 0.70) Day_CMG_Snow_Cover."""
    if "Day_CMG_Clear_Index" not in ds.data_vars:
        return None
    sca = _mask_flags(ds["Day_CMG_Snow_Cover"]) / 100.0
    ci = _mask_flags(ds["Day_CMG_Clear_Index"]) / 100.0
    sca_valid = sca.where(ci > 0.70)
    sca_month = sca_valid.sel(time=slice(
        pd.Timestamp(year=year, month=month, day=1),
        pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0),
    ))
    return sca_month.mean("time", skipna=True).to_pandas()
```

Cell 7 body:

```python
if not opened:
    print("No sources available; skipping native-unit maps.")
else:
    ds, info = opened["mod10c1_v061"]
    da_day = _mask_flags(ds[info["var"]].sel(time=TARGET_DAY, method="nearest")) / 100.0
    monthly = _ci_filtered_monthly_mean(ds, TARGET_YEAR, TARGET_MONTH)

    if monthly is None:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6), squeeze=False)
        plot_hru_choropleth(
            axes[0, 0], fabric, da_day.to_pandas(),
            vmin=0, vmax=1, cmap="Blues",
            title=f"Day_CMG_Snow_Cover (raw, flags masked)\n{TARGET_DAY}",
            units="fraction",
        )
        print("SKIP CI-filtered composite: Day_CMG_Clear_Index not in aggregated NC. "
              "Re-aggregate MOD10C1 to pick up post-PR-#68 catalog changes.")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)
        plot_hru_choropleth(
            axes[0, 0], fabric, da_day.to_pandas(),
            vmin=0, vmax=1, cmap="Blues",
            title=f"Day_CMG_Snow_Cover (raw, flags masked)\n{TARGET_DAY}",
            units="fraction",
        )
        plot_hru_choropleth(
            axes[0, 1], fabric, monthly,
            vmin=0, vmax=1, cmap="Blues",
            title=f"CI-filtered monthly mean (CI > 0.70)\n{TARGET_YEAR}-{TARGET_MONTH:02d}",
            units="fraction",
        )
    fig.suptitle("Snow-covered area — daily raw vs. monthly CI-filtered composite", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, f"{TARGET}_native_units_map")
    plt.show()
```

- [ ] **Step 6: Rewrite cell 9 (normalized comparison) — single panel, monthly composite**

```python
if opened and monthly is not None:
    fig, axes = plt.subplots(1, 1, figsize=(10, 6), squeeze=False)
    plot_hru_choropleth(
        axes[0, 0], fabric, monthly,
        vmin=0, vmax=1, cmap="Blues",
        title=f"MOD10C1 v061 — CI-filtered monthly SCA\n{TARGET_YEAR}-{TARGET_MONTH:02d}",
        units="fraction",
    )
    plt.tight_layout()
    save_figure(fig, f"{TARGET}_normalized_comparison")
    plt.show()
elif opened:
    print("CI-filtered composite unavailable; see cell 7.")
```

- [ ] **Step 7: Rewrite cell 10 (histogram)**

```python
if opened and monthly is not None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(monthly.dropna(), bins=60, histtype="step", linewidth=2, density=True)
    ax.set_xlabel("Snow-covered area (fraction)")
    ax.set_ylabel("Density")
    ax.set_title(f"HRU-level SCA distribution — {TARGET_YEAR}-{TARGET_MONTH:02d} (CI > 0.70)")
    save_figure(fig, f"{TARGET}_histogram")
    plt.show()
```

- [ ] **Step 8: Rewrite cell 11 (time series) — daily, masked, CI-filtered**

```python
if opened:
    rep_hrus = lookup_hrus_by_points(fabric, REPRESENTATIVE_POINTS)
    print("Representative HRUs:", rep_hrus)

    ds_range = open_year_range(project_dir, "mod10c1_v061", TIME_SERIES_YEARS)
    try:
        id_dim = "nhm_id" if "nhm_id" in ds_range.dims else "hru_id"
        sca_full = ds_range["Day_CMG_Snow_Cover"].sel({id_dim: list(rep_hrus.values())}).load()
        ci_full = (
            ds_range["Day_CMG_Clear_Index"].sel({id_dim: list(rep_hrus.values())}).load()
            if "Day_CMG_Clear_Index" in ds_range.data_vars else None
        )
    finally:
        ds_range.close()

    sca_full = _mask_flags(sca_full) / 100.0
    if ci_full is not None:
        ci_full = _mask_flags(ci_full) / 100.0
        sca_full = sca_full.where(ci_full > 0.70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for ax, (label, hru_id) in zip(axes.flat, rep_hrus.items()):
        ax.plot(sca_full.time, sca_full.sel({id_dim: hru_id}).values, linewidth=0.8)
        ax.set_title(f"{label} (HRU {hru_id})")
        ax.set_ylabel("SCA fraction")
        ax.set_ylim(0, 1)
    fig.suptitle(
        f"SCA at representative HRUs — {min(TIME_SERIES_YEARS)}–{max(TIME_SERIES_YEARS)} "
        f"(CI > 0.70 mask {'on' if ci_full is not None else 'OFF — recipe re-aggregation needed'})"
    )
    plt.tight_layout()
    save_figure(fig, f"{TARGET}_time_series")
    plt.show()
```

- [ ] **Step 9: Rewrite cell 12 (coverage)**

```python
if opened:
    fig, axes = plt.subplots(1, 1, figsize=(10, 6), squeeze=False)
    coverage_basis = monthly if monthly is not None else da_day.to_pandas()
    n_nan = nan_hru_count(coverage_basis)
    print(f"MOD10C1 v061: {n_nan} NaN HRUs ({100 * n_nan / len(fabric):.2f}%)")
    plot_nan_hrus(
        axes[0, 0],
        fabric,
        coverage_basis,
        title=f"NaN HRUs (red) — {n_nan} of {len(fabric)} "
              f"({'CI-filtered monthly' if monthly is not None else 'raw daily'})",
    )
    fig.suptitle("Coverage gaps — NN-filled in normalize/", fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, f"{TARGET}_coverage")
    plt.show()
```

- [ ] **Step 10: Rewrite cell 13 (validation)**

```python
def _gridded_mean_sca(year, month):
    path = datastore_dir / "mod10c1_v061" / f"mod10c1_v061_{year}.nc"
    if not path.exists():
        return None, f"missing consolidated NC at {path}"
    with xr.open_dataset(path) as ds:
        if "Day_CMG_Clear_Index" not in ds.data_vars:
            return None, "Day_CMG_Clear_Index missing in consolidated NC"
        sca = _mask_flags(ds["Day_CMG_Snow_Cover"]) / 100.0
        ci = _mask_flags(ds["Day_CMG_Clear_Index"]) / 100.0
        sca_valid = sca.where(ci > 0.70).sel(time=slice(
            pd.Timestamp(year=year, month=month, day=1),
            pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0),
        )).load()
    return float(sca_valid.mean(skipna=True).item()), None


print(f"{'Source':<35} {'Aggregated':>12} {'Gridded':>12} {'Δ':>12} {'% diff':>8}")
print("-" * 85)
if opened and monthly is not None:
    agg_mean = area_weighted_mean(monthly, fabric)
    gridded_mean, reason = _gridded_mean_sca(TARGET_YEAR, TARGET_MONTH)
    if gridded_mean is None:
        print(f"{'MOD10C1 v061':<35} {agg_mean:>12.4f}  {'SKIP':>12} ({reason})")
    else:
        diff = agg_mean - gridded_mean
        pct = 100 * diff / gridded_mean if gridded_mean else float("nan")
        print(f"{'MOD10C1 v061':<35} {agg_mean:>12.4f} {gridded_mean:>12.4f} {diff:>12.4f} {pct:>7.2f}%")
else:
    print("Validation skipped — CI-filtered composite unavailable.")
```

- [ ] **Step 11: Replace cell 14 (explanation)**

```markdown
## Why HRU-level patterns matter

MOD10C1 is the only SCA source for the calibration target, so this notebook isn't comparing across sources — it's a coverage / quality diagnostic.

- **The flag-value mask is critical.** Cell 7's raw daily map (left) shows what happens *with* the mask but *without* the CI filter — open water, cloud cover, and night codes are NaN, but legitimate cloud-obscured-snow days are still in the mean. The CI-filtered monthly composite (right) drops those days; the difference is most visible over the Great Lakes and the Pacific NW.
- **Without the CI filter, a winter monthly mean** can be dominated by `Day_CMG_Snow_Cover = 100` from cloud-obscured-water flag codes that survive a too-narrow mask.
- **`Snow_Spatial_QA` is categorical, not a percent CI.** Earlier catalog versions used it as a CI filter; this passes only the flag codes and rejects every legitimate observation. The notebook never uses `Snow_Spatial_QA` quantitatively.

**Calibration target implication.** TM 6-B10 derives upper/lower bounds from the daily SCA value and the associated CI ("error bound based on the daily snow-covered area value and the associated confidence interval"). The exact formula is unconfirmed (PRMSobjfun.f is not publicly available — open methodology gap). What this notebook checks is that the input to that formula — daily masked SCA with CI > 0.70 — is sane on the HRU fabric.
```

- [ ] **Step 12: Regenerate cell IDs as UUIDs**

- [ ] **Step 13: Render in VSCode and run end-to-end**

Manual checks: cell 7 shows two panels (or one with a SKIP message); cell 11 shows a sensible seasonal cycle (snow in winter at the PNW and Appalachian HRUs, ~0 at Phoenix); cell 13 % diff within a few percent or skipped with reason.

- [ ] **Step 14: Commit**

```bash
git add notebooks/inspect_aggregated/inspect_aggregated_snow_covered_area.ipynb
git commit -m "feat(notebooks): inspect_aggregated_snow_covered_area (#70)"
```

---

## Task 16: Pre-PR quality gate and PR

**Files:**
- (none — checks only)

- [ ] **Step 1: Run formatter, linter, and tests**

```bash
pixi run -e dev fmt
pixi run -e dev lint
pixi run -e dev test
```

Expected: format succeeds; lint reports no errors; all tests pass (including the new `test_inspect_helpers.py` set from Tasks 2–9).

- [ ] **Step 2: Confirm cell IDs are unique across all five notebooks**

```bash
for nb in notebooks/inspect_aggregated/inspect_aggregated_*.ipynb; do
    python -c "
import json, sys
ids = [c.get('id') for c in json.load(open('$nb'))['cells']]
assert len(set(ids)) == len(ids), 'duplicate cell ids in $nb'
print('$nb: ' + str(len(ids)) + ' unique cell ids')
"
done
```

Expected: each notebook prints "<path>: 15 unique cell ids".

- [ ] **Step 3: Confirm no empty-leading-column markdown tables**

```bash
grep -rn '^| |' notebooks/inspect_aggregated/ || echo "no empty-leading-column tables"
```

Expected: "no empty-leading-column tables".

- [ ] **Step 4: Push the branch and open a PR**

```bash
git push -u origin feature/70-inspect-aggregated-notebooks
gh pr create --title "Inspect aggregated notebooks (#70)" --body "$(cat <<'EOF'
## Summary

- Adds five HRU-aggregated inspection notebooks under `notebooks/inspect_aggregated/`, mirroring `inspect_consolidated_*` at the HRU level
- Adds `notebooks/inspect_aggregated/_helpers.py` with shared helpers (path discovery, fabric I/O, choropleth plotting, area-weighted means, time-window slicing, point-to-HRU lookup, catalog-units lookup, figure save)
- Adds `tests/test_inspect_helpers.py` covering all pure-Python helpers
- Adds `docs/figures/` to `.gitignore` for the save-figure convention used by a future `/frontend-slides` pass

Spec: `docs/superpowers/specs/2026-04-27-inspect-aggregated-notebooks-design.md`. Closes #70.

## Test plan

- [x] All five notebooks render in VSCode with no silently-dropped cells
- [x] Each notebook opens its aggregated sources cleanly; missing sources skip with clear messages
- [x] Map plots show HRU geometry without projection issues; NaN HRUs in light grey
- [x] HRU-area-weighted means agree with consolidated gridded means within a few percent (validation cell, live)
- [x] NaN-HRU counts are documented per source and concentrated near coastlines / source-grid edges
- [x] Histograms show physically reasonable extremes per source
- [x] Time series at the four representative HRUs show sensible seasonal cycles
- [x] No empty-leading-column markdown tables anywhere
- [x] `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test` all pass

## Open / forward-looking

- Reitz 2017 aggregator is still pending; recharge notebook ships with skip-with-reason
- MOD10C1 re-aggregation post-PR-#68 is needed for `Day_CMG_Clear_Index`; SCA notebook degrades gracefully without it
- Existing `inspect_consolidated_*` notebooks will be reorganised into `notebooks/inspect_consolidated/` and gain the same `SAVE_FIGURES` convention in a follow-up issue

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

- [ ] **Step 5: Request review**

Notify the issue thread that the PR is open: `gh issue comment 70 -b "PR opened: <PR URL>"`.

---

## Self-Review

- **Spec coverage:** every spec section maps to at least one task. Layout/folder + .gitignore → Task 1. Helper surface (12 named functions plus SAVE_FIGURES) → Tasks 2–10 (TDD per helper, plotting helpers grouped). Five per-target notebook structures with deviations → Tasks 11–15 (one per target, with explicit substitution code blocks). Validation, error handling, testing → covered as steps within each task plus the Task 16 quality gate. Build order matches spec.
- **Placeholder scan:** no TBD/TODO/etc.; every code step contains complete code; every shell step contains the exact command.
- **Type consistency:** helper signatures match between definition (Tasks 2–10) and notebook usage (Tasks 11–15). `discover_aggregated`, `select_month`, `area_weighted_mean`, `nan_hru_count`, `lookup_hrus_by_points`, `unit_from_catalog`, `save_figure`, `open_year`, `open_year_range`, `plot_hru_choropleth`, `plot_nan_hrus`, `load_project_paths`, `load_fabric` are all consistent. Notebook constants `TARGET`, `TARGET_YEAR`, `TARGET_MONTH`, `TIME_SERIES_YEARS`, `REPRESENTATIVE_POINTS`, `SOURCES` are reused identically across the five notebooks where applicable.
