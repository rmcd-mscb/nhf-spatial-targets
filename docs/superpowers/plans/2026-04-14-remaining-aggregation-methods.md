# Remaining Aggregation Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build spatial-aggregation modules for every remaining catalog source (ERA5-Land, GLDAS, MERRA-2, NCEP/NCAR, NLDAS-MOSAIC, NLDAS-NOAH, WaterGAP 2.2d, MOD16A2, MOD10C1) so their consolidated NetCDFs are aggregated to project HRU polygons and written to `data/aggregated/<source>_agg.nc`.

**Architecture:** Two-tier — a shared declarative driver (`_driver.py` + `_adapter.py`) for seven well-behaved lat/lon sources, and bespoke modules for MOD16A2 (sinusoidal CRS) and MOD10C1 (CI masking at source + valid-area fraction). SSEBop stays unchanged except for a minor refactor extracting its manifest helper so the new driver reuses it. Aggregators consume the full temporal range of the source NC — period-of-interest clipping lives downstream in target builders.

**Tech Stack:** Python 3.11+, `gdptools` (WeightGen/AggGen), `geopandas`, `xarray`, `rioxarray`, `pyyaml`, `cyclopts` (CLI), `pytest`.

**Spec:** `docs/superpowers/specs/2026-04-14-remaining-aggregation-methods-design.md`

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `src/nhf_spatial_targets/aggregate/_adapter.py` | `SourceAdapter` dataclass |
| `src/nhf_spatial_targets/aggregate/_driver.py` | Shared engine: `aggregate_source`, batching+weights helpers, `update_manifest` |
| `src/nhf_spatial_targets/aggregate/era5_land.py` | Tier-1 adapter (runoff) |
| `src/nhf_spatial_targets/aggregate/gldas.py` | Tier-1 adapter (runoff, derived `runoff_total`) |
| `src/nhf_spatial_targets/aggregate/merra2.py` | Tier-1 adapter (soil moisture) |
| `src/nhf_spatial_targets/aggregate/ncep_ncar.py` | Tier-1 adapter (soil moisture) |
| `src/nhf_spatial_targets/aggregate/nldas_mosaic.py` | Tier-1 adapter (soil moisture) |
| `src/nhf_spatial_targets/aggregate/nldas_noah.py` | Tier-1 adapter (soil moisture) |
| `src/nhf_spatial_targets/aggregate/watergap22d.py` | Tier-1 adapter (recharge) |
| `src/nhf_spatial_targets/aggregate/mod16a2.py` | Tier-2 bespoke (sinusoidal CRS, AET) |
| `src/nhf_spatial_targets/aggregate/mod10c1.py` | Tier-2 bespoke (CI masking, SCA) |
| `tests/test_aggregate_driver.py` | Driver unit tests |
| `tests/test_aggregate_era5_land.py` | Per-source adapter unit test |
| `tests/test_aggregate_gldas.py` | Per-source adapter unit test |
| `tests/test_aggregate_merra2.py` | Per-source adapter unit test |
| `tests/test_aggregate_ncep_ncar.py` | Per-source adapter unit test |
| `tests/test_aggregate_nldas_mosaic.py` | Per-source adapter unit test |
| `tests/test_aggregate_nldas_noah.py` | Per-source adapter unit test |
| `tests/test_aggregate_watergap22d.py` | Per-source adapter unit test |
| `tests/test_aggregate_mod16a2.py` | Tier-2 unit test (sinusoidal CRS propagation) |
| `tests/test_aggregate_mod10c1.py` | Tier-2 unit test (CI masking, valid_area) |

### Modified files

| Path | Change |
|---|---|
| `src/nhf_spatial_targets/aggregate/ssebop.py` | Delete local `_update_manifest`, call shared `update_manifest` |
| `src/nhf_spatial_targets/aggregate/__init__.py` | Re-export `SourceAdapter`, `aggregate_source`, `update_manifest` |
| `src/nhf_spatial_targets/cli.py` | Add 10 new `agg` subcommands (+`agg all`) |
| `tests/test_aggregate_ssebop.py` | Update manifest-helper import if needed |

---

## Task 1: Extract shared manifest helper

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/_driver.py`
- Modify: `src/nhf_spatial_targets/aggregate/ssebop.py` (remove local `_update_manifest`, call shared)
- Modify: `src/nhf_spatial_targets/aggregate/__init__.py`

**Context:** `ssebop.py` contains a private `_update_manifest` that atomically writes provenance under `manifest["sources"][source_key]`. Lift it verbatim into `_driver.py` as the public `update_manifest`, generalising so any source can call it.

- [ ] **Step 1: Write failing test for shared helper**

Create `tests/test_aggregate_driver.py`:

```python
"""Tests for the shared aggregation driver."""

from __future__ import annotations

import json

import pytest
import yaml

from nhf_spatial_targets.aggregate._driver import update_manifest
from nhf_spatial_targets.workspace import load as load_project


@pytest.fixture()
def project(tmp_path):
    datastore = tmp_path / "datastore"
    datastore.mkdir()
    config = {
        "fabric": {"path": "", "id_col": "hru_id"},
        "datastore": str(datastore),
    }
    (tmp_path / "config.yml").write_text(yaml.dump(config))
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "abc123"}))
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    return load_project(tmp_path)


def test_update_manifest_writes_source_entry(project):
    update_manifest(
        project=project,
        source_key="foo",
        access={"type": "nasa_gesdisc", "short_name": "FOO"},
        period="2000-01-01/2009-12-31",
        output_file="data/aggregated/foo_agg.nc",
        weight_files=["weights/foo_batch0.csv"],
    )
    manifest = json.loads((project.workdir / "manifest.json").read_text())
    entry = manifest["sources"]["foo"]
    assert entry["source_key"] == "foo"
    assert entry["access_type"] == "nasa_gesdisc"
    assert entry["period"] == "2000-01-01/2009-12-31"
    assert entry["fabric_sha256"] == "abc123"
    assert entry["output_file"] == "data/aggregated/foo_agg.nc"
    assert entry["weight_files"] == ["weights/foo_batch0.csv"]
    assert "timestamp" in entry


def test_update_manifest_preserves_existing_sources(project):
    manifest_path = project.workdir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"sources": {"bar": {"source_key": "bar"}}, "steps": []})
    )
    update_manifest(
        project=project,
        source_key="foo",
        access={"type": "local"},
        period="2000/2001",
        output_file="data/aggregated/foo_agg.nc",
        weight_files=[],
    )
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["sources"].keys()) == {"foo", "bar"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_aggregate_driver.py -v`
Expected: FAIL — `ImportError: cannot import name 'update_manifest' from 'nhf_spatial_targets.aggregate._driver'`

- [ ] **Step 3: Create `_driver.py` with shared helper**

Write `src/nhf_spatial_targets/aggregate/_driver.py`:

```python
"""Shared aggregation driver: manifest helper, weight cache, and tier-1 engine."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


def update_manifest(
    project: Project,
    source_key: str,
    access: dict,
    period: str,
    output_file: str,
    weight_files: list[str],
) -> None:
    """Merge an aggregation provenance entry into ``manifest.json`` atomically.

    The manifest is keyed as ``sources[source_key]``; existing entries for
    other sources are preserved. ``period`` is stored as-is for provenance;
    ``fabric_sha256`` is read from ``fabric.json``.
    """
    manifest_path = project.manifest_path
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {project.workdir} is corrupt: {exc}"
            ) from exc
    else:
        manifest = {"sources": {}, "steps": []}

    manifest.setdefault("sources", {})

    fabric_json = project.workdir / "fabric.json"
    fabric_sha = ""
    if fabric_json.exists():
        fabric_meta = json.loads(fabric_json.read_text())
        fabric_sha = fabric_meta.get("sha256", "")

    entry: dict = {
        "source_key": source_key,
        "access_type": access.get("type", ""),
        "period": period,
        "fabric_sha256": fabric_sha,
        "output_file": output_file,
        "weight_files": list(weight_files),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Carry a few optional access identifiers through for provenance parity
    # with ssebop.py's existing behaviour.
    for extra_key in ("collection_id", "short_name", "version"):
        if extra_key in access:
            entry[extra_key] = access[extra_key]

    manifest["sources"][source_key] = entry

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=manifest_path.parent, suffix=".json.tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp_path).replace(manifest_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    logger.info("Updated manifest.json with '%s' aggregation provenance", source_key)
```

- [ ] **Step 4: Replace `_update_manifest` in `ssebop.py`**

Edit `src/nhf_spatial_targets/aggregate/ssebop.py`:

1. Delete the local `_update_manifest` function (the whole definition starting at `def _update_manifest(ws, period, meta, n_batches)`).
2. Replace the import block near the top to include the shared helper:

```python
from nhf_spatial_targets.aggregate._driver import update_manifest
```

3. Replace the `_update_manifest(ws, period, meta, n_batches)` call inside `aggregate_ssebop` with:

```python
weight_files = [
    str(Path("weights") / f"ssebop_batch{i}.csv") for i in range(n_batches)
]
access = meta["access"]
access_with_doi = {**access}
# ssebop.py previously wrote "doi" into its manifest entry; preserve that:
if meta.get("doi"):
    access_with_doi["doi"] = meta["doi"]
time_period = _parse_period(period)
update_manifest(
    project=ws,
    source_key=_SOURCE_KEY,
    access=access_with_doi,
    period=f"{time_period[0]}/{time_period[1]}",
    output_file=str(Path("data") / "aggregated" / "ssebop_agg_aet.nc"),
    weight_files=weight_files,
)
```

4. In `update_manifest` add `"doi"` to the optional `extra_key` tuple so SSEBop's manifest still carries its DOI:

```python
for extra_key in ("collection_id", "short_name", "version", "doi"):
```

- [ ] **Step 5: Re-export helper from package**

Edit `src/nhf_spatial_targets/aggregate/__init__.py`:

```python
"""gdptools-based spatial aggregation to HRU fabric."""

from nhf_spatial_targets.aggregate._driver import update_manifest

__all__ = ["update_manifest"]
```

- [ ] **Step 6: Run tests**

Run: `pixi run -e dev pytest tests/test_aggregate_driver.py tests/test_aggregate_ssebop.py -v`
Expected: PASS on both files.

- [ ] **Step 7: Format + lint**

Run: `pixi run -e dev fmt && pixi run -e dev lint`
Expected: both exit 0.

- [ ] **Step 8: Commit**

```bash
git add src/nhf_spatial_targets/aggregate/_driver.py \
        src/nhf_spatial_targets/aggregate/__init__.py \
        src/nhf_spatial_targets/aggregate/ssebop.py \
        tests/test_aggregate_driver.py
git commit -m "refactor: extract aggregation manifest helper into _driver"
```

---

## Task 2: `SourceAdapter` dataclass

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/_adapter.py`
- Modify: `src/nhf_spatial_targets/aggregate/__init__.py`
- Modify: `tests/test_aggregate_driver.py` (add adapter test)

- [ ] **Step 1: Write failing test**

Append to `tests/test_aggregate_driver.py`:

```python
import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter


def test_source_adapter_defaults():
    adapter = SourceAdapter(
        source_key="foo",
        output_name="foo_agg.nc",
        variables=["a", "b"],
    )
    assert adapter.source_crs == "EPSG:4326"
    assert adapter.x_coord == "lon"
    assert adapter.y_coord == "lat"
    assert adapter.time_coord == "time"
    assert adapter.open_hook is None


def test_source_adapter_open_hook_invocable(project):
    def _open(proj):
        return xr.Dataset({"a": (("time",), [1.0])}, coords={"time": [0]})

    adapter = SourceAdapter(
        source_key="foo",
        output_name="foo_agg.nc",
        variables=["a"],
        open_hook=_open,
    )
    ds = adapter.open_hook(project)
    assert "a" in ds
```

- [ ] **Step 2: Run and verify failure**

Run: `pixi run -e dev pytest tests/test_aggregate_driver.py::test_source_adapter_defaults -v`
Expected: FAIL — `ImportError: cannot import name 'SourceAdapter'`.

- [ ] **Step 3: Create `_adapter.py`**

Write `src/nhf_spatial_targets/aggregate/_adapter.py`:

```python
"""Declarative adapter for tier-1 gridded sources aggregated via gdptools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import xarray as xr

from nhf_spatial_targets.workspace import Project


@dataclass(frozen=True)
class SourceAdapter:
    """Declarative description of a tier-1 source for the aggregation driver.

    ``open_hook`` receives the resolved :class:`Project` and must return an
    :class:`xarray.Dataset` with CRS set and all ``variables`` present
    (including any derived variables). When ``None``, the driver opens the
    single consolidated NetCDF under ``project.raw_dir(source_key)``.
    """

    source_key: str
    output_name: str
    variables: list[str]
    x_coord: str = "lon"
    y_coord: str = "lat"
    time_coord: str = "time"
    source_crs: str = "EPSG:4326"
    open_hook: Callable[[Project], xr.Dataset] | None = field(default=None)
```

- [ ] **Step 4: Re-export**

Update `src/nhf_spatial_targets/aggregate/__init__.py`:

```python
"""gdptools-based spatial aggregation to HRU fabric."""

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import update_manifest

__all__ = ["SourceAdapter", "update_manifest"]
```

- [ ] **Step 5: Run tests**

Run: `pixi run -e dev pytest tests/test_aggregate_driver.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/nhf_spatial_targets/aggregate/_adapter.py \
        src/nhf_spatial_targets/aggregate/__init__.py \
        tests/test_aggregate_driver.py
git commit -m "feat: add SourceAdapter dataclass for tier-1 aggregators"
```

---

## Task 3: Shared batching + weight-cache helpers

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py`
- Modify: `tests/test_aggregate_driver.py`

**Context:** Factor the three boilerplate operations out of `ssebop.py`:
(1) load-and-batch the fabric, (2) compute-or-load weights for a batch, (3) aggregate a Dataset of N variables to a batch. The tier-1 driver will chain these; tier-2 modules will call them individually.

- [ ] **Step 1: Write failing test for `load_and_batch_fabric`**

Append to `tests/test_aggregate_driver.py`:

```python
import geopandas as gpd
from shapely.geometry import box

from nhf_spatial_targets.aggregate._driver import load_and_batch_fabric


@pytest.fixture()
def tiny_fabric(tmp_path):
    polys = [box(i, 0, i + 1, 1) for i in range(4)]
    gdf = gpd.GeoDataFrame(
        {"hru_id": range(4)},
        geometry=polys,
        crs="EPSG:4326",
    )
    path = tmp_path / "fabric.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def test_load_and_batch_fabric_single_batch(tiny_fabric):
    batched = load_and_batch_fabric(tiny_fabric, batch_size=500)
    assert "batch_id" in batched.columns
    assert batched["batch_id"].nunique() == 1
```

- [ ] **Step 2: Verify failure**

Run: `pixi run -e dev pytest tests/test_aggregate_driver.py::test_load_and_batch_fabric_single_batch -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement `load_and_batch_fabric`**

Append to `src/nhf_spatial_targets/aggregate/_driver.py`:

```python
import geopandas as gpd
import pandas as pd
import xarray as xr

from nhf_spatial_targets.aggregate.batching import spatial_batch


def load_and_batch_fabric(
    fabric_path: Path, batch_size: int = 500
) -> gpd.GeoDataFrame:
    """Load the fabric GeoPackage (or GeoParquet) and attach ``batch_id``."""
    fabric_path = Path(fabric_path)
    if fabric_path.suffix.lower() in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(fabric_path)
    else:
        gdf = gpd.read_file(fabric_path)
    return spatial_batch(gdf, batch_size=batch_size)
```

- [ ] **Step 4: Run test**

Run: `pixi run -e dev pytest tests/test_aggregate_driver.py::test_load_and_batch_fabric_single_batch -v`
Expected: PASS.

- [ ] **Step 5: Add `compute_or_load_weights` and `aggregate_variables_for_batch`**

Append to `src/nhf_spatial_targets/aggregate/_driver.py`:

```python
_WEIGHT_GEN_CRS = 5070  # NAD83 / CONUS Albers (equal-area)


def weight_cache_path(workdir: Path, source_key: str, batch_id: int) -> Path:
    """Return the per-batch weight CSV path."""
    return Path(workdir) / "weights" / f"{source_key}_batch{batch_id}.csv"


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
) -> pd.DataFrame:
    """Compute (or load from cache) the per-batch weight table."""
    from gdptools import UserCatData, WeightGen

    wp = weight_cache_path(workdir, source_key, batch_id)
    if wp.exists():
        logger.info("Batch %d: loading cached weights from %s", batch_id, wp)
        return pd.read_csv(wp)

    logger.info(
        "Batch %d: computing weights (%d HRUs, source_var=%s)",
        batch_id,
        len(batch_gdf),
        source_var,
    )
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
        period=[
            str(source_ds[time_coord].values[0]),
            str(source_ds[time_coord].values[-1]),
        ],
    )
    wg = WeightGen(
        user_data=user_data,
        method="serial",
        weight_gen_crs=_WEIGHT_GEN_CRS,
    )
    weights = wg.calculate_weights()
    wp.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(wp, index=False)
    logger.info("Batch %d: weights saved to %s", batch_id, wp)
    return weights


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
) -> xr.Dataset:
    """Run gdptools ``AggGen`` once per variable, merge results on ``id_col``."""
    from gdptools import AggGen, UserCatData

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
            period=[
                str(source_ds[time_coord].values[0]),
                str(source_ds[time_coord].values[-1]),
            ],
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

- [ ] **Step 6: Write test for batch aggregation using mocks**

Append to `tests/test_aggregate_driver.py`:

```python
import numpy as np
from unittest.mock import MagicMock, patch


def _fake_user_data():
    return MagicMock()


def _fake_weights():
    return pd.DataFrame({"i": [0, 1], "j": [0, 0], "wght": [0.5, 0.5], "hru_id": [0, 0]})


def _fake_agg_result(var_name, hru_ids):
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    data = np.array([[1.0] * len(hru_ids), [2.0] * len(hru_ids)])
    ds = xr.Dataset(
        {var_name: (["time", "hru_id"], data)},
        coords={"time": times, "hru_id": hru_ids},
    )
    gdf = gpd.GeoDataFrame({"hru_id": hru_ids})
    return gdf, ds


def test_aggregate_variables_for_batch_merges_variables(tiny_fabric):
    from nhf_spatial_targets.aggregate._driver import (
        aggregate_variables_for_batch,
    )

    batch_gdf = gpd.read_file(tiny_fabric)
    batch_gdf["batch_id"] = 0
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    source_ds = xr.Dataset(
        {
            "a": (["time", "lat", "lon"], np.ones((2, 2, 2))),
            "b": (["time", "lat", "lon"], np.ones((2, 2, 2)) * 2),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )

    with patch("nhf_spatial_targets.aggregate._driver.AggGen") as mock_agg, patch(
        "nhf_spatial_targets.aggregate._driver.UserCatData"
    ) as mock_ucd:
        mock_ucd.return_value = _fake_user_data()
        agg_instance = MagicMock()
        mock_agg.return_value = agg_instance
        agg_instance.calculate_agg.side_effect = [
            _fake_agg_result("a", [0, 1, 2, 3]),
            _fake_agg_result("b", [0, 1, 2, 3]),
        ]

        result = aggregate_variables_for_batch(
            batch_gdf=batch_gdf,
            source_ds=source_ds,
            variables=["a", "b"],
            source_crs="EPSG:4326",
            x_coord="lon",
            y_coord="lat",
            time_coord="time",
            id_col="hru_id",
            weights=_fake_weights(),
        )
    assert set(result.data_vars) == {"a", "b"}
    assert result.sizes["hru_id"] == 4
```

- [ ] **Step 7: Run tests**

Run: `pixi run -e dev pytest tests/test_aggregate_driver.py -v`
Expected: all PASS.

- [ ] **Step 8: Format + lint + commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/aggregate/_driver.py tests/test_aggregate_driver.py
git commit -m "feat: driver helpers for fabric batching, weights, and agg"
```

---

## Task 4: `aggregate_source` end-to-end driver

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py`
- Modify: `src/nhf_spatial_targets/aggregate/__init__.py`
- Modify: `tests/test_aggregate_driver.py`

**Context:** Wire the helpers from Task 3 into a single `aggregate_source(adapter, fabric_path, id_col, workdir, batch_size)` function that drives the full flow: load project → open source via adapter → batch fabric → per-batch weights+aggregate → concat → write NC → manifest.

- [ ] **Step 1: Write failing end-to-end test with mocks**

Append to `tests/test_aggregate_driver.py`:

```python
def test_aggregate_source_writes_multi_var_nc_and_manifest(tmp_path, tiny_fabric):
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    # --- minimal project ---
    datastore = tmp_path / "datastore"
    (datastore / "foo").mkdir(parents=True)
    # write a placeholder consolidated NC so default open_hook has something
    src_nc = datastore / "foo" / "foo.nc"
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    xr.Dataset(
        {
            "a": (["time", "lat", "lon"], np.ones((2, 2, 2))),
            "b": (["time", "lat", "lon"], np.ones((2, 2, 2)) * 2.0),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
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
    (tmp_path / "manifest.json").write_text(
        json.dumps({"sources": {}, "steps": []})
    )
    (tmp_path / "data" / "aggregated").mkdir(parents=True)
    (tmp_path / "weights").mkdir()

    adapter = SourceAdapter(
        source_key="foo",
        output_name="foo_agg.nc",
        variables=["a", "b"],
    )

    # Patch catalog.source to supply access metadata for the manifest
    fake_meta = {"access": {"type": "local_nc"}}

    with patch(
        "nhf_spatial_targets.aggregate._driver.catalog_source",
        return_value=fake_meta,
    ), patch(
        "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
        return_value=_fake_weights(),
    ), patch(
        "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch"
    ) as mock_agg_batch:
        times = pd.date_range("2000-01-01", periods=2, freq="MS")
        mock_agg_batch.return_value = xr.Dataset(
            {
                "a": (["time", "hru_id"], np.ones((2, 4))),
                "b": (["time", "hru_id"], np.ones((2, 4)) * 2.0),
            },
            coords={"time": times, "hru_id": [0, 1, 2, 3]},
        )
        out = aggregate_source(
            adapter,
            fabric_path=tiny_fabric,
            id_col="hru_id",
            workdir=tmp_path,
            batch_size=500,
        )

    assert set(out.data_vars) == {"a", "b"}
    output_nc = tmp_path / "data" / "aggregated" / "foo_agg.nc"
    assert output_nc.exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "foo" in manifest["sources"]
    assert manifest["sources"]["foo"]["output_file"] == (
        "data/aggregated/foo_agg.nc"
    )
```

- [ ] **Step 2: Verify failure**

Run: `pixi run -e dev pytest tests/test_aggregate_driver.py::test_aggregate_source_writes_multi_var_nc_and_manifest -v`
Expected: FAIL — `ImportError: cannot import name 'aggregate_source'`.

- [ ] **Step 3: Implement `aggregate_source`**

Append to `src/nhf_spatial_targets/aggregate/_driver.py`:

```python
from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.catalog import source as catalog_source
from nhf_spatial_targets.workspace import load as load_project


def _default_open_hook(project: Project, source_key: str) -> xr.Dataset:
    """Open the single consolidated NC in ``project.raw_dir(source_key)``."""
    raw_dir = project.raw_dir(source_key)
    ncs = sorted(raw_dir.glob("*.nc"))
    if not ncs:
        raise FileNotFoundError(
            f"No consolidated NC found in {raw_dir}. "
            f"Run 'nhf-targets fetch {source_key}' first."
        )
    if len(ncs) > 1:
        logger.info(
            "Multiple NCs in %s; opening first lexicographic: %s",
            raw_dir,
            ncs[0].name,
        )
    return xr.open_dataset(ncs[0])


def aggregate_source(
    adapter: SourceAdapter,
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
) -> xr.Dataset:
    """Aggregate a tier-1 source to fabric HRU polygons.

    Processes the full temporal range present in the consolidated source NC;
    no period clipping is applied. Weights are cached per batch under
    ``workdir/weights/<source_key>_batch<id>.csv``.
    """
    workdir = Path(workdir)
    project = load_project(workdir)
    meta = catalog_source(adapter.source_key)

    if adapter.open_hook is not None:
        source_ds = adapter.open_hook(project)
    else:
        source_ds = _default_open_hook(project, adapter.source_key)

    missing = [v for v in adapter.variables if v not in source_ds.data_vars]
    if missing:
        raise ValueError(
            f"{adapter.source_key}: variables {missing} missing from source "
            f"dataset (have {list(source_ds.data_vars)})"
        )

    batched = load_and_batch_fabric(fabric_path, batch_size=batch_size)
    n_batches = int(batched["batch_id"].nunique())
    logger.info(
        "%s: fabric split into %d spatial batches",
        adapter.source_key,
        n_batches,
    )

    datasets: list[xr.Dataset] = []
    for bid in sorted(batched["batch_id"].unique()):
        batch_gdf = batched[batched["batch_id"] == bid].drop(columns=["batch_id"])
        weights = compute_or_load_weights(
            batch_gdf=batch_gdf,
            source_ds=source_ds,
            source_var=adapter.variables[0],
            source_crs=adapter.source_crs,
            x_coord=adapter.x_coord,
            y_coord=adapter.y_coord,
            time_coord=adapter.time_coord,
            id_col=id_col,
            source_key=adapter.source_key,
            batch_id=int(bid),
            workdir=workdir,
        )
        ds = aggregate_variables_for_batch(
            batch_gdf=batch_gdf,
            source_ds=source_ds,
            variables=adapter.variables,
            source_crs=adapter.source_crs,
            x_coord=adapter.x_coord,
            y_coord=adapter.y_coord,
            time_coord=adapter.time_coord,
            id_col=id_col,
            weights=weights,
        )
        datasets.append(ds)

    combined = xr.concat(datasets, dim=id_col)
    logger.info(
        "%s: combined dataset: %s time steps x %s HRUs",
        adapter.source_key,
        combined.sizes.get(adapter.time_coord, "?"),
        combined.sizes.get(id_col, "?"),
    )

    output_dir = project.aggregated_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / adapter.output_name
    combined.to_netcdf(output_path)
    logger.info("%s: output written to %s", adapter.source_key, output_path)

    t0 = str(combined[adapter.time_coord].values[0])[:10]
    t1 = str(combined[adapter.time_coord].values[-1])[:10]
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

- [ ] **Step 4: Re-export from package**

Update `src/nhf_spatial_targets/aggregate/__init__.py`:

```python
"""gdptools-based spatial aggregation to HRU fabric."""

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import (
    aggregate_source,
    update_manifest,
)

__all__ = ["SourceAdapter", "aggregate_source", "update_manifest"]
```

- [ ] **Step 5: Run tests**

Run: `pixi run -e dev pytest tests/test_aggregate_driver.py tests/test_aggregate_ssebop.py -v`
Expected: all PASS.

- [ ] **Step 6: Format + lint + commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/aggregate/_driver.py \
        src/nhf_spatial_targets/aggregate/__init__.py \
        tests/test_aggregate_driver.py
git commit -m "feat: aggregate_source driver entry point"
```

---

## Task 5: ERA5-Land adapter

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/era5_land.py`
- Create: `tests/test_aggregate_era5_land.py`

**Context:** ERA5-Land fetch produces two consolidated NCs in the datastore: a daily and a monthly. Aggregation operates on the monthly NC (the one whose filename contains `monthly`). Variables: `ro`, `sro`, `ssro`. Standard lat/lon, 4326.

- [ ] **Step 1: Write failing test**

Create `tests/test_aggregate_era5_land.py`:

```python
"""Tests for ERA5-Land aggregation adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nhf_spatial_targets.aggregate.era5_land import ADAPTER, _open_monthly


@pytest.fixture()
def monthly_nc(tmp_path):
    ds_dir = tmp_path / "era5_land"
    ds_dir.mkdir()
    times = pd.date_range("2000-01-01", periods=3, freq="MS")
    ds = xr.Dataset(
        {
            "ro": (["time", "lat", "lon"], np.ones((3, 2, 2))),
            "sro": (["time", "lat", "lon"], np.ones((3, 2, 2)) * 0.5),
            "ssro": (["time", "lat", "lon"], np.ones((3, 2, 2)) * 0.5),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )
    # Distinguishable from any daily neighbour by the "monthly" token.
    ds.to_netcdf(ds_dir / "era5_land_monthly_2000_2002.nc")
    ds.to_netcdf(ds_dir / "era5_land_daily_2000_2002.nc")
    return ds_dir


def test_adapter_declares_runoff_vars():
    assert ADAPTER.source_key == "era5_land"
    assert ADAPTER.output_name == "era5_land_agg.nc"
    assert set(ADAPTER.variables) == {"ro", "sro", "ssro"}


def test_open_monthly_selects_monthly_nc(monthly_nc):
    project = MagicMock()
    project.raw_dir.return_value = monthly_nc
    ds = _open_monthly(project)
    assert "ro" in ds
    assert ds.sizes["time"] == 3
```

- [ ] **Step 2: Verify failure**

Run: `pixi run -e dev pytest tests/test_aggregate_era5_land.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement adapter**

Create `src/nhf_spatial_targets/aggregate/era5_land.py`:

```python
"""ERA5-Land aggregation adapter (runoff: ro, sro, ssro)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source
from nhf_spatial_targets.workspace import Project


def _open_monthly(project: Project) -> xr.Dataset:
    """Open the monthly consolidated ERA5-Land NC in the datastore.

    The ERA5-Land fetch stores both a daily and monthly NC; pick the monthly
    one by filename token.
    """
    raw_dir = project.raw_dir("era5_land")
    monthly_ncs = sorted(Path(raw_dir).glob("*monthly*.nc"))
    if not monthly_ncs:
        raise FileNotFoundError(
            f"No monthly ERA5-Land NC found in {raw_dir}. "
            "Run 'nhf-targets fetch era5-land' first."
        )
    return xr.open_dataset(monthly_ncs[0])


ADAPTER = SourceAdapter(
    source_key="era5_land",
    output_name="era5_land_agg.nc",
    variables=["ro", "sro", "ssro"],
    x_coord="longitude",
    y_coord="latitude",
    open_hook=_open_monthly,
)


def aggregate_era5_land(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    """Aggregate ERA5-Land monthly runoff (ro, sro, ssro) to HRU polygons."""
    return aggregate_source(
        ADAPTER, fabric_path, id_col, workdir, batch_size,
    )
```

**Note:** ERA5-Land consolidated NCs use coord names `latitude`/`longitude` (CDS convention). If the existing fetch writes `lat`/`lon` instead, drop the two `*_coord` overrides in `ADAPTER`. Verify with `ncdump -h` on an existing consolidated monthly NC before the tests pass in CI.

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_aggregate_era5_land.py -v`
Expected: PASS.

- [ ] **Step 5: Format + lint + commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/aggregate/era5_land.py tests/test_aggregate_era5_land.py
git commit -m "feat: ERA5-Land aggregation adapter"
```

---

## Task 6: GLDAS adapter (with derived `runoff_total`)

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/gldas.py`
- Create: `tests/test_aggregate_gldas.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_aggregate_gldas.py`:

```python
"""Tests for GLDAS aggregation adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nhf_spatial_targets.aggregate.gldas import ADAPTER, _open


@pytest.fixture()
def gldas_nc(tmp_path):
    ds_dir = tmp_path / "gldas_noah_v21_monthly"
    ds_dir.mkdir()
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    xr.Dataset(
        {
            "Qs_acc": (["time", "lat", "lon"], np.ones((2, 2, 2))),
            "Qsb_acc": (["time", "lat", "lon"], np.ones((2, 2, 2)) * 3.0),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    ).to_netcdf(ds_dir / "gldas_noah_v21_monthly.nc")
    return ds_dir


def test_adapter_declares_runoff_vars():
    assert ADAPTER.source_key == "gldas_noah_v21_monthly"
    assert ADAPTER.output_name == "gldas_agg.nc"
    assert set(ADAPTER.variables) == {"Qs_acc", "Qsb_acc", "runoff_total"}


def test_open_adds_runoff_total(gldas_nc):
    project = MagicMock()
    project.raw_dir.return_value = gldas_nc
    ds = _open(project)
    assert "runoff_total" in ds
    # Qs_acc (1.0) + Qsb_acc (3.0) == 4.0 for every cell
    np.testing.assert_allclose(ds["runoff_total"].values, 4.0)
    assert ds["runoff_total"].attrs["units"] == "kg m-2"
```

- [ ] **Step 2: Verify failure**

Run: `pixi run -e dev pytest tests/test_aggregate_gldas.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement adapter**

Create `src/nhf_spatial_targets/aggregate/gldas.py`:

```python
"""GLDAS-2.1 NOAH monthly runoff adapter (Qs_acc, Qsb_acc, runoff_total)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source
from nhf_spatial_targets.workspace import Project

_SOURCE_KEY = "gldas_noah_v21_monthly"


def _open(project: Project) -> xr.Dataset:
    """Open the consolidated GLDAS NC and derive ``runoff_total``."""
    raw_dir = project.raw_dir(_SOURCE_KEY)
    ncs = sorted(Path(raw_dir).glob("*.nc"))
    if not ncs:
        raise FileNotFoundError(
            f"No GLDAS NC found in {raw_dir}. "
            "Run 'nhf-targets fetch gldas' first."
        )
    ds = xr.open_dataset(ncs[0])
    total = ds["Qs_acc"] + ds["Qsb_acc"]
    total.attrs = {
        "long_name": "total runoff (Qs_acc + Qsb_acc, derived)",
        "units": "kg m-2",
        "cell_methods": "time: sum",
        "derived_from": "Qs_acc + Qsb_acc",
    }
    ds["runoff_total"] = total
    return ds


ADAPTER = SourceAdapter(
    source_key=_SOURCE_KEY,
    output_name="gldas_agg.nc",
    variables=["Qs_acc", "Qsb_acc", "runoff_total"],
    open_hook=_open,
)


def aggregate_gldas(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    """Aggregate GLDAS-2.1 NOAH monthly runoff variables to HRU polygons."""
    return aggregate_source(
        ADAPTER, fabric_path, id_col, workdir, batch_size,
    )
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_aggregate_gldas.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/aggregate/gldas.py tests/test_aggregate_gldas.py
git commit -m "feat: GLDAS-2.1 NOAH aggregation adapter with derived runoff_total"
```

---

## Task 7: MERRA-2 adapter

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/merra2.py`
- Create: `tests/test_aggregate_merra2.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_aggregate_merra2.py`:

```python
"""Tests for MERRA-2 aggregation adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nhf_spatial_targets.aggregate.merra2 import ADAPTER


@pytest.fixture()
def merra2_nc(tmp_path):
    ds_dir = tmp_path / "merra2"
    ds_dir.mkdir()
    times = pd.date_range("2000-01-15", periods=2, freq="MS")
    xr.Dataset(
        {
            "GWETTOP": (["time", "lat", "lon"], np.full((2, 2, 2), 0.3)),
            "GWETROOT": (["time", "lat", "lon"], np.full((2, 2, 2), 0.5)),
            "GWETPROF": (["time", "lat", "lon"], np.full((2, 2, 2), 0.4)),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    ).to_netcdf(ds_dir / "merra2.nc")
    return ds_dir


def test_adapter_declares_soil_wetness_vars():
    assert ADAPTER.source_key == "merra2"
    assert ADAPTER.output_name == "merra2_agg.nc"
    assert set(ADAPTER.variables) == {"GWETTOP", "GWETROOT", "GWETPROF"}
```

- [ ] **Step 2: Verify failure**

Run: `pixi run -e dev pytest tests/test_aggregate_merra2.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement adapter**

Create `src/nhf_spatial_targets/aggregate/merra2.py`:

```python
"""MERRA-2 M2TMNXLND monthly soil wetness adapter (GWETTOP/GWETROOT/GWETPROF)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="merra2",
    output_name="merra2_agg.nc",
    variables=["GWETTOP", "GWETROOT", "GWETPROF"],
)


def aggregate_merra2(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    """Aggregate MERRA-2 monthly soil wetness to HRU polygons."""
    return aggregate_source(
        ADAPTER, fabric_path, id_col, workdir, batch_size,
    )
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_aggregate_merra2.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/aggregate/merra2.py tests/test_aggregate_merra2.py
git commit -m "feat: MERRA-2 aggregation adapter"
```

---

## Task 8: NCEP/NCAR adapter

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/ncep_ncar.py`
- Create: `tests/test_aggregate_ncep_ncar.py`

**Context:** NCEP/NCAR Reanalysis is on a T62 Gaussian grid (~1.875°). The consolidated NC carries `soilw_0_10cm` and `soilw_10_200cm`. Its coord names are `lat`/`lon` (gdptools handles the non-uniform Gaussian spacing provided the lat coord is present).

- [ ] **Step 1: Write failing test**

Create `tests/test_aggregate_ncep_ncar.py`:

```python
"""Tests for NCEP/NCAR aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets.aggregate.ncep_ncar import ADAPTER


def test_adapter_declares_soil_moisture_vars():
    assert ADAPTER.source_key == "ncep_ncar"
    assert ADAPTER.output_name == "ncep_ncar_agg.nc"
    assert set(ADAPTER.variables) == {"soilw_0_10cm", "soilw_10_200cm"}
```

- [ ] **Step 2: Verify failure**

Run: `pixi run -e dev pytest tests/test_aggregate_ncep_ncar.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement adapter**

Create `src/nhf_spatial_targets/aggregate/ncep_ncar.py`:

```python
"""NCEP/NCAR Reanalysis monthly soil moisture adapter."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="ncep_ncar",
    output_name="ncep_ncar_agg.nc",
    variables=["soilw_0_10cm", "soilw_10_200cm"],
)


def aggregate_ncep_ncar(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    """Aggregate NCEP/NCAR monthly soil moisture to HRU polygons."""
    return aggregate_source(
        ADAPTER, fabric_path, id_col, workdir, batch_size,
    )
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_aggregate_ncep_ncar.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/aggregate/ncep_ncar.py tests/test_aggregate_ncep_ncar.py
git commit -m "feat: NCEP/NCAR aggregation adapter"
```

---

## Task 9: NLDAS-MOSAIC adapter

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/nldas_mosaic.py`
- Create: `tests/test_aggregate_nldas_mosaic.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_aggregate_nldas_mosaic.py`:

```python
"""Tests for NLDAS-MOSAIC aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets.aggregate.nldas_mosaic import ADAPTER


def test_adapter_declares_three_layers():
    assert ADAPTER.source_key == "nldas_mosaic"
    assert ADAPTER.output_name == "nldas_mosaic_agg.nc"
    assert set(ADAPTER.variables) == {
        "SoilM_0_10cm",
        "SoilM_10_40cm",
        "SoilM_40_200cm",
    }
```

- [ ] **Step 2: Verify failure**

Run: `pixi run -e dev pytest tests/test_aggregate_nldas_mosaic.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement adapter**

Create `src/nhf_spatial_targets/aggregate/nldas_mosaic.py`:

```python
"""NLDAS-2 MOSAIC monthly soil moisture adapter (three layers)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="nldas_mosaic",
    output_name="nldas_mosaic_agg.nc",
    variables=["SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_200cm"],
)


def aggregate_nldas_mosaic(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    """Aggregate NLDAS-2 MOSAIC monthly soil moisture to HRU polygons."""
    return aggregate_source(
        ADAPTER, fabric_path, id_col, workdir, batch_size,
    )
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_aggregate_nldas_mosaic.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/aggregate/nldas_mosaic.py tests/test_aggregate_nldas_mosaic.py
git commit -m "feat: NLDAS-MOSAIC aggregation adapter"
```

---

## Task 10: NLDAS-NOAH adapter

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/nldas_noah.py`
- Create: `tests/test_aggregate_nldas_noah.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_aggregate_nldas_noah.py`:

```python
"""Tests for NLDAS-NOAH aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets.aggregate.nldas_noah import ADAPTER


def test_adapter_declares_four_layers():
    assert ADAPTER.source_key == "nldas_noah"
    assert ADAPTER.output_name == "nldas_noah_agg.nc"
    assert set(ADAPTER.variables) == {
        "SoilM_0_10cm",
        "SoilM_10_40cm",
        "SoilM_40_100cm",
        "SoilM_100_200cm",
    }
```

- [ ] **Step 2: Verify failure**

Run: `pixi run -e dev pytest tests/test_aggregate_nldas_noah.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement adapter**

Create `src/nhf_spatial_targets/aggregate/nldas_noah.py`:

```python
"""NLDAS-2 NOAH monthly soil moisture adapter (four layers)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="nldas_noah",
    output_name="nldas_noah_agg.nc",
    variables=[
        "SoilM_0_10cm",
        "SoilM_10_40cm",
        "SoilM_40_100cm",
        "SoilM_100_200cm",
    ],
)


def aggregate_nldas_noah(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    """Aggregate NLDAS-2 NOAH monthly soil moisture to HRU polygons."""
    return aggregate_source(
        ADAPTER, fabric_path, id_col, workdir, batch_size,
    )
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_aggregate_nldas_noah.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/aggregate/nldas_noah.py tests/test_aggregate_nldas_noah.py
git commit -m "feat: NLDAS-NOAH aggregation adapter"
```

---

## Task 11: WaterGAP 2.2d adapter

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/watergap22d.py`
- Create: `tests/test_aggregate_watergap22d.py`

**Context:** WaterGAP 2.2d consolidated NC has variable `qrdif` (kg m-2 s-1, diffuse groundwater recharge), monthly, 0.5° global.

- [ ] **Step 1: Write failing test**

Create `tests/test_aggregate_watergap22d.py`:

```python
"""Tests for WaterGAP 2.2d aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets.aggregate.watergap22d import ADAPTER


def test_adapter_declares_qrdif():
    assert ADAPTER.source_key == "watergap22d"
    assert ADAPTER.output_name == "watergap22d_agg.nc"
    assert ADAPTER.variables == ["qrdif"]
```

- [ ] **Step 2: Verify failure**

Run: `pixi run -e dev pytest tests/test_aggregate_watergap22d.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement adapter**

Create `src/nhf_spatial_targets/aggregate/watergap22d.py`:

```python
"""WaterGAP 2.2d monthly diffuse groundwater recharge adapter (qrdif)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="watergap22d",
    output_name="watergap22d_agg.nc",
    variables=["qrdif"],
)


def aggregate_watergap22d(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    """Aggregate WaterGAP 2.2d monthly diffuse groundwater recharge to HRUs."""
    return aggregate_source(
        ADAPTER, fabric_path, id_col, workdir, batch_size,
    )
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_aggregate_watergap22d.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/aggregate/watergap22d.py tests/test_aggregate_watergap22d.py
git commit -m "feat: WaterGAP 2.2d aggregation adapter"
```

---

## Task 12: MOD16A2 tier-2 module (sinusoidal CRS)

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/mod16a2.py`
- Create: `tests/test_aggregate_mod16a2.py`

**Context:** MOD16A2 v061 consolidated NetCDFs carry sinusoidal projection. The adapter declares the MODIS sinusoidal PROJ string as `source_crs`; gdptools reprojects the source polygons into EPSG:5070 for weighting. Variable: `ET_500m`. 8-day cadence passes through — monthly resampling happens in `targets/aet.py`.

- [ ] **Step 1: Write failing test**

Create `tests/test_aggregate_mod16a2.py`:

```python
"""Tests for MOD16A2 aggregation adapter (sinusoidal CRS)."""

from __future__ import annotations

from nhf_spatial_targets.aggregate.mod16a2 import ADAPTER, MODIS_SINUSOIDAL_PROJ


def test_adapter_declares_sinusoidal_crs():
    assert ADAPTER.source_key == "mod16a2_v061"
    assert ADAPTER.output_name == "mod16a2_agg.nc"
    assert ADAPTER.variables == ["ET_500m"]
    assert ADAPTER.source_crs == MODIS_SINUSOIDAL_PROJ
    assert "+proj=sinu" in ADAPTER.source_crs
    assert ADAPTER.x_coord == "x"
    assert ADAPTER.y_coord == "y"
```

- [ ] **Step 2: Verify failure**

Run: `pixi run -e dev pytest tests/test_aggregate_mod16a2.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement adapter**

Create `src/nhf_spatial_targets/aggregate/mod16a2.py`:

```python
"""MOD16A2 v061 AET adapter (sinusoidal MODIS projection)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source

# MODIS sinusoidal PROJ4 string — Earth as a sphere of radius 6371007.181 m.
MODIS_SINUSOIDAL_PROJ = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
)


ADAPTER = SourceAdapter(
    source_key="mod16a2_v061",
    output_name="mod16a2_agg.nc",
    variables=["ET_500m"],
    x_coord="x",
    y_coord="y",
    source_crs=MODIS_SINUSOIDAL_PROJ,
)


def aggregate_mod16a2(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    """Aggregate MOD16A2 v061 8-day AET to HRU polygons.

    gdptools reprojects the declared sinusoidal source onto EPSG:5070 for
    area-weighted intersection.
    """
    return aggregate_source(
        ADAPTER, fabric_path, id_col, workdir, batch_size,
    )
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_aggregate_mod16a2.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/aggregate/mod16a2.py tests/test_aggregate_mod16a2.py
git commit -m "feat: MOD16A2 aggregation adapter (sinusoidal CRS)"
```

---

## Task 13: MOD10C1 tier-2 module (CI masking + valid_area_fraction)

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/mod10c1.py`
- Create: `tests/test_aggregate_mod10c1.py`

**Context:** MOD10C1 v061 consolidated NC carries `Day_CMG_Snow_Cover` (0-100) and `Snow_Spatial_QA` (confidence interval, 0-100). The aggregator must build three derived source DataArrays before handing to the shared pipeline:

- `sca = Day_CMG_Snow_Cover / 100.0` where `Snow_Spatial_QA / 100 > 0.70`, else `NaN`.
- `ci = Snow_Spatial_QA / 100.0` (passed through, no masking).
- `valid_mask = 1.0` where `Snow_Spatial_QA / 100 > 0.70`, else `0.0` (pre-NaN float indicator; aggregated as `masked_mean` this yields per-HRU valid-area fraction).

After aggregation the `valid_mask` variable is renamed `valid_area_fraction`.

This module does not use `SourceAdapter` — it calls the driver helpers directly so it can hook the rename between AggGen and write.

- [ ] **Step 1: Write failing test for masking logic**

Create `tests/test_aggregate_mod10c1.py`:

```python
"""Tests for MOD10C1 tier-2 aggregator (CI masking)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nhf_spatial_targets.aggregate.mod10c1 import build_masked_source


@pytest.fixture()
def raw_mod10c1():
    times = pd.date_range("2000-01-01", periods=1, freq="D")
    snow = np.array([[[50.0, 50.0], [50.0, 50.0]]])  # day, y, x
    qa = np.array([[[80.0, 60.0], [30.0, 100.0]]])   # CI in percent
    return xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["time", "lat", "lon"], snow),
            "Snow_Spatial_QA": (["time", "lat", "lon"], qa),
        },
        coords={"time": times, "lat": [0.25, 0.75], "lon": [0.5, 1.5]},
    )


def test_build_masked_source_variables_present(raw_mod10c1):
    out = build_masked_source(raw_mod10c1)
    assert set(["sca", "ci", "valid_mask"]).issubset(out.data_vars)


def test_sca_is_nan_where_ci_below_threshold(raw_mod10c1):
    out = build_masked_source(raw_mod10c1)
    sca = out["sca"].isel(time=0).values
    # Cells with CI 80, 100 pass (>70); CI 60, 30 fail.
    assert np.isclose(sca[0, 0], 0.5)   # CI=80  -> keep, 50/100=0.5
    assert np.isnan(sca[0, 1])          # CI=60  -> drop
    assert np.isnan(sca[1, 0])          # CI=30  -> drop
    assert np.isclose(sca[1, 1], 0.5)   # CI=100 -> keep


def test_ci_passes_through_unmasked(raw_mod10c1):
    out = build_masked_source(raw_mod10c1)
    ci = out["ci"].isel(time=0).values
    # ci is raw QA / 100 — no NaNs even where SCA was masked
    np.testing.assert_allclose(ci, np.array([[0.8, 0.6], [0.3, 1.0]]))


def test_valid_mask_is_zero_one_float(raw_mod10c1):
    out = build_masked_source(raw_mod10c1)
    vm = out["valid_mask"].isel(time=0).values
    np.testing.assert_array_equal(vm, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert out["valid_mask"].dtype.kind == "f"
```

- [ ] **Step 2: Verify failure**

Run: `pixi run -e dev pytest tests/test_aggregate_mod10c1.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement masking + aggregator**

Create `src/nhf_spatial_targets/aggregate/mod10c1.py`:

```python
"""MOD10C1 v061 daily snow-covered area aggregator (CI-masked)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from nhf_spatial_targets.aggregate._driver import (
    aggregate_variables_for_batch,
    compute_or_load_weights,
    load_and_batch_fabric,
    update_manifest,
)
from nhf_spatial_targets.catalog import source as catalog_source
from nhf_spatial_targets.workspace import load as load_project

logger = logging.getLogger(__name__)

_SOURCE_KEY = "mod10c1_v061"
_CI_THRESHOLD = 0.70   # TM 6-B10: keep cells where CI > 0.70
_OUTPUT_NAME = "mod10c1_agg.nc"


def build_masked_source(ds: xr.Dataset) -> xr.Dataset:
    """Derive ``sca``, ``ci``, ``valid_mask`` from raw MOD10C1 variables.

    - ``sca``        = Day_CMG_Snow_Cover / 100, NaN where CI <= 0.70.
    - ``ci``         = Snow_Spatial_QA / 100 (passed through, unmasked).
    - ``valid_mask`` = 1.0 where CI > 0.70, 0.0 otherwise (float so
                       area-weighted mean gives valid-area fraction per HRU).
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


def _open(project) -> xr.Dataset:
    raw_dir = project.raw_dir(_SOURCE_KEY)
    ncs = sorted(Path(raw_dir).glob("*.nc"))
    if not ncs:
        raise FileNotFoundError(
            f"No MOD10C1 NC found in {raw_dir}. "
            "Run 'nhf-targets fetch mod10c1' first."
        )
    return xr.open_dataset(ncs[0])


def aggregate_mod10c1(
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
) -> xr.Dataset:
    """Aggregate MOD10C1 v061 daily SCA to HRU polygons with CI masking.

    Writes three variables to ``data/aggregated/mod10c1_agg.nc``:
    ``sca``, ``ci``, and ``valid_area_fraction``.
    """
    workdir = Path(workdir)
    project = load_project(workdir)
    meta = catalog_source(_SOURCE_KEY)

    raw = _open(project)
    source_ds = build_masked_source(raw)
    variables = ["sca", "ci", "valid_mask"]

    batched = load_and_batch_fabric(fabric_path, batch_size=batch_size)
    n_batches = int(batched["batch_id"].nunique())
    logger.info("mod10c1: fabric split into %d spatial batches", n_batches)

    datasets: list[xr.Dataset] = []
    for bid in sorted(batched["batch_id"].unique()):
        batch_gdf = batched[batched["batch_id"] == bid].drop(columns=["batch_id"])
        weights = compute_or_load_weights(
            batch_gdf=batch_gdf,
            source_ds=source_ds,
            source_var="sca",
            source_crs="EPSG:4326",
            x_coord="lon",
            y_coord="lat",
            time_coord="time",
            id_col=id_col,
            source_key=_SOURCE_KEY,
            batch_id=int(bid),
            workdir=workdir,
        )
        ds = aggregate_variables_for_batch(
            batch_gdf=batch_gdf,
            source_ds=source_ds,
            variables=variables,
            source_crs="EPSG:4326",
            x_coord="lon",
            y_coord="lat",
            time_coord="time",
            id_col=id_col,
            weights=weights,
        )
        datasets.append(ds)

    combined = xr.concat(datasets, dim=id_col)
    combined = combined.rename({"valid_mask": "valid_area_fraction"})
    combined["valid_area_fraction"].attrs = {
        "long_name": "fraction of HRU area that passed CI filter",
        "units": "1",
        "ci_threshold": _CI_THRESHOLD,
    }

    output_dir = project.aggregated_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _OUTPUT_NAME
    combined.to_netcdf(output_path)
    logger.info("mod10c1: output written to %s", output_path)

    t0 = str(combined["time"].values[0])[:10]
    t1 = str(combined["time"].values[-1])[:10]
    update_manifest(
        project=project,
        source_key=_SOURCE_KEY,
        access=meta.get("access", {}),
        period=f"{t0}/{t1}",
        output_file=str(Path("data") / "aggregated" / _OUTPUT_NAME),
        weight_files=[
            str(Path("weights") / f"{_SOURCE_KEY}_batch{i}.csv")
            for i in range(n_batches)
        ],
    )
    return combined
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_aggregate_mod10c1.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/aggregate/mod10c1.py tests/test_aggregate_mod10c1.py
git commit -m "feat: MOD10C1 SCA aggregation with CI masking and valid_area"
```

---

## Task 14: CLI subcommands for tier-1 aggregators

**Files:**
- Modify: `src/nhf_spatial_targets/cli.py`
- Create: `tests/test_cli_agg.py`

**Context:** Each source gets an `agg <source>` subcommand. All tier-1/2 commands share a boilerplate shape — read project config, resolve fabric path + id_col, dispatch to the source's `aggregate_*` function, report result.

- [ ] **Step 1: Write failing smoke test for dispatch**

Create `tests/test_cli_agg.py`:

```python
"""Smoke tests for the `agg` CLI subcommands."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from cyclopts import App


@pytest.mark.parametrize(
    "subcommand,target_fn",
    [
        ("era5-land", "nhf_spatial_targets.cli.aggregate_era5_land"),
        ("gldas", "nhf_spatial_targets.cli.aggregate_gldas"),
        ("merra2", "nhf_spatial_targets.cli.aggregate_merra2"),
        ("ncep-ncar", "nhf_spatial_targets.cli.aggregate_ncep_ncar"),
        ("nldas-mosaic", "nhf_spatial_targets.cli.aggregate_nldas_mosaic"),
        ("nldas-noah", "nhf_spatial_targets.cli.aggregate_nldas_noah"),
        ("watergap22d", "nhf_spatial_targets.cli.aggregate_watergap22d"),
        ("mod16a2", "nhf_spatial_targets.cli.aggregate_mod16a2"),
        ("mod10c1", "nhf_spatial_targets.cli.aggregate_mod10c1"),
    ],
)
def test_agg_subcommand_dispatches(
    subcommand, target_fn, tmp_path, monkeypatch
):
    import json
    import yaml
    from nhf_spatial_targets.cli import app

    # Minimal project
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": str(tmp_path / "fabric.gpkg"), "id_col": "nhm_id"},
                "datastore": str(tmp_path / "datastore"),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))

    with patch(target_fn) as mock_agg:
        app(["agg", subcommand, "--project-dir", str(tmp_path)])
    mock_agg.assert_called_once()
    _args, kwargs = mock_agg.call_args
    assert kwargs.get("id_col", "nhm_id") == "nhm_id"
```

- [ ] **Step 2: Verify failure**

Run: `pixi run -e dev pytest tests/test_cli_agg.py -v`
Expected: FAIL (subcommands not registered).

- [ ] **Step 3: Add subcommands to `cli.py`**

Near the top of `src/nhf_spatial_targets/cli.py`, add imports of the nine new aggregator functions:

```python
from nhf_spatial_targets.aggregate.era5_land import aggregate_era5_land
from nhf_spatial_targets.aggregate.gldas import aggregate_gldas
from nhf_spatial_targets.aggregate.merra2 import aggregate_merra2
from nhf_spatial_targets.aggregate.mod10c1 import aggregate_mod10c1
from nhf_spatial_targets.aggregate.mod16a2 import aggregate_mod16a2
from nhf_spatial_targets.aggregate.ncep_ncar import aggregate_ncep_ncar
from nhf_spatial_targets.aggregate.nldas_mosaic import aggregate_nldas_mosaic
from nhf_spatial_targets.aggregate.nldas_noah import aggregate_nldas_noah
from nhf_spatial_targets.aggregate.watergap22d import aggregate_watergap22d
```

Then add one helper function and nine subcommand wrappers at the bottom of the file, before the final `main = app.meta` line:

```python
def _run_tier_agg(
    aggregate_fn,
    label: str,
    workdir: Path,
    batch_size: int,
) -> None:
    """Common boilerplate for tier-1/tier-2 aggregator CLI wrappers."""
    from rich.console import Console

    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)
    if not (workdir / "fabric.json").exists():
        print(
            f"Error: fabric.json not found in {workdir}. "
            "Run 'nhf-targets validate' first.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        cfg = yaml.safe_load((workdir / "config.yml").read_text())
    except yaml.YAMLError as exc:
        print(f"Error: Cannot parse config.yml: {exc}", file=sys.stderr)
        sys.exit(1)
    fabric_path = cfg["fabric"]["path"]
    id_col = cfg["fabric"].get("id_col", "nhm_id")

    console = Console()
    console.print(f"[bold]Aggregating {label} (batch_size={batch_size})...[/bold]")
    try:
        ds = aggregate_fn(
            fabric_path=fabric_path,
            id_col=id_col,
            workdir=workdir,
            batch_size=batch_size,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during %s aggregation", label)
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    console.print(
        f"[green]{label} aggregation complete: "
        f"{ds.sizes.get('time', '?')} time steps x "
        f"{ds.sizes.get(id_col, '?')} HRUs[/green]"
    )


@agg_app.command(name="era5-land")
def agg_era5_land_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate ERA5-Land monthly runoff to HRU polygons."""
    _run_tier_agg(aggregate_era5_land, "ERA5-Land", workdir, batch_size)


@agg_app.command(name="gldas")
def agg_gldas_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate GLDAS-2.1 NOAH monthly runoff to HRU polygons."""
    _run_tier_agg(aggregate_gldas, "GLDAS", workdir, batch_size)


@agg_app.command(name="merra2")
def agg_merra2_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate MERRA-2 monthly soil wetness to HRU polygons."""
    _run_tier_agg(aggregate_merra2, "MERRA-2", workdir, batch_size)


@agg_app.command(name="ncep-ncar")
def agg_ncep_ncar_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate NCEP/NCAR monthly soil moisture to HRU polygons."""
    _run_tier_agg(aggregate_ncep_ncar, "NCEP/NCAR", workdir, batch_size)


@agg_app.command(name="nldas-mosaic")
def agg_nldas_mosaic_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate NLDAS-2 MOSAIC monthly soil moisture to HRU polygons."""
    _run_tier_agg(aggregate_nldas_mosaic, "NLDAS-MOSAIC", workdir, batch_size)


@agg_app.command(name="nldas-noah")
def agg_nldas_noah_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate NLDAS-2 NOAH monthly soil moisture to HRU polygons."""
    _run_tier_agg(aggregate_nldas_noah, "NLDAS-NOAH", workdir, batch_size)


@agg_app.command(name="watergap22d")
def agg_watergap22d_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate WaterGAP 2.2d monthly diffuse recharge to HRU polygons."""
    _run_tier_agg(aggregate_watergap22d, "WaterGAP 2.2d", workdir, batch_size)


@agg_app.command(name="mod16a2")
def agg_mod16a2_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate MOD16A2 v061 8-day AET to HRU polygons."""
    _run_tier_agg(aggregate_mod16a2, "MOD16A2", workdir, batch_size)


@agg_app.command(name="mod10c1")
def agg_mod10c1_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate MOD10C1 v061 daily SCA to HRU polygons."""
    _run_tier_agg(aggregate_mod10c1, "MOD10C1", workdir, batch_size)
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_cli_agg.py tests/test_aggregate_ssebop.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/cli.py tests/test_cli_agg.py
git commit -m "feat: CLI subcommands for tier-1 and tier-2 aggregators"
```

---

## Task 15: `agg all` command

**Files:**
- Modify: `src/nhf_spatial_targets/cli.py`
- Modify: `tests/test_cli_agg.py`

**Context:** Mirror `fetch all` — iterate the nine new aggregators (plus SSEBop with a config-driven period) and run them sequentially, stopping on first failure.

- [ ] **Step 1: Write failing test**

Append to `tests/test_cli_agg.py`:

```python
def test_agg_all_runs_every_source(tmp_path):
    import json
    import yaml
    from nhf_spatial_targets.cli import app

    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": str(tmp_path / "fabric.gpkg"), "id_col": "nhm_id"},
                "datastore": str(tmp_path / "datastore"),
                "targets": {
                    "aet": {"ssebop_period": "2000/2023"},
                },
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))

    target_fns = [
        "nhf_spatial_targets.cli.aggregate_era5_land",
        "nhf_spatial_targets.cli.aggregate_gldas",
        "nhf_spatial_targets.cli.aggregate_merra2",
        "nhf_spatial_targets.cli.aggregate_ncep_ncar",
        "nhf_spatial_targets.cli.aggregate_nldas_mosaic",
        "nhf_spatial_targets.cli.aggregate_nldas_noah",
        "nhf_spatial_targets.cli.aggregate_watergap22d",
        "nhf_spatial_targets.cli.aggregate_mod16a2",
        "nhf_spatial_targets.cli.aggregate_mod10c1",
    ]
    with ExitStack() as stack:
        mocks = [stack.enter_context(patch(fn)) for fn in target_fns]
        app(["agg", "all", "--project-dir", str(tmp_path)])
    for m in mocks:
        m.assert_called_once()
```

Add `from contextlib import ExitStack` to imports at top of file.

- [ ] **Step 2: Verify failure**

Run: `pixi run -e dev pytest tests/test_cli_agg.py::test_agg_all_runs_every_source -v`
Expected: FAIL — subcommand not registered.

- [ ] **Step 3: Implement `agg all`**

Add to `src/nhf_spatial_targets/cli.py` (after the individual `agg_*_cmd` definitions):

```python
@agg_app.command(name="all")
def agg_all_cmd(
    workdir: Annotated[
        Path,
        Parameter(name=["--project-dir"], help="Project created by 'nhf-targets init'."),
    ],
    batch_size: Annotated[
        int,
        Parameter(name="--batch-size", help="Target HRUs per spatial batch."),
    ] = 500,
):
    """Aggregate every registered source for this project.

    Runs tier-1/tier-2 aggregators in sequence; stops on first failure.
    SSEBop is not included here — run ``agg ssebop --period`` separately.
    """
    from rich.console import Console

    console = Console()
    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    sources: list[tuple[str, callable]] = [
        ("era5-land", aggregate_era5_land),
        ("gldas", aggregate_gldas),
        ("merra2", aggregate_merra2),
        ("ncep-ncar", aggregate_ncep_ncar),
        ("nldas-mosaic", aggregate_nldas_mosaic),
        ("nldas-noah", aggregate_nldas_noah),
        ("watergap22d", aggregate_watergap22d),
        ("mod16a2", aggregate_mod16a2),
        ("mod10c1", aggregate_mod10c1),
    ]
    for label, fn in sources:
        console.print(f"\n[bold]{'─' * 60}[/bold]")
        _run_tier_agg(fn, label, workdir, batch_size)

    console.print(
        f"\n[bold green]All {len(sources)} sources aggregated successfully.[/bold green]"
    )
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_cli_agg.py -v`
Expected: PASS.

- [ ] **Step 5: Format + lint + commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/cli.py tests/test_cli_agg.py
git commit -m "feat: agg all command to run every tier-1/2 aggregator"
```

---

## Task 16: Documentation touch-up

**Files:**
- Modify: `README.md` (if it enumerates `agg` commands)
- Modify: `CLAUDE.md` pipeline commands section

- [ ] **Step 1: Update CLAUDE.md command list**

Edit the "Environment & Commands" section of `CLAUDE.md` to add:

```bash
# Aggregate sources to fabric (full source period; clipping happens in targets)
pixi run nhf-targets agg era5-land    --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg gldas        --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg merra2       --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg ncep-ncar    --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg nldas-mosaic --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg nldas-noah   --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg watergap22d  --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg mod16a2      --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg mod10c1      --project-dir /data/nhf-runs/my-run
pixi run nhf-targets agg all          --project-dir /data/nhf-runs/my-run
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: list new aggregation subcommands in CLAUDE.md"
```

---

## Final check

- [ ] **Full test suite passes**

Run: `pixi run -e dev fmt-check && pixi run -e dev lint && pixi run -e dev test`
Expected: exit code 0.

- [ ] **Manual smoke test (optional, requires a real project + datastore)**

Pick one small source and run end-to-end against a real project:

```bash
pixi run nhf-targets agg watergap22d --project-dir /data/nhf-runs/test-run
```

Verify:
- `data/aggregated/watergap22d_agg.nc` created with `qrdif` variable on `(time, nhm_id)`.
- `weights/watergap22d_batch*.csv` created.
- `manifest.json` updated with a `sources.watergap22d` entry.

---

## Notes for the implementer

- **Coordinate names may drift.** Most fetch modules consolidate to `lat`/`lon`, but ERA5-Land CDS downloads typically use `latitude`/`longitude`. Run `ncdump -h <datastore>/<source>/*.nc` on a real file before each adapter's first real run and adjust the adapter's `x_coord`/`y_coord` overrides if needed. The adapter tests don't exercise the real coord names — they verify the adapter *declaration*, not gdptools integration.
- **Weight cache invalidation.** The existing `ssebop.py` does not invalidate weight caches when the fabric changes (same behaviour preserved here). If the fabric is edited, delete `weights/*.csv` manually before re-running.
- **Running against a real datastore** requires `fabric.json` (via `nhf-targets validate`) and consolidated source NCs (via `nhf-targets fetch <source>`).
- **gdptools real calls** are not exercised by unit tests. Integration tests (`pytest.mark.integration`) should be added later — out of scope for this plan.
