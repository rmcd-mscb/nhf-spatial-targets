# UA SWE Aggregate Layer (PR-B of #237) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `aggregate/ua_swe.py` — a `SourceAdapter` that area-weights UA daily SWE / snow-depth to the HRU fabric and derives a depth-thresholded `snow_covered_fraction` binary pre-aggregation — plus its CLI wiring, tests, docs, and inspect notebook.

**Architecture:** Templated on `aggregate/snodas.py` (pre-projected EPSG:5070 CONUS SWE, `masked_mean`) and `aggregate/mod10c1.py` (binary pre-aggregate hook on a synthesized variable). The nonlinear `snow_depth > depth_threshold_mm` binary is computed per-pixel in a `pre_aggregate_hook` (threshold and area-weighted mean do not commute), re-NaNing fill pixels (`.where(snow_depth.notnull())`) so border HRUs are not diluted by phantom zeros — the **opposite** NaN policy from mod10c1's `valid_mask`. The threshold is read from project config (`snow_covered_area.depth_threshold_mm`, default 1.0 — the key is wired by PR-D, so PR-B reads it defensively), bound into the hooks by closure, and stamped on `snow_covered_fraction` so the agg NC records the threshold that defined it.

**Tech Stack:** Python ≥3.11, xarray, gdptools (via the shared `aggregate/_driver.py`), cyclopts CLI, pytest, pixi.

**Scope guardrails (from the spec — do NOT do these here):**
- No `rebuild_manifest` / `release/lineage.py` change. The ua_swe agg NC uses the standard `data/aggregated/<key>/` layout, which the generic catalog-keyed projection already handles. The `depth_threshold_mm` → manifest-step-params lift and the publish staleness gate are **PR-B2**, out of scope.
- No `targets/sca.py`, `targets/swe.py`, `defaults.py`, or config-schema changes. Adding `ua_swe` as a SWE source is **PR-C**; the SCA multi-source refactor and the `depth_threshold_mm` config stub are **PR-D**.
- No change to the consolidator, fetch path, or catalog `ua_swe` entry (PR-A2, merged).

**Caldera test discipline:** Run `pixi run -e dev fmt && pixi run -e dev lint` locally. Do **not** run the full pytest suite locally — run only targeted serial tests (`pixi run -e dev test -k <name>`, no `-n`/xdist) when needed, then push and let GitHub Actions run the full suite. Commit via `pixi run git commit`. Fresh worktree needs `pixi install -e dev` once before committing.

---

## File Structure

- **Create** `src/nhf_spatial_targets/aggregate/ua_swe.py` — the adapter factory, the two closure-bound hooks, the config-threshold resolver, and `aggregate_ua_swe`.
- **Create** `tests/test_aggregate_ua_swe.py` — pure-function unit tests on the hooks + adapter contract + config-threshold resolution + CLI registration. (Repo convention `test_aggregate_<src>.py`; deviates from the spec's literal `test_ua_swe_aggregate.py` to match 20 sibling files.)
- **Modify** `src/nhf_spatial_targets/cli/__init__.py` — import + `__all__` entry for `aggregate_ua_swe`.
- **Modify** `src/nhf_spatial_targets/cli/agg.py` — new `agg ua-swe` command (with `--period`, mirroring `snodas`) + registration in `agg all`.
- **Modify** `CLAUDE.md` — add the `agg ua-swe` line to the aggregation command list.
- **Create** `notebooks/aggregated/inspect_aggregated_ua_swe.ipynb` — per-source HRU aggregate inspection (eyeball `snow_covered_fraction` against a snowy region).
- **Modify** `notebooks/aggregated/inspect_aggregated_swe.ipynb` and `inspect_aggregated_snow_covered_area.ipynb` — one `datasets={}` registry entry each.

---

## Task 1: `aggregate/ua_swe.py` module + unit tests

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/ua_swe.py`
- Test: `tests/test_aggregate_ua_swe.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_aggregate_ua_swe.py`:

```python
"""Tests for the UA SWE aggregator (``aggregate/ua_swe.py``).

Covers the derived ``snow_covered_fraction`` binary hook (the NaN policy
that is the OPPOSITE of mod10c1's ``valid_mask`` — fill pixels must become
NaN, never a phantom 0.0), the depth-threshold provenance stamp, the static
adapter contract, the post-aggregate re-stamp, and config-driven threshold
resolution. Heavy gdptools aggregation is left to the integration suite;
these are pure-function unit tests on the hooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nhf_spatial_targets import catalog
from nhf_spatial_targets.aggregate.ua_swe import (
    ADAPTER,
    DEFAULT_DEPTH_THRESHOLD_MM,
    aggregate_ua_swe,
    build_adapter,
    make_post_aggregate_hook,
)


@pytest.fixture()
def raw_ua_swe():
    """One-day synthetic UA grid: snowy, bare, and off-CONUS fill (NaN) pixels.

    snow_depth in mm; 2x3 grid. Row 0: 50, 50, 0 (two snowy, one bare).
    Row 1: 0, NaN, NaN (one bare, two fill).
    """
    times = pd.date_range("2000-01-15", periods=1, freq="D")
    depth = np.array([[[50.0, 50.0, 0.0], [0.0, np.nan, np.nan]]])
    swe = np.array([[[10.0, 10.0, 0.0], [0.0, np.nan, np.nan]]])
    return xr.Dataset(
        {
            "swe": (["time", "y", "x"], swe),
            "snow_depth": (["time", "y", "x"], depth),
        },
        coords={"time": times, "y": [0.0, 4000.0], "x": [0.0, 4000.0, 8000.0]},
    )


def test_adapter_static_contract():
    assert ADAPTER.source_key == "ua_swe"
    assert ADAPTER.output_name == "ua_swe_agg.nc"
    assert ADAPTER.variables == ("swe", "snow_depth", "snow_covered_fraction")
    assert ADAPTER.source_crs == "EPSG:5070"
    assert ADAPTER.files_glob == "daily/ua_swe_daily_*.nc"
    assert ADAPTER.stat_method == "masked_mean"
    assert ADAPTER.output_cadence == "daily"
    assert ADAPTER.pre_aggregate_hook is not None
    assert ADAPTER.post_aggregate_hook is not None
    # grid_variable defaults to "swe" (a genuine raw var), so raw_grid_variable
    # resolves correctly without an explicit override.
    assert ADAPTER.grid_variable == "swe"
    assert ADAPTER.raw_grid_variable == "swe"


def test_snow_covered_fraction_is_a_catalog_variable():
    """CF attrs the hook stamps must line up with the catalog declaration."""
    names = {v.get("name") for v in catalog.source("ua_swe")["variables"]}
    assert "snow_covered_fraction" in names


def test_hook_derives_binary_with_nan_on_fill(raw_ua_swe):
    out = ADAPTER.pre_aggregate_hook(raw_ua_swe)
    assert "snow_covered_fraction" in out.data_vars
    scf = out["snow_covered_fraction"].isel(time=0).values
    assert scf[0, 0] == 1.0  # depth 50 > 1
    assert scf[0, 1] == 1.0  # depth 50 > 1
    assert scf[0, 2] == 0.0  # depth 0 -> 0 (real "no snow")
    assert scf[1, 0] == 0.0  # depth 0 -> 0
    assert np.isnan(scf[1, 1])  # fill -> NaN (NOT a phantom 0.0)
    assert np.isnan(scf[1, 2])  # fill -> NaN
    # swe / snow_depth pass through untouched.
    np.testing.assert_array_equal(
        out["snow_depth"].values, raw_ua_swe["snow_depth"].values
    )


def test_all_nan_depth_hru_yields_nan_not_zero():
    """THE load-bearing test: an all-fill footprint must be NaN, never 0.

    Without ``.where(snow_depth.notnull())``, ``NaN > 1`` is False, so every
    fill pixel would become a hard 0.0 "no snow" vote and an all-fill HRU
    would read 0% covered instead of unobserved.
    """
    times = pd.date_range("2000-01-15", periods=1, freq="D")
    depth = np.full((1, 2, 2), np.nan)
    ds = xr.Dataset(
        {"snow_depth": (["time", "y", "x"], depth)},
        coords={"time": times, "y": [0.0, 4000.0], "x": [0.0, 4000.0]},
    )
    scf = ADAPTER.pre_aggregate_hook(ds)["snow_covered_fraction"].values
    assert np.isnan(scf).all()
    assert not (scf == 0.0).any()


def test_half_snow_footprint_mean_is_half():
    """Pre-aggregation binary mean over a half-snowy footprint is 0.5.

    This is the count-of-snowy-pixels-before-averaging quantity that makes
    the binary belong in the hook rather than in targets/sca.py.
    """
    times = pd.date_range("2000-01-15", periods=1, freq="D")
    depth = np.array([[[50.0, 50.0, 0.0, 0.0]]])
    ds = xr.Dataset(
        {"snow_depth": (["time", "y", "x"], depth)},
        coords={"time": times, "y": [0.0], "x": [0.0, 4000.0, 8000.0, 12000.0]},
    )
    scf = ADAPTER.pre_aggregate_hook(ds)["snow_covered_fraction"].isel(time=0).values
    assert np.nanmean(scf) == 0.5


def test_depth_threshold_attr_stamped_default(raw_ua_swe):
    scf = ADAPTER.pre_aggregate_hook(raw_ua_swe)["snow_covered_fraction"]
    assert scf.attrs["depth_threshold_mm"] == DEFAULT_DEPTH_THRESHOLD_MM


def test_cf_var_attrs_match_catalog(raw_ua_swe):
    """(d) CF-1.6: snow_covered_fraction carries catalog units/long_name/cell_methods."""
    cat_scf = next(
        v
        for v in catalog.source("ua_swe")["variables"]
        if v.get("name") == "snow_covered_fraction"
    )
    attrs = ADAPTER.pre_aggregate_hook(raw_ua_swe)["snow_covered_fraction"].attrs
    assert attrs["units"] == cat_scf["cf_units"]
    assert attrs["long_name"] == cat_scf["long_name"]
    assert attrs["cell_methods"] == cat_scf["cell_methods"]


def test_custom_threshold_closure_flips_binary_and_stamp():
    """build_adapter(5.0) must gate at 5 mm and stamp 5.0, proving the closure."""
    times = pd.date_range("2000-01-15", periods=1, freq="D")
    depth = np.array([[[3.0, 10.0]]])  # 3 mm: bare at t=5; 10 mm: snowy at t=5
    ds = xr.Dataset(
        {"snow_depth": (["time", "y", "x"], depth)},
        coords={"time": times, "y": [0.0], "x": [0.0, 4000.0]},
    )
    out = build_adapter(5.0).pre_aggregate_hook(ds)["snow_covered_fraction"]
    vals = out.isel(time=0).values
    assert vals[0] == 0.0  # 3 > 5 is False
    assert vals[1] == 1.0  # 10 > 5 is True
    assert out.attrs["depth_threshold_mm"] == 5.0


def test_post_aggregate_hook_restamps_attrs():
    """gdptools may drop synthesized-var attrs; the post hook re-asserts them."""
    hru = xr.Dataset(
        {"snow_covered_fraction": (["nhm_id"], [0.5, 0.2])},
        coords={"nhm_id": [1, 2]},
    )
    hru["snow_covered_fraction"].attrs = {}  # simulate attr loss through agg
    out = make_post_aggregate_hook(1.0)(hru)
    assert out["snow_covered_fraction"].attrs["depth_threshold_mm"] == 1.0
    assert out["snow_covered_fraction"].attrs["units"] == "1"


def test_aggregate_ua_swe_reads_default_threshold(tmp_path, monkeypatch):
    """No depth_threshold_mm in config -> default 1.0 bound into the hook."""
    (tmp_path / "config.yml").write_text("fabric:\n  path: /x\n  id_col: nhm_id\n")
    captured = {}

    def fake_agg_source(adapter, *args, **kwargs):
        captured["adapter"] = adapter

    monkeypatch.setattr(
        "nhf_spatial_targets.aggregate.ua_swe.aggregate_source", fake_agg_source
    )
    aggregate_ua_swe("/fab", "nhm_id", tmp_path, 500)
    times = pd.date_range("2000-01-15", periods=1, freq="D")
    ds = xr.Dataset(
        {"snow_depth": (["time", "y", "x"], np.array([[[50.0]]]))},
        coords={"time": times, "y": [0.0], "x": [0.0]},
    )
    scf = captured["adapter"].pre_aggregate_hook(ds)["snow_covered_fraction"]
    assert scf.attrs["depth_threshold_mm"] == 1.0


def test_aggregate_ua_swe_reads_custom_threshold(tmp_path, monkeypatch):
    """depth_threshold_mm: 5.0 under targets.snow_covered_area is honored."""
    (tmp_path / "config.yml").write_text(
        "fabric:\n  path: /x\n  id_col: nhm_id\n"
        "targets:\n  snow_covered_area:\n    depth_threshold_mm: 5.0\n"
    )
    captured = {}

    def fake_agg_source(adapter, *args, **kwargs):
        captured["adapter"] = adapter

    monkeypatch.setattr(
        "nhf_spatial_targets.aggregate.ua_swe.aggregate_source", fake_agg_source
    )
    aggregate_ua_swe("/fab", "nhm_id", tmp_path, 500)
    times = pd.date_range("2000-01-15", periods=1, freq="D")
    ds = xr.Dataset(
        {"snow_depth": (["time", "y", "x"], np.array([[[3.0]]]))},
        coords={"time": times, "y": [0.0], "x": [0.0]},
    )
    scf = captured["adapter"].pre_aggregate_hook(ds)["snow_covered_fraction"]
    assert scf.isel(time=0).values[0] == 0.0  # 3 > 5 False
    assert scf.attrs["depth_threshold_mm"] == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -k test_aggregate_ua_swe`
Expected: FAIL — `ModuleNotFoundError: No module named 'nhf_spatial_targets.aggregate.ua_swe'`

- [ ] **Step 3: Write the module**

Create `src/nhf_spatial_targets/aggregate/ua_swe.py`:

```python
"""University of Arizona daily SWE / snow-depth aggregator (CONUS).

Reads per-calendar-year consolidated NCs at
``<datastore>/ua_swe/daily/ua_swe_daily_<year>.nc`` (written by PR-A2's
``fetch.ua_swe.consolidate_calendar_year_ua_swe``), pre-projected to
EPSG:5070 at consolidate time, and emits area-weighted HRU means via the
shared aggregation driver.

Output variables:

- ``swe`` (``kg m-2`` ≡ mm) and ``snow_depth`` (mm): native pass-through.
- ``snow_covered_fraction`` (``1``): **derived** in the
  ``pre_aggregate_hook`` as the per-pixel binary
  ``snow_depth > depth_threshold_mm``, then area-weighted to the HRU. The
  binary must precede aggregation because a threshold is nonlinear and does
  not commute with the area-weighted mean (``mean(d > t) != mean(d) > t``).
  See ``docs/architecture/transformation-pipeline.md`` and the design spec
  ``docs/superpowers/specs/2026-06-09-ua-swe-wiring-design.md``.

``.where(snow_depth.notnull())`` is load-bearing: ``NaN > t`` is ``False``,
so a naive ``.astype(float)`` would turn every off-CONUS / fill pixel into a
hard ``0.0`` ("no snow") that dilutes the fraction for any HRU touching the
CONUS boundary or an internal gap. Re-NaNing those pixels and pairing with
``stat_method="masked_mean"`` excludes them (the HRU is NaN only when its
whole footprint is unobserved). This is the **opposite** NaN policy from
``mod10c1.py``'s ``valid_mask``, which deliberately lets unobserved pixels
become ``0.0`` because its derived ``valid_area_fraction`` means "fraction of
HRU with usable observations." Same mechanical shape, opposite intent —
copying mod10c1's hook verbatim yields a subtly wrong SCA fraction along
every coastline.

``depth_threshold_mm`` is conceptually an SCA-target knob
(``snow_covered_area.depth_threshold_mm`` in project config) but must be
evaluated on the pixel grid, so it is read at agg time, bound into the hooks
via closure, and stamped on ``snow_covered_fraction`` so the agg NC records
the threshold that defined it. Changing the threshold invalidates the agg
NC's ``snow_covered_fraction`` (re-run ``agg ua-swe``); ``swe`` /
``snow_depth`` are untouched. The ``depth_threshold_mm`` config key is wired
by PR-D; this module reads it defensively, defaulting to
``DEFAULT_DEPTH_THRESHOLD_MM`` when absent.

The attrs the ``pre_aggregate_hook`` stamps on the synthesized variable do
not reliably survive gdptools' aggregation, so a closure-bound
``post_aggregate_hook`` re-asserts them on the HRU-scale result — the same
pattern ``mod10c1.py`` uses to re-stamp its derived ``valid_area_fraction``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import xarray as xr
import yaml

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source

_SOURCE_KEY = "ua_swe"
_OUTPUT_NAME = "ua_swe_agg.nc"
_DEPTH_VAR = "snow_depth"
_SCF_VAR = "snow_covered_fraction"

#: Pixel snow-depth threshold (mm) for the snow-covered binary, used when the
#: project config carries no ``snow_covered_area.depth_threshold_mm`` (that
#: key is introduced by PR-D, so this module must read it defensively).
DEFAULT_DEPTH_THRESHOLD_MM = 1.0


def _scf_cf_attrs() -> dict[str, str]:
    """CF-1.6 variable attrs for ``snow_covered_fraction``, from the catalog.

    Read from ``catalog/sources.yml`` (the single source of truth) rather
    than hardcoded, so a units / long_name correction flows through on the
    next aggregation.
    """
    from nhf_spatial_targets import catalog

    for var in catalog.source(_SOURCE_KEY)["variables"]:
        if var.get("name") == _SCF_VAR:
            return {
                "long_name": var["long_name"],
                "units": var["cf_units"],
                "cell_methods": var["cell_methods"],
            }
    raise KeyError(
        f"{_SOURCE_KEY}: catalog has no {_SCF_VAR!r} variable entry; cannot "
        f"stamp CF attrs on the derived snow-covered fraction."
    )


def make_pre_aggregate_hook(
    depth_threshold_mm: float,
) -> Callable[[xr.Dataset], xr.Dataset]:
    """Return a pre-aggregate hook that derives ``snow_covered_fraction``.

    The hook adds the per-pixel binary ``snow_depth > depth_threshold_mm`` as
    a float field, re-NaNing fill pixels (``.where(snow_depth.notnull())``) so
    ``masked_mean`` excludes them. ``swe`` and ``snow_depth`` pass through
    untouched. ``depth_threshold_mm`` is closed over so the module-level
    adapter can be re-parameterized per run.
    """
    cf_attrs = _scf_cf_attrs()

    def _hook(ds: xr.Dataset) -> xr.Dataset:
        if _DEPTH_VAR not in ds.data_vars:
            raise KeyError(
                f"{_SOURCE_KEY}: pre_aggregate_hook expected raw variable "
                f"{_DEPTH_VAR!r}, found {list(ds.data_vars)}."
            )
        depth = ds[_DEPTH_VAR]
        scf = (depth > depth_threshold_mm).astype("float64").where(depth.notnull())
        scf.attrs = {**cf_attrs, "depth_threshold_mm": float(depth_threshold_mm)}
        return ds.assign({_SCF_VAR: scf})

    return _hook


def make_post_aggregate_hook(
    depth_threshold_mm: float,
) -> Callable[[xr.Dataset], xr.Dataset]:
    """Return a post-aggregate hook that re-stamps ``snow_covered_fraction`` attrs.

    gdptools does not reliably carry a synthesized variable's attrs through
    aggregation (the same reason ``mod10c1.py`` re-stamps its derived
    ``valid_area_fraction`` post-aggregation), so re-assert the CF attrs and
    the ``depth_threshold_mm`` provenance stamp on the HRU-scale result.
    """
    cf_attrs = _scf_cf_attrs()

    def _hook(year_ds: xr.Dataset) -> xr.Dataset:
        if _SCF_VAR in year_ds.data_vars:
            year_ds[_SCF_VAR].attrs = {
                **cf_attrs,
                "depth_threshold_mm": float(depth_threshold_mm),
            }
        return year_ds

    return _hook


def build_adapter(
    depth_threshold_mm: float = DEFAULT_DEPTH_THRESHOLD_MM,
) -> SourceAdapter:
    """Build the ua_swe ``SourceAdapter`` with hooks bound to the threshold."""
    return SourceAdapter(
        source_key=_SOURCE_KEY,
        output_cadence="daily",
        output_name=_OUTPUT_NAME,
        variables=("swe", "snow_depth", "snow_covered_fraction"),
        # Pre-projected at consolidate time (PR-A2); matches the driver's
        # WEIGHT_GEN_CRS so gdptools skips reprojection during weight gen.
        source_crs="EPSG:5070",
        files_glob="daily/ua_swe_daily_*.nc",
        # grid_variable defaults to variables[0] = "swe" (a genuine raw var),
        # so raw_grid_variable resolves without an explicit override.
        pre_aggregate_hook=make_pre_aggregate_hook(depth_threshold_mm),
        post_aggregate_hook=make_post_aggregate_hook(depth_threshold_mm),
        # CONUS-masked product + a masked binary -> masked_mean (mirrors
        # snodas #151). Under the default "mean", one NaN border pixel would
        # poison every HRU it touches.
        stat_method="masked_mean",
    )


#: Module-level adapter at the default threshold, for the static contract and
#: tests. ``aggregate_ua_swe`` re-builds a per-run adapter from project config.
ADAPTER = build_adapter()


def _resolve_depth_threshold_mm(workdir: Path) -> float:
    """Read ``snow_covered_area.depth_threshold_mm`` from the project config.

    Defaults to ``DEFAULT_DEPTH_THRESHOLD_MM`` when absent — the config key is
    wired by PR-D, so PR-B must not assume it exists.
    """
    from nhf_spatial_targets.defaults import apply_defaults

    cfg = apply_defaults(yaml.safe_load((Path(workdir) / "config.yml").read_text()))
    sca_cfg = (cfg.get("targets") or {}).get("snow_covered_area") or {}
    return float(sca_cfg.get("depth_threshold_mm", DEFAULT_DEPTH_THRESHOLD_MM))


def aggregate_ua_swe(
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
    period: str | None = None,
    *,
    worker_index: int = 0,
    n_workers: int = 1,
) -> None:
    """Aggregate UA daily SWE / snow-depth / snow-covered-fraction to HRUs.

    Reads ``snow_covered_area.depth_threshold_mm`` (default
    ``DEFAULT_DEPTH_THRESHOLD_MM``) from the project config, binds it into the
    derive/stamp hooks, and runs the shared driver. ``worker_index`` /
    ``n_workers`` enable SLURM-array year sharding (default ``(0, 1)`` serial).
    """
    adapter = build_adapter(_resolve_depth_threshold_mm(workdir))
    aggregate_source(
        adapter,
        fabric_path,
        id_col,
        workdir,
        batch_size,
        period=period,
        worker_index=worker_index,
        n_workers=n_workers,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev test -k test_aggregate_ua_swe`
Expected: PASS (all tests in `tests/test_aggregate_ua_swe.py`)

- [ ] **Step 5: Lint + format**

Run: `pixi run -e dev fmt && pixi run -e dev lint`
Expected: no changes / no errors

- [ ] **Step 6: Commit**

```bash
git add src/nhf_spatial_targets/aggregate/ua_swe.py tests/test_aggregate_ua_swe.py
pixi run git commit -m "feature(#237): aggregate/ua_swe.py — derive snow_covered_fraction pre-agg (PR-B)"
```

---

## Task 2: CLI wiring (`agg ua-swe` command + `agg all` + export)

**Files:**
- Modify: `src/nhf_spatial_targets/cli/__init__.py` (import + `__all__`)
- Modify: `src/nhf_spatial_targets/cli/agg.py` (new command + `agg all` entry)
- Test: `tests/test_aggregate_ua_swe.py` (add CLI-registration tests)

- [ ] **Step 1: Add the failing CLI tests**

Append to `tests/test_aggregate_ua_swe.py`:

```python
def test_cli_exposes_aggregate_ua_swe():
    """The agg CLI resolves aggregators via the cli package namespace."""
    from nhf_spatial_targets import cli

    assert hasattr(cli, "aggregate_ua_swe")
    assert "aggregate_ua_swe" in cli.__all__


def test_cli_registers_agg_ua_swe():
    """`nhf-targets agg ua-swe` is a registered sub-command."""
    from nhf_spatial_targets.cli.agg import agg_app

    names = {cmd for cmd in agg_app}  # cyclopts App is iterable over command names
    assert "ua-swe" in names
```

> Note: if `agg_app` is not directly iterable in this cyclopts version, mirror the existing `test_cli_registers_agg_snodas` in `tests/test_aggregate_snodas.py` (around line 86) verbatim — it already proves the registration idiom for this codebase. Read that test first and copy its mechanism.

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev test -k "test_cli_exposes_aggregate_ua_swe or test_cli_registers_agg_ua_swe"`
Expected: FAIL — `aggregate_ua_swe` not exported / `ua-swe` not registered

- [ ] **Step 3: Export `aggregate_ua_swe` from the cli package**

In `src/nhf_spatial_targets/cli/__init__.py`, add the import next to the other aggregate imports (keep alphabetical-ish ordering near `snodas`/`watergap22d`):

```python
from nhf_spatial_targets.aggregate.ua_swe import aggregate_ua_swe
```

and add to the `__all__` list (next to `"aggregate_snodas"`):

```python
    "aggregate_ua_swe",
```

- [ ] **Step 4: Add the `agg ua-swe` command**

In `src/nhf_spatial_targets/cli/agg.py`, add a command modeled on `agg_snodas_cmd` (daily CONUS source with an optional `--period` clip). Place it after `agg_snodas_cmd`:

```python
@agg_app.command(name="ua-swe")
def agg_ua_swe_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
    period: Annotated[
        str | None,
        Parameter(
            name=["--period", "-p"],
            help=(
                "Optional 'YYYY/YYYY' clip applied at agg time. UA SWE "
                "consolidated NCs span calendar years 1982-2022; pass e.g. "
                "'2000/2010' to restrict aggregation. Omit to aggregate every "
                "year present in the datastore."
            ),
        ),
    ] = None,
):
    """Aggregate UA daily SWE / snow-depth / snow-covered-fraction to HRU polygons."""
    _run_tier_agg(
        _resolve_agg_fn("aggregate_ua_swe"),
        "UA SWE",
        workdir,
        batch_size,
        period=period,
        worker_index=worker_index,
        n_workers=n_workers,
    )
```

- [ ] **Step 5: Register `ua-swe` in `agg all`**

In `agg_all_cmd`'s `sources` list (after the `("snodas", ...)` entry), add:

```python
        ("ua-swe", _resolve_agg_fn("aggregate_ua_swe")),
```

> `agg all` does not pass `--period`, so ua_swe aggregates every calendar year present — consistent with how `snodas` behaves in `agg all`.

- [ ] **Step 6: Run the CLI tests + the module tests together**

Run: `pixi run -e dev test -k test_aggregate_ua_swe`
Expected: PASS (module + CLI tests)

- [ ] **Step 7: Lint + format**

Run: `pixi run -e dev fmt && pixi run -e dev lint`
Expected: no errors

- [ ] **Step 8: Commit**

```bash
git add src/nhf_spatial_targets/cli/__init__.py src/nhf_spatial_targets/cli/agg.py tests/test_aggregate_ua_swe.py
pixi run git commit -m "feature(#237): wire 'agg ua-swe' CLI command + agg all (PR-B)"
```

---

## Task 3: CLAUDE.md aggregation command list

**Files:**
- Modify: `CLAUDE.md` (the `# Aggregate sources to fabric` block in "Environment & Commands")

- [ ] **Step 1: Add the `agg ua-swe` line**

In `CLAUDE.md`, in the aggregation command list (the block containing `pixi run nhf-targets agg snodas ...`), add a line after the `snodas` line, matching the surrounding column alignment:

```
pixi run nhf-targets agg ua-swe       --project-dir /data/nhf-runs/my-run
```

- [ ] **Step 2: Verify the edit reads correctly**

Run: `grep -n "agg ua-swe" CLAUDE.md`
Expected: one match in the aggregation command block.

- [ ] **Step 3: Commit (docs-only, `--no-verify` allowed)**

```bash
git add CLAUDE.md
pixi run git commit --no-verify -m "docs(#237): add 'agg ua-swe' to CLAUDE.md command list (PR-B)"
```

---

## Task 4: Inspect notebooks (per-source + register in SWE/SCA aggregated notebooks)

**Files:**
- Create: `notebooks/aggregated/inspect_aggregated_ua_swe.ipynb`
- Modify: `notebooks/aggregated/inspect_aggregated_swe.ipynb` (one `datasets={}` registry entry)
- Modify: `notebooks/aggregated/inspect_aggregated_snow_covered_area.ipynb` (one `datasets={}` registry entry)

> **Notebooks are best built and run interactively against a real project + aggregated NC, not headless.** This task is split out so it can be done by the operator (or by Claude with the user present) after Tasks 1–2 land and `agg ua-swe` has produced a `data/aggregated/ua_swe/ua_swe_*_agg.nc`. The registry edits in the two multi-source notebooks are "one dict entry each" (the established `datasets={...}`-drives-every-cell pattern); the new per-source notebook eyeballs `snow_covered_fraction` against a known snowy region (e.g. the Colorado Rockies / Sierra Nevada in midwinter) to confirm the fraction is in [0, 1] and high where expected.

- [ ] **Step 1: Inspect the existing registry shape**

Read `notebooks/aggregated/inspect_aggregated_swe.ipynb` and `inspect_aggregated_snow_covered_area.ipynb` and locate the `datasets = {...}` registry cell (mirrors `notebooks/aggregated/_helpers.py`). Note the exact entry shape used for an existing per-source aggregate (e.g. `snodas`, `mod10c1_v061`): the dict key, the aggregated-NC path glob, the variable name, and any per-source label/units.

- [ ] **Step 2: Register `ua_swe` in the SWE aggregated notebook**

Add one `datasets` entry for `ua_swe` pointing at `data/aggregated/ua_swe/ua_swe_*_agg.nc`, variable `swe` (units `mm`), mirroring the `snodas` entry. Per the deck-annotation convention, ensure any "sources dropped from the target build" note still reads correctly.

- [ ] **Step 3: Register `ua_swe` in the SCA aggregated notebook**

Add one `datasets` entry for `ua_swe` pointing at the same agg NC, variable `snow_covered_fraction` (units `1`), so it appears alongside MOD10C1's snow-cover field.

- [ ] **Step 4: Create `inspect_aggregated_ua_swe.ipynb`**

Model on `inspect_aggregated_swe.ipynb`. Single `datasets={}` registry entry for `ua_swe` with all three variables (`swe`, `snow_depth`, `snow_covered_fraction`); choropleth cells over the HRU fabric for a midwinter date; a sanity cell asserting `0 <= snow_covered_fraction <= 1` and reporting the `depth_threshold_mm` attr. Use bullet-list summaries, not markdown tables with empty leading header columns (VSCode's renderer drops those).

- [ ] **Step 5: Eyeball against a snowy region (operator/interactive)**

Open `inspect_aggregated_ua_swe.ipynb`, point it at a project with a real `ua_swe` agg NC, and confirm `snow_covered_fraction` is high (→1) over the midwinter Rockies/Sierra and 0/NaN over the bare desert and off-CONUS. Confirm the stamped `depth_threshold_mm` attr is present and equals the project's threshold.

- [ ] **Step 6: Commit (docs/notebooks-only, `--no-verify` allowed)**

```bash
git add notebooks/aggregated/inspect_aggregated_ua_swe.ipynb \
        notebooks/aggregated/inspect_aggregated_swe.ipynb \
        notebooks/aggregated/inspect_aggregated_snow_covered_area.ipynb
pixi run git commit --no-verify -m "docs(#237): inspect_aggregated_ua_swe notebook + SWE/SCA registry entries (PR-B)"
```

---

## Task 5: Final verification + PR

**Files:** none (verification + PR)

- [ ] **Step 1: Local lint/format gate**

Run: `pixi run -e dev fmt-check && pixi run -e dev lint`
Expected: clean. (Per caldera discipline, do **not** run the full pytest suite locally.)

- [ ] **Step 2: Push and let CI run the full suite**

```bash
git push -u origin <feature-branch>
```
Then watch the GitHub Actions run; the full pytest suite runs there.

- [ ] **Step 3: Open the PR**

Open a PR titled `feature(#237): ua_swe aggregate layer (PR-B)` referencing #237's checklist (check off PR-B). Body should note: derives `snow_covered_fraction` pre-aggregation with the load-bearing `.where(notnull())` NaN policy (opposite of mod10c1); `depth_threshold_mm` read defensively (PR-D wires the config key); stamp re-asserted in a `post_aggregate_hook`; no manifest/target/config changes (those are PR-B2/C/D). Confirm CI is green before requesting review.

---

## Self-Review Notes (spec coverage)

- **Adapter** (spec §"Adapter"): Task 1 — `build_adapter` with `source_key`, `output_cadence="daily"`, `output_name="ua_swe_agg.nc"`, the three variables, `source_crs="EPSG:5070"`, `files_glob`, `stat_method="masked_mean"`. ✓
- **Pre-aggregate hook + load-bearing `.where(notnull())`** (spec §"Pre-aggregate hook"): Task 1 module + `test_all_nan_depth_hru_yields_nan_not_zero`, `test_half_snow_footprint_mean_is_half`, `test_hook_derives_binary_with_nan_on_fill`. ✓
- **Threshold read defensively + bound by closure + stamped** (spec §"Threshold coupling"; user note): Task 1 `_resolve_depth_threshold_mm` (default 1.0), `make_pre/post_aggregate_hook` closures, `depth_threshold_mm` attr; `test_depth_threshold_attr_stamped_default`, `test_custom_threshold_closure_flips_binary_and_stamp`, `test_aggregate_ua_swe_reads_{default,custom}_threshold`. ✓
- **Tests (a)–(d)** (spec §"Tests + notebook"): all-NaN→NaN (a), half-snow→~0.5 (b), threshold attr (c), CF-1.6 attrs (d, `test_cf_var_attrs_match_catalog` + `test_post_aggregate_hook_restamps_attrs`). ✓
- **Notebook + registry + CLAUDE.md** (spec §"Docs + notebooks (PR-B)"): Tasks 3 & 4. ✓
- **Out of scope** held: no PR-B2 manifest lift / publish gate, no PR-C/D target/config/defaults edits, no consolidator/catalog edits. ✓
