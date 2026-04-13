# ERA5-Land + GLDAS Runoff Replacement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the unimplemented `nhm_mwbm` source and replace the runoff target with a `multi_source_minmax` over ERA5-Land (Copernicus CDS) + GLDAS-2.1 NOAH monthly (NASA GES DISC); add ERA5-Land `ssro` as a third recharge source; remove MWBM from the AET source list.

**Architecture:** Two new fetch modules follow the existing pattern (`pangaea.py`, `merra2.py`): per-source download → CF-compliant consolidation in the shared datastore → manifest update. ERA5-Land is unique in needing hourly→daily→monthly aggregation; daily and monthly NCs are both persisted. Targets reuse existing `aggregate/gdptools_agg.py` and `multi_source_minmax` machinery.

**Tech Stack:** Python 3.11+, `cdsapi`, `earthaccess`, `xarray`, `gdptools`, `cyclopts`, `pixi`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-04-13-era5-land-gldas-runoff-design.md`

---

## File Structure

**Create:**
- `src/nhf_spatial_targets/fetch/era5_land.py` — CDS download, hourly→daily→monthly aggregation
- `src/nhf_spatial_targets/fetch/gldas.py` — earthaccess download, bbox-clip, derive `runoff_total`
- `tests/test_era5_land.py`
- `tests/test_gldas.py`
- `tests/test_run_target.py`

**Modify:**
- `catalog/sources.yml` — drop `nhm_mwbm`, add `era5_land` and `gldas_noah_v21_monthly`
- `catalog/variables.yml` — runoff sources/method, drop MWBM from aet, add era5_land to recharge
- `src/nhf_spatial_targets/targets/run.py` — implement multi-source minmax
- `src/nhf_spatial_targets/targets/aet.py` — drop MWBM source reference
- `src/nhf_spatial_targets/targets/rch.py` — add ERA5-Land source
- `src/nhf_spatial_targets/validate.py` — CDS credential check
- `src/nhf_spatial_targets/cli.py` — register `fetch era5-land` and `fetch gldas` commands
- `src/nhf_spatial_targets/fetch/consolidate.py` — extend `apply_cf_metadata` if new sources need entries
- `pixi.toml` — add `cdsapi` dependency
- `tests/test_catalog.py` — replace MWBM assertions with new sources
- `CLAUDE.md` — Known Gaps update
- `README.md` — runoff source description

**Constants module (inline in `fetch/era5_land.py`):**
- `BBOX_NWSE = [53.0, -125.0, 24.7, -66.0]` (CDS area parameter, snapped to 0.1°)

---

## Task 1: Drop `nhm_mwbm` from catalog and update `tests/test_catalog.py`

**Files:**
- Modify: `catalog/sources.yml` (remove `nhm_mwbm:` block, lines 11-38)
- Modify: `catalog/variables.yml` (remove `nhm_mwbm` from `aet.sources`, line 45)
- Modify: `tests/test_catalog.py`

- [ ] **Step 1: Read existing test to understand assertions**

Run: `cat tests/test_catalog.py | head -60`

- [ ] **Step 2: Write failing test for MWBM removal**

In `tests/test_catalog.py`, add:

```python
def test_nhm_mwbm_removed():
    """nhm_mwbm has been replaced by ERA5-Land + GLDAS."""
    from nhf_spatial_targets import catalog
    sources = catalog.sources()
    assert "nhm_mwbm" not in sources, (
        "nhm_mwbm should be removed; replaced by era5_land + gldas_noah_v21_monthly"
    )
    aet = catalog.variable("aet")
    assert "nhm_mwbm" not in aet["sources"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pixi run -e dev test -- tests/test_catalog.py::test_nhm_mwbm_removed -v`
Expected: FAIL (`nhm_mwbm` still in sources)

- [ ] **Step 4: Remove the `nhm_mwbm:` block from `catalog/sources.yml`**

Delete lines 11-38 inclusive (the whole block including the `# RUNOFF` comment header — keep the divider comment but remove the source). The runoff section will be repopulated in Task 2.

- [ ] **Step 5: Remove `nhm_mwbm` from `catalog/variables.yml` `aet.sources`**

In the `aet:` block, change:
```yaml
    sources:
      - nhm_mwbm
      - mod16a2_v061
      - ssebop
```
to:
```yaml
    sources:
      - mod16a2_v061
      - ssebop
```

- [ ] **Step 6: Run all catalog tests**

Run: `pixi run -e dev test -- tests/test_catalog.py -v`
Expected: PASS (test from Step 2 passes; pre-existing tests still pass — if any pre-existing test references `nhm_mwbm`, update it to drop the reference)

- [ ] **Step 7: Commit**

```bash
git add catalog/sources.yml catalog/variables.yml tests/test_catalog.py
git commit -m "refactor: drop unimplemented nhm_mwbm source from catalog"
```

---

## Task 2: Add `era5_land` and `gldas_noah_v21_monthly` to `catalog/sources.yml`

**Files:**
- Modify: `catalog/sources.yml`
- Modify: `tests/test_catalog.py`

- [ ] **Step 1: Write failing tests for new sources**

Append to `tests/test_catalog.py`:

```python
def test_era5_land_source_present():
    from nhf_spatial_targets import catalog
    s = catalog.source("era5_land")
    assert s["access"]["type"] == "copernicus_cds"
    assert s["access"]["dataset"] == "reanalysis-era5-land"
    var_names = [v["name"] for v in s["variables"]]
    assert var_names == ["ro", "sro", "ssro"]
    assert s["time_step"] == "hourly (aggregated to daily and monthly)"
    assert s["status"] == "current"


def test_gldas_source_present():
    from nhf_spatial_targets import catalog
    s = catalog.source("gldas_noah_v21_monthly")
    assert s["access"]["short_name"] == "GLDAS_NOAH025_M"
    assert s["access"]["version"] == "2.1"
    var_names = [v["name"] for v in s["variables"]]
    assert "Qs_acc" in var_names
    assert "Qsb_acc" in var_names
    assert "runoff_total" in var_names  # derived
    assert s["status"] == "current"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -- tests/test_catalog.py::test_era5_land_source_present tests/test_catalog.py::test_gldas_source_present -v`
Expected: FAIL (KeyError)

- [ ] **Step 3: Add `era5_land` block under the RUNOFF divider in `catalog/sources.yml`**

```yaml
  era5_land:
    name: ECMWF ERA5-Land Reanalysis
    description: >
      ECMWF ERA5-Land hourly reanalysis. Total runoff (ro), surface runoff
      (sro), and sub-surface runoff (ssro) downloaded for CONUS plus
      contributing watersheds (Canada/Mexico). Used as a source for the
      runoff calibration target (ro) and the recharge calibration target
      (ssro, as drainage proxy).
    citations:
      - "Muñoz-Sabater, J., and others, 2021, doi:10.5194/essd-13-4349-2021"
    access:
      type: copernicus_cds
      dataset: reanalysis-era5-land
      url: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
      bbox_nwse: [53.0, -125.0, 24.7, -66.0]
      bbox_notes: >
        Encompasses CONUS contributing watersheds (Canada/Mexico) with
        ~10 km buffer, snapped to the 0.1° ERA5-Land grid.
      notes: >
        Access via cdsapi (Copernicus CDS account required). Hourly
        accumulated fields reset at 00 UTC. Pipeline aggregates
        hourly→daily (via .diff('time') + resample sum) and daily→monthly.
        Both daily and monthly consolidated NCs are stored in the datastore.
    variables:
      - name: ro
        long_name: total runoff
        cf_units: "m"
        cell_methods: "time: sum"
        notes: Used for runoff target.
      - name: sro
        long_name: surface runoff
        cf_units: "m"
        cell_methods: "time: sum"
        notes: Stored for future use.
      - name: ssro
        long_name: sub-surface runoff
        cf_units: "m"
        cell_methods: "time: sum"
        notes: Used as recharge proxy in recharge target.
    time_step: hourly (aggregated to daily and monthly)
    period: "1979/present"
    spatial_extent: CONUS+contributing-watersheds
    spatial_resolution: 0.1 degree
    units: m of water equivalent
    license: Copernicus license (free, attribution)
    status: current
```

- [ ] **Step 4: Add `gldas_noah_v21_monthly` block under the RUNOFF divider**

```yaml
  gldas_noah_v21_monthly:
    name: GLDAS-2.1 NOAH Monthly Land Surface Runoff
    description: >
      Global Land Data Assimilation System version 2.1, NOAH land
      surface model, monthly product. Storm surface runoff (Qs_acc)
      and baseflow-groundwater runoff (Qsb_acc) summed at consolidation
      time to derived total runoff (runoff_total). Used as second source
      for the runoff calibration target.
    citations:
      - "Rodell, M., and others, 2004, doi:10.1175/BAMS-85-3-381"
    access:
      type: nasa_gesdisc
      short_name: GLDAS_NOAH025_M
      version: "2.1"
      url: https://disc.gsfc.nasa.gov/datasets/GLDAS_NOAH025_M_2.1/summary
      bbox_nwse: [53.0, -125.0, 24.7, -66.0]
      notes: >
        Access via earthaccess (NASA Earthdata login required). Granules
        are global (~few MB monthly); downloaded full and clipped to bbox
        at consolidation time, mirroring the merra2 pattern.
    variables:
      - name: Qs_acc
        long_name: storm surface runoff
        cf_units: "kg m-2"
        cell_methods: "time: sum"
      - name: Qsb_acc
        long_name: baseflow-groundwater runoff
        cf_units: "kg m-2"
        cell_methods: "time: sum"
      - name: runoff_total
        long_name: total runoff (Qs_acc + Qsb_acc, derived)
        cf_units: "kg m-2"
        cell_methods: "time: sum"
        derived: true
    time_step: monthly
    period: "2000/present"
    spatial_extent: global (clipped to CONUS+contributing-watersheds)
    spatial_resolution: 0.25 degree
    units: kg m-2 (equivalent to mm over the month)
    license: public domain (NASA)
    status: current
```

- [ ] **Step 5: Run catalog tests**

Run: `pixi run -e dev test -- tests/test_catalog.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add catalog/sources.yml tests/test_catalog.py
git commit -m "feat(catalog): add era5_land and gldas_noah_v21_monthly sources"
```

---

## Task 3: Update `catalog/variables.yml` runoff and recharge

**Files:**
- Modify: `catalog/variables.yml`
- Modify: `tests/test_catalog.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_catalog.py`:

```python
def test_runoff_uses_era5_and_gldas():
    from nhf_spatial_targets import catalog
    v = catalog.variable("runoff")
    assert v["sources"] == ["era5_land", "gldas_noah_v21_monthly"]
    assert v["range_method"] == "multi_source_minmax"


def test_recharge_includes_era5_land():
    from nhf_spatial_targets import catalog
    v = catalog.variable("recharge")
    assert "era5_land" in v["sources"]
    assert v["range_method"] == "normalized_minmax"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -- tests/test_catalog.py -k "runoff_uses or recharge_includes" -v`
Expected: FAIL

- [ ] **Step 3: Replace the `runoff:` block in `catalog/variables.yml`**

Replace the entire `runoff:` block (currently lines 15-32 referencing `basin_cfs`, `nhm_mwbm`, `mwbm_uncertainty`) with:

```yaml
  runoff:
    prms_variable: basin_cfs
    description: >
      Monthly basin mean runoff. Range is min/max across two reanalysis
      sources per HRU and time step.
    prms_reference: "TM 6-B7 (Markstrom et al. 2015)"
    time_step: monthly
    period: "2000/2023"
    units: cfs
    sources:
      - era5_land
      - gldas_noah_v21_monthly
    range_method: multi_source_minmax
    range_notes: >
      ERA5-Land 'ro' (m water equivalent) and GLDAS-2.1 NOAH
      'Qs_acc + Qsb_acc' (kg/m² ≡ mm) are aggregated to HRU polygons
      via gdptools, harmonized to mm/month, then converted to cfs using
      HRU area and days-in-month. Per-HRU per-month:
        lower_bound = min(era5, gldas), upper_bound = max(era5, gldas).
      Replaces the original NHM-MWBM source which had pre-computed
      uncertainty bounds (Bock et al. 2018) but no public fetch path.
    normalize: false
    output_format: netcdf
```

- [ ] **Step 4: Update the `recharge:` block sources list**

In `catalog/variables.yml` change:
```yaml
    sources:
      - reitz2017
      - watergap22d
```
to:
```yaml
    sources:
      - reitz2017
      - watergap22d
      - era5_land    # ssro (sub-surface runoff) as recharge proxy
```

Also extend `range_notes` with this paragraph (append at end of the block):
```
      ERA5-Land contribution: ssro is summed monthly→annual in mm,
      then normalized 0-1 over 2000-2009 alongside the other sources
      before the multi-source min/max.
```

- [ ] **Step 5: Run tests**

Run: `pixi run -e dev test -- tests/test_catalog.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add catalog/variables.yml tests/test_catalog.py
git commit -m "feat(catalog): switch runoff to era5/gldas, add era5 to recharge"
```

---

## Task 4: Add `cdsapi` dependency and CDS credential validation

**Files:**
- Modify: `pixi.toml`
- Modify: `src/nhf_spatial_targets/validate.py`
- Test: `tests/test_validate.py` (create or extend)

- [ ] **Step 1: Read current `validate.py` to find credential check pattern**

Run: `grep -n "credentials\|earthdata" src/nhf_spatial_targets/validate.py`

- [ ] **Step 2: Add `cdsapi` to `pixi.toml`**

Open `pixi.toml`. Under `[pypi-dependencies]` (or `[dependencies]` if `cdsapi` is in conda-forge — prefer conda-forge), add:

```toml
cdsapi = "*"
```

- [ ] **Step 3: Run pixi install**

Run: `pixi install`
Expected: cdsapi resolved and installed.

- [ ] **Step 4: Write failing test for CDS credential validation**

Add to `tests/test_validate.py` (create file if missing):

```python
from pathlib import Path
import pytest
import yaml
from nhf_spatial_targets.validate import validate_credentials


def test_validate_credentials_missing_cds(tmp_path):
    creds = tmp_path / ".credentials.yml"
    creds.write_text(yaml.safe_dump({"earthdata": {"username": "u", "password": "p"}}))
    with pytest.raises(ValueError, match="cds"):
        validate_credentials(creds, required=["earthdata", "cds"])


def test_validate_credentials_with_cds(tmp_path):
    creds = tmp_path / ".credentials.yml"
    creds.write_text(yaml.safe_dump({
        "earthdata": {"username": "u", "password": "p"},
        "cds": {"url": "https://cds.climate.copernicus.eu/api", "key": "uid:abc"},
    }))
    validate_credentials(creds, required=["earthdata", "cds"])  # no raise
```

- [ ] **Step 5: Run test to verify failure**

Run: `pixi run -e dev test -- tests/test_validate.py -v`
Expected: FAIL (function may not exist or signature differs).

- [ ] **Step 6: Update `validate.py`**

Locate the existing credential-check logic. Refactor (or add) so that `validate_credentials(path, required=[...])`:
- Reads YAML at `path`
- For each name in `required`:
  - `earthdata`: requires keys `username` and `password`, both non-empty
  - `cds`: requires keys `url` and `key`, both non-empty
- Raises `ValueError` with the missing section name in the message

Wire it into the existing `validate` CLI command so that, if any catalog source's `access.type` is `copernicus_cds`, `cds` is added to the required list. Use the catalog API (`catalog.sources()`) to determine this — do not hard-code.

- [ ] **Step 7: Run tests**

Run: `pixi run -e dev test -- tests/test_validate.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add pixi.toml pixi.lock src/nhf_spatial_targets/validate.py tests/test_validate.py
git commit -m "feat(validate): add cdsapi dep and CDS credential check"
```

---

## Task 5: ERA5-Land hourly→daily aggregation function (pure, testable)

**Files:**
- Create: `src/nhf_spatial_targets/fetch/era5_land.py`
- Test: `tests/test_era5_land.py`

This task isolates the aggregation math so it can be tested without the network.

- [ ] **Step 1: Write failing test for hourly→daily**

Create `tests/test_era5_land.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import pytest

from nhf_spatial_targets.fetch.era5_land import hourly_to_daily


def _make_hourly(start: str, hours: int, value_per_hour: float = 1.0) -> xr.DataArray:
    """Synthetic ERA5-Land accumulated field.

    ERA5-Land accumulates from 00 UTC: at hour H, value = H * per_hour
    (resetting to 0 at the next 00 UTC, where it then represents the
    accumulation step from 23->00 of the prior day).
    """
    times = pd.date_range(start, periods=hours, freq="1h")
    vals = np.zeros(hours)
    for i, t in enumerate(times):
        # value at time t = accumulation since 00 UTC of t's date
        hours_since_midnight = t.hour if t.hour != 0 else (24 if i > 0 else 0)
        vals[i] = hours_since_midnight * value_per_hour
    da = xr.DataArray(
        vals.reshape(-1, 1, 1),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": [40.0], "longitude": [-100.0]},
        name="ro",
    )
    return da


def test_hourly_to_daily_full_24h():
    # 48 hours starting 2020-01-01 00:00 → two full days of accumulation
    da = _make_hourly("2020-01-01 00:00", hours=49, value_per_hour=0.001)
    daily = hourly_to_daily(da)
    # Two complete days should each show 24 * 0.001 = 0.024 m
    assert daily.sizes["time"] == 2
    np.testing.assert_allclose(daily.isel(time=0).values, 0.024, rtol=1e-6)
    np.testing.assert_allclose(daily.isel(time=1).values, 0.024, rtol=1e-6)


def test_hourly_to_daily_preserves_units_attr():
    da = _make_hourly("2020-01-01 00:00", hours=49, value_per_hour=0.001)
    da.attrs["units"] = "m"
    daily = hourly_to_daily(da)
    assert daily.attrs["units"] == "m"
```

- [ ] **Step 2: Run test to verify failure**

Run: `pixi run -e dev test -- tests/test_era5_land.py -v`
Expected: FAIL (module/function does not exist).

- [ ] **Step 3: Create `fetch/era5_land.py` with aggregation helpers**

```python
"""Fetch ERA5-Land hourly runoff from Copernicus CDS.

Downloads hourly accumulated runoff variables (ro, sro, ssro) for the
CONUS+contributing-watersheds bbox, then aggregates hourly→daily and
daily→monthly. Both daily and monthly consolidated NetCDFs are written
to the shared datastore.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# CDS area parameter [N, W, S, E], snapped to ERA5-Land 0.1° grid.
# Encompasses CONUS contributing watersheds (Canada/Mexico) + ~10 km buffer.
BBOX_NWSE = [53.0, -125.0, 24.7, -66.0]

VARIABLES = ("ro", "sro", "ssro")


def hourly_to_daily(da: xr.DataArray) -> xr.DataArray:
    """Aggregate ERA5-Land hourly accumulated runoff to daily totals.

    ERA5-Land accumulated fields (ro, sro, ssro) reset at 00 UTC each day
    and represent meters of water equivalent accumulated since 00 UTC.
    Daily total = sum of hourly increments computed via .diff('time'),
    then resampled to daily sums. The diff approach is robust to the
    midnight-reset boundary and to missing hours within a day.

    The first hourly step of each day, after the 00 UTC reset, equals the
    accumulation over the 23->00 hour of the prior day. We discard the
    pre-first-midnight increment (which is meaningless without a prior
    23 UTC value) by requiring the result to come from `.diff` with
    matching valid timestamps.

    Parameters
    ----------
    da : xr.DataArray
        Hourly accumulated runoff with a 'time' dimension. Time stamps
        must be regular hourly.

    Returns
    -------
    xr.DataArray
        Daily-summed runoff. Time coordinate is the date (00:00) of each
        complete day. Original attrs are preserved.
    """
    # Hourly increment: value(t) - value(t-1). Negative jumps occur at the
    # 00 UTC reset; clip them to 0 because the post-midnight value is the
    # accumulation 23→00 of the prior day, which we attribute to the prior
    # day via resampling.
    incr = da.diff("time", label="upper")
    # The 00 UTC sample's diff is from 23->00, an increment that belongs
    # to the prior day. Resampling by day-of-timestamp is right because
    # the timestamp is 00 UTC of the new day; we instead want it credited
    # to the prior day. Shift the time coord back by 1 hour so that the
    # 00 UTC increment lands inside the prior day.
    incr = incr.assign_coords(time=incr.time - pd.Timedelta(hours=1))
    daily = incr.resample(time="1D").sum()
    daily.attrs = dict(da.attrs)
    return daily
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev test -- tests/test_era5_land.py -v`
Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/era5_land.py tests/test_era5_land.py
git commit -m "feat(fetch): add ERA5-Land hourly→daily aggregation"
```

---

## Task 6: ERA5-Land daily→monthly aggregation

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/era5_land.py`
- Modify: `tests/test_era5_land.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_era5_land.py`:

```python
from nhf_spatial_targets.fetch.era5_land import daily_to_monthly


def test_daily_to_monthly_sum():
    times = pd.date_range("2020-01-01", periods=60, freq="1D")
    vals = np.full((60, 1, 1), 0.001)
    da = xr.DataArray(
        vals,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": [40.0], "longitude": [-100.0]},
        name="ro",
        attrs={"units": "m"},
    )
    monthly = daily_to_monthly(da)
    # January (31 days) and February 2020 (29 days, leap year)
    assert monthly.sizes["time"] == 2
    np.testing.assert_allclose(monthly.isel(time=0).values, 0.031, rtol=1e-6)
    np.testing.assert_allclose(monthly.isel(time=1).values, 0.029, rtol=1e-6)
    assert monthly.attrs["units"] == "m"
```

- [ ] **Step 2: Run test**

Run: `pixi run -e dev test -- tests/test_era5_land.py::test_daily_to_monthly_sum -v`
Expected: FAIL (function not defined).

- [ ] **Step 3: Add `daily_to_monthly` to `fetch/era5_land.py`**

Append:

```python
def daily_to_monthly(da: xr.DataArray) -> xr.DataArray:
    """Sum daily totals to monthly totals.

    Uses month-end frequency ('1ME') so the time coordinate marks the
    last day of each month — consistent with other monthly products in
    this codebase. Original attrs are preserved.
    """
    monthly = da.resample(time="1ME").sum()
    monthly.attrs = dict(da.attrs)
    return monthly
```

- [ ] **Step 4: Run test**

Run: `pixi run -e dev test -- tests/test_era5_land.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/era5_land.py tests/test_era5_land.py
git commit -m "feat(fetch): add ERA5-Land daily→monthly aggregation"
```

---

## Task 7: ERA5-Land CDS download wrapper

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/era5_land.py`
- Modify: `tests/test_era5_land.py`

- [ ] **Step 1: Write failing test using a mock CDS client**

Append to `tests/test_era5_land.py`:

```python
from unittest.mock import MagicMock, patch


def test_download_year_calls_cds_client(tmp_path, monkeypatch):
    from nhf_spatial_targets.fetch.era5_land import download_year_variable

    fake_client = MagicMock()
    fake_client.retrieve = MagicMock()
    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.era5_land._cds_client", lambda: fake_client
    )

    out = tmp_path / "era5_land_ro_2020.nc"
    download_year_variable(year=2020, variable="ro", output_path=out)

    fake_client.retrieve.assert_called_once()
    args, kwargs = fake_client.retrieve.call_args
    assert args[0] == "reanalysis-era5-land"
    request = args[1]
    assert request["variable"] == "runoff"
    assert request["year"] == "2020"
    assert request["area"] == [53.0, -125.0, 24.7, -66.0]
    assert request["format"] == "netcdf"
    assert "month" in request and len(request["month"]) == 12
    assert args[2] == str(out)


def test_download_year_skips_existing(tmp_path, monkeypatch):
    from nhf_spatial_targets.fetch.era5_land import download_year_variable

    fake_client = MagicMock()
    monkeypatch.setattr(
        "nhf_spatial_targets.fetch.era5_land._cds_client", lambda: fake_client
    )
    out = tmp_path / "era5_land_ro_2020.nc"
    out.write_bytes(b"existing")
    download_year_variable(year=2020, variable="ro", output_path=out)
    fake_client.retrieve.assert_not_called()
```

- [ ] **Step 2: Run tests**

Run: `pixi run -e dev test -- tests/test_era5_land.py -k download_year -v`
Expected: FAIL (function not defined).

- [ ] **Step 3: Add download wrapper to `fetch/era5_land.py`**

Append:

```python
# Map short variable name to the CDS request name
_VARIABLE_REQUEST_NAME = {
    "ro": "runoff",
    "sro": "surface_runoff",
    "ssro": "sub_surface_runoff",
}


def _cds_client():
    """Construct a cdsapi.Client. Separated for test injection."""
    import cdsapi
    return cdsapi.Client()


def download_year_variable(
    year: int,
    variable: str,
    output_path: Path,
) -> Path:
    """Download one year of one ERA5-Land variable to ``output_path``.

    Idempotent: if ``output_path`` already exists, returns immediately.
    Submits a single CDS request covering all 12 months × all hours of
    the given year, clipped to ``BBOX_NWSE``.

    Parameters
    ----------
    year : int
    variable : {"ro", "sro", "ssro"}
    output_path : Path
        Target NetCDF file. Parent directory is created if missing.

    Returns
    -------
    Path
        ``output_path`` (for caller convenience).
    """
    if variable not in _VARIABLE_REQUEST_NAME:
        raise ValueError(f"Unknown ERA5-Land variable: {variable!r}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.info("Skipping existing ERA5-Land file: %s", output_path)
        return output_path

    request = {
        "variable": _VARIABLE_REQUEST_NAME[variable],
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": BBOX_NWSE,
        "format": "netcdf",
    }
    client = _cds_client()
    logger.info("Submitting CDS request for %s %d → %s", variable, year, output_path)
    client.retrieve("reanalysis-era5-land", request, str(output_path))
    return output_path
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev test -- tests/test_era5_land.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/era5_land.py tests/test_era5_land.py
git commit -m "feat(fetch): add ERA5-Land CDS download wrapper"
```

---

## Task 8: ERA5-Land orchestrator (`fetch_era5_land`) with consolidation + manifest

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/era5_land.py`
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py` (add ERA5-Land entry to `apply_cf_metadata`)
- Modify: `tests/test_era5_land.py`

- [ ] **Step 1: Inspect `apply_cf_metadata` to learn the registry pattern**

Run: `grep -n "apply_cf_metadata\|SOURCE_METADATA\|watergap22d" src/nhf_spatial_targets/fetch/consolidate.py | head -40`

- [ ] **Step 2: Add ERA5-Land entry to `consolidate.py`**

Add a registry entry in `consolidate.py` for `era5_land` with both `daily` and `monthly` time steps. Pattern (adapt to the dispatch shape used in the file):

```python
"era5_land": {
    "daily": {
        "title": "ERA5-Land daily runoff (CONUS+ buffered)",
        "institution": "ECMWF",
        "source": "reanalysis-era5-land",
        "references": "doi:10.5194/essd-13-4349-2021",
        "Conventions": "CF-1.6",
        "cell_methods": "time: sum",
        "frequency": "day",
    },
    "monthly": {
        "title": "ERA5-Land monthly runoff (CONUS+ buffered)",
        "institution": "ECMWF",
        "source": "reanalysis-era5-land",
        "references": "doi:10.5194/essd-13-4349-2021",
        "Conventions": "CF-1.6",
        "cell_methods": "time: sum",
        "frequency": "month",
    },
},
```

If `apply_cf_metadata` does not currently support per-time-step variants, extend its signature to take a `time_step` argument and look up the matching sub-entry.

- [ ] **Step 3: Write failing integration-style test (no network) for the orchestrator**

Append to `tests/test_era5_land.py`:

```python
def test_consolidate_year_writes_daily_and_updates_monthly(tmp_path, monkeypatch):
    """Given pre-existing hourly NCs, consolidation produces daily and monthly NCs."""
    from nhf_spatial_targets.fetch.era5_land import consolidate_year

    hourly_dir = tmp_path / "hourly"
    daily_dir = tmp_path / "daily"
    monthly_dir = tmp_path / "monthly"
    hourly_dir.mkdir()

    # Build a synthetic hourly NetCDF for each variable, 2020 full year (sparse)
    times = pd.date_range("2020-01-01", "2020-01-03 23:00", freq="1h")
    for var in ("ro", "sro", "ssro"):
        vals = np.tile(
            np.arange(24, dtype=float).reshape(24, 1, 1) * 0.001,
            (len(times) // 24, 1, 1),
        )
        ds = xr.Dataset(
            {var: (("time", "latitude", "longitude"), vals)},
            coords={
                "time": times[: vals.shape[0]],
                "latitude": [40.0],
                "longitude": [-100.0],
            },
        )
        ds[var].attrs["units"] = "m"
        ds.to_netcdf(hourly_dir / f"era5_land_{var}_2020.nc")

    daily_path, monthly_path = consolidate_year(
        year=2020,
        hourly_dir=hourly_dir,
        daily_dir=daily_dir,
        monthly_dir=monthly_dir,
    )

    assert daily_path.exists()
    daily = xr.open_dataset(daily_path)
    try:
        assert set(daily.data_vars) == {"ro", "sro", "ssro"}
        assert daily.sizes["time"] >= 2
    finally:
        daily.close()

    assert monthly_path.exists()
```

- [ ] **Step 4: Run test**

Run: `pixi run -e dev test -- tests/test_era5_land.py::test_consolidate_year_writes_daily_and_updates_monthly -v`
Expected: FAIL.

- [ ] **Step 5: Implement `consolidate_year` and `fetch_era5_land`**

Append to `fetch/era5_land.py`:

```python
import json
import os
import tempfile
from datetime import datetime, timezone

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import parse_period
from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata
from nhf_spatial_targets.workspace import load as _load_project

_SOURCE_KEY = "era5_land"


def _atomic_to_netcdf(ds: xr.Dataset, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        ds.to_netcdf(tmp, format="NETCDF4")
        tmp.rename(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def consolidate_year(
    year: int,
    hourly_dir: Path,
    daily_dir: Path,
    monthly_dir: Path,
) -> tuple[Path, Path]:
    """Build daily and monthly consolidated NCs for one year.

    Reads ``era5_land_{ro,sro,ssro}_{year}.nc`` from ``hourly_dir``,
    aggregates each variable hourly→daily, merges into a single daily
    dataset, applies CF metadata, writes atomically. Then aggregates
    daily→monthly and writes/updates the monthly file.

    Returns (daily_path, monthly_path).
    """
    daily_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir.mkdir(parents=True, exist_ok=True)
    daily_arrays: dict[str, xr.DataArray] = {}
    for var in VARIABLES:
        path = hourly_dir / f"era5_land_{var}_{year}.nc"
        if not path.exists():
            raise FileNotFoundError(f"Missing hourly file: {path}")
        with xr.open_dataset(path) as ds:
            ds.load()
        da = ds[var]
        daily_arrays[var] = hourly_to_daily(da)

    daily_ds = xr.Dataset(daily_arrays)
    daily_ds = apply_cf_metadata(daily_ds, "era5_land", "daily")
    daily_path = daily_dir / f"era5_land_daily_{year}.nc"
    _atomic_to_netcdf(daily_ds, daily_path)
    logger.info("Wrote %s", daily_path)

    # Monthly: rebuild from the (possibly multi-year) collection of daily files
    daily_files = sorted(daily_dir.glob("era5_land_daily_*.nc"))
    with xr.open_mfdataset(daily_files, combine="by_coords") as ds_all:
        ds_all.load()
    monthly_ds = xr.Dataset(
        {v: daily_to_monthly(ds_all[v]) for v in VARIABLES}
    )
    monthly_ds = apply_cf_metadata(monthly_ds, "era5_land", "monthly")
    start_year = pd.Timestamp(monthly_ds.time.min().values).year
    end_year = pd.Timestamp(monthly_ds.time.max().values).year
    monthly_path = monthly_dir / f"era5_land_monthly_{start_year}_{end_year}.nc"
    # Remove any stale monthly file with a different year range
    for stale in monthly_dir.glob("era5_land_monthly_*.nc"):
        if stale != monthly_path:
            stale.unlink()
    _atomic_to_netcdf(monthly_ds, monthly_path)
    logger.info("Wrote %s", monthly_path)

    return daily_path, monthly_path


def fetch_era5_land(workdir: Path, period: str) -> dict:
    """Download ERA5-Land hourly runoff and produce daily/monthly NCs.

    Loops over years in ``period``, downloading per-year per-variable
    hourly NCs from CDS into the project's datastore, then consolidating
    each year into the daily file and rebuilding the rolling monthly
    file. Idempotent on already-downloaded years.
    """
    ws = _load_project(workdir)
    meta = _catalog.source(_SOURCE_KEY)
    start, end = parse_period(period)

    raw_root = ws.raw_dir(_SOURCE_KEY)
    hourly_dir = raw_root / "hourly"
    daily_dir = raw_root / "daily"
    monthly_dir = raw_root / "monthly"

    now_utc = datetime.now(timezone.utc).isoformat()
    files: list[dict] = []

    for year in range(start.year, end.year + 1):
        for var in VARIABLES:
            out = hourly_dir / f"era5_land_{var}_{year}.nc"
            download_year_variable(year, var, out)
        daily_path, monthly_path = consolidate_year(
            year, hourly_dir, daily_dir, monthly_dir
        )
        files.append(
            {
                "year": year,
                "daily_path": str(daily_path),
                "monthly_path": str(monthly_path),
                "consolidated_utc": now_utc,
            }
        )

    bbox = ws.fabric["bbox_buffered"]
    license_str = meta.get("license", "Copernicus license")
    _update_manifest(workdir, period, bbox, meta, license_str, files)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "license": license_str,
        "variables": [v["name"] for v in meta["variables"]],
        "period": period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "files": files,
    }


def _update_manifest(workdir, period, bbox, meta, license_str, files):
    """Merge ERA5-Land provenance into manifest.json (atomic write)."""
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"sources": {}, "steps": []}
    manifest.setdefault("sources", {})[_SOURCE_KEY] = {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "license": license_str,
        "period": period,
        "bbox": bbox,
        "variables": [v["name"] for v in meta["variables"]],
        "files": files,
    }
    fd, tmp = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp).replace(manifest_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
```

- [ ] **Step 6: Run all ERA5-Land tests**

Run: `pixi run -e dev test -- tests/test_era5_land.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/nhf_spatial_targets/fetch/era5_land.py src/nhf_spatial_targets/fetch/consolidate.py tests/test_era5_land.py
git commit -m "feat(fetch): ERA5-Land orchestrator with daily/monthly consolidation"
```

---

## Task 9: GLDAS fetch module

**Files:**
- Create: `src/nhf_spatial_targets/fetch/gldas.py`
- Create: `tests/test_gldas.py`
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py` (add `gldas_noah_v21_monthly` registry entry)

- [ ] **Step 1: Read MERRA-2 fetch as the closest pattern**

Run: `cat src/nhf_spatial_targets/fetch/merra2.py | head -120`

- [ ] **Step 2: Write failing tests**

Create `tests/test_gldas.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import pytest

from nhf_spatial_targets.fetch.gldas import (
    derive_runoff_total,
    clip_to_bbox,
    BBOX_NWSE,
)


def _global_grid(value_qs=2.0, value_qsb=3.0):
    lat = np.arange(-89.875, 90.0, 0.25)
    lon = np.arange(-179.875, 180.0, 0.25)
    times = pd.date_range("2020-01-01", periods=2, freq="1MS")
    shape = (len(times), len(lat), len(lon))
    return xr.Dataset(
        {
            "Qs_acc": (("time", "lat", "lon"), np.full(shape, value_qs)),
            "Qsb_acc": (("time", "lat", "lon"), np.full(shape, value_qsb)),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )


def test_derive_runoff_total_sums_qs_and_qsb():
    ds = _global_grid()
    out = derive_runoff_total(ds)
    assert "runoff_total" in out.data_vars
    np.testing.assert_allclose(out.runoff_total.values, 5.0)
    assert out.runoff_total.attrs["long_name"] == "total runoff (Qs_acc + Qsb_acc, derived)"
    assert out.runoff_total.attrs["units"] == "kg m-2"


def test_clip_to_bbox_reduces_extent():
    ds = _global_grid()
    clipped = clip_to_bbox(ds, BBOX_NWSE)
    # bbox is N=53.0, W=-125.0, S=24.7, E=-66.0
    assert clipped.lat.min() >= 24.7 - 0.25
    assert clipped.lat.max() <= 53.0 + 0.25
    assert clipped.lon.min() >= -125.0 - 0.25
    assert clipped.lon.max() <= -66.0 + 0.25
    assert clipped.lat.size < ds.lat.size
    assert clipped.lon.size < ds.lon.size
```

- [ ] **Step 3: Run tests**

Run: `pixi run -e dev test -- tests/test_gldas.py -v`
Expected: FAIL (module/functions don't exist).

- [ ] **Step 4: Create `fetch/gldas.py`**

```python
"""Fetch GLDAS-2.1 NOAH monthly runoff from NASA GES DISC."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import parse_period
from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata
from nhf_spatial_targets.workspace import load as _load_project

logger = logging.getLogger(__name__)

_SOURCE_KEY = "gldas_noah_v21_monthly"

# Lat/lon bbox matches the ERA5-Land bbox: N=53.0, W=-125.0, S=24.7, E=-66.0
BBOX_NWSE = [53.0, -125.0, 24.7, -66.0]


def derive_runoff_total(ds: xr.Dataset) -> xr.Dataset:
    """Add ``runoff_total = Qs_acc + Qsb_acc`` to the dataset.

    Both inputs are kg m-2 (mm equivalent); their sum has the same units.
    """
    total = ds["Qs_acc"] + ds["Qsb_acc"]
    total.attrs = {
        "long_name": "total runoff (Qs_acc + Qsb_acc, derived)",
        "units": "kg m-2",
        "cell_methods": "time: sum",
    }
    return ds.assign(runoff_total=total)


def clip_to_bbox(ds: xr.Dataset, bbox_nwse: list[float]) -> xr.Dataset:
    """Clip a global dataset (lat/lon coords) to a [N, W, S, E] bbox."""
    n, w, s, e = bbox_nwse
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat = ds[lat_name]
    if float(lat[0]) < float(lat[-1]):
        lat_slice = slice(s, n)
    else:
        lat_slice = slice(n, s)
    return ds.sel({lat_name: lat_slice, lon_name: slice(w, e)})


def fetch_gldas(workdir: Path, period: str) -> dict:
    """Download GLDAS-2.1 NOAH monthly granules and consolidate.

    Uses earthaccess (NASA EDL) to download monthly granules covering
    ``period``, then concatenates, derives ``runoff_total``, clips to
    the project bbox, applies CF metadata, and writes a single
    consolidated NC to the datastore.
    """
    import earthaccess

    ws = _load_project(workdir)
    meta = _catalog.source(_SOURCE_KEY)
    start, end = parse_period(period)

    raw_dir = ws.raw_dir(_SOURCE_KEY) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cf_path = ws.raw_dir(_SOURCE_KEY) / "gldas_noah_v21_monthly.nc"
    now_utc = datetime.now(timezone.utc).isoformat()

    earthaccess.login(strategy="netrc")
    results = earthaccess.search_data(
        short_name=meta["access"]["short_name"],
        version=meta["access"]["version"],
        temporal=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
    )
    files = earthaccess.download(results, str(raw_dir))
    logger.info("Downloaded %d GLDAS granules to %s", len(files), raw_dir)

    with xr.open_mfdataset(sorted(files), combine="by_coords") as ds:
        ds = ds[["Qs_acc", "Qsb_acc"]].load()
    ds = derive_runoff_total(ds)
    ds = clip_to_bbox(ds, BBOX_NWSE)
    ds = apply_cf_metadata(ds, _SOURCE_KEY, "monthly")

    tmp = cf_path.with_suffix(".nc.tmp")
    try:
        ds.to_netcdf(tmp, format="NETCDF4")
        tmp.rename(cf_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    bbox = ws.fabric["bbox_buffered"]
    file_info = {
        "path": str(cf_path),
        "size_bytes": cf_path.stat().st_size,
        "downloaded_utc": now_utc,
        "n_granules": len(files),
    }
    _update_manifest(workdir, period, bbox, meta, file_info)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "license": meta.get("license", "public domain (NASA)"),
        "variables": [v["name"] for v in meta["variables"]],
        "period": period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "file": file_info,
    }


def _update_manifest(workdir, period, bbox, meta, file_info):
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"sources": {}, "steps": []}
    )
    manifest.setdefault("sources", {})[_SOURCE_KEY] = {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "license": meta.get("license", "public domain (NASA)"),
        "period": period,
        "bbox": bbox,
        "variables": [v["name"] for v in meta["variables"]],
        "file": file_info,
    }
    fd, tmp = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp).replace(manifest_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
```

- [ ] **Step 5: Add `gldas_noah_v21_monthly` to `consolidate.py` registry**

Add a registry entry analogous to other monthly sources:

```python
"gldas_noah_v21_monthly": {
    "monthly": {
        "title": "GLDAS-2.1 NOAH monthly runoff (CONUS+ buffered)",
        "institution": "NASA GES DISC",
        "source": "GLDAS_NOAH025_M v2.1",
        "references": "doi:10.1175/BAMS-85-3-381",
        "Conventions": "CF-1.6",
        "cell_methods": "time: sum",
        "frequency": "month",
    },
},
```

- [ ] **Step 6: Run tests**

Run: `pixi run -e dev test -- tests/test_gldas.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/nhf_spatial_targets/fetch/gldas.py src/nhf_spatial_targets/fetch/consolidate.py tests/test_gldas.py
git commit -m "feat(fetch): add GLDAS-2.1 NOAH monthly fetch module"
```

---

## Task 10: Wire ERA5-Land and GLDAS into the CLI

**Files:**
- Modify: `src/nhf_spatial_targets/cli.py`

- [ ] **Step 1: Read existing CLI command pattern**

Run: `sed -n '300,360p' src/nhf_spatial_targets/cli.py`

- [ ] **Step 2: Add per-source CLI commands**

Following the pattern of `fetch_merra2_cmd`, add two commands:

```python
@fetch_app.command(name="era5-land")
def fetch_era5_land_cmd(
    project_dir: Path,
    period: str = "1979/2024",
):
    """Download ERA5-Land hourly runoff (CDS) and consolidate to daily/monthly."""
    from nhf_spatial_targets.fetch.era5_land import fetch_era5_land
    workdir = Path(project_dir).resolve()
    result = fetch_era5_land(workdir=workdir, period=period)
    # use the same logging/printing pattern as fetch_merra2_cmd


@fetch_app.command(name="gldas")
def fetch_gldas_cmd(
    project_dir: Path,
    period: str = "2000/2023",
):
    """Download GLDAS-2.1 NOAH monthly runoff (NASA GES DISC)."""
    from nhf_spatial_targets.fetch.gldas import fetch_gldas
    workdir = Path(project_dir).resolve()
    result = fetch_gldas(workdir=workdir, period=period)
```

Look at `fetch_merra2_cmd` for the surrounding boilerplate (logging, error handling, return) and replicate it exactly.

- [ ] **Step 3: Add both sources to the `fetch all` registry**

In `fetch_all_cmd`, add to the `sources` list (around line 246-253):

```python
("era5-land", "era5_land", fetch_era5_land),
("gldas", "gldas_noah_v21_monthly", fetch_gldas),
```

And add the matching imports near the existing fetch imports.

- [ ] **Step 4: Add a `pixi run` task entry**

In `pixi.toml`, under `[tasks]` (alongside other `fetch-*` tasks if present), add:

```toml
fetch-era5-land = "nhf-targets fetch era5-land"
fetch-gldas = "nhf-targets fetch gldas"
```

- [ ] **Step 5: Smoke check the CLI registers**

Run: `pixi run -- nhf-targets fetch --help`
Expected: output lists `era5-land` and `gldas` subcommands.

- [ ] **Step 6: Commit**

```bash
git add src/nhf_spatial_targets/cli.py pixi.toml
git commit -m "feat(cli): register era5-land and gldas fetch commands"
```

---

## Task 11: Runoff target builder

**Files:**
- Modify: `src/nhf_spatial_targets/targets/run.py`
- Create: `tests/test_run_target.py`

- [ ] **Step 1: Inspect the AET builder for the multi-source minmax pattern**

Run: `cat src/nhf_spatial_targets/targets/aet.py`

- [ ] **Step 2: Write failing tests for unit harmonization and minmax**

Create `tests/test_run_target.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.targets.run import (
    era5_to_mm_per_month,
    gldas_to_mm_per_month,
    mm_per_month_to_cfs,
    multi_source_runoff_bounds,
)

HRU_AREA_M2 = 1.0e8  # 100 km²


def _series(values, units):
    times = pd.date_range("2020-01-01", periods=len(values), freq="1MS")
    da = xr.DataArray(values, dims=("time",), coords={"time": times})
    da.attrs["units"] = units
    return da


def test_era5_meters_to_mm():
    da = _series([0.05, 0.10], "m")
    out = era5_to_mm_per_month(da)
    np.testing.assert_allclose(out.values, [50.0, 100.0])
    assert out.attrs["units"] == "mm"


def test_gldas_kgm2_passthrough_to_mm():
    da = _series([20.0, 40.0], "kg m-2")
    out = gldas_to_mm_per_month(da)
    np.testing.assert_allclose(out.values, [20.0, 40.0])
    assert out.attrs["units"] == "mm"


def test_mm_to_cfs_uses_days_in_month():
    # 31 mm in January = 0.001 m/day over 100 km² = 100 m³/day = ~0.0409 cfs
    da = _series([31.0], "mm")
    out = mm_per_month_to_cfs(da, hru_area_m2=HRU_AREA_M2)
    expected_m3_per_day = 0.001 * HRU_AREA_M2  # 100 000 m³/day
    expected_cfs = expected_m3_per_day / 86400.0 * 35.3147
    np.testing.assert_allclose(out.values, [expected_cfs], rtol=1e-4)
    assert out.attrs["units"] == "cfs"


def test_multi_source_minmax():
    a = _series([10.0, 20.0, 30.0], "cfs")
    b = _series([15.0, 18.0, 32.0], "cfs")
    lower, upper = multi_source_runoff_bounds([a, b])
    np.testing.assert_allclose(lower.values, [10.0, 18.0, 30.0])
    np.testing.assert_allclose(upper.values, [15.0, 20.0, 32.0])
```

- [ ] **Step 3: Run tests**

Run: `pixi run -e dev test -- tests/test_run_target.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement `targets/run.py`**

Replace the current stub with:

```python
"""Build runoff calibration targets from ERA5-Land + GLDAS-2.1 NOAH."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

_M3_PER_DAY_TO_CFS = 35.3146667 / 86400.0  # cubic feet per second per m³/day


def era5_to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """ERA5-Land runoff (m water-eq / month) → mm/month."""
    out = da * 1000.0
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


def gldas_to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """GLDAS Qs_acc + Qsb_acc (kg m-2) ≡ mm/month directly."""
    out = da.copy()
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


def mm_per_month_to_cfs(da: xr.DataArray, hru_area_m2: float | xr.DataArray) -> xr.DataArray:
    """Convert mm/month → cfs given HRU area and the month length.

    mm/month × 1e-3 m/mm × area_m2 / days_in_month → m³/day → cfs.
    Days-in-month is read from ``da.time.dt.days_in_month``.
    """
    days = da["time"].dt.days_in_month
    m_per_day = (da * 1e-3) / days
    m3_per_day = m_per_day * hru_area_m2
    cfs = m3_per_day * _M3_PER_DAY_TO_CFS
    cfs.attrs = dict(da.attrs)
    cfs.attrs["units"] = "cfs"
    return cfs


def multi_source_runoff_bounds(
    sources: list[xr.DataArray],
) -> tuple[xr.DataArray, xr.DataArray]:
    """Per-coord minimum and maximum across input sources.

    All inputs must share dimensions and coords. Returns (lower, upper).
    """
    stacked = xr.concat(sources, dim="source")
    return stacked.min("source"), stacked.max("source")


def build(config: dict, fabric_path: str, output_path: str) -> None:
    """Build runoff target dataset.

    Reads HRU-aggregated monthly runoff for ERA5-Land (`ro`) and GLDAS
    (`runoff_total`), harmonizes units to cfs, computes per-HRU
    per-month min/max bounds, writes a CF-compliant NetCDF with
    `lower_bound` and `upper_bound` variables (dims: hru, time).
    """
    raise NotImplementedError(
        "Builder wiring depends on aggregate/ output paths; "
        "implement once aggregate output schema is finalized."
    )
```

The pure functions are exercised by tests; `build()` remains a stub until the aggregation output schema is concretized — see Task 12.

- [ ] **Step 5: Run tests**

Run: `pixi run -e dev test -- tests/test_run_target.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/nhf_spatial_targets/targets/run.py tests/test_run_target.py
git commit -m "feat(targets): runoff unit harmonization + multi-source minmax"
```

---

## Task 12: Implement runoff `build()` end-to-end

**Files:**
- Modify: `src/nhf_spatial_targets/targets/run.py`
- Modify: `tests/test_run_target.py`

- [ ] **Step 1: Read aggregate output conventions**

Run: `cat src/nhf_spatial_targets/aggregate/gdptools_agg.py | head -80`
Run: `grep -n "agg" src/nhf_spatial_targets/cli.py | head -20`

Identify how aggregated outputs are named/located for each source. The expected pattern (per CLAUDE.md) is `<project>/data/aggregated/<source_key>/<var>.nc`.

- [ ] **Step 2: Write failing test for build() with synthetic aggregated inputs**

Append to `tests/test_run_target.py`:

```python
def test_build_writes_lower_upper_bounds(tmp_path):
    from nhf_spatial_targets.targets.run import build

    # Synthetic per-HRU per-month aggregated inputs in mm/month equivalents
    times = pd.date_range("2020-01-01", periods=3, freq="1MS")
    hrus = np.arange(3)

    def _save(path: Path, var: str, vals_m_or_kg, units: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        da = xr.DataArray(
            vals_m_or_kg,
            dims=("time", "hru"),
            coords={"time": times, "hru": hrus},
            name=var,
        )
        da.attrs["units"] = units
        ds = da.to_dataset()
        ds.to_netcdf(path)

    agg = tmp_path / "data" / "aggregated"
    _save(agg / "era5_land" / "ro.nc", "ro",
          np.full((3, 3), 0.05), "m")  # 0.05 m = 50 mm
    _save(agg / "gldas_noah_v21_monthly" / "runoff_total.nc", "runoff_total",
          np.full((3, 3), 30.0), "kg m-2")  # 30 mm

    out = tmp_path / "targets" / "runoff_target.nc"
    hru_area = xr.DataArray(np.full(3, 1.0e8), dims=("hru",), coords={"hru": hrus})

    build(
        config={"aggregated_dir": str(agg), "hru_area_m2": hru_area},
        fabric_path="unused",
        output_path=str(out),
    )

    assert out.exists()
    ds = xr.open_dataset(out)
    try:
        assert "lower_bound" in ds.data_vars
        assert "upper_bound" in ds.data_vars
        assert ds["lower_bound"].dims == ("time", "hru")
        # GLDAS (30 mm) should be the lower bound; ERA5 (50 mm) the upper.
        # Both convert via mm_per_month_to_cfs with the same area/days, so
        # ordering is preserved in cfs.
        assert (ds["lower_bound"].values <= ds["upper_bound"].values).all()
    finally:
        ds.close()
```

- [ ] **Step 3: Run test**

Run: `pixi run -e dev test -- tests/test_run_target.py::test_build_writes_lower_upper_bounds -v`
Expected: FAIL (NotImplementedError).

- [ ] **Step 4: Implement `build()`**

Replace the `build` function body with:

```python
def build(config: dict, fabric_path: str, output_path: str) -> None:
    """Build runoff target dataset.

    config keys:
      aggregated_dir : str | Path
          Directory containing per-source per-variable aggregated NCs at
          ``<aggregated_dir>/<source_key>/<var>.nc``.
      hru_area_m2 : xr.DataArray
          Per-HRU area in m², dims=('hru',), coord aligned with the
          aggregated outputs.
    """
    agg_dir = Path(config["aggregated_dir"])
    hru_area = config["hru_area_m2"]

    with xr.open_dataset(agg_dir / "era5_land" / "ro.nc") as ds:
        era5 = ds["ro"].load()
    with xr.open_dataset(
        agg_dir / "gldas_noah_v21_monthly" / "runoff_total.nc"
    ) as ds:
        gldas = ds["runoff_total"].load()

    era5_cfs = mm_per_month_to_cfs(era5_to_mm_per_month(era5), hru_area)
    gldas_cfs = mm_per_month_to_cfs(gldas_to_mm_per_month(gldas), hru_area)

    lower, upper = multi_source_runoff_bounds([era5_cfs, gldas_cfs])
    lower.name = "lower_bound"
    upper.name = "upper_bound"
    out_ds = xr.Dataset(
        {"lower_bound": lower, "upper_bound": upper},
        attrs={
            "title": "NHM runoff calibration target (ERA5-Land + GLDAS-2.1)",
            "Conventions": "CF-1.6",
            "cell_methods": "time: sum",
            "units": "cfs",
        },
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".nc.tmp")
    try:
        out_ds.to_netcdf(tmp, format="NETCDF4")
        tmp.rename(output_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    logger.info("Wrote runoff target: %s", output_path)
```

- [ ] **Step 5: Run tests**

Run: `pixi run -e dev test -- tests/test_run_target.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/nhf_spatial_targets/targets/run.py tests/test_run_target.py
git commit -m "feat(targets): runoff target builder writes lower/upper bounds"
```

---

## Task 13: Drop MWBM from AET target builder

**Files:**
- Modify: `src/nhf_spatial_targets/targets/aet.py`

- [ ] **Step 1: Inspect current aet.py**

Run: `cat src/nhf_spatial_targets/targets/aet.py`

- [ ] **Step 2: Remove any code path referencing `nhm_mwbm`**

Edit `targets/aet.py` to drop:
- Imports related to MWBM (none expected, but check)
- Any source-key reference to `"nhm_mwbm"`
- Any list element naming MWBM in the sources iteration

If the file is currently a stub (raise NotImplementedError) with no MWBM reference, this task is a verification-only step — note that fact in the commit message and skip the edit.

- [ ] **Step 3: Run all target tests**

Run: `pixi run -e dev test -- tests/ -v`
Expected: PASS (no regressions).

- [ ] **Step 4: Commit (only if changes made)**

```bash
git add src/nhf_spatial_targets/targets/aet.py
git commit -m "refactor(targets/aet): drop nhm_mwbm source"
```

---

## Task 14: Add ERA5-Land `ssro` to recharge target

**Files:**
- Modify: `src/nhf_spatial_targets/targets/rch.py`
- Modify (or create) corresponding test file

- [ ] **Step 1: Inspect current rch.py and any normalization helper**

Run: `cat src/nhf_spatial_targets/targets/rch.py`
Run: `grep -rn "normalize" src/nhf_spatial_targets/normalize/ src/nhf_spatial_targets/targets/`

- [ ] **Step 2: If `rch.py` is a stub, add a minimum scaffold for the third source**

If the file is `raise NotImplementedError`, leave it as a stub but document the new third source in a module docstring so the upcoming implementation includes it:

```python
"""Build recharge calibration targets.

Sources (per catalog/variables.yml):
  - reitz2017: total_recharge (in/yr)
  - watergap22d: groundwater_recharge (kg m-2 s-1, → mm/yr)
  - era5_land: ssro summed monthly→annual (m water-eq, → mm/yr)

Method: each source aggregated to HRU, normalized 0-1 per HRU over
2000-2009, then per-HRU per-year lower/upper = min/max across the three
normalized sources.
"""
```

If the file already has logic, add an `ssro` branch that:
- Loads aggregated `<aggregated_dir>/era5_land/ssro.nc` (monthly)
- Sums to annual
- Converts m → mm (×1000)
- Normalizes 0-1 over 2000-2009
- Joins the existing min/max stack

- [ ] **Step 3: Add a test that asserts the recharge sources include era5_land**

Test file `tests/test_rch_target.py` (create if missing):

```python
from nhf_spatial_targets import catalog


def test_recharge_target_lists_three_sources():
    v = catalog.variable("recharge")
    assert set(v["sources"]) == {"reitz2017", "watergap22d", "era5_land"}
```

(Catalog assertion; behavioral test follows when `rch.py` is implemented.)

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev test -- tests/test_rch_target.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/targets/rch.py tests/test_rch_target.py
git commit -m "feat(targets/rch): document era5_land ssro as third recharge source"
```

---

## Task 15: Update CLAUDE.md and README

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md "Known Gaps" section**

In `CLAUDE.md`, under "Known Gaps":
- Remove the "MWBM ScienceBase item ID — confirmed: ..." line from "Resolved"
- Add to "Resolved":
  - `Runoff source replacement — NHM-MWBM removed; replaced by ERA5-Land (CDS) + GLDAS-2.1 NOAH monthly. ERA5-Land ssro also added as third recharge source. See PR #<N>.`

- [ ] **Step 2: Update README runoff description**

In `README.md`, find the runoff target description (likely a table or paragraph naming MWBM) and replace with:

> Runoff (`basin_cfs`, monthly): multi-source min/max across ERA5-Land
> total runoff (`ro`) and GLDAS-2.1 NOAH monthly runoff
> (`Qs_acc + Qsb_acc`). Both aggregated to HRUs via gdptools, harmonized
> to mm/month, then converted to cfs.

- [ ] **Step 3: Run lint/format/tests once more**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update CLAUDE.md and README for ERA5-Land+GLDAS runoff"
```

---

## Task 16: Final verification

- [ ] **Step 1: Confirm catalog parses cleanly**

Run: `pixi run catalog-sources`
Expected: lists `era5_land` and `gldas_noah_v21_monthly`; no `nhm_mwbm`.

Run: `pixi run catalog-variables`
Expected: runoff sources are `[era5_land, gldas_noah_v21_monthly]`; recharge sources include `era5_land`.

- [ ] **Step 2: CLI smoke test**

Run: `pixi run -- nhf-targets fetch --help`
Expected: `era5-land` and `gldas` subcommands listed.

- [ ] **Step 3: Full test suite**

Run: `pixi run -e dev test`
Expected: all PASS.

- [ ] **Step 4: Confirm git log shows the expected commit sequence**

Run: `git log --oneline main..HEAD`
Expected: ~14 commits, one per implementation task.

---

## Notes for the executor

- **CDS account.** The user is provisioning a Copernicus CDS account in parallel. Network-dependent integration tests against the live CDS API are out of scope for this plan; unit tests use mocks (Tasks 5-8).
- **GLDAS Earthdata.** Earthdata credentials are already configured in this project. `fetch_gldas` uses `earthaccess.login(strategy="netrc")` — make sure `~/.netrc` is set (it is, per existing modules).
- **Atomic writes.** Every NetCDF write goes through a `.tmp` rename. Don't shortcut this.
- **No backwards-compat shims.** `nhm_mwbm` is removed cleanly; do not add a deprecation alias.
