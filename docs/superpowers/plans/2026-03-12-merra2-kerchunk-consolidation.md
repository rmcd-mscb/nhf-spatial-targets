# MERRA-2 Kerchunk Consolidation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After fetching MERRA-2 monthly files, build a Kerchunk virtual Zarr store containing only the three soil wetness variables (GWETTOP, GWETROOT, GWETPROF) with CF-compliant mid-month timestamps, and support incremental fetch/rebuild when the period changes.

**Architecture:** `fetch_merra2()` reads `manifest.json` to determine which months are already downloaded, fetches only missing months, then calls `consolidate_merra2()` to build/rebuild a Kerchunk JSON reference store. The reference store uses relative paths so the entire `data/raw/merra2/` directory is portable. Time coordinates are shifted to mid-month (15th) with CF `time_bnds`. The Kerchunk store is always fully rebuilt from all files on disk (not incrementally appended) since `MultiZarrToZarr` does not support partial append.

**Tech Stack:** kerchunk (>=0.2), ujson, fsspec, xarray, h5py (transitive via kerchunk)

**Spec:** `docs/superpowers/specs/2026-03-12-merra2-kerchunk-consolidation-design.md`

**Catalog update already done:** `catalog/sources.yml` has been updated to replace SFMC with GWETTOP/GWETROOT/GWETPROF (commit f8629a7).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `pixi.toml` | Add `kerchunk` and `ujson` dependencies |
| `src/nhf_spatial_targets/fetch/consolidate.py` (create) | Kerchunk reference building: scan NetCDFs, filter variables, fix time, write JSON |
| `src/nhf_spatial_targets/fetch/merra2.py` (modify) | Incremental fetch: read manifest, compute delta, download missing, call consolidate, update manifest |
| `tests/test_consolidate.py` (create) | Tests for variable filtering, time fixup, relative paths, incremental rebuild |
| `tests/test_merra2.py` (modify) | Update tests for incremental fetch and manifest interaction |

---

## Chunk 1: Dependencies and Consolidation Module

### Task 1: Add kerchunk and ujson dependencies

**Files:**
- Modify: `pixi.toml`

- [ ] **Step 1: Add kerchunk and ujson to pixi.toml**

In `pixi.toml`, add `kerchunk` and `ujson` to `[pypi-dependencies]`:

```toml
[pypi-dependencies]
nhf-spatial-targets = { path = ".", editable = true }
gdptools = "*"
sciencebasepy = "*"
earthaccess = "*"
kerchunk = ">=0.2"
ujson = "*"
```

- [ ] **Step 2: Install and verify**

Run: `pixi install && pixi run python -c "import kerchunk; import ujson; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add pixi.toml pixi.lock
git commit -m "feat: add kerchunk and ujson dependencies"
```

---

### Task 2: Create consolidation module — variable filtering

**Files:**
- Create: `src/nhf_spatial_targets/fetch/consolidate.py`
- Create: `tests/test_consolidate.py`

The consolidation module builds a Kerchunk JSON reference store from MERRA-2 NetCDF files. This task implements the core variable filtering logic.

**IMPORTANT API note:** `kerchunk.hdf.SingleHdf5ToZarr` takes an **open file object** as its first argument, not a string path. The second argument is the URL/path string used for reference paths.

- [ ] **Step 1: Write the failing test for variable filtering**

Create `tests/test_consolidate.py`:

```python
"""Tests for MERRA-2 Kerchunk consolidation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


@pytest.fixture()
def merra2_dir(tmp_path: Path) -> Path:
    """Create a directory with small synthetic MERRA-2 NetCDF files."""
    out = tmp_path / "data" / "raw" / "merra2"
    out.mkdir(parents=True)

    lat = np.arange(-90, 91, 45.0)
    lon = np.arange(-180, 180, 60.0)

    for month in range(1, 4):  # 3 months
        time = np.array(
            [f"2010-{month:02d}-01T00:30:00"],
            dtype="datetime64[ns]",
        )
        ds = xr.Dataset(
            {
                "GWETTOP": (["time", "lat", "lon"], np.random.rand(1, len(lat), len(lon)).astype(np.float32)),
                "GWETROOT": (["time", "lat", "lon"], np.random.rand(1, len(lat), len(lon)).astype(np.float32)),
                "GWETPROF": (["time", "lat", "lon"], np.random.rand(1, len(lat), len(lon)).astype(np.float32)),
                "SFMC": (["time", "lat", "lon"], np.random.rand(1, len(lat), len(lon)).astype(np.float32)),
                "BASEFLOW": (["time", "lat", "lon"], np.random.rand(1, len(lat), len(lon)).astype(np.float32)),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )
        fname = f"MERRA2_300.tavgM_2d_lnd_Nx.2010{month:02d}.nc4"
        ds.to_netcdf(out / fname)

    return out


def test_filter_variables(merra2_dir):
    """Reference store contains only requested variables plus coordinates."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    run_dir = merra2_dir.parent.parent.parent
    result = consolidate_merra2(
        run_dir=run_dir,
        variables=["GWETTOP", "GWETROOT", "GWETPROF"],
    )

    ref_path = merra2_dir / "merra2_refs.json"
    assert ref_path.exists()

    import fsspec
    fs = fsspec.filesystem(
        "reference",
        fo=str(ref_path),
        target_protocol="file",
    )
    ds = xr.open_zarr(fs.get_mapper(""), consolidated=False)

    assert "GWETTOP" in ds.data_vars
    assert "GWETROOT" in ds.data_vars
    assert "GWETPROF" in ds.data_vars
    assert "SFMC" not in ds.data_vars
    assert "BASEFLOW" not in ds.data_vars
    assert len(ds.time) == 3
    ds.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_consolidate.py::test_filter_variables -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nhf_spatial_targets.fetch.consolidate'`

- [ ] **Step 3: Write minimal consolidation implementation**

Create `src/nhf_spatial_targets/fetch/consolidate.py`:

```python
"""Build Kerchunk virtual Zarr reference store from MERRA-2 NetCDF files."""

from __future__ import annotations

import logging
from pathlib import Path

import ujson

from nhf_spatial_targets import __version__

logger = logging.getLogger(__name__)

# Variables to keep in addition to user-requested ones.
# Coordinate variables and their bounds are always retained.
_COORD_VARS = {"time", "lat", "lon", "time_bnds", "lat_bnds", "lon_bnds"}


def _filter_refs(refs: dict, keep_vars: set[str]) -> dict:
    """Remove variable keys from a kerchunk reference dict.

    Keeps:
    - Top-level metadata keys (no "/" in key): .zattrs, .zgroup
    - Coordinate variables: time, lat, lon (and any *_bnds)
    - Only the data variables listed in keep_vars

    Drops all other variable keys (e.g., "SFMC/.zarray", "SFMC/0.0.0").
    """
    all_keep = keep_vars | _COORD_VARS
    filtered = {}
    for key in refs:
        # Global metadata keys (no "/") are always kept
        if "/" not in key:
            filtered[key] = refs[key]
            continue
        # Variable-level keys: "VARNAME/.zarray", "VARNAME/0.0.0", etc.
        var_name = key.split("/")[0]
        if var_name in all_keep:
            filtered[key] = refs[key]
    return filtered


def consolidate_merra2(
    run_dir: Path,
    variables: list[str],
) -> dict:
    """Build a Kerchunk JSON reference store for MERRA-2 files.

    Always performs a full rebuild from all .nc4 files on disk.
    MultiZarrToZarr does not support incremental append, so every
    call re-scans all files. For ~500 monthly files this takes
    seconds, not minutes.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory containing ``data/raw/merra2/*.nc4``.
    variables : list[str]
        Variable names to include (e.g. ["GWETTOP", "GWETROOT", "GWETPROF"]).

    Returns
    -------
    dict
        Provenance record with reference file path and timestamp.
    """
    from datetime import datetime, timezone

    import kerchunk.hdf
    from kerchunk.combine import MultiZarrToZarr

    merra2_dir = run_dir / "data" / "raw" / "merra2"
    nc_files = sorted(merra2_dir.glob("*.nc4"))

    if not nc_files:
        raise FileNotFoundError(
            f"No .nc4 files found in {merra2_dir}. "
            f"Run 'nhf-targets fetch merra2' first."
        )

    logger.info("Scanning %d NetCDF files for Kerchunk references", len(nc_files))

    keep_vars = set(variables)
    singles = []
    for nc in nc_files:
        with open(nc, "rb") as f:
            h5chunks = kerchunk.hdf.SingleHdf5ToZarr(f, str(nc))
            refs = h5chunks.translate()
        refs["refs"] = _filter_refs(refs["refs"], keep_vars)
        singles.append(refs)

    mzz = MultiZarrToZarr(
        singles,
        concat_dims=["time"],
        identical_dims=["lat", "lon"],
    )
    combined = mzz.translate()

    # Make paths relative to merra2_dir
    combined["refs"] = _make_relative(combined["refs"], merra2_dir)

    ref_path = merra2_dir / "merra2_refs.json"
    ref_path.write_text(ujson.dumps(combined, indent=2))
    logger.info("Wrote Kerchunk reference store: %s", ref_path)

    return {
        "kerchunk_ref": str(ref_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(nc_files),
        "variables": variables,
    }


def _make_relative(refs: dict, base_dir: Path) -> dict:
    """Convert absolute file paths in kerchunk refs to relative paths."""
    base_str = str(base_dir)
    out = {}
    for key, val in refs.items():
        if isinstance(val, list) and len(val) >= 1 and isinstance(val[0], str):
            path = val[0]
            if path.startswith(base_str):
                rel = "./" + str(Path(path).relative_to(base_dir))
                val = [rel] + val[1:]
        out[key] = val
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_consolidate.py::test_filter_variables -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate.py
git commit -m "feat: add Kerchunk consolidation with variable filtering"
```

---

### Task 3: Add CF-compliant time representation

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py`
- Modify: `tests/test_consolidate.py`

Add mid-month timestamp shifting and `time_bnds` generation. The approach: open the combined refs as an xarray dataset to read original times, compute corrected time + bounds, build a small xarray dataset with the corrected arrays, write it to an in-memory zarr store, then copy the zarr metadata and chunk bytes back into the kerchunk reference dict.

- [ ] **Step 1: Write the failing test for time fixup**

Append to `tests/test_consolidate.py`:

```python
def test_time_midmonth(merra2_dir):
    """Timestamps are shifted to the 15th of each month."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2
    import pandas as pd

    run_dir = merra2_dir.parent.parent.parent
    consolidate_merra2(
        run_dir=run_dir,
        variables=["GWETTOP"],
    )

    ref_path = merra2_dir / "merra2_refs.json"
    import fsspec
    fs = fsspec.filesystem("reference", fo=str(ref_path), target_protocol="file")
    ds = xr.open_zarr(fs.get_mapper(""), consolidated=False)

    for t in pd.DatetimeIndex(ds.time.values):
        assert t.day == 15
        assert t.hour == 0
    ds.close()


def test_time_bounds(merra2_dir):
    """time_bnds spans first-of-month to first-of-next-month."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2
    import pandas as pd

    run_dir = merra2_dir.parent.parent.parent
    consolidate_merra2(
        run_dir=run_dir,
        variables=["GWETTOP"],
    )

    ref_path = merra2_dir / "merra2_refs.json"
    import fsspec
    fs = fsspec.filesystem("reference", fo=str(ref_path), target_protocol="file")
    ds = xr.open_zarr(fs.get_mapper(""), consolidated=False)

    assert "time_bnds" in ds.data_vars
    assert ds.time.attrs.get("bounds") == "time_bnds"
    # First month: Jan 2010
    bnds = pd.DatetimeIndex(ds.time_bnds.values[0])
    assert bnds[0] == pd.Timestamp("2010-01-01")
    assert bnds[1] == pd.Timestamp("2010-02-01")
    ds.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_consolidate.py::test_time_midmonth tests/test_consolidate.py::test_time_bounds -v`
Expected: FAIL — timestamps are still at original `YYYY-01-01T00:30:00`, no `time_bnds`

- [ ] **Step 3: Implement time fixup**

The approach uses xarray + zarr's in-memory store to serialize corrected time arrays, then patches the references. This avoids manual base64 encoding and ensures zarr metadata (dtype, `_ARRAY_DIMENSIONS`) is correct.

Add to `consolidate.py`:

```python
import numpy as np


def _fix_time(combined: dict, merra2_dir: Path) -> dict:
    """Shift timestamps to mid-month and add time_bnds.

    Operates on the combined kerchunk reference dict. Opens the virtual
    dataset to read original times, computes corrected arrays, serializes
    via xarray-to-zarr into an in-memory store, then copies the zarr
    metadata and chunk bytes into the reference dict.
    """
    import fsspec
    import pandas as pd
    import xarray as xr
    import zarr

    # Open the combined refs to get the time coordinate
    fs = fsspec.filesystem(
        "reference",
        fo=combined,
        target_protocol="file",
        remote_protocol="file",
    )
    ds = xr.open_zarr(fs.get_mapper(""), consolidated=False)
    original_times = pd.DatetimeIndex(ds.time.values)
    ds.close()

    # Build mid-month timestamps
    mid_month = np.array(
        [t.replace(day=15, hour=0, minute=0, second=0, microsecond=0)
         for t in original_times],
        dtype="datetime64[ns]",
    )

    # Build time bounds: [first-of-month, first-of-next-month]
    bounds = []
    for t in original_times:
        month_start = t.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if t.month == 12:
            month_end = t.replace(year=t.year + 1, month=1, day=1,
                                  hour=0, minute=0, second=0, microsecond=0)
        else:
            month_end = t.replace(month=t.month + 1, day=1,
                                  hour=0, minute=0, second=0, microsecond=0)
        bounds.append([month_start, month_end])
    time_bnds = np.array(bounds, dtype="datetime64[ns]")

    # Build xarray dataset with corrected time + bounds
    nv = np.arange(2)
    time_ds = xr.Dataset(
        {"time_bnds": (["time", "nv"], time_bnds)},
        coords={"time": mid_month, "nv": nv},
    )
    time_ds.time.attrs["bounds"] = "time_bnds"
    time_ds.time.attrs["calendar"] = "standard"
    time_ds.time.attrs["cell_methods"] = "time: mean"
    time_ds.time.encoding["dtype"] = "int64"
    time_ds.time.encoding["units"] = "nanoseconds since 1970-01-01"

    # Serialize to in-memory zarr store — this produces correct metadata
    mem_store = zarr.storage.MemoryStore()
    time_ds.to_zarr(mem_store, mode="w")

    # Copy time-related keys from zarr store into kerchunk refs
    refs = combined["refs"]
    for key in list(mem_store):
        key_str = key if isinstance(key, str) else key.decode()
        prefix = key_str.split("/")[0]
        if prefix in ("time", "time_bnds", "nv"):
            data = mem_store[key]
            if key_str.endswith(".zattrs") or key_str.endswith(".zarray") or key_str.endswith(".zgroup"):
                refs[key_str] = data.decode() if isinstance(data, bytes) else data
            else:
                # Chunk data — inline as base64
                import base64
                refs[key_str] = "base64:" + base64.b64encode(data).decode()

    return combined
```

Then call `_fix_time(combined, merra2_dir)` in `consolidate_merra2()` right after `mzz.translate()` and before `_make_relative()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_consolidate.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate.py
git commit -m "feat: add CF-compliant time representation to Kerchunk store"
```

---

### Task 4: Add global attributes and relative path tests

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py`
- Modify: `tests/test_consolidate.py`

- [ ] **Step 1: Write tests for global attributes and relative paths**

Append to `tests/test_consolidate.py`:

```python
def test_global_attributes(merra2_dir):
    """Reference store has CF and provenance global attributes."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    run_dir = merra2_dir.parent.parent.parent
    consolidate_merra2(run_dir=run_dir, variables=["GWETTOP"])

    ref_path = merra2_dir / "merra2_refs.json"
    refs = json.loads(ref_path.read_text())
    root_attrs = json.loads(refs["refs"][".zattrs"])

    assert root_attrs["Conventions"] == "CF-1.8"
    assert "nhf-spatial-targets" in root_attrs["history"]
    assert "M2TMNXLND" in root_attrs["source"]
    assert "time_modification_note" in root_attrs


def test_relative_paths(merra2_dir):
    """All file references use relative paths starting with './'."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    run_dir = merra2_dir.parent.parent.parent
    consolidate_merra2(run_dir=run_dir, variables=["GWETTOP"])

    ref_path = merra2_dir / "merra2_refs.json"
    refs = json.loads(ref_path.read_text())

    for key, val in refs["refs"].items():
        if isinstance(val, list) and len(val) >= 1 and isinstance(val[0], str):
            path = val[0]
            # Should be relative, not absolute
            assert not path.startswith("/"), f"Absolute path in ref '{key}': {path}"
            assert path.startswith("./"), f"Non-relative path in ref '{key}': {path}"


def test_provenance_return(merra2_dir):
    """consolidate_merra2 returns a dict with provenance keys."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    run_dir = merra2_dir.parent.parent.parent
    result = consolidate_merra2(run_dir=run_dir, variables=["GWETTOP", "GWETROOT", "GWETPROF"])

    assert result["kerchunk_ref"] == "data/raw/merra2/merra2_refs.json"
    assert "last_consolidated_utc" in result
    assert result["n_files"] == 3
    assert result["variables"] == ["GWETTOP", "GWETROOT", "GWETPROF"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_consolidate.py::test_global_attributes tests/test_consolidate.py::test_relative_paths tests/test_consolidate.py::test_provenance_return -v`
Expected: `test_global_attributes` FAILS (no CF attributes yet), others may pass

- [ ] **Step 3: Add global attribute injection to consolidate_merra2**

In `consolidate_merra2()`, after `_fix_time()` and before `_make_relative()`, add:

```python
    # Add CF and provenance global attributes
    import nhf_spatial_targets.catalog as _catalog

    meta = _catalog.source("merra2")
    root_attrs = ujson.loads(combined["refs"].get(".zattrs", "{}"))
    if isinstance(root_attrs, str):
        root_attrs = ujson.loads(root_attrs)
    root_attrs.update({
        "Conventions": "CF-1.8",
        "history": f"Kerchunk virtual Zarr created by nhf-spatial-targets v{__version__}",
        "source": f"NASA MERRA-2 {meta['access']['short_name']} v{meta['access'].get('version', 'unknown')}",
        "time_modification_note": (
            "Original timestamps (YYYY-01-01T00:30:00) shifted to mid-month "
            "(15th) for consistency. See time_bnds for exact averaging periods."
        ),
        "references": meta["access"]["url"],
    })
    combined["refs"][".zattrs"] = ujson.dumps(root_attrs)
```

- [ ] **Step 4: Run all consolidation tests**

Run: `pixi run -e dev pytest tests/test_consolidate.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All tests pass, no lint errors

- [ ] **Step 6: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate.py
git commit -m "feat: add CF global attributes and relative path handling"
```

---

## Chunk 2: Incremental Fetch and Integration

### Task 5: Add incremental fetch logic to merra2.py

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/merra2.py`
- Modify: `tests/test_merra2.py`

The current `fetch_merra2()` downloads all granules for the requested period. This task adds manifest-aware incremental downloading.

- [ ] **Step 1: Write test for incremental fetch — skips existing months**

Append to `tests/test_merra2.py`:

```python
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("earthaccess.login")
def test_incremental_skips_existing(mock_login, mock_search, mock_dl, run_dir):
    """Months already in manifest are not re-downloaded."""
    mock_login.return_value = MagicMock(authenticated=True)

    # Pre-populate manifest with Jan 2010 already downloaded
    manifest = {
        "sources": {
            "merra2": {
                "period": "2010/2010",
                "files": [
                    {
                        "path": "data/raw/merra2/MERRA2_300.tavgM_2d_lnd_Nx.201001.nc4",
                        "year_month": "2010-01",
                        "size_bytes": 100,
                        "downloaded_utc": "2026-01-01T00:00:00+00:00",
                    }
                ],
            }
        }
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    # Create the existing file on disk
    existing = run_dir / "data" / "raw" / "merra2" / "MERRA2_300.tavgM_2d_lnd_Nx.201001.nc4"
    existing.write_bytes(b"fake")

    from nhf_spatial_targets.fetch.merra2 import _months_in_period, _existing_months

    existing_months = _existing_months(run_dir)
    assert "2010-01" in existing_months

    requested = _months_in_period("2010/2010")
    assert len(requested) == 12

    missing = [m for m in requested if m not in existing_months]
    assert "2010-01" not in missing
    assert len(missing) == 11
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_merra2.py::test_incremental_skips_existing -v`
Expected: FAIL with `ImportError: cannot import name '_months_in_period'`

- [ ] **Step 3: Implement helper functions**

Add to `src/nhf_spatial_targets/fetch/merra2.py`:

```python
import re

_MERRA2_DATE_RE = re.compile(r"\.(\d{4})(\d{2})\.nc4$")


def _months_in_period(period: str) -> list[str]:
    """Return list of 'YYYY-MM' strings for every month in the period."""
    parts = period.split("/")
    start_year, end_year = int(parts[0]), int(parts[1])
    months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            months.append(f"{year}-{month:02d}")
    return months


def _existing_months(run_dir: Path) -> set[str]:
    """Read manifest.json and return set of year_month values already fetched."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text())
        files = manifest.get("sources", {}).get("merra2", {}).get("files", [])
        return {f["year_month"] for f in files if "year_month" in f}
    except (json.JSONDecodeError, KeyError):
        return set()


def _year_month_from_path(path: Path) -> str:
    """Extract 'YYYY-MM' from a MERRA-2 filename like '...201007.nc4'."""
    m = _MERRA2_DATE_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot extract date from MERRA-2 filename: {path.name}")
    return f"{m.group(1)}-{m.group(2)}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_merra2.py::test_incremental_skips_existing -v`
Expected: PASS

- [ ] **Step 5: Write test for year_month extraction from filenames**

Append to `tests/test_merra2.py`:

```python
def test_year_month_from_filename():
    """Extract YYYY-MM from MERRA-2 filename for all collection numbers."""
    from nhf_spatial_targets.fetch.merra2 import _year_month_from_path

    # Collection 300 (2001-2010)
    assert _year_month_from_path(Path("MERRA2_300.tavgM_2d_lnd_Nx.201007.nc4")) == "2010-07"
    # Collection 100 (1980-1991)
    assert _year_month_from_path(Path("MERRA2_100.tavgM_2d_lnd_Nx.198001.nc4")) == "1980-01"
    # Collection 200 (1992-2000)
    assert _year_month_from_path(Path("MERRA2_200.tavgM_2d_lnd_Nx.199506.nc4")) == "1995-06"
    # Collection 400 (2011-present)
    assert _year_month_from_path(Path("MERRA2_400.tavgM_2d_lnd_Nx.202312.nc4")) == "2023-12"
```

- [ ] **Step 6: Run test and verify**

Run: `pixi run -e dev pytest tests/test_merra2.py::test_year_month_from_filename -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/nhf_spatial_targets/fetch/merra2.py tests/test_merra2.py
git commit -m "feat: add incremental fetch helpers (period months, existing months, filename parsing)"
```

---

### Task 6: Refactor fetch_merra2() for incremental download + consolidation

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/merra2.py`
- Modify: `tests/test_merra2.py`

This task modifies the main `fetch_merra2()` function to:
1. Check manifest for already-downloaded months
2. Filter search results to only download missing granules
3. Call `consolidate_merra2()` after download
4. Update `manifest.json` with complete provenance, merging (not overwriting) file records

**IMPORTANT:** After this refactor, `fetch_merra2()` calls `consolidate_merra2()` internally. All existing tests that call `fetch_merra2()` to completion must mock `consolidate_merra2` at its **import site** in merra2.py (not its definition site in consolidate.py).

- [ ] **Step 1: Update _fake_download to produce MERRA-2-conforming filenames**

The existing `_fake_download` creates files like `MERRA2_201001.nc4` which don't match `_MERRA2_DATE_RE` (expects a dot before `YYYYMM`). Update the helper:

```python
def _fake_download(run_dir: Path, n: int = 1) -> list[str]:
    """Create fake downloaded files and return their paths."""
    paths = []
    for i in range(n):
        f = run_dir / "data" / "raw" / "merra2" / f"MERRA2_300.tavgM_2d_lnd_Nx.2010{i + 1:02d}.nc4"
        f.write_bytes(b"fake")
        paths.append(str(f))
    return paths
```

- [ ] **Step 2: Add consolidate_merra2 mock to all existing tests that call fetch_merra2**

Add `@patch("nhf_spatial_targets.fetch.merra2.consolidate_merra2")` to these existing tests:
- `test_search_params`
- `test_output_dir`
- `test_provenance_record`
- `test_superseded_warning`

The mock should return a minimal consolidation dict:

```python
_MOCK_CONSOLIDATION = {
    "kerchunk_ref": "data/raw/merra2/merra2_refs.json",
    "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
    "n_files": 1,
    "variables": ["GWETTOP", "GWETROOT", "GWETPROF"],
}
```

For each test, add the mock decorator and accept the mock argument. Example for `test_search_params`:

```python
@patch("nhf_spatial_targets.fetch.merra2.consolidate_merra2", return_value=_MOCK_CONSOLIDATION)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("earthaccess.login")
def test_search_params(mock_login, mock_search, mock_dl, mock_consolidate, run_dir):
    ...
```

Also remove `test_missing_downloaded_file_raises` — the refactored code no longer checks individual downloaded paths against disk (it scans all `.nc4` files via glob instead). The consolidation module's own `FileNotFoundError` covers the no-files case.

- [ ] **Step 3: Write test for manifest update**

```python
@patch("nhf_spatial_targets.fetch.merra2.consolidate_merra2", return_value=_MOCK_CONSOLIDATION)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("earthaccess.login")
def test_manifest_updated(mock_login, mock_search, mock_dl, mock_consolidate, run_dir):
    """fetch_merra2 writes provenance to manifest.json."""
    mock_login.return_value = MagicMock(authenticated=True)
    mock_search.return_value = [_mock_granule("g1")]
    mock_dl.return_value = _fake_download(run_dir)

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    result = fetch_merra2(run_dir=run_dir, period="2010/2010")

    updated_manifest = json.loads((run_dir / "manifest.json").read_text())
    merra2_entry = updated_manifest["sources"]["merra2"]
    assert merra2_entry["period"] == "2010/2010"
    assert len(merra2_entry["files"]) > 0
    assert "year_month" in merra2_entry["files"][0]
    assert "kerchunk_ref" in merra2_entry
```

- [ ] **Step 4: Write test for provenance preservation on re-run**

```python
@patch("nhf_spatial_targets.fetch.merra2.consolidate_merra2", return_value=_MOCK_CONSOLIDATION)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("earthaccess.login")
def test_manifest_preserves_download_timestamp(mock_login, mock_search, mock_dl, mock_consolidate, run_dir):
    """Re-running fetch preserves original downloaded_utc for existing files."""
    mock_login.return_value = MagicMock(authenticated=True)
    mock_search.return_value = [_mock_granule("g1")]

    # Pre-populate manifest with an existing file record
    original_ts = "2026-01-01T00:00:00+00:00"
    manifest = {
        "sources": {
            "merra2": {
                "files": [
                    {
                        "path": "data/raw/merra2/MERRA2_300.tavgM_2d_lnd_Nx.201001.nc4",
                        "year_month": "2010-01",
                        "size_bytes": 100,
                        "downloaded_utc": original_ts,
                    }
                ],
            }
        }
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    # File exists on disk
    existing = run_dir / "data" / "raw" / "merra2" / "MERRA2_300.tavgM_2d_lnd_Nx.201001.nc4"
    existing.write_bytes(b"fake")

    # Download returns nothing new (empty needed list triggers skip)
    mock_dl.return_value = []

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2
    fetch_merra2(run_dir=run_dir, period="2010/2010")

    updated = json.loads((run_dir / "manifest.json").read_text())
    jan_file = next(f for f in updated["sources"]["merra2"]["files"] if f["year_month"] == "2010-01")
    assert jan_file["downloaded_utc"] == original_ts
```

- [ ] **Step 5: Refactor fetch_merra2()**

Replace the current `fetch_merra2()` body. Key changes from original:

1. After auth + bbox, check `_existing_months()` and skip download if all months present
2. Build file inventory from disk, preserving `downloaded_utc` from manifest for known files
3. Call `consolidate_merra2()` after download
4. `_update_manifest()` merges file records by `year_month` and preserves non-merra2 keys

```python
def fetch_merra2(run_dir: Path, period: str) -> dict:
    """Download MERRA-2 M2TMNXLND granules for the given period.

    Supports incremental download — months already recorded in
    ``manifest.json`` are skipped. After downloading, builds a
    Kerchunk virtual Zarr reference store and updates the manifest.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory. Reads ``fabric.json`` for bbox,
        writes files to ``data/raw/merra2/``.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    meta = _catalog.source(_SOURCE_KEY)
    short_name = meta["access"]["short_name"]

    if meta.get("status") == "superseded":
        warnings.warn(
            f"Source '{_SOURCE_KEY}' is superseded. "
            f"Consider using '{meta.get('superseded_by', 'unknown')}'.",
            DeprecationWarning,
            stacklevel=2,
        )

    auth = earthaccess.login()
    if auth is None or not auth.authenticated:
        raise RuntimeError(
            "NASA Earthdata login failed. Register at "
            "https://urs.earthdata.nasa.gov/users/new"
        )
    logger.info("Authenticated with NASA Earthdata")

    fabric_path = run_dir / "fabric.json"
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"fabric.json not found in {run_dir}. "
            f"Run 'nhf-targets init' to create a run workspace first."
        )
    try:
        fabric = json.loads(fabric_path.read_text())
        bbox = fabric["bbox_buffered"]
        bbox_tuple = (bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"])
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(
            f"fabric.json in {run_dir} is malformed or missing required "
            f"fields (bbox_buffered.{{minx,miny,maxx,maxy}}). "
            f"Re-run 'nhf-targets init' to regenerate it."
        ) from exc

    # Determine which months need downloading
    already_have = _existing_months(run_dir)
    all_months = _months_in_period(period)
    needed = [m for m in all_months if m not in already_have]

    if not needed:
        logger.info(
            "All %d months already downloaded, skipping to consolidation",
            len(all_months),
        )
    else:
        temporal = _parse_period(period)
        logger.debug("bbox=%s, temporal=%s", bbox_tuple, temporal)

        granules = earthaccess.search_data(
            short_name=short_name,
            bounding_box=bbox_tuple,
            temporal=temporal,
        )
        logger.info("Found %d granules for %s", len(granules), short_name)

        if not granules:
            raise ValueError(
                f"No granules found for {short_name} with "
                f"bbox={bbox_tuple}, temporal={temporal}"
            )

        output_dir = run_dir / "data" / "raw" / _SOURCE_KEY
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = earthaccess.download(
            granules,
            local_path=str(output_dir),
        )

        if not downloaded:
            raise RuntimeError(
                f"earthaccess.download() returned no files for "
                f"{len(granules)} granules. Check network connectivity "
                f"and Earthdata credentials."
            )
        if len(downloaded) < len(granules):
            logger.warning(
                "Partial download: got %d of %d granules",
                len(downloaded),
                len(granules),
            )
        logger.info("Downloaded %d files to %s", len(downloaded), output_dir)

    # Build file inventory from all .nc4 files on disk
    output_dir = run_dir / "data" / "raw" / _SOURCE_KEY
    all_nc_files = sorted(output_dir.glob("*.nc4"))

    # Preserve original downloaded_utc for files already in manifest
    existing_timestamps = _existing_file_timestamps(run_dir)
    now_utc = datetime.now(timezone.utc).isoformat()

    files = []
    for p in all_nc_files:
        rel = str(p.relative_to(run_dir))
        ym = _year_month_from_path(p)
        files.append({
            "path": rel,
            "year_month": ym,
            "size_bytes": p.stat().st_size,
            "downloaded_utc": existing_timestamps.get(ym, now_utc),
        })

    # Consolidate into Kerchunk reference store
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    var_names = [v["name"] for v in meta["variables"]]
    consolidation = consolidate_merra2(run_dir=run_dir, variables=var_names)

    # Compute effective period from actual files on disk
    if files:
        all_ym = sorted(f["year_month"] for f in files)
        effective_start = all_ym[0][:4]
        effective_end = all_ym[-1][:4]
        effective_period = f"{effective_start}/{effective_end}"
    else:
        effective_period = period

    # Update manifest.json (merge, don't overwrite)
    _update_manifest(run_dir, effective_period, bbox, meta, files, consolidation)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "variables": meta["variables"],
        "period": effective_period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "files": files,
        "kerchunk_ref": consolidation["kerchunk_ref"],
    }


def _existing_file_timestamps(run_dir: Path) -> dict[str, str]:
    """Return {year_month: downloaded_utc} from existing manifest."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text())
        files = manifest.get("sources", {}).get("merra2", {}).get("files", [])
        return {
            f["year_month"]: f["downloaded_utc"]
            for f in files
            if "year_month" in f and "downloaded_utc" in f
        }
    except (json.JSONDecodeError, KeyError):
        return {}


def _update_manifest(
    run_dir: Path,
    period: str,
    bbox: dict,
    meta: dict,
    files: list[dict],
    consolidation: dict,
) -> None:
    """Merge MERRA-2 provenance into manifest.json.

    Uses dict.update() to preserve any non-merra2 keys in the
    sources block (e.g., from other pipeline steps).
    """
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"sources": {}, "steps": []}

    merra2 = manifest["sources"].get("merra2", {})
    merra2.update({
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "period": period,
        "bbox": bbox,
        "variables": [v["name"] for v in meta["variables"]],
        "files": files,
        "kerchunk_ref": consolidation["kerchunk_ref"],
        "last_consolidated_utc": consolidation["last_consolidated_utc"],
    })
    manifest["sources"]["merra2"] = merra2

    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Updated manifest.json with MERRA-2 provenance")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_merra2.py -v`
Expected: All tests pass

- [ ] **Step 7: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add src/nhf_spatial_targets/fetch/merra2.py tests/test_merra2.py
git commit -m "feat: incremental MERRA-2 fetch with manifest tracking and Kerchunk consolidation"
```

---

### Task 7: Update integration test and CLI output

**Files:**
- Modify: `tests/test_merra2.py`
- Modify: `src/nhf_spatial_targets/cli.py`

- [ ] **Step 1: Update integration test for new variables and Kerchunk**

In `tests/test_merra2.py`, replace `test_fetch_merra2_real_download`:

```python
@pytest.mark.integration
def test_fetch_merra2_real_download(tmp_path):
    """End-to-end download of one year of MERRA-2 data."""
    import xarray as xr

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "data" / "raw" / "merra2").mkdir(parents=True)
    fabric = {
        "bbox_buffered": {
            "minx": -105.0,
            "miny": 39.0,
            "maxx": -104.0,
            "maxy": 40.0,
        }
    }
    (run_dir / "fabric.json").write_text(json.dumps(fabric))

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    result = fetch_merra2(run_dir=run_dir, period="2010/2010")

    assert result["source_key"] == "merra2"
    assert len(result["files"]) == 12
    assert "kerchunk_ref" in result

    # Verify Kerchunk reference store works
    ref_path = run_dir / result["kerchunk_ref"]
    assert ref_path.exists()

    import fsspec
    fs = fsspec.filesystem("reference", fo=str(ref_path), target_protocol="file")
    ds = xr.open_zarr(fs.get_mapper(""), consolidated=False)
    assert "GWETTOP" in ds.data_vars
    assert "GWETROOT" in ds.data_vars
    assert "GWETPROF" in ds.data_vars
    assert "SFMC" not in ds.data_vars
    assert len(ds.time) == 12
    ds.close()

    # Verify manifest was written
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "merra2" in manifest["sources"]
```

- [ ] **Step 2: Update CLI to show Kerchunk info**

In `src/nhf_spatial_targets/cli.py`, update `fetch_merra2_cmd` to show Kerchunk ref path:

```python
    console.print(
        f"[green]Downloaded {len(result['files'])} files "
        f"to {run_dir / 'data' / 'raw' / 'merra2'}[/green]"
    )
    if "kerchunk_ref" in result:
        console.print(
            f"[green]Kerchunk reference store: {run_dir / result['kerchunk_ref']}[/green]"
        )
```

- [ ] **Step 3: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/nhf_spatial_targets/cli.py tests/test_merra2.py
git commit -m "feat: update integration test and CLI for Kerchunk consolidation"
```

---

## Implementation Notes

### Kerchunk API Reference

```python
# Scan single file — MUST pass open file object, not string path
from kerchunk.hdf import SingleHdf5ToZarr
with open(filepath, "rb") as f:
    h5chunks = SingleHdf5ToZarr(f, filepath)
    refs = h5chunks.translate()  # returns dict with "version", "refs" keys

# Filter variables from refs dict
# Drop all keys where key.split("/")[0] is not in the keep set
# Preserve top-level keys (no "/") like .zattrs, .zgroup
refs["refs"] = {k: v for k, v in refs["refs"].items() if should_keep(k)}

# Combine multiple files
from kerchunk.combine import MultiZarrToZarr
mzz = MultiZarrToZarr(singles, concat_dims=["time"], identical_dims=["lat", "lon"])
combined = mzz.translate()

# Write to JSON
import ujson
Path("refs.json").write_text(ujson.dumps(combined))

# Open virtual dataset — must pass target_protocol="file" for local files
import fsspec, xarray
fs = fsspec.filesystem("reference", fo="refs.json", target_protocol="file")
ds = xarray.open_zarr(fs.get_mapper(""), consolidated=False)
```

### MERRA-2 Filename Convention

Files follow: `MERRA2_{collection}.tavgM_2d_lnd_Nx.{YYYYMM}.nc4`

Collection numbers vary by era:
- `100`: 1980–1991
- `200`: 1992–2000
- `300`: 2001–2010
- `400`: 2011–present

### Variable Names from Actual Files

| Variable | long_name | units | shape |
|----------|-----------|-------|-------|
| GWETTOP | surface_soil_wetness | 1 (dimensionless) | (1, 361, 576) |
| GWETROOT | root_zone_soil_wetness | 1 (dimensionless) | (1, 361, 576) |
| GWETPROF | ave_prof_soil_moisture | 1 (dimensionless) | (1, 361, 576) |

### Time Coordinate

Original: `2010-01-01T00:30:00` (30 minutes into the first day)
After fixup: `2010-01-15T00:00:00` (mid-month)
Bounds: `[2010-01-01, 2010-02-01]` (first-of-month to first-of-next-month)
December bounds wrap correctly: `[2010-12-01, 2011-01-01]`

### Reviewer Findings Addressed

| Issue | Fix |
|-------|-----|
| `SingleHdf5ToZarr` needs file object, not string | Use `open(nc, "rb")` context manager |
| `time_bnds` needs `_ARRAY_DIMENSIONS` and datetime metadata | Use xarray→zarr in-memory serialization instead of manual base64 |
| Mock patch path for `consolidate_merra2` | Patch at import site: `nhf_spatial_targets.fetch.merra2.consolidate_merra2` |
| `_fake_download` filenames don't match regex | Updated to `MERRA2_300.tavgM_2d_lnd_Nx.YYYYMM.nc4` |
| `_update_manifest` overwrites instead of merging | Use `dict.update()` to preserve existing keys |
| `downloaded_utc` refreshed on re-run | Preserve original timestamp via `_existing_file_timestamps()` |
| Period overlap loses start year | Compute effective period from actual files on disk |
| Consumer pattern needs `target_protocol="file"` | Added to all fsspec calls |
| Kerchunk version constraint missing | Added `>=0.2` to pixi.toml |
