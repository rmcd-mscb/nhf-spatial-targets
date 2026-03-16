# WaterGAP 2.2d Fetch Module Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the WaterGAP 2.2d groundwater recharge fetch module, downloading from PANGAEA via `pangaeapy`, fixing CF compliance issues, and updating the catalog/docs.

**Architecture:** Single-file download from PANGAEA via `pangaeapy`, CF compliance fix-up (time encoding reconstruction, grid_mapping addition), manifest-based skip logic, no consolidation. Module named `pangaea.py` (provider-grouped, consistent with `sciencebase.py`).

**Tech Stack:** `pangaeapy` (PANGAEA client), `xarray` (NetCDF read/write), `numpy`/`pandas` (time reconstruction)

**Spec:** `docs/superpowers/specs/2026-03-16-watergap22d-fetch-design.md`

---

## Chunk 1: Catalog, Variables, and Documentation Updates

### Task 1: Update catalog/sources.yml

**Files:**
- Modify: `catalog/sources.yml:155-174`

- [ ] **Step 1: Update the watergap22d entry**

Replace lines 155-174 of `catalog/sources.yml` with:

```yaml
  watergap22d:
    name: WaterGAP 2.2d Global Hydrological Model (substitute for 2.2a)
    description: >
      Open-access successor to WaterGAP 2.2a on PANGAEA. Diffuse groundwater
      recharge (Rg) used as one of two sources for the recharge calibration
      target range (normalized min-max method).
    citations:
      - "Müller Schmied, H., and others, 2021, doi:10.5194/gmd-14-1037-2021"
    access:
      type: pangaea
      doi: "10.1594/PANGAEA.918447"
      url: https://doi.pangaea.de/10.1594/PANGAEA.918447
      file: watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4
    variables:
      - name: groundwater_recharge
        file_variable: qrdif
        long_name: diffuse groundwater recharge
        units: kg m-2 s-1
    time_step: monthly
    period: "1901/2016"
    spatial_extent: global
    spatial_resolution: 0.5 degree
    units: kg m-2 s-1
    license: CC BY-NC 4.0
    status: current
```

- [ ] **Step 2: Run catalog tests to verify YAML is valid**

Run: `pixi run -e dev test tests/test_catalog.py -v`
Expected: All catalog tests PASS

- [ ] **Step 3: Commit**

```bash
git add catalog/sources.yml
git commit -m "catalog: update watergap22d with confirmed PANGAEA access details"
```

### Task 2: Update catalog/variables.yml

**Files:**
- Modify: `catalog/variables.yml:69-70`

- [ ] **Step 1: Update recharge sources list**

Change line 69-70 from:
```yaml
      - reitz2017
      - watergap22a   # or watergap22d (open-access substitute on PANGAEA)
```
to:
```yaml
      - reitz2017
      - watergap22d
```

- [ ] **Step 2: Run catalog tests**

Run: `pixi run -e dev test tests/test_catalog.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add catalog/variables.yml
git commit -m "catalog: switch recharge source from watergap22a to watergap22d"
```

### Task 3: Update CLAUDE.md and README.md

**Files:**
- Modify: `CLAUDE.md:102-104`
- Modify: `README.md:139,154-156`

- [ ] **Step 1: Move WaterGAP from "Still open" to "Resolved" in CLAUDE.md**

In `CLAUDE.md`, add to the end of the "Resolved" list (after the SSEBop line):
```
- WaterGAP 2.2d — confirmed: doi:10.1594/PANGAEA.918447, variable qrdif (diffuse groundwater recharge), 1901-2016 monthly, 0.5° global, CC BY-NC 4.0
```

Remove from "Still open":
```
- WaterGAP 2.2a — registration-gated; substitute candidate is WaterGAP 2.2d on PANGAEA (doi:10.1594/PANGAEA.918447)
```

- [ ] **Step 2: Update README.md task status**

Change line 139 from:
```
| WaterGAP 2.2d fetch (PANGAEA) | Not started |
```
to:
```
| WaterGAP 2.2d fetch (PANGAEA) | Done |
```

In the "Known Gaps" section, remove the WaterGAP line from "Still open" and add to "Resolved":
```
- WaterGAP 2.2d — confirmed on PANGAEA (doi:10.1594/PANGAEA.918447), CC BY-NC 4.0
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: mark WaterGAP 2.2d as resolved in CLAUDE.md and README"
```

---

## Chunk 2: CF Compliance Fix-up Function and Tests (TDD)

### Task 4: Write failing tests for CF time reconstruction

**Files:**
- Create: `tests/test_pangaea.py`

- [ ] **Step 1: Write the test for time reconstruction**

Create `tests/test_pangaea.py`:

```python
"""Tests for PANGAEA fetch module (WaterGAP 2.2d)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


def _make_watergap_nc(path: Path, n_times: int = 24) -> Path:
    """Create a synthetic WaterGAP-style NC4 with non-CF time encoding.

    Mimics the real file: time as float offsets with
    units='months since 1901-01-01', calendar='proleptic_gregorian'.
    """
    time_vals = np.arange(n_times, dtype=np.float32)
    lat = np.array([89.75, 45.25, 0.25, -44.75], dtype=np.float32)
    lon = np.array([-179.75, -90.25, 0.25, 89.75], dtype=np.float32)
    data = np.random.rand(n_times, len(lat), len(lon)).astype(np.float32)

    ds = xr.Dataset(
        {"qrdif": (["time", "lat", "lon"], data)},
        coords={
            "time": xr.Variable("time", time_vals, attrs={
                "units": "months since 1901-01-01",
                "calendar": "proleptic_gregorian",
                "standard_name": "time",
            }),
            "lat": xr.Variable("lat", lat, attrs={
                "units": "degrees_north",
                "standard_name": "latitude",
            }),
            "lon": xr.Variable("lon", lon, attrs={
                "units": "degrees_east",
                "standard_name": "longitude",
            }),
        },
        attrs={
            "conventions": "partly ALMA, CF and ISIMIP2b protocol",
            "title": "Test WaterGAP",
        },
    )
    ds["qrdif"].attrs.update({
        "standard_name": "qrdif",
        "long_name": "diffuse groundwater recharge",
        "units": "kg m-2 s-1",
    })
    ds.to_netcdf(path, format="NETCDF4")
    return path


def test_cf_fixup_reconstructs_time(tmp_path: Path):
    """Time coordinate must be proper datetime64 after CF fix-up."""
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    raw = _make_watergap_nc(tmp_path / "raw.nc4", n_times=24)
    fixed = _cf_fixup(raw, tmp_path / "fixed.nc")

    ds = xr.open_dataset(fixed)
    assert ds.time.dtype == np.dtype("datetime64[ns]")
    # 24 months from 1901-01 = 1901-01 through 1902-12
    assert str(ds.time.values[0])[:7] == "1901-01"
    assert str(ds.time.values[12])[:7] == "1902-01"
    assert str(ds.time.values[23])[:7] == "1902-12"
    ds.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev test tests/test_pangaea.py::test_cf_fixup_reconstructs_time -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

### Task 5: Write failing tests for grid_mapping and Conventions

**Files:**
- Modify: `tests/test_pangaea.py`

- [ ] **Step 1: Add grid_mapping and Conventions tests**

Append to `tests/test_pangaea.py`:

```python
def test_cf_fixup_adds_grid_mapping(tmp_path: Path):
    """CF fix-up must add crs variable and grid_mapping attr to qrdif."""
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    raw = _make_watergap_nc(tmp_path / "raw.nc4")
    fixed = _cf_fixup(raw, tmp_path / "fixed.nc")

    ds = xr.open_dataset(fixed)
    assert "crs" in ds.data_vars
    assert ds["qrdif"].attrs.get("grid_mapping") == "crs"
    assert ds["crs"].attrs.get("grid_mapping_name") == "latitude_longitude"
    ds.close()


def test_cf_fixup_sets_conventions(tmp_path: Path):
    """CF fix-up must set Conventions to CF-1.6."""
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    raw = _make_watergap_nc(tmp_path / "raw.nc4")
    fixed = _cf_fixup(raw, tmp_path / "fixed.nc")

    ds = xr.open_dataset(fixed)
    assert ds.attrs.get("Conventions") == "CF-1.6"
    ds.close()


def test_cf_fixup_preserves_data(tmp_path: Path):
    """CF fix-up must not alter the qrdif data values."""
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    raw = _make_watergap_nc(tmp_path / "raw.nc4", n_times=12)
    raw_ds = xr.open_dataset(raw, decode_times=False)
    raw_values = raw_ds["qrdif"].values.copy()
    raw_ds.close()

    fixed = _cf_fixup(raw, tmp_path / "fixed.nc")
    fixed_ds = xr.open_dataset(fixed)
    np.testing.assert_array_equal(fixed_ds["qrdif"].values, raw_values)
    fixed_ds.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test tests/test_pangaea.py -v`
Expected: All 4 tests FAIL

### Task 6: Implement _cf_fixup function

**Files:**
- Create: `src/nhf_spatial_targets/fetch/pangaea.py`

- [ ] **Step 1: Create the pangaea.py module with _cf_fixup**

Create `src/nhf_spatial_targets/fetch/pangaea.py`:

```python
"""Fetch datasets hosted on PANGAEA (WaterGAP 2.2d)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

_SOURCE_KEY = "watergap22d"


def _cf_fixup(raw_path: Path, output_path: Path) -> Path:
    """Fix CF compliance issues in a WaterGAP NC4 file.

    Addresses:
    - Time encoding: reconstructs 'months since 1901-01-01' offsets as datetime64
    - Grid mapping: adds WGS84 crs variable and grid_mapping attr on data vars
    - Conventions: sets to CF-1.6

    Parameters
    ----------
    raw_path : Path
        Path to the original (non-CF-compliant) NetCDF file.
    output_path : Path
        Path to write the corrected NetCDF file.

    Returns
    -------
    Path
        Path to the written CF-compliant file (same as output_path).
    """
    ds = xr.open_dataset(raw_path, decode_times=False)

    # --- Reconstruct time coordinate ---
    time_offsets = ds.time.values.astype(int)
    base_year = 1901
    base_month = 1
    dates = []
    for offset in time_offsets:
        total_months = base_month - 1 + offset
        year = base_year + total_months // 12
        month = 1 + total_months % 12
        dates.append(pd.Timestamp(year=int(year), month=int(month), day=1))
    new_time = pd.DatetimeIndex(dates)
    ds = ds.assign_coords(time=new_time)
    ds.time.attrs = {"standard_name": "time", "long_name": "time", "axis": "T"}

    # --- Add CRS variable with WGS84 grid mapping ---
    crs = xr.DataArray(
        np.int32(0),
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
            "longitude_of_prime_meridian": 0.0,
            "crs_wkt": 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]',
        },
    )
    ds["crs"] = crs

    # --- Set grid_mapping on data variables ---
    for var in ds.data_vars:
        if var != "crs":
            ds[var].attrs["grid_mapping"] = "crs"

    # --- Set Conventions ---
    ds.attrs["Conventions"] = "CF-1.6"
    # Remove the old non-standard conventions attr if present
    ds.attrs.pop("conventions", None)

    # --- Write ---
    ds.to_netcdf(output_path, format="NETCDF4")
    ds.close()

    logger.info("Wrote CF-compliant file: %s", output_path)
    return output_path
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pixi run -e dev test tests/test_pangaea.py -v`
Expected: All 4 tests PASS

- [ ] **Step 3: Run fmt and lint**

Run: `pixi run -e dev fmt && pixi run -e dev lint`
Expected: No issues

- [ ] **Step 4: Commit**

```bash
git add src/nhf_spatial_targets/fetch/pangaea.py tests/test_pangaea.py
git commit -m "feat: add CF compliance fix-up for WaterGAP NC4 files"
```

---

## Chunk 3: Fetch Function and Tests (TDD)

### Task 7: Write failing tests for fetch_watergap22d

**Files:**
- Modify: `tests/test_pangaea.py`

- [ ] **Step 1: Add run_dir fixture and test for missing fabric.json**

Append to `tests/test_pangaea.py`:

```python
from unittest.mock import patch, MagicMock


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """Create a minimal run workspace."""
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "watergap22d").mkdir(parents=True)
    fabric = {
        "bbox_buffered": {
            "minx": -125.1, "miny": 23.9,
            "maxx": -65.9, "maxy": 50.1,
        }
    }
    (rd / "fabric.json").write_text(json.dumps(fabric))
    return rd


def test_missing_fabric_raises(tmp_path: Path):
    """FileNotFoundError when fabric.json is absent."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    rd = tmp_path / "run"
    rd.mkdir()
    with pytest.raises(FileNotFoundError, match="fabric.json"):
        fetch_watergap22d(run_dir=rd, period="2000/2009")
```

- [ ] **Step 2: Add test for skipping existing file**

Append to `tests/test_pangaea.py`:

```python
def test_skips_existing_file(run_dir: Path):
    """Skip download when CF-corrected file already exists."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    output_dir = run_dir / "data" / "raw" / "watergap22d"
    cf_file = output_dir / "watergap22d_qrdif_cf.nc"
    _make_watergap_nc(output_dir / "watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4", n_times=12)

    # Create a pre-existing CF-corrected file
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup
    _cf_fixup(
        output_dir / "watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4",
        cf_file,
    )

    with patch("nhf_spatial_targets.fetch.pangaea.PanDataSet") as mock_pan:
        result = fetch_watergap22d(run_dir=run_dir, period="2000/2009")
        mock_pan.assert_not_called()
        assert result["source_key"] == "watergap22d"
```

- [ ] **Step 3: Add test for download and manifest update**

Append to `tests/test_pangaea.py`:

```python
def test_downloads_and_updates_manifest(run_dir: Path):
    """Download via pangaeapy, CF fix-up, and manifest update."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    output_dir = run_dir / "data" / "raw" / "watergap22d"
    raw_filename = "watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4"

    # Mock PanDataSet to create a fake NC4 in the cache dir
    cache_dir = run_dir / "pangaea_cache"
    cache_dir.mkdir()
    cached_file = cache_dir / raw_filename
    _make_watergap_nc(cached_file, n_times=12)

    mock_ds = MagicMock()
    # Mock .data as a DataFrame with the expected filename at index 30
    import pandas as pd_mock
    mock_data = pd_mock.DataFrame({"File name": {30: raw_filename}})
    mock_ds.data = mock_data
    mock_ds.download.return_value = [cached_file]

    with patch("nhf_spatial_targets.fetch.pangaea.PanDataSet", return_value=mock_ds):
        result = fetch_watergap22d(run_dir=run_dir, period="2000/2009")

    # Verify raw file was moved to output dir
    assert (output_dir / raw_filename).exists()
    # Verify CF-corrected file was created
    assert (output_dir / "watergap22d_qrdif_cf.nc").exists()
    # Verify manifest updated
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "watergap22d" in manifest["sources"]
    assert manifest["sources"]["watergap22d"]["license"] == "CC BY-NC 4.0"
    assert result["source_key"] == "watergap22d"


def test_preserves_original_file(run_dir: Path):
    """Both original and CF-corrected files must exist after fetch."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    output_dir = run_dir / "data" / "raw" / "watergap22d"
    raw_filename = "watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4"

    cache_dir = run_dir / "pangaea_cache"
    cache_dir.mkdir()
    cached_file = cache_dir / raw_filename
    _make_watergap_nc(cached_file, n_times=12)

    mock_ds = MagicMock()
    import pandas as pd_mock
    mock_data = pd_mock.DataFrame({"File name": {30: raw_filename}})
    mock_ds.data = mock_data
    mock_ds.download.return_value = [cached_file]

    with patch("nhf_spatial_targets.fetch.pangaea.PanDataSet", return_value=mock_ds):
        fetch_watergap22d(run_dir=run_dir, period="2000/2009")

    assert (output_dir / raw_filename).exists(), "Original file must be preserved"
    assert (output_dir / "watergap22d_qrdif_cf.nc").exists(), "CF-corrected file must exist"


def test_download_failure_raises(run_dir: Path):
    """RuntimeError when pangaeapy download fails."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    raw_filename = "watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4"
    mock_ds = MagicMock()
    # Set up data so file-index validation passes, then download fails
    import pandas as pd_mock
    mock_data = pd_mock.DataFrame({"File name": {30: raw_filename}})
    mock_ds.data = mock_data
    mock_ds.download.side_effect = Exception("PANGAEA unavailable")

    with patch("nhf_spatial_targets.fetch.pangaea.PanDataSet", return_value=mock_ds):
        with pytest.raises(RuntimeError, match="PANGAEA"):
            fetch_watergap22d(run_dir=run_dir, period="2000/2009")


def test_wrong_file_index_raises(run_dir: Path):
    """RuntimeError when file at expected index doesn't match expected filename."""
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    mock_ds = MagicMock()
    import pandas as pd_mock
    mock_data = pd_mock.DataFrame({"File name": {30: "wrong_filename.nc4"}})
    mock_ds.data = mock_data

    with patch("nhf_spatial_targets.fetch.pangaea.PanDataSet", return_value=mock_ds):
        with pytest.raises(RuntimeError, match="file index"):
            fetch_watergap22d(run_dir=run_dir, period="2000/2009")
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pixi run -e dev test tests/test_pangaea.py -v -k "not cf_fixup"`
Expected: All new tests FAIL (fetch_watergap22d not defined yet)

### Task 8: Implement fetch_watergap22d

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/pangaea.py`

- [ ] **Step 1: Add fetch_watergap22d and _update_manifest to pangaea.py**

Add imports and functions after the `_cf_fixup` function in `pangaea.py`:

```python
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone

from pangaeapy import PanDataSet

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import parse_period

_PANGAEA_DATASET_ID = 918447
_PANGAEA_FILE_INDEX = 30  # row index for qrdif file in PanDataSet.data
_RAW_FILENAME = "watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4"
_CF_FILENAME = "watergap22d_qrdif_cf.nc"


def fetch_watergap22d(run_dir: Path, period: str) -> dict:
    """Download WaterGAP 2.2d diffuse groundwater recharge from PANGAEA.

    Downloads the single NC4 file via pangaeapy, applies CF compliance
    fixes, and updates manifest.json. Skips download if the CF-corrected
    file already exists.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory.
    period : str
        Temporal range as ``"YYYY/YYYY"``. Used for provenance only —
        the downloaded file covers the full 1901–2016 period.

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    meta = _catalog.source(_SOURCE_KEY)
    parse_period(period)  # validate format

    fabric_path = run_dir / "fabric.json"
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"fabric.json not found in {run_dir}. "
            f"Run 'nhf-targets init' to create a run workspace first."
        )
    try:
        fabric = json.loads(fabric_path.read_text())
        bbox = fabric["bbox_buffered"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(
            f"fabric.json in {run_dir} is malformed or missing required "
            f"fields (bbox_buffered.{{minx,miny,maxx,maxy}}). "
            f"Re-run 'nhf-targets init' to regenerate it."
        ) from exc

    output_dir = run_dir / "data" / "raw" / _SOURCE_KEY
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / _RAW_FILENAME
    cf_path = output_dir / _CF_FILENAME
    now_utc = datetime.now(timezone.utc).isoformat()

    if cf_path.exists():
        logger.info("CF-corrected file already exists, skipping download: %s", cf_path)
    else:
        # Download from PANGAEA
        if not raw_path.exists():
            logger.info("Downloading WaterGAP 2.2d from PANGAEA (dataset %d)...", _PANGAEA_DATASET_ID)
            try:
                cache_dir = output_dir / ".pangaea_cache"
                cache_dir.mkdir(exist_ok=True)
                pan_ds = PanDataSet(
                    _PANGAEA_DATASET_ID,
                    cachedir=str(cache_dir),
                )

                # Validate file index before downloading
                actual_name = pan_ds.data.loc[_PANGAEA_FILE_INDEX, "File name"]
                if actual_name != _RAW_FILENAME:
                    raise RuntimeError(
                        f"PANGAEA file index {_PANGAEA_FILE_INDEX} contains "
                        f"'{actual_name}', expected '{_RAW_FILENAME}'. "
                        f"The dataset listing may have changed."
                    )

                downloaded = pan_ds.download(indices=[_PANGAEA_FILE_INDEX])
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download WaterGAP 2.2d from PANGAEA "
                    f"(dataset {_PANGAEA_DATASET_ID}): {exc}. "
                    f"Check network connectivity and PANGAEA availability."
                ) from exc

            if not downloaded:
                raise RuntimeError(
                    f"pangaeapy returned no files for dataset {_PANGAEA_DATASET_ID}, "
                    f"index {_PANGAEA_FILE_INDEX}."
                )

            cached_path = Path(downloaded[0])
            shutil.move(str(cached_path), str(raw_path))
            logger.info("Moved %s -> %s", cached_path, raw_path)

        # CF compliance fix-up
        logger.info("Applying CF compliance fixes...")
        _cf_fixup(raw_path, cf_path)

    # Build provenance
    file_info = {
        "path": str(cf_path.relative_to(run_dir)),
        "raw_path": str(raw_path.relative_to(run_dir)) if raw_path.exists() else None,
        "size_bytes": cf_path.stat().st_size,
        "downloaded_utc": now_utc,
    }

    _update_manifest(run_dir, period, bbox, meta, file_info)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "doi": meta["access"]["doi"],
        "license": "CC BY-NC 4.0",
        "variables": meta["variables"],
        "period": period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "file": file_info,
        "cf_corrected_file": str(cf_path.relative_to(run_dir)),
    }


def _update_manifest(
    run_dir: Path,
    period: str,
    bbox: dict,
    meta: dict,
    file_info: dict,
) -> None:
    """Merge WaterGAP 2.2d provenance into manifest.json."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"sources": {}, "steps": []}

    if "sources" not in manifest:
        manifest["sources"] = {}

    entry = manifest["sources"].get(_SOURCE_KEY, {})
    entry.update(
        {
            "source_key": _SOURCE_KEY,
            "access_url": meta["access"]["url"],
            "doi": meta["access"]["doi"],
            "license": "CC BY-NC 4.0",
            "period": period,
            "bbox": bbox,
            "variables": [v["name"] for v in meta["variables"]],
            "file": file_info,
        }
    )
    manifest["sources"][_SOURCE_KEY] = entry

    tmp_fd, tmp_path = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp_path).replace(manifest_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    logger.info("Updated manifest.json with WaterGAP 2.2d provenance")
```

- [ ] **Step 2: Run all pangaea tests**

Run: `pixi run -e dev test tests/test_pangaea.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run fmt, lint, full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add src/nhf_spatial_targets/fetch/pangaea.py tests/test_pangaea.py
git commit -m "feat: implement WaterGAP 2.2d fetch with CF fix-up and tests"
```

### Task 8b: Add integration test

**Files:**
- Modify: `tests/test_pangaea.py`

- [ ] **Step 1: Add integration test**

Append to `tests/test_pangaea.py`:

```python
@pytest.mark.integration
def test_fetch_watergap22d_real_download(tmp_path: Path):
    """Integration: fetch from real PANGAEA and verify CF-compliant output.

    Requires network access. Run with: pixi run -e dev test-integration
    """
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    # Create minimal run workspace
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "watergap22d").mkdir(parents=True)
    fabric = {
        "bbox_buffered": {
            "minx": -125.1, "miny": 23.9,
            "maxx": -65.9, "maxy": 50.1,
        }
    }
    (rd / "fabric.json").write_text(json.dumps(fabric))

    result = fetch_watergap22d(run_dir=rd, period="2000/2009")

    assert result["source_key"] == "watergap22d"

    # Verify CF-corrected file
    cf_path = rd / "data" / "raw" / "watergap22d" / "watergap22d_qrdif_cf.nc"
    assert cf_path.exists()

    ds = xr.open_dataset(cf_path)
    assert "qrdif" in ds.data_vars
    assert ds.time.dtype == np.dtype("datetime64[ns]")
    assert ds.attrs.get("Conventions") == "CF-1.6"
    assert "crs" in ds.data_vars
    assert ds["qrdif"].attrs.get("grid_mapping") == "crs"
    ds.close()

    # Verify original file preserved
    raw_path = rd / "data" / "raw" / "watergap22d" / "watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4"
    assert raw_path.exists()
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_pangaea.py
git commit -m "test: add integration test for WaterGAP 2.2d real download"
```

---

## Chunk 4: CLI Wiring and Final Integration

### Task 9: Add CLI command for watergap22d fetch

**Files:**
- Modify: `src/nhf_spatial_targets/cli.py` (after the mod10c1 command, ~line 554)

- [ ] **Step 1: Add the fetch command**

Add after the `fetch_mod10c1_cmd` function in `cli.py`:

```python
@fetch_app.command(name="watergap22d")
def fetch_watergap22d_cmd(
    run_dir: Annotated[
        Path,
        Parameter(
            name=["--run-dir", "-r"],
            help="Run workspace created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
):
    """Download WaterGAP 2.2d groundwater recharge from PANGAEA.

    Downloads the diffuse groundwater recharge (qrdif) NC4 file,
    applies CF compliance fixes, and prints the provenance record.
    """
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching WaterGAP 2.2d for period {period}...[/bold]")

    try:
        result = fetch_watergap22d(run_dir=run_dir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during WaterGAP 2.2d fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print(
        f"[green]Downloaded WaterGAP 2.2d to "
        f"{run_dir / 'data' / 'raw' / 'watergap22d'}[/green]"
    )
    console.print(json_mod.dumps(result, indent=2))
```

- [ ] **Step 2: Add CLI test**

Add to `tests/test_pangaea.py`:

```python
def test_cli_watergap22d_nonexistent_run_dir(tmp_path: Path):
    """CLI exits with error for nonexistent run directory."""
    from nhf_spatial_targets.cli import app

    with pytest.raises(SystemExit) as exc_info:
        app(["fetch", "watergap22d", "--run-dir", str(tmp_path / "nope"), "--period", "2000/2009"], exit_on_error=False)
    assert exc_info.value.code == 2


def test_cli_watergap22d_calls_fetch(run_dir: Path):
    """CLI wires through to fetch_watergap22d correctly."""
    from nhf_spatial_targets.cli import app

    mock_result = {"source_key": "watergap22d", "files": [], "file": {}}
    with patch("nhf_spatial_targets.fetch.pangaea.fetch_watergap22d", return_value=mock_result) as mock_fetch:
        app(["fetch", "watergap22d", "--run-dir", str(run_dir), "--period", "2000/2009"])
        mock_fetch.assert_called_once_with(run_dir=run_dir, period="2000/2009")
```

- [ ] **Step 3: Run all tests**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add src/nhf_spatial_targets/cli.py tests/test_pangaea.py
git commit -m "feat: add nhf-targets fetch watergap22d CLI command"
```

### Task 10: Add pixi task for fetch-watergap22d

**Files:**
- Modify: `pixi.toml`

- [ ] **Step 1: Check existing fetch tasks in pixi.toml for pattern**

Look for entries like `fetch-merra2`, `fetch-ncep-ncar`, etc. in `pixi.toml` and add:

```toml
fetch-watergap22d = { cmd = "nhf-targets fetch watergap22d", description = "Download WaterGAP 2.2d groundwater recharge from PANGAEA" }
```

- [ ] **Step 2: Run full quality gate**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add pixi.toml
git commit -m "chore: add fetch-watergap22d pixi task"
```

### Task 11: Update README fetch pipeline table

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add WaterGAP row to the fetch pipeline table**

In the "Fetch & Consolidation Pipeline" table, add after the SSEBop row:

```
| WaterGAP 2.2d | `fetch-watergap22d` | Single NC4 via pangaeapy | `watergap22d_qrdif_cf.nc` (CF-corrected) |
```

Also add `fetch-watergap22d` to the Quick Start section:

```
pixi run fetch-watergap22d -- --run-dir /data/nhf-runs/2026-03-12_gfv11_v0.1.0 --period 1901/2016
```

And add `pangaea.py` to the repo structure tree.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add WaterGAP 2.2d to README fetch pipeline table"
```

### Task 12: Final integration verification

- [ ] **Step 1: Run full quality gate**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All pass, no regressions

- [ ] **Step 2: Verify catalog loads correctly**

Run: `pixi run catalog-sources | grep -A 5 watergap22d`
Expected: Shows updated watergap22d entry with `pangaea` access type

- [ ] **Step 3: Review git log**

Run: `git log --oneline -10`
Expected: Clean commit history for the feature
