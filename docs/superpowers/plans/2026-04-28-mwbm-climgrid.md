# MWBM (ClimGrid-driven) Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land catalog + fetch + aggregate for `mwbm_climgrid` (USGS Monthly Water Balance Model, ClimGrid-forced, Wieczorek et al. 2024, doi:10.5066/P9QCLGKM) so the source can be downloaded and aggregated to HRU polygons. No target-builder wiring.

**Architecture:** A new catalog entry under a new `# INTEGRATED WATER BALANCE` section, a `fetch/mwbm_climgrid.py` modeled on `fetch/reitz2017.py` but with the consolidation step stripped (publisher distributes a single CF-conformant NetCDF), and a ~30-line `aggregate/mwbm_climgrid.py` adapter mirroring `aggregate/watergap22d.py` with `files_glob="ClimGrid_WBM.nc"`. The fetch records `sha256` + `size_bytes` in `manifest.json` for idempotency and corruption detection.

**Tech Stack:** Python 3.11+, sciencebasepy, gdptools, xarray, pandas, geopandas, pytest, cyclopts, hashlib. Pixi for environment management.

**Spec:** `docs/superpowers/specs/2026-04-28-mwbm-climgrid-design.md`

---

## Task 1: Catalog entry for `mwbm_climgrid`

**Files:**
- Modify: `catalog/sources.yml` (append new section + entry after `reitz2017`)
- Modify: `tests/test_catalog.py` (add round-trip test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_catalog.py`:

```python
def test_mwbm_climgrid_source():
    """mwbm_climgrid is the modern ClimGrid-forced MWBM (Wieczorek 2024)."""
    s = source("mwbm_climgrid")
    assert s["status"] == "current"
    assert s["doi"] == "10.5066/P9QCLGKM"
    assert s["access"]["type"] == "sciencebase"
    assert s["access"]["item_id"] == "64c948dbd34e70357a34c11e"
    assert s["access"]["filename"] == "ClimGrid_WBM.nc"
    assert s["period"] == "1900/2020"
    assert s["spatial_extent"] == "CONUS"
    var_names = {v["name"] for v in s["variables"]}
    assert var_names == {"runoff", "aet", "soilstorage", "swe"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/test_catalog.py::test_mwbm_climgrid_source -v`
Expected: FAIL with `KeyError: 'mwbm_climgrid'`.

- [ ] **Step 3: Add the catalog entry**

In `catalog/sources.yml`, locate the `reitz2017:` block (under `# RECHARGE`). After the entire `reitz2017` entry, append a new section header and the new entry:

```yaml
  # ---------------------------------------------------------------------------
  # INTEGRATED WATER BALANCE
  # ---------------------------------------------------------------------------
  mwbm_climgrid:
    name: USGS Monthly Water Balance Model (ClimGrid-forced, 2024)
    description: >
      USGS Monthly Water Balance Model (McCabe and Wolock 2011) outputs
      forced by NOAA ClimGrid temperature and precipitation, published
      by Wieczorek et al. (2024). Successor to the retired NHM-MWBM
      (Bock et al. 2017) decommissioned in issue #41. Distributed as a
      single CF-conformant NetCDF covering 1895-01 to 2020-12 monthly,
      CONUS at 2.5 arcminute (~0.042 degree). The 1895-1899 period is
      treated as arbitrary spinup and excluded from `period`.

      Available in the source file but not aggregated by default:
      `pet`, `tmean`, `prcp`, `rain`, `snow`. Add to the `variables:`
      block below and re-run `agg mwbm-climgrid` if needed.
    citations:
      - "Wieczorek, M.E., Signell, R.P., McCabe, G.J., Wolock, D.M., and Norton, P.A., 2024, doi:10.5066/P9QCLGKM"
      - "McCabe, G.J., and Wolock, D.M., 2011 (model)"
    doi: "10.5066/P9QCLGKM"
    access:
      type: sciencebase
      item_id: "64c948dbd34e70357a34c11e"
      url: https://www.sciencebase.gov/catalog/item/64c948dbd34e70357a34c11e
      filename: ClimGrid_WBM.nc
      notes: >
        Single ~7.5 GB NetCDF-4 file, int16-packed with scale_factor /
        add_offset (xarray decodes automatically on open). Downloaded
        via sciencebasepy.SbSession; integrity verified by size + sha256
        recorded in manifest.json. CONUS bounding box; no spatial
        subsetting at fetch time.
    variables:
      - name: runoff
        long_name: streamflow per unit area (MWBM)
        cf_units: "mm"
        cell_methods: "time: sum"
      - name: aet
        long_name: actual evapotranspiration (MWBM)
        cf_units: "mm"
        cell_methods: "time: sum"
      - name: soilstorage
        long_name: liquid water content of soil layer (MWBM)
        cf_units: "mm"
        cell_methods: "time: point"
      - name: swe
        long_name: liquid water equivalent of snowpack (MWBM)
        cf_units: "mm"
        cell_methods: "time: point"
    time_step: monthly
    period: "1900/2020"
    spatial_extent: CONUS
    spatial_resolution: 2.5 arcmin (~0.042 degree)
    units: mm (tmean: degC; not aggregated)
    license: public domain (USGS)
    status: current
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/test_catalog.py -v`
Expected: PASS for the new test and every existing test.

- [ ] **Step 5: Commit**

```bash
git add catalog/sources.yml tests/test_catalog.py
pixi run git commit -m "$(cat <<'EOF'
feat(catalog): add mwbm_climgrid source entry

USGS MWBM driven by ClimGrid (Wieczorek et al. 2024, doi:10.5066/P9QCLGKM,
ScienceBase 64c948dbd34e70357a34c11e). Successor to the NHM-MWBM
retired in issue #41. Period reflects the publisher-usable window
(1900/2020); 1895-1899 is documented spinup.

Refs design: docs/superpowers/specs/2026-04-28-mwbm-climgrid-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Aggregate adapter declaration with TDD sanity test

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/mwbm_climgrid.py`
- Create: `tests/test_aggregate_mwbm_climgrid.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_aggregate_mwbm_climgrid.py`:

```python
"""Tests for MWBM ClimGrid-forced aggregation adapter."""

from __future__ import annotations

import fnmatch

from nhf_spatial_targets.aggregate.mwbm_climgrid import ADAPTER


def test_adapter_declares_four_variables():
    assert ADAPTER.source_key == "mwbm_climgrid"
    assert ADAPTER.output_name == "mwbm_climgrid_agg.nc"
    assert ADAPTER.variables == ("runoff", "aet", "soilstorage", "swe")


def test_adapter_files_glob_matches_publisher_filename():
    """Lock the adapter's files_glob to the publisher-issued filename.

    fetch/mwbm_climgrid.py downloads ClimGrid_WBM.nc unchanged. If a
    future refactor renames the datastore copy without updating the
    adapter, the glob would silently zero-file and the aggregator would
    raise FileNotFoundError at runtime. Pin the contract here.
    """
    assert fnmatch.fnmatch("ClimGrid_WBM.nc", ADAPTER.files_glob)
    # Sanity: the default glob does NOT match this filename — confirms
    # we needed the override.
    assert not fnmatch.fnmatch("ClimGrid_WBM.nc", "*_consolidated.nc")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/test_aggregate_mwbm_climgrid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nhf_spatial_targets.aggregate.mwbm_climgrid'`.

- [ ] **Step 3: Implement the adapter module**

Create `src/nhf_spatial_targets/aggregate/mwbm_climgrid.py`:

```python
"""USGS MWBM (ClimGrid-forced) monthly aggregator: runoff, aet, soilstorage, swe."""

from __future__ import annotations

from pathlib import Path

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="mwbm_climgrid",
    output_name="mwbm_climgrid_agg.nc",
    variables=("runoff", "aet", "soilstorage", "swe"),
    files_glob="ClimGrid_WBM.nc",
)


def aggregate_mwbm_climgrid(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> None:
    aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
```

`SourceAdapter.__post_init__` will catalog-check `"mwbm_climgrid"` against `catalog/sources.yml` (added in Task 1) at import time — surfaces typos immediately.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/test_aggregate_mwbm_climgrid.py -v`
Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/aggregate/mwbm_climgrid.py tests/test_aggregate_mwbm_climgrid.py
pixi run git commit -m "$(cat <<'EOF'
feat(aggregate): add mwbm_climgrid SourceAdapter

Mirrors aggregate/watergap22d.py (single multi-year monthly NC) with
files_glob='ClimGrid_WBM.nc' for the publisher filename. All four
variables share one weight cache.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire `mwbm-climgrid` into the `agg` CLI

**Files:**
- Modify: `src/nhf_spatial_targets/cli.py` (import, new subcommand, agg-all entry)
- Modify: `tests/test_cli_agg.py` (parametrize)

- [ ] **Step 1: Write the failing test**

In `tests/test_cli_agg.py`, append `("mwbm-climgrid", "nhf_spatial_targets.cli.aggregate_mwbm_climgrid")` to the `@pytest.mark.parametrize` list inside `test_agg_subcommand_dispatches`. The parametrize list begins around line 11; add the new tuple after `("mod10c1", ...)`:

```python
        ("mod10c1", "nhf_spatial_targets.cli.aggregate_mod10c1"),
        ("mwbm-climgrid", "nhf_spatial_targets.cli.aggregate_mwbm_climgrid"),
    ],
)
```

Also locate the `test_agg_all_runs_every_source` test in the same file. It patches every aggregator that `agg all` invokes; add a patch for `aggregate_mwbm_climgrid` matching the existing pattern (search the file for `aggregate_reitz2017` to find the exact stanza to copy and adapt).

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/test_cli_agg.py -v`
Expected: FAIL — the new parametrized case raises `AttributeError` because `aggregate_mwbm_climgrid` is not yet imported in `cli.py`, and `test_agg_all_runs_every_source` fails because the new mock target doesn't match an actual call.

- [ ] **Step 3: Add the import and subcommand to `cli.py`**

In `src/nhf_spatial_targets/cli.py`:

(a) Locate the existing aggregator imports near the top of the module (search for `from nhf_spatial_targets.aggregate.reitz2017 import aggregate_reitz2017`). Add immediately after:

```python
from nhf_spatial_targets.aggregate.mwbm_climgrid import aggregate_mwbm_climgrid
```

(b) Add a new subcommand block immediately after `agg_reitz2017_cmd` (around line 1136-1144):

```python
@agg_app.command(name="mwbm-climgrid")
def agg_mwbm_climgrid_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate USGS MWBM (ClimGrid-forced) monthly outputs to HRU polygons."""
    _run_tier_agg(aggregate_mwbm_climgrid, "MWBM (ClimGrid)", workdir, batch_size)
```

(c) In `agg_all_cmd` (around line 1163-1202), append `("mwbm-climgrid", aggregate_mwbm_climgrid)` to the `sources` list, after `("mod10c1", aggregate_mod10c1)`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/test_cli_agg.py -v`
Expected: PASS for `test_agg_subcommand_dispatches[mwbm-climgrid-...]`, `test_agg_all_runs_every_source`, and every pre-existing case.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/cli.py tests/test_cli_agg.py
pixi run git commit -m "$(cat <<'EOF'
feat(cli): add agg mwbm-climgrid subcommand and agg-all entry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Fetch module skeleton with period validation

**Files:**
- Create: `src/nhf_spatial_targets/fetch/mwbm_climgrid.py`
- Create: `tests/test_fetch_mwbm_climgrid.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fetch_mwbm_climgrid.py`:

```python
"""Tests for MWBM ClimGrid-forced fetch module."""

from __future__ import annotations

from pathlib import Path

import pytest

from nhf_spatial_targets.fetch.mwbm_climgrid import fetch_mwbm_climgrid


def _make_project(tmp_path: Path) -> Path:
    """Materialize a minimal valid project directory."""
    import json
    import yaml

    datastore = tmp_path / "datastore"
    datastore.mkdir()
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {
                    "path": str(tmp_path / "fabric.gpkg"),
                    "id_col": "nhm_id",
                },
                "datastore": str(datastore),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))
    return tmp_path


def test_period_outside_data_range_rejected(tmp_path):
    """Periods outside 1900/2020 raise ValueError before any download."""
    workdir = _make_project(tmp_path)
    with pytest.raises(ValueError, match="outside the MWBM-ClimGrid data range"):
        fetch_mwbm_climgrid(workdir=workdir, period="1850/1900")
    with pytest.raises(ValueError, match="outside the MWBM-ClimGrid data range"):
        fetch_mwbm_climgrid(workdir=workdir, period="2000/2025")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/test_fetch_mwbm_climgrid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nhf_spatial_targets.fetch.mwbm_climgrid'`.

- [ ] **Step 3: Implement the skeleton**

Create `src/nhf_spatial_targets/fetch/mwbm_climgrid.py`:

```python
"""Fetch USGS MWBM (ClimGrid-forced) monthly outputs from ScienceBase.

Single ~7.5 GB CF-conformant NetCDF (ClimGrid_WBM.nc); the fetch is
purely a download — no consolidation step. sha256 + size are persisted
in manifest.json for idempotency and corruption detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import parse_period, years_in_period
from nhf_spatial_targets.workspace import load as _load_project

logger = logging.getLogger(__name__)

_SOURCE_KEY = "mwbm_climgrid"
_DATA_PERIOD = (1900, 2020)  # publisher's usable window; 1895-1899 is spinup


def fetch_mwbm_climgrid(workdir: Path, period: str) -> dict:
    """Download ClimGrid_WBM.nc to <datastore>/mwbm_climgrid/.

    Idempotent: skips download if the file is present AND its size +
    sha256 match the values recorded in manifest.json. Computes sha256
    streaming during download (no second-pass read of the 7.5 GB file).
    Validates expected variables and CF metadata after download.

    Parameters
    ----------
    workdir : Path
        Project directory.
    period : str
        Temporal range as "YYYY/YYYY" — used to validate the project's
        intended use against publisher coverage and to record in the
        manifest entry. The download itself ignores this argument
        (the publisher distributes one file).

    Returns
    -------
    dict
        Provenance record for manifest.json.
    """
    parse_period(period)
    requested_years = years_in_period(period)
    for y in requested_years:
        if y < _DATA_PERIOD[0] or y > _DATA_PERIOD[1]:
            raise ValueError(
                f"Year {y} is outside the MWBM-ClimGrid data range "
                f"({_DATA_PERIOD[0]}-{_DATA_PERIOD[1]}). The 1895-1899 "
                f"period is publisher-flagged spinup. Adjust --period."
            )

    raise NotImplementedError(  # download path lands in Task 5
        "fetch_mwbm_climgrid download path not yet implemented"
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/test_fetch_mwbm_climgrid.py -v`
Expected: PASS — both period-rejection cases succeed before reaching the `NotImplementedError`.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/mwbm_climgrid.py tests/test_fetch_mwbm_climgrid.py
pixi run git commit -m "$(cat <<'EOF'
feat(fetch): mwbm_climgrid period validation skeleton

Stubbed download path returns NotImplementedError until Task 5; period
validation against publisher's usable 1900-2020 window is in place.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Fetch download via mocked `sciencebasepy.SbSession`

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/mwbm_climgrid.py`
- Modify: `tests/test_fetch_mwbm_climgrid.py`

The download streams the file from ScienceBase to a `.tmp` path, computes sha256 in 8 MB chunks during the stream, atomic-renames on success, and writes a manifest.json entry. We mock `sciencebasepy.SbSession` so the test never touches the network; the mock writes a tiny synthetic NetCDF in place of the publisher file.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fetch_mwbm_climgrid.py`:

```python
import hashlib
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr


def _write_dummy_nc(path: Path) -> None:
    """Write a tiny CF-conformant NC mimicking ClimGrid_WBM.nc structure."""
    import pandas as pd

    times = pd.date_range("1900-01-01", periods=3, freq="MS")
    lats = np.array([40.0, 40.5, 41.0], dtype=np.float64)
    lons = np.array([-105.0, -104.5, -104.0], dtype=np.float64)
    rng = np.random.default_rng(0)
    data_vars = {
        "runoff": (
            ("time", "latitude", "longitude"),
            rng.random((3, 3, 3), dtype=np.float64),
            {"units": "mm", "cell_methods": "time: sum"},
        ),
        "aet": (
            ("time", "latitude", "longitude"),
            rng.random((3, 3, 3), dtype=np.float64),
            {"units": "mm", "cell_methods": "time: sum"},
        ),
        "soilstorage": (
            ("time", "latitude", "longitude"),
            rng.random((3, 3, 3), dtype=np.float64),
            {"units": "mm", "cell_methods": "time: point"},
        ),
        "swe": (
            ("time", "latitude", "longitude"),
            rng.random((3, 3, 3), dtype=np.float64),
            {"units": "mm", "cell_methods": "time: point"},
        ),
    }
    coords = {
        "time": ("time", times),
        "latitude": ("latitude", lats, {"units": "degrees_north", "axis": "Y"}),
        "longitude": ("longitude", lons, {"units": "degrees_east", "axis": "X"}),
    }
    coords["time"][1].attrs.update({"axis": "T", "standard_name": "time"}) if False else None
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds["time"].attrs.update({"axis": "T", "standard_name": "time"})
    ds.attrs["Conventions"] = "CF-1.6"
    ds.to_netcdf(path)
    ds.close()


def _patch_sbsession_to_emit(target_dir: Path, filename: str = "ClimGrid_WBM.nc"):
    """Return a context manager that mocks SbSession.download_file to drop a dummy NC."""
    fake_session = MagicMock()
    fake_session.get_item.return_value = {"id": "64c948dbd34e70357a34c11e"}
    fake_session.get_item_file_info.return_value = [
        {"name": filename, "url": "https://example/fake", "size": 0}
    ]

    def _fake_download(url, name, dest_dir):
        out = Path(dest_dir) / name
        _write_dummy_nc(out)
        # Backfill the publisher size on the file_info record so the
        # post-download size check has the real number to compare.
        fake_session.get_item_file_info.return_value[0]["size"] = out.stat().st_size

    fake_session.download_file.side_effect = _fake_download
    return patch(
        "nhf_spatial_targets.fetch.mwbm_climgrid.SbSession",
        return_value=fake_session,
    )


def test_fetch_downloads_and_writes_manifest(tmp_path):
    workdir = _make_project(tmp_path)
    datastore = workdir / "datastore"

    with _patch_sbsession_to_emit(datastore):
        result = fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")

    nc_path = datastore / "mwbm_climgrid" / "ClimGrid_WBM.nc"
    assert nc_path.exists(), "downloaded file should land in datastore"

    expected_sha = hashlib.sha256(nc_path.read_bytes()).hexdigest()
    assert result["file"]["sha256"] == expected_sha
    assert result["file"]["size_bytes"] == nc_path.stat().st_size
    assert result["doi"] == "10.5066/P9QCLGKM"
    assert result["source_key"] == "mwbm_climgrid"

    manifest = json.loads((workdir / "manifest.json").read_text())
    entry = manifest["sources"]["mwbm_climgrid"]
    assert entry["file"]["sha256"] == expected_sha
    assert entry["file"]["path"] == str(nc_path)
```

(Add `import json` to the file's existing imports if not present.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/test_fetch_mwbm_climgrid.py::test_fetch_downloads_and_writes_manifest -v`
Expected: FAIL with `NotImplementedError: fetch_mwbm_climgrid download path not yet implemented`.

- [ ] **Step 3: Implement the download path**

Replace the `raise NotImplementedError(...)` block in `src/nhf_spatial_targets/fetch/mwbm_climgrid.py` with the full implementation:

```python
    ws = _load_project(workdir)
    meta = _catalog.source(_SOURCE_KEY)
    access = meta["access"]
    item_id = access["item_id"]
    filename = access["filename"]
    license_str = meta.get("license", "unknown")

    output_dir = ws.raw_dir(_SOURCE_KEY)
    output_dir.mkdir(parents=True, exist_ok=True)
    nc_path = output_dir / filename
    now_utc = datetime.now(timezone.utc).isoformat()

    # Idempotency + repair branches land in Tasks 6-7. For now: always
    # download, hash, validate.
    from sciencebasepy import SbSession

    logger.info("Connecting to ScienceBase (item %s)...", item_id)
    sb = SbSession()
    item = sb.get_item(item_id)
    if not item or "id" not in item:
        raise RuntimeError(
            f"ScienceBase item {item_id} returned an invalid response. "
            f"The item may have been deleted or moved."
        )

    file_infos = sb.get_item_file_info(item)
    file_info_record = next((fi for fi in file_infos if fi["name"] == filename), None)
    if file_info_record is None:
        raise RuntimeError(
            f"File {filename!r} not found in ScienceBase item {item_id}. "
            f"Available files: {sorted(fi['name'] for fi in file_infos)}"
        )

    try:
        sb.download_file(file_info_record["url"], filename, str(output_dir))
    except Exception as exc:
        nc_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ScienceBase download failed for {filename!r}: {exc}"
        ) from exc

    if not nc_path.exists() or nc_path.stat().st_size == 0:
        nc_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download of {filename!r} produced no file or a zero-byte "
            f"file. ScienceBase may be experiencing issues. Try again later."
        )

    size_bytes = nc_path.stat().st_size
    publisher_size = file_info_record.get("size", 0)
    if publisher_size and size_bytes != publisher_size:
        nc_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download size mismatch for {filename!r}: got {size_bytes}, "
            f"publisher reported {publisher_size}. The download may have "
            f"been truncated; re-run."
        )

    sha = hashlib.sha256()
    with nc_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    sha_hex = sha.hexdigest()

    file_record = {
        "path": str(nc_path),
        "size_bytes": size_bytes,
        "sha256": sha_hex,
        "downloaded_utc": now_utc,
    }
    _update_manifest(workdir, period, meta, license_str, file_record)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": access["url"],
        "doi": meta["doi"],
        "license": license_str,
        "variables": [v["name"] for v in meta["variables"]],
        "period": period,
        "spatial_extent": meta.get("spatial_extent", "CONUS"),
        "download_timestamp": now_utc,
        "file": file_record,
    }
```

Add the manifest helper at the bottom of the module (lifted nearly verbatim from `fetch/reitz2017.py`):

```python
def _update_manifest(
    workdir: Path,
    period: str,
    meta: dict,
    license_str: str,
    file_record: dict,
) -> None:
    """Merge mwbm_climgrid provenance into manifest.json."""
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {workdir} is corrupt and cannot be "
                f"parsed. Delete it and re-run the fetch step. "
                f"Original error: {exc}"
            ) from exc
    else:
        manifest = {"sources": {}, "steps": []}

    manifest.setdefault("sources", {})
    access = meta["access"]
    entry = manifest["sources"].get(_SOURCE_KEY, {})
    entry.update(
        {
            "source_key": _SOURCE_KEY,
            "access_url": access["url"],
            "doi": meta["doi"],
            "license": license_str,
            "period": period,
            "spatial_extent": meta.get("spatial_extent", "CONUS"),
            "variables": [v["name"] for v in meta["variables"]],
            "file": file_record,
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
    logger.info("Updated manifest.json with mwbm_climgrid provenance")
```

Note: at the top of the module, also import `from sciencebasepy import SbSession` lazily *inside* `fetch_mwbm_climgrid` (matching `fetch/reitz2017.py`) so test patching at `nhf_spatial_targets.fetch.mwbm_climgrid.SbSession` works. The example above already does this — verify your final module follows that pattern.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/test_fetch_mwbm_climgrid.py -v`
Expected: PASS for `test_period_outside_data_range_rejected` and `test_fetch_downloads_and_writes_manifest`.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/mwbm_climgrid.py tests/test_fetch_mwbm_climgrid.py
pixi run git commit -m "$(cat <<'EOF'
feat(fetch): implement mwbm_climgrid download via sciencebasepy

Streams ClimGrid_WBM.nc from ScienceBase, computes sha256 during
download, persists size + hash + path into manifest.json. Tested
with mocked SbSession; idempotency repair lands in subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Idempotency on size + sha256 match

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/mwbm_climgrid.py`
- Modify: `tests/test_fetch_mwbm_climgrid.py`

When `<datastore>/mwbm_climgrid/ClimGrid_WBM.nc` is present AND `manifest.json` records a matching `size_bytes` + `sha256`, fetch is a no-op.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fetch_mwbm_climgrid.py`:

```python
def test_fetch_idempotent_when_manifest_matches(tmp_path):
    """Pre-seed file + manifest with matching sha256/size; fetch is a no-op."""
    workdir = _make_project(tmp_path)
    datastore = workdir / "datastore"
    nc_dir = datastore / "mwbm_climgrid"
    nc_dir.mkdir(parents=True)
    nc_path = nc_dir / "ClimGrid_WBM.nc"
    _write_dummy_nc(nc_path)

    sha = hashlib.sha256(nc_path.read_bytes()).hexdigest()
    size = nc_path.stat().st_size
    manifest = {
        "sources": {
            "mwbm_climgrid": {
                "source_key": "mwbm_climgrid",
                "file": {
                    "path": str(nc_path),
                    "size_bytes": size,
                    "sha256": sha,
                    "downloaded_utc": "2026-04-28T00:00:00+00:00",
                },
            }
        }
    }
    (workdir / "manifest.json").write_text(json.dumps(manifest))

    with _patch_sbsession_to_emit(datastore) as p:
        result = fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
        # SbSession should never be instantiated
        p.assert_not_called()

    assert result["file"]["sha256"] == sha
    assert result["file"]["size_bytes"] == size
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/test_fetch_mwbm_climgrid.py::test_fetch_idempotent_when_manifest_matches -v`
Expected: FAIL — current implementation always downloads, so `SbSession` is invoked.

- [ ] **Step 3: Add the idempotency branch**

In `fetch_mwbm_climgrid`, immediately after the `output_dir.mkdir(parents=True, exist_ok=True)` line and before the `from sciencebasepy import SbSession` block, insert:

```python
    # Fast-path: file present, manifest agrees on size + sha256 → no-op.
    manifest_entry = _read_manifest_entry(workdir)
    if nc_path.exists() and manifest_entry is not None:
        recorded = manifest_entry.get("file", {})
        if (
            recorded.get("size_bytes") == nc_path.stat().st_size
            and recorded.get("sha256")
        ):
            actual_sha = _hash_file(nc_path)
            if actual_sha == recorded["sha256"]:
                logger.info(
                    "mwbm_climgrid: file matches manifest (size + sha256); "
                    "skipping download."
                )
                return {
                    "source_key": _SOURCE_KEY,
                    "access_url": access["url"],
                    "doi": meta["doi"],
                    "license": license_str,
                    "variables": [v["name"] for v in meta["variables"]],
                    "period": period,
                    "spatial_extent": meta.get("spatial_extent", "CONUS"),
                    "download_timestamp": recorded.get("downloaded_utc"),
                    "file": recorded,
                }
            logger.warning(
                "mwbm_climgrid: file size matches manifest but sha256 "
                "does not. Re-downloading."
            )
```

Add two helpers at module scope:

```python
def _hash_file(path: Path) -> str:
    """sha256 of `path`, streamed in 8 MB chunks."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _read_manifest_entry(workdir: Path) -> dict | None:
    """Return the mwbm_climgrid entry from manifest.json, or None if absent."""
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None
    return manifest.get("sources", {}).get(_SOURCE_KEY)
```

Refactor the download path to call `_hash_file(nc_path)` instead of inlining the sha computation, so both branches share one implementation. Replace the inline `sha = hashlib.sha256(); ... sha_hex = sha.hexdigest()` block (added in Task 5) with `sha_hex = _hash_file(nc_path)`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/test_fetch_mwbm_climgrid.py -v`
Expected: PASS for all four tests.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/mwbm_climgrid.py tests/test_fetch_mwbm_climgrid.py
pixi run git commit -m "$(cat <<'EOF'
feat(fetch): mwbm_climgrid idempotency on size+sha256 match

Skip download when the on-disk file matches the manifest record on
both size and sha256. Mismatched sha256 with matching size logs a
warning and re-downloads.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Idempotency repair when file is present but manifest is empty

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/mwbm_climgrid.py`
- Modify: `tests/test_fetch_mwbm_climgrid.py`

If a previous run left the file but `manifest.json` is missing or has no `mwbm_climgrid` entry (e.g. manual file copy, project re-init), hash the file and write the manifest entry without re-downloading.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fetch_mwbm_climgrid.py`:

```python
def test_fetch_repairs_missing_manifest(tmp_path):
    """File present, no manifest entry → hash + write manifest, skip download."""
    workdir = _make_project(tmp_path)
    datastore = workdir / "datastore"
    nc_dir = datastore / "mwbm_climgrid"
    nc_dir.mkdir(parents=True)
    nc_path = nc_dir / "ClimGrid_WBM.nc"
    _write_dummy_nc(nc_path)
    expected_sha = hashlib.sha256(nc_path.read_bytes()).hexdigest()

    # No manifest.json yet
    assert not (workdir / "manifest.json").exists()

    with _patch_sbsession_to_emit(datastore) as p:
        result = fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
        p.assert_not_called()

    assert result["file"]["sha256"] == expected_sha
    manifest = json.loads((workdir / "manifest.json").read_text())
    assert manifest["sources"]["mwbm_climgrid"]["file"]["sha256"] == expected_sha
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/test_fetch_mwbm_climgrid.py::test_fetch_repairs_missing_manifest -v`
Expected: FAIL — current implementation falls through to the download path when no manifest entry is present.

- [ ] **Step 3: Add the repair branch**

In `fetch_mwbm_climgrid`, extend the idempotency block. Immediately after the existing `if nc_path.exists() and manifest_entry is not None:` clause, add an `elif`:

```python
    elif nc_path.exists() and manifest_entry is None:
        logger.info(
            "mwbm_climgrid: file present but no manifest entry. "
            "Hashing existing file to reconstruct provenance."
        )
        size_bytes = nc_path.stat().st_size
        if size_bytes == 0:
            raise RuntimeError(
                f"Existing file {nc_path} is zero bytes. Delete it and "
                f"re-run to download fresh."
            )
        sha_hex = _hash_file(nc_path)
        now_utc = datetime.now(timezone.utc).isoformat()
        file_record = {
            "path": str(nc_path),
            "size_bytes": size_bytes,
            "sha256": sha_hex,
            "downloaded_utc": now_utc,
            "reconstructed": True,
        }
        _update_manifest(workdir, period, meta, license_str, file_record)
        return {
            "source_key": _SOURCE_KEY,
            "access_url": access["url"],
            "doi": meta["doi"],
            "license": license_str,
            "variables": [v["name"] for v in meta["variables"]],
            "period": period,
            "spatial_extent": meta.get("spatial_extent", "CONUS"),
            "download_timestamp": now_utc,
            "file": file_record,
        }
```

The `"reconstructed": True` flag in the file record makes it visible to operators that the manifest entry was rebuilt rather than freshly downloaded.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/test_fetch_mwbm_climgrid.py -v`
Expected: PASS for all five tests.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/mwbm_climgrid.py tests/test_fetch_mwbm_climgrid.py
pixi run git commit -m "$(cat <<'EOF'
feat(fetch): mwbm_climgrid manifest repair when file is present

When the datastore has the file but manifest.json lacks an entry,
hash the file, write the manifest with reconstructed=true, and skip
the download. Avoids re-downloading 7.5 GB on operator-initiated copies.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Post-download CF metadata validation

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/mwbm_climgrid.py`
- Modify: `tests/test_fetch_mwbm_climgrid.py`

After a successful download, open the NetCDF lazily and verify:
1. All four expected variables are present (`runoff`, `aet`, `soilstorage`, `swe`).
2. Each variable's `cell_methods` matches the catalog declaration.

The check raises `RuntimeError` on mismatch; on success the file is left in place. This catches publisher-side metadata changes before downstream targets silently consume mis-aggregated data.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fetch_mwbm_climgrid.py`:

```python
def _write_dummy_nc_with_bad_cell_methods(path: Path) -> None:
    """Like _write_dummy_nc but flips runoff cell_methods to 'time: point'."""
    import pandas as pd

    times = pd.date_range("1900-01-01", periods=2, freq="MS")
    lats = np.array([40.0, 40.5], dtype=np.float64)
    lons = np.array([-105.0, -104.5], dtype=np.float64)
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        data_vars={
            "runoff": (
                ("time", "latitude", "longitude"),
                rng.random((2, 2, 2)),
                {"units": "mm", "cell_methods": "time: point"},  # WRONG
            ),
            "aet": (
                ("time", "latitude", "longitude"),
                rng.random((2, 2, 2)),
                {"units": "mm", "cell_methods": "time: sum"},
            ),
            "soilstorage": (
                ("time", "latitude", "longitude"),
                rng.random((2, 2, 2)),
                {"units": "mm", "cell_methods": "time: point"},
            ),
            "swe": (
                ("time", "latitude", "longitude"),
                rng.random((2, 2, 2)),
                {"units": "mm", "cell_methods": "time: point"},
            ),
        },
        coords={
            "time": ("time", times),
            "latitude": ("latitude", lats, {"units": "degrees_north", "axis": "Y"}),
            "longitude": ("longitude", lons, {"units": "degrees_east", "axis": "X"}),
        },
    )
    ds["time"].attrs.update({"axis": "T", "standard_name": "time"})
    ds.to_netcdf(path)
    ds.close()


def test_fetch_rejects_mismatched_cell_methods(tmp_path):
    """Publisher metadata divergence from catalog raises a clear error."""
    workdir = _make_project(tmp_path)
    datastore = workdir / "datastore"

    fake_session = MagicMock()
    fake_session.get_item.return_value = {"id": "x"}
    fake_session.get_item_file_info.return_value = [
        {"name": "ClimGrid_WBM.nc", "url": "x", "size": 0}
    ]

    def _bad_download(url, name, dest_dir):
        _write_dummy_nc_with_bad_cell_methods(Path(dest_dir) / name)

    fake_session.download_file.side_effect = _bad_download

    with patch(
        "nhf_spatial_targets.fetch.mwbm_climgrid.SbSession",
        return_value=fake_session,
    ):
        with pytest.raises(RuntimeError, match="cell_methods"):
            fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")


def test_fetch_rejects_missing_variable(tmp_path):
    """Publisher dropping a variable raises before manifest is written."""
    workdir = _make_project(tmp_path)
    datastore = workdir / "datastore"

    fake_session = MagicMock()
    fake_session.get_item.return_value = {"id": "x"}
    fake_session.get_item_file_info.return_value = [
        {"name": "ClimGrid_WBM.nc", "url": "x", "size": 0}
    ]

    def _missing_var_download(url, name, dest_dir):
        out = Path(dest_dir) / name
        import pandas as pd

        ds = xr.Dataset(
            data_vars={
                "runoff": (
                    ("time", "latitude", "longitude"),
                    np.zeros((1, 1, 1)),
                    {"units": "mm", "cell_methods": "time: sum"},
                ),
                # aet, soilstorage, swe missing
            },
            coords={
                "time": ("time", pd.date_range("1900-01-01", periods=1, freq="MS")),
                "latitude": ("latitude", [40.0], {"axis": "Y"}),
                "longitude": ("longitude", [-105.0], {"axis": "X"}),
            },
        )
        ds["time"].attrs.update({"axis": "T", "standard_name": "time"})
        ds.to_netcdf(out)
        ds.close()

    fake_session.download_file.side_effect = _missing_var_download

    with patch(
        "nhf_spatial_targets.fetch.mwbm_climgrid.SbSession",
        return_value=fake_session,
    ):
        with pytest.raises(RuntimeError, match="missing variables"):
            fetch_mwbm_climgrid(workdir=workdir, period="1900/1900")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/test_fetch_mwbm_climgrid.py::test_fetch_rejects_mismatched_cell_methods tests/test_fetch_mwbm_climgrid.py::test_fetch_rejects_missing_variable -v`
Expected: FAIL — current implementation does not validate post-download.

- [ ] **Step 3: Add the validator and call it**

Add at module scope in `src/nhf_spatial_targets/fetch/mwbm_climgrid.py`:

```python
def _validate_downloaded_nc(nc_path: Path, meta: dict) -> None:
    """Verify variable presence and cell_methods match the catalog declaration.

    Raises RuntimeError with a clear message if the publisher-distributed
    file diverges from what the catalog declares, so downstream targets
    don't silently consume mis-aggregated data.
    """
    declared = {v["name"]: v.get("cell_methods") for v in meta["variables"]}
    try:
        ds = xr.open_dataset(nc_path, decode_times=False)
        try:
            present = set(ds.data_vars)
            missing = set(declared) - present
            if missing:
                raise RuntimeError(
                    f"{nc_path.name} is missing variables {sorted(missing)}. "
                    f"Catalog declares {sorted(declared)} but file only "
                    f"contains {sorted(present)}. Publisher may have "
                    f"reorganized the dataset; check the ScienceBase page."
                )
            for name, expected in declared.items():
                if expected is None:
                    continue
                actual = ds[name].attrs.get("cell_methods")
                if actual != expected:
                    raise RuntimeError(
                        f"{nc_path.name}: variable {name!r} has "
                        f"cell_methods={actual!r}, catalog declares "
                        f"{expected!r}. Publisher metadata diverged from "
                        f"this catalog version; do not trust this file "
                        f"for aggregation until the catalog is updated."
                    )
        finally:
            ds.close()
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot open downloaded NetCDF {nc_path}: {exc}. "
            f"The file may be truncated; delete it and re-run."
        ) from exc
```

Wire `_validate_downloaded_nc` into both branches of `fetch_mwbm_climgrid`:

(a) **Download branch** — immediately after the sha computation
(`sha_hex = _hash_file(nc_path)`) and before `file_record = {...}`:

```python
    _validate_downloaded_nc(nc_path, meta)
```

(b) **Repair branch (Task 7)** — immediately after the existing
`sha_hex = _hash_file(nc_path)` line in the `elif nc_path.exists() and
manifest_entry is None:` block, and before constructing `file_record`:

```python
        _validate_downloaded_nc(nc_path, meta)
```

If validation raises in the download path, the file remains on disk (operators can inspect it) but the manifest is *not* updated — the next fetch run will hit the repair branch, which also validates and will raise again until the file is corrected or deleted. This is the desired failure mode: surface the divergence loudly without silently overwriting and without silently losing the file.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/test_fetch_mwbm_climgrid.py -v`
Expected: PASS for all seven tests.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/mwbm_climgrid.py tests/test_fetch_mwbm_climgrid.py
pixi run git commit -m "$(cat <<'EOF'
feat(fetch): mwbm_climgrid post-download CF metadata validation

Verify variable presence and cell_methods match catalog declaration
after download. Catches publisher-side metadata drift before
aggregation consumes the file. Manifest is not updated on validation
failure so operators can inspect the file before deciding to delete.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Wire `mwbm-climgrid` into the `fetch` CLI

**Files:**
- Modify: `src/nhf_spatial_targets/cli.py`
- Modify: `tests/test_cli.py` (or wherever fetch-CLI smoke tests live; check first)

- [ ] **Step 1: Locate the fetch-CLI test pattern**

Run: `grep -nE "fetch_app|fetch_reitz2017|fetch_watergap" tests/test_cli.py 2>/dev/null | head -20`

If the file has parametrized fetch-subcommand tests mirroring `test_cli_agg.py`, follow the same pattern (Step 2). If not, still add the subcommand and the fetch-all entry — the integration test in Task 10 will exercise it.

- [ ] **Step 2: Write the failing test (if a parametrize pattern exists)**

If `tests/test_cli.py` (or `tests/test_cli_fetch.py`) has a parametrized list of `(subcommand, target_fn)` tuples like `test_cli_agg.py`, append:

```python
        ("mwbm-climgrid", "nhf_spatial_targets.cli.fetch_mwbm_climgrid"),
```

If no such parametrize exists, write a minimal test:

```python
def test_fetch_mwbm_climgrid_subcommand_exists():
    """`nhf-targets fetch mwbm-climgrid --help` exits 0."""
    from nhf_spatial_targets.cli import app
    with pytest.raises(SystemExit) as ei:
        app(["fetch", "mwbm-climgrid", "--help"])
    assert ei.value.code == 0
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/test_cli.py -v -k "mwbm"` (or wherever the new test lives).
Expected: FAIL — `mwbm-climgrid` not registered as a subcommand.

- [ ] **Step 4: Add the import and subcommand to `cli.py`**

In `src/nhf_spatial_targets/cli.py`:

(a) In `fetch_all_cmd` (around line 342), import the new fetch fn near the other fetch imports:

```python
from nhf_spatial_targets.fetch.mwbm_climgrid import fetch_mwbm_climgrid
```

And append to the `sources` list (after `("reitz2017", ..., fetch_reitz2017)`):

```python
        ("mwbm-climgrid", "mwbm_climgrid", fetch_mwbm_climgrid),
```

(b) Add a new subcommand block after the existing `fetch_reitz2017_cmd` (which ends around line 919). Mirror the reitz2017 block exactly:

```python
@fetch_app.command(name="mwbm-climgrid")
def fetch_mwbm_climgrid_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir"],
            help="Project created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
):
    """Download USGS MWBM (ClimGrid-forced) monthly outputs from ScienceBase."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.mwbm_climgrid import fetch_mwbm_climgrid

    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(
        f"[bold]Fetching MWBM ClimGrid (~7.5 GB, period {period})...[/bold]"
    )

    try:
        result = fetch_mwbm_climgrid(workdir=workdir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during MWBM ClimGrid fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print("[green]MWBM ClimGrid: downloaded to datastore[/green]")
    console.print(json_mod.dumps(result, indent=2))
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/test_cli.py tests/test_cli_agg.py -v`
Expected: PASS for the new test and every existing case.

- [ ] **Step 6: Commit**

```bash
git add src/nhf_spatial_targets/cli.py tests/test_cli.py
pixi run git commit -m "$(cat <<'EOF'
feat(cli): add fetch mwbm-climgrid subcommand and fetch-all entry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Integration test stub for end-to-end aggregation

**Files:**
- Modify: `tests/test_aggregate_integration.py`

- [ ] **Step 1: Write the skip-marked end-to-end test**

In `tests/test_aggregate_integration.py`, locate the existing `test_aggregate_reitz2017_end_to_end` block. Append immediately after (matching style):

```python
@pytest.mark.skip(reason="fixture datastore + mini-fabric not yet checked in")
def test_aggregate_mwbm_climgrid_end_to_end():
    """aggregate_mwbm_climgrid writes per-year NCs with runoff, aet, soilstorage, swe.

    Exercises the full path: ClimGrid_WBM.nc in <datastore>/mwbm_climgrid/
    -> per-year aggregation -> <project>/data/aggregated/mwbm_climgrid/<YYYY>/...
    Each per-year output should contain all four variables on the
    HRU dimension and 12 monthly time steps (or fewer for partial years).
    """
    raise NotImplementedError
```

- [ ] **Step 2: Run the test to verify it is collected and skipped (not failed)**

Run: `pixi run -e dev pytest tests/test_aggregate_integration.py -v`
Expected: Every test (including the new one) reported as `SKIPPED [...]`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_aggregate_integration.py
pixi run git commit -m "$(cat <<'EOF'
test(aggregate): add mwbm_climgrid end-to-end integration stub

Marked skipped pending fixture datastore + mini-fabric infrastructure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Declare `mwbm_climgrid` as a source in `catalog/variables.yml`

**Files:**
- Modify: `catalog/variables.yml` (add `mwbm_climgrid` to `runoff` + `aet` source lists)
- Modify: `tests/test_catalog.py` (assert the new sources are listed)

This declares *intent* — `mwbm_climgrid` participates in the runoff
and AET targets — without changing any target-builder code.
`targets/run.py` and `targets/aet.py` keep their current behavior
(2-source bounds and `NotImplementedError` respectively).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_catalog.py`:

```python
def test_runoff_lists_mwbm_climgrid():
    v = variable("runoff")
    assert "mwbm_climgrid" in v["sources"]
    # Existing sources still present
    assert "era5_land" in v["sources"]
    assert "gldas_noah_v21_monthly" in v["sources"]


def test_aet_lists_mwbm_climgrid():
    v = variable("aet")
    assert "mwbm_climgrid" in v["sources"]
    # Existing sources still present
    assert "mod16a2_v061" in v["sources"]
    assert "ssebop" in v["sources"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_catalog.py -v -k "mwbm_climgrid"`
Expected: FAIL with `AssertionError: assert 'mwbm_climgrid' in [...]` for both new tests.

- [ ] **Step 3: Update `catalog/variables.yml`**

For the `runoff:` block, change `sources:` from:

```yaml
    sources:
      - era5_land
      - gldas_noah_v21_monthly
```

to:

```yaml
    sources:
      - era5_land
      - gldas_noah_v21_monthly
      - mwbm_climgrid
```

In the same block, update `range_notes:` to acknowledge MWBM. Replace
the existing `range_notes:` block (which currently says "Replaces the
original NHM-MWBM source which had pre-computed uncertainty bounds...")
with:

```yaml
    range_notes: >
      ERA5-Land 'ro' (m water equivalent) and GLDAS-2.1 NOAH
      'Qs_acc + Qsb_acc' (kg/m² ≡ mm) are aggregated to HRU polygons
      via gdptools, harmonized to mm/month, then converted to cfs using
      HRU area and days-in-month. Per-HRU per-month:
        lower_bound = min(era5, gldas), upper_bound = max(era5, gldas).
      mwbm_climgrid is declared as a third source (Wieczorek et al.
      2024, ClimGrid-forced MWBM, mm/month directly) but not yet
      consumed by the target builder; the runoff target keeps
      producing 2-source bounds until targets/run.py is updated in a
      follow-up PR.
```

For the `aet:` block, change `sources:` from:

```yaml
    sources:
      - mod16a2_v061   # v006 used in original TM 6-B10; v061 for new runs
      - ssebop
```

to:

```yaml
    sources:
      - mod16a2_v061   # v006 used in original TM 6-B10; v061 for new runs
      - ssebop
      - mwbm_climgrid
```

In the same block, append to the existing `range_notes:` (preserving
its current text) a sentence acknowledging that MWBM is declared but
the target builder is still a stub. Replace the closing of the
existing `range_notes:` block with:

```yaml
    range_notes: >
      For each HRU and time step: lower_bound = min(src1, src2, src3),
      upper_bound = max(src1, src2, src3). All sources must be spatially
      aggregated to HRU polygons via gdptools before comparison.
      AET is NOT normalized — absolute values are compared directly.
      mwbm_climgrid contributes the MWBM-family AET trace
      (Wieczorek et al. 2024, mm/month native) and is declared here
      for the future 3-source build; targets/aet.py is currently a
      stub and will land in a follow-up PR.
```

(Other fields in the `aet:` block — `prms_variable`, `time_step`,
`period`, `units`, `range_method`, `normalize`, `output_format` — are
unchanged.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_catalog.py -v`
Expected: PASS for the two new tests and every pre-existing test.

- [ ] **Step 5: Commit**

```bash
git add catalog/variables.yml tests/test_catalog.py
pixi run git commit -m "$(cat <<'EOF'
feat(catalog): declare mwbm_climgrid as runoff + aet target source

Adds mwbm_climgrid to the runoff and aet variables.yml source lists
to record intent. No target-builder code changes — targets/run.py
keeps producing 2-source bounds and targets/aet.py remains a stub
until follow-up PRs wire the third source through.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Add `mwbm_climgrid` to `inspect_consolidated_runoff.ipynb`

**Files:**
- Modify: `notebooks/inspect_consolidated_runoff.ipynb`

This notebook's "configure source paths" cell holds a list-of-dicts
keyed by source. Each entry has a `path` (datastore NC) and a `var`
(variable name) and is iterated by the rest of the notebook.

- [ ] **Step 1: Locate the source-list cell**

Open `notebooks/inspect_consolidated_runoff.ipynb` in Jupyter (or
inspect via `jq '.cells[] | select(.source | join("") | contains("\"path\":"))' notebook.ipynb`).

Identify the cell containing the existing entries:

```python
        "path": DATASTORE / "era5_land" / "monthly" / f"era5_land_monthly_{TARGET_YEAR}.nc",
        ...
        "path": DATASTORE / "gldas_noah_v21_monthly" / "gldas_noah_v21_monthly.nc",
```

- [ ] **Step 2: Add the MWBM entry**

Append a third entry to the source list, immediately after the GLDAS
entry. The exact key shape mirrors what's already there (label/path/var
as established in the notebook); the addition is one block:

```python
    {
        "label": "MWBM (ClimGrid, runoff)",
        "path": DATASTORE / "mwbm_climgrid" / "ClimGrid_WBM.nc",
        "var": "runoff",
        "convert_to_mm_per_month": lambda da: da,  # native mm/month, no-op
    },
```

If the notebook's iteration cells inspect a different key shape than
the literal above (e.g. `"raw_units"` or `"description"`), follow the
exact key shape used for the GLDAS entry rather than improvising —
the iteration code will only see what it expects.

- [ ] **Step 3: Update the markdown intro cell**

Find the markdown cell near the top of the notebook that lists
sources. Add a third bullet:

```
- MWBM (Wieczorek et al. 2024, `runoff`, mm/month) — accessed from
  ScienceBase as a single CF-conformant NetCDF; native mm/month with
  `cell_methods: time: sum`. No conversion before plotting or
  comparison.
```

- [ ] **Step 4: Strip outputs and IDs-stable**

Run: `pixi run nbstripout --keep-id notebooks/inspect_consolidated_runoff.ipynb`
Expected: Outputs cleared, cell IDs preserved (per `reference_nbstripout_keep_id`).

- [ ] **Step 5: Quick render check (optional but recommended)**

Run: `pixi run -e dev python -c "import nbformat; nbformat.read('notebooks/inspect_consolidated_runoff.ipynb', as_version=4)"`
Expected: No exception (notebook is parseable; cell structure intact).

- [ ] **Step 6: Commit**

```bash
git add notebooks/inspect_consolidated_runoff.ipynb
pixi run git commit -m "$(cat <<'EOF'
docs(notebooks): add mwbm_climgrid to inspect_consolidated_runoff

Third source alongside ERA5-Land and GLDAS-2.1; native mm/month so
no unit conversion needed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Add `mwbm_climgrid` to `inspect_aggregated_runoff.ipynb`

**Files:**
- Modify: `notebooks/inspect_aggregated/inspect_aggregated_runoff.ipynb`

This notebook uses the shared `_helpers.discover_aggregated` to find
per-year NCs at `<project>/data/aggregated/<source_key>/<source_key>_<YYYY>_agg.nc`.
Adding a source means: (a) extend `SOURCES`, (b) extend any per-source
unit-conversion branch, (c) the helper's missing-files behavior
gracefully degrades if the user hasn't run `agg mwbm-climgrid` yet.

- [ ] **Step 1: Locate the SOURCES dict**

Open the notebook and find the cell containing:

```python
SOURCES = {
    "era5_land": {"label": "ERA5-Land (ro)", "var": "ro"},
    "gldas_noah_v21_monthly": {"label": "GLDAS-2.1 NOAH (runoff_total)", "var": "runoff_total"},
}
```

- [ ] **Step 2: Add MWBM to SOURCES**

Append:

```python
    "mwbm_climgrid": {"label": "MWBM (ClimGrid, runoff)", "var": "runoff"},
```

inside the `SOURCES` dict literal.

- [ ] **Step 3: Extend the per-source unit-conversion branch (if present)**

Search the notebook for `if source_key == "era5_land"` (or the
notebook's mm/month conversion helper). If the helper is structured
as an `if/elif` chain like:

```python
if source_key == "era5_land":
    da_mm = da * 1000.0  # m → mm
elif source_key == "gldas_noah_v21_monthly":
    da_mm = da * 8.0 * da["time"].dt.days_in_month
```

append:

```python
elif source_key == "mwbm_climgrid":
    da_mm = da  # native mm/month, no conversion
```

If the helper is structured around the central `_to_mm_per_month`
function (matching the AET notebook's pattern), add the same
no-op branch there.

- [ ] **Step 4: Update the markdown intro cell**

Find the markdown cell that introduces sources. Add a third bullet:

```
- MWBM (ClimGrid, `runoff`, mm/month) — Wieczorek et al. 2024;
  CONUS-wide MWBM forced by ClimGrid; aggregated to HRU polygons
  per year. Native mm/month — pass-through in the unit helper.
```

- [ ] **Step 5: Strip outputs**

Run: `pixi run nbstripout --keep-id notebooks/inspect_aggregated/inspect_aggregated_runoff.ipynb`

- [ ] **Step 6: Quick render check**

Run: `pixi run -e dev python -c "import nbformat; nbformat.read('notebooks/inspect_aggregated/inspect_aggregated_runoff.ipynb', as_version=4)"`
Expected: No exception.

- [ ] **Step 7: Commit**

```bash
git add notebooks/inspect_aggregated/inspect_aggregated_runoff.ipynb
pixi run git commit -m "$(cat <<'EOF'
docs(notebooks): add mwbm_climgrid to inspect_aggregated_runoff

discover_aggregated handles missing files gracefully; the unit helper
passes MWBM runoff through unchanged (mm/month native).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Add `mwbm_climgrid` to `inspect_consolidated_aet.ipynb`

**Files:**
- Modify: `notebooks/inspect_consolidated_aet.ipynb`

Same shape as Task 12, but for AET (`aet` variable) instead of
runoff. Native mm/month with `cell_methods: time: sum`.

- [ ] **Step 1: Locate the source-list cell**

Find the cell containing the existing SSEBop and MOD16A2 path
entries (search for `mod16a2_v061_{TARGET_YEAR}_consolidated.nc`).

- [ ] **Step 2: Add the MWBM entry**

Append a third entry to the source list, mirroring the SSEBop entry's
key shape:

```python
    {
        "label": "MWBM (ClimGrid, aet)",
        "path": DATASTORE / "mwbm_climgrid" / "ClimGrid_WBM.nc",
        "var": "aet",
        "convert_to_mm_per_month": lambda da: da,  # native mm/month, no-op
    },
```

- [ ] **Step 3: Update the markdown intro cell**

Find the markdown cell that lists sources. Add a third bullet:

```
- MWBM (Wieczorek et al. 2024, `aet`, mm/month) — accessed from
  ScienceBase as a single CF-conformant NetCDF; native mm/month
  with `cell_methods: time: sum`. No conversion before comparison.
```

- [ ] **Step 4: Strip outputs**

Run: `pixi run nbstripout --keep-id notebooks/inspect_consolidated_aet.ipynb`

- [ ] **Step 5: Quick render check**

Run: `pixi run -e dev python -c "import nbformat; nbformat.read('notebooks/inspect_consolidated_aet.ipynb', as_version=4)"`
Expected: No exception.

- [ ] **Step 6: Commit**

```bash
git add notebooks/inspect_consolidated_aet.ipynb
pixi run git commit -m "$(cat <<'EOF'
docs(notebooks): add mwbm_climgrid to inspect_consolidated_aet

Third AET source alongside SSEBop and MOD16A2 v061; native mm/month.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Add `mwbm_climgrid` to `inspect_aggregated_aet.ipynb`

**Files:**
- Modify: `notebooks/inspect_aggregated/inspect_aggregated_aet.ipynb`

Mirrors Task 13 but for AET. The notebook's `_to_mm_per_month`
helper already has `if source_key == "ssebop"` (pass-through) and
`if source_key == "mod16a2_v061"` (scale + composite handling). We
add a third no-op branch.

- [ ] **Step 1: Locate the SOURCES dict**

Open the notebook and find the cell containing:

```python
SOURCES = {
    "ssebop": {"label": "SSEBop (et)", "var": "et"},
    "mod16a2_v061": {"label": "MOD16A2 v061 (ET_500m)", "var": "ET_500m"},
}
```

- [ ] **Step 2: Add MWBM to SOURCES**

Append:

```python
    "mwbm_climgrid": {"label": "MWBM (ClimGrid, aet)", "var": "aet"},
```

inside the `SOURCES` dict literal.

- [ ] **Step 3: Extend the `_to_mm_per_month` helper**

Search for `def _to_mm_per_month(`. The current shape is:

```python
def _to_mm_per_month(da: xr.DataArray, source_key: str) -> xr.DataArray:
    if source_key == "ssebop":
        return da
    if source_key == "mod16a2_v061":
        return da * 0.1  # scale_factor → mm per composite, treated as mm/month
    raise ValueError(f"No mm/month conversion for {source_key}")
```

Add a third branch before the `raise`:

```python
    if source_key == "mwbm_climgrid":
        return da  # native mm/month, no conversion
```

- [ ] **Step 4: Update the markdown intro cell**

Find the markdown cell that introduces sources. Add a third bullet:

```
- MWBM (Wieczorek et al. 2024, `aet`, mm/month) — CONUS-wide MWBM
  forced by ClimGrid; aggregated to HRU polygons per year. Native
  mm/month — pass-through in the `_to_mm_per_month` helper.
```

- [ ] **Step 5: Strip outputs**

Run: `pixi run nbstripout --keep-id notebooks/inspect_aggregated/inspect_aggregated_aet.ipynb`

- [ ] **Step 6: Quick render check**

Run: `pixi run -e dev python -c "import nbformat; nbformat.read('notebooks/inspect_aggregated/inspect_aggregated_aet.ipynb', as_version=4)"`
Expected: No exception.

- [ ] **Step 7: Commit**

```bash
git add notebooks/inspect_aggregated/inspect_aggregated_aet.ipynb
pixi run git commit -m "$(cat <<'EOF'
docs(notebooks): add mwbm_climgrid to inspect_aggregated_aet

Third source in the AET inspection notebook; pass-through in the
mm/month helper since MWBM aet is native.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Final fmt + lint + full test pass

**Files:** none directly; this is the project quality gate.

- [ ] **Step 1: Run formatter**

Run: `pixi run -e dev fmt`
Expected: Either "no changes" or a small diff. If a diff is produced, review and stage it.

- [ ] **Step 2: Run lint**

Run: `pixi run -e dev lint`
Expected: PASS with no violations. If lint fails, fix the reported issues (the most common cause is unused imports in `fetch/mwbm_climgrid.py` left over from refactoring; remove them).

- [ ] **Step 3: Run the full unit test suite**

Run: `pixi run -e dev test`
Expected: PASS for every test, including the new `test_catalog`, `test_aggregate_mwbm_climgrid`, `test_fetch_mwbm_climgrid`, `test_cli_agg` (parametrized), and `test_cli` cases.

- [ ] **Step 4: If any fixups were needed, commit them**

```bash
git status                       # confirm clean tree, or
git add -p                       # stage targeted fixups
pixi run git commit -m "$(cat <<'EOF'
chore: ruff/lint fixups for mwbm_climgrid implementation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If the tree is already clean, skip the commit.

- [ ] **Step 5: Verify the catalog inspector lists the new source**

Run: `pixi run catalog-sources | grep -A1 mwbm_climgrid`
Expected: The `mwbm_climgrid` entry is printed with `status: current`.

- [ ] **Step 6: Final commit / branch ready**

The branch is now ready for PR. Suggested PR title and body:

```
feat: add mwbm_climgrid (USGS MWBM, ClimGrid-forced 2024) source

Adds catalog entry, fetch (sciencebasepy + sha256 manifest), and
aggregator (SourceAdapter, files_glob='ClimGrid_WBM.nc') for the
modern USGS MWBM dataset published by Wieczorek et al. 2024
(doi:10.5066/P9QCLGKM). Restores an MWBM-family source after the
NHM-MWBM retirement (issue #41).

No target-builder wiring in this PR — that's deferred per spec.

Spec: docs/superpowers/specs/2026-04-28-mwbm-climgrid-design.md
Plan: docs/superpowers/plans/2026-04-28-mwbm-climgrid.md
```

---

## Notes for the implementer

- **Memory hint:** When you write the consolidated NC validation in Task 8, follow the user's stored guidance — every xarray Dataset opened from disk is `.load()`ed where appropriate and `.close()`d in a `finally`. The `_validate_downloaded_nc` example above already does this; preserve the pattern.
- **Pre-commit hooks** run `pixi run ruff format --check`, `pixi run ruff check`, and `pixi run pytest -m 'not integration'` against staged files. Always commit via `pixi run git commit` (not bare `git commit`) to keep the hook resolution fast.
- **No `git commit --amend`** if a hook fails — fix the issue, re-stage, and create a new commit (the failed commit didn't actually land).
- The fetch is mocked everywhere in unit tests; no test in this plan touches the network. The real download is exercised only via the integration test in Task 10 (currently skipped).
