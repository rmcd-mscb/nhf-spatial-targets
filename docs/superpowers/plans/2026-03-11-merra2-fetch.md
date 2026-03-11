# MERRA-2 Fetch Module Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the MERRA-2 fetch module to download monthly soil moisture data (SFMC, GWETROOT) via `earthaccess`, subsetted to the run workspace's fabric bounding box.

**Architecture:** Single fetch module reads metadata from `catalog.py`, authenticates via `earthaccess.login()`, searches for M2TMNXLND granules by bbox and temporal range, downloads to `run_dir/data/raw/merra2/`, and returns a provenance dict for `manifest.json`.

**Tech Stack:** earthaccess, xarray, pathlib, json, warnings

**Spec:** `docs/superpowers/specs/2026-03-11-merra2-fetch-design.md`

---

## Chunk 1: Unit tests and implementation

### Task 1: Write unit tests for fetch_merra2

**Files:**
- Create: `tests/test_merra2.py`

- [ ] **Step 1: Write test file with all unit tests**

Create `tests/test_merra2.py`:

```python
"""Tests for MERRA-2 soil moisture fetch module."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """Create a minimal run workspace with fabric.json."""
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "merra2").mkdir(parents=True)
    fabric = {
        "bbox_buffered": {
            "minx": -125.1,
            "miny": 23.9,
            "maxx": -65.9,
            "maxy": 50.1,
        }
    }
    (rd / "fabric.json").write_text(json.dumps(fabric))
    return rd


def _mock_granule(name: str) -> MagicMock:
    """Create a mock granule object with a data_links method."""
    g = MagicMock()
    g.__str__ = lambda self: name
    return g


# ---- Authentication --------------------------------------------------------


@patch("earthaccess.login")
def test_login_called(mock_login, run_dir):
    """earthaccess.login() is called before searching."""
    mock_login.return_value = MagicMock(authenticated=True)
    with patch("earthaccess.search_data", return_value=[]):
        with pytest.raises(ValueError, match="No granules found"):
            from nhf_spatial_targets.fetch.merra2 import fetch_merra2

            fetch_merra2(run_dir=run_dir, period="2010/2010")
    mock_login.assert_called_once()


@patch("earthaccess.login")
def test_login_failure_raises(mock_login, run_dir):
    """RuntimeError raised when earthaccess.login() fails."""
    mock_login.return_value = MagicMock(authenticated=False)
    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with pytest.raises(RuntimeError, match="Earthdata"):
        fetch_merra2(run_dir=run_dir, period="2010/2010")


# ---- Search parameters -----------------------------------------------------


@patch("earthaccess.download", return_value=[])
@patch("earthaccess.search_data")
@patch("earthaccess.login")
def test_search_params(mock_login, mock_search, mock_dl, run_dir):
    """search_data called with correct short_name, bbox tuple, and temporal."""
    mock_login.return_value = MagicMock(authenticated=True)
    mock_search.return_value = [_mock_granule("g1")]

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    fetch_merra2(run_dir=run_dir, period="2005/2006")

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["short_name"] == "M2TMNXLND"
    assert call_kwargs["bounding_box"] == (-125.1, 23.9, -65.9, 50.1)
    assert call_kwargs["temporal"] == ("2005-01-01", "2006-12-31")


# ---- No results ------------------------------------------------------------


@patch("earthaccess.login")
@patch("earthaccess.search_data", return_value=[])
def test_no_granules_raises(mock_search, mock_login, run_dir):
    """ValueError raised when search returns zero granules."""
    mock_login.return_value = MagicMock(authenticated=True)
    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with pytest.raises(ValueError, match="No granules found"):
        fetch_merra2(run_dir=run_dir, period="2010/2010")


@patch("earthaccess.login")
def test_login_returns_none_raises(mock_login, run_dir):
    """RuntimeError raised when earthaccess.login() returns None."""
    mock_login.return_value = None
    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with pytest.raises(RuntimeError, match="Earthdata"):
        fetch_merra2(run_dir=run_dir, period="2010/2010")


# ---- Output directory ------------------------------------------------------


@patch("earthaccess.download", return_value=[])
@patch("earthaccess.search_data")
@patch("earthaccess.login")
def test_output_dir(mock_login, mock_search, mock_dl, run_dir):
    """Download writes to run_dir/data/raw/merra2/."""
    mock_login.return_value = MagicMock(authenticated=True)
    mock_search.return_value = [_mock_granule("g1")]

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    fetch_merra2(run_dir=run_dir, period="2010/2010")

    mock_dl.assert_called_once()
    call_args = mock_dl.call_args
    local_path = call_args[1].get("local_path") or call_args[0][1]
    assert Path(local_path) == run_dir / "data" / "raw" / "merra2"


# ---- Provenance record -----------------------------------------------------


@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("earthaccess.login")
def test_provenance_record(mock_login, mock_search, mock_dl, run_dir):
    """Returned dict has all required provenance keys."""
    mock_login.return_value = MagicMock(authenticated=True)
    mock_search.return_value = [_mock_granule("g1")]

    # Simulate downloaded files
    f1 = run_dir / "data" / "raw" / "merra2" / "MERRA2_201001.nc4"
    f1.write_bytes(b"fake")
    mock_dl.return_value = [str(f1)]

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    result = fetch_merra2(run_dir=run_dir, period="2010/2010")

    assert result["source_key"] == "merra2"
    assert "access_url" in result
    assert result["variables"] == ["SFMC", "GWETROOT"]
    assert result["period"] == "2010/2010"
    assert "bbox" in result
    assert "download_timestamp" in result
    assert isinstance(result["files"], list)
    assert len(result["files"]) == 1
    assert "path" in result["files"][0]
    assert "size_bytes" in result["files"][0]
    # Path should be relative to run_dir
    assert not Path(result["files"][0]["path"]).is_absolute()


# ---- Superseded warning ----------------------------------------------------


@patch("earthaccess.download", return_value=[])
@patch("earthaccess.search_data")
@patch("earthaccess.login")
@patch("nhf_spatial_targets.catalog.source")
def test_superseded_warning(mock_source, mock_login, mock_search, mock_dl, run_dir):
    """DeprecationWarning emitted when catalog status is superseded."""
    mock_source.return_value = {
        "status": "superseded",
        "access": {"url": "https://example.com"},
        "variables": ["SFMC"],
    }
    mock_login.return_value = MagicMock(authenticated=True)
    mock_search.return_value = [_mock_granule("g1")]

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fetch_merra2(run_dir=run_dir, period="2010/2010")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "superseded" in str(dep_warnings[0].message).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -- tests/test_merra2.py -v`
Expected: FAIL — `fetch_merra2` raises `NotImplementedError` or import errors.

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_merra2.py
git commit -m "Add unit tests for MERRA-2 fetch module"
```

---

### Task 2: Implement fetch_merra2

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/merra2.py` (replace stub)

- [ ] **Step 1: Write the implementation**

Replace `src/nhf_spatial_targets/fetch/merra2.py` with:

```python
"""Fetch MERRA-2 monthly soil moisture via earthaccess."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import earthaccess

from nhf_spatial_targets.catalog import source as catalog_source

_SOURCE_KEY = "merra2"
_SHORT_NAME = "M2TMNXLND"


def fetch_merra2(run_dir: Path, period: str) -> dict:
    """Download MERRA-2 monthly soil moisture for the given period.

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
    meta = catalog_source(_SOURCE_KEY)

    # Warn if source is superseded
    if meta.get("status") == "superseded":
        warnings.warn(
            f"Source '{_SOURCE_KEY}' is superseded. "
            f"Consider using '{meta.get('superseded_by', 'unknown')}'.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Authenticate
    auth = earthaccess.login()
    if auth is None or not auth.authenticated:
        raise RuntimeError(
            "NASA Earthdata login failed. Register at "
            "https://urs.earthdata.nasa.gov/users/new"
        )

    # Read bounding box from fabric.json
    fabric_path = run_dir / "fabric.json"
    fabric = json.loads(fabric_path.read_text())
    bbox = fabric["bbox_buffered"]
    bbox_tuple = (bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"])

    # Parse period
    start_year, end_year = period.split("/")
    temporal = (f"{start_year}-01-01", f"{end_year}-12-31")

    # Search for granules
    granules = earthaccess.search_data(
        short_name=_SHORT_NAME,
        bounding_box=bbox_tuple,
        temporal=temporal,
    )

    if not granules:
        raise ValueError(
            f"No granules found for {_SHORT_NAME} with "
            f"bbox={bbox_tuple}, temporal={temporal}"
        )

    # Download
    output_dir = run_dir / "data" / "raw" / _SOURCE_KEY
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = earthaccess.download(
        granules,
        local_path=str(output_dir),
    )

    # Build provenance record
    variables = meta.get("variables", ["SFMC", "GWETROOT"])
    files = []
    for fpath in downloaded:
        p = Path(fpath)
        if p.exists():
            rel = p.relative_to(run_dir)
            files.append({
                "path": str(rel),
                "size_bytes": p.stat().st_size,
            })

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "variables": variables,
        "period": period,
        "bbox": bbox,
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
```

- [ ] **Step 2: Run unit tests**

Run: `pixi run -e dev test -- tests/test_merra2.py -v`
Expected: All tests pass.

- [ ] **Step 3: Run full test suite**

Run: `pixi run -e dev test`
Expected: All 11 existing + new merra2 tests pass.

- [ ] **Step 4: Run lint and format check**

Run: `pixi run -e dev fmt && pixi run -e dev lint`
Expected: Clean.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/merra2.py
git commit -m "Implement MERRA-2 fetch module via earthaccess"
```

---

## Chunk 2: Integration test

### Task 3: Add integration test

**Files:**
- Modify: `tests/test_merra2.py` (append integration test)

- [ ] **Step 1: Add integration test to test_merra2.py**

Append to `tests/test_merra2.py`:

```python
# ---- Integration test (requires NASA Earthdata credentials) ----------------


@pytest.mark.integration
def test_fetch_merra2_real_download(tmp_path):
    """End-to-end download of one year of MERRA-2 data."""
    import xarray as xr

    # Set up minimal run workspace
    run_dir = tmp_path / "run"
    run_dir.mkdir()
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

    # Verify provenance
    assert result["source_key"] == "merra2"
    assert len(result["files"]) > 0

    # Verify at least one file is valid NetCDF with expected variables
    first_file = run_dir / result["files"][0]["path"]
    assert first_file.exists()
    ds = xr.open_dataset(first_file)
    assert "SFMC" in ds.data_vars or "GWETROOT" in ds.data_vars
    ds.close()
```

- [ ] **Step 2: Verify integration test is skipped in normal runs**

Run: `pixi run -e dev test -- tests/test_merra2.py -v`
Expected: Integration test not collected (no `integration` marker registered or skipped).

- [ ] **Step 3: Register the integration marker in pyproject.toml**

Add `markers` to the existing `[tool.pytest.ini_options]` section in `pyproject.toml` (do not duplicate the section header or `testpaths`):

```toml
markers = [
    "integration: tests requiring network access or real data files",
]
```

The section should look like:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: tests requiring network access or real data files",
]
```

- [ ] **Step 4: Verify marker works**

Run: `pixi run -e dev test -- tests/test_merra2.py -v`
Expected: Integration test shows as SKIPPED or not selected (unit tests pass, integration test only runs with `-m integration`).

Run: `pixi run -e dev test-unit -- tests/test_merra2.py -v`
Expected: Integration test excluded, unit tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_merra2.py pyproject.toml
git commit -m "Add integration test for MERRA-2 fetch and register pytest marker"
```
