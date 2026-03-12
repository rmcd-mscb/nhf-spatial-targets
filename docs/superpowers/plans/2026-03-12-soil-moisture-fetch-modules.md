> **SUPERSEDED:** Kerchunk references in this document are outdated — consolidation now uses direct xarray merge + NetCDF (PR #10). Retained for historical context.

# Soil Moisture Fetch Modules Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement fetch modules for NCEP/NCAR, NLDAS-2 MOSAIC, and NLDAS-2 NOAH soil moisture sources, plus shared credential and period-parsing utilities.

**Architecture:** Three fetch modules following the established MERRA-2 pattern — shared earthdata auth (`_auth.py`) and period parsing (`_period.py`) extracted from merra2.py, NLDAS MOSAIC/NOAH sharing a single module with parameterized source key, NCEP/NCAR using direct HTTPS download with daily-to-monthly aggregation. Per-source Kerchunk consolidation functions in consolidate.py.

**Tech Stack:** earthaccess (NASA auth/search/download), urllib (HTTPS download), xarray (NetCDF I/O, resampling), kerchunk (virtual Zarr), cyclopts (CLI), pytest (testing)

**Spec:** `docs/superpowers/specs/2026-03-12-soil-moisture-fetch-modules-design.md`

---

## Chunk 1: Catalog Updates and Shared Utilities

### Task 1: Update catalog sources and variables

**Files:**
- Modify: `catalog/sources.yml:252-338`
- Modify: `catalog/variables.yml:115-119`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update NLDAS MOSAIC variables in sources.yml**

Replace the single `SOILM_UNKNOWN` variable entry (lines 292-301) for `nldas_mosaic` with three confirmed variables and change status:

```yaml
  nldas_mosaic:
    # ... keep name, description, citations, access unchanged ...
    variables:
      - name: SoilM_0_10cm
        long_name: "soil moisture 0-10 cm"
        layer_depth_m: "0.00-0.10"
        units: kg/m2
      - name: SoilM_10_40cm
        long_name: "soil moisture 10-40 cm"
        layer_depth_m: "0.10-0.40"
        units: kg/m2
      - name: SoilM_40_200cm
        long_name: "soil moisture 40-200 cm"
        layer_depth_m: "0.40-2.00"
        units: kg/m2
    # ... keep time_step, period, spatial_extent, spatial_resolution, units unchanged ...
    status: current
```

- [ ] **Step 2: Update NLDAS NOAH variables in sources.yml**

Same replacement for `nldas_noah` (lines 323-332): replace `SOILM_UNKNOWN` with three variables, update description to note 3 shared layers (not 4), change status to `current`.

```yaml
  nldas_noah:
    description: >
      North American Land Data Assimilation System phase 2 (NLDAS-2),
      NOAH land surface model output, soil moisture.
      Three soil layers shared with fetch: 0-10cm, 10-40cm, 40-200cm.
    # ... keep citations, access unchanged ...
    variables:
      - name: SoilM_0_10cm
        long_name: "soil moisture 0-10 cm"
        layer_depth_m: "0.00-0.10"
        units: kg/m2
      - name: SoilM_10_40cm
        long_name: "soil moisture 10-40 cm"
        layer_depth_m: "0.10-0.40"
        units: kg/m2
      - name: SoilM_40_200cm
        long_name: "soil moisture 40-200 cm"
        layer_depth_m: "0.40-2.00"
        units: kg/m2
    status: current
```

- [ ] **Step 3: Update NCEP/NCAR entry in sources.yml**

Replace the single variable and move `file_pattern` from `access` to per-variable. Remove top-level `access.file_pattern`. Add second layer variable:

```yaml
  ncep_ncar:
    # ... keep name, description, citations unchanged ...
    access:
      type: noaa_psl
      url: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html
      notes: Monthly means derived from daily Gaussian grid files.
    variables:
      - name: soilw
        file_variable: soilw.0-10cm.gauss
        file_pattern: "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface_gauss/soilw.0-10cm.gauss.{year}.nc"
        long_name: "volumetric soil moisture 0-10 cm"
        layer_depth_m: "0.00-0.10"
        units: kg/m2
      - name: soilw
        file_variable: soilw.10-200cm.gauss
        file_pattern: "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface_gauss/soilw.10-200cm.gauss.{year}.nc"
        long_name: "volumetric soil moisture 10-200 cm"
        layer_depth_m: "0.10-2.00"
        units: kg/m2
    # ... keep time_step, period, spatial_extent, spatial_resolution, units, status unchanged ...
```

- [ ] **Step 4: Update variables.yml range_notes**

Replace lines 118-119 in `catalog/variables.yml`:
```yaml
        NLDAS-MOSAIC: layer 1 (0-0.10m, kg/m2) — variable name TBD
        NLDAS-NOAH:   layer 1 (0-0.10m, kg/m2) — variable name TBD
```
with:
```yaml
        NLDAS-MOSAIC: SoilM_0_10cm (0-0.10m, kg/m2) — confirmed
        NLDAS-NOAH:   SoilM_0_10cm (0-0.10m, kg/m2) — confirmed
```

- [ ] **Step 5: Update CLAUDE.md known gaps**

Remove the NLDAS-2 variable names bullet from "Still open" section:
```
- NLDAS-2 MOSAIC / NOAH upper-layer variable names — needs file inspection via earthaccess or GES DISC README
```

Add to "Resolved" section:
```
- NLDAS-2 MOSAIC / NOAH variable names — confirmed: SoilM_0_10cm, SoilM_10_40cm, SoilM_40_200cm
```

- [ ] **Step 6: Run lint and tests**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All tests pass, no lint errors.

- [ ] **Step 7: Commit**

```bash
git add catalog/sources.yml catalog/variables.yml CLAUDE.md
git commit -m "catalog: update NLDAS variables, add NCEP/NCAR second layer"
```

---

### Task 2: Extract shared period parsing (`fetch/_period.py`)

**Files:**
- Create: `src/nhf_spatial_targets/fetch/_period.py`
- Modify: `src/nhf_spatial_targets/fetch/merra2.py:33-60`
- Modify: `tests/test_merra2.py:237-258,328-333`
- Create: `tests/test_period.py`

- [ ] **Step 1: Write tests for period parsing**

Create `tests/test_period.py`:

```python
"""Tests for shared period parsing utilities."""

from __future__ import annotations

import pytest

from nhf_spatial_targets.fetch._period import (
    months_in_period,
    parse_period,
    years_in_period,
)


def test_parse_period_valid():
    assert parse_period("2005/2006") == ("2005-01-01", "2006-12-31")


def test_parse_period_single_year():
    assert parse_period("2010/2010") == ("2010-01-01", "2010-12-31")


def test_parse_period_missing_slash():
    with pytest.raises(ValueError, match="YYYY/YYYY"):
        parse_period("2010")


def test_parse_period_non_numeric():
    with pytest.raises(ValueError, match="integers"):
        parse_period("abc/def")


def test_parse_period_reversed():
    with pytest.raises(ValueError, match="before start year"):
        parse_period("2015/2010")


def test_months_in_period():
    months = months_in_period("2010/2010")
    assert len(months) == 12
    assert months[0] == "2010-01"
    assert months[-1] == "2010-12"


def test_months_in_period_multi_year():
    months = months_in_period("2009/2010")
    assert len(months) == 24
    assert months[0] == "2009-01"
    assert months[-1] == "2010-12"


def test_years_in_period():
    years = years_in_period("2005/2008")
    assert years == [2005, 2006, 2007, 2008]


def test_years_in_period_single():
    years = years_in_period("2010/2010")
    assert years == [2010]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -- tests/test_period.py -v`
Expected: FAIL — `_period` module does not exist yet.

- [ ] **Step 3: Create `fetch/_period.py`**

Create `src/nhf_spatial_targets/fetch/_period.py`:

```python
"""Shared period-parsing utilities for fetch modules."""

from __future__ import annotations


def parse_period(period: str) -> tuple[str, str]:
    """Parse ``"YYYY/YYYY"`` into ``("YYYY-01-01", "YYYY-12-31")``."""
    parts = period.split("/")
    if len(parts) != 2:
        raise ValueError(f"period must be 'YYYY/YYYY', got: {period!r}")
    start_year, end_year = parts
    try:
        start_int, end_int = int(start_year), int(end_year)
    except ValueError:
        raise ValueError(
            f"period years must be integers, got: {period!r}"
        ) from None
    if end_int < start_int:
        raise ValueError(
            f"period end year ({end_year}) is before start year "
            f"({start_year}). Use 'YYYY/YYYY' with start <= end."
        )
    return (f"{start_year}-01-01", f"{end_year}-12-31")


def months_in_period(period: str) -> list[str]:
    """Return list of 'YYYY-MM' strings for every month in the period."""
    parse_period(period)  # validate format
    parts = period.split("/")
    start_year, end_year = int(parts[0]), int(parts[1])
    months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            months.append(f"{year}-{month:02d}")
    return months


def years_in_period(period: str) -> list[int]:
    """Return list of year integers for every year in the period."""
    parse_period(period)  # validate format
    parts = period.split("/")
    start_year, end_year = int(parts[0]), int(parts[1])
    return list(range(start_year, end_year + 1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev test -- tests/test_period.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Update merra2.py to import from _period**

In `src/nhf_spatial_targets/fetch/merra2.py`:
- Remove the `_parse_period` and `_months_in_period` function definitions (lines 33-60).
- Add import at top:
  ```python
  from nhf_spatial_targets.fetch._period import months_in_period, parse_period
  ```
- Replace all calls to `_parse_period` with `parse_period` and `_months_in_period` with `months_in_period`.

- [ ] **Step 6: Update test_merra2.py period test imports**

In `tests/test_merra2.py`, update the three period validation tests (lines 237-258) to import from `_period`:
```python
from nhf_spatial_targets.fetch._period import parse_period
```
And the incremental test (line 328):
```python
from nhf_spatial_targets.fetch._period import months_in_period
```

- [ ] **Step 7: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/nhf_spatial_targets/fetch/_period.py tests/test_period.py \
        src/nhf_spatial_targets/fetch/merra2.py tests/test_merra2.py
git commit -m "refactor: extract shared period parsing to fetch/_period.py"
```

---

### Task 3: Create shared earthdata auth (`fetch/_auth.py`)

**Files:**
- Create: `src/nhf_spatial_targets/fetch/_auth.py`
- Create: `tests/test_auth.py`
- Modify: `src/nhf_spatial_targets/fetch/merra2.py:102-138`
- Modify: `tests/test_merra2.py` (auth mock paths)

- [ ] **Step 1: Write tests for auth module**

Create `tests/test_auth.py`:

```python
"""Tests for shared earthdata credential handling."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nhf_spatial_targets.fetch._auth import earthdata_login


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "run"
    rd.mkdir()
    return rd


def _write_credentials(run_dir: Path, username: str, password: str) -> None:
    import yaml

    creds = {
        "nasa_earthdata": {"username": username, "password": password},
        "sciencebase": {"username": "", "password": ""},
    }
    (run_dir / ".credentials.yml").write_text(yaml.dump(creds))


@patch("nhf_spatial_targets.fetch._auth.earthaccess.login")
def test_credentials_file_used(mock_login, run_dir):
    """Credentials from .credentials.yml are set as env vars."""
    _write_credentials(run_dir, "myuser", "mypass")
    mock_login.return_value = MagicMock(authenticated=True)

    earthdata_login(run_dir)

    mock_login.assert_called_once_with(strategy="environment")


@patch("nhf_spatial_targets.fetch._auth.earthaccess.login")
def test_empty_credentials_falls_back(mock_login, run_dir):
    """Empty credentials in .credentials.yml trigger default login."""
    _write_credentials(run_dir, "", "")
    mock_login.return_value = MagicMock(authenticated=True)

    earthdata_login(run_dir)

    mock_login.assert_called_once_with()


@patch("nhf_spatial_targets.fetch._auth.earthaccess.login")
def test_no_credentials_file_falls_back(mock_login, run_dir):
    """Missing .credentials.yml triggers default login."""
    mock_login.return_value = MagicMock(authenticated=True)

    earthdata_login(run_dir)

    mock_login.assert_called_once_with()


@patch("nhf_spatial_targets.fetch._auth.earthaccess.login")
def test_login_failure_raises(mock_login, run_dir):
    """RuntimeError raised when all login strategies fail."""
    mock_login.return_value = MagicMock(authenticated=False)

    with pytest.raises(RuntimeError, match="Earthdata"):
        earthdata_login(run_dir)


@patch("nhf_spatial_targets.fetch._auth.earthaccess.login")
def test_login_returns_none_raises(mock_login, run_dir):
    """RuntimeError raised when login returns None."""
    mock_login.return_value = None

    with pytest.raises(RuntimeError, match="Earthdata"):
        earthdata_login(run_dir)


@patch("nhf_spatial_targets.fetch._auth.earthaccess.login")
def test_credentials_fallback_on_env_failure(mock_login, run_dir):
    """Falls back to default login if environment strategy fails."""
    _write_credentials(run_dir, "myuser", "mypass")
    # First call (strategy="environment") fails, second (default) succeeds
    mock_login.side_effect = [
        MagicMock(authenticated=False),
        MagicMock(authenticated=True),
    ]

    result = earthdata_login(run_dir)

    assert result.authenticated
    assert mock_login.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -- tests/test_auth.py -v`
Expected: FAIL — `_auth` module does not exist yet.

- [ ] **Step 3: Create `fetch/_auth.py`**

Create `src/nhf_spatial_targets/fetch/_auth.py`:

```python
"""Shared NASA Earthdata authentication for fetch modules."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import earthaccess
import yaml

logger = logging.getLogger(__name__)


def earthdata_login(run_dir: Path) -> earthaccess.Auth:
    """Authenticate with NASA Earthdata.

    Reads credentials from ``run_dir/.credentials.yml`` if available,
    otherwise falls back to earthaccess default login strategies
    (netrc, interactive prompt, etc.).

    Parameters
    ----------
    run_dir : Path
        Run workspace directory containing ``.credentials.yml``.

    Returns
    -------
    earthaccess.Auth
        Authenticated session object.

    Raises
    ------
    RuntimeError
        If all login strategies fail.
    """
    creds_path = run_dir / ".credentials.yml"
    username, password = "", ""

    if creds_path.exists():
        try:
            data = yaml.safe_load(creds_path.read_text()) or {}
            nasa = data.get("nasa_earthdata", {})
            username = nasa.get("username", "") or ""
            password = nasa.get("password", "") or ""
        except (yaml.YAMLError, AttributeError):
            logger.warning("Could not parse %s, using default login", creds_path)

    if username and password:
        os.environ["EARTHDATA_USERNAME"] = username
        os.environ["EARTHDATA_PASSWORD"] = password
        logger.info("Using credentials from .credentials.yml")
        auth = earthaccess.login(strategy="environment")
        if auth is not None and auth.authenticated:
            return auth
        logger.warning(
            "Environment-based login failed, falling back to default"
        )

    auth = earthaccess.login()
    if auth is None or not auth.authenticated:
        raise RuntimeError(
            "NASA Earthdata login failed. Either fill in "
            f"{creds_path} or register at "
            "https://urs.earthdata.nasa.gov/users/new"
        )
    return auth
```

- [ ] **Step 4: Run auth tests to verify they pass**

Run: `pixi run -e dev test -- tests/test_auth.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Update merra2.py to use earthdata_login**

In `src/nhf_spatial_targets/fetch/merra2.py`:
- Add import: `from nhf_spatial_targets.fetch._auth import earthdata_login`
- Replace lines 133-139:
  ```python
  auth = earthaccess.login()
  if auth is None or not auth.authenticated:
      raise RuntimeError(
          "NASA Earthdata login failed. Register at "
          "https://urs.earthdata.nasa.gov/users/new"
      )
  logger.info("Authenticated with NASA Earthdata")
  ```
  with:
  ```python
  earthdata_login(run_dir)
  logger.info("Authenticated with NASA Earthdata")
  ```
- Remove the `import earthaccess` at the top of the file (it's no longer used directly for login, but is still needed for `earthaccess.search_data` and `earthaccess.download`).

- [ ] **Step 6: Update test_merra2.py auth mock paths**

In `tests/test_merra2.py`, update auth-related test mocks. The login tests (`test_login_called`, `test_login_failure_raises`, `test_login_returns_none_raises`) should now mock `nhf_spatial_targets.fetch._auth.earthdata_login` instead of `earthaccess.login`. The remaining tests that mock `earthaccess.login` should also be updated.

For tests that need auth to succeed, use:
```python
@patch("nhf_spatial_targets.fetch.merra2.earthdata_login")
```

For auth-specific tests, mock at the source:
```python
@patch("nhf_spatial_targets.fetch._auth.earthaccess.login")
```

Note: `test_login_called`, `test_login_failure_raises`, and `test_login_returns_none_raises` are now redundant with `test_auth.py`. Either remove them or convert them to verify `merra2.py` calls `earthdata_login(run_dir)`.

- [ ] **Step 7: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/nhf_spatial_targets/fetch/_auth.py tests/test_auth.py \
        src/nhf_spatial_targets/fetch/merra2.py tests/test_merra2.py
git commit -m "feat: add shared earthdata auth with .credentials.yml support"
```

---

## Chunk 2: NLDAS Fetch Module

### Task 4: Implement NLDAS fetch module

**Files:**
- Create: `tests/test_nldas.py`
- Modify: `src/nhf_spatial_targets/fetch/nldas.py` (rewrite from stub)

- [ ] **Step 1: Write NLDAS tests**

Create `tests/test_nldas.py`. Follow the `test_merra2.py` patterns exactly, parameterized for both sources. Key test structure:

```python
"""Tests for NLDAS-2 fetch module (MOSAIC and NOAH)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic, fetch_nldas_noah

_MOCK_CONSOLIDATION = {
    "kerchunk_ref": "data/raw/nldas_mosaic/nldas_mosaic_refs.json",
    "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
    "n_files": 1,
    "variables": ["SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_200cm"],
}

_NLDAS_DATE_RE_STR = r"NLDAS_MOS0125_M\.A(\d{4})(\d{2})\.002\.\S+\.nc"


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "nldas_mosaic").mkdir(parents=True)
    (rd / "data" / "raw" / "nldas_noah").mkdir(parents=True)
    fabric = {
        "bbox_buffered": {
            "minx": -125.1, "miny": 23.9, "maxx": -65.9, "maxy": 50.1,
        }
    }
    (rd / "fabric.json").write_text(json.dumps(fabric))
    return rd
```

Tests to include (for both `fetch_nldas_mosaic` and `fetch_nldas_noah` — use `@pytest.mark.parametrize` where possible):
- `test_login_called` — `earthdata_login` called with `run_dir`
- `test_search_params_mosaic` — correct `short_name=NLDAS_MOS0125_M`
- `test_search_params_noah` — correct `short_name=NLDAS_NOAH0125_M`
- `test_no_granules_raises` — ValueError
- `test_output_dir` — writes to `data/raw/<source_key>/`
- `test_provenance_record` — all required keys
- `test_superseded_warning` — DeprecationWarning
- `test_missing_fabric_raises` — FileNotFoundError
- `test_empty_download_raises` — RuntimeError
- `test_incremental_skips_existing` — months in manifest skipped
- `test_manifest_updated` — provenance in manifest.json
- `test_manifest_preserves_download_timestamp` — original timestamps kept

Mock `nhf_spatial_targets.fetch.nldas.earthdata_login` for auth.
Mock `nhf_spatial_targets.fetch.nldas.consolidate_nldas` for consolidation.
Mock `earthaccess.search_data` and `earthaccess.download` for network.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -- tests/test_nldas.py -v`
Expected: FAIL — functions raise `NotImplementedError`.

- [ ] **Step 3: Implement `fetch/nldas.py`**

Rewrite `src/nhf_spatial_targets/fetch/nldas.py`. Follow the `merra2.py` structure exactly:

```python
"""Fetch NLDAS-2 land surface model soil moisture via earthaccess."""

from __future__ import annotations

import json
import logging
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import earthaccess

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._auth import earthdata_login
from nhf_spatial_targets.fetch._period import months_in_period, parse_period
from nhf_spatial_targets.fetch.consolidate import consolidate_nldas

logger = logging.getLogger(__name__)

# Regex to extract YYYY-MM from NLDAS granule URLs/filenames.
# Example: NLDAS_MOS0125_M.A200101.002.grb.SUB.nc4
_NLDAS_DATE_URL_RE = re.compile(r"\.A(\d{4})(\d{2})\.")
_NLDAS_DATE_FILE_RE = re.compile(r"\.A(\d{4})(\d{2})\.")


def fetch_nldas_mosaic(run_dir: Path, period: str) -> dict:
    """Download NLDAS-2 MOSAIC monthly soil moisture."""
    return _fetch_nldas("nldas_mosaic", run_dir, period)


def fetch_nldas_noah(run_dir: Path, period: str) -> dict:
    """Download NLDAS-2 NOAH monthly soil moisture."""
    return _fetch_nldas("nldas_noah", run_dir, period)


def _granule_year_month(granule: object) -> str | None:
    """Extract 'YYYY-MM' from an earthaccess granule's data links."""
    for link in granule.data_links():
        m = _NLDAS_DATE_URL_RE.search(link)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    return None


def _year_month_from_path(path: Path) -> str:
    """Extract 'YYYY-MM' from an NLDAS filename."""
    m = _NLDAS_DATE_FILE_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot extract date from NLDAS filename: {path.name}")
    return f"{m.group(1)}-{m.group(2)}"


def _manifest_source_files(run_dir: Path, source_key: str) -> list[dict]:
    """Read manifest.json and return file records for the given source."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
        return manifest.get("sources", {}).get(source_key, {}).get("files", [])
    except (json.JSONDecodeError, KeyError):
        logger.warning("Malformed manifest.json in %s", run_dir)
        return []


def _existing_months(run_dir: Path, source_key: str) -> set[str]:
    """Return set of year_month values already fetched."""
    return {
        f["year_month"]
        for f in _manifest_source_files(run_dir, source_key)
        if "year_month" in f
    }


def _existing_file_timestamps(run_dir: Path, source_key: str) -> dict[str, str]:
    """Return {year_month: downloaded_utc} from manifest."""
    return {
        f["year_month"]: f["downloaded_utc"]
        for f in _manifest_source_files(run_dir, source_key)
        if "year_month" in f and "downloaded_utc" in f
    }


def _fetch_nldas(source_key: str, run_dir: Path, period: str) -> dict:
    """Shared implementation for NLDAS MOSAIC and NOAH fetch."""
    meta = _catalog.source(source_key)
    short_name = meta["access"]["short_name"]

    if meta.get("status") == "superseded":
        warnings.warn(
            f"Source '{source_key}' is superseded. "
            f"Consider using '{meta.get('superseded_by', 'unknown')}'.",
            DeprecationWarning,
            stacklevel=3,
        )

    earthdata_login(run_dir)
    logger.info("Authenticated with NASA Earthdata")

    fabric_path = run_dir / "fabric.json"
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"fabric.json not found in {run_dir}. "
            "Run 'nhf-targets init' to create a run workspace first."
        )
    fabric = json.loads(fabric_path.read_text())
    bbox = fabric["bbox_buffered"]
    bbox_tuple = (bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"])

    already_have = _existing_months(run_dir, source_key)
    all_months = months_in_period(period)
    needed = [m for m in all_months if m not in already_have]

    if not needed:
        logger.info("All %d months already downloaded, skipping to consolidation",
                     len(all_months))
    else:
        temporal = parse_period(period)
        granules = earthaccess.search_data(
            short_name=short_name,
            bounding_box=bbox_tuple,
            temporal=temporal,
        )
        if not granules:
            raise ValueError(
                f"No granules found for {short_name} with "
                f"bbox={bbox_tuple}, temporal={temporal}"
            )

        needed_set = set(needed)
        granules = [g for g in granules if _granule_year_month(g) in needed_set]
        if granules:
            output_dir = run_dir / "data" / "raw" / source_key
            output_dir.mkdir(parents=True, exist_ok=True)

            downloaded = earthaccess.download(granules, local_path=str(output_dir))
            if not downloaded:
                raise RuntimeError(
                    f"earthaccess.download() returned no files for "
                    f"{len(granules)} granules."
                )
            logger.info("Downloaded %d files to %s", len(downloaded), output_dir)

    # Build file inventory
    output_dir = run_dir / "data" / "raw" / source_key
    all_nc_files = sorted(
        list(output_dir.glob("*.nc4")) + list(output_dir.glob("*.nc"))
    )

    existing_timestamps = _existing_file_timestamps(run_dir, source_key)
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

    var_names = [v["name"] for v in meta["variables"]]
    consolidation = consolidate_nldas(
        run_dir=run_dir, source_key=source_key, variables=var_names
    )

    if files:
        all_ym = sorted(f["year_month"] for f in files)
        effective_period = f"{all_ym[0][:4]}/{all_ym[-1][:4]}"
    else:
        effective_period = period

    _update_manifest(run_dir, source_key, effective_period, bbox, meta,
                     files, consolidation)

    return {
        "source_key": source_key,
        "access_url": meta["access"]["url"],
        "variables": meta["variables"],
        "period": effective_period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "files": files,
        "kerchunk_ref": consolidation["kerchunk_ref"],
    }


def _update_manifest(
    run_dir: Path,
    source_key: str,
    period: str,
    bbox: dict,
    meta: dict,
    files: list[dict],
    consolidation: dict,
) -> None:
    """Merge source provenance into manifest.json."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"sources": {}, "steps": []}
    if "sources" not in manifest:
        manifest["sources"] = {}

    entry = manifest["sources"].get(source_key, {})
    entry.update({
        "source_key": source_key,
        "access_url": meta["access"]["url"],
        "period": period,
        "bbox": bbox,
        "variables": [v["name"] for v in meta["variables"]],
        "files": files,
        "kerchunk_ref": consolidation["kerchunk_ref"],
        "last_consolidated_utc": consolidation["last_consolidated_utc"],
    })
    manifest["sources"][source_key] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2))
```

- [ ] **Step 4: Run NLDAS tests to verify they pass**

Run: `pixi run -e dev test -- tests/test_nldas.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/nhf_spatial_targets/fetch/nldas.py tests/test_nldas.py
git commit -m "feat: implement NLDAS-2 MOSAIC and NOAH fetch module"
```

---

### Task 5: Add NLDAS consolidation function

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py`
- Modify: `tests/test_consolidate.py`

- [ ] **Step 1: Write consolidation tests**

Add to `tests/test_consolidate.py`. Create an NLDAS fixture with synthetic HDF5/NetCDF4 files:

```python
@pytest.fixture()
def nldas_dir(tmp_path: Path) -> Path:
    """Create synthetic NLDAS NetCDF4 files."""
    out = tmp_path / "data" / "raw" / "nldas_mosaic"
    out.mkdir(parents=True)

    lat = np.arange(25.0, 50.0, 5.0)
    lon = np.arange(-125.0, -65.0, 10.0)

    for month in range(1, 4):
        time = np.array(
            [f"2010-{month:02d}-15T00:00:00"], dtype="datetime64[ns]"
        )
        ds = xr.Dataset(
            {
                "SoilM_0_10cm": (["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32)),
                "SoilM_10_40cm": (["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32)),
                "SoilM_40_200cm": (["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32)),
                "EXTRA_VAR": (["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32)),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )
        fname = f"NLDAS_MOS0125_M.A2010{month:02d}.002.grb.SUB.nc4"
        ds.to_netcdf(out / fname)

    return out


def test_nldas_filter_variables(nldas_dir):
    from nhf_spatial_targets.fetch.consolidate import consolidate_nldas
    import fsspec

    run_dir = nldas_dir.parent.parent.parent
    consolidate_nldas(
        run_dir=run_dir, source_key="nldas_mosaic",
        variables=["SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_200cm"],
    )

    ref_path = nldas_dir / "nldas_mosaic_refs.json"
    assert ref_path.exists()

    fs = fsspec.filesystem("reference", fo=str(ref_path), target_protocol="file")
    ds = xr.open_zarr(fs.get_mapper(""), consolidated=False)
    assert "SoilM_0_10cm" in ds.data_vars
    assert "SoilM_10_40cm" in ds.data_vars
    assert "SoilM_40_200cm" in ds.data_vars
    assert "EXTRA_VAR" not in ds.data_vars
    assert len(ds.time) == 3
    ds.close()


def test_nldas_relative_paths(nldas_dir):
    from nhf_spatial_targets.fetch.consolidate import consolidate_nldas

    run_dir = nldas_dir.parent.parent.parent
    consolidate_nldas(
        run_dir=run_dir, source_key="nldas_mosaic",
        variables=["SoilM_0_10cm"],
    )

    ref_path = nldas_dir / "nldas_mosaic_refs.json"
    refs = json.loads(ref_path.read_text())
    for key, val in refs["refs"].items():
        if isinstance(val, list) and len(val) >= 1 and isinstance(val[0], str):
            assert val[0].startswith("./"), f"Non-relative path: {val[0]}"


def test_nldas_provenance_return(nldas_dir):
    from nhf_spatial_targets.fetch.consolidate import consolidate_nldas

    run_dir = nldas_dir.parent.parent.parent
    result = consolidate_nldas(
        run_dir=run_dir, source_key="nldas_mosaic",
        variables=["SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_200cm"],
    )
    assert result["kerchunk_ref"] == "data/raw/nldas_mosaic/nldas_mosaic_refs.json"
    assert "last_consolidated_utc" in result
    assert result["n_files"] == 3


def test_nldas_no_files_raises(tmp_path):
    from nhf_spatial_targets.fetch.consolidate import consolidate_nldas

    (tmp_path / "data" / "raw" / "nldas_mosaic").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        consolidate_nldas(
            run_dir=tmp_path, source_key="nldas_mosaic",
            variables=["SoilM_0_10cm"],
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -- tests/test_consolidate.py::test_nldas_filter_variables -v`
Expected: FAIL — `consolidate_nldas` does not exist.

- [ ] **Step 3: Implement `consolidate_nldas` in consolidate.py**

Add to `src/nhf_spatial_targets/fetch/consolidate.py`:

```python
def consolidate_nldas(
    run_dir: Path,
    source_key: str,
    variables: list[str],
) -> dict:
    """Build a Kerchunk JSON reference store for NLDAS files.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory.
    source_key : str
        Source key (e.g. "nldas_mosaic" or "nldas_noah").
    variables : list[str]
        Variable names to include.

    Returns
    -------
    dict
        Provenance record.
    """
    from datetime import datetime, timezone

    import kerchunk.hdf
    from kerchunk.combine import MultiZarrToZarr

    source_dir = run_dir / "data" / "raw" / source_key
    nc_files = sorted(
        list(source_dir.glob("*.nc4")) + list(source_dir.glob("*.nc"))
    )

    if not nc_files:
        raise FileNotFoundError(
            f"No NetCDF files found in {source_dir}. "
            f"Run 'nhf-targets fetch {source_key.replace('_', '-')}' first."
        )

    logger.info("Scanning %d NetCDF files for %s", len(nc_files), source_key)

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
        coo_map={"time": "cf:time"},
    )
    combined = mzz.translate()

    combined["refs"] = _make_relative(combined["refs"], source_dir)

    ref_path = source_dir / f"{source_key}_refs.json"
    ref_path.write_text(ujson.dumps(combined, indent=2))
    logger.info("Wrote Kerchunk reference store: %s", ref_path)

    return {
        "kerchunk_ref": str(ref_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(nc_files),
        "variables": variables,
    }
```

- [ ] **Step 4: Run consolidation tests**

Run: `pixi run -e dev test -- tests/test_consolidate.py -v`
Expected: All tests PASS (existing MERRA-2 tests + new NLDAS tests).

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate.py
git commit -m "feat: add NLDAS Kerchunk consolidation function"
```

---

## Chunk 3: NCEP/NCAR Fetch Module

### Task 6: Implement NCEP/NCAR fetch module

**Files:**
- Create: `tests/test_ncep_ncar.py`
- Modify: `src/nhf_spatial_targets/fetch/ncep_ncar.py` (rewrite from stub)

- [ ] **Step 1: Write NCEP/NCAR tests**

Create `tests/test_ncep_ncar.py`:

```python
"""Tests for NCEP/NCAR Reanalysis fetch module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

_MOCK_CONSOLIDATION = {
    "kerchunk_ref": "data/raw/ncep_ncar/ncep_ncar_refs.json",
    "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
    "n_files": 1,
    "variables": ["soilw"],
}


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "ncep_ncar").mkdir(parents=True)
    fabric = {
        "bbox_buffered": {
            "minx": -125.1, "miny": 23.9, "maxx": -65.9, "maxy": 50.1,
        }
    }
    (rd / "fabric.json").write_text(json.dumps(fabric))
    return rd


def _make_daily_nc(path: Path, year: int, var_name: str = "soilw"):
    """Create a synthetic daily NetCDF file for one year."""
    import pandas as pd

    times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    lat = np.arange(-90, 91, 45.0)
    lon = np.arange(-180, 180, 60.0)
    ds = xr.Dataset(
        {
            var_name: (
                ["time", "lat", "lon"],
                np.random.rand(len(times), len(lat), len(lon)).astype(np.float32),
            ),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path, format="NETCDF3_CLASSIC")
    return ds
```

Tests to include:
- `test_url_construction` — correct URL from catalog `file_pattern` + year
- `test_download_failure_raises` — RuntimeError on HTTP error
- `test_daily_to_monthly` — verify monthly means from synthetic daily data (create a synthetic daily file, run the aggregation, check 12 monthly values)
- `test_output_dir` — writes to `data/raw/ncep_ncar/`
- `test_provenance_record` — all required keys
- `test_missing_fabric_raises` — FileNotFoundError
- `test_incremental_skips_existing` — years in manifest skipped
- `test_manifest_updated` — provenance in manifest.json

Mock `urllib.request.urlretrieve` for downloads and `consolidate_ncep_ncar` for consolidation.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -- tests/test_ncep_ncar.py -v`
Expected: FAIL — `fetch_ncep_ncar` raises `NotImplementedError`.

- [ ] **Step 3: Implement `fetch/ncep_ncar.py`**

Rewrite `src/nhf_spatial_targets/fetch/ncep_ncar.py`:

```python
"""Fetch NCEP/NCAR Reanalysis soil moisture from NOAA PSL."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import parse_period, years_in_period
from nhf_spatial_targets.fetch.consolidate import consolidate_ncep_ncar

_SOURCE_KEY = "ncep_ncar"
logger = logging.getLogger(__name__)


def fetch_ncep_ncar(run_dir: Path, period: str) -> dict:
    """Download NCEP/NCAR Reanalysis soil moisture for the given period.

    Downloads daily Gaussian grid NetCDF files from NOAA PSL, aggregates
    to monthly means, and consolidates into a Kerchunk reference store.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory.
    period : str
        Temporal range as ``"YYYY/YYYY"``.

    Returns
    -------
    dict
        Provenance record.
    """
    meta = _catalog.source(_SOURCE_KEY)

    if meta.get("status") == "superseded":
        warnings.warn(
            f"Source '{_SOURCE_KEY}' is superseded. "
            f"Consider using '{meta.get('superseded_by', 'unknown')}'.",
            DeprecationWarning,
            stacklevel=2,
        )

    fabric_path = run_dir / "fabric.json"
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"fabric.json not found in {run_dir}. "
            "Run 'nhf-targets init' to create a run workspace first."
        )
    fabric = json.loads(fabric_path.read_text())
    bbox = fabric["bbox_buffered"]

    parse_period(period)  # validate
    all_years = years_in_period(period)
    already_have = _existing_years(run_dir)
    needed = [y for y in all_years if y not in already_have]

    output_dir = run_dir / "data" / "raw" / _SOURCE_KEY
    output_dir.mkdir(parents=True, exist_ok=True)

    if not needed:
        logger.info("All %d years already downloaded", len(all_years))
    else:
        for year in needed:
            for var_entry in meta["variables"]:
                file_var = var_entry["file_variable"]
                url = var_entry["file_pattern"].format(year=year)
                daily_path = output_dir / f"{file_var}.{year}.nc"

                logger.info("Downloading %s", url)
                try:
                    urllib.request.urlretrieve(url, daily_path)
                except urllib.error.HTTPError as exc:
                    raise RuntimeError(
                        f"HTTP {exc.code} downloading {url}"
                    ) from exc

                # Aggregate daily to monthly
                import xarray as xr

                ds = xr.open_dataset(daily_path)
                monthly = ds.resample(time="1ME").mean()
                monthly_path = output_dir / f"{file_var}.{year}.monthly.nc"
                monthly.to_netcdf(monthly_path)
                ds.close()
                monthly.close()

                # Remove daily file
                daily_path.unlink()
                logger.info("Wrote monthly means: %s", monthly_path)

    # Build file inventory from monthly files
    all_monthly = sorted(output_dir.glob("*.monthly.nc"))
    existing_timestamps = _existing_file_timestamps(run_dir)
    now_utc = datetime.now(timezone.utc).isoformat()

    files = []
    for p in all_monthly:
        rel = str(p.relative_to(run_dir))
        # Extract year from filename like soilw.0-10cm.gauss.2010.monthly.nc
        year_str = p.stem.split(".")[-2]  # "2010" from "soilw.0-10cm.gauss.2010.monthly"
        files.append({
            "path": rel,
            "year": year_str,
            "size_bytes": p.stat().st_size,
            "downloaded_utc": existing_timestamps.get(year_str, now_utc),
        })

    var_names = [v["file_variable"] for v in meta["variables"]]
    consolidation = consolidate_ncep_ncar(run_dir=run_dir, variables=var_names)

    if files:
        all_years_str = sorted({f["year"] for f in files})
        effective_period = f"{all_years_str[0]}/{all_years_str[-1]}"
    else:
        effective_period = period

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


def _existing_years(run_dir: Path) -> set[int]:
    """Return set of years already fetched from manifest."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text())
        files = manifest.get("sources", {}).get(_SOURCE_KEY, {}).get("files", [])
        return {int(f["year"]) for f in files if "year" in f}
    except (json.JSONDecodeError, KeyError):
        return set()


def _existing_file_timestamps(run_dir: Path) -> dict[str, str]:
    """Return {year: downloaded_utc} from manifest."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text())
        files = manifest.get("sources", {}).get(_SOURCE_KEY, {}).get("files", [])
        return {
            f["year"]: f["downloaded_utc"]
            for f in files
            if "year" in f and "downloaded_utc" in f
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
    """Merge NCEP/NCAR provenance into manifest.json."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"sources": {}, "steps": []}
    if "sources" not in manifest:
        manifest["sources"] = {}

    entry = manifest["sources"].get(_SOURCE_KEY, {})
    entry.update({
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "period": period,
        "bbox": bbox,
        "variables": [v["file_variable"] for v in meta["variables"]],
        "files": files,
        "kerchunk_ref": consolidation["kerchunk_ref"],
        "last_consolidated_utc": consolidation["last_consolidated_utc"],
    })
    manifest["sources"][_SOURCE_KEY] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2))
```

- [ ] **Step 4: Run NCEP/NCAR tests**

Run: `pixi run -e dev test -- tests/test_ncep_ncar.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/nhf_spatial_targets/fetch/ncep_ncar.py tests/test_ncep_ncar.py
git commit -m "feat: implement NCEP/NCAR Reanalysis fetch with daily-to-monthly aggregation"
```

---

### Task 7: Add NCEP/NCAR consolidation function

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py`
- Modify: `tests/test_consolidate.py`

- [ ] **Step 1: Write consolidation tests**

Add to `tests/test_consolidate.py`:

```python
@pytest.fixture()
def ncep_dir(tmp_path: Path) -> Path:
    """Create synthetic NCEP/NCAR monthly NetCDF3 files."""
    out = tmp_path / "data" / "raw" / "ncep_ncar"
    out.mkdir(parents=True)

    lat = np.arange(-90, 91, 45.0)
    lon = np.arange(-180, 180, 60.0)

    for month in range(1, 4):
        time = np.array(
            [f"2010-{month:02d}-15T00:00:00"], dtype="datetime64[ns]"
        )
        ds = xr.Dataset(
            {
                "soilw": (["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32)),
                "EXTRA": (["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32)),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )
        fname = f"soilw.0-10cm.gauss.2010.monthly.nc"
        # Use separate files per month for consolidation
        fname = f"soilw.0-10cm.gauss.2010-{month:02d}.monthly.nc"
        ds.to_netcdf(out / fname, format="NETCDF3_CLASSIC")

    return out


def test_ncep_filter_variables(ncep_dir):
    from nhf_spatial_targets.fetch.consolidate import consolidate_ncep_ncar
    import fsspec

    run_dir = ncep_dir.parent.parent.parent
    consolidate_ncep_ncar(run_dir=run_dir, variables=["soilw"])

    ref_path = ncep_dir / "ncep_ncar_refs.json"
    assert ref_path.exists()

    fs = fsspec.filesystem("reference", fo=str(ref_path), target_protocol="file")
    ds = xr.open_zarr(fs.get_mapper(""), consolidated=False)
    assert "soilw" in ds.data_vars
    assert "EXTRA" not in ds.data_vars
    ds.close()


def test_ncep_provenance_return(ncep_dir):
    from nhf_spatial_targets.fetch.consolidate import consolidate_ncep_ncar

    run_dir = ncep_dir.parent.parent.parent
    result = consolidate_ncep_ncar(run_dir=run_dir, variables=["soilw"])
    assert result["kerchunk_ref"] == "data/raw/ncep_ncar/ncep_ncar_refs.json"
    assert result["n_files"] == 3


def test_ncep_no_files_raises(tmp_path):
    from nhf_spatial_targets.fetch.consolidate import consolidate_ncep_ncar

    (tmp_path / "data" / "raw" / "ncep_ncar").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        consolidate_ncep_ncar(run_dir=tmp_path, variables=["soilw"])
```

- [ ] **Step 2: Implement `consolidate_ncep_ncar` in consolidate.py**

```python
def consolidate_ncep_ncar(
    run_dir: Path,
    variables: list[str],
) -> dict:
    """Build a Kerchunk JSON reference store for NCEP/NCAR monthly files.

    Uses NetCDF3ToZarr since NCEP/NCAR Reanalysis files are NetCDF-3 classic.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory containing ``data/raw/ncep_ncar/*.monthly.nc``.
    variables : list[str]
        Variable names to include (file_variable values).

    Returns
    -------
    dict
        Provenance record.
    """
    from datetime import datetime, timezone

    from kerchunk.combine import MultiZarrToZarr
    from kerchunk.netCDF3 import NetCDF3ToZarr

    ncep_dir = run_dir / "data" / "raw" / "ncep_ncar"
    nc_files = sorted(ncep_dir.glob("*.monthly.nc"))

    if not nc_files:
        raise FileNotFoundError(
            f"No .monthly.nc files found in {ncep_dir}. "
            "Run 'nhf-targets fetch ncep-ncar' first."
        )

    logger.info("Scanning %d monthly NetCDF files for NCEP/NCAR", len(nc_files))

    keep_vars = set(variables)
    singles = []
    for nc in nc_files:
        refs = NetCDF3ToZarr(str(nc)).translate()
        refs["refs"] = _filter_refs(refs["refs"], keep_vars)
        singles.append(refs)

    mzz = MultiZarrToZarr(
        singles,
        concat_dims=["time"],
        identical_dims=["lat", "lon"],
        coo_map={"time": "cf:time"},
    )
    combined = mzz.translate()

    combined["refs"] = _make_relative(combined["refs"], ncep_dir)

    ref_path = ncep_dir / "ncep_ncar_refs.json"
    ref_path.write_text(ujson.dumps(combined, indent=2))
    logger.info("Wrote Kerchunk reference store: %s", ref_path)

    return {
        "kerchunk_ref": str(ref_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(nc_files),
        "variables": variables,
    }
```

- [ ] **Step 3: Run consolidation tests**

Run: `pixi run -e dev test -- tests/test_consolidate.py -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate.py
git commit -m "feat: add NCEP/NCAR Kerchunk consolidation (NetCDF3)"
```

---

## Chunk 4: CLI Integration and Final Verification

### Task 8: Add CLI fetch subcommands

**Files:**
- Modify: `src/nhf_spatial_targets/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add CLI tests**

Add to `tests/test_cli.py` — test that the three new commands exist and validate input:

```python
def test_fetch_nldas_mosaic_nonexistent_run_dir(capsys):
    from nhf_spatial_targets.cli import main
    with pytest.raises(SystemExit, match="2"):
        main(["fetch", "nldas-mosaic", "--run-dir", "/no/such/dir", "--period", "2010/2010"])


def test_fetch_nldas_noah_nonexistent_run_dir(capsys):
    from nhf_spatial_targets.cli import main
    with pytest.raises(SystemExit, match="2"):
        main(["fetch", "nldas-noah", "--run-dir", "/no/such/dir", "--period", "2010/2010"])


def test_fetch_ncep_ncar_nonexistent_run_dir(capsys):
    from nhf_spatial_targets.cli import main
    with pytest.raises(SystemExit, match="2"):
        main(["fetch", "ncep-ncar", "--run-dir", "/no/such/dir", "--period", "2010/2010"])
```

- [ ] **Step 2: Add CLI commands**

Add to `src/nhf_spatial_targets/cli.py` after the `fetch_merra2_cmd` function:

```python
@fetch_app.command(name="nldas-mosaic")
def fetch_nldas_mosaic_cmd(
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
    """Download NLDAS-2 MOSAIC monthly soil moisture (NLDAS_MOS0125_M)."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching NLDAS-2 MOSAIC for period {period}...[/bold]")

    try:
        result = fetch_nldas_mosaic(run_dir=run_dir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    console.print(
        f"[green]Downloaded {len(result['files'])} files "
        f"to {run_dir / 'data' / 'raw' / 'nldas_mosaic'}[/green]"
    )
    if "kerchunk_ref" in result:
        console.print(
            f"[green]Kerchunk reference store: {run_dir / result['kerchunk_ref']}[/green]"
        )
    console.print(json_mod.dumps(result, indent=2))


@fetch_app.command(name="nldas-noah")
def fetch_nldas_noah_cmd(
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
    """Download NLDAS-2 NOAH monthly soil moisture (NLDAS_NOAH0125_M)."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.nldas import fetch_nldas_noah

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching NLDAS-2 NOAH for period {period}...[/bold]")

    try:
        result = fetch_nldas_noah(run_dir=run_dir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    console.print(
        f"[green]Downloaded {len(result['files'])} files "
        f"to {run_dir / 'data' / 'raw' / 'nldas_noah'}[/green]"
    )
    if "kerchunk_ref" in result:
        console.print(
            f"[green]Kerchunk reference store: {run_dir / result['kerchunk_ref']}[/green]"
        )
    console.print(json_mod.dumps(result, indent=2))


@fetch_app.command(name="ncep-ncar")
def fetch_ncep_ncar_cmd(
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
    """Download NCEP/NCAR Reanalysis soil moisture from NOAA PSL."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.ncep_ncar import fetch_ncep_ncar

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching NCEP/NCAR Reanalysis for period {period}...[/bold]")

    try:
        result = fetch_ncep_ncar(run_dir=run_dir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    console.print(
        f"[green]Downloaded {len(result['files'])} files "
        f"to {run_dir / 'data' / 'raw' / 'ncep_ncar'}[/green]"
    )
    if "kerchunk_ref" in result:
        console.print(
            f"[green]Kerchunk reference store: {run_dir / result['kerchunk_ref']}[/green]"
        )
    console.print(json_mod.dumps(result, indent=2))
```

- [ ] **Step 3: Run CLI tests**

Run: `pixi run -e dev test -- tests/test_cli.py -v`
Expected: All tests PASS.

- [ ] **Step 4: Run full test suite and quality checks**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All tests pass, no lint errors.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/cli.py tests/test_cli.py
git commit -m "feat: add CLI commands for NLDAS and NCEP/NCAR fetch"
```

---

### Task 9: Final verification and cleanup

- [ ] **Step 1: Run full test suite one final time**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All tests pass.

- [ ] **Step 2: Verify all new modules have `from __future__ import annotations`**

Check: `_auth.py`, `_period.py`, `nldas.py`, `ncep_ncar.py`, all test files.

- [ ] **Step 3: Verify no hardcoded URLs or variable names in Python modules**

All source metadata should come from `_catalog.source()` calls.

- [ ] **Step 4: Create issue and PR branch**

```bash
gh issue create --title "Implement NLDAS and NCEP/NCAR soil moisture fetch modules" --body "..."
git checkout -b feature/<issue#>-soil-moisture-fetch-modules
# Cherry-pick or squash commits from working branch
git push -u origin feature/<issue#>-soil-moisture-fetch-modules
gh pr create --title "Add NLDAS and NCEP/NCAR soil moisture fetch modules" --body "..."
```
