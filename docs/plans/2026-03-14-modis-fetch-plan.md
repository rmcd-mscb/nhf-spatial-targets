# MODIS Fetch Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `fetch_mod16a2()` and `fetch_mod10c1()` to download MODIS AET and snow cover data via earthaccess, with per-year consolidation and MOD10C1 CONUS subsetting.

**Architecture:** Two independent fetch functions in `modis.py`, each following the MERRA-2 pattern (catalog lookup, fabric.json bbox, earthaccess search/download, year-based incremental tracking, per-year consolidation, manifest update). MOD10C1 subsets global files to CONUS on the fly to stay within 32 GB RAM.

**Tech Stack:** earthaccess, xarray, rioxarray, cyclopts, pytest

**Design doc:** `docs/plans/2026-03-14-modis-fetch-design.md`

---

## Important Context

### Catalog variable format
The MODIS catalog entries use **plain strings** for variables (e.g., `["ET_500m", "ET_QC_500m"]`), NOT dicts with a `name` key like MERRA-2 (`[{"name": "GWETTOP"}, ...]`). Code that accesses variable names must handle this — use `meta["variables"]` directly, not `[v["name"] for v in meta["variables"]]`.

### Key files to reference
- **Pattern to follow:** `src/nhf_spatial_targets/fetch/merra2.py` (the template for all fetch modules)
- **Consolidation helpers:** `src/nhf_spatial_targets/fetch/consolidate.py` (`_open_datasets`, `_write_netcdf`, `_validate_variables`)
- **Period utilities:** `src/nhf_spatial_targets/fetch/_period.py` (`years_in_period`, `parse_period`)
- **Auth:** `src/nhf_spatial_targets/fetch/_auth.py` (`earthdata_login`)
- **Catalog:** `src/nhf_spatial_targets/catalog.py` (`source()`)
- **CLI pattern:** `src/nhf_spatial_targets/cli.py:212-265` (MERRA-2 fetch command)
- **Test pattern:** `tests/test_merra2.py`
- **Catalog entries:** `catalog/sources.yml:65-79` (mod16a2_v061), `catalog/sources.yml:394-409` (mod10c1_v061)

### MODIS filename patterns
- MOD16A2: `MOD16A2GF.A2005001.h09v05.061.2021238120323.hdf` — `AYYYYDDD` is year + day-of-year
- MOD10C1: `MOD10C1.A2005001.061.2021157180344.hdf` — same `AYYYYDDD` pattern, no tile ID (global CMG)

### Commands
```bash
pixi run -e dev test          # run tests
pixi run -e dev lint          # ruff check
pixi run -e dev fmt           # ruff format
```

---

### Task 1: Update catalog with short_name for MODIS products

**Files:**
- Modify: `catalog/sources.yml:65-79` (mod16a2_v061)
- Modify: `catalog/sources.yml:394-409` (mod10c1_v061)

**Step 1: Add short_name to mod16a2_v061 access block**

In `catalog/sources.yml`, change the `mod16a2_v061` access block from:

```yaml
  mod16a2_v061:
    name: MODIS MOD16A2 v061 (current version)
    description: Updated version of MOD16A2. Use in place of v006.
    access:
      type: lpdaac
      url: https://lpdaac.usgs.gov/products/mod16a2v061/
```

to:

```yaml
  mod16a2_v061:
    name: MODIS MOD16A2 v061 (current version)
    description: Updated version of MOD16A2. Use in place of v006.
    access:
      type: lpdaac
      short_name: MOD16A2GF
      version: "061"
      url: https://lpdaac.usgs.gov/products/mod16a2v061/
```

**Step 2: Add short_name to mod10c1_v061 access block**

Change the `mod10c1_v061` access block from:

```yaml
  mod10c1_v061:
    name: MODIS MOD10C1 v061 (current version)
    description: Updated version of MOD10C1. Use in place of v006.
    access:
      type: nsidc
      url: https://nsidc.org/data/mod10c1/versions/61
```

to:

```yaml
  mod10c1_v061:
    name: MODIS MOD10C1 v061 (current version)
    description: Updated version of MOD10C1. Use in place of v006.
    access:
      type: nsidc
      short_name: MOD10C1
      version: "061"
      url: https://nsidc.org/data/mod10c1/versions/61
```

**Step 3: Verify catalog still loads**

Run: `pixi run -e dev python -c "from nhf_spatial_targets.catalog import source; s = source('mod16a2_v061'); print(s['access']['short_name']); s2 = source('mod10c1_v061'); print(s2['access']['short_name'])"`

Expected: prints `MOD16A2GF` then `MOD10C1`

**Step 4: Run existing tests**

Run: `pixi run -e dev test`

Expected: all existing tests pass (catalog changes don't break anything)

**Step 5: Commit**

```bash
git add catalog/sources.yml
git commit -m "catalog: add short_name and version to MODIS v061 entries"
```

---

### Task 2: Implement year extraction and shared helpers in modis.py

**Files:**
- Create: `src/nhf_spatial_targets/fetch/modis.py`
- Create: `tests/test_modis.py`

**Step 1: Write failing tests for year extraction**

Create `tests/test_modis.py`:

```python
"""Tests for MODIS fetch module."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_year_from_mod16a2_filename():
    """Extract year from MOD16A2 filename with AYYYYDDD pattern."""
    from nhf_spatial_targets.fetch.modis import _year_from_path

    assert _year_from_path(Path("MOD16A2GF.A2005001.h09v05.061.2021238120323.hdf")) == 2005
    assert _year_from_path(Path("MOD16A2GF.A2010361.h08v04.061.2021240012345.hdf")) == 2010


def test_year_from_mod10c1_filename():
    """Extract year from MOD10C1 filename with AYYYYDDD pattern."""
    from nhf_spatial_targets.fetch.modis import _year_from_path

    assert _year_from_path(Path("MOD10C1.A2005001.061.2021157180344.hdf")) == 2005
    assert _year_from_path(Path("MOD10C1.A2014365.061.2021157180344.hdf")) == 2014


def test_year_from_conus_subset_filename():
    """Extract year from CONUS subset NetCDF filename."""
    from nhf_spatial_targets.fetch.modis import _year_from_path

    assert _year_from_path(Path("MOD10C1.A2005001.061.conus.nc")) == 2005


def test_year_from_invalid_filename():
    """ValueError raised for filenames without AYYYYDDD pattern."""
    from nhf_spatial_targets.fetch.modis import _year_from_path

    with pytest.raises(ValueError, match="Cannot extract year"):
        _year_from_path(Path("not_a_modis_file.txt"))
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test tests/test_modis.py -v`

Expected: FAIL — `_year_from_path` not defined

**Step 3: Write modis.py with year extraction and helper functions**

Create `src/nhf_spatial_targets/fetch/modis.py`:

```python
"""Fetch MODIS products: MOD16A2 (AET) and MOD10C1 (SCA) via earthaccess."""

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
from nhf_spatial_targets.fetch._period import parse_period, years_in_period

logger = logging.getLogger(__name__)

_MODIS_YEAR_RE = re.compile(r"\.A(\d{4})\d{3}\.")


def _year_from_path(path: Path) -> int:
    """Extract year from a MODIS filename containing AYYYYDDD pattern."""
    m = _MODIS_YEAR_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot extract year from MODIS filename: {path.name}")
    return int(m.group(1))


def _read_fabric_bbox(run_dir: Path) -> dict:
    """Read bbox_buffered from fabric.json, return the dict."""
    fabric_path = run_dir / "fabric.json"
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"fabric.json not found in {run_dir}. "
            f"Run 'nhf-targets init' to create a run workspace first."
        )
    try:
        fabric = json.loads(fabric_path.read_text())
        bbox = fabric["bbox_buffered"]
        # Validate required keys exist
        _ = (bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"])
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(
            f"fabric.json in {run_dir} is malformed or missing required "
            f"fields (bbox_buffered.{{minx,miny,maxx,maxy}}). "
            f"Re-run 'nhf-targets init' to regenerate it."
        ) from exc
    return bbox


def _bbox_tuple(bbox: dict) -> tuple[float, float, float, float]:
    """Convert bbox dict to (minx, miny, maxx, maxy) tuple."""
    return (bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"])


def _manifest_source_files(run_dir: Path, source_key: str) -> list[dict]:
    """Read manifest.json and return file records for a source."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manifest.json in {run_dir} is corrupted and cannot be parsed. "
            f"Inspect the file manually or restore from backup. Detail: {exc}"
        ) from exc
    return manifest.get("sources", {}).get(source_key, {}).get("files", [])


def _existing_years(run_dir: Path, source_key: str) -> set[int]:
    """Return set of years already fetched from manifest."""
    return {
        f["year"] for f in _manifest_source_files(run_dir, source_key) if "year" in f
    }


def _existing_file_timestamps(run_dir: Path, source_key: str) -> dict[int, str]:
    """Return {year: downloaded_utc} from existing manifest."""
    return {
        f["year"]: f["downloaded_utc"]
        for f in _manifest_source_files(run_dir, source_key)
        if "year" in f and "downloaded_utc" in f
    }


def _check_superseded(meta: dict, source_key: str) -> None:
    """Emit DeprecationWarning if source is superseded."""
    if meta.get("status") == "superseded":
        warnings.warn(
            f"Source '{source_key}' is superseded. "
            f"Consider using '{meta.get('superseded_by', 'unknown')}'.",
            DeprecationWarning,
            stacklevel=3,
        )


def _update_manifest(
    run_dir: Path,
    source_key: str,
    period: str,
    bbox: dict,
    meta: dict,
    files: list[dict],
    consolidated_ncs: dict[str, str],
) -> None:
    """Merge MODIS provenance into manifest.json."""
    import os
    import tempfile

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"sources": {}, "steps": []}

    if "sources" not in manifest:
        manifest["sources"] = {}

    entry = manifest["sources"].get(source_key, {})
    entry.update(
        {
            "source_key": source_key,
            "access_url": meta["access"]["url"],
            "period": period,
            "bbox": bbox,
            "variables": meta["variables"],
            "files": files,
            "consolidated_ncs": consolidated_ncs,
            "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    manifest["sources"][source_key] = entry

    tmp_fd, tmp_path = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp_path).replace(manifest_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    logger.info("Updated manifest.json with %s provenance", source_key)
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev test tests/test_modis.py -v`

Expected: all 4 tests PASS

**Step 5: Run lint and format**

Run: `pixi run -e dev fmt && pixi run -e dev lint`

Expected: no errors

**Step 6: Commit**

```bash
git add src/nhf_spatial_targets/fetch/modis.py tests/test_modis.py
git commit -m "feat: add MODIS year extraction and shared fetch helpers"
```

---

### Task 3: Implement fetch_mod16a2

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/modis.py`
- Modify: `tests/test_modis.py`

**Step 1: Write failing tests for fetch_mod16a2**

Add to `tests/test_modis.py`:

```python
import json
import warnings
from unittest.mock import MagicMock, patch

_SOURCE_KEY_16 = "mod16a2_v061"

_MOCK_CONSOLIDATION_16 = {
    "consolidated_nc": "data/raw/mod16a2_v061/mod16a2_v061_2005.nc",
    "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
    "n_files": 1,
    "variables": ["ET_500m", "ET_QC_500m"],
}


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """Create a minimal run workspace with fabric.json."""
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / _SOURCE_KEY_16).mkdir(parents=True)
    (rd / "data" / "raw" / "mod10c1_v061").mkdir(parents=True)
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


def _mock_granule_mod16(name: str, year: int = 2005, doy: int = 1) -> MagicMock:
    """Create a mock granule with MOD16A2 filename pattern."""
    g = MagicMock()
    fname = f"MOD16A2GF.A{year}{doy:03d}.h09v05.061.2021238120323.hdf"
    g.__str__ = lambda self: fname
    g.data_links.return_value = [f"https://example.com/{fname}"]
    return g


def _fake_download_mod16(run_dir: Path, year: int = 2005, n: int = 1) -> list[str]:
    """Create fake MOD16A2 downloaded files."""
    paths = []
    for i in range(n):
        doy = i * 8 + 1
        f = (
            run_dir / "data" / "raw" / _SOURCE_KEY_16
            / f"MOD16A2GF.A{year}{doy:03d}.h09v05.061.2021238120323.hdf"
        )
        f.write_bytes(b"fake")
        paths.append(str(f))
    return paths


# ---- fetch_mod16a2 tests ---------------------------------------------------


@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_login_called(mock_login, run_dir):
    """earthdata_login() is called before searching."""
    with patch("earthaccess.search_data", return_value=[]):
        with pytest.raises(ValueError, match="No granules found"):
            from nhf_spatial_targets.fetch.modis import fetch_mod16a2

            fetch_mod16a2(run_dir=run_dir, period="2005/2005")
    mock_login.assert_called_once_with(run_dir)


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2",
    return_value=_MOCK_CONSOLIDATION_16,
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_search_params(mock_login, mock_search, mock_dl, mock_consolidate, run_dir):
    """search_data called with correct short_name, bbox, temporal for each year."""
    mock_search.return_value = [_mock_granule_mod16("g1")]
    mock_dl.return_value = _fake_download_mod16(run_dir)

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    fetch_mod16a2(run_dir=run_dir, period="2005/2005")

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["short_name"] == "MOD16A2GF"
    assert call_kwargs["bounding_box"] == (-125.1, 23.9, -65.9, 50.1)
    assert call_kwargs["temporal"] == ("2005-01-01", "2005-12-31")


@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
@patch("earthaccess.search_data", return_value=[])
def test_mod16a2_no_granules_raises(mock_search, mock_login, run_dir):
    """ValueError raised when search returns zero granules."""
    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    with pytest.raises(ValueError, match="No granules found"):
        fetch_mod16a2(run_dir=run_dir, period="2005/2005")


@patch("earthaccess.download", return_value=[])
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_empty_download_raises(mock_login, mock_search, mock_dl, run_dir):
    """RuntimeError raised when download returns no files."""
    mock_search.return_value = [_mock_granule_mod16("g1")]

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    with pytest.raises(RuntimeError, match="returned no files"):
        fetch_mod16a2(run_dir=run_dir, period="2005/2005")


@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_missing_fabric_raises(mock_login, tmp_path):
    """FileNotFoundError raised when fabric.json is missing."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    with pytest.raises(FileNotFoundError, match="fabric.json"):
        fetch_mod16a2(run_dir=run_dir, period="2005/2005")


@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_malformed_fabric_raises(mock_login, tmp_path):
    """ValueError raised when fabric.json is malformed."""
    run_dir = tmp_path / "bad_run"
    run_dir.mkdir()
    (run_dir / "fabric.json").write_text("{}")

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    with pytest.raises(ValueError, match="malformed"):
        fetch_mod16a2(run_dir=run_dir, period="2005/2005")


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2",
    return_value=_MOCK_CONSOLIDATION_16,
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_provenance_record(mock_login, mock_search, mock_dl, mock_consolidate, run_dir):
    """Returned dict has all required provenance keys."""
    mock_search.return_value = [_mock_granule_mod16("g1")]
    mock_dl.return_value = _fake_download_mod16(run_dir)

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    result = fetch_mod16a2(run_dir=run_dir, period="2005/2005")

    assert result["source_key"] == _SOURCE_KEY_16
    assert "access_url" in result
    assert result["variables"] == ["ET_500m", "ET_QC_500m"]
    assert result["period"] == "2005/2005"
    assert "bbox" in result
    assert "download_timestamp" in result
    assert isinstance(result["files"], list)
    assert "consolidated_ncs" in result


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2",
    return_value=_MOCK_CONSOLIDATION_16,
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_manifest_updated(mock_login, mock_search, mock_dl, mock_consolidate, run_dir):
    """fetch_mod16a2 writes provenance to manifest.json."""
    mock_search.return_value = [_mock_granule_mod16("g1")]
    mock_dl.return_value = _fake_download_mod16(run_dir)

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    fetch_mod16a2(run_dir=run_dir, period="2005/2005")

    manifest = json.loads((run_dir / "manifest.json").read_text())
    entry = manifest["sources"][_SOURCE_KEY_16]
    assert entry["period"] == "2005/2005"
    assert len(entry["files"]) > 0
    assert "year" in entry["files"][0]
    assert "consolidated_ncs" in entry


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2",
    return_value=_MOCK_CONSOLIDATION_16,
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_incremental_skips_year(mock_login, mock_search, mock_dl, mock_consolidate, run_dir):
    """Years already in manifest are not re-downloaded."""
    # Pre-populate manifest with 2005 already downloaded
    manifest = {
        "sources": {
            _SOURCE_KEY_16: {
                "files": [
                    {
                        "path": "data/raw/mod16a2_v061/MOD16A2GF.A2005001.h09v05.061.hdf",
                        "year": 2005,
                        "size_bytes": 100,
                        "downloaded_utc": "2026-01-01T00:00:00+00:00",
                    }
                ],
            }
        }
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    # Create the file on disk
    f = run_dir / "data" / "raw" / _SOURCE_KEY_16 / "MOD16A2GF.A2005001.h09v05.061.hdf"
    f.write_bytes(b"fake")

    from nhf_spatial_targets.fetch.modis import _existing_years

    assert 2005 in _existing_years(run_dir, _SOURCE_KEY_16)
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test tests/test_modis.py -v`

Expected: FAIL — `fetch_mod16a2` and `consolidate_mod16a2` not defined

**Step 3: Implement fetch_mod16a2**

Add to `src/nhf_spatial_targets/fetch/modis.py`:

```python
from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2


def fetch_mod16a2(run_dir: Path, period: str) -> dict:
    """Download MOD16A2 v061 AET tiles for CONUS for the given period.

    Supports incremental download — years already recorded in
    ``manifest.json`` are skipped. After downloading, consolidates
    each year into a single NetCDF and updates the manifest.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory. Reads ``fabric.json`` for bbox,
        writes files to ``data/raw/mod16a2_v061/``.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    source_key = "mod16a2_v061"
    meta = _catalog.source(source_key)
    short_name = meta["access"]["short_name"]

    _check_superseded(meta, source_key)
    earthdata_login(run_dir)
    logger.info("Authenticated with NASA Earthdata")

    bbox = _read_fabric_bbox(run_dir)
    bt = _bbox_tuple(bbox)

    all_years = years_in_period(period)
    already_have = _existing_years(run_dir, source_key)
    needed = [y for y in all_years if y not in already_have]

    output_dir = run_dir / "data" / "raw" / source_key
    output_dir.mkdir(parents=True, exist_ok=True)

    if not needed:
        logger.info(
            "All %d years already downloaded, skipping to consolidation",
            len(all_years),
        )
    else:
        for year in needed:
            temporal = (f"{year}-01-01", f"{year}-12-31")
            logger.info("Searching %s for year %d", short_name, year)

            granules = earthaccess.search_data(
                short_name=short_name,
                bounding_box=bt,
                temporal=temporal,
            )
            logger.info("Found %d granules for %s year %d", len(granules), short_name, year)

            if not granules:
                raise ValueError(
                    f"No granules found for {short_name} with "
                    f"bbox={bt}, temporal={temporal}"
                )

            downloaded = earthaccess.download(
                granules,
                local_path=str(output_dir),
            )

            if not downloaded:
                raise RuntimeError(
                    f"earthaccess.download() returned no files for "
                    f"{len(granules)} granules (year {year}). Check network "
                    f"connectivity and Earthdata credentials."
                )
            if len(downloaded) < len(granules):
                logger.warning(
                    "Partial download for year %d: got %d of %d granules.",
                    year,
                    len(downloaded),
                    len(granules),
                )
            logger.info("Downloaded %d files for year %d", len(downloaded), year)

    # Build file inventory from all HDF files on disk
    all_hdf = sorted(output_dir.glob("*.hdf"))

    existing_timestamps = _existing_file_timestamps(run_dir, source_key)
    now_utc = datetime.now(timezone.utc).isoformat()

    files = []
    for p in all_hdf:
        rel = str(p.relative_to(run_dir))
        year = _year_from_path(p)
        files.append(
            {
                "path": rel,
                "year": year,
                "size_bytes": p.stat().st_size,
                "downloaded_utc": existing_timestamps.get(year, now_utc),
            }
        )

    # Consolidate per year
    consolidated_ncs = {}
    for year in all_years:
        year_files = [p for p in all_hdf if _year_from_path(p) == year]
        if year_files:
            result = consolidate_mod16a2(
                run_dir=run_dir,
                source_key=source_key,
                variables=meta["variables"],
                year=year,
            )
            consolidated_ncs[str(year)] = result["consolidated_nc"]

    # Compute effective period from files on disk
    if files:
        all_file_years = sorted({f["year"] for f in files})
        effective_period = f"{all_file_years[0]}/{all_file_years[-1]}"
    else:
        effective_period = period

    _update_manifest(run_dir, source_key, effective_period, bbox, meta, files, consolidated_ncs)

    return {
        "source_key": source_key,
        "access_url": meta["access"]["url"],
        "variables": meta["variables"],
        "period": effective_period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "files": files,
        "consolidated_ncs": consolidated_ncs,
    }
```

**Step 4: Add consolidate_mod16a2 stub to consolidate.py**

Add to `src/nhf_spatial_targets/fetch/consolidate.py`:

```python
def consolidate_mod16a2(
    run_dir: Path,
    source_key: str,
    variables: list[str],
    year: int,
) -> dict:
    """Mosaic and merge MOD16A2 tiles for a single year into one NetCDF.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory.
    source_key : str
        Source key (e.g. "mod16a2_v061").
    variables : list[str]
        Variable names to include (e.g. ["ET_500m", "ET_QC_500m"]).
    year : int
        Year to consolidate.

    Returns
    -------
    dict
        Provenance record with consolidated file path and timestamp.
    """
    raise NotImplementedError("consolidate_mod16a2 not yet implemented")
```

**Step 5: Run tests to verify they pass**

Run: `pixi run -e dev test tests/test_modis.py -v`

Expected: all tests PASS (consolidation is mocked in unit tests)

**Step 6: Run lint and format**

Run: `pixi run -e dev fmt && pixi run -e dev lint`

**Step 7: Commit**

```bash
git add src/nhf_spatial_targets/fetch/modis.py src/nhf_spatial_targets/fetch/consolidate.py tests/test_modis.py
git commit -m "feat: implement fetch_mod16a2 with year-based incremental download"
```

---

### Task 4: Implement fetch_mod10c1 with CONUS subsetting

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/modis.py`
- Modify: `tests/test_modis.py`

**Step 1: Write failing tests for fetch_mod10c1**

Add to `tests/test_modis.py`:

```python
_SOURCE_KEY_10 = "mod10c1_v061"

_MOCK_CONSOLIDATION_10 = {
    "consolidated_nc": "data/raw/mod10c1_v061/mod10c1_v061_2005.nc",
    "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
    "n_files": 1,
    "variables": ["Day_CMG_Snow_Cover", "Snow_Spatial_QA"],
}


def _mock_granule_mod10(name: str, year: int = 2005, doy: int = 1) -> MagicMock:
    """Create a mock granule with MOD10C1 filename pattern."""
    g = MagicMock()
    fname = f"MOD10C1.A{year}{doy:03d}.061.2021157180344.hdf"
    g.__str__ = lambda self: fname
    g.data_links.return_value = [f"https://example.com/{fname}"]
    return g


@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod10c1_login_called(mock_login, run_dir):
    """earthdata_login() is called before searching."""
    with patch("earthaccess.search_data", return_value=[]):
        with pytest.raises(ValueError, match="No granules found"):
            from nhf_spatial_targets.fetch.modis import fetch_mod10c1

            fetch_mod10c1(run_dir=run_dir, period="2005/2005")
    mock_login.assert_called_once_with(run_dir)


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod10c1",
    return_value=_MOCK_CONSOLIDATION_10,
)
@patch("nhf_spatial_targets.fetch.modis._subset_to_conus")
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod10c1_search_params(mock_login, mock_search, mock_dl, mock_subset, mock_consolidate, run_dir):
    """search_data called with correct short_name for MOD10C1."""
    mock_search.return_value = [_mock_granule_mod10("g1")]
    # download returns fake HDF paths; subset converts them to .conus.nc
    dl_paths = [str(run_dir / "data" / "raw" / _SOURCE_KEY_10 / "MOD10C1.A2005001.061.2021157180344.hdf")]
    for p in dl_paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_bytes(b"fake")
    mock_dl.return_value = dl_paths
    conus_path = Path(dl_paths[0]).with_suffix(".conus.nc")
    conus_path.write_bytes(b"fake")
    mock_subset.return_value = conus_path

    from nhf_spatial_targets.fetch.modis import fetch_mod10c1

    fetch_mod10c1(run_dir=run_dir, period="2005/2005")

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["short_name"] == "MOD10C1"


@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
@patch("earthaccess.search_data", return_value=[])
def test_mod10c1_no_granules_raises(mock_search, mock_login, run_dir):
    """ValueError raised when search returns zero granules."""
    from nhf_spatial_targets.fetch.modis import fetch_mod10c1

    with pytest.raises(ValueError, match="No granules found"):
        fetch_mod10c1(run_dir=run_dir, period="2005/2005")


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod10c1",
    return_value=_MOCK_CONSOLIDATION_10,
)
@patch("nhf_spatial_targets.fetch.modis._subset_to_conus")
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod10c1_provenance_record(mock_login, mock_search, mock_dl, mock_subset, mock_consolidate, run_dir):
    """Returned dict has all required provenance keys."""
    mock_search.return_value = [_mock_granule_mod10("g1")]
    dl_paths = [str(run_dir / "data" / "raw" / _SOURCE_KEY_10 / "MOD10C1.A2005001.061.2021157180344.hdf")]
    for p in dl_paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_bytes(b"fake")
    mock_dl.return_value = dl_paths
    conus_path = Path(dl_paths[0]).with_suffix(".conus.nc")
    conus_path.write_bytes(b"fake")
    mock_subset.return_value = conus_path

    from nhf_spatial_targets.fetch.modis import fetch_mod10c1

    result = fetch_mod10c1(run_dir=run_dir, period="2005/2005")

    assert result["source_key"] == _SOURCE_KEY_10
    assert "access_url" in result
    assert result["variables"] == ["Day_CMG_Snow_Cover", "Snow_Spatial_QA"]
    assert "consolidated_ncs" in result


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod10c1",
    return_value=_MOCK_CONSOLIDATION_10,
)
@patch("nhf_spatial_targets.fetch.modis._subset_to_conus")
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod10c1_subset_called(mock_login, mock_search, mock_dl, mock_subset, mock_consolidate, run_dir):
    """_subset_to_conus is called for each downloaded HDF file."""
    mock_search.return_value = [_mock_granule_mod10("g1")]
    hdf_path = run_dir / "data" / "raw" / _SOURCE_KEY_10 / "MOD10C1.A2005001.061.2021157180344.hdf"
    hdf_path.parent.mkdir(parents=True, exist_ok=True)
    hdf_path.write_bytes(b"fake")
    mock_dl.return_value = [str(hdf_path)]
    conus_path = hdf_path.with_suffix(".conus.nc")
    conus_path.write_bytes(b"fake")
    mock_subset.return_value = conus_path

    from nhf_spatial_targets.fetch.modis import fetch_mod10c1

    fetch_mod10c1(run_dir=run_dir, period="2005/2005")

    mock_subset.assert_called_once()
    call_args = mock_subset.call_args
    assert str(call_args[0][0]).endswith(".hdf")
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test tests/test_modis.py -v -k mod10c1`

Expected: FAIL — `fetch_mod10c1`, `_subset_to_conus`, `consolidate_mod10c1` not defined

**Step 3: Implement _subset_to_conus and fetch_mod10c1**

Add to `src/nhf_spatial_targets/fetch/modis.py`:

```python
import xarray as xr

from nhf_spatial_targets.fetch.consolidate import consolidate_mod10c1

# CONUS bounding box for subsetting global CMG files
_CONUS_BBOX = {
    "minx": -129.4134,
    "miny": 22.3380,
    "maxx": -63.2790,
    "maxy": 54.6729,
}


def _subset_to_conus(hdf_path: Path, bbox: dict | None = None) -> Path:
    """Open a global MOD10C1 HDF file, subset to CONUS, save as NetCDF.

    Writes a ``.conus.nc`` file alongside the original and deletes the
    original HDF to save disk space.

    Parameters
    ----------
    hdf_path : Path
        Path to the global MOD10C1 HDF file.
    bbox : dict, optional
        Bounding box dict with minx/miny/maxx/maxy.
        Defaults to CONUS bbox.

    Returns
    -------
    Path
        Path to the CONUS-subset NetCDF file.
    """
    if bbox is None:
        bbox = _CONUS_BBOX

    ds = xr.open_dataset(hdf_path)
    # MOD10C1 CMG uses lat/lon coordinates
    ds_conus = ds.sel(
        lat=slice(bbox["maxy"], bbox["miny"]),
        lon=slice(bbox["minx"], bbox["maxx"]),
    )
    out_path = hdf_path.with_suffix(".conus.nc")
    ds_conus.to_netcdf(out_path)
    ds.close()
    ds_conus.close()

    # Delete original global HDF to save disk space
    hdf_path.unlink()
    logger.debug("Subset %s -> %s, deleted original", hdf_path.name, out_path.name)

    return out_path


def fetch_mod10c1(run_dir: Path, period: str) -> dict:
    """Download MOD10C1 v061 daily snow cover for the given period.

    Downloads global CMG files, subsets each to CONUS, and consolidates
    per year. Supports incremental download — years already recorded in
    ``manifest.json`` are skipped.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory. Reads ``fabric.json`` for bbox,
        writes files to ``data/raw/mod10c1_v061/``.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    source_key = "mod10c1_v061"
    meta = _catalog.source(source_key)
    short_name = meta["access"]["short_name"]

    _check_superseded(meta, source_key)
    earthdata_login(run_dir)
    logger.info("Authenticated with NASA Earthdata")

    bbox = _read_fabric_bbox(run_dir)
    bt = _bbox_tuple(bbox)

    all_years = years_in_period(period)
    already_have = _existing_years(run_dir, source_key)
    needed = [y for y in all_years if y not in already_have]

    output_dir = run_dir / "data" / "raw" / source_key
    output_dir.mkdir(parents=True, exist_ok=True)

    if not needed:
        logger.info(
            "All %d years already downloaded, skipping to consolidation",
            len(all_years),
        )
    else:
        for year in needed:
            temporal = (f"{year}-01-01", f"{year}-12-31")
            logger.info("Searching %s for year %d", short_name, year)

            granules = earthaccess.search_data(
                short_name=short_name,
                bounding_box=bt,
                temporal=temporal,
            )
            logger.info("Found %d granules for %s year %d", len(granules), short_name, year)

            if not granules:
                raise ValueError(
                    f"No granules found for {short_name} with "
                    f"bbox={bt}, temporal={temporal}"
                )

            downloaded = earthaccess.download(
                granules,
                local_path=str(output_dir),
            )

            if not downloaded:
                raise RuntimeError(
                    f"earthaccess.download() returned no files for "
                    f"{len(granules)} granules (year {year}). Check network "
                    f"connectivity and Earthdata credentials."
                )
            if len(downloaded) < len(granules):
                logger.warning(
                    "Partial download for year %d: got %d of %d granules.",
                    year,
                    len(downloaded),
                    len(granules),
                )
            logger.info("Downloaded %d files for year %d", len(downloaded), year)

            # Subset each global HDF to CONUS
            for dl_path in downloaded:
                p = Path(dl_path)
                if p.suffix == ".hdf" and p.exists():
                    _subset_to_conus(p)

    # Build file inventory from CONUS subset files on disk
    all_nc = sorted(output_dir.glob("*.conus.nc"))

    existing_timestamps = _existing_file_timestamps(run_dir, source_key)
    now_utc = datetime.now(timezone.utc).isoformat()

    files = []
    for p in all_nc:
        rel = str(p.relative_to(run_dir))
        year = _year_from_path(p)
        files.append(
            {
                "path": rel,
                "year": year,
                "size_bytes": p.stat().st_size,
                "downloaded_utc": existing_timestamps.get(year, now_utc),
            }
        )

    # Consolidate per year
    consolidated_ncs = {}
    for year in all_years:
        year_files = [p for p in all_nc if _year_from_path(p) == year]
        if year_files:
            result = consolidate_mod10c1(
                run_dir=run_dir,
                source_key=source_key,
                variables=meta["variables"],
                year=year,
            )
            consolidated_ncs[str(year)] = result["consolidated_nc"]

    # Compute effective period
    if files:
        all_file_years = sorted({f["year"] for f in files})
        effective_period = f"{all_file_years[0]}/{all_file_years[-1]}"
    else:
        effective_period = period

    _update_manifest(run_dir, source_key, effective_period, bbox, meta, files, consolidated_ncs)

    return {
        "source_key": source_key,
        "access_url": meta["access"]["url"],
        "variables": meta["variables"],
        "period": effective_period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "files": files,
        "consolidated_ncs": consolidated_ncs,
    }
```

**Step 4: Add consolidate_mod10c1 stub to consolidate.py**

Add to `src/nhf_spatial_targets/fetch/consolidate.py`:

```python
def consolidate_mod10c1(
    run_dir: Path,
    source_key: str,
    variables: list[str],
    year: int,
) -> dict:
    """Merge daily MOD10C1 CONUS subsets for a single year into one NetCDF.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory.
    source_key : str
        Source key (e.g. "mod10c1_v061").
    variables : list[str]
        Variable names to include.
    year : int
        Year to consolidate.

    Returns
    -------
    dict
        Provenance record with consolidated file path and timestamp.
    """
    raise NotImplementedError("consolidate_mod10c1 not yet implemented")
```

**Step 5: Run tests to verify they pass**

Run: `pixi run -e dev test tests/test_modis.py -v`

Expected: all tests PASS

**Step 6: Run lint and format**

Run: `pixi run -e dev fmt && pixi run -e dev lint`

**Step 7: Commit**

```bash
git add src/nhf_spatial_targets/fetch/modis.py src/nhf_spatial_targets/fetch/consolidate.py tests/test_modis.py
git commit -m "feat: implement fetch_mod10c1 with CONUS subsetting"
```

---

### Task 5: Add CLI commands for MODIS fetch

**Files:**
- Modify: `src/nhf_spatial_targets/cli.py`

**Step 1: Write failing test for CLI commands**

Add to `tests/test_modis.py`:

```python
def test_cli_mod16a2_command_exists():
    """The mod16a2 fetch subcommand is registered."""
    from nhf_spatial_targets.cli import fetch_app

    command_names = [cmd.name for cmd in fetch_app._commands.values()]
    assert "mod16a2" in command_names


def test_cli_mod10c1_command_exists():
    """The mod10c1 fetch subcommand is registered."""
    from nhf_spatial_targets.cli import fetch_app

    command_names = [cmd.name for cmd in fetch_app._commands.values()]
    assert "mod10c1" in command_names
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test tests/test_modis.py::test_cli_mod16a2_command_exists tests/test_modis.py::test_cli_mod10c1_command_exists -v`

Expected: FAIL — commands not registered

**Step 3: Add CLI commands to cli.py**

Add after the `fetch_ncep_ncar_cmd` function (around line 433) in `src/nhf_spatial_targets/cli.py`:

```python
@fetch_app.command(name="mod16a2")
def fetch_mod16a2_cmd(
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
    """Download MODIS MOD16A2 v061 AET data (8-day composites, 500m).

    Authenticates via earthaccess, searches for tiles matching the
    fabric bounding box, downloads them, and prints the provenance record.
    """
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching MOD16A2 v061 for period {period}...[/bold]")

    try:
        result = fetch_mod16a2(run_dir=run_dir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during MOD16A2 fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print(
        f"[green]Downloaded {len(result['files'])} files "
        f"to {run_dir / 'data' / 'raw' / 'mod16a2_v061'}[/green]"
    )
    if result.get("consolidated_ncs"):
        for yr, nc in result["consolidated_ncs"].items():
            console.print(f"[green]Consolidated {yr}: {run_dir / nc}[/green]")
    console.print(json_mod.dumps(result, indent=2))


@fetch_app.command(name="mod10c1")
def fetch_mod10c1_cmd(
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
    """Download MODIS MOD10C1 v061 daily snow cover data (0.05deg CMG).

    Authenticates via earthaccess, downloads global CMG files, subsets
    to CONUS, and prints the provenance record.
    """
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.modis import fetch_mod10c1

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching MOD10C1 v061 for period {period}...[/bold]")

    try:
        result = fetch_mod10c1(run_dir=run_dir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during MOD10C1 fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print(
        f"[green]Downloaded {len(result['files'])} files "
        f"to {run_dir / 'data' / 'raw' / 'mod10c1_v061'}[/green]"
    )
    if result.get("consolidated_ncs"):
        for yr, nc in result["consolidated_ncs"].items():
            console.print(f"[green]Consolidated {yr}: {run_dir / nc}[/green]")
    console.print(json_mod.dumps(result, indent=2))
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev test tests/test_modis.py -v`

Expected: all tests PASS

**Step 5: Run full test suite, lint, and format**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`

Expected: all pass

**Step 6: Commit**

```bash
git add src/nhf_spatial_targets/cli.py tests/test_modis.py
git commit -m "feat: add mod16a2 and mod10c1 CLI fetch commands"
```

---

### Task 6: Run full quality gate and final commit

**Step 1: Run format, lint, tests**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`

Expected: all pass, no issues

**Step 2: Verify all new files are tracked**

Run: `git status`

Expected: clean working tree (all changes committed)

**Step 3: Review the implementation**

Verify:
- `catalog/sources.yml` has `short_name` and `version` for both MODIS v061 entries
- `modis.py` has `fetch_mod16a2()`, `fetch_mod10c1()`, `_subset_to_conus()`, and all helpers
- `consolidate.py` has `consolidate_mod16a2()` and `consolidate_mod10c1()` stubs
- `cli.py` has `mod16a2` and `mod10c1` fetch subcommands
- `test_modis.py` covers both products with unit tests
- No hardcoded URLs or product names in Python code (all from catalog)

---

## Notes for implementer

1. **The `consolidate_mod16a2` and `consolidate_mod10c1` functions are left as stubs** (`raise NotImplementedError`). They will be implemented in a follow-up task. The unit tests mock these functions, so all tests pass.

2. **The `_subset_to_conus` function assumes MOD10C1 HDF files have `lat`/`lon` coordinate names.** The actual coordinate names in the HDF-EOS files may differ (e.g., `YDim`/`XDim` or `latitude`/`longitude`). This will need to be verified and adjusted during integration testing with real data.

3. **The MOD16A2 `short_name` may need verification.** The plan uses `MOD16A2GF` (gap-filled version). If the non-gap-filled `MOD16A2` is needed instead, update `catalog/sources.yml`.

4. **Variable format difference:** MODIS catalog entries use plain strings (`["ET_500m"]`), while MERRA-2 uses dicts (`[{"name": "GWETTOP"}]`). The MODIS fetch code uses `meta["variables"]` directly — do NOT use `[v["name"] for v in meta["variables"]]`.
