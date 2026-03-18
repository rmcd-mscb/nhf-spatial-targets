# Configurable Datastore and Workspace Refactor — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate fabric-independent raw data storage (datastore) from fabric-dependent workspaces, with a `validate` preflight gate and centralized path resolution.

**Architecture:** New `workspace.py` module centralizes all path resolution and directory creation. `init` creates a minimal workspace skeleton with a config template. `validate` reads the user-edited config, runs preflight checks (fabric, datastore, credentials, catalog), and writes `fabric.json` as the "validated" marker. All fetch modules rename `run_dir` to `workdir` and call `workspace.load(workdir)` internally to get a `Workspace` object for path resolution. Consolidation functions change from `run_dir` to explicit `source_dir` parameters (no more internal path construction). **Spec deviation:** The spec says fetch modules accept a `Workspace` object directly; this plan instead has them accept `workdir: Path` and load the workspace internally, which keeps the CLI layer simpler (it only passes paths, not objects) and matches the existing pattern where fetch functions accept `Path` arguments.

**Tech Stack:** Python 3.11+, cyclopts (CLI), PyYAML, geopandas (fabric reading), pathlib, platform-conditional os.chmod

**Spec:** `docs/superpowers/specs/2026-03-18-configurable-datastore-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/nhf_spatial_targets/workspace.py` | Path resolution, `Workspace` dataclass, `make_dir()`, `load()` |
| Create | `src/nhf_spatial_targets/validate.py` | All preflight checks, writes `fabric.json` + `manifest.json` |
| Create | `tests/test_workspace.py` | Tests for workspace path resolution, load gating, make_dir |
| Create | `tests/test_validate.py` | Tests for all preflight checks |
| Modify | `src/nhf_spatial_targets/init_run.py` | Simplify to skeleton + config template only |
| Modify | `src/nhf_spatial_targets/cli.py` | New `validate` command, simplified `init`, `--workdir` everywhere |
| Modify | `src/nhf_spatial_targets/fetch/_auth.py` | Rename `run_dir` to `workdir` |
| Modify | `src/nhf_spatial_targets/fetch/merra2.py` | Use `Workspace` for paths |
| Modify | `src/nhf_spatial_targets/fetch/nldas.py` | Use `Workspace` for paths |
| Modify | `src/nhf_spatial_targets/fetch/ncep_ncar.py` | Use `Workspace` for paths |
| Modify | `src/nhf_spatial_targets/fetch/modis.py` | Use `Workspace` for paths |
| Modify | `src/nhf_spatial_targets/fetch/pangaea.py` | Use `Workspace` for paths |
| Modify | `src/nhf_spatial_targets/fetch/reitz2017.py` | Use `Workspace` for paths |
| Modify | `src/nhf_spatial_targets/fetch/consolidate.py` | `run_dir` → `source_dir` param (no more path construction) |
| Modify | `tests/test_consolidate.py` | Update for new consolidate signatures |
| Modify | `tests/test_consolidate_modis.py` | Update for new consolidate signatures |
| Modify | `tests/test_init_run.py` | Update for simplified init |
| Modify | `tests/test_cli.py` | Update for new CLI surface |
| Modify | `tests/test_merra2.py` | Swap run_dir fixtures for Workspace |
| Modify | `tests/test_nldas.py` | Swap run_dir fixtures for Workspace |
| Modify | `tests/test_ncep_ncar.py` | Swap run_dir fixtures for Workspace |
| Modify | `tests/test_modis.py` | Swap run_dir fixtures for Workspace |
| Modify | `tests/test_pangaea.py` | Swap run_dir fixtures for Workspace |
| Modify | `tests/test_reitz2017.py` | Swap run_dir fixtures for Workspace |
| Modify | `tests/test_auth.py` | Update for Workspace |
| Modify | `pixi.toml` | Update task commands and comments for `--workdir` |
| Modify | `CLAUDE.md` | Update CLI docs, layout, conventions |

---

## Chunk 1: workspace.py — Path Resolver and Directory Helper

### Task 1: Create `workspace.py` with `Workspace` dataclass and `make_dir()`

**Files:**
- Create: `src/nhf_spatial_targets/workspace.py`
- Create: `tests/test_workspace.py`

- [ ] **Step 1: Write failing tests for `make_dir()`**

Create `tests/test_workspace.py`:

```python
"""Tests for workspace path resolution and directory helpers."""

from __future__ import annotations

import os
import platform
import stat
from pathlib import Path

import pytest


def test_make_dir_creates_directory(tmp_path):
    """make_dir creates the directory and returns the path."""
    from nhf_spatial_targets.workspace import make_dir

    target = tmp_path / "newdir"
    result = make_dir(target)
    assert result == target
    assert target.is_dir()


def test_make_dir_parents(tmp_path):
    """make_dir creates parent directories."""
    from nhf_spatial_targets.workspace import make_dir

    target = tmp_path / "a" / "b" / "c"
    make_dir(target)
    assert target.is_dir()


def test_make_dir_existing_ok(tmp_path):
    """make_dir does not raise if directory already exists."""
    from nhf_spatial_targets.workspace import make_dir

    target = tmp_path / "existing"
    target.mkdir()
    make_dir(target)  # should not raise
    assert target.is_dir()


@pytest.mark.skipif(platform.system() == "Windows", reason="Unix permissions only")
def test_make_dir_applies_mode(tmp_path):
    """make_dir applies the octal dir_mode on Unix."""
    from nhf_spatial_targets.workspace import make_dir

    target = tmp_path / "modedir"
    make_dir(target, dir_mode=0o2775)
    st = target.stat()
    # Check setgid and group-write bits are set
    assert st.st_mode & stat.S_ISGID
    assert st.st_mode & stat.S_IWGRP


def test_make_dir_no_mode_on_windows(tmp_path, monkeypatch):
    """make_dir ignores dir_mode on Windows (no error)."""
    from nhf_spatial_targets.workspace import make_dir

    monkeypatch.setattr("nhf_spatial_targets.workspace._IS_UNIX", False)
    target = tmp_path / "windir"
    make_dir(target, dir_mode=0o2775)
    assert target.is_dir()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -- tests/test_workspace.py -v`
Expected: FAIL — `nhf_spatial_targets.workspace` does not exist

- [ ] **Step 3: Implement `make_dir()` and module skeleton**

Create `src/nhf_spatial_targets/workspace.py`:

```python
"""Workspace path resolution and directory helpers.

Centralises all path construction so fetch/aggregate/target modules
never build ``data/raw/...`` paths themselves.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

_IS_UNIX = platform.system() != "Windows"


def make_dir(path: Path, *, dir_mode: int | None = None) -> Path:
    """Create a directory (with parents), optionally applying Unix permissions.

    Parameters
    ----------
    path : Path
        Directory to create.
    dir_mode : int | None
        Octal mode to apply (e.g. 0o2775).  Ignored on Windows.

    Returns
    -------
    Path
        The created directory path.
    """
    path.mkdir(parents=True, exist_ok=True)
    if dir_mode is not None and _IS_UNIX:
        os.chmod(path, dir_mode)
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev test -- tests/test_workspace.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/workspace.py tests/test_workspace.py
git commit -m "feat: add workspace.make_dir() with Unix permission support"
```

### Task 2: Add `Workspace` dataclass and `load()` function

**Files:**
- Modify: `src/nhf_spatial_targets/workspace.py`
- Modify: `tests/test_workspace.py`

- [ ] **Step 1: Write failing tests for Workspace and load()**

Append to `tests/test_workspace.py`:

```python
import json
import yaml


def _write_config(workdir: Path, datastore: Path, fabric_path: str = "/fake/fabric.gpkg") -> None:
    """Helper to write a minimal valid config.yml."""
    config = {
        "fabric": {
            "path": fabric_path,
            "id_col": "nhm_id",
            "crs": "EPSG:4326",
            "buffer_deg": 0.1,
        },
        "datastore": str(datastore),
        "dir_mode": "2775",
        "aggregation": {"engine": "gdptools", "method": "area_weighted"},
        "output": {"format": "netcdf", "compress": True},
        "targets": {},
    }
    (workdir / "config.yml").write_text(yaml.dump(config, sort_keys=False))


def _write_fabric_json(workdir: Path) -> None:
    """Helper to write a minimal valid fabric.json."""
    fabric = {
        "path": "/fake/fabric.gpkg",
        "sha256": "abc123",
        "crs": "EPSG:4326",
        "id_col": "nhm_id",
        "hru_count": 42,
        "bbox": {"minx": -125.0, "miny": 24.0, "maxx": -66.0, "maxy": 50.0},
        "bbox_buffered": {"minx": -125.1, "miny": 23.9, "maxx": -65.9, "maxy": 50.1},
        "buffer_deg": 0.1,
    }
    (workdir / "fabric.json").write_text(json.dumps(fabric))


def test_load_returns_workspace(tmp_path):
    """load() returns a Workspace with config and fabric data."""
    from nhf_spatial_targets.workspace import load

    datastore = tmp_path / "datastore"
    datastore.mkdir()
    _write_config(tmp_path, datastore)
    _write_fabric_json(tmp_path)

    ws = load(tmp_path)
    assert ws.workdir == tmp_path
    assert ws.datastore == datastore
    assert ws.fabric["id_col"] == "nhm_id"
    assert ws.config["fabric"]["id_col"] == "nhm_id"


def test_load_fails_without_fabric_json(tmp_path):
    """load() raises if fabric.json is missing (not validated)."""
    from nhf_spatial_targets.workspace import load

    datastore = tmp_path / "datastore"
    datastore.mkdir()
    _write_config(tmp_path, datastore)

    with pytest.raises(FileNotFoundError, match="validate"):
        load(tmp_path)


def test_load_fails_without_config(tmp_path):
    """load() raises if config.yml is missing."""
    from nhf_spatial_targets.workspace import load

    with pytest.raises(FileNotFoundError, match="config.yml"):
        load(tmp_path)


def test_workspace_raw_dir(tmp_path):
    """raw_dir returns <datastore>/<source_key>."""
    from nhf_spatial_targets.workspace import load

    datastore = tmp_path / "datastore"
    datastore.mkdir()
    _write_config(tmp_path, datastore)
    _write_fabric_json(tmp_path)

    ws = load(tmp_path)
    assert ws.raw_dir("merra2") == datastore / "merra2"


def test_workspace_aggregated_dir(tmp_path):
    """aggregated_dir returns <workdir>/data/aggregated."""
    from nhf_spatial_targets.workspace import load

    datastore = tmp_path / "datastore"
    datastore.mkdir()
    _write_config(tmp_path, datastore)
    _write_fabric_json(tmp_path)

    ws = load(tmp_path)
    assert ws.aggregated_dir() == tmp_path / "data" / "aggregated"


def test_workspace_targets_dir(tmp_path):
    """targets_dir returns <workdir>/targets."""
    from nhf_spatial_targets.workspace import load

    datastore = tmp_path / "datastore"
    datastore.mkdir()
    _write_config(tmp_path, datastore)
    _write_fabric_json(tmp_path)

    ws = load(tmp_path)
    assert ws.targets_dir() == tmp_path / "targets"


def test_workspace_manifest_path(tmp_path):
    """manifest_path returns <workdir>/manifest.json."""
    from nhf_spatial_targets.workspace import load

    datastore = tmp_path / "datastore"
    datastore.mkdir()
    _write_config(tmp_path, datastore)
    _write_fabric_json(tmp_path)

    ws = load(tmp_path)
    assert ws.manifest_path == tmp_path / "manifest.json"


def test_workspace_credentials_path(tmp_path):
    """credentials_path returns <workdir>/.credentials.yml."""
    from nhf_spatial_targets.workspace import load

    datastore = tmp_path / "datastore"
    datastore.mkdir()
    _write_config(tmp_path, datastore)
    _write_fabric_json(tmp_path)

    ws = load(tmp_path)
    assert ws.credentials_path == tmp_path / ".credentials.yml"


def test_workspace_dir_mode_parsed_as_octal(tmp_path):
    """dir_mode string '2775' is parsed as octal 0o2775."""
    from nhf_spatial_targets.workspace import load

    datastore = tmp_path / "datastore"
    datastore.mkdir()
    _write_config(tmp_path, datastore)
    _write_fabric_json(tmp_path)

    ws = load(tmp_path)
    assert ws.dir_mode == 0o2775
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -- tests/test_workspace.py -v`
Expected: FAIL — `load` not importable, `Workspace` not defined

- [ ] **Step 3: Implement `Workspace` dataclass and `load()`**

Add to `src/nhf_spatial_targets/workspace.py`:

```python
import json
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class Workspace:
    """Resolved workspace with validated paths.

    Created by ``load()`` which reads config.yml + fabric.json.
    All fetch/aggregate/target modules use this instead of raw paths.
    """

    workdir: Path
    datastore: Path
    config: dict
    fabric: dict
    dir_mode: int | None

    def raw_dir(self, source_key: str) -> Path:
        """Return ``<datastore>/<source_key>/``."""
        return self.datastore / source_key

    def aggregated_dir(self) -> Path:
        """Return ``<workdir>/data/aggregated/``."""
        return self.workdir / "data" / "aggregated"

    def targets_dir(self) -> Path:
        """Return ``<workdir>/targets/``."""
        return self.workdir / "targets"

    @property
    def manifest_path(self) -> Path:
        """Return ``<workdir>/manifest.json``."""
        return self.workdir / "manifest.json"

    @property
    def credentials_path(self) -> Path:
        """Return ``<workdir>/.credentials.yml``."""
        return self.workdir / ".credentials.yml"


def load(workdir: Path) -> Workspace:
    """Load a validated workspace.

    Reads ``config.yml`` and ``fabric.json`` from *workdir*.
    Raises ``FileNotFoundError`` if either is missing.

    Parameters
    ----------
    workdir : Path
        Workspace directory created by ``nhf-targets init``.

    Returns
    -------
    Workspace
    """
    config_path = workdir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yml not found in {workdir}. "
            f"Run 'nhf-targets init --workdir {workdir}' first."
        )

    config = yaml.safe_load(config_path.read_text())

    fabric_path = workdir / "fabric.json"
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"fabric.json not found in {workdir}. "
            f"Run 'nhf-targets validate --workdir {workdir}' first."
        )

    fabric = json.loads(fabric_path.read_text())

    dir_mode_str = config.get("dir_mode")
    dir_mode = int(dir_mode_str, 8) if dir_mode_str else None

    return Workspace(
        workdir=workdir,
        datastore=Path(config["datastore"]),
        config=config,
        fabric=fabric,
        dir_mode=dir_mode,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev test -- tests/test_workspace.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/workspace.py tests/test_workspace.py
git commit -m "feat: add Workspace dataclass with load() and path resolution"
```

---

## Chunk 2: validate.py — Preflight Checks

### Task 3: Create `validate.py` with config and fabric checks

**Files:**
- Create: `src/nhf_spatial_targets/validate.py`
- Create: `tests/test_validate.py`

- [ ] **Step 1: Write failing tests for config completeness and fabric checks**

Create `tests/test_validate.py`:

```python
"""Tests for workspace validation (preflight checks)."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest
import yaml
from shapely.geometry import box

from nhf_spatial_targets.validate import validate_workspace


@pytest.fixture()
def minimal_fabric(tmp_path) -> Path:
    """Create a minimal GeoPackage with nhm_id column."""
    gdf = gpd.GeoDataFrame(
        {"nhm_id": [1, 2, 3]},
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2), box(2, 2, 3, 3)],
        crs="EPSG:4326",
    )
    path = tmp_path / "fabric.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def _write_config(workdir: Path, overrides: dict | None = None) -> Path:
    """Write config.yml with optional overrides."""
    config = {
        "fabric": {
            "path": "",
            "id_col": "nhm_id",
            "crs": "EPSG:4326",
            "buffer_deg": 0.1,
        },
        "datastore": "",
        "dir_mode": "2775",
        "aggregation": {"engine": "gdptools", "method": "area_weighted"},
        "output": {"format": "netcdf", "compress": True},
        "targets": {},
    }
    if overrides:
        for key, val in overrides.items():
            if isinstance(val, dict) and key in config and isinstance(config[key], dict):
                config[key].update(val)
            else:
                config[key] = val
    path = workdir / "config.yml"
    path.write_text(yaml.dump(config, sort_keys=False))
    return path


def _write_credentials(workdir: Path, earthdata_user: str = "user", earthdata_pass: str = "pass") -> None:
    """Write .credentials.yml with earthdata creds."""
    creds = {
        "nasa_earthdata": {"username": earthdata_user, "password": earthdata_pass},
    }
    (workdir / ".credentials.yml").write_text(yaml.dump(creds, sort_keys=False))


def test_validate_missing_config(tmp_path):
    """Fails if config.yml is missing."""
    with pytest.raises(FileNotFoundError, match="config.yml"):
        validate_workspace(tmp_path)


def test_validate_empty_fabric_path(tmp_path):
    """Fails if fabric.path is empty."""
    _write_config(tmp_path)
    _write_credentials(tmp_path)
    with pytest.raises(ValueError, match="fabric.path"):
        validate_workspace(tmp_path)


def test_validate_empty_datastore(tmp_path, minimal_fabric):
    """Fails if datastore is empty."""
    _write_config(tmp_path, {"fabric": {"path": str(minimal_fabric)}, "datastore": ""})
    _write_credentials(tmp_path)
    with pytest.raises(ValueError, match="datastore"):
        validate_workspace(tmp_path)


def test_validate_fabric_not_found(tmp_path):
    """Fails if fabric file does not exist."""
    datastore = tmp_path / "ds"
    _write_config(tmp_path, {
        "fabric": {"path": "/no/such/file.gpkg"},
        "datastore": str(datastore),
    })
    _write_credentials(tmp_path)
    with pytest.raises(FileNotFoundError, match="fabric"):
        validate_workspace(tmp_path)


def test_validate_missing_id_col(tmp_path, minimal_fabric):
    """Fails if fabric.id_col is not in the GeoPackage."""
    datastore = tmp_path / "ds"
    _write_config(tmp_path, {
        "fabric": {"path": str(minimal_fabric), "id_col": "nonexistent_col"},
        "datastore": str(datastore),
    })
    _write_credentials(tmp_path)
    with pytest.raises(ValueError, match="nonexistent_col"):
        validate_workspace(tmp_path)


def test_validate_writes_fabric_json(tmp_path, minimal_fabric):
    """On success, writes fabric.json with expected fields."""
    datastore = tmp_path / "ds"
    _write_config(tmp_path, {
        "fabric": {"path": str(minimal_fabric)},
        "datastore": str(datastore),
    })
    _write_credentials(tmp_path)
    validate_workspace(tmp_path)
    fabric_json = json.loads((tmp_path / "fabric.json").read_text())
    assert fabric_json["sha256"]
    assert fabric_json["id_col"] == "nhm_id"
    assert fabric_json["hru_count"] == 3
    assert "bbox" in fabric_json
    assert "bbox_buffered" in fabric_json
    assert fabric_json["buffer_deg"] == 0.1


def test_validate_writes_manifest_json(tmp_path, minimal_fabric):
    """On success, writes manifest.json with initial metadata."""
    datastore = tmp_path / "ds"
    _write_config(tmp_path, {
        "fabric": {"path": str(minimal_fabric)},
        "datastore": str(datastore),
    })
    _write_credentials(tmp_path)
    validate_workspace(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["sources"] == {}
    assert manifest["steps"] == []
    assert "fabric" in manifest
    assert "nhf_spatial_targets_version" in manifest


def test_validate_creates_datastore(tmp_path, minimal_fabric):
    """validate creates the datastore directory if missing."""
    datastore = tmp_path / "new_datastore"
    _write_config(tmp_path, {
        "fabric": {"path": str(minimal_fabric)},
        "datastore": str(datastore),
    })
    _write_credentials(tmp_path)
    validate_workspace(tmp_path)
    assert datastore.is_dir()


def test_validate_creates_source_subdirs(tmp_path, minimal_fabric):
    """validate creates subdirs for current+implemented sources only."""
    datastore = tmp_path / "ds"
    _write_config(tmp_path, {
        "fabric": {"path": str(minimal_fabric)},
        "datastore": str(datastore),
    })
    _write_credentials(tmp_path)
    validate_workspace(tmp_path)

    expected = {
        "merra2", "nldas_mosaic", "nldas_noah", "ncep_ncar",
        "mod16a2_v061", "mod10c1_v061", "watergap22d", "reitz2017",
    }
    actual = {p.name for p in datastore.iterdir() if p.is_dir()}
    assert actual == expected


def test_validate_missing_credentials(tmp_path, minimal_fabric):
    """Fails if .credentials.yml is missing."""
    datastore = tmp_path / "ds"
    _write_config(tmp_path, {
        "fabric": {"path": str(minimal_fabric)},
        "datastore": str(datastore),
    })
    with pytest.raises(FileNotFoundError, match="credentials"):
        validate_workspace(tmp_path)


def test_validate_empty_earthdata_creds(tmp_path, minimal_fabric):
    """Fails if earthdata credentials are empty."""
    datastore = tmp_path / "ds"
    _write_config(tmp_path, {
        "fabric": {"path": str(minimal_fabric)},
        "datastore": str(datastore),
    })
    _write_credentials(tmp_path, earthdata_user="", earthdata_pass="")
    with pytest.raises(ValueError, match="[Ee]arthdata"):
        validate_workspace(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -- tests/test_validate.py -v`
Expected: FAIL — `nhf_spatial_targets.validate` does not exist

- [ ] **Step 3: Implement `validate.py`**

Create `src/nhf_spatial_targets/validate.py`:

```python
"""Workspace validation — preflight checks before fetch/run.

Run via ``nhf-targets validate --workdir <path>``.
Checks config completeness, fabric existence and schema, datastore
writability, credentials, and catalog consistency.  On success writes
``fabric.json`` and ``manifest.json``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from nhf_spatial_targets import __version__
from nhf_spatial_targets.workspace import make_dir

logger = logging.getLogger(__name__)

# Sources with status: current AND an implemented (non-stub) fetch module.
_IMPLEMENTED_SOURCES = frozenset({
    "merra2",
    "nldas_mosaic",
    "nldas_noah",
    "ncep_ncar",
    "mod16a2_v061",
    "mod10c1_v061",
    "watergap22d",
    "reitz2017",
})


def validate_workspace(workdir: Path) -> None:
    """Run all preflight checks and write fabric.json + manifest.json.

    Parameters
    ----------
    workdir : Path
        Workspace directory created by ``nhf-targets init``.

    Raises
    ------
    FileNotFoundError
        If config.yml, fabric file, or credentials file is missing.
    ValueError
        If required config fields are empty, id_col is not in fabric,
        or earthdata credentials are blank.
    """
    workdir = workdir.resolve()

    # 1. Read config
    config_path = workdir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yml not found in {workdir}. "
            f"Run 'nhf-targets init --workdir {workdir}' first."
        )
    config = yaml.safe_load(config_path.read_text())

    # 2. Check required fields
    fabric_cfg = config.get("fabric", {})
    fabric_file = fabric_cfg.get("path", "")
    id_col = fabric_cfg.get("id_col", "")
    datastore_str = config.get("datastore", "")

    if not fabric_file:
        raise ValueError(
            "fabric.path is empty in config.yml. "
            "Set it to the path of your HRU fabric GeoPackage."
        )
    if not id_col:
        raise ValueError(
            "fabric.id_col is empty in config.yml. "
            "Set it to the HRU ID column name (e.g. 'nhm_id')."
        )
    if not datastore_str:
        raise ValueError(
            "datastore is empty in config.yml. "
            "Set it to the shared raw data directory path."
        )

    # 3. Fabric exists
    fabric_path = Path(fabric_file)
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"Fabric file not found: {fabric_path}. "
            f"Check the fabric.path setting in config.yml."
        )

    # 4. Feature ID column exists
    _check_id_col(fabric_path, id_col)

    # 5. Compute fabric metadata
    buffer_deg = fabric_cfg.get("buffer_deg", 0.1)
    fabric_meta = _fabric_metadata(fabric_path, id_col, buffer_deg)

    # 6. Credentials (check before creating directories)
    creds_path = workdir / ".credentials.yml"
    if not creds_path.exists():
        raise FileNotFoundError(
            f".credentials.yml not found in {workdir}. "
            f"Run 'nhf-targets init' to create a template."
        )
    _check_earthdata_credentials(creds_path)

    # 8. Datastore — create if missing, create source subdirs
    dir_mode_str = config.get("dir_mode")
    dir_mode = int(dir_mode_str, 8) if dir_mode_str else None
    datastore = Path(datastore_str)
    make_dir(datastore, dir_mode=dir_mode)
    for key in sorted(_IMPLEMENTED_SOURCES):
        make_dir(datastore / key, dir_mode=dir_mode)

    # 9. Catalog consistency
    _check_catalog_consistency()

    # 10. Write fabric.json
    (workdir / "fabric.json").write_text(json.dumps(fabric_meta, indent=2))
    logger.info("Wrote fabric.json to %s", workdir)

    # 11. Write manifest.json
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "nhf_spatial_targets_version": __version__,
        "fabric": {
            "path": fabric_meta["path"],
            "sha256": fabric_meta["sha256"],
            "crs": fabric_meta["crs"],
            "id_col": fabric_meta["id_col"],
            "hru_count": fabric_meta["hru_count"],
        },
        "sources": {},
        "steps": [],
    }
    (workdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote manifest.json to %s", workdir)


def _check_id_col(fabric_path: Path, id_col: str) -> None:
    """Verify id_col exists in the fabric's attribute table."""
    import geopandas as gpd

    if fabric_path.suffix.lower() in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(fabric_path)
    else:
        gdf = gpd.read_file(fabric_path, rows=1)
    columns = list(gdf.columns)

    if id_col not in columns:
        raise ValueError(
            f"Column '{id_col}' not found in fabric at {fabric_path}. "
            f"Available columns: {columns}. "
            f"Update fabric.id_col in config.yml."
        )


def _fabric_metadata(fabric_path: Path, id_col: str, buffer_deg: float) -> dict:
    """Compute fabric hash, bbox, and HRU count."""
    import hashlib

    import geopandas as gpd
    from shapely.geometry import box

    sha256 = _sha256(fabric_path)

    if fabric_path.suffix.lower() in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(fabric_path)
    else:
        gdf = gpd.read_file(fabric_path, rows=None)
    native_crs = gdf.crs.to_string() if gdf.crs else "unknown"
    hru_count = len(gdf)

    native_bounds = gdf.total_bounds
    if gdf.crs and not gdf.crs.is_geographic:
        bbox_geom = gpd.GeoSeries([box(*native_bounds)], crs=gdf.crs)
        wgs84_bounds = bbox_geom.to_crs("EPSG:4326").total_bounds
    else:
        wgs84_bounds = native_bounds

    buffered = {
        "minx": float(wgs84_bounds[0]) - buffer_deg,
        "miny": float(wgs84_bounds[1]) - buffer_deg,
        "maxx": float(wgs84_bounds[2]) + buffer_deg,
        "maxy": float(wgs84_bounds[3]) + buffer_deg,
    }

    return {
        "path": str(fabric_path),
        "sha256": sha256,
        "crs": native_crs,
        "id_col": id_col,
        "hru_count": hru_count,
        "bbox": {
            "minx": float(wgs84_bounds[0]),
            "miny": float(wgs84_bounds[1]),
            "maxx": float(wgs84_bounds[2]),
            "maxy": float(wgs84_bounds[3]),
        },
        "bbox_buffered": buffered,
        "buffer_deg": buffer_deg,
    }


def _sha256(path: Path) -> str:
    """Compute SHA-256 of a file, or of all files in a directory."""
    import hashlib

    h = hashlib.sha256()
    if path.is_dir():
        for child in sorted(path.rglob("*"), key=lambda p: str(p)):
            if child.is_file():
                h.update(str(child.relative_to(path)).encode())
                with child.open("rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
    else:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _check_earthdata_credentials(creds_path: Path) -> None:
    """Check that Earthdata username and password are non-empty."""
    data = yaml.safe_load(creds_path.read_text()) or {}
    nasa = data.get("nasa_earthdata", {})
    username = nasa.get("username", "") or ""
    password = nasa.get("password", "") or ""
    if not username or not password:
        raise ValueError(
            f"Earthdata credentials are empty in {creds_path}. "
            f"Fill in nasa_earthdata.username and nasa_earthdata.password."
        )


def _check_catalog_consistency() -> None:
    """Verify sources referenced in variables.yml exist in sources.yml."""
    from nhf_spatial_targets.catalog import sources, variables

    all_sources = sources()
    all_vars = variables()

    for var_name, var_def in all_vars.items():
        for src in var_def.get("sources", []):
            if src not in all_sources:
                raise ValueError(
                    f"Variable '{var_name}' references source '{src}' "
                    f"which is not defined in catalog/sources.yml."
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev test -- tests/test_validate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/validate.py tests/test_validate.py
git commit -m "feat: add validate_workspace() preflight checks"
```

---

## Chunk 3: Simplified init and new validate CLI command

### Task 4: Rewrite `init_run.py` for minimal workspace creation

**Files:**
- Modify: `src/nhf_spatial_targets/init_run.py`
- Modify: `tests/test_init_run.py`

- [ ] **Step 1: Write failing tests for simplified init**

Replace `tests/test_init_run.py` contents:

```python
"""Tests for nhf-targets init workspace creation (simplified)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nhf_spatial_targets.init_run import init_workspace


def test_init_creates_skeleton(tmp_path):
    """init_workspace creates the expected directory structure."""
    workdir = tmp_path / "my-workspace"
    init_workspace(workdir)

    assert workdir.is_dir()
    assert (workdir / "config.yml").exists()
    assert (workdir / ".credentials.yml").exists()
    assert (workdir / "data" / "aggregated").is_dir()
    assert (workdir / "targets").is_dir()
    assert (workdir / "logs").is_dir()


def test_init_config_template_has_required_fields(tmp_path):
    """config.yml template contains all required fields."""
    workdir = tmp_path / "ws"
    init_workspace(workdir)

    config = yaml.safe_load((workdir / "config.yml").read_text())
    assert "fabric" in config
    assert "path" in config["fabric"]
    assert "id_col" in config["fabric"]
    assert "buffer_deg" in config["fabric"]
    assert "datastore" in config
    assert "dir_mode" in config
    assert "targets" in config


def test_init_credentials_template(tmp_path):
    """credentials template has nasa_earthdata section."""
    workdir = tmp_path / "ws"
    init_workspace(workdir)

    creds = yaml.safe_load((workdir / ".credentials.yml").read_text())
    assert "nasa_earthdata" in creds
    assert "username" in creds["nasa_earthdata"]


def test_init_existing_workdir_raises(tmp_path):
    """init_workspace raises if workdir already exists."""
    workdir = tmp_path / "ws"
    init_workspace(workdir)
    with pytest.raises(FileExistsError, match="already exists"):
        init_workspace(workdir)


def test_init_no_fabric_json(tmp_path):
    """init does NOT create fabric.json (that's validate's job)."""
    workdir = tmp_path / "ws"
    init_workspace(workdir)
    assert not (workdir / "fabric.json").exists()


def test_init_no_manifest_json(tmp_path):
    """init does NOT create manifest.json (that's validate's job)."""
    workdir = tmp_path / "ws"
    init_workspace(workdir)
    assert not (workdir / "manifest.json").exists()


def test_init_no_data_raw(tmp_path):
    """init does NOT create data/raw/ (datastore is separate)."""
    workdir = tmp_path / "ws"
    init_workspace(workdir)
    assert not (workdir / "data" / "raw").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -- tests/test_init_run.py -v`
Expected: FAIL — `init_workspace` not importable

- [ ] **Step 3: Rewrite `init_run.py`**

Replace `src/nhf_spatial_targets/init_run.py` with:

```python
"""Logic for 'nhf-targets init' — create a workspace skeleton."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_CONFIG_TEMPLATE = """\
# nhf-spatial-targets workspace configuration
# Edit the fields below, then run: nhf-targets validate --workdir <this-dir>

# ---------------------------------------------------------------------------
# Fabric
# ---------------------------------------------------------------------------
fabric:
  path: ""          # path to fabric GeoPackage (e.g., /data/gfv1.1.gpkg)
  id_col: nhm_id    # feature ID column in the fabric
  crs: "EPSG:4326"
  buffer_deg: 0.1   # degrees to buffer fabric bbox for source downloads
  # Optional: subset to specific HRU IDs for testing
  # subset_ids: [1, 2, 3]

# ---------------------------------------------------------------------------
# Shared raw data store (fabric-independent)
# ---------------------------------------------------------------------------
datastore: ""       # path to shared raw data directory (e.g., /mnt/bigdrive/nhf-raw)

# ---------------------------------------------------------------------------
# Directory permissions (Unix only; ignored on Windows)
# ---------------------------------------------------------------------------
# Octal string parsed as octal (e.g., "2775" = setgid + group rwx + other rx).
dir_mode: "2775"

# ---------------------------------------------------------------------------
# Spatial aggregation (gdptools)
# ---------------------------------------------------------------------------
aggregation:
  engine: gdptools
  method: area_weighted      # area-weighted mean for intensive variables

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
output:
  format: netcdf
  compress: true

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
targets:

  runoff:
    enabled: true
    sources:
      - nhm_mwbm
    time_step: monthly
    period: "1982-01-01/2010-12-31"
    prms_variable: basin_cfs
    range_method: mwbm_uncertainty
    output_file: runoff_targets.nc

  aet:
    enabled: true
    sources:
      - nhm_mwbm
      - mod16a2_v061
      - ssebop
    time_step: monthly
    period: "2000-01-01/2010-12-31"
    prms_variable: hru_actet
    range_method: multi_source_minmax
    output_file: aet_targets.nc

  recharge:
    enabled: true
    sources:
      - reitz2017
      - watergap22d
    time_step: annual
    period: "2000-01-01/2009-12-31"
    prms_variable: recharge
    range_method: normalized_minmax
    normalize: true
    normalize_period: "2000-01-01/2009-12-31"
    output_file: recharge_targets.nc

  soil_moisture:
    enabled: true
    sources:
      - merra2
      - ncep_ncar
      - nldas_mosaic
      - nldas_noah
    time_step:
      - monthly
      - annual
    period: "1982-01-01/2010-12-31"
    prms_variable: soil_rechr
    range_method: normalized_minmax
    normalize: true
    normalize_by: calendar_month
    output_file: soil_moisture_targets.nc

  snow_covered_area:
    enabled: true
    sources:
      - mod10c1_v061
    time_step: daily
    period: "2000-01-01/2010-12-31"
    prms_variable: snowcov_area
    range_method: modis_ci
    ci_threshold: 0.70
    output_file: sca_targets.nc
"""


def init_workspace(workdir: Path) -> Path:
    """Create a new workspace skeleton.

    Parameters
    ----------
    workdir : Path
        Directory to create.  Must not already exist.

    Returns
    -------
    Path to the created workspace directory.
    """
    workdir = workdir.resolve()
    if workdir.exists():
        raise FileExistsError(
            f"Workspace already exists: {workdir}\n"
            "Choose a different --workdir path."
        )

    workdir.mkdir(parents=True)
    (workdir / "data" / "aggregated").mkdir(parents=True)
    (workdir / "targets").mkdir(parents=True)
    (workdir / "logs").mkdir(parents=True)

    # Write config template
    (workdir / "config.yml").write_text(_CONFIG_TEMPLATE)

    # Write credentials template
    _write_credentials_template(workdir)

    return workdir


def _write_credentials_template(workdir: Path) -> None:
    template = {
        "nasa_earthdata": {
            "_comment": "NASA Earthdata login — https://urs.earthdata.nasa.gov",
            "username": "",
            "password": "",
        },
    }
    path = workdir / ".credentials.yml"
    path.write_text(
        "# nhf-spatial-targets workspace credentials\n"
        "# Fill in before running: nhf-targets validate --workdir <this-dir>\n"
        "# This file is gitignored — do not commit it.\n\n"
        + yaml.dump(template, default_flow_style=False, sort_keys=False)
    )


```

Note: The old `_fabric_metadata` and `_sha256` functions are removed from `init_run.py`. They now live exclusively in `validate.py`. The simplified `init_run.py` no longer needs them.

Note: The existing `_sha256` directory hashing tests (`test_sha256_directory_consistent`, `test_sha256_directory_includes_filenames`) and `_fabric_metadata` parquet tests (`test_fabric_metadata_reads_parquet`, `test_fabric_metadata_reads_geoparquet`) should be moved to `tests/test_validate.py` since those functions now live in `validate.py`. Update their imports from `nhf_spatial_targets.init_run` to `nhf_spatial_targets.validate`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev test -- tests/test_init_run.py tests/test_validate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/init_run.py tests/test_init_run.py src/nhf_spatial_targets/validate.py tests/test_validate.py
git commit -m "refactor: simplify init_run to skeleton-only, move fabric utils to validate"
```

### Task 5: Update CLI — simplified `init`, new `validate`, `--workdir` everywhere

**Files:**
- Modify: `src/nhf_spatial_targets/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for new CLI surface**

Replace `tests/test_cli.py` with updated tests. Key changes:
- `init` tests: only `--workdir`, no `--fabric`
- New `validate` tests
- All fetch commands: `--workdir` instead of `--run-dir`
- `run` command: `--workdir` instead of `--run-dir`

```python
"""Tests for the cyclopts CLI layer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from nhf_spatial_targets.cli import app


def _run(*tokens: str) -> None:
    """Invoke the cyclopts app, suppressing the SystemExit(0) on success."""
    try:
        app(list(tokens), exit_on_error=False)
    except SystemExit as exc:
        if exc.code != 0:
            raise


def _run_meta(*tokens: str) -> None:
    """Invoke via the meta launcher, suppressing SystemExit(0)."""
    try:
        app.meta(list(tokens), exit_on_error=False)
    except SystemExit as exc:
        if exc.code != 0:
            raise


# ---- init command ----------------------------------------------------------


def test_init_creates_workspace(tmp_path):
    """init --workdir creates a workspace skeleton."""
    workdir = tmp_path / "ws"
    _run("init", "--workdir", str(workdir))
    assert (workdir / "config.yml").exists()
    assert (workdir / ".credentials.yml").exists()


def test_init_existing_workdir_fails(tmp_path):
    """init fails if workdir already exists."""
    workdir = tmp_path / "ws"
    workdir.mkdir()
    with pytest.raises(SystemExit):
        _run("init", "--workdir", str(workdir))


# ---- validate command ------------------------------------------------------


def test_validate_missing_workdir(tmp_path):
    """validate fails if workdir does not exist."""
    with pytest.raises(SystemExit):
        _run("validate", "--workdir", str(tmp_path / "missing"))


@patch("nhf_spatial_targets.validate.validate_workspace")
def test_validate_calls_validate_workspace(mock_validate, tmp_path):
    """validate wires --workdir to validate_workspace()."""
    workdir = tmp_path / "ws"
    workdir.mkdir()
    _run("validate", "--workdir", str(workdir))
    mock_validate.assert_called_once_with(workdir)


# ---- run command -----------------------------------------------------------


def test_run_missing_workdir():
    """Exit code 2 when --workdir is not provided."""
    with pytest.raises(SystemExit, match="2"):
        _run("run")


def test_run_nonexistent_workdir(tmp_path):
    """Exit code 2 when --workdir does not exist."""
    with pytest.raises(SystemExit, match="2"):
        _run("run", "--workdir", str(tmp_path / "missing"))


def test_run_not_validated(tmp_path):
    """Exit code 1 when workspace has not been validated (no fabric.json)."""
    workdir = tmp_path / "ws"
    workdir.mkdir()
    (workdir / "config.yml").write_text("targets: {}")
    with pytest.raises(SystemExit):
        _run("run", "--workdir", str(workdir))


def test_run_dispatches_enabled_targets(tmp_path):
    """Dispatches to builder for each enabled target."""
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    config = workdir / "config.yml"
    config.write_text(
        "fabric:\n  path: /fake/fabric.gpkg\n"
        "datastore: /fake/ds\n"
        "output:\n  dir: /fake/out\n"
        "targets:\n  runoff:\n    enabled: true\n  aet:\n    enabled: false\n"
    )
    (workdir / "fabric.json").write_text(json.dumps({
        "bbox_buffered": {"minx": -125, "miny": 24, "maxx": -66, "maxy": 50}
    }))

    with patch("nhf_spatial_targets.cli._dispatch") as mock_dispatch:
        _run("run", "--workdir", str(workdir))

    mock_dispatch.assert_called_once()
    assert mock_dispatch.call_args[0][0] == "runoff"


def test_run_single_target(tmp_path):
    """--target selects a single target by name."""
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    config = workdir / "config.yml"
    config.write_text(
        "fabric:\n  path: /fake/fabric.gpkg\n"
        "datastore: /fake/ds\n"
        "output:\n  dir: /fake/out\n"
        "targets:\n  runoff:\n    enabled: true\n  aet:\n    enabled: true\n"
    )
    (workdir / "fabric.json").write_text(json.dumps({
        "bbox_buffered": {"minx": -125, "miny": 24, "maxx": -66, "maxy": 50}
    }))

    with patch("nhf_spatial_targets.cli._dispatch") as mock_dispatch:
        _run("run", "--workdir", str(workdir), "--target", "aet")

    mock_dispatch.assert_called_once()
    assert mock_dispatch.call_args[0][0] == "aet"


# ---- fetch commands --------------------------------------------------------


def test_fetch_merra2_nonexistent_workdir(tmp_path):
    """Exit code 2 when --workdir does not exist."""
    with pytest.raises(SystemExit, match="2"):
        _run("fetch", "merra2", "--workdir", str(tmp_path / "missing"), "--period", "2010/2010")


def test_fetch_merra2_calls_fetch(tmp_path):
    """CLI wires --workdir and --period to fetch_merra2()."""
    workdir = tmp_path / "workspace"
    workdir.mkdir()

    mock_result = {
        "source_key": "merra2",
        "files": [{"path": "data/raw/merra2/f.nc4", "size_bytes": 100}],
        "access_url": "https://example.com",
        "variables": ["SFMC"],
        "period": "2010/2010",
        "bbox": {},
        "download_timestamp": "2026-01-01T00:00:00+00:00",
    }

    with patch(
        "nhf_spatial_targets.fetch.merra2.fetch_merra2",
        return_value=mock_result,
    ) as mock_fetch:
        _run("fetch", "merra2", "--workdir", str(workdir), "--period", "2010/2010")

    mock_fetch.assert_called_once_with(workdir=workdir, period="2010/2010")


@patch("nhf_spatial_targets.fetch.nldas.fetch_nldas_mosaic")
def test_fetch_nldas_mosaic_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --workdir and --period to fetch_nldas_mosaic()."""
    mock_fetch.return_value = {"files": [], "consolidated_nc": "consolidated.nc"}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run("fetch", "nldas-mosaic", "--workdir", str(workdir), "--period", "2010/2010")
    mock_fetch.assert_called_once()


@patch("nhf_spatial_targets.fetch.nldas.fetch_nldas_noah")
def test_fetch_nldas_noah_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --workdir and --period to fetch_nldas_noah()."""
    mock_fetch.return_value = {"files": [], "consolidated_nc": "consolidated.nc"}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run("fetch", "nldas-noah", "--workdir", str(workdir), "--period", "2010/2010")
    mock_fetch.assert_called_once()


@patch("nhf_spatial_targets.fetch.ncep_ncar.fetch_ncep_ncar")
def test_fetch_ncep_ncar_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --workdir and --period to fetch_ncep_ncar()."""
    mock_fetch.return_value = {"files": [], "consolidated_nc": "consolidated.nc"}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run("fetch", "ncep-ncar", "--workdir", str(workdir), "--period", "2010/2010")
    mock_fetch.assert_called_once()


@patch("nhf_spatial_targets.fetch.modis.fetch_mod16a2")
def test_fetch_mod16a2_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --workdir and --period to fetch_mod16a2()."""
    mock_fetch.return_value = {"files": [], "consolidated_ncs": {}}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run("fetch", "mod16a2", "--workdir", str(workdir), "--period", "2005/2005")
    mock_fetch.assert_called_once_with(workdir=workdir, period="2005/2005")


@patch("nhf_spatial_targets.fetch.modis.fetch_mod10c1")
def test_fetch_mod10c1_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --workdir and --period to fetch_mod10c1()."""
    mock_fetch.return_value = {"files": [], "consolidated_ncs": {}}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run("fetch", "mod10c1", "--workdir", str(workdir), "--period", "2005/2005")
    mock_fetch.assert_called_once_with(workdir=workdir, period="2005/2005")


# ---- catalog commands ------------------------------------------------------


def test_catalog_sources():
    """catalog sources runs without error."""
    _run("catalog", "sources")


def test_catalog_variables():
    """catalog variables runs without error."""
    _run("catalog", "variables")


# ---- meta launcher / verbose -----------------------------------------------


def test_verbose_flag():
    """--verbose flag is accepted and calls setup_logging(verbose=True)."""
    with patch("nhf_spatial_targets.cli.setup_logging") as mock_setup:
        _run_meta("--verbose", "catalog", "sources")
    mock_setup.assert_called_once_with(True)


def test_default_no_verbose():
    """Without --verbose, setup_logging is called with False."""
    with patch("nhf_spatial_targets.cli.setup_logging") as mock_setup:
        _run_meta("catalog", "sources")
    mock_setup.assert_called_once_with(False)
```

Note: The `_sha256` directory hashing tests and `_fabric_metadata` parquet tests should be moved to `tests/test_validate.py` since those functions now live in `validate.py` (see Task 4, Step 3 note).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test -- tests/test_cli.py -v`
Expected: FAIL — old CLI still expects `--fabric`, `--run-dir`

- [ ] **Step 3: Rewrite `cli.py`**

Key changes to `src/nhf_spatial_targets/cli.py`:

1. Replace the `init` command: remove `--fabric`, `--id-col`, `--config`, `--id`, `--buffer` parameters. Add only `--workdir`. Call `init_workspace()` instead of `init_run()`.

2. Add `validate` command: takes `--workdir`, calls `validate_workspace()`.

3. Replace `--run-dir` with `--workdir` on `run` command. Add `fabric.json` existence check. Remove `--config` alternative.

4. Replace `--run-dir` with `--workdir` on all fetch commands. Update the parameter name in the call to the fetch function: `fetch_fn(workdir=workdir, period=period)`.

5. Update `fetch_all_cmd` similarly.

6. Remove `_DEFAULT_CONFIG` and `_DEFAULT_WORKDIR` constants.

The full `cli.py` rewrite is large. The implementer should:
- Change `init` to call `init_workspace(workdir)` from `nhf_spatial_targets.init_run`
- Add `validate` command calling `validate_workspace(workdir)` from `nhf_spatial_targets.validate`
- In `run`, replace `--run-dir` with `--workdir`. Remove `--config` alternative. Check for `fabric.json` before proceeding, read `config.yml` from workdir
- In every fetch command, rename parameter from `run_dir` to `workdir` and pass `workdir=workdir` to the fetch function
- In `fetch_all_cmd`, same `--run-dir` → `--workdir` rename, pass `workdir=workdir` to each fetch function
- Update console output messages in each `fetch_*_cmd` that hardcode `run_dir / 'data' / 'raw' / source_key` — these should use `workspace.load(workdir).raw_dir(source_key)` or just print a generic "Downloaded to datastore" message
- Remove `_DEFAULT_CONFIG` constant (no longer used). Remove `_DEFAULT_WORKDIR` constant.
- `_dispatch` signature: change `run_dir` parameter to `workdir`, pass it through to builders
- Keep `catalog_sources`, `catalog_variables`, and `launcher` functions with minimal changes

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev test -- tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Do NOT commit yet**

The CLI now expects `workdir` but fetch modules still use `run_dir`. We must update them together to keep the build green. Continue to the next tasks before committing.

---

## Chunk 4: Update fetch and consolidation modules (commit together with Chunk 3)

**Important:** Tasks 6-13 in this chunk plus Task 5 above are committed together as a single atomic change to keep the test suite passing at every commit.

### Task 5b: Update `consolidate.py` — `run_dir` → explicit `source_dir` parameter

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py`
- Modify: `tests/test_consolidate.py`
- Modify: `tests/test_consolidate_modis.py`

- [ ] **Step 1: Update all consolidation function signatures**

Each consolidation function currently takes `run_dir: Path` and internally computes `run_dir / "data" / "raw" / source_key`. Change these to take `source_dir: Path` directly (the caller passes `ws.raw_dir(source_key)`).

Functions to update:
- `consolidate_merra2(run_dir: Path, ...)` → `consolidate_merra2(source_dir: Path, ...)`
- `consolidate_nldas(run_dir: Path, source_key: str, ...)` → `consolidate_nldas(source_dir: Path, source_key: str, ...)`
- `consolidate_mod10c1(run_dir: Path, ...)` → `consolidate_mod10c1(source_dir: Path, ...)`
- `consolidate_mod16a2(run_dir: Path, ...)` → `consolidate_mod16a2(source_dir: Path, ...)`
- `consolidate_mod16a2_finalize(run_dir: Path, ...)` → `consolidate_mod16a2_finalize(source_dir: Path, ...)`
- `consolidate_ncep_ncar(run_dir: Path, ...)` → `consolidate_ncep_ncar(source_dir: Path, ...)`

For each: remove the internal `source_dir = run_dir / "data" / "raw" / key` line and use `source_dir` directly. Also update the return dict's `"consolidated_nc"` value — it was `str(out_path.relative_to(run_dir))` which no longer makes sense. Change to `str(out_path)` (absolute path) or `str(out_path.relative_to(source_dir))`.

- [ ] **Step 2: Update `tests/test_consolidate.py` and `tests/test_consolidate_modis.py`**

Change all `run_dir` fixture references to `source_dir`. Tests that previously created `run_dir / "data" / "raw" / key /` structures should now just pass the source directory directly.

- [ ] **Step 3: Run consolidation tests**

Run: `pixi run -e dev test -- tests/test_consolidate.py tests/test_consolidate_modis.py -v`
Expected: PASS

### Task 6: Update `_auth.py` — rename `run_dir` to `workdir`

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/_auth.py`
- Modify: `tests/test_auth.py`

- [ ] **Step 1: Update `_auth.py`**

Change `earthdata_login(run_dir: Path)` to `earthdata_login(workdir: Path)`.

The function reads `.credentials.yml` from `workdir`. The internal logic is identical — only the parameter name changes. The `run_dir` was already used as a directory path to find `.credentials.yml`, so this is a rename.

```python
def earthdata_login(workdir: Path) -> earthaccess.Auth:
    """Authenticate with NASA Earthdata.

    Reads credentials from ``workdir/.credentials.yml`` if available,
    otherwise falls back to earthaccess default login strategies.
    """
    creds_path = workdir / ".credentials.yml"
    # ... rest unchanged except error message references workdir
```

- [ ] **Step 2: Update `tests/test_auth.py`**

Rename any `run_dir` references to `workdir` in fixtures and assertions.

- [ ] **Step 3: Run auth tests**

Run: `pixi run -e dev test -- tests/test_auth.py -v`
Expected: PASS

- [ ] **Step 4: Do not commit yet — will be committed atomically in Task 12b**

### Task 7: Update `merra2.py` to use Workspace

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/merra2.py`
- Modify: `tests/test_merra2.py`

- [ ] **Step 1: Update `fetch_merra2()` signature and internals**

Change signature from `fetch_merra2(run_dir: Path, period: str)` to `fetch_merra2(workdir: Path, period: str)`.

Internal changes (search-and-replace pattern used by all fetch modules):
- `run_dir / "fabric.json"` → load fabric from workspace: `from nhf_spatial_targets.workspace import load; ws = load(workdir)`
- `run_dir / "data" / "raw" / _SOURCE_KEY` → `ws.raw_dir(_SOURCE_KEY)`
- `run_dir / "manifest.json"` → `ws.manifest_path`
- `earthdata_login(run_dir)` → `earthdata_login(workdir)`
- `fabric["bbox_buffered"]` → `ws.fabric["bbox_buffered"]`
- The helper functions `_manifest_merra2_files()`, `_existing_months()`, `_existing_file_timestamps()` change from `run_dir: Path` to `workdir: Path`, reading manifest from `workdir / "manifest.json"` (or use `ws.manifest_path`). Since these are called before `ws` is loaded in some cases, keep them reading from `workdir / "manifest.json"` directly.
- `_update_manifest()` changes `run_dir` param to `workdir`.

- [ ] **Step 2: Update `tests/test_merra2.py`**

Change all test fixtures from `run_dir` to `workdir`. Tests that create `fabric.json` and `manifest.json` should also create a `config.yml` with a `datastore` pointing to a temp dir. Or, since the tests mock the fetch function, they may only need the parameter rename.

- [ ] **Step 3: Run merra2 tests**

Run: `pixi run -e dev test -- tests/test_merra2.py -v`
Expected: PASS

- [ ] **Step 4: Do not commit yet — will be committed atomically in Task 12b**

### Task 8: Update `nldas.py` to use Workspace

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/nldas.py`
- Modify: `tests/test_nldas.py`

- [ ] **Step 1: Apply the same pattern as Task 7**

Change `fetch_nldas_mosaic(run_dir: Path, period: str)` and `fetch_nldas_noah(run_dir: Path, period: str)` to use `workdir: Path`.

Apply the same internal changes:
- `run_dir / "fabric.json"` → load via workspace or direct read
- `run_dir / "data" / "raw" / source_key` → compute from config's datastore path
- `run_dir / "manifest.json"` → `workdir / "manifest.json"`
- `earthdata_login(run_dir)` → `earthdata_login(workdir)`

- [ ] **Step 2: Update tests**

- [ ] **Step 3: Run tests**

Run: `pixi run -e dev test -- tests/test_nldas.py -v`
Expected: PASS

- [ ] **Step 4: Do not commit yet — will be committed atomically in Task 12b**

### Task 9: Update `ncep_ncar.py` to use Workspace

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/ncep_ncar.py`
- Modify: `tests/test_ncep_ncar.py`

- [ ] **Step 1: Apply the same pattern as Tasks 7-8**

`ncep_ncar.py` does not use `earthdata_login` (NOAA PSL is public), but it does read `fabric.json` and `manifest.json` from `run_dir`. Apply the same rename and path changes.

- [ ] **Step 2: Update tests**

- [ ] **Step 3: Run tests**

Run: `pixi run -e dev test -- tests/test_ncep_ncar.py -v`
Expected: PASS

- [ ] **Step 4: Do not commit yet — will be committed atomically in Task 12b**

### Task 10: Update `modis.py` to use Workspace

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/modis.py`
- Modify: `tests/test_modis.py`

- [ ] **Step 1: Apply the same pattern**

Both `fetch_mod16a2()` and `fetch_mod10c1()` need `run_dir` → `workdir` rename plus path changes. These modules use `earthdata_login(run_dir)`.

- [ ] **Step 2: Update tests**

- [ ] **Step 3: Run tests**

Run: `pixi run -e dev test -- tests/test_modis.py -v`
Expected: PASS

- [ ] **Step 4: Do not commit yet — will be committed atomically in Task 12b**

### Task 11: Update `pangaea.py` to use Workspace

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/pangaea.py`
- Modify: `tests/test_pangaea.py`

- [ ] **Step 1: Apply the same pattern**

`pangaea.py` does not use `earthdata_login` (PANGAEA is public). Apply `run_dir` → `workdir` rename and path changes.

- [ ] **Step 2: Update tests**

- [ ] **Step 3: Run tests**

Run: `pixi run -e dev test -- tests/test_pangaea.py -v`
Expected: PASS

- [ ] **Step 4: Do not commit yet — will be committed atomically in Task 12b**

### Task 12: Update `reitz2017.py` to use Workspace

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/reitz2017.py`
- Modify: `tests/test_reitz2017.py`

- [ ] **Step 1: Apply the same pattern**

`reitz2017.py` does not use `earthdata_login`. Apply `run_dir` → `workdir` rename and path changes.

- [ ] **Step 2: Update tests**

- [ ] **Step 3: Run tests**

Run: `pixi run -e dev test -- tests/test_reitz2017.py -v`
Expected: PASS

- [ ] **Step 4: Do not commit yet — will be committed atomically in Task 12b**

### Task 12b: Atomic commit for CLI + all fetch/consolidation changes

- [ ] **Step 1: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test -v`
Expected: ALL PASS

- [ ] **Step 2: Commit everything together**

```bash
git add src/nhf_spatial_targets/cli.py tests/test_cli.py \
  src/nhf_spatial_targets/fetch/_auth.py tests/test_auth.py \
  src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate.py tests/test_consolidate_modis.py \
  src/nhf_spatial_targets/fetch/merra2.py tests/test_merra2.py \
  src/nhf_spatial_targets/fetch/nldas.py tests/test_nldas.py \
  src/nhf_spatial_targets/fetch/ncep_ncar.py tests/test_ncep_ncar.py \
  src/nhf_spatial_targets/fetch/modis.py tests/test_modis.py \
  src/nhf_spatial_targets/fetch/pangaea.py tests/test_pangaea.py \
  src/nhf_spatial_targets/fetch/reitz2017.py tests/test_reitz2017.py
git commit -m "refactor: CLI uses --workdir, all fetch/consolidate modules use workdir"
```

---

## Chunk 5: Cleanup and Documentation

### Task 13: Update `pixi.toml` task commands

**Files:**
- Modify: `pixi.toml`

- [ ] **Step 1: Update pixi task comments and commands**

- Update init task comment: `# Usage: pixi run init -- --workdir /data/nhf-runs/my-run`
- Update run task comment: `# Usage: pixi run run -- --workdir /data/nhf-runs/my-run`
- Add validate task: `validate = { cmd = "nhf-targets validate", description = "Validate workspace config and fabric" }`
- Update any individual target run tasks that reference `--run-dir` to use `--workdir`

- [ ] **Step 2: Commit**

```bash
git add pixi.toml
git commit -m "chore: update pixi.toml tasks for --workdir and add validate"
```

### Task 14: Remove `data/` placeholder directory

**Files:**
- Remove: `data/` directory from repo root

- [ ] **Step 1: Remove the empty data directory**

```bash
rm -rf data/
```

Note: Since `data/` is gitignored, this may or may not be tracked. If there's a `.gitkeep` file, remove it. If the directory isn't tracked at all, this is a no-op.

- [ ] **Step 2: Commit if there were tracked changes**

```bash
git add -A data/
git commit -m "chore: remove empty data/ placeholder directory"
```

### Task 15: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Environment & Commands section**

Replace the init/run examples:

```markdown
# Create a workspace
pixi run init -- --workdir /data/nhf-runs/my-run

# Edit config.yml to set fabric path, datastore, and credentials
# Then validate the workspace
pixi run validate -- --workdir /data/nhf-runs/my-run

# Run the full pipeline
pixi run run -- --workdir /data/nhf-runs/my-run

# Run a single target
pixi run run-aet -- --workdir /data/nhf-runs/my-run
```

- [ ] **Step 2: Update Repository Layout section**

Add `workspace.py` and `validate.py`. Remove `data/` placeholder mention.

- [ ] **Step 3: Update Data Provenance & Run Workspaces section**

Replace the current content to describe:
- `nhf-targets init --workdir <dir>` creates workspace skeleton + config template
- User edits `config.yml` to set fabric path, datastore, dir_mode
- `nhf-targets validate --workdir <dir>` runs preflight checks, writes `fabric.json`
- Raw downloads live at `<datastore>/<source_key>/` (shared, fabric-independent)
- Aggregated/target outputs in `<workdir>/data/aggregated/` and `<workdir>/targets/`
- Remove: symlink-to-prior-run description, run ID format, `--fabric` flag docs

- [ ] **Step 4: Update Known Gaps section if needed**

No changes expected — gaps are tracked in catalog.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for workspace/datastore refactor"
```

### Task 16: Keep `config/pipeline.yml` as reference

**Files:**
- Modify: `config/pipeline.yml`

- [ ] **Step 1: Add a header comment**

Add a comment at the top of `config/pipeline.yml`:
```yaml
# REFERENCE ONLY — This file is no longer used directly by the CLI.
# Workspace config is generated by 'nhf-targets init' into <workdir>/config.yml.
# This file is retained as a reference for the default configuration.
```

- [ ] **Step 2: Commit**

```bash
git add config/pipeline.yml
git commit -m "docs: mark config/pipeline.yml as reference-only"
```

### Task 17: Run full test suite and lint

- [ ] **Step 1: Format and lint**

Run: `pixi run -e dev fmt && pixi run -e dev lint`
Expected: PASS with no issues

- [ ] **Step 2: Run full test suite**

Run: `pixi run -e dev test -v`
Expected: ALL PASS

- [ ] **Step 3: Fix any failures**

If any tests fail, fix them and re-run.

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: address test/lint issues from workspace refactor"
```
