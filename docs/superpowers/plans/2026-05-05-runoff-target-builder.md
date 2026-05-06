# Runoff Target Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the runoff calibration-target builder using all three configured sources (ERA5-Land, GLDAS-2.1 NOAH, MWBM ClimGrid) with period-union semantics and optional NN-fill post-processing, write CF-1.6 compliant NetCDF output, introduce a project-wide config defaults layer for all five targets, add `run_all.slurm`, and update related docs/configs to be consistent.

**Architecture:** A new `targets/_common.py` module owns the shared multi-source-minmax target machinery (per-year NC reading, time canonicalization, NaN-aware combination, CF output writing). `targets/run.py` is rewritten to declare the three runoff sources, their per-source unit-conversion shims (existing helpers retained), and orchestrate via `_common`. NN-fill lives in `normalize/methods.py` and produces a separate `*_nn_filled.nc` artifact. A new `defaults.py` + `workspace.py` merge layer makes existing project `config.yml` files continue to work after new keys land; `validate.py` reports the diff and writes `config.effective.yml`.

**Tech Stack:** Python 3.11+, xarray + dask (lazy reads via `open_mfdataset`), geopandas (fabric reprojection / centroids / area), scipy.spatial.cKDTree (NN search), pyyaml, cyclopts (CLI), pytest, ruff. Spec at `docs/superpowers/specs/2026-05-05-runoff-target-builder-design.md`.

**Project conventions to respect:**
- All commands run via `pixi run` (not bare `python`/`pytest`).
- Commits via `pixi run git commit` (a PreToolUse hook blocks bare `git commit` for Claude sessions); pass message via heredoc.
- Quality gate before each commit: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`.
- xarray objects opened from disk must be `.load()`ed and `.close()`d (try/finally); functions returning a Dataset must detach it from disk first.
- Read units from `nhf_spatial_targets.catalog`, not from on-disk NetCDF `attrs["units"]`.
- `from __future__ import annotations` in every new module; type hints on all public functions; docstrings on public functions only (not stubs).
- Ruff line length 88.

---

## Task 1: Create `defaults.py` with DEFAULTS schema and merge helper

**Files:**
- Create: `src/nhf_spatial_targets/defaults.py`
- Test: `tests/test_defaults.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_defaults.py`:

```python
"""Tests for the project-config defaults layer."""

from __future__ import annotations

import pytest

from nhf_spatial_targets.defaults import (
    DEFAULTS,
    REQUIRED,
    apply_defaults,
    find_unknown_keys,
    iter_default_diff,
    missing_required,
)


def test_defaults_has_all_five_targets():
    """Every calibration target has a defaults block."""
    assert set(DEFAULTS["targets"].keys()) == {
        "runoff",
        "aet",
        "recharge",
        "soil_moisture",
        "snow_covered_area",
    }


def test_runoff_defaults_include_all_three_sources():
    """Runoff defaults to era5+gldas+mwbm per the design."""
    assert DEFAULTS["targets"]["runoff"]["sources"] == [
        "era5_land",
        "gldas_noah_v21_monthly",
        "mwbm_climgrid",
    ]


def test_runoff_defaults_nn_fill_on_by_default():
    assert DEFAULTS["targets"]["runoff"]["nn_fill"] is True
    assert DEFAULTS["targets"]["runoff"]["nn_max_candidates"] == 10
    assert DEFAULTS["targets"]["runoff"]["chunk_months"] == 12


def test_fabric_area_crs_default_is_conus_albers():
    assert DEFAULTS["fabric"]["area_crs"] == "EPSG:5070"


def test_apply_defaults_fills_missing_leaf_keys():
    """User config wins; missing keys take defaults."""
    user = {"datastore": "/data", "fabric": {"path": "/p", "id_col": "nhm_id"}}
    merged = apply_defaults(user)
    assert merged["datastore"] == "/data"
    assert merged["fabric"]["path"] == "/p"
    assert merged["fabric"]["area_crs"] == "EPSG:5070"  # filled from defaults
    assert merged["targets"]["runoff"]["nn_fill"] is True


def test_apply_defaults_user_wins_at_leaves():
    user = {
        "datastore": "/data",
        "fabric": {"path": "/p", "id_col": "nhm_id", "area_crs": "EPSG:3338"},
    }
    merged = apply_defaults(user)
    assert merged["fabric"]["area_crs"] == "EPSG:3338"


def test_apply_defaults_lists_replace_wholesale():
    """A user-set list is NOT merged with the default list."""
    user = {
        "datastore": "/data",
        "fabric": {"path": "/p", "id_col": "nhm_id"},
        "targets": {"runoff": {"sources": ["era5_land"]}},
    }
    merged = apply_defaults(user)
    assert merged["targets"]["runoff"]["sources"] == ["era5_land"]


def test_iter_default_diff_lists_filled_keys():
    """Defaults that were filled in are reported with dotted paths."""
    user = {"datastore": "/data", "fabric": {"path": "/p", "id_col": "nhm_id"}}
    diff = list(iter_default_diff(user))
    paths = {p for p, _ in diff}
    assert "fabric.area_crs" in paths
    assert "targets.runoff.nn_fill" in paths
    # Keys the user did set should NOT appear:
    assert "datastore" not in paths


def test_find_unknown_keys_returns_typos():
    """Keys not present in DEFAULTS are reported."""
    user = {
        "datastore": "/data",
        "fabric": {"path": "/p", "id_col": "nhm_id"},
        "targets": {"runoff": {"nn_fil": True}},  # typo
    }
    unknown = find_unknown_keys(user)
    assert "targets.runoff.nn_fil" in unknown


def test_missing_required_reports_paths():
    """Missing required keys return their dotted paths."""
    user = {"fabric": {"id_col": "nhm_id"}}  # no datastore, no fabric.path
    missing = missing_required(apply_defaults(user))
    assert "datastore" in missing
    assert "fabric.path" in missing


def test_missing_required_runoff_period_when_enabled():
    user = {
        "datastore": "/d",
        "fabric": {"path": "/p", "id_col": "nhm_id"},
        # no targets.runoff.period set; runoff.enabled=True by default
    }
    missing = missing_required(apply_defaults(user))
    assert "targets.runoff.period" in missing


def test_missing_required_skips_period_when_disabled():
    user = {
        "datastore": "/d",
        "fabric": {"path": "/p", "id_col": "nhm_id"},
        "targets": {"runoff": {"enabled": False}},
    }
    missing = missing_required(apply_defaults(user))
    assert "targets.runoff.period" not in missing


def test_required_paths_includes_datastore_and_fabric_path():
    """REQUIRED is the canonical list of always-required dotted paths."""
    assert ("datastore",) in REQUIRED
    assert ("fabric", "path") in REQUIRED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev test tests/test_defaults.py -v`
Expected: FAIL — `ModuleNotFoundError: nhf_spatial_targets.defaults`

- [ ] **Step 3: Write `src/nhf_spatial_targets/defaults.py`**

```python
"""Project-config defaults: single source of truth for every settable key.

The merge is non-modifying: ``workspace.load()`` overlays user config from
``config.yml`` onto :data:`DEFAULTS` so old project directories continue to
work after new keys land. The merged dict is what every downstream consumer
sees; the on-disk ``config.yml`` is never touched.

Lists in the user config replace the default list wholesale (no item-wise
merge), so e.g. ``targets.runoff.sources: [era5_land]`` means just that one
source — no surprise inclusion of the default three.

A leaf value of ``None`` in :data:`DEFAULTS` means "no default — required."
"""

from __future__ import annotations

import copy
from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DEFAULTS: dict = {
    "fabric": {
        "path": None,  # required
        "id_col": "nhm_id",
        "crs": "EPSG:4326",
        "buffer_deg": 0.1,
        # Equal-area CRS used for HRU area + NN-fill distances. Override for
        # AK / HI / PR (e.g. EPSG:3338 for Alaska Albers).
        "area_crs": "EPSG:5070",
    },
    "datastore": None,  # required
    "dir_mode": "2775",
    "aggregation": {"engine": "gdptools", "method": "area_weighted"},
    "output": {"dir": "outputs", "format": "netcdf", "compress": True},
    "targets": {
        "runoff": {
            "enabled": True,
            "sources": [
                "era5_land",
                "gldas_noah_v21_monthly",
                "mwbm_climgrid",
            ],
            "time_step": "monthly",
            "period": None,  # required when enabled
            "prms_variable": "basin_cfs",
            "range_method": "multi_source_minmax",
            "output_file": "runoff_targets.nc",
            "nn_fill": True,
            "nn_max_candidates": 10,
            "chunk_months": 12,
        },
        "aet": {
            "enabled": True,
            "sources": ["mod16a2_v061", "ssebop", "mwbm_climgrid"],
            "time_step": "monthly",
            "period": None,
            "prms_variable": "hru_actet",
            "range_method": "multi_source_minmax",
            "output_file": "aet_targets.nc",
            "nn_fill": True,
            "nn_max_candidates": 10,
            "chunk_months": 12,
        },
        "recharge": {
            "enabled": True,
            "sources": ["reitz2017", "watergap22d", "era5_land"],
            "time_step": "annual",
            "period": None,
            "prms_variable": "recharge",
            "range_method": "normalized_minmax",
            "normalize": True,
            "normalize_period": "2000-01-01/2009-12-31",
            "output_file": "recharge_targets.nc",
            "nn_fill": True,
            "nn_max_candidates": 10,
        },
        "soil_moisture": {
            "enabled": True,
            "sources": ["merra2", "ncep_ncar", "nldas_mosaic", "nldas_noah"],
            "time_step": ["monthly", "annual"],
            "period": None,
            "prms_variable": "soil_rechr",
            "range_method": "normalized_minmax",
            "normalize": True,
            "normalize_by": "calendar_month",
            "output_file": "soil_moisture_targets.nc",
            "nn_fill": True,
            "nn_max_candidates": 10,
        },
        "snow_covered_area": {
            "enabled": True,
            "sources": ["mod10c1_v061"],
            "time_step": "daily",
            "period": None,
            "prms_variable": "snowcov_area",
            "range_method": "modis_ci",
            "ci_threshold": 0.70,
            "output_file": "sca_targets.nc",
            "nn_fill": True,
            "nn_max_candidates": 10,
        },
    },
}


# Always-required dotted paths (independent of any toggle).
REQUIRED: list[tuple[str, ...]] = [
    ("datastore",),
    ("fabric", "path"),
]

# Per-target paths that are required iff the target is enabled.
_REQUIRED_PER_TARGET: list[tuple[str, ...]] = [
    ("period",),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_defaults(user: dict | None) -> dict:
    """Return ``user`` deep-merged onto :data:`DEFAULTS`.

    User values win at every leaf. Nested dicts recurse. Lists in ``user``
    replace the default list wholesale (no item-wise merge), so a user-set
    ``sources: [era5_land]`` yields exactly that one element.

    A ``None`` value in the user config is treated as "not set" and falls
    through to the default; pass an explicit empty string or empty dict if
    you genuinely want to suppress a default.
    """
    return _deep_merge(DEFAULTS, user or {})


def iter_default_diff(user: dict | None) -> Iterator[tuple[str, object]]:
    """Yield ``(dotted_path, default_value)`` for every key the user did not set.

    Walks the merged dict and reports leaves where ``user`` had no value.
    Used by ``validate`` to print which defaults took effect.
    """
    user = user or {}
    yield from _walk_diff(DEFAULTS, user, prefix=())


def find_unknown_keys(user: dict | None) -> list[str]:
    """Return dotted paths for user-set keys not present in :data:`DEFAULTS`.

    Catches typos like ``runoff.nn_fil``. Lists are leaves (their items are
    not introspected). Top-level keys not in DEFAULTS are reported as-is.
    """
    user = user or {}
    return list(_walk_unknown(DEFAULTS, user, prefix=()))


def missing_required(merged: dict) -> list[str]:
    """Return dotted paths for required keys missing or ``None`` in ``merged``.

    Includes always-required paths (``datastore``, ``fabric.path``) plus
    per-target ``period`` for targets where ``enabled`` is True.
    """
    missing: list[str] = []
    for path in REQUIRED:
        if _get(merged, path) in (None, ""):
            missing.append(".".join(path))
    targets = merged.get("targets", {}) or {}
    for tname, tcfg in targets.items():
        if not isinstance(tcfg, dict) or not tcfg.get("enabled", False):
            continue
        for path in _REQUIRED_PER_TARGET:
            if _get(tcfg, path) in (None, ""):
                missing.append(".".join(("targets", tname, *path)))
    return missing


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _walk_diff(
    defaults: dict, user: dict, prefix: tuple[str, ...]
) -> Iterator[tuple[str, object]]:
    for k, dv in defaults.items():
        path = (*prefix, k)
        uv = user.get(k) if isinstance(user, dict) else None
        if isinstance(dv, dict):
            if isinstance(uv, dict):
                yield from _walk_diff(dv, uv, path)
            else:
                # User did not set this whole subtree; report each leaf default.
                yield from _walk_diff(dv, {}, path)
        else:
            if uv is None:
                yield (".".join(path), dv)


def _walk_unknown(
    defaults: dict, user: dict, prefix: tuple[str, ...]
) -> Iterator[str]:
    if not isinstance(user, dict):
        return
    for k, uv in user.items():
        path = (*prefix, k)
        if k not in defaults:
            yield ".".join(path)
            continue
        dv = defaults[k]
        if isinstance(dv, dict) and isinstance(uv, dict):
            yield from _walk_unknown(dv, uv, path)


def _get(d: dict, path: tuple[str, ...]) -> object:
    cur: object = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev test tests/test_defaults.py -v`
Expected: PASS for all 12 tests.

- [ ] **Step 5: Format, lint, full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
pixi run git add src/nhf_spatial_targets/defaults.py tests/test_defaults.py
pixi run git commit -m "$(cat <<'EOF'
feat(config): add defaults schema for all five targets

Single-source-of-truth DEFAULTS dict with deep-merge / required-key /
unknown-key / diff helpers. None-leaf marks a value as required. Lists
replace wholesale so user-set sources override defaults cleanly.

Foundation for non-modifying config migration: existing project.yml
files continue to work after new keys are introduced.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Wire `workspace.load()` to apply defaults

**Files:**
- Modify: `src/nhf_spatial_targets/workspace.py`
- Test: `tests/test_workspace_defaults.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_workspace_defaults.py`:

```python
"""Tests for the workspace defaults integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from nhf_spatial_targets.workspace import load


def _write_project(
    tmp_path: Path, *, config: dict, fabric: dict | None = None
) -> Path:
    workdir = tmp_path / "proj"
    workdir.mkdir()
    (workdir / "config.yml").write_text(yaml.safe_dump(config))
    (workdir / "fabric.json").write_text(json.dumps(fabric or {"id_col": "nhm_id"}))
    return workdir


def test_load_applies_defaults(tmp_path: Path):
    workdir = _write_project(
        tmp_path,
        config={
            "datastore": str(tmp_path / "store"),
            "fabric": {"path": "/some/fabric.gpkg", "id_col": "nhm_id"},
        },
    )
    project = load(workdir)
    # area_crs default applied:
    assert project.area_crs == "EPSG:5070"
    # Per-target merge available via Project.target():
    runoff = project.target("runoff")
    assert runoff["nn_fill"] is True
    assert runoff["sources"] == [
        "era5_land",
        "gldas_noah_v21_monthly",
        "mwbm_climgrid",
    ]


def test_load_user_overrides_area_crs(tmp_path: Path):
    workdir = _write_project(
        tmp_path,
        config={
            "datastore": str(tmp_path / "store"),
            "fabric": {
                "path": "/p",
                "id_col": "nhm_id",
                "area_crs": "EPSG:3338",
            },
        },
    )
    project = load(workdir)
    assert project.area_crs == "EPSG:3338"


def test_load_target_user_sources_replaces_default(tmp_path: Path):
    workdir = _write_project(
        tmp_path,
        config={
            "datastore": str(tmp_path / "store"),
            "fabric": {"path": "/p", "id_col": "nhm_id"},
            "targets": {"runoff": {"sources": ["era5_land"]}},
        },
    )
    project = load(workdir)
    assert project.target("runoff")["sources"] == ["era5_land"]


def test_load_unknown_target_raises(tmp_path: Path):
    workdir = _write_project(
        tmp_path,
        config={
            "datastore": str(tmp_path / "store"),
            "fabric": {"path": "/p", "id_col": "nhm_id"},
        },
    )
    project = load(workdir)
    with pytest.raises(KeyError, match="not_a_target"):
        project.target("not_a_target")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev test tests/test_workspace_defaults.py -v`
Expected: FAIL — `Project` has no `area_crs` / `target` attributes.

- [ ] **Step 3: Modify `src/nhf_spatial_targets/workspace.py`**

Replace the file body with:

```python
"""Project path resolution and directory helpers."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import yaml

from nhf_spatial_targets.defaults import apply_defaults

_IS_UNIX = platform.system() != "Windows"


def make_dir(path: Path, *, dir_mode: int | None = None) -> Path:
    """Create a directory (with parents), optionally applying Unix permissions."""
    path.mkdir(parents=True, exist_ok=True)
    if dir_mode is not None and _IS_UNIX:
        os.chmod(path, dir_mode)
    return path


@dataclass(frozen=True)
class Project:
    """Resolved project with validated paths."""

    workdir: Path
    datastore: Path
    config: dict
    fabric: dict
    dir_mode: int | None

    def raw_dir(self, source_key: str) -> Path:
        """Return the raw data directory for a source."""
        return self.datastore / source_key

    def aggregated_dir(self) -> Path:
        """Return the aggregated data directory."""
        return self.workdir / "data" / "aggregated"

    def targets_dir(self) -> Path:
        """Return the targets output directory."""
        return self.workdir / "targets"

    @property
    def manifest_path(self) -> Path:
        """Return path to manifest.json."""
        return self.workdir / "manifest.json"

    @property
    def credentials_path(self) -> Path:
        """Return path to .credentials.yml."""
        return self.workdir / ".credentials.yml"

    @property
    def area_crs(self) -> str:
        """Equal-area CRS used for HRU area + NN-fill distances."""
        return self.config["fabric"]["area_crs"]

    @property
    def id_col(self) -> str:
        """Fabric ID column name (also the HRU dim in aggregated NCs)."""
        return self.config["fabric"]["id_col"]

    def target(self, name: str) -> dict:
        """Return the merged config sub-dict for a calibration target.

        Raises ``KeyError`` if ``name`` is not a recognized target.
        """
        targets = self.config.get("targets", {})
        if name not in targets:
            raise KeyError(
                f"Unknown target '{name}'. Known: {sorted(targets.keys())}"
            )
        return targets[name]


def load(workdir: Path) -> Project:
    """Load a project, merging user config over defaults from ``defaults.py``.

    Reads ``config.yml`` and ``fabric.json``, deep-merges the user config over
    :data:`nhf_spatial_targets.defaults.DEFAULTS` (user wins at every leaf;
    lists replace wholesale; ``None`` user values fall through to defaults),
    and returns a :class:`Project` carrying the merged config.

    Required keys (``datastore``, ``fabric.path``) are checked here so that
    every consumer of ``Project`` can assume they exist.
    """
    config_path = workdir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yml not found in {workdir}. "
            f"Run 'nhf-targets init --project-dir {workdir}' first."
        )
    try:
        user_config = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Cannot parse config.yml in {workdir}: {exc}") from exc
    if user_config is not None and not isinstance(user_config, dict):
        raise ValueError(
            f"config.yml in {workdir} is empty or malformed. "
            f"It must contain YAML key-value pairs."
        )

    config = apply_defaults(user_config)

    if not config.get("datastore"):
        raise ValueError(
            f"'datastore' key missing from config.yml in {workdir}. "
            f"This field is required. Edit config.yml and add the datastore path."
        )

    fabric_path = workdir / "fabric.json"
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"fabric.json not found in {workdir}. "
            f"Run 'nhf-targets validate --project-dir {workdir}' first."
        )
    fabric = json.loads(fabric_path.read_text())

    dir_mode_str = config.get("dir_mode")
    if dir_mode_str:
        try:
            dir_mode = int(dir_mode_str, 8)
        except ValueError:
            raise ValueError(
                f"Invalid dir_mode '{dir_mode_str}' in config.yml. "
                f"Expected an octal string like '2775'."
            ) from None
    else:
        dir_mode = None

    return Project(
        workdir=workdir,
        datastore=Path(config["datastore"]),
        config=config,
        fabric=fabric,
        dir_mode=dir_mode,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev test tests/test_workspace_defaults.py -v`
Expected: PASS for all 4 tests.

- [ ] **Step 5: Run full test suite for regressions**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: PASS. (If any pre-existing tests broke because they passed un-merged config to `Project`, fix them by either constructing `Project` directly with a merged dict or routing through `load()` with a real project skeleton.)

- [ ] **Step 6: Commit**

```bash
pixi run git add src/nhf_spatial_targets/workspace.py tests/test_workspace_defaults.py
pixi run git commit -m "$(cat <<'EOF'
feat(workspace): merge user config over defaults at load time

Project.config now carries the fully-merged dict; new Project.target(),
Project.area_crs, and Project.id_col surface the most-asked values
without making consumers grovel through nested dicts. Existing projects
that lack new keys (area_crs, target.nn_fill, etc.) continue to work
unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire `validate.py` to report defaults diff, warn on unknown keys, write `config.effective.yml`

**Files:**
- Modify: `src/nhf_spatial_targets/validate.py`
- Test: `tests/test_validate_defaults.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_validate_defaults.py`:

```python
"""Tests for the validate-time defaults diff and config.effective.yml."""

from __future__ import annotations

import os
import platform
from pathlib import Path

import pytest
import yaml


def _make_minimal_project(tmp_path: Path) -> Path:
    """Skeleton sufficient for the defaults/effective-config code paths.

    Skips the full validate path (fabric/credentials) by directly invoking
    the pure-config helpers exposed by the validate module.
    """
    workdir = tmp_path / "proj"
    workdir.mkdir()
    (workdir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "datastore": str(tmp_path / "store"),
                "fabric": {
                    "path": str(tmp_path / "fabric.gpkg"),
                    "id_col": "nhm_id",
                },
            }
        )
    )
    return workdir


def test_report_defaults_diff_lists_filled_keys(tmp_path: Path, capsys):
    from nhf_spatial_targets.validate import _report_defaults_and_unknowns

    workdir = _make_minimal_project(tmp_path)
    user_cfg = yaml.safe_load((workdir / "config.yml").read_text())
    _report_defaults_and_unknowns(user_cfg)
    err = capsys.readouterr().err
    assert "[defaults] fabric.area_crs" in err
    assert "[defaults] targets.runoff.nn_fill" in err


def test_report_unknown_keys_warns(tmp_path: Path, capsys):
    from nhf_spatial_targets.validate import _report_defaults_and_unknowns

    user_cfg = {
        "datastore": "/d",
        "fabric": {"path": "/p", "id_col": "nhm_id"},
        "targets": {"runoff": {"nn_fil": True}},  # typo
    }
    _report_defaults_and_unknowns(user_cfg)
    err = capsys.readouterr().err
    assert "[warning] unknown config key: targets.runoff.nn_fil" in err


def test_write_effective_config_writes_merged_yaml(tmp_path: Path):
    from nhf_spatial_targets.validate import _write_effective_config

    workdir = _make_minimal_project(tmp_path)
    user_cfg = yaml.safe_load((workdir / "config.yml").read_text())
    _write_effective_config(workdir, user_cfg)

    out = workdir / "config.effective.yml"
    assert out.exists()
    text = out.read_text()
    assert text.startswith("# AUTOGENERATED")
    body = yaml.safe_load(text)
    assert body["fabric"]["area_crs"] == "EPSG:5070"
    assert body["targets"]["runoff"]["sources"] == [
        "era5_land",
        "gldas_noah_v21_monthly",
        "mwbm_climgrid",
    ]


@pytest.mark.skipif(platform.system() == "Windows", reason="POSIX file modes only")
def test_write_effective_config_is_read_only(tmp_path: Path):
    from nhf_spatial_targets.validate import _write_effective_config

    workdir = _make_minimal_project(tmp_path)
    user_cfg = yaml.safe_load((workdir / "config.yml").read_text())
    _write_effective_config(workdir, user_cfg)

    mode = (workdir / "config.effective.yml").stat().st_mode & 0o777
    assert mode == 0o444


def test_write_effective_config_overwrites_existing(tmp_path: Path):
    """Read-only mode on the existing file must not block re-write."""
    from nhf_spatial_targets.validate import _write_effective_config

    workdir = _make_minimal_project(tmp_path)
    user_cfg = yaml.safe_load((workdir / "config.yml").read_text())
    _write_effective_config(workdir, user_cfg)

    # Mutate the user config and write again — must succeed.
    user_cfg["fabric"]["area_crs"] = "EPSG:3338"
    _write_effective_config(workdir, user_cfg)
    body = yaml.safe_load((workdir / "config.effective.yml").read_text())
    assert body["fabric"]["area_crs"] == "EPSG:3338"


def test_check_required_raises_on_missing_period(tmp_path: Path):
    from nhf_spatial_targets.validate import _check_required_keys

    user_cfg = {
        "datastore": "/d",
        "fabric": {"path": "/p", "id_col": "nhm_id"},
        # runoff.enabled True by default; no period -> should raise
    }
    with pytest.raises(ValueError, match="targets.runoff.period"):
        _check_required_keys(user_cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev test tests/test_validate_defaults.py -v`
Expected: FAIL — `_report_defaults_and_unknowns`, `_write_effective_config`, `_check_required_keys` not defined.

- [ ] **Step 3: Modify `src/nhf_spatial_targets/validate.py`**

Add these new private helpers at the top of the module (just below the existing `_SOURCE_KEYS` constant):

```python
import os
import sys
import stat


def _report_defaults_and_unknowns(user_cfg: dict | None) -> None:
    """Print to stderr which keys took defaults and which are unknown.

    Called once per ``validate`` run so the user sees the migration impact
    of new defaulted keys without their on-disk config being modified.
    """
    from nhf_spatial_targets.defaults import find_unknown_keys, iter_default_diff

    for path, value in iter_default_diff(user_cfg):
        print(f"[defaults] {path} not set; using default: {value!r}", file=sys.stderr)
    for path in find_unknown_keys(user_cfg):
        print(
            f"[warning] unknown config key: {path} (typo? not in defaults schema)",
            file=sys.stderr,
        )


def _check_required_keys(user_cfg: dict | None) -> dict:
    """Apply defaults and raise if any required keys are missing.

    Returns the merged config so callers can re-use it without re-merging.
    """
    from nhf_spatial_targets.defaults import apply_defaults, missing_required

    merged = apply_defaults(user_cfg)
    missing = missing_required(merged)
    if missing:
        raise ValueError(
            "Required config keys missing or empty:\n  - "
            + "\n  - ".join(missing)
            + "\nEdit config.yml and re-run validate."
        )
    return merged


def _write_effective_config(workdir: Path, user_cfg: dict | None) -> Path:
    """Write the fully-merged config to ``<workdir>/config.effective.yml``.

    Always rewritten on every validate run; mode 0o444 so manual edits get a
    clear permission error (the source-of-truth file is ``config.yml``). The
    existing file is unlinked first so the read-only mode does not block the
    re-write.
    """
    from nhf_spatial_targets.defaults import apply_defaults

    merged = apply_defaults(user_cfg)
    body = (
        "# AUTOGENERATED by nhf-targets validate. Do not edit; edit config.yml.\n"
        + yaml.safe_dump(merged, sort_keys=False)
    )
    out = workdir / "config.effective.yml"
    if out.exists():
        out.chmod(0o644)  # so unlink succeeds on systems with strict perms
        out.unlink()
    out.write_text(body)
    if platform.system() != "Windows":
        out.chmod(0o444)
    return out
```

Now wire them into `validate_workspace`. Replace the body:

```python
def validate_workspace(workdir: Path) -> None:
    """Run preflight checks on a project directory, failing fast on errors.

    On success, writes ``fabric.json``, ``manifest.json``, and
    ``config.effective.yml`` into *workdir*. Reports default-application
    and unknown-key warnings to stderr.
    """
    workdir = Path(workdir).resolve()

    # 0. Load raw user config (without merging defaults yet — we want to know
    #    what the user set vs. what came from defaults).
    config_path = workdir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yml not found in {workdir}. "
            f"Run 'nhf-targets init --project-dir {workdir}' first."
        )
    try:
        user_cfg = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Cannot parse config.yml in {workdir}: {exc}") from exc
    if user_cfg is not None and not isinstance(user_cfg, dict):
        raise ValueError(
            f"config.yml in {workdir} is empty or malformed. "
            f"It must contain YAML key-value pairs."
        )

    # 1. Report defaults diff and unknown keys (informational), then
    #    enforce required-key presence on the merged result.
    _report_defaults_and_unknowns(user_cfg)
    config = _check_required_keys(user_cfg)

    # 2. Fabric exists
    fabric_path = Path(config["fabric"]["path"])
    id_col = config["fabric"]["id_col"]
    _check_fabric_exists(fabric_path)

    # 3-4. Fabric metadata (sha256, bbox, hru_count) + id_col check
    buffer_deg = float(config["fabric"]["buffer_deg"])
    fabric_meta = _fabric_metadata(fabric_path, id_col, buffer_deg)

    # 5. Credentials
    _check_credentials(workdir)

    # 6. Datastore — create directory tree
    datastore = Path(config["datastore"])
    dir_mode_str = config.get("dir_mode")
    dir_mode = int(dir_mode_str, 8) if dir_mode_str else None
    _ensure_datastore(datastore, dir_mode)

    # 7. Catalog consistency
    _check_catalog_consistency()

    # Write outputs
    _write_fabric_json(workdir, fabric_meta)
    _write_manifest(workdir, fabric_meta)
    _write_effective_config(workdir, user_cfg)
```

Delete the now-redundant `_check_config` function (its required-key checks are subsumed by `_check_required_keys`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev test tests/test_validate_defaults.py -v`
Expected: PASS for all 6 tests.

- [ ] **Step 5: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: PASS. (Pre-existing `tests/test_init_run.py` and any other `validate_workspace` tests may need a tiny adjustment if they constructed config without the new merge layer — fix in place.)

- [ ] **Step 6: Commit**

```bash
pixi run git add src/nhf_spatial_targets/validate.py tests/test_validate_defaults.py
pixi run git commit -m "$(cat <<'EOF'
feat(validate): report defaults diff, warn unknowns, write effective config

validate now (i) prints to stderr which config keys took defaults,
(ii) warns about user-set keys not in the defaults schema (typo catcher),
(iii) writes <project>/config.effective.yml (mode 0o444) showing the fully
merged config. Required-key check moved into a single helper that is the
authoritative gate before any other validation step runs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `targets/_common.py` — `read_aggregated_source`

**Files:**
- Create: `src/nhf_spatial_targets/targets/_common.py`
- Test: `tests/test_targets_common.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_targets_common.py`:

```python
"""Tests for shared multi-source-minmax target machinery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from nhf_spatial_targets.workspace import load


def _write_year_nc(path: Path, year: int, var: str, id_col: str = "nhm_id"):
    """Write a synthetic per-year aggregated NC at <path>/<source_key>_<year>_agg.nc."""
    times = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    hrus = [1, 2, 3]
    data = np.arange(len(times) * len(hrus), dtype=np.float32).reshape(
        len(times), len(hrus)
    )
    ds = xr.Dataset(
        {var: ((("time", id_col)), data)},
        coords={"time": times, id_col: hrus},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _make_project(tmp_path: Path, source_keys: list[str]) -> Path:
    workdir = tmp_path / "proj"
    workdir.mkdir()
    (workdir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "datastore": str(tmp_path / "store"),
                "fabric": {"path": str(tmp_path / "f.gpkg"), "id_col": "nhm_id"},
            }
        )
    )
    (workdir / "fabric.json").write_text(json.dumps({"id_col": "nhm_id"}))
    return workdir


def test_read_aggregated_source_concats_per_year_nc(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = _make_project(tmp_path, ["era5_land"])
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    _write_year_nc(src_dir / f"{src}_2000_agg.nc", 2000, var)
    _write_year_nc(src_dir / f"{src}_2001_agg.nc", 2001, var)

    project = load(workdir)
    da = read_aggregated_source(
        project, src, var, period=("2000-01-01", "2001-12-31"), chunks={"time": 12}
    )
    assert da.dims == ("time", "nhm_id")
    assert len(da.time) == 24
    assert da.time.values[0] == np.datetime64("2000-01-01")
    assert da.time.values[-1] == np.datetime64("2001-12-01")


def test_read_aggregated_source_slices_to_period(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = _make_project(tmp_path, ["era5_land"])
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    for y in (1999, 2000, 2001, 2002):
        _write_year_nc(src_dir / f"{src}_{y}_agg.nc", y, var)

    project = load(workdir)
    da = read_aggregated_source(
        project, src, var, period=("2000-06-01", "2001-06-30"), chunks={"time": 12}
    )
    # months 2000-06 .. 2001-06 inclusive -> 13 months
    assert len(da.time) == 13


def test_read_aggregated_source_raises_when_dir_empty(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = _make_project(tmp_path, ["era5_land"])
    project = load(workdir)
    with pytest.raises(FileNotFoundError, match="No aggregated NC files found"):
        read_aggregated_source(
            project, "era5_land", "ro", period=("2000-01-01", "2001-12-31")
        )


def test_read_aggregated_source_raises_when_period_outside_coverage(tmp_path: Path):
    from nhf_spatial_targets.targets._common import read_aggregated_source

    workdir = _make_project(tmp_path, ["era5_land"])
    src = "era5_land"
    var = "ro"
    src_dir = workdir / "data" / "aggregated" / src
    _write_year_nc(src_dir / f"{src}_2000_agg.nc", 2000, var)

    project = load(workdir)
    with pytest.raises(ValueError, match="entirely outside source coverage"):
        read_aggregated_source(
            project, src, var, period=("2010-01-01", "2010-12-31")
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev test tests/test_targets_common.py -v`
Expected: FAIL — `targets._common` does not exist.

- [ ] **Step 3: Create `src/nhf_spatial_targets/targets/_common.py`**

```python
"""Shared multi-source-minmax target machinery.

Used by ``targets/run.py`` (and, in future, ``targets/aet.py``) to:

- read per-year aggregated source NCs as a single lazy xarray DataArray;
- canonicalize cross-source monthly time coordinates onto a master
  month-start index;
- combine sources with NaN-aware reductions and emit a per-cell finite-
  source-count diagnostic;
- compute HRU area and centroids in both equal-area and lat/lon CRSes;
- write CF-1.6 compliant NetCDF output atomically.

Linear unit conversions live in the per-target module (e.g. ``run.py``
keeps ``mm_per_month_to_cfs``); this module is unit-agnostic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


def read_aggregated_source(
    project: Project,
    source_key: str,
    var: str,
    period: tuple[str, str],
    chunks: dict | None = None,
) -> xr.DataArray:
    """Open per-year aggregated NCs for one source and return one variable.

    Reads ``<project.aggregated_dir()>/<source_key>/<source_key>_*_agg.nc``
    via ``xr.open_mfdataset`` (lazy / dask-backed), slices to the requested
    period, and returns the requested variable as a DataArray.

    The HRU dim name in the aggregated NCs matches ``project.id_col`` (e.g.
    ``nhm_id``).

    Parameters
    ----------
    project
        Loaded :class:`~nhf_spatial_targets.workspace.Project`.
    source_key
        Catalog key (e.g. ``"era5_land"``).
    var
        Variable name to extract from the aggregated dataset.
    period
        ``(start_iso, end_iso)`` tuple, both inclusive (e.g.
        ``("2000-01-01", "2010-12-31")``).
    chunks
        Forwarded to ``xr.open_mfdataset``. Defaults to
        ``{"time": 12, project.id_col: -1}`` (one calendar year per chunk,
        all HRUs in one chunk).

    Raises
    ------
    FileNotFoundError
        If the source's aggregated directory contains no per-year NCs.
    ValueError
        If the requested period falls entirely outside the source's
        per-year coverage.
    """
    agg_dir = project.aggregated_dir() / source_key
    pattern = f"{source_key}_*_agg.nc"
    paths = sorted(agg_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No aggregated NC files found for source '{source_key}' under "
            f"{agg_dir} (pattern: {pattern}). Run "
            f"'pixi run nhf-targets agg {source_key.replace('_', '-')} "
            f"--project-dir {project.workdir}' first."
        )

    if chunks is None:
        chunks = {"time": 12, project.id_col: -1}

    ds = xr.open_mfdataset(
        [str(p) for p in paths],
        combine="by_coords",
        chunks=chunks,
        engine="netcdf4",
    )
    sliced = ds[var].sel(time=slice(period[0], period[1]))
    if sliced.sizes.get("time", 0) == 0:
        ds_t0 = str(ds["time"].min().values)[:10]
        ds_t1 = str(ds["time"].max().values)[:10]
        ds.close()
        raise ValueError(
            f"Requested period {period[0]} .. {period[1]} is entirely "
            f"outside source coverage for '{source_key}' "
            f"({ds_t0} .. {ds_t1})."
        )
    logger.info(
        "Loaded %s/%s: %d months from %d per-year NCs",
        source_key,
        var,
        sliced.sizes["time"],
        len(paths),
    )
    return sliced
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev test tests/test_targets_common.py -v`
Expected: PASS for all 4 tests.

- [ ] **Step 5: Format and lint**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
pixi run git add src/nhf_spatial_targets/targets/_common.py tests/test_targets_common.py
pixi run git commit -m "$(cat <<'EOF'
feat(targets/_common): add read_aggregated_source

First helper of the new shared multi-source-minmax target machinery:
opens per-year aggregated NCs as a lazy xarray DataArray and slices to
the requested period. Raises with an actionable command if the source's
aggregated dir is empty; raises if the period falls outside coverage.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `targets/_common.py` — `reindex_to_month_start`

**Files:**
- Modify: `src/nhf_spatial_targets/targets/_common.py`
- Modify: `tests/test_targets_common.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_targets_common.py`:

```python
def _da_with_time(times, hrus=(1, 2, 3), values=None) -> xr.DataArray:
    if values is None:
        values = np.arange(len(times) * len(hrus), dtype=np.float32).reshape(
            len(times), len(hrus)
        )
    return xr.DataArray(
        values,
        dims=("time", "nhm_id"),
        coords={"time": pd.DatetimeIndex(times), "nhm_id": list(hrus)},
    )


def test_reindex_to_month_start_maps_eom_to_ms():
    from nhf_spatial_targets.targets._common import reindex_to_month_start

    eom = _da_with_time(["2000-01-31", "2000-02-29", "2000-03-31"])
    master = pd.date_range("2000-01-01", "2000-03-01", freq="MS")
    reindexed = reindex_to_month_start(eom, master)
    assert list(reindexed.time.values) == list(master.values)
    np.testing.assert_array_equal(reindexed.values, eom.values)


def test_reindex_to_month_start_maps_mid_month_to_ms():
    from nhf_spatial_targets.targets._common import reindex_to_month_start

    mid = _da_with_time(["2000-01-15", "2000-02-15", "2000-03-15"])
    master = pd.date_range("2000-01-01", "2000-03-01", freq="MS")
    reindexed = reindex_to_month_start(mid, master)
    np.testing.assert_array_equal(reindexed.values, mid.values)


def test_reindex_to_month_start_pads_missing_months_with_nan():
    """Months in master_index but absent from the source come out as NaN."""
    from nhf_spatial_targets.targets._common import reindex_to_month_start

    partial = _da_with_time(["2000-01-01", "2000-02-01"])
    master = pd.date_range("2000-01-01", "2000-04-01", freq="MS")
    reindexed = reindex_to_month_start(partial, master)
    assert len(reindexed.time) == 4
    assert np.isnan(reindexed.values[2:]).all()
    np.testing.assert_array_equal(reindexed.values[:2], partial.values)


def test_reindex_to_month_start_already_ms_is_idempotent():
    from nhf_spatial_targets.targets._common import reindex_to_month_start

    ms = _da_with_time(["2000-01-01", "2000-02-01", "2000-03-01"])
    master = pd.date_range("2000-01-01", "2000-03-01", freq="MS")
    reindexed = reindex_to_month_start(ms, master)
    np.testing.assert_array_equal(reindexed.values, ms.values)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pixi run -e dev test tests/test_targets_common.py::test_reindex_to_month_start_maps_eom_to_ms -v`
Expected: FAIL — `reindex_to_month_start` not defined.

- [ ] **Step 3: Add the function to `_common.py`**

Append to `src/nhf_spatial_targets/targets/_common.py`:

```python
import pandas as pd


def reindex_to_month_start(
    da: xr.DataArray, master_index: pd.DatetimeIndex
) -> xr.DataArray:
    """Reindex a monthly DataArray onto a master ``freq="MS"`` index.

    Source timestamps may be end-of-month (ERA5-Land), start-of-month (GLDAS,
    MWBM), or mid-month (MERRA-2 etc.). All three convey "which calendar
    month" unambiguously. This helper converts the source's time coordinate
    via ``dt.to_period("M").dt.to_timestamp()`` (yielding the month-start),
    then reindexes onto ``master_index``.

    Months in ``master_index`` that the source does not cover come back as
    NaN — this is what gives the runoff target its period-union semantics:
    a source that ends in 2020 but is asked through 2024 simply contributes
    nothing for the post-2020 cells.

    Parameters
    ----------
    da
        Monthly DataArray to reindex.
    master_index
        Target index. Must be ``DatetimeIndex`` with ``freq="MS"``.
    """
    canon = da.assign_coords(time=da.time.dt.to_period("M").dt.to_timestamp())
    return canon.reindex(time=master_index)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev test tests/test_targets_common.py -v`
Expected: PASS for all reindex tests.

- [ ] **Step 5: Commit**

```bash
pixi run git add src/nhf_spatial_targets/targets/_common.py tests/test_targets_common.py
pixi run git commit -m "$(cat <<'EOF'
feat(targets/_common): add reindex_to_month_start

Maps any monthly source's time coord (EOM, SOM, mid-month) onto a master
freq=MS index via dt.to_period("M").dt.to_timestamp(). Months absent
from the source come back as NaN, which is what makes the period-union
semantics work cleanly downstream.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `targets/_common.py` — `multi_source_nanminmax`

**Files:**
- Modify: `src/nhf_spatial_targets/targets/_common.py`
- Modify: `tests/test_targets_common.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_targets_common.py`:

```python
def test_multi_source_nanminmax_three_finite_sources():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(["2000-01-01"], values=np.array([[10, 20, 30]], dtype=np.float32))
    b = _da_with_time(["2000-01-01"], values=np.array([[15, 25, 35]], dtype=np.float32))
    c = _da_with_time(["2000-01-01"], values=np.array([[20, 30, 40]], dtype=np.float32))
    lower, upper, n = multi_source_nanminmax({"a": a, "b": b, "c": c})
    np.testing.assert_array_equal(lower.values, [[10, 20, 30]])
    np.testing.assert_array_equal(upper.values, [[20, 30, 40]])
    np.testing.assert_array_equal(n.values, [[3, 3, 3]])


def test_multi_source_nanminmax_partial_nan_uses_finite_only():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(
        ["2000-01-01"], values=np.array([[10.0, np.nan, 30.0]], dtype=np.float32)
    )
    b = _da_with_time(
        ["2000-01-01"], values=np.array([[15.0, 25.0, np.nan]], dtype=np.float32)
    )
    lower, upper, n = multi_source_nanminmax({"a": a, "b": b})
    np.testing.assert_array_equal(lower.values, [[10, 25, 30]])
    np.testing.assert_array_equal(upper.values, [[15, 25, 30]])
    np.testing.assert_array_equal(n.values, [[2, 1, 1]])


def test_multi_source_nanminmax_all_nan_returns_nan_and_zero():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(
        ["2000-01-01"], values=np.array([[np.nan]], dtype=np.float32)
    )
    b = _da_with_time(
        ["2000-01-01"], values=np.array([[np.nan]], dtype=np.float32)
    )
    a = a.isel(nhm_id=[0])
    b = b.isel(nhm_id=[0])
    lower, upper, n = multi_source_nanminmax({"a": a, "b": b})
    assert np.isnan(lower.values[0, 0])
    assert np.isnan(upper.values[0, 0])
    assert n.values[0, 0] == 0


def test_multi_source_nanminmax_n_sources_is_int8():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(["2000-01-01"], values=np.array([[10.0, 20.0, 30.0]], np.float32))
    _, _, n = multi_source_nanminmax({"a": a})
    assert n.dtype == np.int8


def test_multi_source_nanminmax_raises_on_hru_mismatch():
    from nhf_spatial_targets.targets._common import multi_source_nanminmax

    a = _da_with_time(["2000-01-01"], hrus=(1, 2, 3))
    b = _da_with_time(["2000-01-01"], hrus=(1, 2, 4))
    with pytest.raises(ValueError, match="HRU coords differ"):
        multi_source_nanminmax({"a": a, "b": b})
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pixi run -e dev test tests/test_targets_common.py::test_multi_source_nanminmax_three_finite_sources -v`
Expected: FAIL — `multi_source_nanminmax` not defined.

- [ ] **Step 3: Add the function to `_common.py`**

Append to `src/nhf_spatial_targets/targets/_common.py`:

```python
import numpy as np


def multi_source_nanminmax(
    sources: dict[str, xr.DataArray],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """NaN-aware per-cell min, max, and finite-source count.

    All input DataArrays must share dims and coords (typically
    ``(time, id_col)``). They are stacked on a new ``source`` dim and
    reduced with ``skipna=True``.

    A bound is defined whenever ≥1 source is finite at that cell; the result
    is NaN only when *every* source is NaN there, which is exactly when
    ``n_sources == 0``.

    Parameters
    ----------
    sources
        Mapping from source key to per-source DataArray.

    Returns
    -------
    lower, upper, n_sources
        ``(time, id_col)`` arrays. ``n_sources`` is int8 with values in
        ``[0, len(sources)]``; ``lower`` / ``upper`` preserve the input
        dtype (typically float32).

    Raises
    ------
    ValueError
        If any two sources have different HRU coords (different fabrics).
    """
    keys = list(sources.keys())
    if not keys:
        raise ValueError("multi_source_nanminmax: empty sources dict")
    ref = sources[keys[0]]
    hru_dim = next(d for d in ref.dims if d != "time")
    for k in keys[1:]:
        other = sources[k]
        if not other[hru_dim].equals(ref[hru_dim]):
            raise ValueError(
                f"HRU coords differ between sources '{keys[0]}' and '{k}'. "
                "All sources must be aggregated to the same fabric."
            )

    stacked = xr.concat(
        [sources[k] for k in keys], dim=xr.Variable("source", keys)
    )
    lower = stacked.min(dim="source", skipna=True)
    upper = stacked.max(dim="source", skipna=True)
    n_sources = stacked.notnull().sum(dim="source").astype(np.int8)
    return lower, upper, n_sources
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev test tests/test_targets_common.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
pixi run git add src/nhf_spatial_targets/targets/_common.py tests/test_targets_common.py
pixi run git commit -m "$(cat <<'EOF'
feat(targets/_common): add multi_source_nanminmax with n_sources diagnostic

NaN-aware min/max across a dict of source DataArrays plus an int8
finite-source count per cell. A bound is defined whenever >=1 source is
finite; NaN only when every source is NaN. Raises on HRU coord mismatch
between sources (different fabrics) — a real bug, not data we silently
drop.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `targets/_common.py` — `compute_hru_area_and_centroids`

**Files:**
- Modify: `src/nhf_spatial_targets/targets/_common.py`
- Modify: `tests/test_targets_common.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_targets_common.py`:

```python
def _write_synthetic_fabric(path: Path, id_col: str = "nhm_id"):
    """Write a 3-polygon GeoPackage in EPSG:4326."""
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {id_col: [1, 2, 3]},
        geometry=[
            box(-105.0, 40.0, -104.9, 40.1),  # ~10x10 km in mid-CONUS
            box(-104.9, 40.0, -104.8, 40.1),
            box(-104.8, 40.0, -104.7, 40.1),
        ],
        crs="EPSG:4326",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")


def _make_project_with_fabric(tmp_path: Path) -> Path:
    workdir = tmp_path / "proj"
    workdir.mkdir()
    fabric_path = tmp_path / "fabric.gpkg"
    _write_synthetic_fabric(fabric_path)
    (workdir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "datastore": str(tmp_path / "store"),
                "fabric": {"path": str(fabric_path), "id_col": "nhm_id"},
            }
        )
    )
    (workdir / "fabric.json").write_text(json.dumps({"id_col": "nhm_id"}))
    return workdir


def test_compute_hru_area_and_centroids_returns_expected_columns(tmp_path: Path):
    from nhf_spatial_targets.targets._common import compute_hru_area_and_centroids

    workdir = _make_project_with_fabric(tmp_path)
    project = load(workdir)
    df = compute_hru_area_and_centroids(project)
    assert df.index.name == "nhm_id"
    assert set(df.columns) == {
        "area_m2",
        "centroid_x",
        "centroid_y",
        "centroid_lat",
        "centroid_lon",
    }
    assert len(df) == 3


def test_compute_hru_area_and_centroids_areas_within_1pct(tmp_path: Path):
    """Each polygon is ~0.1 deg square at lat 40 N ≈ ~85 km² (varies by lat)."""
    from nhf_spatial_targets.targets._common import compute_hru_area_and_centroids

    workdir = _make_project_with_fabric(tmp_path)
    project = load(workdir)
    df = compute_hru_area_and_centroids(project)
    # All three polygons are the same size at this latitude.
    areas = df["area_m2"].values
    assert (areas > 50e6).all() and (areas < 150e6).all()
    # Adjacent polygons should agree to within 1% in area.
    assert abs(areas[0] - areas[1]) / areas[0] < 0.01


def test_compute_hru_area_and_centroids_lat_lon_in_range(tmp_path: Path):
    from nhf_spatial_targets.targets._common import compute_hru_area_and_centroids

    workdir = _make_project_with_fabric(tmp_path)
    project = load(workdir)
    df = compute_hru_area_and_centroids(project)
    assert df["centroid_lon"].between(-180, 180).all()
    assert df["centroid_lat"].between(-90, 90).all()
    # Should be near 40N, -105E given the polygons:
    assert df["centroid_lat"].between(39, 41).all()
    assert df["centroid_lon"].between(-106, -104).all()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pixi run -e dev test tests/test_targets_common.py::test_compute_hru_area_and_centroids_returns_expected_columns -v`
Expected: FAIL — function not defined.

- [ ] **Step 3: Add the function to `_common.py`**

Append to `src/nhf_spatial_targets/targets/_common.py`:

```python
def compute_hru_area_and_centroids(project: Project) -> "pd.DataFrame":
    """Compute per-HRU area (m²) and centroid coords from the fabric.

    Always recomputes from geometry (no fabric-column fallback) so the area
    cannot drift from the geometry actually being processed. Reprojects to
    ``project.area_crs`` (e.g. EPSG:5070 for CONUS) to compute area and
    equal-area centroids; reprojects centroids to EPSG:4326 for ancillary
    lat/lon coords.

    The returned DataFrame is indexed by ``project.id_col`` so callers can
    align to xarray's HRU dim trivially.

    Returns
    -------
    pandas.DataFrame
        Columns: ``area_m2``, ``centroid_x``, ``centroid_y`` (in
        ``area_crs``), ``centroid_lat``, ``centroid_lon`` (EPSG:4326).
    """
    import geopandas as gpd

    fabric_path = Path(project.config["fabric"]["path"])
    if fabric_path.suffix.lower() in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(fabric_path)
    else:
        gdf = gpd.read_file(fabric_path)
    id_col = project.id_col
    if id_col not in gdf.columns:
        raise ValueError(
            f"Column '{id_col}' not found in fabric {fabric_path}. "
            f"Available: {list(gdf.columns)}"
        )

    gdf_eq = gdf.to_crs(project.area_crs)
    centroids_eq = gdf_eq.geometry.centroid
    centroids_ll = centroids_eq.to_crs("EPSG:4326")

    df = gdf_eq[[id_col]].copy()
    df["area_m2"] = gdf_eq.geometry.area.astype(float)
    df["centroid_x"] = centroids_eq.x.astype(float)
    df["centroid_y"] = centroids_eq.y.astype(float)
    df["centroid_lon"] = centroids_ll.x.astype(float)
    df["centroid_lat"] = centroids_ll.y.astype(float)
    df = df.set_index(id_col)
    return df
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev test tests/test_targets_common.py -v`
Expected: PASS for all centroid tests.

- [ ] **Step 5: Commit**

```bash
pixi run git add src/nhf_spatial_targets/targets/_common.py tests/test_targets_common.py
pixi run git commit -m "$(cat <<'EOF'
feat(targets/_common): add compute_hru_area_and_centroids

Reads fabric, reprojects to project.area_crs (EPSG:5070 for CONUS by
default), computes area_m2 plus equal-area and lat/lon centroids.
Always recomputes from geometry — no fabric-column fallback — so the
value cannot drift from what is being processed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `targets/_common.py` — `write_target_nc`

**Files:**
- Modify: `src/nhf_spatial_targets/targets/_common.py`
- Modify: `tests/test_targets_common.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_targets_common.py`:

```python
def _toy_target_dataset(
    id_col: str = "nhm_id",
) -> xr.Dataset:
    """Toy 3-month / 3-HRU dataset with the bound vars and an n_sources."""
    times = pd.date_range("2000-01-01", "2000-03-01", freq="MS")
    bnds = np.stack(
        [times.values, (times + pd.offsets.MonthBegin(1)).values], axis=1
    )
    hrus = np.array([1, 2, 3])
    lower = np.array(
        [[1.0, 2.0, np.nan], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    upper = lower + 1.0
    n = np.array([[2, 2, 0], [2, 2, 2], [2, 2, 2]], dtype=np.int8)
    ds = xr.Dataset(
        {
            "lower_bound": (("time", id_col), lower),
            "upper_bound": (("time", id_col), upper),
            "n_sources": (("time", id_col), n),
        },
        coords={
            "time": times,
            id_col: hrus,
            "time_bnds": (("time", "nv"), bnds),
            "centroid_lat": ((id_col,), np.array([40.0, 40.1, 40.2])),
            "centroid_lon": ((id_col,), np.array([-105.0, -104.9, -104.8])),
        },
    )
    ds["time"].attrs["bounds"] = "time_bnds"
    return ds


def test_write_target_nc_round_trips_via_xarray(tmp_path: Path):
    from nhf_spatial_targets.targets._common import write_target_nc

    ds = _toy_target_dataset()
    out = tmp_path / "runoff_targets.nc"
    write_target_nc(ds, out, title="Test runoff target")
    assert out.exists()
    with xr.open_dataset(out, decode_cf=True) as got:
        assert got.attrs["Conventions"] == "CF-1.6"
        assert got.attrs["title"] == "Test runoff target"
        assert "lower_bound" in got.data_vars
        assert "upper_bound" in got.data_vars
        assert "n_sources" in got.data_vars
        assert got["lower_bound"].dtype == np.float32
        assert got["n_sources"].dtype == np.int8
        assert got["time"].attrs["bounds"] == "time_bnds"
        assert "time_bnds" in got.variables
        assert got["lower_bound"].attrs["units"] == "cfs"
        # NaN preserved (decode_cf maps _FillValue back to NaN):
        assert np.isnan(got["lower_bound"].values[0, 2])


def test_write_target_nc_atomic_no_partial_on_failure(tmp_path: Path, monkeypatch):
    """If to_netcdf raises, the final path must not exist (tempfile cleanup)."""
    from nhf_spatial_targets.targets._common import write_target_nc

    out = tmp_path / "runoff_targets.nc"

    def _boom(self, *a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _boom)
    with pytest.raises(RuntimeError, match="disk full"):
        write_target_nc(_toy_target_dataset(), out, title="x")
    assert not out.exists()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pixi run -e dev test tests/test_targets_common.py::test_write_target_nc_round_trips_via_xarray -v`
Expected: FAIL — `write_target_nc` not defined.

- [ ] **Step 3: Add the function to `_common.py`**

Append to `src/nhf_spatial_targets/targets/_common.py`:

```python
def write_target_nc(
    ds: xr.Dataset,
    output_path: Path,
    title: str,
    extra_global_attrs: dict | None = None,
) -> None:
    """Write a target Dataset to NetCDF atomically with CF-1.6 metadata.

    The Dataset is expected to already carry the data variables, ancillary
    coordinates (``time_bnds``, ``centroid_lat``, ``centroid_lon``), and
    per-variable attrs (``units``, ``long_name``, ``cell_methods``, etc.).
    This helper sets the global ``Conventions`` / ``title`` / ``history`` /
    ``software_version`` attrs, applies float32+zlib encoding for the bound
    variables and int8+zlib encoding for the diagnostic variables, and
    writes via tempfile + rename so a partial NetCDF never lands at the
    final path.
    """
    from datetime import datetime, timezone

    from nhf_spatial_targets import __version__

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = ds.copy()
    ds.attrs.setdefault("Conventions", "CF-1.6")
    ds.attrs["title"] = title
    ds.attrs["history"] = (
        f"{datetime.now(timezone.utc).isoformat()} created by "
        f"nhf_spatial_targets v{__version__}"
    )
    ds.attrs.setdefault("institution", "USGS")
    ds.attrs.setdefault("software_version", __version__)
    if extra_global_attrs:
        ds.attrs.update(extra_global_attrs)

    encoding: dict = {}
    for v in ("lower_bound", "upper_bound"):
        if v in ds.data_vars:
            encoding[v] = {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.float32("nan"),
            }
    for v in ("n_sources", "nn_filled"):
        if v in ds.data_vars:
            encoding[v] = {
                "dtype": "int8",
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.int8(-1),
            }
    encoding["time"] = {
        "dtype": "float64",
        "units": "days since 1970-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
    }
    if "time_bnds" in ds.variables:
        encoding["time_bnds"] = {
            "dtype": "float64",
            "units": "days since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        }

    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        ds.to_netcdf(tmp, format="NETCDF4", encoding=encoding)
        tmp.rename(output_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    logger.info("Wrote %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev test tests/test_targets_common.py -v`
Expected: PASS.

- [ ] **Step 5: Format and lint**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
pixi run git add src/nhf_spatial_targets/targets/_common.py tests/test_targets_common.py
pixi run git commit -m "$(cat <<'EOF'
feat(targets/_common): add CF-1.6 atomic write_target_nc

Sets Conventions / title / history / software_version, applies
float32+zlib encoding for bound vars and int8+zlib for diagnostics
(n_sources, nn_filled), writes time / time_bnds with explicit CF
units, and writes via tempfile+rename so a partial NetCDF never lands
at the final path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `normalize/methods.py` — `nn_fill_bounds`

**Files:**
- Modify: `src/nhf_spatial_targets/normalize/methods.py`
- Create: `tests/test_normalize_nn_fill.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_normalize_nn_fill.py`:

```python
"""Tests for HRU-space NN-fill of multi-source-minmax bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _bounds_dataset(values_lower, values_upper, hrus, times):
    return xr.Dataset(
        {
            "lower_bound": (("time", "nhm_id"), np.asarray(values_lower, np.float32)),
            "upper_bound": (("time", "nhm_id"), np.asarray(values_upper, np.float32)),
        },
        coords={"time": pd.DatetimeIndex(times), "nhm_id": list(hrus)},
    )


def test_nn_fill_bounds_fills_isolated_nan_with_nearest_finite():
    from nhf_spatial_targets.normalize.methods import nn_fill_bounds

    # 3 HRUs in a row at x = 0, 1, 5; HRU 2 (at x=1) is NaN. Nearest finite
    # neighbor is HRU 1 (at x=0, distance 1) over HRU 3 (at x=5, distance 4).
    centroids_xy = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    ds = _bounds_dataset(
        values_lower=[[10.0, np.nan, 50.0]],
        values_upper=[[20.0, np.nan, 60.0]],
        hrus=[1, 2, 3],
        times=["2000-01-01"],
    )
    filled, diag = nn_fill_bounds(ds, centroids_xy, max_candidates=10)
    assert filled["lower_bound"].values[0, 1] == 10.0
    assert filled["upper_bound"].values[0, 1] == 20.0
    assert diag.values[0, 1] == 1
    assert diag.values[0, 0] == 0  # not filled


def test_nn_fill_bounds_walks_when_nearest_is_also_nan():
    from nhf_spatial_targets.normalize.methods import nn_fill_bounds

    # HRU 2 is NaN. Nearest is HRU 1 at distance 1, but HRU 1 is also NaN.
    # Walk to HRU 3 at distance 4.
    centroids_xy = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    ds = _bounds_dataset(
        values_lower=[[np.nan, np.nan, 50.0]],
        values_upper=[[np.nan, np.nan, 60.0]],
        hrus=[1, 2, 3],
        times=["2000-01-01"],
    )
    filled, diag = nn_fill_bounds(ds, centroids_xy, max_candidates=10)
    assert filled["lower_bound"].values[0, 1] == 50.0
    assert filled["upper_bound"].values[0, 1] == 60.0
    assert diag.values[0, 1] == 1


def test_nn_fill_bounds_max_candidates_cap_leaves_nan():
    from nhf_spatial_targets.normalize.methods import nn_fill_bounds

    # All NaN at this time step -> no donor will ever be finite; cell stays NaN.
    centroids_xy = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    ds = _bounds_dataset(
        values_lower=[[np.nan, np.nan, np.nan]],
        values_upper=[[np.nan, np.nan, np.nan]],
        hrus=[1, 2, 3],
        times=["2000-01-01"],
    )
    filled, diag = nn_fill_bounds(ds, centroids_xy, max_candidates=2)
    assert np.isnan(filled["lower_bound"].values).all()
    assert (diag.values == 0).all()


def test_nn_fill_bounds_per_time_step_independence():
    from nhf_spatial_targets.normalize.methods import nn_fill_bounds

    # HRU 2 is NaN at t=0 (filled from HRU 1 = 10.0) but finite at t=1 (untouched).
    centroids_xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    ds = _bounds_dataset(
        values_lower=[[10.0, np.nan], [99.0, 88.0]],
        values_upper=[[20.0, np.nan], [199.0, 188.0]],
        hrus=[1, 2],
        times=["2000-01-01", "2000-02-01"],
    )
    filled, diag = nn_fill_bounds(ds, centroids_xy, max_candidates=10)
    assert filled["lower_bound"].values[0, 1] == 10.0
    assert filled["lower_bound"].values[1, 1] == 88.0
    assert diag.values[0, 1] == 1
    assert diag.values[1, 1] == 0


def test_nn_fill_bounds_preserves_finite_cells():
    from nhf_spatial_targets.normalize.methods import nn_fill_bounds

    centroids_xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    ds = _bounds_dataset(
        values_lower=[[10.0, 20.0]],
        values_upper=[[100.0, 200.0]],
        hrus=[1, 2],
        times=["2000-01-01"],
    )
    filled, diag = nn_fill_bounds(ds, centroids_xy, max_candidates=10)
    np.testing.assert_array_equal(
        filled["lower_bound"].values, ds["lower_bound"].values
    )
    np.testing.assert_array_equal(
        filled["upper_bound"].values, ds["upper_bound"].values
    )
    assert (diag.values == 0).all()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pixi run -e dev test tests/test_normalize_nn_fill.py -v`
Expected: FAIL — `nn_fill_bounds` not defined.

- [ ] **Step 3: Add `nn_fill_bounds` to `normalize/methods.py`**

Replace the body of `src/nhf_spatial_targets/normalize/methods.py` with:

```python
"""Normalization and range-bound construction methods."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def normalize_0_1(da, dim: str = "time") -> object:
    """Normalize an xarray DataArray to [0, 1] over the given dimension."""
    raise NotImplementedError


def normalize_by_calendar_month(da) -> object:
    """Normalize per calendar month: each month normalized independently."""
    raise NotImplementedError


def multi_source_minmax(datasets: list) -> tuple:
    """Compute lower/upper bounds as min/max across a list of DataArrays."""
    raise NotImplementedError


def modis_ci_bounds(sca, ci, ci_threshold: float = 0.70) -> tuple:
    """Construct SCA calibration bounds from MODIS confidence interval."""
    raise NotImplementedError


def nn_fill_bounds(
    ds: xr.Dataset,
    centroids_xy: np.ndarray,
    max_candidates: int = 10,
) -> Tuple[xr.Dataset, xr.DataArray]:
    """Fill NaN bound cells with the nearest *finite* HRU at the same time step.

    For every HRU position that is NaN in either ``lower_bound`` or
    ``upper_bound`` at any time step, this walks ``cKDTree`` neighbors in
    increasing-distance order and adopts the bound values of the first
    donor that is finite *at that time step*. If no donor among the first
    ``max_candidates`` neighbors is finite, the cell stays NaN.

    Cells where both bounds are already finite are untouched.

    Parameters
    ----------
    ds
        Dataset with ``lower_bound(time, id_col)`` and
        ``upper_bound(time, id_col)`` float vars.
    centroids_xy
        Array of shape ``(n_hrus, 2)`` with HRU centroids in an equal-area
        CRS (matching ``ds[id_col]`` order).
    max_candidates
        Maximum number of donor neighbors to consider per (time, hru)
        before giving up.

    Returns
    -------
    filled_ds, nn_filled
        ``filled_ds`` is a copy of ``ds`` with ``lower_bound`` /
        ``upper_bound`` updated; ``nn_filled`` is an int8
        ``(time, id_col)`` flag array (0 = not filled, 1 = filled).
    """
    from scipy.spatial import cKDTree

    if "lower_bound" not in ds or "upper_bound" not in ds:
        raise ValueError(
            "nn_fill_bounds requires 'lower_bound' and 'upper_bound' in ds"
        )
    id_col = next(d for d in ds["lower_bound"].dims if d != "time")
    if centroids_xy.shape != (ds.sizes[id_col], 2):
        raise ValueError(
            f"centroids_xy shape {centroids_xy.shape} does not match "
            f"({ds.sizes[id_col]}, 2)"
        )

    lower = ds["lower_bound"].values.copy()
    upper = ds["upper_bound"].values.copy()
    n_time, n_hru = lower.shape

    # Use up to (1 + max_candidates) neighbors so that index 0 (the cell itself)
    # can be skipped without losing donor budget.
    k = min(1 + max_candidates, n_hru)
    tree = cKDTree(centroids_xy)
    _, neighbor_idx = tree.query(centroids_xy, k=k)
    if k == 1:
        neighbor_idx = neighbor_idx[:, None]

    diag = np.zeros((n_time, n_hru), dtype=np.int8)
    nan_mask = np.isnan(lower) | np.isnan(upper)

    if not nan_mask.any():
        nn_diag = xr.DataArray(
            diag,
            dims=ds["lower_bound"].dims,
            coords={d: ds[d] for d in ds["lower_bound"].dims},
            name="nn_filled",
        )
        return ds.copy(), nn_diag

    # Iterate only HRUs that ever go NaN.
    nan_hrus = np.where(nan_mask.any(axis=0))[0]
    n_unfilled = 0
    for h in nan_hrus:
        candidates = neighbor_idx[h]
        # Skip self (index 0 in the kNN result).
        candidates = candidates[candidates != h]
        for t in np.where(nan_mask[:, h])[0]:
            for cand in candidates[:max_candidates]:
                lo = lower[t, cand]
                up = upper[t, cand]
                if np.isfinite(lo) and np.isfinite(up):
                    lower[t, h] = lo
                    upper[t, h] = up
                    diag[t, h] = 1
                    break
            else:
                n_unfilled += 1
    if n_unfilled:
        logger.warning(
            "nn_fill_bounds: %d (time, hru) cells stayed NaN after exhausting "
            "%d donor candidates.",
            n_unfilled,
            max_candidates,
        )

    out = ds.copy()
    out["lower_bound"] = (ds["lower_bound"].dims, lower)
    out["upper_bound"] = (ds["upper_bound"].dims, upper)
    out["lower_bound"].attrs = dict(ds["lower_bound"].attrs)
    out["upper_bound"].attrs = dict(ds["upper_bound"].attrs)
    nn_diag = xr.DataArray(
        diag,
        dims=ds["lower_bound"].dims,
        coords={d: ds[d] for d in ds["lower_bound"].dims},
        name="nn_filled",
    )
    return out, nn_diag
```

- [ ] **Step 4: Add scipy to `pixi.toml` dependencies if not already present**

Check `pixi.toml` for `scipy`. If absent, run:

```
pixi add scipy
```

This updates `pixi.toml` and `pixi.lock` together.

- [ ] **Step 5: Run tests to verify pass**

Run: `pixi run -e dev test tests/test_normalize_nn_fill.py -v`
Expected: PASS for all 5 tests.

- [ ] **Step 6: Format and lint**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
pixi run git add src/nhf_spatial_targets/normalize/methods.py tests/test_normalize_nn_fill.py pixi.toml pixi.lock
pixi run git commit -m "$(cat <<'EOF'
feat(normalize): add nn_fill_bounds for HRU-space NN-fill of bounds

cKDTree-based donor walk in equal-area CRS. Fills NaN bound cells with
the nearest finite HRU's values at the same time step; walks to the
next-nearest donor if needed; gives up after max_candidates and leaves
NaN. Returns the filled Dataset plus an int8 nn_filled diagnostic
array. Cells already finite are untouched.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Rewrite `targets/run.py` to use `_common` + all three sources

**Files:**
- Modify: `src/nhf_spatial_targets/targets/run.py`
- Create: `tests/test_targets_run.py` (the existing test file does not exist; rewriting `run.py` is purely additive on the test side)

- [ ] **Step 1: Write the failing test**

Create `tests/test_targets_run.py`:

```python
"""Tests for the runoff target builder end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml


def _write_year_nc(
    path: Path, year: int, var: str, value: float, id_col: str = "nhm_id"
):
    """Write a per-year aggregated NC with the given constant value."""
    times = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    hrus = [1, 2, 3]
    arr = np.full((len(times), len(hrus)), value, dtype=np.float32)
    ds = xr.Dataset(
        {var: (("time", id_col), arr)},
        coords={"time": times, id_col: hrus},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _write_synthetic_fabric(path: Path, id_col: str = "nhm_id"):
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {id_col: [1, 2, 3]},
        geometry=[
            box(-105.0, 40.0, -104.9, 40.1),
            box(-104.9, 40.0, -104.8, 40.1),
            box(-104.8, 40.0, -104.7, 40.1),
        ],
        crs="EPSG:4326",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")


def _make_runoff_project(
    tmp_path: Path,
    period: str = "2000-01-01/2001-12-31",
    nn_fill: bool = True,
    sources_per_year: dict | None = None,
) -> Path:
    """Build a project skeleton with synthetic fabric + per-year aggregated NCs.

    sources_per_year: dict[source_key -> dict[year -> (var, value)]]
    Default: era5_land ro=0.05 m/month for 2000-2001;
             gldas_noah_v21_monthly runoff_total=2.0 kg/m2 for 2000-2001;
             mwbm_climgrid runoff=30.0 mm/month for 2000 only (partial period).
    """
    workdir = tmp_path / "proj"
    workdir.mkdir()
    fabric_path = tmp_path / "fabric.gpkg"
    _write_synthetic_fabric(fabric_path)
    (workdir / "fabric.json").write_text(json.dumps({"id_col": "nhm_id"}))

    cfg = {
        "datastore": str(tmp_path / "store"),
        "fabric": {"path": str(fabric_path), "id_col": "nhm_id"},
        "targets": {
            "runoff": {
                "period": period,
                "nn_fill": nn_fill,
            }
        },
    }
    (workdir / "config.yml").write_text(yaml.safe_dump(cfg))

    plan = sources_per_year or {
        "era5_land": {2000: ("ro", 0.05), 2001: ("ro", 0.05)},
        "gldas_noah_v21_monthly": {
            2000: ("runoff_total", 2.0),
            2001: ("runoff_total", 2.0),
        },
        "mwbm_climgrid": {2000: ("runoff", 30.0)},
    }
    agg_dir = workdir / "data" / "aggregated"
    for src, year_map in plan.items():
        for year, (var, value) in year_map.items():
            _write_year_nc(agg_dir / src / f"{src}_{year}_agg.nc", year, var, value)
    return workdir


def test_build_writes_unfilled_and_filled_files(tmp_path: Path):
    from nhf_spatial_targets.targets.run import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_runoff_project(tmp_path)
    project = load(workdir)
    build(project)
    assert (project.targets_dir() / "runoff_targets.nc").exists()
    assert (project.targets_dir() / "runoff_targets_nn_filled.nc").exists()


def test_build_period_union_n_sources_diagnostic(tmp_path: Path):
    """MWBM only covers 2000 -> 2001 cells should have n_sources=2."""
    from nhf_spatial_targets.targets.run import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_runoff_project(tmp_path, nn_fill=False)
    project = load(workdir)
    build(project)
    out = project.targets_dir() / "runoff_targets.nc"
    with xr.open_dataset(out, decode_cf=True) as ds:
        n = ds["n_sources"].values
        # 24 months total: months 0..11 = 2000 (3 sources), 12..23 = 2001 (2 sources)
        assert (n[:12] == 3).all()
        assert (n[12:] == 2).all()


def test_build_output_schema(tmp_path: Path):
    from nhf_spatial_targets.targets.run import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_runoff_project(tmp_path)
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "runoff_targets.nc") as ds:
        assert "lower_bound" in ds and "upper_bound" in ds
        assert "n_sources" in ds
        assert "centroid_lat" in ds.coords and "centroid_lon" in ds.coords
        assert ds["lower_bound"].attrs["units"] == "cfs"
        assert ds.attrs["Conventions"] == "CF-1.6"
        assert ds["time"].attrs["bounds"] == "time_bnds"
        assert "time_bnds" in ds.variables


def test_build_unit_chain_matches_hand_calculation(tmp_path: Path):
    """ERA5 0.05 m/mo on a known-area HRU -> known cfs.

    For HRU 1 (the leftmost ~10x10 km cell), the area in EPSG:5070 is roughly
    8.5e7 m². ERA5 0.05 m/mo = 50 mm/month. cfs =
    (50e-3 * 8.5e7) / (days_in_month) * 35.3147 / 86400. We just check that
    the lower bound is within a reasonable order of magnitude (sanity, not
    full unit pinning — synthetic geometries make exact matching brittle).
    """
    from nhf_spatial_targets.targets.run import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_runoff_project(
        tmp_path,
        sources_per_year={
            "era5_land": {2000: ("ro", 0.05)},  # 50 mm/mo
            "gldas_noah_v21_monthly": {2000: ("runoff_total", 6.25)},
            # ^ 6.25 * 8 * 31 = 1550 mm/mo Jan, much bigger than ERA5
            "mwbm_climgrid": {2000: ("runoff", 30.0)},
        },
        period="2000-01-01/2000-12-31",
        nn_fill=False,
    )
    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "runoff_targets.nc") as ds:
        # Lower bound should be the smallest of three positive numbers > 0:
        assert (ds["lower_bound"].values > 0).all()
        assert (ds["upper_bound"].values >= ds["lower_bound"].values).all()


def test_build_hru_mismatch_raises(tmp_path: Path):
    """Sources aggregated against different HRU sets -> raise."""
    from nhf_spatial_targets.targets.run import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_runoff_project(tmp_path, nn_fill=False)
    # Overwrite the era5 NC with mismatched HRUs:
    bad = workdir / "data" / "aggregated" / "era5_land" / "era5_land_2000_agg.nc"
    times = pd.date_range("2000-01-01", "2000-12-01", freq="MS")
    hrus = [1, 2, 99]  # 99 instead of 3
    ds = xr.Dataset(
        {"ro": (("time", "nhm_id"), np.full((12, 3), 0.05, dtype=np.float32))},
        coords={"time": times, "nhm_id": hrus},
    )
    bad.unlink()
    ds.to_netcdf(bad)

    project = load(workdir)
    with pytest.raises(ValueError, match="HRU coords differ"):
        build(project)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pixi run -e dev test tests/test_targets_run.py -v`
Expected: FAIL — current `run.build` has the wrong signature and reads from the obsolete `<source>/<var>.nc` layout.

- [ ] **Step 3: Replace `src/nhf_spatial_targets/targets/run.py`**

Overwrite the file with:

```python
"""Build runoff calibration targets from ERA5-Land + GLDAS-2.1 NOAH + MWBM.

Three monthly sources contribute to per-HRU per-month bounds in cfs:

  - ERA5-Land ``ro``               (m water-equivalent / month)
  - GLDAS-2.1 NOAH ``runoff_total`` (kg/m², mean of 3-hourly accumulations;
                                    multiply by 8 × days_in_month for mm/month)
  - MWBM ClimGrid ``runoff``       (mm/month, native)

Per-source unit shims convert each to mm/month, ``mm_per_month_to_cfs`` then
converts to cfs using the per-HRU equal-area area and days-in-month. Sources
are stacked on a ``source`` dim and reduced with NaN-aware min/max so a bound
is defined whenever ≥1 source is finite at that (HRU, time). An int8
``n_sources`` diagnostic is also written.

If ``runoff.nn_fill`` is True (default), a second file
``<output>_nn_filled.nc`` is written with bound NaNs filled by the nearest
finite HRU's value at the same time step (cKDTree donor walk in
``project.area_crs``).
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd
import xarray as xr

from nhf_spatial_targets.normalize.methods import nn_fill_bounds
from nhf_spatial_targets.targets._common import (
    compute_hru_area_and_centroids,
    multi_source_nanminmax,
    read_aggregated_source,
    reindex_to_month_start,
    write_target_nc,
)
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


# 1 m³/day = (1/86400) m³/s = (35.3147/86400) ft³/s
_M3_PER_DAY_TO_CFS = 35.3146667 / 86400.0

# Per-source variable name in the aggregated NC.
_SOURCE_VAR: dict[str, str] = {
    "era5_land": "ro",
    "gldas_noah_v21_monthly": "runoff_total",
    "mwbm_climgrid": "runoff",
}


# ---------------------------------------------------------------------------
# Per-source unit shims (mm/month is the common intermediate)
# ---------------------------------------------------------------------------


def era5_to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """ERA5-Land runoff (m water-eq / month) → mm/month."""
    out = da * 1000.0
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


def gldas_to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """GLDAS Qs_acc + Qsb_acc → mm/month.

    GLDAS-2.1 ``_acc`` monthly values are the *mean* of 3-hourly
    accumulations (NOT a monthly sum). Per the NASA GES DISC GLDAS-2.1
    README, the monthly total is recovered via × 8 × days_in_month.
    1 kg/m² ≡ 1 mm depth.
    """
    days = da["time"].dt.days_in_month
    out = da * 8.0 * days
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


def mwbm_to_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    """MWBM ClimGrid runoff is already mm/month — pass through."""
    out = da.copy()
    out.attrs = dict(da.attrs)
    out.attrs["units"] = "mm"
    return out


_TO_MM: dict[str, Callable[[xr.DataArray], xr.DataArray]] = {
    "era5_land": era5_to_mm_per_month,
    "gldas_noah_v21_monthly": gldas_to_mm_per_month,
    "mwbm_climgrid": mwbm_to_mm_per_month,
}


def mm_per_month_to_cfs(
    da: xr.DataArray, hru_area_m2: xr.DataArray
) -> xr.DataArray:
    """Convert mm/month → cfs given per-HRU area and the month length."""
    days = da["time"].dt.days_in_month
    m_per_day = (da * 1e-3) / days
    m3_per_day = m_per_day * hru_area_m2
    cfs = m3_per_day * _M3_PER_DAY_TO_CFS
    cfs.attrs = dict(da.attrs)
    cfs.attrs["units"] = "cfs"
    return cfs


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _parse_period(period_str: str) -> tuple[str, str]:
    """Parse 'YYYY-MM-DD/YYYY-MM-DD' (or 'YYYY/YYYY') into (start, end)."""
    if "/" not in period_str:
        raise ValueError(
            f"Invalid period {period_str!r}. Expected 'YYYY-MM-DD/YYYY-MM-DD'."
        )
    start, end = period_str.split("/", 1)
    return start.strip(), end.strip()


def build(project: Project) -> None:
    """Build the runoff calibration target.

    Reads each enabled source's per-year aggregated NCs, harmonizes time
    coords onto a master month-start index over ``runoff.period``,
    converts each to cfs using per-HRU area, combines via NaN-aware
    min/max, and writes a CF-1.6 NetCDF. If ``runoff.nn_fill`` is True,
    additionally writes ``runoff_targets_nn_filled.nc``.
    """
    runoff_cfg = project.target("runoff")
    period = _parse_period(runoff_cfg["period"])
    sources = list(runoff_cfg["sources"])
    chunk_months = int(runoff_cfg["chunk_months"])

    logger.info(
        "Building runoff target: %d sources (%s), period %s .. %s, fabric=%s",
        len(sources),
        ",".join(sources),
        period[0],
        period[1],
        project.config["fabric"]["path"],
    )

    # 1. Per-HRU area + centroids (computed once from fabric).
    hru_meta = compute_hru_area_and_centroids(project)
    id_col = project.id_col
    hru_area_da = xr.DataArray(
        hru_meta["area_m2"].values,
        dims=(id_col,),
        coords={id_col: hru_meta.index.values},
        name="area_m2",
    )

    # 2. Master month-start index over the requested period.
    master_idx = pd.date_range(period[0], period[1], freq="MS")
    if len(master_idx) == 0:
        raise ValueError(
            f"runoff.period {runoff_cfg['period']} produces no months at "
            f"freq='MS'. Check the date range."
        )

    # 3. Read, convert, reindex each source.
    sources_cfs: dict[str, xr.DataArray] = {}
    for src in sources:
        if src not in _SOURCE_VAR:
            raise ValueError(
                f"runoff.sources includes unknown source '{src}'. "
                f"Known: {sorted(_SOURCE_VAR.keys())}"
            )
        var = _SOURCE_VAR[src]
        da_native = read_aggregated_source(
            project, src, var, period, chunks={"time": chunk_months, id_col: -1}
        )
        da_mm = _TO_MM[src](da_native)
        da_cfs = mm_per_month_to_cfs(da_mm, hru_area_da)
        sources_cfs[src] = reindex_to_month_start(da_cfs, master_idx)

    # 4. NaN-aware combination.
    lower, upper, n_sources = multi_source_nanminmax(sources_cfs)
    lower.name = "lower_bound"
    upper.name = "upper_bound"
    n_sources.name = "n_sources"

    # 5. Assemble Dataset with CF metadata + ancillary coords.
    time_bnds = xr.DataArray(
        list(zip(master_idx.values, (master_idx + pd.offsets.MonthBegin(1)).values)),
        dims=("time", "nv"),
        coords={"time": master_idx.values},
        name="time_bnds",
    )
    centroid_lat = xr.DataArray(
        hru_meta["centroid_lat"].values,
        dims=(id_col,),
        coords={id_col: hru_meta.index.values},
        attrs={
            "units": "degrees_north",
            "standard_name": "latitude",
            "long_name": "HRU centroid latitude",
        },
    )
    centroid_lon = xr.DataArray(
        hru_meta["centroid_lon"].values,
        dims=(id_col,),
        coords={id_col: hru_meta.index.values},
        attrs={
            "units": "degrees_east",
            "standard_name": "longitude",
            "long_name": "HRU centroid longitude",
        },
    )

    lower.attrs.update(
        {
            "units": "cfs",
            "long_name": (
                "lower bound of monthly runoff "
                "(NaN-aware min across sources)"
            ),
            "cell_methods": "time: sum",
            "coordinates": "centroid_lat centroid_lon",
        }
    )
    upper.attrs.update(
        {
            "units": "cfs",
            "long_name": (
                "upper bound of monthly runoff "
                "(NaN-aware max across sources)"
            ),
            "cell_methods": "time: sum",
            "coordinates": "centroid_lat centroid_lon",
        }
    )
    n_sources.attrs.update(
        {
            "units": "1",
            "long_name": "number of finite source contributions",
            "flag_values": list(range(0, len(sources) + 1)),
            "flag_meanings": " ".join(
                ["none", "one", "two", "three", "four", "five"][: len(sources) + 1]
            ),
            "coordinates": "centroid_lat centroid_lon",
        }
    )

    ds = xr.Dataset(
        {
            "lower_bound": lower,
            "upper_bound": upper,
            "n_sources": n_sources,
        },
        coords={
            "time": master_idx,
            id_col: lower[id_col],
            "time_bnds": time_bnds,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
        },
    )
    ds["time"].attrs["bounds"] = "time_bnds"
    ds["time"].attrs["axis"] = "T"
    ds["time"].attrs["standard_name"] = "time"
    ds["time"].attrs["long_name"] = "time at month start"
    ds[id_col].attrs["long_name"] = "HRU identifier"
    ds[id_col].attrs["cf_role"] = "timeseries_id"

    extra_attrs = {
        "source": (
            "ERA5-Land ro; GLDAS-2.1 NOAH Qs_acc+Qsb_acc; "
            "MWBM ClimGrid runoff"
        ),
        "references": "Hay et al. 2022, doi:10.3133/tm6B10",
        "fabric": project.config["fabric"]["path"],
        "fabric_sha256": project.fabric.get("sha256", ""),
        "period": runoff_cfg["period"],
        "area_crs": project.area_crs,
    }

    output_path = project.targets_dir() / runoff_cfg["output_file"]
    write_target_nc(
        ds,
        output_path,
        title="NHM runoff calibration target (lower/upper bounds in cfs)",
        extra_global_attrs=extra_attrs,
    )

    # Coverage summary log line.
    n = ds["n_sources"].values
    total = n.size
    none = int((n == 0).sum())
    logger.info(
        "Coverage: %d/%d cells have >=1 finite source (%.2f%% all-NaN)",
        total - none,
        total,
        100.0 * none / total if total else 0.0,
    )

    # 6. NN-fill (optional).
    if runoff_cfg["nn_fill"]:
        # Materialize the bounds for the in-memory NN walk.
        ds_loaded = ds.compute()
        centroids_xy = hru_meta[["centroid_x", "centroid_y"]].values
        filled_ds, nn_diag = nn_fill_bounds(
            ds_loaded,
            centroids_xy,
            max_candidates=int(runoff_cfg["nn_max_candidates"]),
        )
        nn_diag.attrs.update(
            {
                "units": "1",
                "long_name": "nearest-neighbor fill flag",
                "flag_values": [0, 1],
                "flag_meanings": "not_filled filled",
                "coordinates": "centroid_lat centroid_lon",
            }
        )
        filled_ds["nn_filled"] = nn_diag
        filled_attrs = dict(extra_attrs)
        filled_attrs["nn_fill_max_candidates"] = int(runoff_cfg["nn_max_candidates"])
        filled_attrs["nn_fill_distance_crs"] = project.area_crs

        nn_path = output_path.with_name(
            output_path.stem + "_nn_filled" + output_path.suffix
        )
        write_target_nc(
            filled_ds,
            nn_path,
            title="NHM runoff calibration target (NN-filled, cfs)",
            extra_global_attrs=filled_attrs,
        )
```

- [ ] **Step 4: Run the runoff tests**

Run: `pixi run -e dev test tests/test_targets_run.py -v`
Expected: PASS for all 5 tests.

- [ ] **Step 5: Run the full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: PASS. (If pre-existing tests of the old `run.build(target_cfg, output_path)` signature exist, delete them — the entire builder is replaced by this rewrite.)

- [ ] **Step 6: Commit**

```bash
pixi run git add src/nhf_spatial_targets/targets/run.py tests/test_targets_run.py
pixi run git commit -m "$(cat <<'EOF'
feat(targets/run): rewrite runoff builder for 3-source NaN-aware bounds

Replaces the obsolete <source>/<var>.nc reader with the per-year
aggregated NC layout established in PR #51, adds MWBM ClimGrid as the
third source, switches combination to NaN-aware nanmin/nanmax with an
int8 n_sources diagnostic, canonicalizes time coords across EOM/SOM/
mid-month sources onto a master month-start index, writes time_bnds
and centroid lat/lon ancillary coords, and (when runoff.nn_fill is
True) emits a parallel runoff_targets_nn_filled.nc with NaN bound
cells filled by nearest finite HRU at the same time step.

Closes the runoff portion of the spec at
docs/superpowers/specs/2026-05-05-runoff-target-builder-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Refactor `cli.py` `_dispatch` to pass `Project` to `run.build`

**Files:**
- Modify: `src/nhf_spatial_targets/cli.py`
- Modify: `tests/test_cli.py` (if existing tests cover `run`/`_dispatch`)

- [ ] **Step 1: Read existing CLI tests for the `run` command**

```bash
grep -n "run(" /home/rmcd/projects/usgs/nhf-spatial-targets/tests/test_cli.py 2>/dev/null | head
```
Note any tests that need updating; they typically live under names like `test_run_cmd_*`. If there are none, skip the test edit step below.

- [ ] **Step 2: Edit `_dispatch` in `cli.py`**

Replace the `_dispatch` function body to load a `Project` once and pass it to runoff (which uses the new signature) while preserving the older signature for not-yet-rewritten targets:

```python
def _dispatch(
    name: str,
    target_cfg: dict,
    pipeline_cfg: dict,
    workdir: Path | None = None,
) -> None:
    """Dispatch to the appropriate target builder module."""
    from nhf_spatial_targets.targets import aet, rch, run, sca, som
    from nhf_spatial_targets.workspace import load as load_project

    if name == "runoff":
        if workdir is None:
            print(
                "Error: --project-dir is required for runoff target",
                file=sys.stderr,
            )
            sys.exit(1)
        project = load_project(workdir)
        run.build(project)
        return

    builders = {
        "aet": aet.build,
        "recharge": rch.build,
        "soil_moisture": som.build,
        "snow_covered_area": sca.build,
    }
    if name not in builders:
        print(f"Error: No builder registered for target: {name}", file=sys.stderr)
        sys.exit(1)

    fabric_path = pipeline_cfg["fabric"]["path"]
    if workdir is not None:
        output_path = str(workdir / "targets")
    else:
        output_path = pipeline_cfg["output"]["dir"]

    builders[name](target_cfg, fabric_path, output_path)
```

Also update the `run` command itself to ensure it loads via `workspace.load` so defaults flow through. Replace the body of the existing `run` command:

```python
@app.command
def run(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir", "-d"],
            help="Project created by 'nhf-targets init'.",
        ),
    ],
    target: Annotated[
        str | None,
        Parameter(
            name=["--target", "-t"],
            help="Run a single target (default: all enabled).",
        ),
    ] = None,
):
    """Run the calibration target pipeline."""
    from nhf_spatial_targets.workspace import load as load_project

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
        project = load_project(workdir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    targets_cfg = project.config.get("targets", {})
    to_run = (
        [target]
        if target
        else [k for k, v in targets_cfg.items() if v.get("enabled", False)]
    )

    for name in to_run:
        if name not in targets_cfg:
            print(f"Error: Unknown target: {name}", file=sys.stderr)
            sys.exit(1)
        print(f"Building target: {name}")
        try:
            _dispatch(name, targets_cfg[name], project.config, workdir=workdir)
        except NotImplementedError as exc:
            print(
                f"WARNING: target '{name}' not yet implemented; skipping ({exc})",
                file=sys.stderr,
            )
            continue
        except Exception as exc:
            _logger.exception("Error building target '%s'", name)
            print(f"Error building target '{name}': {exc}", file=sys.stderr)
            sys.exit(1)
```

The `NotImplementedError → skip` clause is what makes `nhf-targets run` (and the slurm array) succeed even when AET/RCH/SOM/SCA stubs are still in place.

- [ ] **Step 3: Add a smoke test**

Append to `tests/test_cli.py` (or create the file if absent):

```python
def test_run_runoff_smoke(tmp_path: Path):
    """Invoking the CLI's run --target runoff dispatches to the new build()."""
    from nhf_spatial_targets.cli import _dispatch
    from tests.test_targets_run import _make_runoff_project

    workdir = _make_runoff_project(tmp_path)
    target_cfg = {}  # _dispatch doesn't use target_cfg for runoff anymore
    pipeline_cfg = {"fabric": {"path": str(tmp_path / "fabric.gpkg")}}
    _dispatch("runoff", target_cfg, pipeline_cfg, workdir=workdir)
    assert (workdir / "targets" / "runoff_targets.nc").exists()
```

- [ ] **Step 4: Run the test**

Run: `pixi run -e dev test tests/test_cli.py::test_run_runoff_smoke -v`
Expected: PASS.

- [ ] **Step 5: Full quality gate**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
pixi run git add src/nhf_spatial_targets/cli.py tests/test_cli.py
pixi run git commit -m "$(cat <<'EOF'
feat(cli): wire run/--target to load Project once and dispatch

The runoff dispatch now constructs a Project via workspace.load (so
config defaults are applied) and calls run.build(project). NotImplementedError
from a stub target is downgraded to a 'skipping' warning so bulk
'nhf-targets run' invocations don't fail noisily until each target lands.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Add `run_all.slurm`

**Files:**
- Create: `run_all.slurm`

- [ ] **Step 1: Verify existing `agg_all.slurm` pattern is still current**

```bash
head -10 /home/rmcd/projects/usgs/nhf-spatial-targets/agg_all.slurm
```
Expected: SLURM directives for `--account=impd`, `--partition=cpu`, `--mem=128G`. Confirm before proceeding.

- [ ] **Step 2: Create `run_all.slurm`**

Write the file at the repo root:

```bash
#!/bin/bash
#SBATCH --job-name=nhf-run
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/run_%a_%A.out
#SBATCH --error=logs/run_%a_%A.err

# NHF Spatial Targets — parallel target-build array.
# Submits one SLURM job per calibration target (5 total). Each target depends
# on its source aggregations being complete (run agg_all.slurm first); targets
# are otherwise independent and can run concurrently on separate compute nodes.
#
# Stub targets (AET, RCH, SOM, SCA) raise NotImplementedError under the hood;
# the CLI catches that and logs "skipping", so the array won't fail noisily on
# bulk submits. As each target's builder lands, the skip becomes a real build
# with no slurm-script change.
#
# Prerequisites:
#   - agg_all.slurm has completed for the sources each target needs
#   - $PROJECT_DIR/fabric.json exists (validate has run)
#
# Usage:
#   mkdir -p logs
#   export PROJECT_DIR=/path/to/your/project
#   sbatch run_all.slurm
#
# Run a single target by index (e.g. runoff = 0):
#   sbatch --array=0 run_all.slurm
#
# Bump memory for an unusually large fabric:
#   sbatch --mem=64G run_all.slurm

set -euo pipefail

REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"

cd "$REPO_DIR" || { echo "ERROR: REPO_DIR=$REPO_DIR not found" >&2; exit 1; }

RUN_TASKS=(
    "run-runoff"   # 0 — Runoff (3 sources: ERA5-Land, GLDAS, MWBM)
    "run-aet"      # 1 — AET (stub until targets/aet.py implemented)
    "run-rch"      # 2 — Recharge (stub)
    "run-som"      # 3 — Soil moisture (stub)
    "run-sca"      # 4 — SCA (stub)
)

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#RUN_TASKS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range" >&2
    exit 2
fi

TASK="${RUN_TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Array task $SLURM_ARRAY_TASK_ID: $TASK ==="
echo "=== Project: $PROJECT_DIR ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:    $(hostname) ==="

pixi run "$TASK" -- --project-dir "$PROJECT_DIR"

echo "=== Done:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
```

The pixi tasks (`run-runoff`, `run-aet`, etc.) already exist in `pixi.toml` (lines 71–75). No `pixi.toml` change is needed for the slurm wiring.

- [ ] **Step 3: Make executable**

```bash
chmod +x /home/rmcd/projects/usgs/nhf-spatial-targets/run_all.slurm
```

- [ ] **Step 4: Sanity-check syntax**

```bash
bash -n /home/rmcd/projects/usgs/nhf-spatial-targets/run_all.slurm
```
Expected: no output (script parses).

- [ ] **Step 5: Commit**

```bash
pixi run git add run_all.slurm
pixi run git commit -m "$(cat <<'EOF'
chore(hpc): add run_all.slurm for parallel target builds

5-element array, one slurm task per calibration target, 32GB / 4h
defaults. Dispatches to existing pixi run-{runoff,aet,rch,som,sca}
tasks. Stub targets are caught by cli.py and logged as 'skipping' so
the array exits cleanly until each target's builder lands.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Update `init_run.py` template + `config/pipeline.yml`

**Files:**
- Modify: `src/nhf_spatial_targets/init_run.py`
- Modify: `config/pipeline.yml`

- [ ] **Step 1: Update `_CONFIG_TEMPLATE` in `init_run.py`**

Replace the `runoff:` block in `_CONFIG_TEMPLATE` (around lines 56–66 of the file as last read) with the new defaults-aligned version, and add `area_crs` under `fabric:`. Use `Edit` on the two relevant strings.

Edit 1 — under `fabric:` block, after `id_col: nhm_id`, before `buffer_deg`:

```python
# OLD
fabric:
  path: /path/to/fabric.gpkg        # absolute path to the HRU fabric
  id_col: nhm_id
  crs: EPSG:4326
  buffer_deg: 0.1                    # degrees to buffer bbox for downloads
```

```python
# NEW
fabric:
  path: /path/to/fabric.gpkg        # absolute path to the HRU fabric
  id_col: nhm_id
  crs: EPSG:4326
  buffer_deg: 0.1                    # degrees to buffer bbox for downloads
  # Equal-area CRS used for HRU area + NN-fill distances. EPSG:5070 is
  # CONUS Albers — override for AK / HI / PR (e.g. EPSG:3338 for Alaska).
  area_crs: "EPSG:5070"
```

Edit 2 — replace the runoff block:

```python
# OLD
  runoff:
    enabled: true
    sources:
      - era5_land
      - gldas_noah_v21_monthly
    time_step: monthly
    period: "2000-01-01/2010-12-31"
    prms_variable: basin_cfs
    range_method: multi_source_minmax
    output_file: runoff_targets.nc
```

```python
# NEW
  runoff:
    enabled: true
    sources:
      - era5_land
      - gldas_noah_v21_monthly
      - mwbm_climgrid
    time_step: monthly
    period: "2000-01-01/2010-12-31"
    prms_variable: basin_cfs
    range_method: multi_source_minmax
    output_file: runoff_targets.nc
    nn_fill: true
    nn_max_candidates: 10
    chunk_months: 12
```

- [ ] **Step 2: Make the same two edits in `config/pipeline.yml`**

The reference config has the same blocks. Apply the same `area_crs` addition and the same runoff-block replacement.

- [ ] **Step 3: Run any `init_run` tests**

Run: `pixi run -e dev test tests/test_init_run.py -v`
Expected: PASS. If a test asserts the exact runoff block content, update its expected value to match the new block.

- [ ] **Step 4: Full quality gate**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
pixi run git add src/nhf_spatial_targets/init_run.py config/pipeline.yml tests/test_init_run.py
pixi run git commit -m "$(cat <<'EOF'
chore(config): add area_crs + runoff defaults to init template + ref config

init_project's config.yml template and the reference config/pipeline.yml
now include fabric.area_crs (EPSG:5070 default) and the runoff target
block with all three sources (era5_land, gldas_noah_v21_monthly,
mwbm_climgrid) plus nn_fill / nn_max_candidates / chunk_months keys.
These match defaults.DEFAULTS so existing-project migration just sees
the same values surfaced explicitly in config.effective.yml.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Update `catalog/variables.yml` runoff range_notes

**Files:**
- Modify: `catalog/variables.yml`

- [ ] **Step 1: Edit the runoff `range_notes`**

Use `Edit` on `catalog/variables.yml`. Replace:

```yaml
        lower_bound = min(era5, gldas, mwbm), upper_bound = max(era5, gldas, mwbm).
```

with:

```yaml
        lower_bound = nanmin(era5, gldas, mwbm), upper_bound = nanmax(era5, gldas, mwbm).
        nanmin/nanmax (xr.concat([...]).min(skipna=True)) means a bound is
        defined whenever >=1 source is finite at that (HRU, time); only
        all-NaN cells produce NaN.
```

- [ ] **Step 2: Catalog test still passes**

Run: `pixi run -e dev test tests/test_catalog.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
pixi run git add catalog/variables.yml
pixi run git commit -m "$(cat <<'EOF'
docs(catalog): clarify NaN-aware semantics for runoff range_notes

The previous wording 'min(...)' was ambiguous about NaN propagation.
The runoff target uses nanmin/nanmax so a bound is defined whenever
at least one source is finite. Documenting this in the catalog keeps
the variable definition aligned with targets/run.py and the recipes doc.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Rewrite `docs/references/calibration-target-recipes.md` § 1 Runoff

**Files:**
- Modify: `docs/references/calibration-target-recipes.md`

- [ ] **Step 1: Edit § 1 Runoff (RUN)**

Open the file. The section to replace begins at the heading `### 1. Runoff (RUN)` and ends just before `### 2. Actual evapotranspiration (AET)`. Replace its body with:

```markdown
### 1. Runoff (RUN)

- **PRMS variable:** `runoff`
- **Sources:** ERA5-Land `ro`, GLDAS-2.1 NOAH `runoff_total = Qs_acc + Qsb_acc`,
  MWBM ClimGrid `runoff` — **all three**
- **Builder:** `src/nhf_spatial_targets/targets/run.py` (implemented)

**Native-unit conversion to mm/month**

- ERA5-Land `ro`: m of water / month (monthly accumulation, `cell_methods: time: sum`).
  Multiply by 1000 → mm/month.
- GLDAS NOAH `runoff_total`: kg m⁻², stored as the **mean of 3-hourly accumulations**
  for the month (NOT a monthly sum), per the NASA GES DISC GLDAS-2.1 README.
  Multiply by `8 × days_in_month` → mm/month.
- MWBM ClimGrid `runoff`: already mm/month native — no conversion. The catalog is
  authoritative for this unit; the builder validates against
  `catalog.source("mwbm_climgrid")["variables"]`.

**mm/month → cfs**

`mm/month × 1e-3 m/mm × HRU_area_m² / days_in_month × 35.3147 ÷ 86400`,
implemented in `mm_per_month_to_cfs`. HRU area is computed by the builder
from the fabric, reprojecting to `project.area_crs` (EPSG:5070 default,
configurable for AK/HI/PR).

**Period semantics (period union, not period intersection)**

The builder takes a single `runoff.period: "<start>/<end>"` from project
config and constructs a master month-start time index covering it. Each
source contributes for `[period_min, period_max] ∩ source_native_range` —
months in the master index that a source does not cover come back as NaN
after `reindex_to_month_start`. The `multi_source_nanminmax` reduction is
NaN-aware, so a source that ends in 2020 simply contributes nothing for
post-2020 cells; the bound is still well-defined as long as ≥1 other
source is finite there.

**Time canonicalization**

ERA5-Land timestamps are end-of-month, GLDAS start-of-month, MWBM
start-of-month. `reindex_to_month_start` converts every source's time
coord via `dt.to_period("M").dt.to_timestamp()` so all three land on the
same month-start slot regardless of native convention. The output NC
carries `time_bnds(time, nv=2)` recording `[month_start, next_month_start)`
per CF convention.

**Spatial extent**

ERA5-Land is subset to CONUS+contributing-watersheds at fetch time
(lon −125 to −66, lat 24.7 to 53). GLDAS is global, MWBM is CONUS at
2.5 arcmin; both are aggregated through gdptools onto the same HRU
fabric, so spatial alignment lands at HRU resolution. The builder raises
if the three sources do not agree on the HRU coord set after reindex —
that indicates different fabrics, a real upstream bug.

**Combination rule**

Per-HRU per-month: `lower_bound = nanmin(era5_cfs, gldas_cfs, mwbm_cfs)`,
`upper_bound = nanmax(era5_cfs, gldas_cfs, mwbm_cfs)` via
`xr.concat([...], dim="source").min(skipna=True)` (and `.max`). An int8
`n_sources(time, id_col)` diagnostic records how many sources contributed
at each cell (0 / 1 / 2 / 3). Cross-link: see "Aggregation, masked_mean,
and target-time NN-fill" later in this document.

**NN-fill (optional, default-on)**

When `runoff.nn_fill: true` (default), the builder writes a second file
`runoff_targets_nn_filled.nc` alongside the honest-NaN
`runoff_targets.nc`. The fill is purely additive: a cKDTree donor walk
in `project.area_crs` finds the nearest finite HRU at each time step and
adopts its bound values; if the nearest is also NaN, the walk continues
until a finite donor is found or `runoff.nn_max_candidates` (default 10)
is exhausted (in which case the cell stays NaN). The filled file carries
an int8 `nn_filled(time, id_col)` flag (0/1) so analysts can see which
cells were imputed. The honest-NaN file is the canonical output; NN-fill
is a downstream-friendly artifact.

**Project config keys**

- `runoff.sources`, `runoff.period`, `runoff.nn_fill`, `runoff.nn_max_candidates`,
  `runoff.chunk_months`, `runoff.output_file`
- Project-level: `fabric.area_crs`

All carry defaults — see `src/nhf_spatial_targets/defaults.py` for the
authoritative source. After `nhf-targets validate`, the fully merged
config is written to `<project>/config.effective.yml` (read-only), and
the diff between user config and defaults is logged to stderr.

**Open / verify**

- ERA5-Land's CONUS-and-contributing-watersheds bbox is in
  `src/nhf_spatial_targets/fetch/era5_land.py`. Confirm GLDAS and MWBM
  aggregations use the same fabric so the multi-source min/max is
  per-HRU consistent (the builder will raise on mismatch).
```

- [ ] **Step 2: Verify the file is still well-formed**

```bash
python -c "open('/home/rmcd/projects/usgs/nhf-spatial-targets/docs/references/calibration-target-recipes.md').read()" 2>&1 | tail
```
Expected: no traceback.

- [ ] **Step 3: Commit**

```bash
pixi run git add docs/references/calibration-target-recipes.md
pixi run git commit -m "$(cat <<'EOF'
docs: rewrite calibration-target-recipes runoff section

Adds MWBM ClimGrid as the third source, documents period-union and
time canonicalization semantics, the new NN-fill artifact, and the
project config keys (with pointer to defaults.py as authoritative).
Cross-references the existing aggregation/NN-fill section already in
this file.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Reconcile `docs/references/processing-steps-reference.md`

**Files:**
- Modify: `docs/references/processing-steps-reference.md`

- [ ] **Step 1: Update the cross-cutting "NaN HRU fill" sub-block**

Find the section `### NaN HRU fill` under `## Cross-cutting conventions`. Replace its body with:

```markdown
### NaN HRU fill

- NaN HRUs after aggregation (partial or no source coverage) are honest:
  the aggregated NCs preserve NaN, no imputation happens at the aggregation stage.
- NN-fill is a **target-stage post-processing step** on the per-HRU per-time
  *bounds* (after the multi-source NaN-aware min/max combination), not on the
  aggregated NCs themselves. When `target.nn_fill` is true (default), the
  target builder writes a separate `<target>_nn_filled.nc` alongside the
  honest-NaN `<target>.nc`. Donor walk in equal-area space (`project.area_crs`).
- Diagnostics written by every multi-source-minmax target:
  - `n_sources(time, id_col)` int8 (0/1/2/3) — finite-source count per cell
  - `nn_filled(time, id_col)` int8 (0/1) — only present in the `*_nn_filled.nc`
    file; flags which cells were imputed
```

- [ ] **Step 2: Verify the per-source bullets in § 1 Runoff still match**

The existing § 1 Runoff already names all three sources and the unit conversions are correct. No further edits to that section.

- [ ] **Step 3: Commit**

```bash
pixi run git add docs/references/processing-steps-reference.md
pixi run git commit -m "$(cat <<'EOF'
docs: align processing-steps NN-fill section with target-stage placement

Earlier wording put NN-fill in normalize/ before any combination; the
agreed design (PR for runoff target builder) puts it after combination,
on the bounds, in a separate *_nn_filled.nc file. Documents the new
n_sources / nn_filled diagnostics.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Update `CLAUDE.md` example commands and NN-fill bullet

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add `pixi run run-runoff` example**

Find the example commands block at the top of `CLAUDE.md` (`# Run a single target` section). Update the example by replacing:

```
# Run a single target
pixi run run-aet -- --project-dir /data/nhf-runs/my-run
```

with:

```
# Run a single target (runoff is implemented; aet/rch/som/sca are stubs)
pixi run run-runoff -- --project-dir /data/nhf-runs/my-run
pixi run run-aet -- --project-dir /data/nhf-runs/my-run
```

- [ ] **Step 2: Expand the NN-fill bullet under Aggregation Transformation Policy**

Find the "Per-HRU transforms (`normalize/methods.py`)" bullet. Append a sentence:

```
NN-fill, when applied, is a **target-stage post-processing step** on the
multi-source-combined bounds (not on aggregated NCs); the target builder
writes a parallel ``<target>_nn_filled.nc`` alongside the honest-NaN
``<target>.nc``. See ``targets/run.py`` for the canonical implementation.
```

- [ ] **Step 3: Commit**

```bash
pixi run git add CLAUDE.md
pixi run git commit -m "$(cat <<'EOF'
docs(claude): document run-runoff command and NN-fill placement

CLAUDE.md gains a one-line pixi run-runoff example and a sentence on
the NN-fill placement convention (target-stage, separate file). Keeps
new agentic sessions on the same page as the recipes / processing-steps
docs without having to read the spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: Final sweep — full test suite + sanity checks

**Files:** none (verification only)

- [ ] **Step 1: Run full quality gate one more time**

```bash
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
```
Expected: all PASS, no formatting changes.

- [ ] **Step 2: Verify the runoff target end-to-end against synthetic data**

```bash
pixi run -e dev test tests/test_targets_run.py tests/test_targets_common.py tests/test_normalize_nn_fill.py tests/test_workspace_defaults.py tests/test_validate_defaults.py tests/test_defaults.py -v
```
Expected: PASS for every one.

- [ ] **Step 3: Skim the manifest of files changed in this branch**

```bash
git log --oneline main..HEAD
git diff --stat main..HEAD
```
Confirm: changes match the expected file map in the spec (Section "Module map" + "Updated config / docs / scripts").

- [ ] **Step 4: Check that no notebook ID was accidentally stripped**

This branch should not modify any notebook, but verify:
```bash
git diff main..HEAD -- '*.ipynb' | head
```
Expected: empty.

- [ ] **Step 5: Confirm `.claude/settings.local.json` is committed if changed**

```bash
git status -- .claude/settings.local.json
```
If it's modified, include it in a final commit per the project memory note that `.claude/settings.local.json` is tracked and should land with whatever PR is in progress.

- [ ] **Step 6: Open the PR**

Branch + PR per the project's git workflow (branch type `feature/<issue#>-runoff-target-builder`). Ensure the PR description references the spec at `docs/superpowers/specs/2026-05-05-runoff-target-builder-design.md` and the design's "Build sequence" so reviewers can map commits to design intent.

---

## Self-review notes

- Every spec section maps to at least one task: defaults layer (1–3), shared infra (4–8), NN-fill (9), runoff builder (10), CLI integration (11), HPC (12), config templates (13), catalog (14), recipes (15), processing-steps (16), CLAUDE.md (17), final sweep (18).
- No placeholders or "implement later" left in any task.
- Type/name consistency: `Project.target(name)`, `Project.area_crs`, `Project.id_col`, `read_aggregated_source`, `reindex_to_month_start`, `multi_source_nanminmax`, `compute_hru_area_and_centroids`, `write_target_nc`, `nn_fill_bounds` are all spelled identically wherever they appear across tasks 2–11.
- Per-source variable names used in `_SOURCE_VAR` (`ro`, `runoff_total`, `runoff`) match the catalog entries verified in `catalog/sources.yml`.
- The `pixi.toml` task aliases the slurm script invokes already exist (lines 71–75); Task 12 does not add them and Task 13 does not modify them.
- Stub-target skip behavior is implemented in `cli.py` (Task 11) by catching `NotImplementedError` from `_dispatch`; the four stub modules themselves are untouched, so nothing else needs to change for the slurm array to exit cleanly.
