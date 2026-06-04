"""Shared pytest fixtures for unit tests.

The preflight system-file checks (_check_cdsapirc, _check_netrc_earthdata)
require ~/.cdsapirc and ~/.netrc to exist on the developer's machine.  Unit
tests that call validate_workspace() use the ``no_system_cred_checks`` fixture
to suppress these checks so that CI and dev machines without those files can
still run the full unit suite.

Tests that exercise the check functions themselves call the functions directly
with the ``_home`` parameter rather than relying on module-level patching.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from nhf_spatial_targets.workspace import Project


@pytest.fixture()
def no_system_cred_checks():
    """Suppress ~/.cdsapirc and ~/.netrc existence checks (for validate_workspace tests)."""
    with (
        patch(
            "nhf_spatial_targets.validate._check_cdsapirc",
            return_value=None,
        ),
        patch(
            "nhf_spatial_targets.validate._check_netrc_earthdata",
            return_value=None,
        ),
    ):
        yield


def write_year_nc(
    path,
    year: int,
    var: str,
    id_col: str = "nhm_id",
    value: float | None = None,
):
    """Write a synthetic per-year aggregated NC at the given path.

    Used by tests for the ``targets/`` helper modules and ``targets/run.py``.
    The file matches the per-year aggregated layout established in PR #51:
    ``<source_key>/<source_key>_<YYYY>_agg.nc`` with ``time`` and ``id_col``
    dims.

    If ``value`` is None, fills with sequential floats; if a number, fills
    constant.
    """
    times = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    hrus = [1, 2, 3]
    if value is None:
        data = np.arange(len(times) * len(hrus), dtype=np.float32).reshape(
            len(times), len(hrus)
        )
    else:
        data = np.full((len(times), len(hrus)), value, dtype=np.float32)
    ds = xr.Dataset(
        {var: (("time", id_col), data)},
        coords={"time": times, id_col: hrus},
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def make_minimal_project(tmp_path, fabric_path: str | None = None):
    """Build a minimal project skeleton for tests of workspace.load() consumers.

    Writes ``config.yml`` with a fake datastore + fabric path and a stub
    ``fabric.json``. Returns the project directory.
    """
    workdir = tmp_path / "proj"
    workdir.mkdir()
    (workdir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "datastore": str(tmp_path / "store"),
                "fabric": {
                    "path": fabric_path or str(tmp_path / "f.gpkg"),
                    "id_col": "nhm_id",
                },
            }
        )
    )
    (workdir / "fabric.json").write_text(json.dumps({"id_col": "nhm_id"}))
    return workdir


# ---------------------------------------------------------------------------
# Release publish/dry-run test helpers
# ---------------------------------------------------------------------------

# A populated author list satisfies the publish pre-flight; tests pass
# ``authors=[]`` to exercise the empty-authors gate.
DEFAULT_RELEASE_AUTHORS = [
    {"given": "Jane", "family": "Doe", "affiliation": "U.S. Geological Survey"}
]


def build_release_project(
    tmp_path,
    *,
    fabric_label: str = "gfv_test",
    authors: list | None = None,
    with_manifest: bool = True,
) -> Project:
    """Synthesize a release-ready ``Project`` on disk for publish/dry-run tests.

    Mirrors tests/test_release_build.py's project fixture (publishable
    era5_land kept, watergap22d excluded, daymet metadata-only) and adds the
    knobs the publish pre-flight exercises: ``authors`` (``[]`` triggers the
    empty-authors gate) and ``with_manifest`` (``False`` removes manifest.json
    to trigger the missing-manifest gate).
    """
    workdir = tmp_path / "proj"
    datastore = tmp_path / "store"
    fabric_file = tmp_path / "fab" / "gfv_test.gpkg"
    fabric_file.parent.mkdir(parents=True, exist_ok=True)
    fabric_file.write_bytes(b"GPKG-bytes")

    def _touch(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    agg = workdir / "data" / "aggregated"
    _touch(agg / "era5_land" / "era5_land_1980_agg.nc", b"era5-1980")
    _touch(agg / "watergap22d" / "watergap22d_1980_agg.nc", b"wg-held")
    _touch(agg / "daymet" / "daymet_na_1980_agg.nc", b"daymet-1980")
    _touch(workdir / "targets" / "runoff_targets.nc", b"runoff")
    _touch(workdir / "targets" / "aet_targets.nc", b"aet")

    _touch(datastore / "era5_land" / "daily" / "era5_land_daily_1980.nc", b"d-1980")
    _touch(datastore / "era5_land" / "monthly" / "era5_land_monthly_1980.nc", b"m-1980")
    _touch(datastore / "daymet" / "daymet_consolidated.nc", b"should-not-stage")

    config = {
        "fabric": {"path": str(fabric_file), "id_col": "nhm_id"},
        "release": {
            "authors": DEFAULT_RELEASE_AUTHORS if authors is None else authors,
            "fabric_label": fabric_label,
            "doi": None,
            "abstract_notes": "Synthetic project for publish tests.",
            "ipds_number": "IP-123456",
        },
        "targets": {
            "runoff": {"enabled": True, "period": "1980/2020"},
            "aet": {"enabled": True, "period": "2000/2010"},
        },
    }

    # config.yml on disk + a CURRENT config.effective.yml (the version/hash
    # stamp written by validate._write_effective_config). The PR-4 publish
    # staleness gate reads both and refuses if config.effective.yml's recorded
    # source_config_sha256 != sha256(config.yml); seed them in sync so the
    # happy-path publish flows clear the gate. Tests that exercise staleness
    # mutate config.yml or the effective config after this call.
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "config.yml").write_text(yaml.safe_dump(config))
    from nhf_spatial_targets.validate import _write_effective_config

    _write_effective_config(workdir, config)

    fabric = {
        "path": str(fabric_file),
        "bbox": {"minx": -125.0, "miny": 24.0, "maxx": -66.0, "maxy": 53.0},
        "hru_count": 4242,
    }
    project = Project(
        workdir=workdir,
        datastore=datastore,
        config=config,
        fabric=fabric,
        dir_mode=None,
    )
    if with_manifest:
        # A release-ready fixture carries a COMPLETE manifest: seed the fabric
        # authorship block (as ``validate`` would) then project the on-disk
        # datastore / aggregated / targets artifacts into sources[]/steps[] via
        # ``rebuild_manifest``, so the PR-3 publish completeness gate clears for
        # the happy paths. Tests that need an incomplete/stale manifest mutate
        # disk (add a source dir) or the manifest after this call.
        from nhf_spatial_targets.rebuild_manifest import rebuild_manifest
        from nhf_spatial_targets.release.lineage import (
            _new_manifest_skeleton,
            atomic_write_manifest,
        )

        skeleton = _new_manifest_skeleton()
        skeleton["fabric"] = {
            "path": str(fabric_file),
            "id_col": "nhm_id",
            "hru_count": 4242,
        }
        atomic_write_manifest(project.manifest_path, skeleton)
        rebuild_manifest(project, dry_run=False)
    return project


def _md5_file(filename: str) -> str:
    h = hashlib.md5()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class FakeSbSession:
    """In-memory stand-in for ``sciencebasepy.SbSession`` for offline tests.

    Maintains an item store keyed by id and records every mutating call so a
    test can assert read-only behavior (``created`` / ``updated`` / ``uploaded``
    / ``deleted``). ``upload_file_to_item`` records each file's **MD5** as its
    checksum value -- matching ScienceBase, which stores MD5 -- so the
    SbClient skip-if-checksum-matches fast path fires on an unchanged re-upload.

    A missing id raises ``Exception("Resource not found")``, mirroring
    sciencebasepy, so the publish layer's stale-registry detection is exercised.
    """

    def __init__(self) -> None:
        self.items: dict[str, dict] = {}
        self._counter = 0
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self.uploaded: list[tuple[str, str]] = []
        self.deleted: list[tuple[str, str]] = []

    # -- helpers for tests to pre-seed remote state -----------------------
    def seed_item(
        self,
        sb_id: str,
        *,
        title: str,
        parent_id: str | None = None,
        files: list[dict] | None = None,
    ) -> dict:
        item = {"id": sb_id, "title": title, "files": list(files or [])}
        if parent_id is not None:
            item["parentId"] = parent_id
        self.items[sb_id] = item
        return item

    # -- sciencebasepy surface used by SbClient ---------------------------
    def get_item(self, itemid, params=None):
        if itemid not in self.items:
            raise Exception("Resource not found")
        return copy.deepcopy(self.items[itemid])

    def create_item(self, item_json):
        self.created.append(copy.deepcopy(item_json))
        self._counter += 1
        new_id = f"sb{self._counter}"
        item = copy.deepcopy(item_json)
        item["id"] = new_id
        item.setdefault("files", [])
        self.items[new_id] = item
        return copy.deepcopy(item)

    def update_item(self, item_json):
        self.updated.append(copy.deepcopy(item_json))
        sb_id = item_json["id"]
        self.items[sb_id].update(copy.deepcopy(item_json))
        return copy.deepcopy(self.items[sb_id])

    def get_child_ids(self, parentid):
        return [i for i, it in self.items.items() if it.get("parentId") == parentid]

    def upload_file_to_item(self, item, filename, scrape_file=True):
        sb_id = item["id"]
        stored = self.items[sb_id]
        name = os.path.basename(filename)
        files = [f for f in stored.get("files", []) if f["name"] != name]
        files.append(
            {
                "name": name,
                "size": os.path.getsize(filename),
                "checksum": {"type": "MD5", "value": _md5_file(filename)},
            }
        )
        stored["files"] = files
        self.uploaded.append((sb_id, name))
        return copy.deepcopy(stored)

    def delete_file(self, sb_filename, item):
        sb_id = item["id"]
        stored = self.items[sb_id]
        stored["files"] = [
            f for f in stored.get("files", []) if f["name"] != sb_filename
        ]
        self.deleted.append((sb_filename, sb_id))
        return copy.deepcopy(stored)

    def is_logged_in(self):
        return True

    def ping(self):
        return {"ok": True}


def make_sb_client(session, **kwargs):
    """Wrap *session* in an SbClient with a no-op sleep (no wall-clock backoff)."""
    from nhf_spatial_targets.release.sb_client import SbClient

    kwargs.setdefault("sleep", lambda _: None)
    return SbClient(session, **kwargs)
