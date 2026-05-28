"""Live-path consolidate-step output checksums (issue #263).

The live fetch path of ``ua_swe`` / ``snodas`` / ``margulis_wus_sr`` builds
``manifest.json.steps[]`` ``kind=consolidate`` records from the per-year /
per-water-year records produced by the download+consolidate loop. Each of
those records stores the consolidated NC path under ``daily_path`` (set from
the consolidator's return value on **both** the fresh-consolidate path and the
mtime-idempotent skip path -- the consolidators return the existing ``out_path``
when skipping).

Issue #263: the step-output builders probed a fixed key tuple
(``consolidated_nc`` / ``consolidated_path`` / ``path``) that never included
``daily_path``, so ``step["outputs"]`` came out empty and the consolidate steps
carried no output sha256 -- undercutting the release checksum/provenance
feature (PR-C #258).

These tests drive each module's ``_update_manifest`` directly with a record
shaped exactly like the live loop produces (``daily_path`` -> a real NC on
disk). They fail before the fix (empty ``outputs[]``) and pin the contract that
a record carrying ``daily_path`` -- whether freshly consolidated or
mtime-skipped -- attaches an output entry with a correct sha256.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

import nhf_spatial_targets.catalog as catalog
from nhf_spatial_targets.fetch import margulis_wus_sr, snodas, ua_swe
from nhf_spatial_targets.release.lineage import sha256_file


def _make_project(tmp_path: Path) -> Path:
    """Materialize a minimal project the workspace loader accepts.

    ``_load_project`` needs ``config.yml`` + ``fabric.json``; the stub
    fabric path need not exist on disk for these manifest-only tests.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    datastore = tmp_path / "datastore"
    datastore.mkdir(exist_ok=True)
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


def _write_tiny_nc(path: Path) -> Path:
    """Write a small real NetCDF to stand in for a consolidated daily NC."""
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"swe": ("x", np.arange(3.0, dtype="float32"))}).to_netcdf(path)
    return path


def _consolidate_step(workdir: Path, source_key: str) -> dict:
    """Return the single ``kind=consolidate`` step for *source_key*."""
    manifest = json.loads((workdir / "manifest.json").read_text())
    steps = [
        s
        for s in manifest["steps"]
        if s["kind"] == "consolidate" and s["source_key"] == source_key
    ]
    assert len(steps) == 1, f"expected one consolidate step, got {len(steps)}"
    return steps[0]


def _assert_outputs_match(step: dict, expected_ncs: list[Path]) -> None:
    """The step's outputs must cover every consolidated NC with a real sha256."""
    outputs = step["outputs"]
    assert outputs, "consolidate step recorded empty outputs[] (issue #263)"
    by_path = {o["path"]: o for o in outputs}
    for nc in expected_ncs:
        entry = by_path.get(str(nc))
        assert entry is not None, f"no output entry for {nc}"
        assert entry.get("sha256") == sha256_file(nc), f"sha256 mismatch for {nc}"


def test_ua_swe_consolidate_step_records_output_checksums(tmp_path: Path) -> None:
    workdir = _make_project(tmp_path)
    daily_dir = workdir / "datastore" / "ua_swe" / "daily"
    ncs = [
        _write_tiny_nc(daily_dir / "ua_swe_daily_WY2010.nc"),
        _write_tiny_nc(daily_dir / "ua_swe_daily_WY2011.nc"),
    ]
    wy_records = [
        {"water_year": 2010, "status": "downloaded", "daily_path": str(ncs[0])},
        {"water_year": 2011, "status": "downloaded", "daily_path": str(ncs[1])},
    ]
    ua_swe._update_manifest(
        workdir,
        "2010/2010",
        catalog.source("ua_swe"),
        wy_records,
        "https://example.org/archive",
        None,
    )
    _assert_outputs_match(_consolidate_step(workdir, "ua_swe"), ncs)


def test_snodas_consolidate_step_records_output_checksums(tmp_path: Path) -> None:
    workdir = _make_project(tmp_path)
    daily_dir = workdir / "datastore" / "snodas" / "daily"
    nc = _write_tiny_nc(daily_dir / "snodas_daily_2020.nc")
    year_records = [{"year": 2020, "daily_path": str(nc)}]
    snodas._update_manifest(
        workdir,
        "2020/2020",
        catalog.source("snodas"),
        year_records,
        "https://example.org/archive",
    )
    _assert_outputs_match(_consolidate_step(workdir, "snodas"), [nc])


def test_margulis_consolidate_step_records_output_checksums(tmp_path: Path) -> None:
    workdir = _make_project(tmp_path)
    daily_dir = workdir / "datastore" / "margulis_wus_sr" / "daily"
    nc = _write_tiny_nc(daily_dir / "margulis_wus_sr_daily_2000.nc")
    year_records = [{"year": 2000, "daily_path": str(nc)}]
    margulis_wus_sr._update_manifest(
        workdir,
        "2000/2000",
        catalog.source("margulis_wus_sr"),
        year_records,
        {"fabrics": ["or"], "notes": "test"},
        (-124.0, 42.0, -116.0, 46.0),
    )
    _assert_outputs_match(_consolidate_step(workdir, "margulis_wus_sr"), [nc])
