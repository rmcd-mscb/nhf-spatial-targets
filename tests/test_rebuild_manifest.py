"""Tests for ``nhf_spatial_targets.rebuild_manifest``.

``rebuild-manifest`` is the one authoritative deterministic projection of
(datastore consolidated dirs x catalog) U project ``data/aggregated/`` dirs
U ``targets/`` NCs U ``fabric.json`` into a complete ``manifest.json``.

Invariants locked here (spec decisions E/F, plan PR-2):

- Deterministic, byte-identical on re-run (the idempotency contract).
- year/period parse from aggregated filenames incl. ``ssebop_2000_agg.nc``
  and ``daymet_na_1980_agg.nc``.
- Non-catalog-key dir -> ``derived_variant: True`` (never orphan a shipped NC).
- Non-publishable source (``watergap22d``) still recorded.
- No ``datetime.now()`` reachable from the rebuild path.
- Concurrent flock merge preserves identity fields.
- ``reconcile-manifest`` removed; not a registered command.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nhf_spatial_targets.rebuild_manifest import parse_aggregated_filename


@pytest.mark.parametrize(
    "name,year",
    [
        ("ssebop_2000_agg.nc", 2000),
        ("daymet_na_1980_agg.nc", 1980),
        ("mod10c1_v061_2020_agg.nc", 2020),
        ("merra2_agg.nc", None),  # single-shot, no year
        ("snodas_agg.nc", None),
    ],
)
def test_parse_year(name, year):
    assert parse_aggregated_filename(name)[1] == year


# ---------------------------------------------------------------------------
# Task 2.3: build_source_entry
# ---------------------------------------------------------------------------


def test_source_entry_catalog_key(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import build_source_entry

    d = tmp_path / "merra2"
    d.mkdir()
    (d / "merra2_2000.nc").write_bytes(b"x")
    entry = build_source_entry("merra2", d, compute_sha256=False)
    assert entry["source_key"] == "merra2"
    assert entry["provenance"] == "reconstructed"
    assert entry["derived_variant"] is False
    assert len(entry["files"]) == 1
    assert "sha256" not in entry["files"][0]  # opt-in only


def test_source_entry_noncatalog_is_derived_variant(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import build_source_entry

    d = tmp_path / "era5_land_sd"
    d.mkdir()
    (d / "era5_land_sd_agg.nc").write_bytes(b"x")
    entry = build_source_entry("era5_land_sd", d, compute_sha256=False)
    assert entry["derived_variant"] is True


def test_source_entry_period_from_years(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import build_source_entry

    d = tmp_path / "ssebop"
    d.mkdir()
    (d / "ssebop_2000_agg.nc").write_bytes(b"x")
    (d / "ssebop_2003_agg.nc").write_bytes(b"x")
    entry = build_source_entry("ssebop", d, compute_sha256=False)
    assert entry["period"] == "2000/2003"


def test_source_entry_sha256_optin(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import build_source_entry

    d = tmp_path / "merra2"
    d.mkdir()
    (d / "merra2_2000.nc").write_bytes(b"x")
    entry = build_source_entry("merra2", d, compute_sha256=True)
    assert "sha256" in entry["files"][0]


def test_source_entry_files_sorted(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import build_source_entry

    d = tmp_path / "merra2"
    d.mkdir()
    for name in ("merra2_2002.nc", "merra2_2000.nc", "merra2_2001.nc"):
        (d / name).write_bytes(b"x")
    entry = build_source_entry("merra2", d, compute_sha256=False)
    paths = [f["path"] for f in entry["files"]]
    assert paths == sorted(paths)


# ---------------------------------------------------------------------------
# Task 2.4: synthesize_steps
# ---------------------------------------------------------------------------


def _kind_order(kind):
    return ["consolidate", "aggregate", "target", "validate"].index(kind)


def test_synthesize_steps_sorted_and_kinds(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import synthesize_steps

    ds = tmp_path / "datastore" / "merra2"
    ds.mkdir(parents=True)
    (ds / "merra2_2000.nc").write_bytes(b"x")
    agg = tmp_path / "proj" / "data" / "aggregated" / "merra2"
    agg.mkdir(parents=True)
    (agg / "merra2_agg.nc").write_bytes(b"x")
    tgt = tmp_path / "proj" / "targets"
    tgt.mkdir(parents=True)
    (tgt / "aet_targets.nc").write_bytes(b"x")
    (tmp_path / "proj" / "fabric.json").write_text("{}")

    steps = synthesize_steps(
        datastore=tmp_path / "datastore",
        project_dir=tmp_path / "proj",
        compute_sha256=False,
    )
    kinds = [s["kind"] for s in steps]
    assert kinds == sorted(kinds, key=_kind_order)
    assert {"consolidate", "aggregate", "target", "validate"} <= set(kinds)
    assert all(s["provenance"] == "reconstructed" for s in steps)
    # Timestamps are mtime-derived, never None.
    assert all(s["timestamp_utc"] for s in steps)


def test_synthesize_steps_validate_only_when_minimal(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import synthesize_steps

    (tmp_path / "datastore").mkdir()
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "fabric.json").write_text("{}")

    steps = synthesize_steps(
        datastore=tmp_path / "datastore",
        project_dir=proj,
        compute_sha256=False,
    )
    assert [s["kind"] for s in steps] == ["validate"]
    assert steps[0]["source_key"] is None


# ---------------------------------------------------------------------------
# Task 2.5: rebuild_manifest (assemble + read-merge + write/dry-run)
# ---------------------------------------------------------------------------

import json  # noqa: E402

import yaml  # noqa: E402

from nhf_spatial_targets.workspace import load as load_project  # noqa: E402


def _make_project(tmp_path, *, datastore_dirs=(), aggregated_dirs=(), targets=()):
    """Build a loadable Project over a synthetic datastore/aggregated/targets tree.

    ``datastore_dirs`` / ``aggregated_dirs`` are ``{dirname: [filenames]}``;
    ``targets`` is a list of target NC filenames.
    """
    datastore = tmp_path / "datastore"
    datastore.mkdir()
    for name, files in dict(datastore_dirs).items():
        d = datastore / name
        d.mkdir(parents=True)
        for fn in files:
            (d / fn).write_bytes(b"x")

    proj = tmp_path / "proj"
    proj.mkdir()
    agg_root = proj / "data" / "aggregated"
    for name, files in dict(aggregated_dirs).items():
        d = agg_root / name
        d.mkdir(parents=True)
        for fn in files:
            (d / fn).write_bytes(b"x")
    if targets:
        tdir = proj / "targets"
        tdir.mkdir(parents=True)
        for fn in targets:
            (tdir / fn).write_bytes(b"x")

    (proj / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": "/fake/fabric.gpkg", "id_col": "hru_id"},
                "datastore": str(datastore),
            }
        )
    )
    (proj / "fabric.json").write_text(
        json.dumps(
            {"sha256": "abc", "hru_count": 3, "id_col": "hru_id", "id_col_sorted": True}
        )
    )
    return load_project(proj)


def test_rebuild_is_byte_identical_on_rerun(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import rebuild_manifest

    project = _make_project(
        tmp_path,
        datastore_dirs={"merra2": ["merra2_2000.nc", "merra2_2001.nc"]},
        aggregated_dirs={
            "merra2": ["merra2_agg.nc"],
            "ssebop": ["ssebop_2000_agg.nc"],
            "era5_land_sd": ["era5_land_sd_agg.nc"],
        },
        targets=["aet_targets.nc"],
    )
    m1 = rebuild_manifest(project, dry_run=True)
    m2 = rebuild_manifest(project, dry_run=True)
    assert json.dumps(m1, indent=2) == json.dumps(m2, indent=2)


def test_rebuild_dry_run_writes_nothing(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import rebuild_manifest

    project = _make_project(tmp_path, aggregated_dirs={"merra2": ["merra2_agg.nc"]})
    assert not project.manifest_path.exists()
    rebuild_manifest(project, dry_run=True)
    assert not project.manifest_path.exists()


def test_rebuild_writes_canonical_manifest(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import rebuild_manifest
    from nhf_spatial_targets.release.lineage import (
        CURRENT_MANIFEST_SCHEMA_VERSION,
        _new_manifest_skeleton,
    )

    project = _make_project(tmp_path, aggregated_dirs={"merra2": ["merra2_agg.nc"]})
    rebuild_manifest(project)
    m = json.loads(project.manifest_path.read_text())
    assert set(m) >= set(_new_manifest_skeleton())
    assert m["manifest_schema_version"] == CURRENT_MANIFEST_SCHEMA_VERSION
    assert "merra2" in m["sources"]
    assert all(s["provenance"] == "reconstructed" for s in m["steps"])


def test_rebuild_preserves_created_utc_and_fabric(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import rebuild_manifest

    project = _make_project(tmp_path, aggregated_dirs={"merra2": ["merra2_agg.nc"]})
    seeded = {
        "manifest_schema_version": 1,
        "created_utc": "2020-01-01T00:00:00+00:00",
        "last_validated_utc": "2020-06-01T00:00:00+00:00",
        "nhf_spatial_targets_version": "0.0.1",
        "fabric": {"id_col": "hru_id", "sha256": "SEEDED"},
        "release": {"sb_id": "abc123"},
        "sources": {},
        "steps": [],
    }
    project.manifest_path.write_text(json.dumps(seeded))

    rebuild_manifest(project)
    m = json.loads(project.manifest_path.read_text())
    assert m["created_utc"] == "2020-01-01T00:00:00+00:00"  # never re-minted
    assert m["last_validated_utc"] == "2020-06-01T00:00:00+00:00"  # not re-minted
    assert m["fabric"]["sha256"] == "SEEDED"  # authorship preserved
    assert m["release"] == {"sb_id": "abc123"}  # extra blocks read-merged
    assert "merra2" in m["sources"]  # derived catalog regenerated


def test_rebuild_includes_derived_variant(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import rebuild_manifest

    project = _make_project(
        tmp_path, aggregated_dirs={"era5_land_sd": ["era5_land_sd_agg.nc"]}
    )
    m = rebuild_manifest(project, dry_run=True)
    assert m["sources"]["era5_land_sd"]["derived_variant"] is True


def test_rebuild_records_nonpublishable_source(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import rebuild_manifest

    project = _make_project(
        tmp_path, aggregated_dirs={"watergap22d": ["watergap22d_agg.nc"]}
    )
    m = rebuild_manifest(project, dry_run=True)
    assert "watergap22d" in m["sources"]
    assert m["sources"]["watergap22d"]["derived_variant"] is False


def test_rebuild_concurrent_flock_merge(tmp_path):
    import threading

    from nhf_spatial_targets.rebuild_manifest import rebuild_manifest

    project = _make_project(
        tmp_path,
        aggregated_dirs={"merra2": ["merra2_agg.nc"], "ssebop": ["ssebop_2000_agg.nc"]},
    )
    project.manifest_path.write_text(
        json.dumps(
            {
                "manifest_schema_version": 1,
                "created_utc": "2020-01-01T00:00:00+00:00",
                "last_validated_utc": None,
                "nhf_spatial_targets_version": "0.0.1",
                "fabric": {"id_col": "hru_id"},
                "sources": {},
                "steps": [],
            }
        )
    )

    errors = []

    def _run():
        try:
            rebuild_manifest(project)
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=_run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    m = json.loads(project.manifest_path.read_text())
    assert m["created_utc"] == "2020-01-01T00:00:00+00:00"  # identity survived races
    assert "merra2" in m["sources"]


# ---------------------------------------------------------------------------
# Task 2.6: no datetime.now() reachable from the rebuild path
# ---------------------------------------------------------------------------


def test_no_datetime_now_in_rebuild_module():
    import inspect

    from nhf_spatial_targets import rebuild_manifest as rm

    src = inspect.getsource(rm)
    assert "datetime.now" not in src, (
        "rebuild_manifest must derive all timestamps from file mtime "
        "(spec decision E). Use lineage.iso_from_mtime."
    )


def test_frozen_clock_guard_rebuild_dry_run(tmp_path, monkeypatch):
    import datetime as _dt

    from nhf_spatial_targets.rebuild_manifest import rebuild_manifest
    from nhf_spatial_targets.release import lineage

    class _NoNow(_dt.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ARG003
            raise AssertionError("datetime.now() called on the rebuild path")

    # build_step_record + iso_from_mtime + _file_basics all reference
    # lineage.datetime; freezing .now here proves the rebuild never mints a clock.
    monkeypatch.setattr(lineage, "datetime", _NoNow)
    project = _make_project(
        tmp_path,
        datastore_dirs={"merra2": ["merra2_2000.nc"]},
        aggregated_dirs={"merra2": ["merra2_agg.nc"]},
        targets=["aet_targets.nc"],
    )
    # Must not raise: every timestamp is mtime-derived.
    rebuild_manifest(project, dry_run=True)


# ---------------------------------------------------------------------------
# Task 2.7: CLI wiring + reconcile removal
# ---------------------------------------------------------------------------


def test_rebuild_manifest_cmd_dry_run_writes_nothing(tmp_path):
    from nhf_spatial_targets.cli.run import rebuild_manifest_cmd

    project = _make_project(tmp_path, aggregated_dirs={"merra2": ["merra2_agg.nc"]})
    rebuild_manifest_cmd(project.workdir, dry_run=True)
    assert not project.manifest_path.exists()


def test_rebuild_manifest_cmd_writes(tmp_path):
    from nhf_spatial_targets.cli.run import rebuild_manifest_cmd

    project = _make_project(tmp_path, aggregated_dirs={"merra2": ["merra2_agg.nc"]})
    rebuild_manifest_cmd(project.workdir)
    assert project.manifest_path.exists()
    m = json.loads(project.manifest_path.read_text())
    assert "merra2" in m["sources"]


def test_reconcile_manifest_is_removed():
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("nhf_spatial_targets.reconcile")

    from nhf_spatial_targets.cli import app

    assert "reconcile-manifest" not in app  # command de-registered
    assert "rebuild-manifest" in app  # replacement registered


# ---------------------------------------------------------------------------
# Review fixup (PR #281): content-derivation + shared-Lustre resilience
# ---------------------------------------------------------------------------


def test_target_nc_params_reads_real_attrs(tmp_path):
    """A real target NC's period/sources attrs land on the target step.

    Every other fixture writes placeholder bytes (-> {} via the fallback);
    this exercises the actual xarray attr read + numpy .tolist() conversion,
    which is the one runtime path manifest JSON-stability depends on.
    """
    import numpy as np
    import xarray as xr

    from nhf_spatial_targets.rebuild_manifest import rebuild_manifest

    project = _make_project(tmp_path, aggregated_dirs={"merra2": ["merra2_agg.nc"]})
    tdir = project.workdir / "targets"
    tdir.mkdir(parents=True)
    ds = xr.Dataset(attrs={"period": "2000-01-01/2010-12-31"})
    # numpy array attr -> must round-trip to a plain JSON list via .tolist().
    ds.attrs["sources"] = np.array(["merra2", "gldas"])
    ds.to_netcdf(tdir / "aet_targets.nc")

    m = rebuild_manifest(project, dry_run=True)
    target_steps = [s for s in m["steps"] if s["kind"] == "target"]
    assert len(target_steps) == 1
    params = target_steps[0]["params"]
    assert params["period"] == "2000-01-01/2010-12-31"
    assert params["sources"] == ["merra2", "gldas"]  # numpy -> list
    # The whole manifest must remain JSON-serializable (no numpy leaks).
    json.dumps(m)


def test_target_nc_params_placeholder_is_benign(tmp_path, caplog):
    """A non-NetCDF placeholder (no NetCDF/HDF5 magic) yields {} AND warns.

    Target writers may not yet persist resolved-param attrs, and many fixtures
    write plain-text placeholder bytes; such a file is NOT a corrupt published
    artifact -- it degrades to empty params with a warning, never fatally.
    """
    import logging

    from nhf_spatial_targets.rebuild_manifest import _target_nc_params

    bad = tmp_path / "aet_targets.nc"
    bad.write_bytes(b"not a netcdf")
    with caplog.at_level(logging.WARNING):
        assert _target_nc_params(bad) == {}
    assert any("aet_targets.nc" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "magic",
    [
        b"\x89HDF\r\n\x1a\n",  # HDF5 (NetCDF4) magic
        b"CDF\x01",  # classic NetCDF-1 magic
        b"CDF\x02",  # 64-bit-offset NetCDF-2 magic
    ],
)
def test_target_nc_params_corrupt_netcdf_is_fatal(tmp_path, magic):
    """A file carrying NetCDF/HDF5 magic but unopenable is a corrupt published
    artifact -- it must RAISE (issue #283), not degrade to {}. Otherwise the
    publish gate green-lights a truncated/corrupt target NC: the projection
    still records the target step (built from the file's existence), only its
    params go empty, so both the 'matching step' and the drift checks pass.
    """
    from nhf_spatial_targets.rebuild_manifest import _target_nc_params

    corrupt = tmp_path / "aet_targets.nc"
    corrupt.write_bytes(magic + b"\x00truncated-garbage")
    with pytest.raises(ValueError, match="truncated or corrupt"):
        _target_nc_params(corrupt)


def test_rebuild_aggregated_dir_wins_union(tmp_path):
    """When a key is in BOTH datastore and aggregated, the aggregated dir wins.

    The published fabric ships the aggregated NC, so sources[key].files must
    reflect it; the datastore NCs are still recorded by the consolidate step.
    """
    from nhf_spatial_targets.rebuild_manifest import rebuild_manifest

    project = _make_project(
        tmp_path,
        datastore_dirs={"merra2": ["merra2_2000.nc", "merra2_2001.nc"]},
        aggregated_dirs={"merra2": ["merra2_agg.nc"]},
    )
    m = rebuild_manifest(project, dry_run=True)
    files = [Path(f["path"]).name for f in m["sources"]["merra2"]["files"]]
    assert files == ["merra2_agg.nc"]  # aggregated wins, not the datastore NCs
    # The datastore NCs still surface via the consolidate step.
    consolidate = [
        s
        for s in m["steps"]
        if s["kind"] == "consolidate" and s["source_key"] == "merra2"
    ]
    assert len(consolidate) == 1
    cons_files = [Path(o["path"]).name for o in consolidate[0]["outputs"]]
    assert cons_files == ["merra2_2000.nc", "merra2_2001.nc"]


def test_rebuild_skips_vanished_file_without_aborting(tmp_path, monkeypatch, caplog):
    """A file that vanishes mid-scan (TOCTOU) is logged and skipped, not fatal.

    Restores the log-and-continue resilience the deleted reconcile.py /
    release/rebuild.py carried for the shared Lustre datastore.
    """
    import logging

    from nhf_spatial_targets import rebuild_manifest as rm

    project = _make_project(
        tmp_path,
        aggregated_dirs={"merra2": ["merra2_2000_agg.nc", "merra2_2001_agg.nc"]},
    )
    real = rm.lineage.output_file_entry

    def flaky(path, *, compute_sha256):
        if path.name == "merra2_2000_agg.nc":
            raise OSError("stale NFS handle")
        return real(path, compute_sha256=compute_sha256)

    monkeypatch.setattr(rm.lineage, "output_file_entry", flaky)
    with caplog.at_level(logging.WARNING):
        m = rm.rebuild_manifest(project, dry_run=True)  # must NOT raise

    # The surviving file is recorded; the vanished one is skipped + logged.
    agg = [s for s in m["steps"] if s["kind"] == "aggregate"]
    assert len(agg) == 1
    names = [Path(o["path"]).name for o in agg[0]["outputs"]]
    assert names == ["merra2_2001_agg.nc"]
    assert any("merra2_2000_agg.nc" in r.message for r in caplog.records)
