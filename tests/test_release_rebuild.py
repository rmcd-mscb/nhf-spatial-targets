"""Tests for ``nhf_spatial_targets.release.rebuild``.

Bootstraps lineage steps for projects that predate PR-B (#246). The
invariants we lock here:

- Each manifest source entry produces the right kind+source_key step
  shapes from on-disk evidence (consolidate, aggregate).
- Target NCs in ``<project>/targets/`` produce target / nn_fill steps.
- Idempotent: re-running rebuild_lineage doesn't duplicate steps even
  when synthesized variants would conflict with live ones from the
  post-PR-B pipeline.
- SHA256 is opt-in.
- ``dry_run=True`` returns the same summary without mutating the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from nhf_spatial_targets.release.rebuild import rebuild_lineage
from nhf_spatial_targets.workspace import load as load_project


@pytest.fixture
def project(tmp_path: Path):
    """Minimal Project fixture matching test_aggregate_driver.py shape."""
    datastore = tmp_path / "datastore"
    datastore.mkdir()
    config = {
        "fabric": {"path": "/fake/fabric.gpkg", "id_col": "hru_id"},
        "datastore": str(datastore),
    }
    (tmp_path / "config.yml").write_text(yaml.dump(config))
    (tmp_path / "fabric.json").write_text(
        json.dumps(
            {
                "sha256": "abc123",
                "hru_count": 3,
                "id_col": "hru_id",
                "id_col_sorted": True,
            }
        )
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    return load_project(tmp_path)


def _write_consolidated_nc(datastore: Path, source_key: str, name: str) -> Path:
    """Create a tiny placeholder NC for output_file_entry to stat."""
    src_dir = datastore / source_key
    src_dir.mkdir(parents=True, exist_ok=True)
    nc = src_dir / name
    nc.write_bytes(b"placeholder-netcdf")
    return nc


def _write_aggregated_nc(project_dir: Path, source_key: str, name: str) -> Path:
    agg_dir = project_dir / "data" / "aggregated" / source_key
    agg_dir.mkdir(parents=True, exist_ok=True)
    nc = agg_dir / name
    nc.write_bytes(b"placeholder-agg-nc")
    return nc


def _write_target_nc(project_dir: Path, name: str) -> Path:
    targets = project_dir / "targets"
    targets.mkdir(parents=True, exist_ok=True)
    nc = targets / name
    nc.write_bytes(b"placeholder-target-nc")
    return nc


# ---------------------------------------------------------------------------
# Synthesis basics
# ---------------------------------------------------------------------------


def test_rebuild_empty_project_only_synthesizes_validate(project) -> None:
    """A project with no sources / no targets gets a single validate step."""
    summary = rebuild_lineage(project)
    assert summary["steps_added"] == 1
    assert summary["by_kind"] == {"validate": 1}
    manifest = json.loads(project.manifest_path.read_text())
    assert [s["kind"] for s in manifest["steps"]] == ["validate"]
    assert manifest["steps"][0]["params"]["synthesized"] is True
    assert manifest["steps"][0]["params"]["fabric_sha256"] == "abc123"


def test_rebuild_single_source_consolidate(project) -> None:
    """A source entry with consolidated_nc synthesizes one consolidate step."""
    nc = _write_consolidated_nc(project.datastore, "merra2", "merra2_consolidated.nc")
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["merra2"] = {
        "source_key": "merra2",
        "consolidated_nc": str(nc),
        "period": "2010/2020",
        "last_consolidated_utc": "2026-05-27T10:00:00+00:00",
    }
    project.manifest_path.write_text(json.dumps(manifest))

    summary = rebuild_lineage(project)

    manifest = json.loads(project.manifest_path.read_text())
    consolidate_steps = [s for s in manifest["steps"] if s["kind"] == "consolidate"]
    assert len(consolidate_steps) == 1
    step = consolidate_steps[0]
    assert step["source_key"] == "merra2"
    assert step["timestamp_utc"] == "2026-05-27T10:00:00+00:00"
    assert step["outputs"][0]["path"] == str(nc)
    assert step["params"]["period"] == "2010/2020"
    assert step["params"]["synthesized"] is True
    # sha256 default off — output entry has no sha256 key
    assert "sha256" not in step["outputs"][0]
    assert summary["by_kind"]["consolidate"] == 1


def test_rebuild_aggregate_step_from_output_files(project) -> None:
    """``output_files`` (project-workdir-relative) becomes one aggregate step."""
    nc = _write_aggregated_nc(project.workdir, "era5_land", "era5_2010.nc")
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["era5_land"] = {
        "source_key": "era5_land",
        "output_files": [str(nc.relative_to(project.workdir))],
        "period": "2010/2020",
        "fabric_sha256": "abc123",
        "timestamp": "2026-05-27T11:00:00+00:00",
    }
    project.manifest_path.write_text(json.dumps(manifest))

    rebuild_lineage(project)

    manifest = json.loads(project.manifest_path.read_text())
    agg_steps = [s for s in manifest["steps"] if s["kind"] == "aggregate"]
    assert len(agg_steps) == 1
    step = agg_steps[0]
    assert step["source_key"] == "era5_land"
    assert step["timestamp_utc"] == "2026-05-27T11:00:00+00:00"
    assert step["params"]["period"] == "2010/2020"
    assert step["params"]["fabric_sha256"] == "abc123"
    assert Path(step["outputs"][0]["path"]).name == "era5_2010.nc"


def test_rebuild_multi_year_consolidate(project) -> None:
    """``years[].consolidated_nc`` shape (snodas, margulis) flattens correctly."""
    nc_2010 = _write_consolidated_nc(project.datastore, "snodas", "snodas_2010.nc")
    nc_2011 = _write_consolidated_nc(project.datastore, "snodas", "snodas_2011.nc")
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["snodas"] = {
        "source_key": "snodas",
        "years": [
            {"year": 2010, "consolidated_nc": str(nc_2010)},
            {"year": 2011, "consolidated_nc": str(nc_2011)},
        ],
        "period": "2010/2011",
    }
    project.manifest_path.write_text(json.dumps(manifest))

    rebuild_lineage(project)

    manifest = json.loads(project.manifest_path.read_text())
    consolidate_steps = [s for s in manifest["steps"] if s["kind"] == "consolidate"]
    assert len(consolidate_steps) == 1
    out_paths = sorted(o["path"] for o in consolidate_steps[0]["outputs"])
    assert out_paths == sorted([str(nc_2010), str(nc_2011)])
    assert consolidate_steps[0]["params"]["years"] == [2010, 2011]


def test_rebuild_modis_consolidated_ncs_dict(project) -> None:
    """``consolidated_ncs`` (MODIS year-keyed dict) flattens correctly."""
    nc_a = _write_consolidated_nc(
        project.datastore, "mod16a2_v061", "mod16a2_2010_consolidated.nc"
    )
    nc_b = _write_consolidated_nc(
        project.datastore, "mod16a2_v061", "mod16a2_2011_consolidated.nc"
    )
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["mod16a2_v061"] = {
        "source_key": "mod16a2_v061",
        "consolidated_ncs": {"2010": str(nc_a), "2011": str(nc_b)},
        "period": "2010/2011",
    }
    project.manifest_path.write_text(json.dumps(manifest))

    rebuild_lineage(project)

    manifest = json.loads(project.manifest_path.read_text())
    step = next(s for s in manifest["steps"] if s["kind"] == "consolidate")
    out_paths = sorted(o["path"] for o in step["outputs"])
    assert out_paths == sorted([str(nc_a), str(nc_b)])


# ---------------------------------------------------------------------------
# Target NCs
# ---------------------------------------------------------------------------


def test_rebuild_target_and_nn_fill_steps(project) -> None:
    """``*_targets.nc`` becomes target; ``*_nn_filled.nc`` becomes nn_fill."""
    runoff = _write_target_nc(project.workdir, "runoff_targets.nc")
    runoff_nn = _write_target_nc(project.workdir, "runoff_targets_nn_filled.nc")
    _write_target_nc(project.workdir, "aet_targets.nc")

    rebuild_lineage(project)

    manifest = json.loads(project.manifest_path.read_text())
    targets = [s for s in manifest["steps"] if s["kind"] == "target"]
    nn_fills = [s for s in manifest["steps"] if s["kind"] == "nn_fill"]
    assert len(targets) == 2  # runoff + aet
    assert len(nn_fills) == 1
    target_paths = {Path(t["outputs"][0]["path"]).name for t in targets}
    assert target_paths == {"runoff_targets.nc", "aet_targets.nc"}
    assert Path(nn_fills[0]["outputs"][0]["path"]).name == "runoff_targets_nn_filled.nc"
    # Per-year intermediates dir is NOT walked.
    intermediates = project.targets_dir() / ".runoff_intermediates"
    intermediates.mkdir(parents=True, exist_ok=True)
    (intermediates / "runoff_targets_1980.nc").write_bytes(b"x")
    # Re-run -- intermediate must not appear as a step.
    rebuild_lineage(project)
    manifest = json.loads(project.manifest_path.read_text())
    all_output_paths = [
        o["path"] for s in manifest["steps"] for o in s.get("outputs", [])
    ]
    assert not any("runoff_targets_1980.nc" in p for p in all_output_paths), (
        "per-year intermediate must not be enumerated as a step output"
    )
    # runoff in scope check
    assert runoff.name == "runoff_targets.nc"
    assert runoff_nn.name == "runoff_targets_nn_filled.nc"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_rebuild_is_idempotent(project) -> None:
    """Re-running rebuild produces zero new steps."""
    nc = _write_consolidated_nc(project.datastore, "merra2", "merra2_consolidated.nc")
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["merra2"] = {
        "source_key": "merra2",
        "consolidated_nc": str(nc),
    }
    project.manifest_path.write_text(json.dumps(manifest))

    first = rebuild_lineage(project)
    second = rebuild_lineage(project)

    assert first["steps_added"] > 0
    assert second["steps_added"] == 0
    assert second["skipped_existing"] >= first["steps_added"]


def test_rebuild_skips_live_steps_from_post_prb_pipeline(project) -> None:
    """A live step from the post-PR-B pipeline wins the dedupe.

    Operators who ran agg AFTER PR-B will have one live aggregate step
    per source. Re-running rebuild_lineage must not synthesize a
    duplicate.
    """
    nc = _write_aggregated_nc(project.workdir, "era5_land", "era5_2010.nc")
    rel = str(nc.relative_to(project.workdir))
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["era5_land"] = {
        "source_key": "era5_land",
        "output_files": [rel],
        "period": "2010/2020",
        "timestamp": "2026-05-27T11:00:00+00:00",
    }
    # Pre-existing live step from the post-PR-B aggregate run.
    manifest["steps"].append(
        {
            "kind": "aggregate",
            "source_key": "era5_land",
            "timestamp_utc": "2026-05-27T11:00:00+00:00",
            "software_version": "0.1.0",
            "tool": "nhf-targets",
            "command": "agg era5-land",
            "inputs": [],
            "outputs": [{"path": str(nc), "size_bytes": 0, "mtime_utc": "x"}],
            "params": {},
        }
    )
    project.manifest_path.write_text(json.dumps(manifest))

    summary = rebuild_lineage(project)

    # The aggregate slot is taken by the live step; rebuild skips it.
    by_kind = summary["by_kind"]
    assert "aggregate" not in by_kind
    assert summary["skipped_existing"] >= 1


# ---------------------------------------------------------------------------
# Knobs: dry_run, compute_sha256
# ---------------------------------------------------------------------------


def test_rebuild_dry_run_does_not_write(project) -> None:
    """``dry_run=True`` computes the synthesis but never touches manifest.json."""
    nc = _write_consolidated_nc(project.datastore, "merra2", "merra2_consolidated.nc")
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["merra2"] = {
        "source_key": "merra2",
        "consolidated_nc": str(nc),
    }
    project.manifest_path.write_text(json.dumps(manifest))
    before = project.manifest_path.read_bytes()

    summary = rebuild_lineage(project, dry_run=True)

    assert summary["dry_run"] is True
    assert summary["steps_added"] > 0
    # File is byte-identical to the pre-rebuild state.
    assert project.manifest_path.read_bytes() == before


def test_rebuild_with_sha256_includes_digest(project) -> None:
    """``compute_sha256=True`` populates the ``sha256`` field on outputs."""
    nc = _write_consolidated_nc(project.datastore, "merra2", "merra2_consolidated.nc")
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["merra2"] = {
        "source_key": "merra2",
        "consolidated_nc": str(nc),
    }
    project.manifest_path.write_text(json.dumps(manifest))

    rebuild_lineage(project, compute_sha256=True)

    manifest = json.loads(project.manifest_path.read_text())
    step = next(s for s in manifest["steps"] if s["kind"] == "consolidate")
    assert "sha256" in step["outputs"][0]
    assert len(step["outputs"][0]["sha256"]) == 64


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_rebuild_missing_manifest_raises(tmp_path) -> None:
    """``rebuild_lineage`` requires the project's manifest.json to exist."""
    # Build a project shell without a manifest.
    datastore = tmp_path / "datastore"
    datastore.mkdir()
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": "/fake/fabric.gpkg", "id_col": "hru_id"},
                "datastore": str(datastore),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "abc"}))
    project = load_project(tmp_path)
    # Now remove the manifest.json validate would have written.
    project.manifest_path.unlink(missing_ok=True)

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        rebuild_lineage(project)


def test_rebuild_missing_output_files_are_skipped(project) -> None:
    """A manifest path pointing to a vanished NC does not crash the rebuild.

    Stale on-disk evidence (a fetched NC the operator deleted to
    reclaim disk) yields a consolidate step with no outputs, which
    PR-D's FGDC processstep handles fine as long as the params still
    record the period / bbox / license.
    """
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["merra2"] = {
        "source_key": "merra2",
        "consolidated_nc": str(project.datastore / "merra2" / "vanished.nc"),
        "period": "2010/2020",
    }
    project.manifest_path.write_text(json.dumps(manifest))

    rebuild_lineage(project)

    manifest = json.loads(project.manifest_path.read_text())
    step = next(s for s in manifest["steps"] if s["kind"] == "consolidate")
    assert step["outputs"] == []
    assert step["params"]["period"] == "2010/2020"
