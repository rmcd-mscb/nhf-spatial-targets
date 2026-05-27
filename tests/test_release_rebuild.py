"""Tests for ``nhf_spatial_targets.release.rebuild``.

Bootstraps lineage steps for legacy projects with empty ``steps[]``.
The invariants we lock here:

- Each manifest source entry produces the right kind+source_key step
  shapes from on-disk evidence (consolidate, aggregate).
- Target NCs in ``<project>/targets/`` produce target / nn_fill steps.
- Idempotent: re-running rebuild_lineage doesn't duplicate steps even
  when synthesized variants would conflict with live steps from the
  live pipeline.
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


def test_rebuild_skips_live_steps_from_live_pipeline(project) -> None:
    """A live step from the live pipeline wins the dedupe.

    Operators who ran agg under the live (step-emitting) pipeline have
    one live aggregate step per source. Re-running rebuild_lineage must
    not synthesize a duplicate.
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
    # Pre-existing live step from a prior aggregate run.
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


def test_rebuild_missing_output_files_are_skipped(project, caplog) -> None:
    """A manifest path pointing to a vanished NC does not crash the rebuild.

    Stale on-disk evidence (a fetched NC the operator deleted to
    reclaim disk) yields a consolidate step with no outputs, a WARNING
    log line, and the vanished path recorded in
    ``params["missing_outputs"]`` so downstream consumers can surface
    the gap.
    """
    import logging

    manifest = json.loads(project.manifest_path.read_text())
    vanished = str(project.datastore / "merra2" / "vanished.nc")
    manifest["sources"]["merra2"] = {
        "source_key": "merra2",
        "consolidated_nc": vanished,
        "period": "2010/2020",
    }
    project.manifest_path.write_text(json.dumps(manifest))

    with caplog.at_level(logging.WARNING, logger="nhf_spatial_targets.release.rebuild"):
        rebuild_lineage(project)

    manifest = json.loads(project.manifest_path.read_text())
    step = next(s for s in manifest["steps"] if s["kind"] == "consolidate")
    assert step["outputs"] == []
    assert step["params"]["period"] == "2010/2020"
    assert step["params"]["missing_outputs"] == [vanished]
    assert any(
        "not on disk" in rec.message and "vanished.nc" in rec.message
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# Additional source-entry shapes
# ---------------------------------------------------------------------------


def test_rebuild_era5_land_files_top_level_shape(project) -> None:
    """ERA5-Land emits ``files: [{year, daily_path, monthly_path}]`` at top level.

    Pre-fixup, ``_consolidate_outputs`` only walked ``years[]`` /
    ``water_years[]`` / ``regions{}`` -- ERA5-Land projects synthesized
    empty-output consolidate steps. The fix extends the year-record key
    list to also cover ``daily_path`` and walks the top-level ``files``
    list.
    """
    daily_2010 = _write_consolidated_nc(
        project.datastore, "era5_land/daily", "era5_land_daily_2010.nc"
    )
    daily_2011 = _write_consolidated_nc(
        project.datastore, "era5_land/daily", "era5_land_daily_2011.nc"
    )
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["era5_land"] = {
        "source_key": "era5_land",
        "period": "2010/2011",
        "files": [
            {"year": 2010, "daily_path": str(daily_2010), "monthly_path": "/m/2010.nc"},
            {"year": 2011, "daily_path": str(daily_2011), "monthly_path": "/m/2011.nc"},
        ],
    }
    project.manifest_path.write_text(json.dumps(manifest))

    rebuild_lineage(project)

    manifest = json.loads(project.manifest_path.read_text())
    step = next(s for s in manifest["steps"] if s["kind"] == "consolidate")
    out_paths = sorted(o["path"] for o in step["outputs"])
    assert out_paths == sorted([str(daily_2010), str(daily_2011)])
    assert step["params"]["n_files"] == 2


def test_rebuild_year_record_daily_path_shape(project) -> None:
    """snodas / margulis / ua_swe year records use ``daily_path``, not ``consolidated_nc``."""
    daily_2010 = _write_consolidated_nc(project.datastore, "snodas", "snodas_2010.nc")
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["snodas"] = {
        "source_key": "snodas",
        "period": "2010/2010",
        "years": [{"year": 2010, "daily_path": str(daily_2010)}],
    }
    project.manifest_path.write_text(json.dumps(manifest))

    rebuild_lineage(project)

    manifest = json.loads(project.manifest_path.read_text())
    step = next(s for s in manifest["steps"] if s["kind"] == "consolidate")
    assert [o["path"] for o in step["outputs"]] == [str(daily_2010)]
    assert step["params"]["years"] == [2010]


def test_rebuild_file_path_shape(project) -> None:
    """mwbm_climgrid / reitz2017 / pangaea use ``file: {path}`` single-NC registration."""
    nc = _write_consolidated_nc(project.datastore, "mwbm_climgrid", "ClimGrid_WBM.nc")
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["mwbm_climgrid"] = {
        "source_key": "mwbm_climgrid",
        "period": "1900/2020",
        "file": {"path": str(nc), "size_bytes": 100, "sha256": "x" * 64},
    }
    project.manifest_path.write_text(json.dumps(manifest))

    rebuild_lineage(project)

    manifest = json.loads(project.manifest_path.read_text())
    step = next(s for s in manifest["steps"] if s["kind"] == "consolidate")
    assert [o["path"] for o in step["outputs"]] == [str(nc)]


def test_rebuild_file_path_dedups_against_consolidated_nc(project) -> None:
    """When both ``consolidated_nc`` and ``file.path`` point at the same NC,
    the synthesized step has exactly one output entry.
    """
    nc = _write_consolidated_nc(project.datastore, "merra2", "merra2_consolidated.nc")
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["merra2"] = {
        "source_key": "merra2",
        "consolidated_nc": str(nc),
        "file": {"path": str(nc)},
    }
    project.manifest_path.write_text(json.dumps(manifest))

    rebuild_lineage(project)

    manifest = json.loads(project.manifest_path.read_text())
    step = next(s for s in manifest["steps"] if s["kind"] == "consolidate")
    assert len(step["outputs"]) == 1


def test_rebuild_daymet_zarr_dir_skipped(project) -> None:
    """Daymet ``regions[r].path`` points at a zarr directory, not a file.

    ``output_file_entry`` on a directory would crash under
    ``compute_sha256=True`` (``IsADirectoryError`` on ``open()``).
    ``_consolidate_outputs`` must skip dir-valued paths so the step
    lands with empty outputs but the regions param is preserved -- the
    live daymet step has the same shape (zarr stores aren't republished).
    """
    zarr_dir = project.datastore / "daymet" / "na.zarr"
    zarr_dir.mkdir(parents=True)
    (zarr_dir / "zmetadata").write_bytes(b"{}")  # zarr metadata file
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["daymet"] = {
        "source_key": "daymet",
        "period": "1980/2024",
        "regions": {"na": {"path": str(zarr_dir), "zmetadata_sha256": "abc"}},
    }
    project.manifest_path.write_text(json.dumps(manifest))

    # Even with compute_sha256=True, the dir must not crash the rebuild.
    rebuild_lineage(project, compute_sha256=True)

    manifest = json.loads(project.manifest_path.read_text())
    step = next(s for s in manifest["steps"] if s["kind"] == "consolidate")
    assert step["outputs"] == []
    assert step["params"]["regions"] == ["na"]


# ---------------------------------------------------------------------------
# Failure-mode contracts
# ---------------------------------------------------------------------------


def test_rebuild_missing_fabric_json_raises(project) -> None:
    """``rebuild_lineage`` raises FileNotFoundError when fabric.json is missing.

    Asymmetric-with-manifest-only behavior would silently produce
    fabric-anchorless lineage; rebuild matches its manifest-json
    contract (raise loud) so the operator sees the gap immediately.
    """
    (project.workdir / "fabric.json").unlink()
    with pytest.raises(FileNotFoundError, match="fabric.json"):
        rebuild_lineage(project)


def test_rebuild_skips_live_validate_step(project) -> None:
    """A live ``kind=validate`` step from the live pipeline must dedup.

    The synthesizer's outputs must exactly match the live writer's
    outputs (fabric.json + config.effective.yml; manifest.json is
    deliberately excluded). If the synthesizer accidentally listed
    manifest.json, the sorted-path tuples would differ and the dedup
    would miss, producing a duplicate validate step on every rebuild.
    """
    # Materialize the on-disk artifacts the live writer records.
    fabric_json = project.workdir / "fabric.json"
    effective_config = project.workdir / "config.effective.yml"
    effective_config.write_bytes(b"effective: yaml")
    live_outputs = sorted([str(fabric_json), str(effective_config)])
    manifest = json.loads(project.manifest_path.read_text())
    manifest["steps"].append(
        {
            "kind": "validate",
            "source_key": None,
            "timestamp_utc": "2026-05-27T10:00:00+00:00",
            "software_version": "0.1.0",
            "tool": "nhf-targets",
            "command": "validate",
            "inputs": [],
            "outputs": [
                {"path": p, "size_bytes": 0, "mtime_utc": "x"} for p in live_outputs
            ],
            "params": {"fabric_sha256": "abc123"},
        }
    )
    project.manifest_path.write_text(json.dumps(manifest))

    summary = rebuild_lineage(project)

    # The synthesizer's validate step matches the live one, so it dedups.
    assert "validate" not in summary["by_kind"]
    assert summary["skipped_existing"] >= 1
    # Only ONE validate step survives (the live one).
    manifest = json.loads(project.manifest_path.read_text())
    validate_steps = [s for s in manifest["steps"] if s["kind"] == "validate"]
    assert len(validate_steps) == 1


def test_rebuild_validate_step_excludes_manifest_json(project) -> None:
    """The synthesized validate step must NOT list ``manifest.json`` in outputs.

    Recording the sha256 of the file the step lives in is the
    self-hash paradox: the recorded hash would describe the pre-append
    state, not the on-disk file as it now exists. The synthesizer must
    match the live ``validate`` writer's behavior here.
    """
    rebuild_lineage(project, compute_sha256=True)
    manifest = json.loads(project.manifest_path.read_text())
    step = next(s for s in manifest["steps"] if s["kind"] == "validate")
    out_paths = {o["path"] for o in step["outputs"]}
    assert str(project.manifest_path) not in out_paths


def test_rebuild_default_stamps_sha256_skipped(project) -> None:
    """With ``compute_sha256=False`` (default), every output-bearing
    synthesized step is stamped with ``params.sha256_skipped=True``.

    The downstream ``release publish`` stage gate reads this flag to
    refuse to publish a release whose outputs lack integrity hashes.
    """
    nc = _write_consolidated_nc(project.datastore, "merra2", "merra2_consolidated.nc")
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["merra2"] = {
        "source_key": "merra2",
        "consolidated_nc": str(nc),
    }
    project.manifest_path.write_text(json.dumps(manifest))

    summary = rebuild_lineage(project)

    assert summary["compute_sha256"] is False
    manifest = json.loads(project.manifest_path.read_text())
    for step in manifest["steps"]:
        # The validate step has outputs so it's stamped too.
        assert step["params"].get("sha256_skipped") is True


def test_rebuild_with_sha256_does_not_stamp_skipped(project) -> None:
    """``compute_sha256=True`` must NOT stamp ``sha256_skipped=True`` --
    that flag is the gate the downstream ``release publish`` stage
    reads to refuse unhashed outputs.
    """
    nc = _write_consolidated_nc(project.datastore, "merra2", "merra2_consolidated.nc")
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["merra2"] = {
        "source_key": "merra2",
        "consolidated_nc": str(nc),
    }
    project.manifest_path.write_text(json.dumps(manifest))

    summary = rebuild_lineage(project, compute_sha256=True)

    assert summary["compute_sha256"] is True
    manifest = json.loads(project.manifest_path.read_text())
    for step in manifest["steps"]:
        assert "sha256_skipped" not in step["params"]


def test_rebuild_aborted_aggregate_emits_marker_step(project) -> None:
    """``output_files: []`` (aborted prior aggregate) emits a step with
    ``params.status='aborted'`` rather than silently eliding the evidence.
    """
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["era5_land"] = {
        "source_key": "era5_land",
        "output_files": [],
        "period": "2010/2020",
        "fabric_sha256": "abc123",
    }
    project.manifest_path.write_text(json.dumps(manifest))

    rebuild_lineage(project)

    manifest = json.loads(project.manifest_path.read_text())
    agg_steps = [s for s in manifest["steps"] if s["kind"] == "aggregate"]
    assert len(agg_steps) == 1
    assert agg_steps[0]["params"]["status"] == "aborted"


def test_rebuild_dry_run_reports_full_summary(project) -> None:
    """``dry_run=True`` reports the would-be summary, not just steps_added.

    Operators inspecting what would change need the by_kind breakdown
    too, not just the count.
    """
    nc = _write_consolidated_nc(project.datastore, "merra2", "merra2_consolidated.nc")
    manifest = json.loads(project.manifest_path.read_text())
    manifest["sources"]["merra2"] = {
        "source_key": "merra2",
        "consolidated_nc": str(nc),
    }
    project.manifest_path.write_text(json.dumps(manifest))

    summary = rebuild_lineage(project, dry_run=True)

    assert summary["dry_run"] is True
    assert summary["steps_added"] >= 1
    assert summary["by_kind"]  # non-empty
    assert summary["compute_sha256"] is False
    # On a fresh project nothing should be skipped.
    assert summary["skipped_existing"] == 0
