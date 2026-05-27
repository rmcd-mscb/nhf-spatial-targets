"""Tests for ``nhf_spatial_targets.release.lineage``.

The lineage module is the foundation under FGDC / ISO / README
rendering, so the invariants we lock here are:

- ``append_step`` records the canonical step-record shape (no field
  renames downstream),
- output file entries carry sha256 + size + mtime + path, input
  entries carry the same minus sha256 (cost asymmetry: aggregate
  reads dozens of multi-GB inputs and rehashing dominates),
- concurrent appenders never lose a step (the SLURM array hazard
  the existing aggregator's ``update_manifest`` already addresses),
- ``merge_source_and_append_step`` is the single canonical
  read-modify-write that fetch modules can adopt without recreating
  their bespoke flock code.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from nhf_spatial_targets.release.lineage import (
    STEP_KINDS,
    append_step,
    input_file_entry,
    merge_source_and_append_step,
    output_file_entry,
    sha256_file,
)


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    """Fresh ``manifest.json`` path inside a tmp project."""
    return tmp_path / "manifest.json"


@pytest.fixture
def sample_output(tmp_path: Path) -> Path:
    """A small file standing in for an output NetCDF."""
    p = tmp_path / "out.nc"
    p.write_bytes(b"netcdf-payload-bytes-for-test")
    return p


# ---------------------------------------------------------------------------
# File-entry shape
# ---------------------------------------------------------------------------


def test_output_file_entry_includes_sha256_and_metadata(sample_output: Path) -> None:
    """Outputs include sha256, size, mtime, path -- the canonical shape
    the FGDC ``processstep`` consumer reads.
    """
    entry = output_file_entry(sample_output)
    assert set(entry) == {"path", "sha256", "size_bytes", "mtime_utc"}
    assert entry["path"] == str(sample_output)
    assert entry["size_bytes"] == len(b"netcdf-payload-bytes-for-test")
    assert entry["sha256"] == sha256_file(sample_output)
    # ISO-8601 with timezone offset
    assert "T" in entry["mtime_utc"]
    assert entry["mtime_utc"].endswith("+00:00")


def test_output_file_entry_skip_sha256(sample_output: Path) -> None:
    """``compute_sha256=False`` is the only knob for skipping the hash."""
    entry = output_file_entry(sample_output, compute_sha256=False)
    assert "sha256" not in entry
    assert set(entry) == {"path", "size_bytes", "mtime_utc"}


def test_input_file_entry_has_no_sha256(sample_output: Path) -> None:
    """Inputs omit sha256 by design: aggregate over 30 sources reads dozens
    of consolidated NCs, and full-file hashing would dominate runtime.
    """
    entry = input_file_entry(sample_output)
    assert "sha256" not in entry
    assert set(entry) == {"path", "size_bytes", "mtime_utc"}


def test_sha256_file_matches_hashlib(sample_output: Path) -> None:
    """Streamed sha256 matches a one-shot hashlib digest."""
    import hashlib

    one_shot = hashlib.sha256(sample_output.read_bytes()).hexdigest()
    assert sha256_file(sample_output) == one_shot


# ---------------------------------------------------------------------------
# append_step: schema + lifecycle
# ---------------------------------------------------------------------------


def test_append_step_creates_manifest_with_skeleton(manifest_path: Path) -> None:
    """First append into a fresh project creates manifest.json + skeleton."""
    assert not manifest_path.exists()
    append_step(
        manifest_path,
        kind="validate",
        source_key=None,
        timestamp_utc="2026-05-27T12:00:00+00:00",
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest == {
        "sources": {},
        "steps": [
            {
                "kind": "validate",
                "source_key": None,
                "timestamp_utc": "2026-05-27T12:00:00+00:00",
                "software_version": manifest["steps"][0]["software_version"],
                "tool": "nhf-targets",
                "command": None,
                "inputs": [],
                "outputs": [],
                "params": {},
            }
        ],
    }


def test_append_step_preserves_existing_sources(manifest_path: Path) -> None:
    """Appending a step must NEVER clobber existing ``sources`` keys
    (regression for #97 -- the validate-overwrites-fetch-provenance bug).
    """
    manifest_path.write_text(
        json.dumps(
            {
                "sources": {"era5_land": {"period": "1980/2024"}},
                "steps": [],
            }
        )
    )
    append_step(manifest_path, kind="validate", source_key=None)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["sources"] == {"era5_land": {"period": "1980/2024"}}
    assert len(manifest["steps"]) == 1


def test_append_step_appends_in_order(manifest_path: Path) -> None:
    """Sequential appends preserve insertion order."""
    for ts in ("2026-05-27T01:00:00+00:00", "2026-05-27T02:00:00+00:00"):
        append_step(
            manifest_path,
            kind="aggregate",
            source_key="era5_land",
            timestamp_utc=ts,
        )
    manifest = json.loads(manifest_path.read_text())
    assert [s["timestamp_utc"] for s in manifest["steps"]] == [
        "2026-05-27T01:00:00+00:00",
        "2026-05-27T02:00:00+00:00",
    ]


def test_append_step_records_inputs_and_outputs(
    manifest_path: Path, sample_output: Path
) -> None:
    """Step records carry the input/output file lists verbatim."""
    append_step(
        manifest_path,
        kind="target",
        source_key=None,
        inputs=[input_file_entry(sample_output)],
        outputs=[output_file_entry(sample_output)],
        params={"sort_dim": "nhm_id"},
        command="run-runoff",
    )
    step = json.loads(manifest_path.read_text())["steps"][0]
    assert step["inputs"][0]["path"] == str(sample_output)
    assert "sha256" not in step["inputs"][0]
    assert step["outputs"][0]["sha256"] == sha256_file(sample_output)
    assert step["params"] == {"sort_dim": "nhm_id"}
    assert step["command"] == "run-runoff"


def test_append_step_rejects_unknown_kind(manifest_path: Path) -> None:
    """Unknown kinds fail loud so the FGDC pattern-match stays closed."""
    with pytest.raises(ValueError, match="Unknown step kind"):
        append_step(manifest_path, kind="not-a-real-kind", source_key=None)


def test_step_kinds_set_matches_plan() -> None:
    """The closed set of step kinds matches the plan's Lineage capture
    table; downstream FGDC generation depends on this contract.
    """
    assert STEP_KINDS == frozenset(
        {"fetch", "consolidate", "aggregate", "target", "nn_fill", "validate"}
    )


def test_append_step_raises_on_corrupt_manifest(manifest_path: Path) -> None:
    """A corrupt manifest is a loud failure mode -- matches the
    aggregator's existing behavior so an operator can't silently
    blow away their provenance.
    """
    manifest_path.write_text("{not valid json")
    with pytest.raises(ValueError, match="corrupt"):
        append_step(manifest_path, kind="validate", source_key=None)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_append_step_concurrent_writers_lose_no_steps(manifest_path: Path) -> None:
    """4-thread concurrent appends produce exactly 4 steps, none lost.

    The flock semantics under threads-of-one-process aren't a perfect
    proxy for SLURM array workers (separate OS processes), but the
    read-modify-write window is the same, and a faulty implementation
    that just did ``manifest["steps"].append(); write()`` without
    serializing would lose appends here too.
    """
    n_threads = 4

    def _worker(worker_id: int) -> None:
        append_step(
            manifest_path,
            kind="aggregate",
            source_key=f"src_{worker_id}",
            params={"worker_id": worker_id},
        )

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    manifest = json.loads(manifest_path.read_text())
    seen_workers = {s["source_key"] for s in manifest["steps"]}
    assert seen_workers == {f"src_{i}" for i in range(n_threads)}
    assert len(manifest["steps"]) == n_threads


# ---------------------------------------------------------------------------
# merge_source_and_append_step
# ---------------------------------------------------------------------------


def test_merge_source_and_append_step_updates_both(manifest_path: Path) -> None:
    """Single call writes both the source entry and the step atomically."""
    merge_source_and_append_step(
        manifest_path,
        "merra2",
        source_updates={
            "source_key": "merra2",
            "period": "2010/2020",
            "consolidated_nc": "/data/merra2/merra2_consolidated.nc",
        },
        kind="consolidate",
        outputs=[],
        params={"n_files": 132},
        timestamp_utc="2026-05-27T12:00:00+00:00",
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["sources"]["merra2"]["period"] == "2010/2020"
    assert manifest["sources"]["merra2"]["consolidated_nc"].endswith(
        "merra2_consolidated.nc"
    )
    assert len(manifest["steps"]) == 1
    step = manifest["steps"][0]
    assert step["kind"] == "consolidate"
    assert step["source_key"] == "merra2"
    assert step["params"] == {"n_files": 132}


def test_merge_source_preserves_existing_source_keys(manifest_path: Path) -> None:
    """``dict.update`` semantics: existing keys survive unless overridden.

    Mirrors the fetch-module read-merge-write contract from #97 / #119.
    """
    manifest_path.write_text(
        json.dumps(
            {
                "sources": {
                    "snodas": {
                        "source_key": "snodas",
                        "years": [2003, 2004, 2005],
                        "license": "Public domain",
                    }
                },
                "steps": [],
            }
        )
    )
    merge_source_and_append_step(
        manifest_path,
        "snodas",
        source_updates={"last_consolidated_utc": "2026-05-27T12:00:00+00:00"},
        kind="consolidate",
    )
    snodas = json.loads(manifest_path.read_text())["sources"]["snodas"]
    assert snodas["years"] == [2003, 2004, 2005]
    assert snodas["license"] == "Public domain"
    assert snodas["last_consolidated_utc"] == "2026-05-27T12:00:00+00:00"


def test_merge_source_rejects_top_level_steps_collision(manifest_path: Path) -> None:
    """``steps`` is top-level, not nested under a source; misuse must
    fail loud so an accidentally nested step list is never created.
    """
    with pytest.raises(ValueError, match="must not include 'steps'"):
        merge_source_and_append_step(
            manifest_path,
            "merra2",
            source_updates={"steps": []},
            kind="consolidate",
        )


def test_merge_source_strips_legacy_nested_steps(manifest_path: Path, caplog) -> None:
    """A legacy manifest that nested ``steps`` under a source entry has
    those records popped (with the dropped payload logged) and the new
    step lands at the top-level ``manifest["steps"]`` array.
    """
    legacy = [{"step": "legacy-1", "utc": "2020-01-01"}]
    manifest_path.write_text(
        json.dumps(
            {
                "sources": {
                    "merra2": {"source_key": "merra2", "steps": legacy},
                },
                "steps": [],
            }
        )
    )

    with caplog.at_level("WARNING"):
        merge_source_and_append_step(
            manifest_path,
            "merra2",
            source_updates={"period": "2010/2020"},
            kind="consolidate",
        )

    after = json.loads(manifest_path.read_text())
    # Nested array is gone, top-level got the new record only.
    assert "steps" not in after["sources"]["merra2"]
    assert len(after["steps"]) == 1
    assert after["steps"][0]["kind"] == "consolidate"
    assert after["sources"]["merra2"]["period"] == "2010/2020"

    # Exactly one WARNING fired and the dropped payload is in the log line
    # so an operator can recover it.
    drop_records = [
        r
        for r in caplog.records
        if "dropped" in r.getMessage() and "legacy" in r.getMessage()
    ]
    assert len(drop_records) == 1
    assert drop_records[0].levelname == "WARNING"
    assert "legacy-1" in drop_records[0].getMessage()


def test_merge_source_returns_appended_record(manifest_path: Path) -> None:
    """Returns the appended step record so callers can log + assert."""
    record = merge_source_and_append_step(
        manifest_path,
        "merra2",
        source_updates={"period": "2010/2020"},
        kind="consolidate",
    )
    assert record["kind"] == "consolidate"
    assert record["source_key"] == "merra2"
