"""Tests for nhf_spatial_targets.release.checksums."""

from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from nhf_spatial_targets.release.checksums import (
    CHECKSUMS_CSV,
    SHA256SUMS,
    ChecksumMismatch,
    compute_checksums,
    verify_csv,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_stage(tmp_path) -> Path:
    stage = tmp_path / "stage"
    (stage / "aggregated" / "era5_land").mkdir(parents=True)
    (stage / "fabric.gpkg").write_bytes(b"GPKG")
    (stage / "manifest.json").write_text('{"a": 1}')
    (stage / "aggregated" / "era5_land" / "y1980.nc").write_bytes(b"nineteen-eighty")
    (stage / "targets").mkdir()
    (stage / "targets" / "runoff.nc").write_bytes(b"runoff-bytes")
    return stage


def test_compute_returns_sorted_entries(tmp_path):
    stage = _make_stage(tmp_path)
    entries = compute_checksums(stage)
    paths = [e.path for e in entries]
    assert paths == sorted(paths)
    assert "fabric.gpkg" in paths
    assert "aggregated/era5_land/y1980.nc" in paths


def test_checksum_files_excluded_from_themselves(tmp_path):
    stage = _make_stage(tmp_path)
    entries = compute_checksums(stage)
    paths = {e.path for e in entries}
    assert CHECKSUMS_CSV not in paths
    assert SHA256SUMS not in paths


def test_sha256sums_format_matches_expected(tmp_path):
    stage = _make_stage(tmp_path)
    entries = compute_checksums(stage)
    expected = "".join(f"{e.sha256}  {e.path}\n" for e in entries)
    assert (stage / SHA256SUMS).read_text() == expected
    # Spot-check one digest against an independent hash.
    fabric_entry = next(e for e in entries if e.path == "fabric.gpkg")
    assert fabric_entry.sha256 == _sha256(b"GPKG")


@pytest.mark.skipif(
    shutil.which("sha256sum") is None, reason="GNU sha256sum not on PATH"
)
def test_sha256sums_byte_matches_gnu_sha256sum(tmp_path):
    stage = _make_stage(tmp_path)
    compute_checksums(stage)
    # `sha256sum -c` must accept our file unchanged.
    result = subprocess.run(
        ["sha256sum", "-c", SHA256SUMS],
        cwd=stage,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # And a single-file line must be byte-identical to GNU output.
    gnu = subprocess.run(
        ["sha256sum", "fabric.gpkg"],
        cwd=stage,
        capture_output=True,
        text=True,
    ).stdout
    assert gnu in (stage / SHA256SUMS).read_text()


def test_checksums_csv_columns(tmp_path):
    stage = _make_stage(tmp_path)
    compute_checksums(stage)
    with (stage / CHECKSUMS_CSV).open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert {"path", "sha256", "size_bytes", "mtime_utc"} == set(rows[0])
    fabric_row = next(r for r in rows if r["path"] == "fabric.gpkg")
    assert fabric_row["sha256"] == _sha256(b"GPKG")
    assert fabric_row["size_bytes"] == "4"


def test_compute_on_missing_dir_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        compute_checksums(tmp_path / "nope")


def test_compute_on_empty_stage_writes_header_only_and_empty_sums(tmp_path):
    """Umbrella / metadata-only stages are empty; the manifests must still
    be well-formed: a header-only CSV and a zero-length SHA256SUMS."""
    stage = tmp_path / "empty"
    stage.mkdir()
    entries = compute_checksums(stage)
    assert entries == []
    assert (stage / SHA256SUMS).read_text() == ""
    header = (stage / CHECKSUMS_CSV).read_text().splitlines()[0]
    assert header == "path,sha256,size_bytes,mtime_utc"
    verify_csv(stage)  # an empty stage verifies clean


def test_dangling_symlink_is_a_hard_error_not_a_silent_skip(tmp_path):
    """A staged symlink whose target vanished must fail loudly, not be
    silently dropped from the integrity manifest."""
    stage = tmp_path / "stage"
    stage.mkdir()
    target = tmp_path / "gone.nc"
    target.write_bytes(b"data")
    (stage / "linked.nc").symlink_to(target)
    target.unlink()  # break the link
    with pytest.raises(FileNotFoundError, match="missing target.*linked.nc"):
        compute_checksums(stage)


def test_verify_passes_after_compute(tmp_path):
    stage = _make_stage(tmp_path)
    compute_checksums(stage)
    verify_csv(stage)  # no raise


def test_verify_detects_sha256_drift_same_size(tmp_path):
    stage = _make_stage(tmp_path)
    compute_checksums(stage)
    # Same length (4 bytes), different content -> size matches, so the
    # hash check is what catches the drift.
    (stage / "fabric.gpkg").write_bytes(b"XXXX")
    with pytest.raises(ChecksumMismatch, match="sha256 drift.*fabric.gpkg"):
        verify_csv(stage)


def test_verify_detects_size_drift(tmp_path):
    stage = _make_stage(tmp_path)
    compute_checksums(stage)
    # Different length -> size is the cheap pre-filter that catches it.
    (stage / "fabric.gpkg").write_bytes(b"TAMPERED")
    with pytest.raises(ChecksumMismatch, match="size drift.*fabric.gpkg"):
        verify_csv(stage)


def test_verify_detects_missing_file(tmp_path):
    stage = _make_stage(tmp_path)
    compute_checksums(stage)
    (stage / "targets" / "runoff.nc").unlink()
    with pytest.raises(ChecksumMismatch, match="missing.*runoff.nc"):
        verify_csv(stage)


def test_verify_detects_unlisted_extra_file(tmp_path):
    stage = _make_stage(tmp_path)
    compute_checksums(stage)
    (stage / "targets" / "sneaky.nc").write_bytes(b"new")
    with pytest.raises(ChecksumMismatch, match="unlisted.*sneaky.nc"):
        verify_csv(stage)


def test_verify_without_csv_raises(tmp_path):
    stage = _make_stage(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_csv(stage)
