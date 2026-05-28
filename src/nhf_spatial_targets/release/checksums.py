"""Per-file checksums for a staged release item.

Two integrity artifacts are written next to a finished stage tree:

- ``checksums.csv`` -- columns ``path,sha256,size_bytes,mtime_utc``; the
  rich record (sizes + mtimes) for status reporting and publish-time drift
  detection.
- ``SHA256SUMS`` -- GNU ``sha256sum`` format (``<hex>␠␠<path>``) so an
  operator can run ``sha256sum -c SHA256SUMS`` from inside the stage
  directory to verify a download.

Both list every staged file except the two integrity files themselves,
with paths relative to the stage directory and rows sorted by path for
reproducible output. Symlinked entries are hashed by *content* (the link
target), which is what a ScienceBase consumer downloads.

:func:`verify_csv` re-hashes the recorded files and raises
:class:`ChecksumMismatch` on any content drift, size change, missing file,
or unlisted extra file. Verification is content-based: mtime is recorded
but not compared (touching a file must not fail verification).
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

from nhf_spatial_targets.release._models import CHECKSUM_FILES, FileEntry
from nhf_spatial_targets.release.lineage import sha256_file

CHECKSUMS_CSV = "checksums.csv"
SHA256SUMS = "SHA256SUMS"
_CSV_FIELDS = ("path", "sha256", "size_bytes", "mtime_utc")


class ChecksumMismatch(Exception):
    """Raised when staged files no longer match their recorded checksums."""


def _iter_staged_files(stage_dir: Path) -> list[Path]:
    """Return stage-relative paths of every file to checksum, sorted.

    Skips the top-level integrity files (a manifest must not list itself);
    a same-named file nested in a subdirectory is still included.

    A *dangling* symlink (the default stager links data files, and the
    datastore can live on a separately-mounted drive) is a hard error, not
    a skip: ``is_file()`` is False for a broken link, so a silent skip would
    drop the file from the integrity manifest and let ``verify_csv`` /
    ``sha256sum -c`` pass over an incomplete payload.
    """
    skip = set(CHECKSUM_FILES)
    rels: list[Path] = []
    for path in stage_dir.rglob("*"):
        if path.is_symlink() and not path.exists():
            raise FileNotFoundError(
                f"staged entry points at a missing target: {path} -> "
                f"{os.readlink(path)}"
            )
        if not path.is_file():
            continue
        rel = path.relative_to(stage_dir)
        if rel.as_posix() in skip:
            continue
        rels.append(rel)
    return sorted(rels, key=lambda r: r.as_posix())


def _file_entry(stage_dir: Path, rel: Path) -> FileEntry:
    abs_path = stage_dir / rel
    st = abs_path.stat()
    return FileEntry(
        path=rel.as_posix(),
        sha256=sha256_file(abs_path),
        size=int(st.st_size),
        mtime=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    )


def _write_checksums_csv(path: Path, entries: list[FileEntry]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_FIELDS)
        for e in entries:
            writer.writerow([e.path, e.sha256, e.size, e.mtime])


def _write_sha256sums(path: Path, entries: list[FileEntry]) -> None:
    # GNU sha256sum text-mode line: digest, two spaces, path. Matches the
    # output of `sha256sum <file>` so `sha256sum -c` accepts the file.
    lines = [f"{e.sha256}  {e.path}\n" for e in entries]
    path.write_text("".join(lines))


def compute_checksums(stage_dir: Path) -> list[FileEntry]:
    """Hash every staged file and write ``checksums.csv`` + ``SHA256SUMS``.

    Parameters
    ----------
    stage_dir
        A finished stage directory (e.g. ``build/fabric/<label>``).

    Returns
    -------
    list[FileEntry]
        One entry per staged file, sorted by relative path. The same list
        backs both written artifacts.
    """
    if not stage_dir.is_dir():
        raise NotADirectoryError(f"stage directory not found: {stage_dir}")
    entries = [_file_entry(stage_dir, rel) for rel in _iter_staged_files(stage_dir)]
    _write_checksums_csv(stage_dir / CHECKSUMS_CSV, entries)
    _write_sha256sums(stage_dir / SHA256SUMS, entries)
    return entries


def _read_checksums_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def verify_csv(stage_dir: Path) -> None:
    """Verify staged files against ``checksums.csv``; raise on drift.

    Re-hashes each recorded file and compares sha256 + size. Also flags
    recorded files that have gone missing and staged files absent from the
    record. mtime is intentionally not compared.

    Raises
    ------
    FileNotFoundError
        ``checksums.csv`` does not exist (run :func:`compute_checksums`).
    ChecksumMismatch
        Any hash drift, size change, missing recorded file, or unlisted
        extra file. The message enumerates every discrepancy found.
    """
    csv_path = stage_dir / CHECKSUMS_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found; run compute_checksums(stage_dir) first."
        )

    recorded = _read_checksums_csv(csv_path)
    problems: list[str] = []
    recorded_paths: set[str] = set()

    for row in recorded:
        rel = row["path"]
        recorded_paths.add(rel)
        abs_path = stage_dir / rel
        if not abs_path.is_file():
            problems.append(f"missing: {rel}")
            continue
        # Size is the cheap pre-filter: a mismatch already proves drift, so
        # skip hashing a (potentially multi-GB) file whose size changed, and
        # report one unambiguous problem per file rather than a redundant
        # size+sha256 pair.
        actual_size = abs_path.stat().st_size
        if str(actual_size) != row["size_bytes"]:
            problems.append(
                f"size drift: {rel} (recorded {row['size_bytes']} bytes, "
                f"now {actual_size})"
            )
            continue
        actual_sha = sha256_file(abs_path)
        if actual_sha != row["sha256"]:
            problems.append(
                f"sha256 drift: {rel} (recorded {row['sha256'][:12]}…, "
                f"now {actual_sha[:12]}…)"
            )

    on_disk = {rel.as_posix() for rel in _iter_staged_files(stage_dir)}
    for extra in sorted(on_disk - recorded_paths):
        problems.append(f"unlisted: {extra}")

    if problems:
        raise ChecksumMismatch(
            f"{len(problems)} checksum discrepanc"
            f"{'y' if len(problems) == 1 else 'ies'} in {stage_dir}:\n  "
            + "\n  ".join(problems)
        )
