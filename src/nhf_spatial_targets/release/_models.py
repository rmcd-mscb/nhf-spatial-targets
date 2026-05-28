"""Dataclasses describing a ScienceBase release payload.

A *plan* records what files a single ScienceBase item (the umbrella
parent, a consolidated-source child, or a fabric child) should carry and
where each lands inside the item's stage directory. ``payload.py`` builds
these plans by walking the project + datastore, then materializes them as
symlink (default) or copy trees under ``<project>/release/build/``.

A :class:`FileEntry` is the post-staging fingerprint of a single staged
file (relative path + sha256 + size + mtime); ``checksums.py`` produces
these by walking a finished stage directory.

The README / FGDC / ISO files are *reserved* here (their canonical
filenames are recorded on every plan) but **not** generated in this
phase -- MCF + FGDC + ISO + README rendering is PR-D. ``checksums.py``
only fingerprints files that actually exist, so the reserved slots do not
appear in ``SHA256SUMS`` until PR-D fills them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Metadata files every item carries. Generated in PR-D (MCF + FGDC + ISO
# + README); the staging code in PR-C records these names on each plan
# but writes no placeholder, so they stay out of the checksum manifest
# until they exist.
RESERVED_METADATA_FILES: tuple[str, ...] = ("README.md", "fgdc.xml", "iso.xml")

# Integrity files written by checksums.py. Named here so the checksum
# walker can skip them when fingerprinting a stage directory (a manifest
# must not list itself).
CHECKSUM_FILES: tuple[str, ...] = ("checksums.csv", "SHA256SUMS")


@dataclass(frozen=True)
class FileEntry:
    """A single staged file's integrity record.

    ``path`` is the POSIX-style path **relative to the stage directory**
    so that ``SHA256SUMS`` verifies with ``sha256sum -c`` run from inside
    that directory. ``size`` is in bytes; ``mtime`` is an ISO-8601 UTC
    timestamp. For a symlinked stage entry these reflect the *target*
    file's content and metadata, not the link.
    """

    path: str
    sha256: str
    size: int
    mtime: str


@dataclass(frozen=True)
class UmbrellaPlan:
    """Metadata-only umbrella (DOI parent) item.

    Carries no data payload -- only the reserved README / FGDC / ISO
    slots filled in PR-D. ``stage_dir`` is ``<build>/umbrella``.
    """

    stage_dir: Path
    reserved_files: tuple[str, ...] = RESERVED_METADATA_FILES


@dataclass(frozen=True)
class SourceChildPlan:
    """A consolidated-source child item (one per publishable source).

    ``data_files`` are absolute datastore paths to stage (the source's
    consolidated NetCDFs); it is empty for ``metadata_only`` sources
    (e.g. daymet), whose child carries only metadata pointing upstream.
    """

    source_key: str
    distribution_kind: str
    stage_dir: Path
    data_files: tuple[Path, ...] = ()
    reserved_files: tuple[str, ...] = RESERVED_METADATA_FILES


@dataclass(frozen=True)
class FabricChildPlan:
    """A fabric child item (one per NHM fabric).

    Carries the fabric GeoPackage, every aggregated NetCDF from a
    publishable source, every target NetCDF, and a point-in-time copy of
    ``manifest.json``. ``aggregated_files`` and ``target_files`` are
    absolute source paths; ``payload.py`` derives their staged layout
    (``aggregated/<source>/<file>``, ``targets/<file>``).
    """

    fabric_label: str
    stage_dir: Path
    fabric_gpkg: Path
    manifest_src: Path
    aggregated_files: tuple[Path, ...] = ()
    target_files: tuple[Path, ...] = ()
    reserved_files: tuple[str, ...] = RESERVED_METADATA_FILES


@dataclass(frozen=True)
class ReleasePayload:
    """The set of item plans produced for one ``release build`` invocation."""

    build_root: Path
    umbrella: UmbrellaPlan | None = None
    sources: tuple[SourceChildPlan, ...] = field(default_factory=tuple)
    fabric: FabricChildPlan | None = None
