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
phase -- metadata rendering is a later phase. ``checksums.py`` only
fingerprints files that actually exist, so the reserved slots do not
appear in ``SHA256SUMS`` until those files exist.

This module also hosts :class:`ReleaseError`, the shared base for every
deliberate release-layer error. It lives here -- the dependency-free leaf
both ``sb_client`` and ``publish`` import -- so they can subclass it without
an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class ReleaseError(RuntimeError):
    """Base for every error the release tooling raises deliberately.

    Lives here (the dependency-free shared-types module) so both the
    foundational :mod:`~nhf_spatial_targets.release.sb_client` primitive and
    the publish/dry-run orchestration can subclass it without an import cycle.
    Catching :class:`ReleaseError` catches every release-layer condition --
    a stale registry, an ambiguous title, a failed pre-flight gate -- as
    opposed to a transient transport failure (which the client retries) or a
    genuine bug (which should propagate).
    """


# The two recognized source distribution kinds, mirroring
# ``catalog.DISTRIBUTION_KIND_TOKENS``. A test asserts the two stay in
# sync. "data" ships the consolidated NetCDFs; "metadata_only" ships only
# metadata pointing at the upstream archive.
DistributionKind = Literal["data", "metadata_only"]

# Metadata files every item carries. Generated in a later phase (MCF +
# FGDC + ISO + README); the staging code here records these names on each
# plan but writes no placeholder, so they stay out of the checksum
# manifest until they exist.
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
    slots filled in a later phase. ``stage_dir`` is ``<build>/umbrella``.
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
    distribution_kind: DistributionKind
    stage_dir: Path
    data_files: tuple[Path, ...] = ()
    reserved_files: tuple[str, ...] = RESERVED_METADATA_FILES

    def __post_init__(self) -> None:
        # A metadata-only child publishes no data, so carrying data files
        # would be a contradiction that misleads downstream FGDC emission.
        if self.distribution_kind == "metadata_only" and self.data_files:
            raise ValueError(
                f"metadata_only source {self.source_key!r} must carry no "
                f"data_files; got {len(self.data_files)}."
            )


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
    sources: tuple[SourceChildPlan, ...] = ()
    fabric: FabricChildPlan | None = None
