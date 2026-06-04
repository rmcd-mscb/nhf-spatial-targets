"""Offline build orchestrator for the ScienceBase release.

This module is the glue that turns the existing release pieces -- payload
staging (:mod:`~nhf_spatial_targets.release.payload`), MCF construction
(:mod:`~nhf_spatial_targets.release.mcf`), FGDC / ISO / README rendering
(:mod:`~nhf_spatial_targets.release.fgdc` / :mod:`.iso` / :mod:`.readme`), and
checksums (:mod:`~nhf_spatial_targets.release.checksums`) -- into a complete,
self-contained release payload on disk under
``<project>/release/build/<scope>/``.

It is **entirely offline**: no ScienceBase client, registry, or network I/O.
Publishing, the registry, and dry-run live in separate modules
(:mod:`~nhf_spatial_targets.release.publish` /
:mod:`~nhf_spatial_targets.release.registry`); the CLI that drives this is
:mod:`~nhf_spatial_targets.cli.release`. The build is **catalog- + manifest-driven**: source
children flow from :func:`payload.sources_used` (the intersection of
:func:`catalog.publishable_sources` with the sources the project actually
aggregated on disk) and fabric participation from the project ``manifest.json``
-- never a hard-coded source list.

Per-item build sequence (deterministic for a fixed ``now``)::

    1. stage the payload         payload.stage_*()  (symlink by default; --copy)
    2. load defaults + manifest  release_defaults.yml + <project>/manifest.json
    3. declare distribution list  the staged relative paths (+ the not-yet-
                                  written README/checksums for the fabric child)
    4. build the MCF dict        mcf.build_*_mcf(distribution_files=<list>)
    5. render metadata           fgdc.xml + iso.xml + README.md into the stage
    6. compute checksums LAST     checksums.csv + SHA256SUMS (after every file
                                  exists -- a manifest must not list itself)

Distribution-list decision:
    The MCF ``distribution.online`` list records the staged **downloadable
    artifacts**, with the metadata *records* (``fgdc.xml`` / ``iso.xml``)
    excluded -- they describe the item, they are not its data. Because the MCF
    builder only records names (strings), the list can name files written
    after the MCF is rendered (README, checksums), which is why the order is
    *declare list -> render -> checksum last*.

    - **Fabric child:** ``fabric.gpkg`` + every aggregated NC + every target NC
      + ``manifest.json`` + ``README.md`` + ``checksums.csv`` + ``SHA256SUMS``.
      This matches the fgdc-field-map's "every staged file" intent.
    - **Source child:** the staged consolidated NetCDFs only. The
      source-distribution builder labels every listed entry "Consolidated
      CF-1.6 NetCDF.", so listing README / checksums there would mislabel them;
      those files are still staged and checksummed, just not enumerated in
      ``distribution.online``. ``metadata_only`` sources (daymet) stage no data
      and so carry only the upstream landing entry.
    - **Umbrella:** metadata-only; no data files, just the landing entry.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from nhf_spatial_targets.release import checksums, fgdc, iso, mcf, payload, readme
from nhf_spatial_targets.release._models import (
    CHECKSUM_FILES,
    RESERVED_METADATA_FILES,
    FileEntry,
)
from nhf_spatial_targets.release.defaults import load_release_defaults
from nhf_spatial_targets.workspace import Project

# release build --scope choices; "fabric" is the default (one fabric child).
BuildScope = Literal["fabric", "sources", "all"]
ItemKind = Literal["umbrella", "source", "fabric"]

# Generated metadata records (fgdc.xml / iso.xml) deliberately omitted from the
# fabric distribution list -- they describe the item, they are not its data.
# The human-readable README and the integrity manifests are kept.
_FABRIC_EXTRA_DISTRIBUTION = ("README.md", "checksums.csv", "SHA256SUMS")


@dataclass(frozen=True)
class BuildResult:
    """The on-disk result of building one release item.

    ``key`` is the source key for a source child, the fabric label for a
    fabric child, and ``None`` for the umbrella. ``mcf`` is the dict the
    metadata files were rendered from; ``entries`` is the checksum record of
    every staged file (the same list backing ``checksums.csv`` / ``SHA256SUMS``).
    """

    kind: ItemKind
    key: str | None
    stage_dir: Path
    mcf: dict
    entries: tuple[FileEntry, ...]

    def __post_init__(self) -> None:
        # The umbrella is keyless; source/fabric children always carry a key.
        # Guard the correlation the downstream publish phase will rely on
        # (mirrors SourceChildPlan.__post_init__ in _models.py).
        if (self.kind == "umbrella") != (self.key is None):
            raise ValueError(
                f"{self.kind!r} child key invariant violated: umbrella must "
                f"have key=None and source/fabric a non-None key; got "
                f"key={self.key!r}."
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_manifest(project: Project) -> dict:
    """Load the project's live ``manifest.json`` (``{}`` if it is absent).

    The fabric stage carries a point-in-time *copy* of this file (staged by
    :func:`payload.stage_fabric_child`); the MCF lineage reads the same
    content here so the metadata and the staged snapshot agree.

    An absent manifest is a legitimate state (e.g. a freshly-validated project)
    and the MCF builders accept an empty mapping. A *corrupt* manifest is not:
    silently falling back to ``{}`` would build a release with empty lineage
    and an empty source list, so it is a loud failure here -- mirroring
    :func:`nhf_spatial_targets.release.lineage._load_or_init`.
    """
    path = project.manifest_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manifest.json at {path} is corrupt: {exc}. The release lineage "
            f"and source list are built from this file; inspect it manually or "
            f"restore it before building."
        ) from exc


def _release_cfg(project: Project) -> dict:
    """The merged ``release:`` config block (``{}`` if the project omits it)."""
    return project.config.get("release") or {}


def _staged_data_relpaths(stage_dir: Path) -> list[str]:
    """Sorted POSIX relpaths of every staged **data** file under *stage_dir*.

    Excludes the generated metadata records and integrity manifests (so the
    list is identical whether or not a previous build left them behind), which
    lets us declare the distribution list *before* those files are written.
    Symlinks are kept (the default stager links data files); only directories
    are skipped.

    A *dangling* symlink (the datastore can live on a separately-mounted drive)
    is a hard error, not a silent inclusion: were it enumerated, it would be
    advertised in ``distribution.online`` and baked into the rendered FGDC /
    ISO / README before :func:`checksums.compute_checksums` caught it. Failing
    here -- mirroring :func:`checksums._iter_staged_files` -- keeps the failure
    early and names the real cause.
    """
    skip = set(RESERVED_METADATA_FILES) | set(CHECKSUM_FILES)
    rels: list[str] = []
    for path in stage_dir.rglob("*"):
        if path.is_symlink() and not path.exists():
            raise FileNotFoundError(
                f"staged entry points at a missing target: {path} -> "
                f"{os.readlink(path)}. The datastore drive may be unmounted, "
                f"or the source file was removed after staging."
            )
        if path.is_dir():
            continue
        rel = path.relative_to(stage_dir).as_posix()
        if rel in skip:
            continue
        rels.append(rel)
    return sorted(rels)


def _render_metadata(stage_dir: Path, mcf_dict: dict, *, kind: ItemKind) -> None:
    """Render and write ``README.md`` / ``fgdc.xml`` / ``iso.xml`` to *stage_dir*.

    A pure function of *mcf_dict* (every date is already baked into the MCF),
    so re-rendering an unchanged MCF rewrites byte-identical files.
    """
    (stage_dir / "fgdc.xml").write_text(fgdc.render_fgdc(mcf_dict, kind=kind))
    (stage_dir / "iso.xml").write_text(iso.render_iso(mcf_dict))
    (stage_dir / "README.md").write_text(readme.render_readme(mcf_dict, kind=kind))


def _finish(
    stage_dir: Path, mcf_dict: dict, *, kind: ItemKind, key: str | None
) -> BuildResult:
    """Render metadata, checksum the finished stage, and package the result.

    Shared tail of every ``build_*`` function: metadata records are written
    first, then :func:`checksums.compute_checksums` runs **last** so it
    fingerprints a complete payload (and never lists ``checksums.csv`` /
    ``SHA256SUMS`` themselves).
    """
    _render_metadata(stage_dir, mcf_dict, kind=kind)
    entries = checksums.compute_checksums(stage_dir)
    return BuildResult(
        kind=kind, key=key, stage_dir=stage_dir, mcf=mcf_dict, entries=tuple(entries)
    )


def _child_mcfs(
    project: Project, *, defaults: dict, manifest: dict, now: datetime | None
) -> list[dict]:
    """Build the fabric + source child MCF dicts (no staging, no rendering).

    The umbrella only needs each child's bbox / temporal extent / title to
    union them, so this builds the MCF dicts directly without touching disk.
    Distribution-file lists are irrelevant to the union and left unset.
    """
    children = [
        mcf.build_fabric_mcf(
            config=project.config,
            fabric=project.fabric,
            defaults=defaults,
            manifest=manifest,
            doi=_release_cfg(project).get("doi"),
            now=now,
        )
    ]
    for key in payload.sources_used(project):
        children.append(
            mcf.build_source_mcf(key, defaults=defaults, manifest=manifest, now=now)
        )
    return children


# ---------------------------------------------------------------------------
# Per-item builders
# ---------------------------------------------------------------------------


def build_source_child(
    project: Project, key: str, *, copy: bool = False, now: datetime | None = None
) -> BuildResult:
    """Build the consolidated-source child for *key* under ``build/sources/<key>``.

    Stages the source's consolidated NetCDFs (none for ``metadata_only``
    sources such as daymet), then renders FGDC / ISO / README and checksums.
    The distribution list is the staged NetCDFs; see the module docstring for
    why README / checksums are not enumerated there for source children.
    """
    defaults = load_release_defaults()
    manifest = _load_manifest(project)
    plan = payload.stage_source_child(project, key, copy=copy)
    mcf_dict = mcf.build_source_mcf(
        key,
        defaults=defaults,
        manifest=manifest,
        distribution_files=_staged_data_relpaths(plan.stage_dir),
        now=now,
    )
    return _finish(plan.stage_dir, mcf_dict, kind="source", key=key)


def build_fabric_child(
    project: Project, *, copy: bool = False, now: datetime | None = None
) -> BuildResult:
    """Build the fabric child under ``build/fabric/<label>``.

    Stages ``fabric.gpkg`` + aggregated NCs + target NCs + a ``manifest.json``
    snapshot, then renders FGDC / ISO / README and checksums. The distribution
    list is every staged downloadable artifact (the metadata records aside);
    see the module docstring.
    """
    defaults = load_release_defaults()
    manifest = _load_manifest(project)
    plan = payload.stage_fabric_child(project, copy=copy)
    data = _staged_data_relpaths(plan.stage_dir)
    distribution_files = sorted(set(data) | set(_FABRIC_EXTRA_DISTRIBUTION))
    mcf_dict = mcf.build_fabric_mcf(
        config=project.config,
        fabric=project.fabric,
        defaults=defaults,
        manifest=manifest,
        distribution_files=distribution_files,
        doi=_release_cfg(project).get("doi"),
        now=now,
    )
    return _finish(plan.stage_dir, mcf_dict, kind="fabric", key=plan.fabric_label)


def build_umbrella(
    project: Project,
    *,
    copy: bool = False,
    children: list[dict] | None = None,
    now: datetime | None = None,
) -> BuildResult:
    """Build the metadata-only umbrella under ``build/umbrella``.

    The umbrella unions its children's spatial / temporal extents and lists
    their titles. ``children`` is the list of already-built child MCF dicts;
    when ``None`` (a standalone umbrella build), the fabric + source child MCFs
    are rebuilt locally -- a local build unions exactly the children this
    project would produce. The registry-driven cross-project union and DOI
    refresh live in the publish/registry layer, not here; any operator-set DOI
    is read from ``config.release.doi`` here.
    """
    defaults = load_release_defaults()
    release_cfg = _release_cfg(project)
    if children is None:
        children = _child_mcfs(
            project, defaults=defaults, manifest=_load_manifest(project), now=now
        )
    plan = payload.stage_umbrella(project, copy=copy)
    mcf_dict = mcf.build_umbrella_mcf(
        defaults=defaults,
        children=children,
        authors=list(release_cfg.get("authors") or []),
        doi=release_cfg.get("doi"),
        now=now,
    )
    return _finish(plan.stage_dir, mcf_dict, kind="umbrella", key=None)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def build_all(
    project: Project,
    *,
    scope: BuildScope = "fabric",
    copy: bool = False,
    now: datetime | None = None,
) -> list[BuildResult]:
    """Build the items selected by *scope* and return their results in order.

    - ``"fabric"`` (default): the fabric child only.
    - ``"sources"``: one source child per publishable source the project used.
    - ``"all"``: the fabric child, every source child, and the umbrella -- the
      umbrella unions the children built in this same invocation.
    """
    if scope == "fabric":
        return [build_fabric_child(project, copy=copy, now=now)]
    if scope == "sources":
        return [
            build_source_child(project, key, copy=copy, now=now)
            for key in payload.sources_used(project)
        ]
    if scope == "all":
        results = [build_fabric_child(project, copy=copy, now=now)]
        results += [
            build_source_child(project, key, copy=copy, now=now)
            for key in payload.sources_used(project)
        ]
        results.append(
            build_umbrella(
                project, copy=copy, children=[r.mcf for r in results], now=now
            )
        )
        return results
    raise ValueError(
        f"Unknown scope {scope!r}; expected 'fabric', 'sources', or 'all'."
    )
