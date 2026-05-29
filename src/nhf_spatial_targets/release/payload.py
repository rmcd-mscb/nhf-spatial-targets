"""Stage ScienceBase release payloads under ``<project>/release/build/``.

A *stage* is the on-disk tree for one ScienceBase item:

- ``build/fabric/<label>/``  -- fabric.gpkg + aggregated NCs + target NCs
  + a copied ``manifest.json``.
- ``build/sources/<key>/``   -- the source's consolidated NetCDFs from the
  shared datastore (empty for ``metadata_only`` sources such as daymet).
- ``build/umbrella/``        -- metadata only; no data payload.

Data files are **symlinked** by default -- the consolidated and aggregated
NetCDFs are routinely multi-GB and a release spans dozens of them, so
copying would waste disk and time. Pass ``copy=True`` for filesystems
without symlink support. ``manifest.json`` is the one exception: it is
always *copied*, because it is a point-in-time provenance snapshot for the
release and must not mutate when a later pipeline run rewrites the live
``manifest.json``.

Publishability filtering (fabric children):
    Aggregated subdirectories whose name is a known catalog source key that
    is **not** publishable -- e.g. ``watergap22d`` (held for legal review)
    or any superseded source -- are excluded, so held-back data never
    leaks into a fabric child. Aggregated subdirectories that are not
    catalog keys (derived variants such as ``era5_land_sd``) are *kept*:
    they are products of a publishable source, not independently gated.

The README / FGDC / ISO slots are *reserved* (recorded on each plan) but
not written here -- metadata generation is a later phase. Re-running a
stage overwrites each managed file in place (idempotent for the same
inputs); pruning entries whose inputs were removed is deferred to the
build orchestrator.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from nhf_spatial_targets import catalog
from nhf_spatial_targets.release._models import (
    FabricChildPlan,
    ReleaseError,
    ReleasePayload,
    SourceChildPlan,
    UmbrellaPlan,
)
from nhf_spatial_targets.workspace import Project


def resolve_fabric_label(project: Project) -> str:
    """Return the human-readable fabric label for *project*.

    Uses ``config.release.fabric_label`` when set, else the stem of the
    fabric file path (e.g. ``/data/fab/gfv2_nhru_merged.gpkg`` ->
    ``gfv2_nhru_merged``).
    """
    release_cfg = project.config.get("release") or {}
    label = release_cfg.get("fabric_label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return Path(project.fabric["path"]).stem


def _nc_files(directory: Path) -> tuple[Path, ...]:
    """Return all ``*.nc`` regular files under *directory*, sorted by path."""
    if not directory.exists():
        return ()
    return tuple(sorted(p for p in directory.rglob("*.nc") if p.is_file()))


def _publishable_agg_dir(name: str) -> bool:
    """Whether an aggregated subdir named *name* may ship in a fabric child.

    A known catalog key must be publishable; an unknown name (a derived
    variant) is always allowed.
    """
    all_keys = set(catalog.sources())
    if name not in all_keys:
        return True
    return name in catalog.publishable_sources()


def _link_or_copy(src: Path, dest: Path, *, copy: bool) -> None:
    """Materialize *dest* from *src* as a symlink (default) or a copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    if copy:
        shutil.copy2(src, dest)
    else:
        os.symlink(os.path.abspath(src), dest)


def _copy_file(src: Path, dest: Path) -> None:
    """Copy *src* to *dest* (used for the manifest snapshot, never symlinked)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    shutil.copy2(src, dest)


# ---------------------------------------------------------------------------
# Plan builders (read-only: enumerate what would be staged)
# ---------------------------------------------------------------------------


def sources_used(project: Project) -> list[str]:
    """Return publishable catalog source keys the project aggregated.

    Membership is decided by the per-source aggregated subdirectories
    present on disk, intersected with :func:`catalog.publishable_sources`.
    Derived-variant subdirs that are not catalog keys (e.g.
    ``era5_land_sd``) are not independent source children and are omitted.
    """
    agg = project.aggregated_dir()
    if not agg.exists():
        return []
    publishable = catalog.publishable_sources()
    present = {p.name for p in agg.iterdir() if p.is_dir()}
    return sorted(name for name in present if name in publishable)


def plan_fabric_child(project: Project) -> FabricChildPlan:
    """Build the :class:`FabricChildPlan` for *project* without staging."""
    aggregated_root = project.aggregated_dir()
    aggregated: list[Path] = []
    if aggregated_root.exists():
        for source_dir in sorted(p for p in aggregated_root.iterdir() if p.is_dir()):
            if not _publishable_agg_dir(source_dir.name):
                continue
            aggregated.extend(_nc_files(source_dir))

    label = resolve_fabric_label(project)
    return FabricChildPlan(
        fabric_label=label,
        stage_dir=project.release_build_dir / "fabric" / label,
        fabric_gpkg=Path(project.fabric["path"]),
        manifest_src=project.manifest_path,
        aggregated_files=tuple(aggregated),
        target_files=_nc_files(project.targets_dir()),
    )


def plan_source_child(project: Project, source_key: str) -> SourceChildPlan:
    """Build the :class:`SourceChildPlan` for *source_key* without staging.

    ``metadata_only`` sources (e.g. daymet) carry no data files -- their
    child item ships metadata pointing at the upstream archive.
    """
    rb = catalog.release_block(source_key)
    distribution_kind = rb["distribution_kind"]
    data_files: tuple[Path, ...] = ()
    if distribution_kind == "data":
        data_files = _nc_files(project.raw_dir(source_key))
    return SourceChildPlan(
        source_key=source_key,
        distribution_kind=distribution_kind,
        stage_dir=project.release_build_dir / "sources" / source_key,
        data_files=data_files,
    )


def plan_umbrella(project: Project) -> UmbrellaPlan:
    """Build the :class:`UmbrellaPlan` (metadata-only) for *project*."""
    return UmbrellaPlan(stage_dir=project.release_build_dir / "umbrella")


# ---------------------------------------------------------------------------
# Stagers (materialize the plan on disk)
# ---------------------------------------------------------------------------


def stage_fabric_child(project: Project, *, copy: bool = False) -> FabricChildPlan:
    """Stage the fabric child tree for *project*; return the plan staged.

    Layout under ``build/fabric/<label>/``::

        fabric.gpkg                  (symlink, or copy with copy=True)
        manifest.json                (always copied -- provenance snapshot)
        aggregated/<source>/<nc>     (symlinks)
        targets/<nc>                 (symlinks)

    README.md / fgdc.xml / iso.xml are reserved on the returned plan but
    not yet generated.
    """
    plan = plan_fabric_child(project)
    plan.stage_dir.mkdir(parents=True, exist_ok=True)

    _link_or_copy(plan.fabric_gpkg, plan.stage_dir / "fabric.gpkg", copy=copy)

    if not plan.manifest_src.exists():
        raise ReleaseError(
            f"Cannot stage fabric child: manifest.json missing at "
            f"{plan.manifest_src}. Run 'nhf-targets validate' then "
            f"'nhf-targets rebuild-manifest' for this project before publishing."
        )
    _copy_file(plan.manifest_src, plan.stage_dir / "manifest.json")

    aggregated_root = project.aggregated_dir()
    for src in plan.aggregated_files:
        rel = src.relative_to(aggregated_root)
        _link_or_copy(src, plan.stage_dir / "aggregated" / rel, copy=copy)

    for src in plan.target_files:
        _link_or_copy(src, plan.stage_dir / "targets" / src.name, copy=copy)

    return plan


def stage_source_child(
    project: Project, source_key: str, *, copy: bool = False
) -> SourceChildPlan:
    """Stage a consolidated-source child tree; return the plan staged.

    Datastore subpaths are preserved (e.g. ``daily/era5_land_daily_1979.nc``
    stages to ``<stage>/daily/era5_land_daily_1979.nc``). ``metadata_only``
    sources stage no data files -- only the reserved metadata slots.
    """
    plan = plan_source_child(project, source_key)
    plan.stage_dir.mkdir(parents=True, exist_ok=True)

    datastore_root = project.raw_dir(source_key)
    for src in plan.data_files:
        rel = src.relative_to(datastore_root)
        _link_or_copy(src, plan.stage_dir / rel, copy=copy)

    return plan


def stage_umbrella(project: Project, *, copy: bool = False) -> UmbrellaPlan:
    """Create the metadata-only umbrella stage directory; return the plan.

    ``copy`` is accepted for a uniform stager signature; the umbrella
    carries no data files, so it is unused.
    """
    plan = plan_umbrella(project)
    plan.stage_dir.mkdir(parents=True, exist_ok=True)
    return plan


def stage_all(project: Project, *, copy: bool = False) -> ReleasePayload:
    """Stage the fabric child, every publishable source child, and umbrella.

    Returns a :class:`ReleasePayload` collecting the staged plans.
    """
    fabric = stage_fabric_child(project, copy=copy)
    sources = tuple(
        stage_source_child(project, key, copy=copy) for key in sources_used(project)
    )
    umbrella = stage_umbrella(project, copy=copy)
    return ReleasePayload(
        build_root=project.release_build_dir,
        umbrella=umbrella,
        sources=sources,
        fabric=fabric,
    )
