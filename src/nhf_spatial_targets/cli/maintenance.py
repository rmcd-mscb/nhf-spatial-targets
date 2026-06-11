"""``nhf-targets maintenance`` sub-app: existing-project catch-up commands.

Groups the maintenance verbs so report-vs-mutate is legible from the
grammar (issue #319):

- ``check-config`` / ``check-manifest`` are **report-only** — they never
  mutate the operator's intent artifacts; they print what's missing or
  behind and exit non-zero on drift so scripted heartbeats can detect it.
- ``rebuild-manifest`` / ``rechunk`` **mutate derived artifacts** —
  ``manifest.json`` is regenerated as a deterministic projection; NCs are
  rewritten in place to the canonical encoding. Both accept ``--dry-run``.

The config actuator remains ``nhf-targets validate`` (it regenerates
``config.effective.yml``); see docs/maintenance.md for the catch-up
sequences.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import yaml
from cyclopts import App, Parameter

from nhf_spatial_targets.cli._params import _PROJECT_DIR_PARAM

_logger = logging.getLogger(__name__)

maintenance_app = App(
    name="maintenance",
    help=(
        "Catch up an existing project: check-* report drift (read-only); "
        "rebuild-manifest / rechunk regenerate derived artifacts."
    ),
)


@maintenance_app.command(name="rechunk")
def rechunk(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
    layer: Annotated[
        str | None,
        Parameter(
            name=["--layer"],
            help="Restrict to 'aggregated' or 'target' (default: both).",
        ),
    ] = None,
    source: Annotated[
        str | None,
        Parameter(
            name=["--source"],
            help="Restrict the aggregated layer to one source key.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        Parameter(
            name=["--dry-run"],
            help="Report what would be rechunked without writing.",
        ),
    ] = False,
):
    """Backfill existing aggregated/target NCs to the chunked+compressed layout.

    Rewrites contiguous/uncompressed NetCDFs (built before #165) in place to the
    canonical io_nc layout: value-preserving, atomic, and idempotent. Does not
    touch the shared datastore's consolidated NCs, nor the daymet/ssebop
    aggregated outputs (left as-is by #165 ST3).
    """
    from nhf_spatial_targets.rechunk import rechunk_project
    from nhf_spatial_targets.workspace import load as load_project

    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)
    if not (workdir / "fabric.json").exists():
        print(
            f"Error: fabric.json not found in {workdir}. "
            "Run 'nhf-targets validate' first.",
            file=sys.stderr,
        )
        sys.exit(2)
    if layer is not None and layer not in ("aggregated", "target"):
        print(
            f"Error: invalid --layer {layer!r}; expected 'aggregated' or 'target'.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        project = load_project(workdir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # A mistyped --source would otherwise be a silent no-op; fail loudly.
    if source is not None and not (project.aggregated_dir() / source).is_dir():
        print(
            f"Error: no aggregated source directory '{source}' under "
            f"{project.aggregated_dir()}.",
            file=sys.stderr,
        )
        sys.exit(2)

    results = rechunk_project(project, layer=layer, source=source, dry_run=dry_run)

    rechunked = [r for r in results if r["status"] == "rechunked"]
    skipped = [r for r in results if r["status"].startswith("skipped")]
    planned = [r for r in results if r["status"] == "would-rechunk"]
    failed = [r for r in results if r["status"] == "failed"]
    total_before = sum(r["size_before"] for r in results)
    total_after = sum(r["size_after"] for r in results if r["size_after"] is not None)

    if dry_run:
        print(
            f"[dry-run] {len(planned)} file(s) would be rechunked, "
            f"{len(skipped)} skipped "
            f"({total_before / 1e9:.2f} GB candidate)."
        )
    else:
        saved = total_before - total_after if rechunked else 0
        reclaimed_from = sum(r["size_before"] for r in rechunked)
        pct = 100.0 * saved / reclaimed_from if reclaimed_from else 0.0
        print(
            f"Rechunked {len(rechunked)} file(s), skipped {len(skipped)}. "
            f"Reclaimed {saved / 1e9:.2f} GB "
            f"({pct:.0f}% of the rewritten files)."
        )

    if failed:
        print(f"Error: {len(failed)} file(s) failed to rechunk:", file=sys.stderr)
        for r in failed:
            print(f"  {r['path'].name}: {r.get('error')}", file=sys.stderr)
        sys.exit(1)


@maintenance_app.command(name="rebuild-manifest")
def rebuild_manifest_cmd(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
    compute_sha256: Annotated[
        bool,
        Parameter(
            name=["--compute-sha256"],
            help="Fingerprint every file (slow; multi-GB NCs). Default: off.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        Parameter(name=["--dry-run"], help="Report the projection; write nothing."),
    ] = False,
):
    """Regenerate manifest.json as a deterministic projection of on-disk artifacts.

    The one authoritative provenance command: manifest.json = f(datastore x
    catalog, data/aggregated/, targets/, fabric.json). Sources are the
    (datastore n catalog) U aggregated-dirs union; steps are synthesized
    deterministically. Identity fields (created_utc, fabric authorship, any
    release block) are read-merged, never re-minted; the result is
    byte-identical on re-run. Subsumes the former 'reconcile-manifest'.
    """
    from rich.console import Console
    from rich.table import Table

    from nhf_spatial_targets import workspace
    from nhf_spatial_targets.rebuild_manifest import rebuild_manifest

    console = Console()
    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    try:
        project = workspace.load(workdir)
        manifest = rebuild_manifest(
            project, compute_sha256=compute_sha256, dry_run=dry_run
        )
    except (FileNotFoundError, ValueError, KeyError, OSError) as e:
        print(f"rebuild-manifest failed: {e}", file=sys.stderr)
        sys.exit(1)

    sources = manifest["sources"]
    derived = sum(1 for s in sources.values() if s.get("derived_variant"))
    title = (
        "rebuild-manifest (dry run — no changes written)"
        if dry_run
        else "rebuild-manifest"
    )
    table = Table(title=title)
    table.add_column("metric", style="bold")
    table.add_column("count", justify="right")
    table.add_row("sources", str(len(sources)))
    table.add_row("  of which derived variants", str(derived))
    table.add_row("steps", str(len(manifest["steps"])))
    console.print(table)

    verb = "would write" if dry_run else "wrote"
    console.print(
        f"[bold green]{verb} manifest.json with {len(sources)} source(s) "
        f"and {len(manifest['steps'])} step(s).[/bold green]"
    )


@maintenance_app.command(name="check-config")
def check_config_cmd(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
):
    """Report optional-config features missing from the project's config.yml.

    Existing projects don't pick up new optional features (e.g.
    representative_points) added to the init template, because they were
    created before those features existed. This command compares the project's
    config.yml against the registry in upgrade_config.OPTIONAL_CONFIG_FEATURES
    and prints the literal commented block to paste for each missing feature.

    Report-only: never mutates the operator's config.yml. Exits 0 if in sync,
    1 on drift (so scripted heartbeats can detect it).
    """
    from rich.console import Console
    from rich.table import Table

    from nhf_spatial_targets.upgrade_config import (
        check_available_sources,
        check_drift,
        check_missing_targets,
    )

    console = Console()
    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)
    try:
        missing = check_drift(workdir)
        missing_targets = check_missing_targets(workdir)
        available = check_available_sources(workdir)
    except FileNotFoundError as e:
        print(f"check-config failed: {e}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        # check_missing_targets / check_available_sources parse config.yml
        # (check_drift was text-only). A malformed config must fail with an
        # actionable message, not a raw traceback -- this command exists to
        # help the operator fix config drift.
        print(f"check-config failed to parse config.yml: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Optional-feature stubs (the only signal that drives the exit code) ---
    if not missing:
        console.print(
            "[bold green]Project config is in sync with the latest "
            "optional-feature stubs in the init template.[/bold green]"
        )
    else:
        table = Table(title="Optional features missing from this project's config.yml")
        table.add_column("feature", style="bold")
        table.add_column("added")
        table.add_column("why")
        for feat in missing:
            table.add_row(feat.name, feat.added, feat.why)
        console.print(table)

        console.print(
            "\nPaste each block below into config.yml (or see "
            "src/nhf_spatial_targets/init_run.py:_CONFIG_TEMPLATE for context). "
            "This command never edits your file.\n"
        )
        for feat in missing:
            console.print(f"[bold]# --- {feat.name} ---[/bold]")
            console.print(feat.block)

    # --- Whole-target additions (report-only hint; issue #279, PR-5) ---
    if missing_targets:
        ttable = Table(
            title="Targets in the defaults schema absent from your config.yml"
        )
        ttable.add_column("target", style="bold")
        ttable.add_column("note")
        for tname in missing_targets:
            ttable.add_row(
                tname,
                "in defaults; add a targets: entry to build it on this fabric",
            )
        console.print(ttable)

    # --- Newly-available catalog sources (report-only hint; issue #279, PR-5) ---
    if available:
        stable = Table(title="Catalog sources available but not in targets.*.sources[]")
        stable.add_column("target", style="bold")
        stable.add_column("available source(s)")
        for tname, srcs in available.items():
            stable.add_row(tname, ", ".join(srcs))
        console.print(stable)

    # Exit code stays driven solely by optional-feature drift, preserving the
    # report-only contract; the whole-target and new-source tables above are
    # informational hints and never, on their own, fail the command.
    if missing:
        sys.exit(1)


@maintenance_app.command(name="check-manifest")
def check_manifest_cmd(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
):
    """Report whether manifest.json predates the current manifest schema.

    Report-only: never mutates the manifest. Exits 0 if current, 1 if behind
    (so scripted heartbeats detect drift). To normalize a behind manifest, run
    'nhf-targets maintenance rebuild-manifest -d <dir>'.
    """
    from rich.console import Console

    from nhf_spatial_targets.release.lineage import CURRENT_MANIFEST_SCHEMA_VERSION
    from nhf_spatial_targets.upgrade_manifest import check_manifest_schema

    console = Console()
    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)
    try:
        behind = check_manifest_schema(workdir)
    except (FileNotFoundError, ValueError) as e:
        print(f"check-manifest failed: {e}", file=sys.stderr)
        sys.exit(1)

    if behind is None:
        console.print(
            "[bold green]manifest.json is at the current schema version "
            f"({CURRENT_MANIFEST_SCHEMA_VERSION}).[/bold green]"
        )
        return

    console.print(
        f"[bold yellow]manifest.json is schema version {behind}; current is "
        f"{CURRENT_MANIFEST_SCHEMA_VERSION}.[/bold yellow]\n\n"
        "Run 'nhf-targets maintenance rebuild-manifest -d <dir>' to regenerate "
        "it as a complete, version-stamped projection of the on-disk artifacts. "
        "This command never edits your manifest."
    )
    sys.exit(1)
