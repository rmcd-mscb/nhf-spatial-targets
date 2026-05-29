"""Root-level ``nhf-targets`` commands.

Contains: ``init``, ``materialize-credentials``, ``validate``, ``run``,
``rechunk``, ``reconcile-manifest``, and ``upgrade-config``. These attach
to the root ``app`` defined in :mod:`nhf_spatial_targets.cli` rather
than to one of the per-stage sub-apps (``fetch``, ``agg``, ``catalog``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import yaml
from cyclopts import Parameter

if TYPE_CHECKING:
    from cyclopts import App

    from nhf_spatial_targets.workspace import Project

_logger = logging.getLogger(__name__)


def register(app: "App") -> None:
    """Register the root-level commands on ``app``."""
    app.command(run)
    app.command(rechunk)
    app.command(reconcile_manifest_cmd, name="reconcile-manifest")
    app.command(upgrade_config_cmd, name="upgrade-config")
    app.command(upgrade_manifest_cmd, name="upgrade-manifest")
    app.command(init)
    app.command(materialize_credentials_cmd, name="materialize-credentials")
    app.command(validate)


def run(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir", "-d"],
            help="Project created by 'nhf-targets init'.",
        ),
    ],
    target: Annotated[
        str | None,
        Parameter(
            name=["--target", "-t"],
            help="Run a single target (default: all enabled).",
        ),
    ] = None,
):
    """Run the calibration target pipeline."""
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

    try:
        project = load_project(workdir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    targets_cfg = project.config.get("targets", {})
    to_run = (
        [target]
        if target
        else [k for k, v in targets_cfg.items() if v.get("enabled", False)]
    )

    # Late import + module-level lookup so tests that patch
    # ``nhf_spatial_targets.cli._dispatch`` intercept the call.
    from nhf_spatial_targets import cli as _cli

    for name in to_run:
        if name not in targets_cfg:
            print(f"Error: Unknown target: {name}", file=sys.stderr)
            sys.exit(1)
        print(f"Building target: {name}")
        try:
            _cli._dispatch(name, project)
        except NotImplementedError as exc:
            print(
                f"WARNING: target '{name}' not yet implemented; skipping ({exc})",
                file=sys.stderr,
            )
            continue
        except Exception as exc:
            _logger.exception("Error building target '%s'", name)
            print(f"Error building target '{name}': {exc}", file=sys.stderr)
            sys.exit(1)


def _dispatch(
    name: str,
    project: "Project",
) -> None:
    """Dispatch to the appropriate target builder module.

    All builders share the ``build(project: Project) -> None`` signature
    of :func:`targets.run.build`; per-target config is read from
    ``project.target(name)``.
    """
    from nhf_spatial_targets.targets import aet, rch, run, sca, som, swe

    builders = {
        "runoff": run.build,
        "aet": aet.build,
        "recharge": rch.build,
        "soil_moisture": som.build,
        "snow_covered_area": sca.build,
        "snow_water_equivalent": swe.build,
    }
    if name not in builders:
        print(f"Error: No builder registered for target: {name}", file=sys.stderr)
        sys.exit(1)
    builders[name](project)


def rechunk(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir", "-d"],
            help="Project created by 'nhf-targets init'.",
        ),
    ],
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


def reconcile_manifest_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir", "-d"],
            help="Project created by 'nhf-targets init'.",
        ),
    ],
    source: Annotated[
        list[str] | None,
        Parameter(
            name=["--source"],
            help="Catalog source key to reconcile (repeatable). Default: all.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        Parameter(name=["--dry-run"], help="Report what would change; write nothing."),
    ] = False,
    checksum: Annotated[
        bool,
        Parameter(name=["--checksum"], help="Compute sha256 for each record (slow)."),
    ] = False,
):
    """Backfill manifest.json from consolidated NCs already in the datastore.

    Use after creating a new project against a datastore that another project
    already populated. Adds 'provenance: reconciled' file records for sources
    found on disk but missing from this project's manifest; never overwrites
    existing records. See docs/architecture/reconcile-manifest.md.
    """
    from rich.console import Console
    from rich.table import Table

    from nhf_spatial_targets import workspace
    from nhf_spatial_targets.reconcile import reconcile_manifest

    console = Console()
    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    try:
        project = workspace.load(workdir)
        results = reconcile_manifest(
            project, sources=source, dry_run=dry_run, checksum=checksum
        )
    except (FileNotFoundError, ValueError, KeyError, OSError) as e:
        print(f"reconcile-manifest failed: {e}", file=sys.stderr)
        sys.exit(1)

    title = (
        "reconcile-manifest (dry run — no changes written)"
        if dry_run
        else "reconcile-manifest"
    )
    table = Table(title=title)
    table.add_column("source", style="bold")
    table.add_column("status")
    table.add_column("on disk", justify="right")
    table.add_column("already recorded", justify="right")
    table.add_column("added", justify="right")
    for r in results:
        table.add_row(
            r.source_key,
            r.status,
            str(r.on_disk),
            str(r.already_recorded),
            str(r.added),
        )
    console.print(table)

    total_added = sum(r.added for r in results)
    verb = "would add" if dry_run else "added"
    console.print(
        f"[bold green]{verb} {total_added} reconciled record(s).[/bold green]"
    )


def upgrade_config_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir", "-d"],
            help="Project created by 'nhf-targets init'.",
        ),
    ],
):
    """Report optional-config features missing from the project's config.yml.

    Existing projects don't pick up new optional features (e.g. fabric.token,
    representative_points) added to the init template, because they were
    created before those features existed. This command compares the project's
    config.yml against the registry in upgrade_config.OPTIONAL_CONFIG_FEATURES
    and prints the literal commented block to paste for each missing feature.

    Report-only: never mutates the operator's config.yml. Exits 0 if in sync,
    1 on drift (so scripted heartbeats can detect it).
    """
    from rich.console import Console
    from rich.table import Table

    from nhf_spatial_targets.upgrade_config import check_drift

    console = Console()
    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)
    try:
        missing = check_drift(workdir)
    except FileNotFoundError as e:
        print(f"upgrade-config failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not missing:
        console.print(
            "[bold green]Project config is in sync with the latest "
            "optional-feature stubs in the init template.[/bold green]"
        )
        return

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
    sys.exit(1)


def upgrade_manifest_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir", "-d"],
            help="Project created by 'nhf-targets init'.",
        ),
    ],
):
    """Report whether manifest.json predates the current manifest schema.

    Report-only: never mutates the manifest. Exits 0 if current, 1 if behind
    (so scripted heartbeats detect drift). A behind manifest will be normalized
    by the forthcoming 'rebuild-manifest' command (issue #279, lands in a later
    PR).
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
        print(f"upgrade-manifest failed: {e}", file=sys.stderr)
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
        "The forthcoming 'rebuild-manifest' command (issue #279, lands in a "
        "later PR) will regenerate it as a complete, version-stamped projection "
        "of the on-disk artifacts. This command never edits your manifest."
    )
    sys.exit(1)


def init(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir", "-d"],
            help="Directory to create as the new project.",
        ),
    ],
):
    """Initialise a new project with a config template.

    Creates a directory skeleton with config.yml and .credentials.yml.
    Edit those files, then run 'nhf-targets validate --project-dir <dir>'.
    """
    from nhf_spatial_targets.init_run import init_project
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()

    try:
        result = init_project(workdir)
    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    msg = Text()
    msg.append("Project created:\n", style="bold green")
    msg.append(f"  {result}\n\n")
    msg.append("Next steps:\n", style="bold")
    msg.append(f"  1. Edit   {result / 'config.yml'}\n")
    msg.append(f"  2. Fill   {result / '.credentials.yml'}\n")
    msg.append(f"  3. Run    nhf-targets validate --project-dir {result}\n")
    console.print(Panel(msg, title="nhf-targets init", border_style="green"))


def materialize_credentials_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir", "-d"],
            help="Project directory containing .credentials.yml.",
        ),
    ],
):
    """Copy credentials from .credentials.yml into ~/.cdsapirc and ~/.netrc.

    Reads the 'cds' and 'nasa_earthdata' sections from the project's
    .credentials.yml and writes the corresponding dotfiles consumed by
    cdsapi and earthaccess at runtime.

    Both files are written atomically and set to mode 0600.  Run this command
    after editing or rotating .credentials.yml.

    Each section (cds, nasa_earthdata) is processed independently — one
    section failing does not prevent the other from being written.  The
    command exits non-zero if any section fails.

    Exit codes:
      0 — all sections written successfully
      1 — incomplete credentials (ValueError) — user action required
      2 — project directory not found
      3 — write failure (OSError) — system action required
    """
    from rich.console import Console
    from rich.table import Table

    from nhf_spatial_targets.credentials import (
        materialize_cdsapirc,
        materialize_netrc_earthdata,
    )

    console = Console()

    cred_path = workdir / ".credentials.yml"
    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)
    if not cred_path.exists():
        print(
            f"Error: .credentials.yml not found in {workdir}. "
            "Run 'nhf-targets init --project-dir <dir>' to create a template.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        raw = yaml.safe_load(cred_path.read_text())
    except yaml.YAMLError as exc:
        print(f"Error: Cannot parse {cred_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    if raw is None:
        print(
            f"Error: {cred_path} is empty — did you save your edits?",
            file=sys.stderr,
        )
        sys.exit(1)
    creds = raw if isinstance(raw, dict) else {}

    table = Table(title="Credential materialisation", show_header=True)
    table.add_column("Section", style="bold")
    table.add_column("Target file")
    table.add_column("Status")

    errors: list[str] = []

    # --- CDS ---
    try:
        cds_path = materialize_cdsapirc(creds)
        table.add_row("cds", str(cds_path), "[green]written[/green]")
    except ValueError as exc:
        msg = f"{cred_path}: {exc}"
        table.add_row("cds", "~/.cdsapirc", f"[yellow]skipped[/yellow]: {exc}")
        errors.append(("user", msg))
    except OSError as exc:
        msg = f"{cred_path}: {exc}"
        table.add_row("cds", "~/.cdsapirc", f"[red]error[/red]: {exc}")
        errors.append(("system", msg))

    # --- NASA Earthdata ---
    try:
        netrc_path = materialize_netrc_earthdata(creds)
        table.add_row("nasa_earthdata", str(netrc_path), "[green]written[/green]")
    except ValueError as exc:
        msg = f"{cred_path}: {exc}"
        table.add_row("nasa_earthdata", "~/.netrc", f"[yellow]skipped[/yellow]: {exc}")
        errors.append(("user", msg))
    except OSError as exc:
        msg = f"{cred_path}: {exc}"
        table.add_row("nasa_earthdata", "~/.netrc", f"[red]error[/red]: {exc}")
        errors.append(("system", msg))

    console.print(table)

    if errors:
        has_system_error = any(kind == "system" for kind, _ in errors)
        has_user_error = any(kind == "user" for kind, _ in errors)
        if has_user_error:
            console.print(
                "\n[yellow]One or more sections were skipped due to missing or "
                "incomplete credentials.  Fill in .credentials.yml and re-run.[/yellow]"
            )
        if has_system_error:
            console.print(
                "\n[red]One or more sections failed due to a system error "
                "(e.g. filesystem permissions).  See the table above for details.[/red]"
            )
        sys.exit(3 if has_system_error else 1)

    console.print(
        "\n[bold green]All credentials materialised successfully.[/bold green]"
    )


def validate(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir", "-d"],
            help="Project directory to validate.",
        ),
    ],
):
    """Validate a project: check config, fabric, credentials, and catalog.

    On success, writes fabric.json and manifest.json into the project.
    """
    from rich.console import Console

    from nhf_spatial_targets.validate import validate_workspace

    console = Console()

    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    try:
        validate_workspace(workdir)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during validation")
        print(
            f"Unexpected validation error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print(f"[bold green]Project validated successfully:[/bold green] {workdir}")
