"""Root-level ``nhf-targets`` commands.

Contains: ``init``, ``materialize-credentials``, ``validate``, and ``run``.
These attach to the root ``app`` defined in :mod:`nhf_spatial_targets.cli`
rather than to one of the per-stage sub-apps (``fetch``, ``agg``,
``catalog``, ``release``, ``maintenance``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import yaml
from cyclopts import Parameter

from nhf_spatial_targets.cli._params import _PROJECT_DIR_PARAM

if TYPE_CHECKING:
    from cyclopts import App

    from nhf_spatial_targets.workspace import Project

_logger = logging.getLogger(__name__)

# Short nicknames accepted by ``run --target`` in addition to the long
# config keys, matching the ``pixi run run-<nick>`` task names (issue #319).
_TARGET_NICKNAMES = {
    "rch": "recharge",
    "som": "soil_moisture",
    "sca": "snow_covered_area",
    "swe": "snow_water_equivalent",
}


def register(app: "App") -> None:
    """Register the root-level commands on ``app``."""
    app.command(run)
    app.command(init)
    app.command(materialize_credentials_cmd, name="materialize-credentials")
    app.command(validate)


def run(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
    target: Annotated[
        str | None,
        Parameter(
            name=["--target", "-t"],
            help=(
                "Run a single target (default: all enabled). Accepts the "
                "config keys runoff / aet / recharge / soil_moisture / "
                "snow_covered_area / snow_water_equivalent or the short "
                "nicknames rch / som / sca / swe."
            ),
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

    requested = target
    if target is not None:
        target = _TARGET_NICKNAMES.get(target, target)

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
            # Report the token the user typed, not the nickname expansion.
            print(f"Error: Unknown target: {requested or name}", file=sys.stderr)
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
