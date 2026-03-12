"""nhf-targets command-line interface."""

from __future__ import annotations

import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated

import yaml
from cyclopts import App, Parameter

from nhf_spatial_targets._logging import setup_logging

_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "pipeline.yml"
_DEFAULT_WORKDIR = Path("runs")

app = App(
    name="nhf-targets",
    help="nhf-spatial-targets: build NHM calibration target datasets.",
    version=_pkg_version("nhf-spatial-targets"),
)
fetch_app = App(name="fetch", help="Download source datasets into a run workspace.")
catalog_app = App(name="catalog", help="Inspect the data source catalog.")
app.command(fetch_app)
app.command(catalog_app)


@app.meta.default
def launcher(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    verbose: Annotated[bool, Parameter(name=["--verbose", "-v"])] = False,
):
    """Global options for nhf-targets."""
    setup_logging(verbose)
    app(tokens)  # dispatch remaining tokens to the root app


@app.command
def run(
    run_dir: Annotated[
        Path | None,
        Parameter(
            name=["--run-dir", "-r"],
            help="Run workspace created by 'nhf-targets init'.",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        Parameter(name=["--config", "-c"], help="Explicit pipeline.yml path."),
    ] = None,
    target: Annotated[
        str | None,
        Parameter(
            name=["--target", "-t"], help="Run a single target (default: all enabled)."
        ),
    ] = None,
):
    """Run the calibration target pipeline."""
    if run_dir is None and config is None:
        print("Error: Provide either --run-dir or --config.", file=sys.stderr)
        sys.exit(2)
    if run_dir is not None and config is not None:
        print("Error: Provide --run-dir or --config, not both.", file=sys.stderr)
        sys.exit(2)
    if run_dir is not None and not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(2)

    config_path = (run_dir / "config.yml") if run_dir else config
    cfg = yaml.safe_load(config_path.read_text())
    targets_cfg = cfg.get("targets", {})

    to_run = (
        [target]
        if target
        else [k for k, v in targets_cfg.items() if v.get("enabled", False)]
    )

    for name in to_run:
        if name not in targets_cfg:
            print(f"Error: Unknown target: {name}", file=sys.stderr)
            sys.exit(1)
        print(f"Building target: {name}")
        _dispatch(name, targets_cfg[name], cfg, run_dir=run_dir)


def _dispatch(
    name: str,
    target_cfg: dict,
    pipeline_cfg: dict,
    run_dir: Path | None = None,
) -> None:
    """Dispatch to the appropriate target builder module."""
    from nhf_spatial_targets.targets import aet, rch, run, sca, som

    builders = {
        "runoff": run.build,
        "aet": aet.build,
        "recharge": rch.build,
        "soil_moisture": som.build,
        "snow_covered_area": sca.build,
    }
    if name not in builders:
        print(f"Error: No builder registered for target: {name}", file=sys.stderr)
        sys.exit(1)

    fabric_path = pipeline_cfg["fabric"]["path"]
    if run_dir is not None:
        output_path = str(run_dir / "targets")
    else:
        output_path = pipeline_cfg["output"]["dir"]

    builders[name](target_cfg, fabric_path, output_path)


@app.command
def init(
    fabric: Annotated[
        Path,
        Parameter(name=["--fabric", "-f"], help="Path to the HRU fabric GeoPackage."),
    ],
    id_col: Annotated[
        str,
        Parameter(name="--id-col", help="HRU ID column name in the fabric."),
    ] = "nhm_id",
    config: Annotated[
        Path | None,
        Parameter(
            name=["--config", "-c"],
            help="Pipeline config to copy into the run workspace.",
        ),
    ] = None,
    workdir: Annotated[
        Path | None,
        Parameter(name=["--workdir", "-w"], help="Root directory for run workspaces."),
    ] = None,
    run_label: Annotated[
        str | None,
        Parameter(
            name="--id", help="Short label embedded in the run ID (e.g. 'gfv11')."
        ),
    ] = None,
    buffer: Annotated[
        float,
        Parameter(name="--buffer", help="Degrees to buffer the fabric bounding box."),
    ] = 0.1,
):
    """Initialise a new run workspace tied to a specific fabric.

    Creates a dated directory under WORKDIR containing a folder skeleton,
    a snapshot of the pipeline config, a credentials template, and a
    fabric metadata file (path, bounding box, sha256).
    """
    from nhf_spatial_targets.init_run import init_run
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()

    if not fabric.exists():
        print(f"Error: Fabric file not found: {fabric}", file=sys.stderr)
        sys.exit(1)
    if fabric.is_dir():
        print(
            f"Error: Fabric path is a directory, not a file: {fabric}", file=sys.stderr
        )
        sys.exit(1)

    config_path = config or _DEFAULT_CONFIG
    if not config_path.exists():
        print(
            f"Error: Config file not found: {config_path}\n"
            "Pass --config to specify a different path.",
            file=sys.stderr,
        )
        sys.exit(1)

    workdir_path = workdir or _DEFAULT_WORKDIR

    console.print(f"[bold]Fabric:[/bold]  {fabric}")
    console.print(f"[bold]Workdir:[/bold] {workdir_path.resolve()}")
    console.print(f"[bold]Buffer:[/bold]  {buffer}°\n")
    console.print("[dim]Computing fabric bbox and sha256 (reading full file)...[/dim]")

    try:
        run_dir = init_run(
            fabric_path=fabric,
            id_col=id_col,
            config_path=config_path,
            workdir=workdir_path,
            run_label=run_label,
            buffer_deg=buffer,
        )
    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    msg = Text()
    msg.append("Run workspace created:\n", style="bold green")
    msg.append(f"  {run_dir}\n\n")
    msg.append("Next steps:\n", style="bold")
    msg.append(f"  1. Edit   {run_dir / 'config.yml'}\n")
    msg.append(f"  2. Fill   {run_dir / '.credentials.yml'}\n")
    msg.append(f"  3. Run    nhf-targets run --run-dir {run_dir}\n")
    console.print(Panel(msg, title="nhf-targets init", border_style="green"))


@fetch_app.command(name="merra2")
def fetch_merra2_cmd(
    run_dir: Annotated[
        Path,
        Parameter(
            name=["--run-dir", "-r"],
            help="Run workspace created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
):
    """Download MERRA-2 monthly land surface data (M2TMNXLND).

    Authenticates via earthaccess, searches for granules matching the
    fabric bounding box, downloads them, and prints the provenance record.
    """
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching MERRA-2 for period {period}...[/bold]")

    try:
        result = fetch_merra2(run_dir=run_dir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    console.print(
        f"[green]Downloaded {len(result['files'])} files "
        f"to {run_dir / 'data' / 'raw' / 'merra2'}[/green]"
    )
    if "kerchunk_ref" in result:
        console.print(
            f"[green]Kerchunk reference store: {run_dir / result['kerchunk_ref']}[/green]"
        )
    console.print(json_mod.dumps(result, indent=2))


@catalog_app.command(name="sources")
def catalog_sources():
    """List all registered data sources."""
    from nhf_spatial_targets.catalog import sources
    from rich import print as rprint

    rprint(sources())


@catalog_app.command(name="variables")
def catalog_variables():
    """List all calibration variable definitions."""
    from nhf_spatial_targets.catalog import variables
    from rich import print as rprint

    rprint(variables())


main = app.meta
