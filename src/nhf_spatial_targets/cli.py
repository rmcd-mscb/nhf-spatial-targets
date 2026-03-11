"""nhf-targets command-line interface."""

from __future__ import annotations

from pathlib import Path

import click
import yaml


_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "pipeline.yml"
_DEFAULT_WORKDIR = Path("runs")


@click.group()
def main():
    """nhf-spatial-targets: build NHM calibration target datasets."""


@main.command()
@click.option(
    "--run-dir",
    "-r",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Run workspace created by 'nhf-targets init'. "
    "Uses run-dir/config.yml as the pipeline config.",
)
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Explicit pipeline.yml path (alternative to --run-dir).",
)
@click.option(
    "--target",
    "-t",
    default=None,
    help="Run a single target (default: all enabled targets).",
)
def run(run_dir: Path | None, config: Path | None, target: str | None):
    """Run the calibration target pipeline.

    Provide either --run-dir (preferred, points to an init workspace) or
    --config (legacy, points directly to a pipeline.yml).
    """
    if run_dir is None and config is None:
        raise click.UsageError("Provide either --run-dir or --config.")
    if run_dir is not None and config is not None:
        raise click.UsageError("Provide --run-dir or --config, not both.")

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
            raise click.ClickException(f"Unknown target: {name}")
        click.echo(f"Building target: {name}")
        _dispatch(name, targets_cfg[name], cfg, run_dir=run_dir)


def _dispatch(
    name: str,
    target_cfg: dict,
    pipeline_cfg: dict,
    run_dir: Path | None = None,
) -> None:
    """Dispatch to the appropriate target builder module."""
    from nhf_spatial_targets.targets import run, aet, rch, som, sca

    builders = {
        "runoff": run.build,
        "aet": aet.build,
        "recharge": rch.build,
        "soil_moisture": som.build,
        "snow_covered_area": sca.build,
    }
    if name not in builders:
        raise click.ClickException(f"No builder registered for target: {name}")

    # When run_dir is supplied, data paths are relative to the workspace.
    fabric_path = pipeline_cfg["fabric"]["path"]
    if run_dir is not None:
        output_path = str(run_dir / "targets")
    else:
        output_path = pipeline_cfg["output"]["dir"]

    builders[name](target_cfg, fabric_path, output_path)


@main.command()
@click.option(
    "--fabric",
    "-f",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the HRU fabric GeoPackage.",
)
@click.option(
    "--id-col",
    default="nhm_id",
    show_default=True,
    help="HRU ID column name in the fabric.",
)
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Pipeline config to copy into the run workspace. "
    "Defaults to config/pipeline.yml in the repo root.",
)
@click.option(
    "--workdir",
    "-w",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Root directory for run workspaces. "
    "Defaults to runs/ relative to the current directory. "
    "Can be outside the repository for large data volumes.",
)
@click.option(
    "--id",
    "run_label",
    default=None,
    help="Short label embedded in the run ID (e.g. 'gfv11'). "
    "Produces: 2026-03-11T1500_gfv11_v0.1.0",
)
@click.option(
    "--buffer",
    default=0.1,
    show_default=True,
    type=float,
    help="Degrees to buffer the fabric bounding box for source downloads.",
)
def init(
    fabric: Path,
    id_col: str,
    config: Path | None,
    workdir: Path | None,
    run_label: str | None,
    buffer: float,
):
    """Initialise a new run workspace tied to a specific fabric.

    Creates a dated directory under WORKDIR containing a folder skeleton,
    a snapshot of the pipeline config, a credentials template, and a
    fabric metadata file (path, bounding box, sha256). No data is
    downloaded and no validation is performed at this stage.

    Example
    -------
      nhf-targets init --fabric /data/gfv1.1_fabric.gpkg --id gfv11 --workdir /data/nhf-runs
      # creates: /data/nhf-runs/2026-03-11T1500_gfv11_v0.1.0/
    """
    from nhf_spatial_targets.init_run import init_run
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()

    config_path = config or _DEFAULT_CONFIG
    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            "Pass --config to specify a different path."
        )

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
        raise click.ClickException(str(e)) from e

    msg = Text()
    msg.append("Run workspace created:\n", style="bold green")
    msg.append(f"  {run_dir}\n\n")
    msg.append("Next steps:\n", style="bold")
    msg.append(f"  1. Edit   {run_dir / 'config.yml'}\n")
    msg.append(f"  2. Fill   {run_dir / '.credentials.yml'}\n")
    msg.append(f"  3. Run    nhf-targets run --run-dir {run_dir}\n")
    console.print(Panel(msg, title="nhf-targets init", border_style="green"))


@main.group()
def fetch():
    """Download source datasets into a run workspace."""


@fetch.command("merra2")
@click.option(
    "--run-dir",
    "-r",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Run workspace created by 'nhf-targets init'.",
)
@click.option(
    "--period",
    "-p",
    required=True,
    help="Temporal range as 'YYYY/YYYY' (start/end years inclusive).",
)
def fetch_merra2_cmd(run_dir: Path, period: str):
    """Download MERRA-2 monthly land surface data (M2TMNXLND).

    Authenticates via earthaccess, downloads granules subsetted to the
    fabric bounding box, and prints the provenance record.

    Example
    -------
      nhf-targets fetch merra2 --run-dir /data/runs/2026-03-11T1500_v0.1.0 --period 2010/2010
    """
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.merra2 import fetch_merra2

    console = Console()
    console.print(f"[bold]Fetching MERRA-2 for period {period}...[/bold]")

    result = fetch_merra2(run_dir=run_dir, period=period)

    console.print(
        f"[green]Downloaded {len(result['files'])} files "
        f"to {run_dir / 'data' / 'raw' / 'merra2'}[/green]"
    )
    console.print(json_mod.dumps(result, indent=2))


@main.group()
def catalog():
    """Inspect the data source catalog."""


@catalog.command("sources")
def catalog_sources():
    """List all registered data sources."""
    from nhf_spatial_targets.catalog import sources
    from rich import print as rprint

    rprint(sources())


@catalog.command("variables")
def catalog_variables():
    """List all calibration variable definitions."""
    from nhf_spatial_targets.catalog import variables
    from rich import print as rprint

    rprint(variables())
