"""nhf-targets command-line interface."""

from __future__ import annotations

import logging
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated

import yaml
from cyclopts import App, Parameter

from nhf_spatial_targets._logging import setup_logging

_logger = logging.getLogger(__name__)

app = App(
    name="nhf-targets",
    help="nhf-spatial-targets: build NHM calibration target datasets.",
    version=_pkg_version("nhf-spatial-targets"),
)
fetch_app = App(name="fetch", help="Download source datasets into a run workspace.")
agg_app = App(name="agg", help="Aggregate source datasets to HRU fabric polygons.")
catalog_app = App(name="catalog", help="Inspect the data source catalog.")
app.command(fetch_app)
app.command(agg_app)
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
    workdir: Annotated[
        Path,
        Parameter(
            name=["--workdir", "-w"],
            help="Workspace created by 'nhf-targets init'.",
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
    if not workdir.exists():
        print(f"Error: Workspace not found: {workdir}", file=sys.stderr)
        sys.exit(2)
    if not (workdir / "fabric.json").exists():
        print(
            f"Error: fabric.json not found in {workdir}. "
            "Run 'nhf-targets validate' first.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        cfg = yaml.safe_load((workdir / "config.yml").read_text())
    except yaml.YAMLError as exc:
        print(f"Error: Cannot parse config.yml: {exc}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(cfg, dict):
        print("Error: config.yml is empty or malformed.", file=sys.stderr)
        sys.exit(1)

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
        try:
            _dispatch(name, targets_cfg[name], cfg, workdir=workdir)
        except Exception as exc:
            _logger.exception("Error building target '%s'", name)
            print(f"Error building target '{name}': {exc}", file=sys.stderr)
            sys.exit(1)


def _dispatch(
    name: str,
    target_cfg: dict,
    pipeline_cfg: dict,
    workdir: Path | None = None,
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
    if workdir is not None:
        output_path = str(workdir / "targets")
    else:
        output_path = pipeline_cfg["output"]["dir"]

    builders[name](target_cfg, fabric_path, output_path)


@app.command
def init(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--workdir", "-w"],
            help="Directory to create as the new workspace.",
        ),
    ],
):
    """Initialise a new workspace with a config template.

    Creates a directory skeleton with config.yml and .credentials.yml.
    Edit those files, then run 'nhf-targets validate --workdir <dir>'.
    """
    from nhf_spatial_targets.init_run import init_workspace
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()

    try:
        result = init_workspace(workdir)
    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    msg = Text()
    msg.append("Workspace created:\n", style="bold green")
    msg.append(f"  {result}\n\n")
    msg.append("Next steps:\n", style="bold")
    msg.append(f"  1. Edit   {result / 'config.yml'}\n")
    msg.append(f"  2. Fill   {result / '.credentials.yml'}\n")
    msg.append(f"  3. Run    nhf-targets validate --workdir {result}\n")
    console.print(Panel(msg, title="nhf-targets init", border_style="green"))


@app.command
def validate(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--workdir", "-w"],
            help="Workspace directory to validate.",
        ),
    ],
):
    """Validate a workspace: check config, fabric, credentials, and catalog.

    On success, writes fabric.json and manifest.json into the workspace.
    """
    from rich.console import Console

    from nhf_spatial_targets.validate import validate_workspace

    console = Console()

    if not workdir.exists():
        print(f"Error: Workspace not found: {workdir}", file=sys.stderr)
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

    console.print(
        f"[bold green]Workspace validated successfully:[/bold green] {workdir}"
    )


@fetch_app.command(name="all")
def fetch_all_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--workdir", "-w"],
            help="Workspace created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
):
    """Download all source datasets into the workspace datastore.

    Iterates through every registered fetch module in sequence.
    Stops on the first failure.
    """
    from rich.console import Console

    if not workdir.exists():
        print(f"Error: Workspace not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()

    # Import all fetch functions
    import nhf_spatial_targets.catalog as _catalog
    from nhf_spatial_targets.fetch._period import clamp_period
    from nhf_spatial_targets.fetch.merra2 import fetch_merra2
    from nhf_spatial_targets.fetch.modis import fetch_mod10c1, fetch_mod16a2
    from nhf_spatial_targets.fetch.ncep_ncar import fetch_ncep_ncar
    from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic, fetch_nldas_noah
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d
    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017

    # (display name, catalog source key, fetch function)
    sources = [
        ("merra2", "merra2", fetch_merra2),
        ("nldas-mosaic", "nldas_mosaic", fetch_nldas_mosaic),
        ("nldas-noah", "nldas_noah", fetch_nldas_noah),
        ("ncep-ncar", "ncep_ncar", fetch_ncep_ncar),
        ("mod16a2", "mod16a2_v061", fetch_mod16a2),
        ("mod10c1", "mod10c1_v061", fetch_mod10c1),
        ("watergap22d", "watergap22d", fetch_watergap22d),
        ("reitz2017", "reitz2017", fetch_reitz2017),
    ]

    results = {}
    for name, source_key, fetch_fn in sources:
        console.print(f"\n[bold]{'─' * 60}[/bold]")

        # Clamp requested period to the source's available range
        meta = _catalog.source(source_key)
        available = meta.get("period", "")
        clamped = clamp_period(period, available) if available else period

        if clamped is None:
            console.print(
                f"[yellow]{name}: skipped (no overlap between "
                f"requested {period} and available {available})[/yellow]"
            )
            continue

        if clamped != period:
            console.print(
                f"[bold]Fetching {name} for period {clamped} "
                f"(clamped from {period} to available {available})...[/bold]"
            )
        else:
            console.print(f"[bold]Fetching {name} for period {period}...[/bold]")

        try:
            result = fetch_fn(workdir=workdir, period=clamped)
            results[name] = result
            console.print(f"[green]{name}: downloaded to datastore[/green]")
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            print(f"Error fetching {name}: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            _logger.exception("Unexpected error during %s fetch", name)
            print(
                f"Unexpected error fetching {name} ({type(exc).__name__}): {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    console.print(
        f"\n[bold green]All {len(results)} sources fetched successfully.[/bold green]"
    )


@fetch_app.command(name="merra2")
def fetch_merra2_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--workdir", "-w"],
            help="Workspace created by 'nhf-targets init'.",
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

    if not workdir.exists():
        print(f"Error: Workspace not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching MERRA-2 for period {period}...[/bold]")

    try:
        result = fetch_merra2(workdir=workdir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during MERRA-2 fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print("[green]MERRA-2: downloaded to datastore[/green]")
    console.print(json_mod.dumps(result, indent=2))


@fetch_app.command(name="nldas-mosaic")
def fetch_nldas_mosaic_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--workdir", "-w"],
            help="Workspace created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
):
    """Download NLDAS-2 MOSAIC soil moisture data."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic

    if not workdir.exists():
        print(f"Error: Workspace not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching NLDAS-2 MOSAIC for period {period}...[/bold]")

    try:
        result = fetch_nldas_mosaic(workdir=workdir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during NLDAS-2 MOSAIC fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print("[green]NLDAS-2 MOSAIC: downloaded to datastore[/green]")
    console.print(json_mod.dumps(result, indent=2))


@fetch_app.command(name="nldas-noah")
def fetch_nldas_noah_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--workdir", "-w"],
            help="Workspace created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
):
    """Download NLDAS-2 NOAH soil moisture data."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.nldas import fetch_nldas_noah

    if not workdir.exists():
        print(f"Error: Workspace not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching NLDAS-2 NOAH for period {period}...[/bold]")

    try:
        result = fetch_nldas_noah(workdir=workdir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during NLDAS-2 NOAH fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print("[green]NLDAS-2 NOAH: downloaded to datastore[/green]")
    console.print(json_mod.dumps(result, indent=2))


@fetch_app.command(name="ncep-ncar")
def fetch_ncep_ncar_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--workdir", "-w"],
            help="Workspace created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
):
    """Download NCEP/NCAR Reanalysis soil moisture data."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.ncep_ncar import fetch_ncep_ncar

    if not workdir.exists():
        print(f"Error: Workspace not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching NCEP/NCAR Reanalysis for period {period}...[/bold]")

    try:
        result = fetch_ncep_ncar(workdir=workdir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during NCEP/NCAR fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print("[green]NCEP/NCAR: downloaded to datastore[/green]")
    console.print(json_mod.dumps(result, indent=2))


@fetch_app.command(name="mod16a2")
def fetch_mod16a2_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--workdir", "-w"],
            help="Workspace created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
):
    """Download MODIS MOD16A2 v061 AET data (8-day composites, 500m)."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    if not workdir.exists():
        print(f"Error: Workspace not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching MOD16A2 v061 for period {period}...[/bold]")

    try:
        result = fetch_mod16a2(workdir=workdir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during MOD16A2 fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print("[green]MOD16A2: downloaded to datastore[/green]")
    console.print(json_mod.dumps(result, indent=2))


@fetch_app.command(name="mod10c1")
def fetch_mod10c1_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--workdir", "-w"],
            help="Workspace created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
):
    """Download MODIS MOD10C1 v061 daily snow cover data (0.05deg CMG)."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.modis import fetch_mod10c1

    if not workdir.exists():
        print(f"Error: Workspace not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching MOD10C1 v061 for period {period}...[/bold]")

    try:
        result = fetch_mod10c1(workdir=workdir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during MOD10C1 fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print("[green]MOD10C1: downloaded to datastore[/green]")
    console.print(json_mod.dumps(result, indent=2))


@fetch_app.command(name="watergap22d")
def fetch_watergap22d_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--workdir", "-w"],
            help="Workspace created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
):
    """Download WaterGAP 2.2d groundwater recharge from PANGAEA."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d

    if not workdir.exists():
        print(f"Error: Workspace not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching WaterGAP 2.2d for period {period}...[/bold]")

    try:
        result = fetch_watergap22d(workdir=workdir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during WaterGAP 2.2d fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print("[green]WaterGAP 2.2d: downloaded to datastore[/green]")
    console.print(json_mod.dumps(result, indent=2))


@fetch_app.command(name="reitz2017")
def fetch_reitz2017_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--workdir", "-w"],
            help="Workspace created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
):
    """Download Reitz 2017 annual recharge estimates from USGS ScienceBase."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017

    if not workdir.exists():
        print(f"Error: Workspace not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching Reitz 2017 recharge for period {period}...[/bold]")

    try:
        result = fetch_reitz2017(workdir=workdir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during Reitz 2017 fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print("[green]Reitz 2017: downloaded to datastore[/green]")
    console.print(json_mod.dumps(result, indent=2))


@agg_app.command(name="ssebop")
def agg_ssebop_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--workdir", "-w"],
            help="Workspace created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
    batch_size: Annotated[
        int,
        Parameter(name="--batch-size", help="Target HRUs per spatial batch."),
    ] = 500,
):
    """Aggregate SSEBop monthly AET to HRU fabric polygons.

    Reads SSEBop data from the USGS NHGF STAC catalog (Zarr), computes
    area-weighted means per HRU, and writes the result to NetCDF.
    """
    from rich.console import Console

    from nhf_spatial_targets.aggregate.ssebop import aggregate_ssebop

    if not workdir.exists():
        print(f"Error: Workspace not found: {workdir}", file=sys.stderr)
        sys.exit(2)
    if not (workdir / "fabric.json").exists():
        print(
            f"Error: fabric.json not found in {workdir}. "
            "Run 'nhf-targets validate' first.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Read fabric path and id_col from workspace config
    try:
        cfg = yaml.safe_load((workdir / "config.yml").read_text())
    except yaml.YAMLError as exc:
        print(f"Error: Cannot parse config.yml: {exc}", file=sys.stderr)
        sys.exit(1)
    fabric_path = cfg["fabric"]["path"]
    id_col = cfg["fabric"].get("id_col", "nhm_id")

    console = Console()
    console.print(
        f"[bold]Aggregating SSEBop AET for period {period} "
        f"(batch_size={batch_size})...[/bold]"
    )

    try:
        ds = aggregate_ssebop(
            fabric_path=fabric_path,
            id_col=id_col,
            period=period,
            workdir=workdir,
            batch_size=batch_size,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during SSEBop aggregation")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print(
        f"[green]SSEBop aggregation complete: "
        f"{ds.sizes.get('time', '?')} time steps x "
        f"{ds.sizes.get(id_col, '?')} HRUs[/green]"
    )
    console.print(
        f"[green]Output: {workdir / 'data' / 'aggregated' / 'ssebop_agg_aet.nc'}[/green]"
    )


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
