"""nhf-targets command-line interface."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated


import yaml
from cyclopts import App, Parameter

from nhf_spatial_targets._logging import setup_logging
from nhf_spatial_targets.aggregate.era5_land import aggregate_era5_land
from nhf_spatial_targets.aggregate.gldas import aggregate_gldas
from nhf_spatial_targets.aggregate.merra2 import aggregate_merra2
from nhf_spatial_targets.aggregate.mod10c1 import aggregate_mod10c1
from nhf_spatial_targets.aggregate.mod16a2 import aggregate_mod16a2
from nhf_spatial_targets.aggregate.mwbm_climgrid import aggregate_mwbm_climgrid
from nhf_spatial_targets.aggregate.ncep_ncar import aggregate_ncep_ncar
from nhf_spatial_targets.aggregate.nldas_mosaic import aggregate_nldas_mosaic
from nhf_spatial_targets.aggregate.nldas_noah import aggregate_nldas_noah
from nhf_spatial_targets.aggregate.reitz2017 import aggregate_reitz2017
from nhf_spatial_targets.aggregate.watergap22d import aggregate_watergap22d

_logger = logging.getLogger(__name__)

app = App(
    name="nhf-targets",
    help="nhf-spatial-targets: build NHM calibration target datasets.",
    version=_pkg_version("nhf-spatial-targets"),
)
fetch_app = App(name="fetch", help="Download source datasets into a project datastore.")
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

    if name == "runoff":
        # run.build does not use fabric_path (HRU area comes from config)
        builders[name](target_cfg, output_path)
    else:
        builders[name](target_cfg, fabric_path, output_path)


@app.command
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


@app.command(name="materialize-credentials")
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


@app.command
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


@fetch_app.command(name="all")
def fetch_all_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir"],
            help="Project created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
):
    """Download all source datasets into the project datastore.

    Iterates through every registered fetch module in sequence.
    Stops on the first failure.
    """
    from rich.console import Console

    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()

    # Import all fetch functions
    import nhf_spatial_targets.catalog as _catalog
    from nhf_spatial_targets.fetch._period import clamp_period
    from nhf_spatial_targets.fetch.era5_land import fetch_era5_land
    from nhf_spatial_targets.fetch.gldas import fetch_gldas
    from nhf_spatial_targets.fetch.merra2 import fetch_merra2
    from nhf_spatial_targets.fetch.modis import fetch_mod10c1, fetch_mod16a2
    from nhf_spatial_targets.fetch.mwbm_climgrid import fetch_mwbm_climgrid
    from nhf_spatial_targets.fetch.ncep_ncar import fetch_ncep_ncar
    from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic, fetch_nldas_noah
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d
    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017

    # (display name, catalog source key, fetch function)
    sources = [
        ("era5-land", "era5_land", fetch_era5_land),
        ("gldas", "gldas_noah_v21_monthly", fetch_gldas),
        ("merra2", "merra2", fetch_merra2),
        ("nldas-mosaic", "nldas_mosaic", fetch_nldas_mosaic),
        ("nldas-noah", "nldas_noah", fetch_nldas_noah),
        ("ncep-ncar", "ncep_ncar", fetch_ncep_ncar),
        ("mod16a2", "mod16a2_v061", fetch_mod16a2),
        ("mod10c1", "mod10c1_v061", fetch_mod10c1),
        ("watergap22d", "watergap22d", fetch_watergap22d),
        ("reitz2017", "reitz2017", fetch_reitz2017),
        ("mwbm-climgrid", "mwbm_climgrid", fetch_mwbm_climgrid),
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
        except FileNotFoundError as exc:
            # mwbm-climgrid requires a manual (CAPTCHA-gated) download;
            # treat its absence as a skip rather than a fatal error so
            # the rest of the pipeline can still run when the operator
            # hasn't placed the file yet.
            if name == "mwbm-climgrid":
                console.print(
                    f"[yellow]{name}: skipped (manual download not yet "
                    f"placed; see docs/sources/mwbm_climgrid.md)[/yellow]"
                )
                continue
            print(f"Error fetching {name}: {exc}", file=sys.stderr)
            sys.exit(1)
        except (ValueError, RuntimeError) as exc:
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
            name=["--project-dir"],
            help="Project created by 'nhf-targets init'.",
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
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
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
            name=["--project-dir"],
            help="Project created by 'nhf-targets init'.",
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
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
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
            name=["--project-dir"],
            help="Project created by 'nhf-targets init'.",
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
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
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
            name=["--project-dir"],
            help="Project created by 'nhf-targets init'.",
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
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
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
            name=["--project-dir"],
            help="Project created by 'nhf-targets init'.",
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
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
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
            name=["--project-dir"],
            help="Project created by 'nhf-targets init'.",
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
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
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
            name=["--project-dir"],
            help="Project created by 'nhf-targets init'.",
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
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
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


@fetch_app.command(name="era5-land")
def fetch_era5_land_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir"],
            help="Project created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ] = "1979/2024",
    worker_index: Annotated[
        int,
        Parameter(
            name=["--worker-index"],
            help="0-based index of this worker within the pool (default 0). "
            "Set to $SLURM_ARRAY_TASK_ID in array jobs.",
        ),
    ] = 0,
    n_workers: Annotated[
        int,
        Parameter(
            name=["--n-workers"],
            help="Total number of parallel workers (default 1 = serial). "
            "Must match the SLURM array size.",
        ),
    ] = 1,
):
    """Download ERA5-Land hourly runoff (ro, sro, ssro) via CDS API and consolidate to daily/monthly NetCDFs."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.era5_land import fetch_era5_land

    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    if n_workers > 1:
        console.print(
            f"[bold]Fetching ERA5-Land for period {period} "
            f"(worker {worker_index}/{n_workers})...[/bold]"
        )
    else:
        console.print(f"[bold]Fetching ERA5-Land for period {period}...[/bold]")

    try:
        result = fetch_era5_land(
            workdir=workdir,
            period=period,
            worker_index=worker_index,
            n_workers=n_workers,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during ERA5-Land fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print("[green]ERA5-Land: downloaded to datastore[/green]")
    console.print(json_mod.dumps(result, indent=2))


@fetch_app.command(name="gldas")
def fetch_gldas_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir"],
            help="Project created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ] = "2000/2023",
):
    """Download GLDAS-2 monthly runoff data via NASA earthaccess."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.gldas import fetch_gldas

    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching GLDAS for period {period}...[/bold]")

    try:
        result = fetch_gldas(workdir=workdir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during GLDAS fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print("[green]GLDAS: downloaded to datastore[/green]")
    console.print(json_mod.dumps(result, indent=2))


@fetch_app.command(name="reitz2017")
def fetch_reitz2017_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir"],
            help="Project created by 'nhf-targets init'.",
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
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
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


@fetch_app.command(name="mwbm-climgrid")
def fetch_mwbm_climgrid_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir"],
            help="Project created by 'nhf-targets init'.",
        ),
    ],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
):
    """Register a manually-placed USGS MWBM (ClimGrid-forced) NetCDF.

    The ScienceBase distribution is gated by a CAPTCHA, so the ~7.5 GB
    ClimGrid_WBM.nc cannot be retrieved automatically. Download it via
    a browser and place it at <datastore>/mwbm_climgrid/ClimGrid_WBM.nc
    before invoking this command — see docs/sources/mwbm_climgrid.md
    for the procedure. This command then fingerprints the file and
    writes its provenance to manifest.json.
    """
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.mwbm_climgrid import fetch_mwbm_climgrid

    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Registering MWBM ClimGrid (period {period})...[/bold]")

    try:
        result = fetch_mwbm_climgrid(workdir=workdir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during MWBM ClimGrid registration")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print("[green]MWBM ClimGrid: registered in manifest[/green]")
    console.print(json_mod.dumps(result, indent=2))


@agg_app.command(name="ssebop")
def agg_ssebop_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir"],
            help="Project created by 'nhf-targets init'.",
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
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)
    if not (workdir / "fabric.json").exists():
        print(
            f"Error: fabric.json not found in {workdir}. "
            "Run 'nhf-targets validate' first.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Read fabric path and id_col from project config
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


def _run_tier_agg(
    aggregate_fn,
    label: str,
    workdir: Path,
    batch_size: int,
    period: str | None = None,
) -> None:
    """Common boilerplate for tier-1/tier-2 aggregator CLI wrappers.

    ``period`` is forwarded to ``aggregate_fn`` only when set, so
    aggregators that don't accept it (most sources, where fetch already
    clips by file) are unaffected.
    """
    from rich.console import Console

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
        cfg = yaml.safe_load((workdir / "config.yml").read_text())
    except yaml.YAMLError as exc:
        print(f"Error: Cannot parse config.yml: {exc}", file=sys.stderr)
        sys.exit(1)
    fabric_path = cfg["fabric"]["path"]
    id_col = cfg["fabric"].get("id_col", "nhm_id")

    console = Console()
    period_suffix = f", period={period}" if period is not None else ""
    console.print(
        f"[bold]Aggregating {label} (batch_size={batch_size}{period_suffix})...[/bold]"
    )
    try:
        kwargs = {
            "fabric_path": fabric_path,
            "id_col": id_col,
            "workdir": workdir,
            "batch_size": batch_size,
        }
        if period is not None:
            kwargs["period"] = period
        aggregate_fn(**kwargs)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error ({type(exc).__name__}): {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during %s aggregation", label)
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    console.print(
        f"[green]{label} aggregation complete; per-year NCs and manifest "
        f"updated under {workdir}/data/aggregated/[/green]"
    )


@agg_app.command(name="era5-land")
def agg_era5_land_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate ERA5-Land monthly runoff to HRU polygons."""
    _run_tier_agg(aggregate_era5_land, "ERA5-Land", workdir, batch_size)


@agg_app.command(name="gldas")
def agg_gldas_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate GLDAS-2.1 NOAH monthly runoff to HRU polygons."""
    _run_tier_agg(aggregate_gldas, "GLDAS", workdir, batch_size)


@agg_app.command(name="merra2")
def agg_merra2_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate MERRA-2 monthly soil wetness to HRU polygons."""
    _run_tier_agg(aggregate_merra2, "MERRA-2", workdir, batch_size)


@agg_app.command(name="ncep-ncar")
def agg_ncep_ncar_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate NCEP/NCAR monthly soil moisture to HRU polygons."""
    _run_tier_agg(aggregate_ncep_ncar, "NCEP/NCAR", workdir, batch_size)


@agg_app.command(name="nldas-mosaic")
def agg_nldas_mosaic_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate NLDAS-2 MOSAIC monthly soil moisture to HRU polygons."""
    _run_tier_agg(aggregate_nldas_mosaic, "NLDAS-MOSAIC", workdir, batch_size)


@agg_app.command(name="nldas-noah")
def agg_nldas_noah_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate NLDAS-2 NOAH monthly soil moisture to HRU polygons."""
    _run_tier_agg(aggregate_nldas_noah, "NLDAS-NOAH", workdir, batch_size)


@agg_app.command(name="watergap22d")
def agg_watergap22d_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate WaterGAP 2.2d monthly diffuse recharge to HRU polygons."""
    _run_tier_agg(aggregate_watergap22d, "WaterGAP 2.2d", workdir, batch_size)


@agg_app.command(name="reitz2017")
def agg_reitz2017_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate Reitz 2017 annual recharge to HRU polygons."""
    _run_tier_agg(aggregate_reitz2017, "Reitz 2017", workdir, batch_size)


@agg_app.command(name="mod16a2")
def agg_mod16a2_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate MOD16A2 v061 8-day AET to HRU polygons."""
    _run_tier_agg(aggregate_mod16a2, "MOD16A2", workdir, batch_size)


@agg_app.command(name="mod10c1")
def agg_mod10c1_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate MOD10C1 v061 daily SCA to HRU polygons."""
    _run_tier_agg(aggregate_mod10c1, "MOD10C1", workdir, batch_size)


@agg_app.command(name="mwbm-climgrid")
def agg_mwbm_climgrid_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
    period: Annotated[
        str | None,
        Parameter(
            name=["--period", "-p"],
            help=(
                "Optional 'YYYY/YYYY' clip applied to the per-year output. "
                "ClimGrid_WBM.nc spans 1895-2020 in a single file; pass e.g. "
                "'1979/2020' to skip the publisher's spinup years and "
                "anything outside the NHM run window. Omit to aggregate "
                "every year in the file."
            ),
        ),
    ] = None,
):
    """Aggregate USGS MWBM (ClimGrid-forced) monthly outputs to HRU polygons."""
    _run_tier_agg(
        aggregate_mwbm_climgrid,
        "MWBM (ClimGrid)",
        workdir,
        batch_size,
        period=period,
    )


@agg_app.command(name="all")
def agg_all_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir"], help="Project created by 'nhf-targets init'."
        ),
    ],
    batch_size: Annotated[
        int,
        Parameter(name="--batch-size", help="Target HRUs per spatial batch."),
    ] = 500,
):
    """Aggregate every registered source for this project.

    Runs tier-1/tier-2 aggregators in sequence; stops on first failure.
    SSEBop is not included here — run ``agg ssebop --period`` separately.
    """
    from rich.console import Console

    console = Console()
    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    sources: list[tuple[str, Callable[..., None]]] = [
        ("era5-land", aggregate_era5_land),
        ("gldas", aggregate_gldas),
        ("merra2", aggregate_merra2),
        ("ncep-ncar", aggregate_ncep_ncar),
        ("nldas-mosaic", aggregate_nldas_mosaic),
        ("nldas-noah", aggregate_nldas_noah),
        ("watergap22d", aggregate_watergap22d),
        ("reitz2017", aggregate_reitz2017),
        ("mod16a2", aggregate_mod16a2),
        ("mod10c1", aggregate_mod10c1),
        ("mwbm-climgrid", aggregate_mwbm_climgrid),
    ]
    for label, fn in sources:
        console.print(f"\n[bold]{'─' * 60}[/bold]")
        _run_tier_agg(fn, label, workdir, batch_size)

    console.print(
        f"\n[bold green]All {len(sources)} sources aggregated successfully.[/bold green]"
    )


main = app.meta
