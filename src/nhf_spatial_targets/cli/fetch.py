"""``nhf-targets fetch`` sub-app: download source datasets to the datastore."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from nhf_spatial_targets.cli._params import _PROJECT_DIR_PARAM

_logger = logging.getLogger(__name__)

# Exit code for a run that completed but produced incomplete data — some
# years / calendar-years failed to consolidate or hit download errors while
# the rest succeeded (issue #299). Distinct from 1 (hard error) and 2 (bad
# invocation) so SLURM array jobs and scripts can treat "partial" separately
# from a clean failure. Reserved for *genuine* trouble (n_failed / n_errors);
# skip-only runs (expected archive gaps) stay exit 0 — see _emit_fetch_banner.
EXIT_PARTIAL = 3

fetch_app = App(name="fetch", help="Download source datasets into a project datastore.")


def _emit_fetch_banner(console, display_name: str, summary: object) -> bool:
    """Print a green-success / yellow-partial banner for one fetch result.

    Reads the optional ``n_consolidated`` / ``n_failed`` / ``n_skipped`` /
    ``n_errors`` rollup that the per-year / per-calendar-year fetchers
    (``ua_swe``, ``snodas``) attach to their summary dict. Fetchers that
    download in a single shot omit the rollup, so the banner stays green for
    them and this helper is safe to call uniformly across every command.

    A run is *partial* whenever any of ``n_failed`` / ``n_skipped`` /
    ``n_errors`` is non-zero — the banner turns yellow in that case so the
    operator does not read an unconditional green "downloaded" line over an
    incomplete datastore. Only *genuine* trouble (``n_failed`` or
    ``n_errors``) is treated as actionable: that is what this function
    returns, and what the caller maps to :data:`EXIT_PARTIAL`. Skip-only
    runs (e.g. expected SNODAS archive day-gaps, all-404 years) stay green
    on exit code but still surface as a yellow banner.

    Returns
    -------
    bool
        ``True`` when the run carries genuine failures (``n_failed`` or
        ``n_errors`` > 0) and the caller should exit :data:`EXIT_PARTIAL`.
    """
    rollup = summary if isinstance(summary, dict) else {}
    n_failed = int(rollup.get("n_failed", 0) or 0)
    n_skipped = int(rollup.get("n_skipped", 0) or 0)
    n_errors = int(rollup.get("n_errors", 0) or 0)

    if not (n_failed or n_skipped or n_errors):
        console.print(f"[green]{display_name}: downloaded to datastore[/green]")
        return False

    n_consolidated = int(rollup.get("n_consolidated", 0) or 0)
    parts = [f"{n_consolidated} consolidated"]
    if n_failed:
        parts.append(f"{n_failed} failed")
    if n_skipped:
        parts.append(f"{n_skipped} skipped")
    if n_errors:
        parts.append(f"{n_errors} download errors")
    console.print(
        f"[yellow]{display_name}: {', '.join(parts)} — inspect the per-record "
        f"diagnostics (consolidate_error / wy_status / n_errors, "
        f"source-dependent) in the JSON summary below (also recorded in "
        f"manifest.json).[/yellow]"
    )
    return bool(n_failed or n_errors)


@fetch_app.command(name="all")
def fetch_all_cmd(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
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
    from nhf_spatial_targets.fetch.daymet import fetch_daymet
    from nhf_spatial_targets.fetch.era5_land import fetch_era5_land
    from nhf_spatial_targets.fetch.gldas import fetch_gldas
    from nhf_spatial_targets.fetch.margulis_wus_sr import fetch_margulis_wus_sr
    from nhf_spatial_targets.fetch.merra2 import fetch_merra2
    from nhf_spatial_targets.fetch.modis import fetch_mod10c1, fetch_mod16a2
    from nhf_spatial_targets.fetch.mwbm_climgrid import fetch_mwbm_climgrid
    from nhf_spatial_targets.fetch.ncep_ncar import fetch_ncep_ncar
    from nhf_spatial_targets.fetch.nldas import fetch_nldas_mosaic, fetch_nldas_noah
    from nhf_spatial_targets.fetch.pangaea import fetch_watergap22d
    from nhf_spatial_targets.fetch.reitz2017 import fetch_reitz2017
    from nhf_spatial_targets.fetch.snodas import fetch_snodas
    from nhf_spatial_targets.fetch.ua_swe import fetch_ua_swe

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
        # SWE category (issue #99) — fetch scaffolding only; aggregate +
        # target wiring deferred to per-source follow-up issues.
        ("daymet", "daymet", fetch_daymet),
        ("snodas", "snodas", fetch_snodas),
        ("margulis-wus-sr", "margulis_wus_sr", fetch_margulis_wus_sr),
        ("ua-swe", "ua_swe", fetch_ua_swe),
    ]

    results = {}
    partial_sources: list[str] = []
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
            if _emit_fetch_banner(console, name, result):
                partial_sources.append(name)
        except FileNotFoundError as exc:
            # Manual-placement / operator-staged sources can legitimately
            # be missing on a freshly-bootstrapped project — skip them
            # rather than fail the whole `fetch all` run so the rest of
            # the pipeline can proceed. Operators inspect the per-source
            # skip lines and stage the missing inputs as needed.
            #   - mwbm-climgrid: CAPTCHA-gated ScienceBase download.
            #   - daymet: pre-staged zarr root not yet configured.
            #   - margulis-wus-sr: Western-US coverage; legitimately absent
            #     when the project fabric's bbox doesn't reach the WUS domain.
            manual_skip = {"mwbm-climgrid", "daymet", "margulis-wus-sr"}
            if name in manual_skip:
                console.print(
                    f"[yellow]{name}: skipped (manual download not yet "
                    f"placed; see docs/sources/{name.replace('-', '_')}.md)[/yellow]"
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

    if partial_sources:
        console.print(
            f"\n[bold yellow]{len(results)} sources fetched, but "
            f"{len(partial_sources)} produced incomplete data: "
            f"{', '.join(partial_sources)}. Inspect each source's JSON "
            f"summary / manifest.json and re-run the affected fetch.[/bold yellow]"
        )
        sys.exit(EXIT_PARTIAL)

    console.print(
        f"\n[bold green]All {len(results)} sources fetched successfully.[/bold green]"
    )


@fetch_app.command(name="merra2")
def fetch_merra2_cmd(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
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
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
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
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
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
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
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
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
    force: Annotated[
        bool,
        Parameter(
            name=["--force"],
            help=(
                "Re-fetch every year in --period, ignoring the manifest's "
                "year-skip. Use after a pipeline change has invalidated the "
                "existing consolidated NCs (e.g. PR #88's fill-mask fix). The "
                "manifest entry for mod16a2_v061 is overwritten when the run "
                "completes."
            ),
        ),
    ] = False,
):
    """Download MODIS MOD16A2 v061 AET data (8-day composites, 500m)."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    if force:
        console.print(
            f"[bold]Force re-fetching MOD16A2 v061 for period {period}...[/bold]"
        )
    else:
        console.print(f"[bold]Fetching MOD16A2 v061 for period {period}...[/bold]")

    try:
        result = fetch_mod16a2(workdir=workdir, period=period, force=force)
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
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ],
    force: Annotated[
        bool,
        Parameter(
            name=["--force"],
            help=(
                "Re-fetch every year in --period, ignoring the manifest's "
                "year-skip. The manifest entry for mod10c1_v061 is "
                "overwritten when the run completes."
            ),
        ),
    ] = False,
):
    """Download MODIS MOD10C1 v061 daily snow cover data (0.05deg CMG)."""
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.modis import fetch_mod10c1

    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    if force:
        console.print(
            f"[bold]Force re-fetching MOD10C1 v061 for period {period}...[/bold]"
        )
    else:
        console.print(f"[bold]Fetching MOD10C1 v061 for period {period}...[/bold]")

    try:
        result = fetch_mod10c1(workdir=workdir, period=period, force=force)
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
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
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
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
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
    """Download ERA5-Land hourly fields (ro, sro, ssro, sd) via CDS API and consolidate to daily/monthly NetCDFs.

    The runoff variables (ro/sro/ssro) are accumulated and aggregated as
    daily/monthly sums; sd (snow depth water equivalent) is instantaneous
    and aggregated as daily/monthly means.
    """
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
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
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
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
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
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
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


# ---------------------------------------------------------------------------
# SWE category (issue #99) — fetch scaffolding
# Aggregate + target wiring deferred to per-source follow-up issues.
# ---------------------------------------------------------------------------


@fetch_app.command(name="daymet")
def fetch_daymet_cmd(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ] = "1980/2024",
    source_path: Annotated[
        Path | None,
        Parameter(
            name=["--source-path"],
            help="Directory containing daymet_{na,hi,pr}.zarr. Overrides "
            "'daymet_root' from config.yml.",
        ),
    ] = None,
    region: Annotated[
        str,
        Parameter(
            name=["--region"],
            help="One of 'na' | 'hi' | 'pr' | 'all' (default: all).",
        ),
    ] = "all",
):
    """Register operator-staged Daymet V4 R1 regional zarr stores.

    Daymet zarrs are pre-staged on a shared filesystem; this command
    fingerprints the structural metadata of each region and records a
    per-region manifest entry. No downloading or copying. See
    docs/sources/daymet.md for the staging procedure.
    """
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.daymet import fetch_daymet

    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(
        f"[bold]Registering Daymet (region={region}, period {period})...[/bold]"
    )

    try:
        result = fetch_daymet(
            workdir=workdir,
            period=period,
            source_path=source_path,
            region=region,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during Daymet registration")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print("[green]Daymet: registered in manifest[/green]")
    console.print(json_mod.dumps(result, indent=2))


@fetch_app.command(name="snodas")
def fetch_snodas_cmd(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ] = "2003/2024",
    worker_index: Annotated[
        int,
        Parameter(
            name=["--worker-index"],
            help="0-based index of this worker within the pool (default 0).",
        ),
    ] = 0,
    n_workers: Annotated[
        int,
        Parameter(
            name=["--n-workers"],
            help="Total number of parallel workers (default 1 = serial).",
        ),
    ] = 1,
):
    """Download SNODAS daily SWE granules from NSIDC (G02158) via earthaccess.

    Fetch-only: raw tar/.Hdr bundles land in <datastore>/snodas/raw/<year>/.
    Decoding into CF NetCDFs is deferred to the SNODAS aggregate follow-up.
    """
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.snodas import fetch_snodas

    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    if n_workers > 1:
        console.print(
            f"[bold]Fetching SNODAS for period {period} "
            f"(worker {worker_index}/{n_workers})...[/bold]"
        )
    else:
        console.print(f"[bold]Fetching SNODAS for period {period}...[/bold]")

    try:
        result = fetch_snodas(
            workdir=workdir,
            period=period,
            worker_index=worker_index,
            n_workers=n_workers,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during SNODAS fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    partial = _emit_fetch_banner(console, "SNODAS", result)
    console.print(json_mod.dumps(result, indent=2))
    if partial:
        sys.exit(EXIT_PARTIAL)


@fetch_app.command(name="margulis-wus-sr")
def fetch_margulis_wus_sr_cmd(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ] = "1985/2021",
):
    """Download Margulis Western US Snow Reanalysis (WUS_UCLA_SR) via earthaccess.

    The CMR search bbox is the project fabric's buffered bbox (the
    source itself covers the Western US). Fetch-only:
    consolidation is deferred to the Margulis aggregate follow-up.
    """
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.margulis_wus_sr import fetch_margulis_wus_sr

    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    console.print(f"[bold]Fetching Margulis WUS-SR for period {period}...[/bold]")

    try:
        result = fetch_margulis_wus_sr(workdir=workdir, period=period)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during Margulis WUS-SR fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print("[green]Margulis WUS-SR: downloaded to datastore[/green]")
    console.print(json_mod.dumps(result, indent=2))


@fetch_app.command(name="ua-swe")
def fetch_ua_swe_cmd(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
    period: Annotated[
        str,
        Parameter(name=["--period", "-p"], help="Temporal range as 'YYYY/YYYY'."),
    ] = "1981/2023",
    worker_index: Annotated[
        int,
        Parameter(
            name=["--worker-index"],
            help="0-based index of this worker within the pool (default 0).",
        ),
    ] = 0,
    n_workers: Annotated[
        int,
        Parameter(
            name=["--n-workers"],
            help="Total number of parallel workers (default 1 = serial).",
        ),
    ] = 1,
):
    """Download UA daily 4-km SWE + snow depth (NSIDC-0719) from NSIDC.

    Per-WY NCs are fetched from the NSIDC HTTPS archive via the
    earthaccess auth session, then assembled into CF-1.6 calendar-year
    NCs pre-projected to EPSG:5070 at
    ``<datastore>/ua_swe/daily/ua_swe_daily_<YYYY>.nc``. Raw downloads
    are preserved under ``<datastore>/ua_swe/raw/``.
    """
    import json as json_mod

    from rich.console import Console

    from nhf_spatial_targets.fetch.ua_swe import fetch_ua_swe

    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    console = Console()
    if n_workers > 1:
        console.print(
            f"[bold]Fetching UA SWE for period {period} "
            f"(worker {worker_index}/{n_workers})...[/bold]"
        )
    else:
        console.print(f"[bold]Fetching UA SWE for period {period}...[/bold]")

    try:
        result = fetch_ua_swe(
            workdir=workdir,
            period=period,
            worker_index=worker_index,
            n_workers=n_workers,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during UA SWE fetch")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    partial = _emit_fetch_banner(console, "UA SWE", result)
    console.print(json_mod.dumps(result, indent=2))
    if partial:
        sys.exit(EXIT_PARTIAL)
