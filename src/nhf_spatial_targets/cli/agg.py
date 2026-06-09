"""``nhf-targets agg`` sub-app: aggregate source datasets to HRU polygons.

The per-source ``aggregate_*`` callables are looked up via the
:mod:`nhf_spatial_targets.cli` package namespace at command-invocation
time (``_resolve_agg_fn``). This indirection keeps existing tests that
patch ``nhf_spatial_targets.cli.aggregate_era5_land`` (etc.) working
after the cli split, without changing the call signatures or behavior
of any aggregator.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import yaml
from cyclopts import App, Parameter

from nhf_spatial_targets.cli._params import (
    _AGG_BATCH_SIZE_PARAM,
    _AGG_N_WORKERS_PARAM,
    _AGG_WORKER_INDEX_PARAM,
)

_logger = logging.getLogger(__name__)

agg_app = App(name="agg", help="Aggregate source datasets to HRU fabric polygons.")


def _resolve_agg_fn(name: str) -> Callable[..., object]:
    """Look up an ``aggregate_*`` callable on the ``cli`` package.

    Routing the lookup through ``nhf_spatial_targets.cli`` rather than
    importing directly from ``nhf_spatial_targets.aggregate.<src>`` is
    what makes ``unittest.mock.patch("nhf_spatial_targets.cli.aggregate_<src>")``
    intercept the call. See ``tests/test_cli_agg.py``.
    """
    from nhf_spatial_targets import cli as _cli

    return getattr(_cli, name)


def _resolve_agg_config(
    workdir: Path, cli_batch_size: int | None
) -> tuple[str, str, int]:
    """Return (fabric_path, id_col, batch_size) from project config.

    ``batch_size`` resolution order: CLI ``--batch-size`` flag (if non-None)
    > ``config.yml[fabric.batch_size]`` > 500 (defaults.py).
    """
    from nhf_spatial_targets.defaults import apply_defaults

    try:
        user_cfg = yaml.safe_load((workdir / "config.yml").read_text())
    except yaml.YAMLError as exc:
        print(f"Error: Cannot parse config.yml: {exc}", file=sys.stderr)
        sys.exit(1)
    cfg = apply_defaults(user_cfg)
    fabric_cfg = cfg.get("fabric") or {}
    fabric_path = fabric_cfg["path"]
    id_col = fabric_cfg.get("id_col", "nhm_id")
    if cli_batch_size is not None:
        batch_size = int(cli_batch_size)
    else:
        batch_size = int(fabric_cfg.get("batch_size", 500))
    return fabric_path, id_col, batch_size


def _run_tier_agg(
    aggregate_fn,
    label: str,
    workdir: Path,
    batch_size: int | None,
    period: str | None = None,
    *,
    worker_index: int = 0,
    n_workers: int = 1,
) -> None:
    """Common boilerplate for tier-1/tier-2 aggregator CLI wrappers.

    ``period`` is forwarded to ``aggregate_fn`` only when set, so
    aggregators that don't accept it (most sources, where fetch already
    clips by file) are unaffected.

    ``batch_size`` of ``None`` means "fall back to ``config.yml``" via
    :func:`_resolve_agg_config`. The CLI per-command default is now
    ``None`` so projects can pin batch_size in their config rather than
    relying on every operator passing ``--batch-size 500`` (issue #156).

    ``worker_index`` / ``n_workers`` enable SLURM-array year-sharding
    (issue #156). Default ``(0, 1)`` is the single-worker serial path.
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

    fabric_path, id_col, resolved_batch_size = _resolve_agg_config(workdir, batch_size)

    console = Console()
    period_suffix = f", period={period}" if period is not None else ""
    worker_suffix = f", worker={worker_index}/{n_workers}" if n_workers > 1 else ""
    console.print(
        f"[bold]Aggregating {label} (batch_size={resolved_batch_size}"
        f"{period_suffix}{worker_suffix})...[/bold]"
    )
    try:
        kwargs = {
            "fabric_path": fabric_path,
            "id_col": id_col,
            "workdir": workdir,
            "batch_size": resolved_batch_size,
            "worker_index": worker_index,
            "n_workers": n_workers,
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
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
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

    fabric_path, id_col, resolved_batch_size = _resolve_agg_config(workdir, batch_size)

    console = Console()
    worker_suffix = f", worker={worker_index}/{n_workers}" if n_workers > 1 else ""
    console.print(
        f"[bold]Aggregating SSEBop AET for period {period} "
        f"(batch_size={resolved_batch_size}{worker_suffix})...[/bold]"
    )

    try:
        aggregate_ssebop(
            fabric_path=fabric_path,
            id_col=id_col,
            period=period,
            workdir=workdir,
            batch_size=resolved_batch_size,
            worker_index=worker_index,
            n_workers=n_workers,
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
        f"[green]SSEBop aggregation complete; per-year NCs and manifest "
        f"updated under {workdir / 'data' / 'aggregated' / 'ssebop'}[/green]"
    )


@agg_app.command(name="daymet")
def agg_daymet_cmd(
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
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
    region: Annotated[
        str,
        Parameter(
            name=["--region", "-r"],
            help=(
                "Daymet region: 'na', 'hi', or 'pr'. Only 'na' is wired "
                "up in this build; 'hi' and 'pr' raise NotImplementedError "
                "until the corresponding fabrics land (issue #101)."
            ),
        ),
    ] = "na",
):
    """Aggregate Daymet V4 R1 daily SWE to HRU fabric polygons.

    Reads the per-region zarr path from manifest.json (written by
    'nhf-targets fetch daymet'), opens the zarr directly, computes
    area-weighted means per HRU, and writes per-year NetCDFs.
    """
    from rich.console import Console

    aggregate_daymet = _resolve_agg_fn("aggregate_daymet")

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

    fabric_path, id_col, resolved_batch_size = _resolve_agg_config(workdir, batch_size)

    console = Console()
    worker_suffix = f", worker={worker_index}/{n_workers}" if n_workers > 1 else ""
    console.print(
        f"[bold]Aggregating Daymet SWE (region={region}, period={period}, "
        f"batch_size={resolved_batch_size}{worker_suffix})...[/bold]"
    )

    try:
        aggregate_daymet(
            fabric_path=fabric_path,
            id_col=id_col,
            period=period,
            workdir=workdir,
            batch_size=resolved_batch_size,
            region=region,
            worker_index=worker_index,
            n_workers=n_workers,
        )
    except (ValueError, FileNotFoundError, RuntimeError, NotImplementedError) as exc:
        print(f"Error ({type(exc).__name__}): {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _logger.exception("Unexpected error during Daymet aggregation")
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    console.print(
        f"[green]Daymet aggregation complete; per-year NCs and manifest "
        f"updated under {workdir / 'data' / 'aggregated' / 'daymet'}[/green]"
    )


@agg_app.command(name="era5-land")
def agg_era5_land_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
):
    """Aggregate ERA5-Land monthly runoff to HRU polygons."""
    _run_tier_agg(
        _resolve_agg_fn("aggregate_era5_land"),
        "ERA5-Land",
        workdir,
        batch_size,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="gldas")
def agg_gldas_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
):
    """Aggregate GLDAS-2.1 NOAH monthly runoff to HRU polygons."""
    _run_tier_agg(
        _resolve_agg_fn("aggregate_gldas"),
        "GLDAS",
        workdir,
        batch_size,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="merra2")
def agg_merra2_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
):
    """Aggregate MERRA-2 monthly soil wetness to HRU polygons."""
    _run_tier_agg(
        _resolve_agg_fn("aggregate_merra2"),
        "MERRA-2",
        workdir,
        batch_size,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="ncep-ncar")
def agg_ncep_ncar_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
):
    """Aggregate NCEP/NCAR monthly soil moisture to HRU polygons."""
    _run_tier_agg(
        _resolve_agg_fn("aggregate_ncep_ncar"),
        "NCEP/NCAR",
        workdir,
        batch_size,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="nldas-mosaic")
def agg_nldas_mosaic_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
):
    """Aggregate NLDAS-2 MOSAIC monthly soil moisture to HRU polygons."""
    _run_tier_agg(
        _resolve_agg_fn("aggregate_nldas_mosaic"),
        "NLDAS-MOSAIC",
        workdir,
        batch_size,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="nldas-noah")
def agg_nldas_noah_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
):
    """Aggregate NLDAS-2 NOAH monthly soil moisture to HRU polygons."""
    _run_tier_agg(
        _resolve_agg_fn("aggregate_nldas_noah"),
        "NLDAS-NOAH",
        workdir,
        batch_size,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="watergap22d")
def agg_watergap22d_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
):
    """Aggregate WaterGAP 2.2d monthly diffuse recharge to HRU polygons."""
    _run_tier_agg(
        _resolve_agg_fn("aggregate_watergap22d"),
        "WaterGAP 2.2d",
        workdir,
        batch_size,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="reitz2017")
def agg_reitz2017_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
):
    """Aggregate Reitz 2017 annual recharge to HRU polygons."""
    _run_tier_agg(
        _resolve_agg_fn("aggregate_reitz2017"),
        "Reitz 2017",
        workdir,
        batch_size,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="mod16a2")
def agg_mod16a2_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
):
    """Aggregate MOD16A2 v061 8-day AET to HRU polygons."""
    _run_tier_agg(
        _resolve_agg_fn("aggregate_mod16a2"),
        "MOD16A2",
        workdir,
        batch_size,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="mod10c1")
def agg_mod10c1_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
):
    """Aggregate MOD10C1 v061 daily SCA to HRU polygons."""
    _run_tier_agg(
        _resolve_agg_fn("aggregate_mod10c1"),
        "MOD10C1",
        workdir,
        batch_size,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="snodas")
def agg_snodas_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
    period: Annotated[
        str | None,
        Parameter(
            name=["--period", "-p"],
            help=(
                "Optional 'YYYY/YYYY' clip applied at agg time. SNODAS "
                "consolidated NCs span 2003-present; pass e.g. '2003/2020' "
                "to restrict aggregation. Omit to aggregate every year "
                "present in the datastore."
            ),
        ),
    ] = None,
):
    """Aggregate SNODAS daily SWE to HRU polygons."""
    _run_tier_agg(
        _resolve_agg_fn("aggregate_snodas"),
        "SNODAS",
        workdir,
        batch_size,
        period=period,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="ua-swe")
def agg_ua_swe_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
    period: Annotated[
        str | None,
        Parameter(
            name=["--period", "-p"],
            help=(
                "Optional 'YYYY/YYYY' clip applied at agg time. UA SWE "
                "consolidated NCs span calendar years 1982-2022; pass e.g. "
                "'2000/2010' to restrict aggregation. Omit to aggregate every "
                "year present in the datastore."
            ),
        ),
    ] = None,
):
    """Aggregate UA daily SWE / snow-depth / snow-covered-fraction to HRU polygons."""
    _run_tier_agg(
        _resolve_agg_fn("aggregate_ua_swe"),
        "UA SWE",
        workdir,
        batch_size,
        period=period,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="era5-land-sd")
def agg_era5_land_sd_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
):
    """Aggregate ERA5-Land daily snow depth water equivalent to HRU polygons.

    Reads from ``<datastore>/era5_land/daily/era5_land_daily_*.nc`` and
    writes to ``<project>/data/aggregated/era5_land_sd/`` so the daily
    SWE outputs stay separate from the monthly runoff aggregations under
    ``era5_land/`` (which the runoff and recharge targets consume).
    """
    _run_tier_agg(
        _resolve_agg_fn("aggregate_era5_land_sd"),
        "ERA5-Land sd",
        workdir,
        batch_size,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="margulis-wus-sr")
def agg_margulis_wus_sr_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
    period: Annotated[
        str | None,
        Parameter(
            name=["--period", "-p"],
            help=(
                "Optional 'YYYY/YYYY' clip applied at agg time. Margulis "
                "consolidated NCs span 1985-2020 in the datastore; pass "
                "e.g. '2003/2020' to restrict aggregation to the SWE "
                "target window. Omit to aggregate every year present."
            ),
        ),
    ] = None,
):
    """Aggregate Margulis WUS-SR daily SWE to HRU polygons (Oregon-scoped)."""
    _run_tier_agg(
        _resolve_agg_fn("aggregate_margulis_wus_sr"),
        "Margulis WUS-SR",
        workdir,
        batch_size,
        period=period,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="mwbm-climgrid")
def agg_mwbm_climgrid_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
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
        _resolve_agg_fn("aggregate_mwbm_climgrid"),
        "MWBM (ClimGrid)",
        workdir,
        batch_size,
        period=period,
        worker_index=worker_index,
        n_workers=n_workers,
    )


@agg_app.command(name="all")
def agg_all_cmd(
    workdir: Annotated[
        Path,
        Parameter(
            name=["--project-dir"], help="Project created by 'nhf-targets init'."
        ),
    ],
    batch_size: Annotated[int | None, _AGG_BATCH_SIZE_PARAM] = None,
    worker_index: Annotated[int, _AGG_WORKER_INDEX_PARAM] = 0,
    n_workers: Annotated[int, _AGG_N_WORKERS_PARAM] = 1,
):
    """Aggregate every registered source for this project.

    Runs tier-1/tier-2 aggregators in sequence; stops on first failure.
    Sources that require an explicit ``--period`` argument are not
    included here — run them separately:

    - ``agg ssebop --period YYYY/YYYY``
    - ``agg daymet --period YYYY/YYYY [--region na]``

    SLURM-array support: when called with ``--n-workers > 1`` every
    source's aggregator gets the same ``worker_index`` / ``n_workers``
    pair. Year sharding is per-source, so all 14 aggregators run on
    this one worker for its slice of each source's years.
    """
    from rich.console import Console

    console = Console()
    if not workdir.exists():
        print(f"Error: Project not found: {workdir}", file=sys.stderr)
        sys.exit(2)

    sources: list[tuple[str, Callable[..., None]]] = [
        ("era5-land", _resolve_agg_fn("aggregate_era5_land")),
        ("gldas", _resolve_agg_fn("aggregate_gldas")),
        ("merra2", _resolve_agg_fn("aggregate_merra2")),
        ("ncep-ncar", _resolve_agg_fn("aggregate_ncep_ncar")),
        ("nldas-mosaic", _resolve_agg_fn("aggregate_nldas_mosaic")),
        ("nldas-noah", _resolve_agg_fn("aggregate_nldas_noah")),
        ("watergap22d", _resolve_agg_fn("aggregate_watergap22d")),
        ("reitz2017", _resolve_agg_fn("aggregate_reitz2017")),
        ("mod16a2", _resolve_agg_fn("aggregate_mod16a2")),
        ("mod10c1", _resolve_agg_fn("aggregate_mod10c1")),
        ("snodas", _resolve_agg_fn("aggregate_snodas")),
        ("ua-swe", _resolve_agg_fn("aggregate_ua_swe")),
        ("mwbm-climgrid", _resolve_agg_fn("aggregate_mwbm_climgrid")),
        ("era5-land-sd", _resolve_agg_fn("aggregate_era5_land_sd")),
        ("margulis-wus-sr", _resolve_agg_fn("aggregate_margulis_wus_sr")),
    ]
    for label, fn in sources:
        console.print(f"\n[bold]{'─' * 60}[/bold]")
        _run_tier_agg(
            fn,
            label,
            workdir,
            batch_size,
            worker_index=worker_index,
            n_workers=n_workers,
        )

    console.print(
        f"\n[bold green]All {len(sources)} sources aggregated successfully.[/bold green]"
    )
