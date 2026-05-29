"""nhf-targets command-line interface.

The CLI is split across per-stage submodules:

- ``cli.fetch``    — ``fetch`` sub-app (downloads to the datastore)
- ``cli.agg``      — ``agg`` sub-app (HRU aggregation)
- ``cli.catalog``  — ``catalog`` sub-app (registry inspection)
- ``cli.run``      — root-level commands (init / validate / run / rechunk
  / rebuild-manifest / upgrade-config / materialize-credentials)
- ``cli._params``  — shared ``Annotated[..., Parameter(...)]`` aliases for
  the agg sub-app

The package re-exports ``app`` (the root cyclopts ``App``), ``main``
(``app.meta``, the ``pyproject.toml`` script entry), ``_dispatch``,
``setup_logging``, ``materialize_credentials_cmd``, and every
``aggregate_*`` callable. Tests rely on those names being present at the
``nhf_spatial_targets.cli`` namespace so ``unittest.mock.patch(...)``
intercepts continue to work after the split.
"""

from __future__ import annotations

from importlib.metadata import version as _pkg_version
from typing import Annotated

from cyclopts import App, Parameter

from nhf_spatial_targets._logging import setup_logging

# Aggregate callables are re-exported so tests can do
# ``patch("nhf_spatial_targets.cli.aggregate_<src>", ...)``. The agg
# sub-app looks them up by name via ``cli._resolve_agg_fn`` so the patch
# intercepts the call.
from nhf_spatial_targets.aggregate.daymet import aggregate_daymet
from nhf_spatial_targets.aggregate.era5_land import (
    aggregate_era5_land,
    aggregate_era5_land_sd,
)
from nhf_spatial_targets.aggregate.gldas import aggregate_gldas
from nhf_spatial_targets.aggregate.margulis_wus_sr import aggregate_margulis_wus_sr
from nhf_spatial_targets.aggregate.merra2 import aggregate_merra2
from nhf_spatial_targets.aggregate.mod10c1 import aggregate_mod10c1
from nhf_spatial_targets.aggregate.mod16a2 import aggregate_mod16a2
from nhf_spatial_targets.aggregate.mwbm_climgrid import aggregate_mwbm_climgrid
from nhf_spatial_targets.aggregate.ncep_ncar import aggregate_ncep_ncar
from nhf_spatial_targets.aggregate.nldas_mosaic import aggregate_nldas_mosaic
from nhf_spatial_targets.aggregate.nldas_noah import aggregate_nldas_noah
from nhf_spatial_targets.aggregate.reitz2017 import aggregate_reitz2017
from nhf_spatial_targets.aggregate.snodas import aggregate_snodas
from nhf_spatial_targets.aggregate.watergap22d import aggregate_watergap22d

app = App(
    name="nhf-targets",
    help="nhf-spatial-targets: build NHM calibration target datasets.",
    version=_pkg_version("nhf-spatial-targets"),
)


@app.meta.default
def launcher(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    verbose: Annotated[bool, Parameter(name=["--verbose", "-v"])] = False,
):
    """Global options for nhf-targets."""
    setup_logging(verbose)
    app(tokens)  # dispatch remaining tokens to the root app


# Wire the per-stage sub-apps. Import after ``app`` exists because each
# submodule pulls Parameter aliases / shared helpers from this package.
from nhf_spatial_targets.cli.agg import agg_app  # noqa: E402
from nhf_spatial_targets.cli.catalog import catalog_app  # noqa: E402
from nhf_spatial_targets.cli.fetch import fetch_app  # noqa: E402
from nhf_spatial_targets.cli.release import release_app  # noqa: E402
from nhf_spatial_targets.cli.run import (  # noqa: E402
    _dispatch,
    init,
    materialize_credentials_cmd,
    rebuild_manifest_cmd,
    rechunk,
    register as _register_run_commands,
    run,
    upgrade_config_cmd,
    upgrade_manifest_cmd,
    validate,
)

app.command(fetch_app)
app.command(agg_app)
app.command(catalog_app)
app.command(release_app)
_register_run_commands(app)


# Script entry point: ``pyproject.toml`` has
# ``nhf-targets = "nhf_spatial_targets.cli:main"``.
main = app.meta


__all__ = [
    "app",
    "main",
    "setup_logging",
    "_dispatch",
    # root-level command callables (kept importable for tests)
    "init",
    "validate",
    "run",
    "rechunk",
    "rebuild_manifest_cmd",
    "upgrade_config_cmd",
    "upgrade_manifest_cmd",
    "materialize_credentials_cmd",
    # aggregate callables (test-patch targets)
    "aggregate_daymet",
    "aggregate_era5_land",
    "aggregate_era5_land_sd",
    "aggregate_gldas",
    "aggregate_margulis_wus_sr",
    "aggregate_merra2",
    "aggregate_mod10c1",
    "aggregate_mod16a2",
    "aggregate_mwbm_climgrid",
    "aggregate_ncep_ncar",
    "aggregate_nldas_mosaic",
    "aggregate_nldas_noah",
    "aggregate_reitz2017",
    "aggregate_snodas",
    "aggregate_watergap22d",
]
