"""``nhf-targets catalog`` sub-app: inspect the data source catalog."""

from __future__ import annotations

from cyclopts import App

catalog_app = App(name="catalog", help="Inspect the data source catalog.")


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
