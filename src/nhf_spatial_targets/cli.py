"""nhf-targets command-line interface."""

from __future__ import annotations

import click
import yaml
from pathlib import Path


@click.group()
def main():
    """nhf-spatial-targets: build NHM calibration target datasets."""


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Path to pipeline.yml config file.")
@click.option("--target", "-t", default=None,
              help="Run a single target (default: all enabled targets).")
def run(config: str, target: str | None):
    """Run the calibration target pipeline."""
    cfg = yaml.safe_load(Path(config).read_text())
    targets_cfg = cfg.get("targets", {})

    to_run = [target] if target else [k for k, v in targets_cfg.items()
                                      if v.get("enabled", False)]

    for name in to_run:
        if name not in targets_cfg:
            raise click.ClickException(f"Unknown target: {name}")
        click.echo(f"Building target: {name}")
        _dispatch(name, targets_cfg[name], cfg)


def _dispatch(name: str, target_cfg: dict, pipeline_cfg: dict) -> None:
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

    fabric_path = pipeline_cfg["fabric"]["path"]
    output_path = pipeline_cfg["output"]["dir"]
    builders[name](target_cfg, fabric_path, output_path)


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
