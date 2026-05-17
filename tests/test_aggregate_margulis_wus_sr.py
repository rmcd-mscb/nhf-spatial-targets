"""Tests for the Margulis WUS-SR daily SWE aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets import catalog as cat
from nhf_spatial_targets.aggregate.margulis_wus_sr import (
    ADAPTER,
    aggregate_margulis_wus_sr,
)


def test_adapter_declares_swe_only():
    assert ADAPTER.source_key == "margulis_wus_sr"
    assert ADAPTER.output_name == "margulis_wus_sr_agg.nc"
    assert ADAPTER.variables == ("SWE",)


def test_adapter_uses_daily_files_glob():
    """Consolidated NCs land under <datastore>/margulis_wus_sr/daily/."""
    assert ADAPTER.files_glob == "daily/margulis_wus_sr_daily_*.nc"


def test_adapter_wgs84_and_plain_mean():
    """Consolidator reprojects native EASE-Grid to regular WGS84 lat/lon;
    no per-pixel mask in a pre_aggregate_hook means default ``mean`` is
    correct (same as SNODAS)."""
    assert ADAPTER.source_crs == "EPSG:4326"
    assert ADAPTER.stat_method == "mean"


def test_adapter_has_no_hooks():
    """Per-pixel NaN over WUS-domain edges decodes via mask_and_scale;
    no pre/post hook needed."""
    assert ADAPTER.pre_aggregate_hook is None
    assert ADAPTER.post_aggregate_hook is None


def test_adapter_variable_matches_catalog():
    """The adapter's declared SWE var must appear in catalog/sources.yml
    so the cf attrs applied by the driver line up with the consolidated NCs.
    """
    entry = cat.source("margulis_wus_sr")
    catalog_vars = {v["name"] for v in entry["variables"]}
    assert set(ADAPTER.variables).issubset(catalog_vars)


def test_aggregate_margulis_is_callable_with_period_kwarg():
    """CLI dispatcher forwards --period via kwargs; the function must
    accept it for the _run_tier_agg(period=...) path."""
    import inspect

    sig = inspect.signature(aggregate_margulis_wus_sr)
    assert "period" in sig.parameters
    assert sig.parameters["period"].default is None


def test_cli_registers_agg_margulis():
    """``nhf-targets agg margulis-wus-sr`` must be wired in cli.py."""
    from nhf_spatial_targets import cli

    assert cli.aggregate_margulis_wus_sr is aggregate_margulis_wus_sr


def test_catalog_declares_fabric_scope_oregon():
    """Target builder relies on catalog fabric_scope to keep Margulis
    out of non-OR fabrics; if the catalog scope is removed/widened the
    target builder default behaviour will silently change."""
    entry = cat.source("margulis_wus_sr")
    scope = entry.get("fabric_scope")
    assert scope is not None
    cat.validate_fabric_scope("margulis_wus_sr", scope)
    assert "or" in scope["fabrics"]
