"""Tests for SNODAS daily SWE aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets import catalog as cat
from nhf_spatial_targets.aggregate.snodas import ADAPTER, aggregate_snodas


def test_adapter_declares_swe_only():
    assert ADAPTER.source_key == "snodas"
    assert ADAPTER.output_name == "snodas_agg.nc"
    assert ADAPTER.variables == ("swe",)


def test_adapter_targets_consolidated_daily_subdir():
    """SNODAS consolidated NCs live under <datastore>/snodas/daily/."""
    assert ADAPTER.files_glob == "daily/snodas_daily_*.nc"


def test_adapter_uses_wgs84_and_plain_mean():
    """SNODAS arrives without per-pixel masking (fills decode to NaN via
    mask_and_scale); ``mean`` is correct for honest geometric NaN at the
    HRU level. See CLAUDE.md "Aggregation Transformation Policy"."""
    assert ADAPTER.source_crs == "EPSG:4326"
    assert ADAPTER.stat_method == "mean"


def test_adapter_has_no_hooks():
    """Per-pixel decoding is xarray's job; no pre/post hook needed."""
    assert ADAPTER.pre_aggregate_hook is None
    assert ADAPTER.post_aggregate_hook is None


def test_adapter_variable_matches_catalog():
    """The adapter's declared swe var must appear in catalog/sources.yml
    so the cf attrs applied by the driver line up with the consolidated NCs."""
    entry = cat.source("snodas")
    catalog_vars = {v["name"] for v in entry["variables"]}
    assert set(ADAPTER.variables).issubset(catalog_vars)


def test_aggregate_snodas_is_callable_with_period_kwarg():
    """The CLI dispatcher forwards --period via kwargs; the function must
    accept it for the _run_tier_agg(period=...) path."""
    import inspect

    sig = inspect.signature(aggregate_snodas)
    assert "period" in sig.parameters
    # Default None so omitting --period aggregates every year on disk.
    assert sig.parameters["period"].default is None


def test_cli_registers_agg_snodas():
    """``nhf-targets agg snodas`` must be wired in cli.py.

    Checks the imported aggregate function rather than the cyclopts
    command registry (which is private) — this catches any future
    accidental removal of the import.
    """
    from nhf_spatial_targets import cli

    assert cli.aggregate_snodas is aggregate_snodas
