"""Tests for SNODAS daily SWE aggregation adapter."""

from __future__ import annotations

import numpy as np
import xarray as xr

from nhf_spatial_targets import catalog as cat
from nhf_spatial_targets.aggregate.snodas import ADAPTER, aggregate_snodas


def test_adapter_declares_swe_only():
    assert ADAPTER.source_key == "snodas"
    assert ADAPTER.output_name == "snodas_agg.nc"
    assert ADAPTER.variables == ("swe",)


def test_adapter_targets_consolidated_daily_subdir():
    """SNODAS consolidated NCs live under <datastore>/snodas/daily/."""
    assert ADAPTER.files_glob == "daily/snodas_daily_*.nc"


def test_adapter_uses_epsg5070_and_masked_mean():
    """SNODAS is pre-projected to EPSG:5070 at consolidate time (#121),
    matching the driver's WEIGHT_GEN_CRS so gdptools does not reproject.
    ``stat_method="masked_mean"`` (#151) treats the SNODAS CONUS mask as
    a deliberate per-pixel mask (justified by the pre_aggregate_hook
    below), so HRUs whose footprint touches a single fill pixel are no
    longer poisoned to NaN. See CLAUDE.md "Aggregation Transformation
    Policy" → ``stat_method`` choice."""
    assert ADAPTER.source_crs == "EPSG:5070"
    assert ADAPTER.stat_method == "masked_mean"


def test_adapter_has_pre_aggregate_hook():
    """A pre_aggregate_hook is required to justify masked_mean per the
    transformation policy: the hook documents the deliberate per-pixel
    fill mask. post_aggregate_hook is unused — no diagnostic emitted."""
    assert ADAPTER.pre_aggregate_hook is not None
    assert ADAPTER.post_aggregate_hook is None


def test_pre_aggregate_hook_masks_fill_values():
    """The hook converts SNODAS ``-9999`` (and similar fill codes) to NaN.

    xarray's mask_and_scale already does this on read, but the hook
    re-asserts it so downstream gdptools sees float NaN regardless of
    how the source NC was opened. Values strictly greater than -9990
    survive untouched.
    """
    ds = xr.Dataset(
        {
            "swe": (
                ("time", "y", "x"),
                np.array([[[100.0, -9999.0], [-9999.0, 50.0]]], dtype=np.float32),
            ),
        }
    )
    out = ADAPTER.pre_aggregate_hook(ds)
    masked = out["swe"].values
    assert masked[0, 0, 0] == 100.0
    assert masked[0, 1, 1] == 50.0
    assert np.isnan(masked[0, 0, 1])
    assert np.isnan(masked[0, 1, 0])


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
