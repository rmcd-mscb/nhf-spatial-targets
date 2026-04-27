"""Tests for Reitz 2017 aggregation adapter."""

from __future__ import annotations

import fnmatch

from nhf_spatial_targets.aggregate.reitz2017 import ADAPTER


def test_adapter_declares_recharge_variables():
    assert ADAPTER.source_key == "reitz2017"
    assert ADAPTER.output_name == "reitz2017_agg.nc"
    assert ADAPTER.variables == ("total_recharge", "eff_recharge")
    assert ADAPTER.source_crs == "EPSG:4269"


def test_adapter_files_glob_matches_fetch_output():
    """Lock the adapter's files_glob to the fetch module's filename.

    fetch/reitz2017.py writes 'reitz2017_consolidated.nc'. If a future
    refactor renames the fetch output without updating the adapter, the
    glob would silently zero-file and the aggregator would raise
    FileNotFoundError at runtime. Pin the contract here.
    """
    assert fnmatch.fnmatch("reitz2017_consolidated.nc", ADAPTER.files_glob)
