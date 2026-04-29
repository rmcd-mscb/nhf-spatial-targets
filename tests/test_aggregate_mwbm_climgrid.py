"""Tests for MWBM ClimGrid-forced aggregation adapter."""

from __future__ import annotations

import fnmatch

from nhf_spatial_targets.aggregate.mwbm_climgrid import ADAPTER


def test_adapter_declares_four_variables():
    assert ADAPTER.source_key == "mwbm_climgrid"
    assert ADAPTER.output_name == "mwbm_climgrid_agg.nc"
    assert ADAPTER.variables == ("runoff", "aet", "soilstorage", "swe")


def test_adapter_files_glob_matches_publisher_filename():
    """Lock the adapter's files_glob to the publisher-issued filename.

    fetch/mwbm_climgrid.py downloads ClimGrid_WBM.nc unchanged. If a
    future refactor renames the datastore copy without updating the
    adapter, the glob would silently zero-file and the aggregator would
    raise FileNotFoundError at runtime. Pin the contract here.
    """
    assert fnmatch.fnmatch("ClimGrid_WBM.nc", ADAPTER.files_glob)
    # Sanity: the default glob does NOT match this filename — confirms
    # we needed the override.
    assert not fnmatch.fnmatch("ClimGrid_WBM.nc", "*_consolidated.nc")
