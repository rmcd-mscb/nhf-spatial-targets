"""Tests for ERA5-Land aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets.aggregate.era5_land import ADAPTER


def test_adapter_declares_runoff_vars():
    assert ADAPTER.source_key == "era5_land"
    assert ADAPTER.output_name == "era5_land_agg.nc"
    assert set(ADAPTER.variables) == {"ro", "sro", "ssro"}


def test_adapter_uses_monthly_files_glob():
    # Glob carries the `monthly/` subdir component because the era5_land
    # fetch script writes consolidated monthly NCs into <datastore>/
    # era5_land/monthly/ alongside daily/ and hourly/ subdirs.
    assert ADAPTER.files_glob == "monthly/era5_land_monthly_*.nc"
