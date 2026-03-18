"""Tests for the catalog interface."""

import pytest
from nhf_spatial_targets.catalog import sources, variables, source, variable


def test_sources_loads():
    s = sources()
    assert isinstance(s, dict)
    assert "nhm_mwbm" in s


def test_variables_loads():
    v = variables()
    assert isinstance(v, dict)
    assert "runoff" in v


def test_source_lookup():
    s = source("nhm_mwbm")
    assert s["time_step"] == "monthly"


def test_variable_lookup():
    v = variable("aet")
    assert "mod16a2_v061" in v["sources"]


def test_ssebop_source_is_usgs_gdp_stac():
    s = source("ssebop")
    assert s["access"]["type"] == "usgs_gdp_stac"
    assert s["access"]["collection_id"] == "ssebopeta_monthly"
    assert s["status"] == "current"


def test_source_missing():
    with pytest.raises(KeyError):
        source("not_a_real_source")
