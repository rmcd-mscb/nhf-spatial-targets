"""Tests for the catalog interface."""

import pytest
from nhf_spatial_targets.catalog import sources, variables, source, variable


def test_sources_loads():
    s = sources()
    assert isinstance(s, dict)
    assert "ssebop" in s


def test_variables_loads():
    v = variables()
    assert isinstance(v, dict)
    assert "runoff" in v


def test_source_lookup():
    s = source("ssebop")
    assert s["status"] == "current"


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


def test_nhm_mwbm_removed():
    """nhm_mwbm has been replaced by ERA5-Land + GLDAS."""
    from nhf_spatial_targets import catalog

    sources = catalog.sources()
    assert "nhm_mwbm" not in sources, (
        "nhm_mwbm should be removed; replaced by era5_land + gldas_noah_v21_monthly"
    )
    aet = catalog.variable("aet")
    assert "nhm_mwbm" not in aet["sources"]


def test_era5_land_source_present():
    from nhf_spatial_targets import catalog

    s = catalog.source("era5_land")
    assert s["access"]["type"] == "copernicus_cds"
    assert s["access"]["dataset"] == "reanalysis-era5-land"
    var_names = [v["name"] for v in s["variables"]]
    assert var_names == ["ro", "sro", "ssro"]
    assert s["time_step"] == "hourly (aggregated to daily and monthly)"
    assert s["status"] == "current"


def test_gldas_source_present():
    from nhf_spatial_targets import catalog

    s = catalog.source("gldas_noah_v21_monthly")
    assert s["access"]["short_name"] == "GLDAS_NOAH025_M"
    assert s["access"]["version"] == "2.1"
    var_names = [v["name"] for v in s["variables"]]
    assert "Qs_acc" in var_names
    assert "Qsb_acc" in var_names
    assert "runoff_total" in var_names  # derived
    assert s["status"] == "current"
