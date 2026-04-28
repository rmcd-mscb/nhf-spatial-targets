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


def test_runoff_uses_era5_and_gldas():
    from nhf_spatial_targets import catalog

    v = catalog.variable("runoff")
    assert v["sources"] == ["era5_land", "gldas_noah_v21_monthly", "mwbm_climgrid"]
    assert v["range_method"] == "multi_source_minmax"


def test_recharge_includes_era5_land():
    from nhf_spatial_targets import catalog

    v = catalog.variable("recharge")
    assert "era5_land" in v["sources"]
    assert v["range_method"] == "normalized_minmax"


def test_every_current_source_has_status_field():
    """Every non-superseded source must declare ``status: current``.

    Guards against accidents where status is dropped (e.g., the
    mod10c1_v061 regression where a trailing key got overwritten).
    """
    for key, src in sources().items():
        if src.get("superseded_by") is not None:
            continue
        assert src.get("status") == "current", (
            f"Source {key!r} is missing 'status: current'"
        )


def test_mwbm_climgrid_source():
    """mwbm_climgrid is the modern ClimGrid-forced MWBM (Wieczorek 2024)."""
    s = source("mwbm_climgrid")
    assert s["status"] == "current"
    assert s["doi"] == "10.5066/P9QCLGKM"
    assert s["access"]["type"] == "sciencebase"
    assert s["access"]["item_id"] == "64c948dbd34e70357a34c11e"
    assert s["access"]["filename"] == "ClimGrid_WBM.nc"
    assert s["period"] == "1900/2020"
    assert s["spatial_extent"] == "CONUS"
    var_names = {v["name"] for v in s["variables"]}
    assert var_names == {"runoff", "aet", "soilstorage", "swe"}


def test_runoff_lists_mwbm_climgrid():
    v = variable("runoff")
    assert "mwbm_climgrid" in v["sources"]
    # Existing sources still present
    assert "era5_land" in v["sources"]
    assert "gldas_noah_v21_monthly" in v["sources"]


def test_aet_lists_mwbm_climgrid():
    v = variable("aet")
    assert "mwbm_climgrid" in v["sources"]
    # Existing sources still present
    assert "mod16a2_v061" in v["sources"]
    assert "ssebop" in v["sources"]
