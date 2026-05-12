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
    assert var_names == ["ro", "sro", "ssro", "sd"]
    assert s["time_step"] == "hourly (aggregated to daily and monthly)"
    assert s["status"] == "current"


def test_era5_land_sd_is_instantaneous():
    """`sd` (snow depth water equivalent) is instantaneous, not accumulated.

    cell_methods must mark it as a point-in-time field so the consolidation
    pipeline picks the .mean() reducer rather than the diff-of-accumulations
    sum used for ro/sro/ssro.
    """
    s = source("era5_land")
    sd = next(v for v in s["variables"] if v["name"] == "sd")
    assert sd["cell_methods"] == "time: point"
    assert sd["cf_units"] == "m"


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
    assert s["access"]["type"] == "sciencebase_manual"
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


# ---------------------------------------------------------------------------
# SWE category — fetch-layer scaffolding (issue #99)
# ---------------------------------------------------------------------------


def test_snow_water_equivalent_variable_present():
    v = variable("snow_water_equivalent")
    assert v["prms_variable"] == "pkwater_equiv"
    assert v["range_method"] == "multi_source_minmax"
    assert v["normalize"] is False
    assert v["sources"] == ["daymet", "snodas", "era5_land", "margulis_wus_sr"]


def test_daymet_source_present():
    s = source("daymet")
    assert s["status"] == "current"
    assert s["access"]["type"] == "zarr_verify"
    stores = s["access"]["stores"]
    assert set(stores.keys()) == {"na", "hi", "pr"}
    # Stores keyed via the {daymet_root} template so projects can override.
    for region_path in stores.values():
        assert "{daymet_root}" in region_path
    var_names = {v["name"] for v in s["variables"]}
    assert "swe" in var_names
    assert s["doi"] == "10.3334/ORNLDAAC/2129"


def test_snodas_source_present():
    s = source("snodas")
    assert s["status"] == "current"
    assert s["access"]["type"] == "nasa_nsidc"
    assert s["access"]["short_name"] == "G02158"
    var_names = {v["name"] for v in s["variables"]}
    assert "swe" in var_names
    assert s["period"] == "2003/present"


def test_margulis_wus_sr_source_present():
    s = source("margulis_wus_sr")
    assert s["status"] == "current"
    assert s["access"]["type"] == "nasa_nsidc"
    assert s["access"]["short_name"] == "WUS_UCLA_SR"
    assert s["doi"] == "10.5067/PP7T2GBI52I2"
    assert s["period"] == "1985/2021"
    var_names = {v["name"] for v in s["variables"]}
    assert "SWE" in var_names


def test_margulis_wus_sr_fabric_scope_oregon_only():
    """Margulis WUS-SR carries the new optional fabric_scope field."""
    s = source("margulis_wus_sr")
    scope = s["fabric_scope"]
    assert scope["fabrics"] == ["or"]
    assert "notes" in scope


def test_fabric_scope_field_only_on_scoped_sources():
    """No other current source declares fabric_scope.

    Defensive: catches accidental copy/paste of the field onto sources
    that should be available to all fabrics.
    """
    scoped = {key for key, src in sources().items() if "fabric_scope" in src}
    assert scoped == {"margulis_wus_sr"}
