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
    # Required-fields sanity check (matches the existing target-variable
    # schema across runoff/aet/recharge/soil_moisture/sca).
    for field in (
        "description",
        "prms_reference",
        "time_step",
        "period",
        "units",
        "range_notes",
        "output_format",
    ):
        assert field in v, f"snow_water_equivalent missing required field {field!r}"
    # Units rework (C8): post-conversion units alongside original.
    assert v["units"] == "inches"
    assert v.get("original_units")
    assert v["time_step"] == "daily"


def test_daymet_source_present():
    s = source("daymet")
    assert s["status"] == "current"
    assert s["access"]["type"] == "zarr_verify"
    assert s["access"]["url"].startswith("https://")
    stores = s["access"]["stores"]
    assert set(stores.keys()) == {"na", "hi", "pr"}
    # Stores keyed via the {daymet_root} template so projects can override,
    # and the resolved per-region path must end with a `.zarr` directory.
    for r, region_path in stores.items():
        assert "{daymet_root}" in region_path
        assert region_path.endswith(f"daymet_{r}.zarr"), region_path
    var_names = {v["name"] for v in s["variables"]}
    assert "swe" in var_names
    # The SWE variable carries a unit and cell_methods (drift-prevention).
    swe = next(v for v in s["variables"] if v["name"] == "swe")
    assert swe["cf_units"] == "kg m-2"
    assert swe["cell_methods"] == "time: point"
    assert s["doi"] == "10.3334/ORNLDAAC/2129"
    assert s["citations"]
    assert s["units"]
    assert s["spatial_resolution"]


def test_snodas_source_present():
    s = source("snodas")
    assert s["status"] == "current"
    # SNODAS is fetched directly from NSIDC's HTTPS archive (issue #107).
    # The CMR collection record is a granule-less stub, so earthaccess.
    # search_data is not used; the catalog records both the human-facing
    # landing URL and the machine-facing archive_url for the fetcher.
    assert s["access"]["type"] == "nsidc_https"
    assert s["access"]["url"].startswith("https://")
    assert s["access"]["archive_url"].startswith(
        "https://noaadata.apps.nsidc.org/NOAA/G02158/"
    )
    # CMR record exists as a metadata-only stub (zero granules) and is
    # preserved for provenance; the fetch module deliberately does NOT
    # call earthaccess.search_data on it (see #107).
    assert s["access"]["cmr"]["short_name"] == "G02158"
    assert s["access"]["cmr"]["concept_id"] == "C1386246263-NSIDCV0"
    var_names = {v["name"] for v in s["variables"]}
    assert "swe" in var_names
    swe = next(v for v in s["variables"] if v["name"] == "swe")
    assert swe["cf_units"] == "kg m-2"
    assert swe["cell_methods"] == "time: point"
    assert s["period"] == "2003/present"
    assert s["citations"]
    assert s["units"]


def test_margulis_wus_sr_source_present():
    s = source("margulis_wus_sr")
    assert s["status"] == "current"
    # `nsidc` = CMR-mediated path; distinct from SNODAS's `nsidc_https`
    # which uses the granule-less CMR record only as metadata.
    assert s["access"]["type"] == "nsidc"
    assert s["access"]["short_name"] == "WUS_UCLA_SR"
    assert s["access"]["url"].startswith("https://")
    assert s["doi"] == "10.5067/PP7T2GBI52I2"
    assert s["period"] == "1985/2021"
    var_names = {v["name"] for v in s["variables"]}
    assert "SWE" in var_names
    swe = next(v for v in s["variables"] if v["name"] == "SWE")
    assert swe["cf_units"] == "m"
    assert swe["cell_methods"] == "time: point"
    assert s["citations"]
    assert s["units"]
    # N7 — spatial_extent is now a pure source-extent string, not a
    # project-scoping mention. Fabric scoping lives in fabric_scope.
    assert "Oregon-fabric only" not in s["spatial_extent"]


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


def test_every_fabric_scope_block_validates():
    """Every declared `fabric_scope` block passes the validator.

    Catches typos like `fabrics: [oregon]` that would otherwise
    silently disable scoping at the target-builder boundary.
    """
    from nhf_spatial_targets.catalog import validate_fabric_scope

    for key, src in sources().items():
        validate_fabric_scope(key, src.get("fabric_scope"))


def test_validate_fabric_scope_rejects_unknown_token():
    from nhf_spatial_targets.catalog import validate_fabric_scope

    with pytest.raises(ValueError, match="unknown token"):
        validate_fabric_scope("test_src", {"fabrics": ["oregon"]})


def test_validate_fabric_scope_rejects_non_list_fabrics():
    from nhf_spatial_targets.catalog import validate_fabric_scope

    with pytest.raises(ValueError, match="non-empty list"):
        validate_fabric_scope("test_src", {"fabrics": "or"})


def test_validate_fabric_scope_rejects_empty_fabrics():
    from nhf_spatial_targets.catalog import validate_fabric_scope

    with pytest.raises(ValueError, match="non-empty list"):
        validate_fabric_scope("test_src", {"fabrics": []})


def test_validate_fabric_scope_none_is_ok():
    """`fabric_scope: None` (i.e. field absent) is the global default and OK."""
    from nhf_spatial_targets.catalog import validate_fabric_scope

    validate_fabric_scope("test_src", None)


# ---------------------------------------------------------------------------
# source_var_cf_units (issue #130 — target-builder unit-drift guard)
# ---------------------------------------------------------------------------


def test_source_var_cf_units_resolves_dict_form():
    """Standard dict-form variables resolve via the `name` key."""
    from nhf_spatial_targets.catalog import source_var_cf_units

    assert source_var_cf_units("era5_land", "ro") == "m"
    assert source_var_cf_units("gldas_noah_v21_monthly", "runoff_total") == "kg m-2"
    assert source_var_cf_units("mwbm_climgrid", "runoff") == "mm"


def test_source_var_cf_units_resolves_via_file_variable():
    """Reitz 2017 distributes `TotalRecharge` but stores it under `total_recharge`."""
    from nhf_spatial_targets.catalog import source_var_cf_units

    assert source_var_cf_units("reitz2017", "TotalRecharge") == "m yr-1"
    assert source_var_cf_units("reitz2017", "total_recharge") == "m yr-1"


def test_source_var_cf_units_raises_on_unknown_variable():
    from nhf_spatial_targets.catalog import source_var_cf_units

    with pytest.raises(KeyError, match="not found"):
        source_var_cf_units("era5_land", "no_such_var")


def test_source_var_cf_units_raises_on_flat_string_variable():
    """Flat-string entries (kept only on superseded sources) have no cf_units."""
    from nhf_spatial_targets.catalog import source_var_cf_units

    # merra_land is superseded and still uses the original `variables: [SFMC]`
    # flat-list shape. Target builders that point at it would need to add
    # cf_units to the catalog or opt out via expected_cf_units=None.
    with pytest.raises(KeyError, match="not found"):
        source_var_cf_units("merra_land", "SFMC")


def test_ssebop_et_resolves_to_mm():
    """SSEBop migrated from flat-string to dict form with cf_units: mm
    so the AET target builder can validate units at startup (issue #130).
    """
    from nhf_spatial_targets.catalog import source_var_cf_units

    assert source_var_cf_units("ssebop", "et") == "mm"
