"""Tests for the project-config defaults layer."""

from __future__ import annotations

from nhf_spatial_targets.defaults import (
    DEFAULTS,
    REQUIRED,
    apply_defaults,
    find_unknown_keys,
    iter_default_diff,
    missing_required,
)


def test_defaults_has_all_five_targets():
    """Every calibration target has a defaults block."""
    assert set(DEFAULTS["targets"].keys()) == {
        "runoff",
        "aet",
        "recharge",
        "soil_moisture",
        "snow_covered_area",
    }


def test_runoff_defaults_include_all_three_sources():
    """Runoff defaults to era5+gldas+mwbm per the design."""
    assert DEFAULTS["targets"]["runoff"]["sources"] == [
        "era5_land",
        "gldas_noah_v21_monthly",
        "mwbm_climgrid",
    ]


def test_runoff_defaults_nn_fill_on_by_default():
    assert DEFAULTS["targets"]["runoff"]["nn_fill"] is True
    assert DEFAULTS["targets"]["runoff"]["nn_max_candidates"] == 10
    assert DEFAULTS["targets"]["runoff"]["chunk_months"] == 12


def test_fabric_area_crs_default_is_conus_albers():
    assert DEFAULTS["fabric"]["area_crs"] == "EPSG:5070"


def test_apply_defaults_fills_missing_leaf_keys():
    """User config wins; missing keys take defaults."""
    user = {"datastore": "/data", "fabric": {"path": "/p", "id_col": "nhm_id"}}
    merged = apply_defaults(user)
    assert merged["datastore"] == "/data"
    assert merged["fabric"]["path"] == "/p"
    assert merged["fabric"]["area_crs"] == "EPSG:5070"  # filled from defaults
    assert merged["targets"]["runoff"]["nn_fill"] is True


def test_apply_defaults_user_wins_at_leaves():
    user = {
        "datastore": "/data",
        "fabric": {"path": "/p", "id_col": "nhm_id", "area_crs": "EPSG:3338"},
    }
    merged = apply_defaults(user)
    assert merged["fabric"]["area_crs"] == "EPSG:3338"


def test_apply_defaults_lists_replace_wholesale():
    """A user-set list is NOT merged with the default list."""
    user = {
        "datastore": "/data",
        "fabric": {"path": "/p", "id_col": "nhm_id"},
        "targets": {"runoff": {"sources": ["era5_land"]}},
    }
    merged = apply_defaults(user)
    assert merged["targets"]["runoff"]["sources"] == ["era5_land"]


def test_iter_default_diff_lists_filled_keys():
    """Defaults that were filled in are reported with dotted paths."""
    user = {"datastore": "/data", "fabric": {"path": "/p", "id_col": "nhm_id"}}
    diff = list(iter_default_diff(user))
    paths = {p for p, _ in diff}
    assert "fabric.area_crs" in paths
    assert "targets.runoff.nn_fill" in paths
    # Keys the user did set should NOT appear:
    assert "datastore" not in paths
    # Yielded values must be the actual defaults (not None sentinel):
    values = dict(diff)
    assert values["fabric.area_crs"] == "EPSG:5070"
    assert values["targets.runoff.nn_fill"] is True
    assert values["targets.runoff.nn_max_candidates"] == 10


def test_iter_default_diff_skips_required_sentinel_none():
    """Keys whose default is None (required-sentinel) are not yielded."""
    user = {}  # nothing set; both required and default keys are absent
    diff = list(iter_default_diff(user))
    paths = {p for p, _ in diff}
    # Real defaults appear:
    assert "fabric.area_crs" in paths
    # Required-sentinel keys (whose default is None) do NOT appear:
    assert "datastore" not in paths
    assert "fabric.path" not in paths
    assert "targets.runoff.period" not in paths
    # And no yielded value is None:
    assert all(v is not None for _, v in diff)


def test_find_unknown_keys_returns_typos():
    """Keys not present in DEFAULTS are reported."""
    user = {
        "datastore": "/data",
        "fabric": {"path": "/p", "id_col": "nhm_id"},
        "targets": {"runoff": {"nn_fil": True}},  # typo
    }
    unknown = find_unknown_keys(user)
    assert "targets.runoff.nn_fil" in unknown


def test_missing_required_reports_paths():
    """Missing required keys return their dotted paths."""
    user = {"fabric": {"id_col": "nhm_id"}}  # no datastore, no fabric.path
    missing = missing_required(apply_defaults(user))
    assert "datastore" in missing
    assert "fabric.path" in missing


def test_missing_required_runoff_period_when_enabled():
    user = {
        "datastore": "/d",
        "fabric": {"path": "/p", "id_col": "nhm_id"},
        # no targets.runoff.period set; runoff.enabled=True by default
    }
    missing = missing_required(apply_defaults(user))
    assert "targets.runoff.period" in missing


def test_missing_required_skips_period_when_disabled():
    user = {
        "datastore": "/d",
        "fabric": {"path": "/p", "id_col": "nhm_id"},
        "targets": {"runoff": {"enabled": False}},
    }
    missing = missing_required(apply_defaults(user))
    assert "targets.runoff.period" not in missing


def test_required_paths_includes_datastore_and_fabric_path():
    """REQUIRED is the canonical list of always-required dotted paths."""
    assert ("datastore",) in REQUIRED
    assert ("fabric", "path") in REQUIRED
