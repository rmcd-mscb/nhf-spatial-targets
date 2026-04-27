"""Unit tests for notebooks/inspect_aggregated/_helpers.py.

The helper module lives outside the package, so we load it via importlib
rather than a regular import.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPERS_PATH = REPO_ROOT / "notebooks" / "inspect_aggregated" / "_helpers.py"


@pytest.fixture(scope="session")
def helpers():
    spec = importlib.util.spec_from_file_location(
        "inspect_aggregated_helpers", HELPERS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_loads_with_default_save_figures_off(helpers):
    assert helpers.SAVE_FIGURES is False
    assert helpers.FIGURES_DIR == Path("docs/figures/inspect_aggregated/")


def test_unit_from_catalog_dict_variables(helpers):
    # gldas_noah_v21_monthly has dict-form variables with cf_units
    units = helpers.unit_from_catalog("gldas_noah_v21_monthly", "Qs_acc")
    assert units == "kg m-2"


def test_unit_from_catalog_flat_variables(helpers):
    # ssebop has a flat variables list and a source-level units field
    units = helpers.unit_from_catalog("ssebop", "actual_et")
    assert units == "mm/month"


def test_unit_from_catalog_unknown_variable_raises(helpers):
    with pytest.raises(KeyError):
        helpers.unit_from_catalog("ssebop", "nonexistent_var")


def test_unit_from_catalog_file_variable_lookup(helpers):
    # reitz2017: name=total_recharge, file_variable=TotalRecharge
    # Caller passes the on-disk variable name; should resolve via file_variable path
    units = helpers.unit_from_catalog("reitz2017", "TotalRecharge")
    assert units == "m yr-1"


def test_unit_from_catalog_cf_units_takes_precedence(helpers):
    # reitz2017 total_recharge: cf_units="m yr-1", units="m/year" — cf_units wins
    units = helpers.unit_from_catalog("reitz2017", "total_recharge")
    assert units == "m yr-1"
