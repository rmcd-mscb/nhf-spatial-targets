"""Shared pytest fixtures for unit tests.

The preflight system-file checks (_check_cdsapirc, _check_netrc_earthdata)
require ~/.cdsapirc and ~/.netrc to exist on the developer's machine.  Unit
tests that call validate_workspace() use the ``no_system_cred_checks`` fixture
to suppress these checks so that CI and dev machines without those files can
still run the full unit suite.

Tests that exercise the check functions themselves call the functions directly
with the ``_home`` parameter rather than relying on module-level patching.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture()
def no_system_cred_checks():
    """Suppress ~/.cdsapirc and ~/.netrc existence checks (for validate_workspace tests)."""
    with (
        patch(
            "nhf_spatial_targets.validate._check_cdsapirc",
            return_value=None,
        ),
        patch(
            "nhf_spatial_targets.validate._check_netrc_earthdata",
            return_value=None,
        ),
    ):
        yield
