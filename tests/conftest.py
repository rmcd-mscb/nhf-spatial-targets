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

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml


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


def write_year_nc(
    path,
    year: int,
    var: str,
    id_col: str = "nhm_id",
    value: float | None = None,
):
    """Write a synthetic per-year aggregated NC at the given path.

    Used by tests for ``targets/_common.py`` helpers and ``targets/run.py``.
    The file matches the per-year aggregated layout established in PR #51:
    ``<source_key>/<source_key>_<YYYY>_agg.nc`` with ``time`` and ``id_col``
    dims.

    If ``value`` is None, fills with sequential floats; if a number, fills
    constant.
    """
    times = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    hrus = [1, 2, 3]
    if value is None:
        data = np.arange(len(times) * len(hrus), dtype=np.float32).reshape(
            len(times), len(hrus)
        )
    else:
        data = np.full((len(times), len(hrus)), value, dtype=np.float32)
    ds = xr.Dataset(
        {var: (("time", id_col), data)},
        coords={"time": times, id_col: hrus},
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def make_minimal_project(tmp_path, fabric_path: str | None = None):
    """Build a minimal project skeleton for tests of workspace.load() consumers.

    Writes ``config.yml`` with a fake datastore + fabric path and a stub
    ``fabric.json``. Returns the project directory.
    """
    workdir = tmp_path / "proj"
    workdir.mkdir()
    (workdir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "datastore": str(tmp_path / "store"),
                "fabric": {
                    "path": fabric_path or str(tmp_path / "f.gpkg"),
                    "id_col": "nhm_id",
                },
            }
        )
    )
    (workdir / "fabric.json").write_text(json.dumps({"id_col": "nhm_id"}))
    return workdir
