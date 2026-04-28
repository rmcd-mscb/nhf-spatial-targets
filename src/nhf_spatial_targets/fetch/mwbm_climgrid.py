"""Fetch USGS MWBM (ClimGrid-forced) monthly outputs from ScienceBase.

Single ~7.5 GB CF-conformant NetCDF (ClimGrid_WBM.nc); the fetch is
purely a download — no consolidation step. sha256 + size are persisted
in manifest.json for idempotency and corruption detection.
"""

from __future__ import annotations

import hashlib  # noqa: F401 — used in Tasks 5-8
import json  # noqa: F401 — used in Tasks 5-8
import logging
import os  # noqa: F401 — used in Tasks 5-8
import tempfile  # noqa: F401 — used in Tasks 5-8
from datetime import datetime, timezone  # noqa: F401 — used in Tasks 5-8
from pathlib import Path

import xarray as xr  # noqa: F401 — used in Tasks 5-8

import nhf_spatial_targets.catalog as _catalog  # noqa: F401 — used in Tasks 5-8
from nhf_spatial_targets.fetch._period import parse_period, years_in_period
from nhf_spatial_targets.workspace import load as _load_project  # noqa: F401 — used in Tasks 5-8

logger = logging.getLogger(__name__)

_SOURCE_KEY = "mwbm_climgrid"
_DATA_PERIOD = (1900, 2020)  # publisher's usable window; 1895-1899 is spinup


def fetch_mwbm_climgrid(workdir: Path, period: str) -> dict:
    """Download ClimGrid_WBM.nc to <datastore>/mwbm_climgrid/.

    Idempotent: skips download if the file is present AND its size +
    sha256 match the values recorded in manifest.json. Computes sha256
    streaming during download (no second-pass read of the 7.5 GB file).
    Validates expected variables and CF metadata after download.

    Parameters
    ----------
    workdir : Path
        Project directory.
    period : str
        Temporal range as "YYYY/YYYY" — used to validate the project's
        intended use against publisher coverage and to record in the
        manifest entry. The download itself ignores this argument
        (the publisher distributes one file).

    Returns
    -------
    dict
        Provenance record for manifest.json.
    """
    parse_period(period)
    requested_years = years_in_period(period)
    for y in requested_years:
        if y < _DATA_PERIOD[0] or y > _DATA_PERIOD[1]:
            raise ValueError(
                f"Year {y} is outside the MWBM-ClimGrid data range "
                f"({_DATA_PERIOD[0]}-{_DATA_PERIOD[1]}). The 1895-1899 "
                f"period is publisher-flagged spinup. Adjust --period."
            )

    raise NotImplementedError(  # download path lands in Task 5
        "fetch_mwbm_climgrid download path not yet implemented"
    )
