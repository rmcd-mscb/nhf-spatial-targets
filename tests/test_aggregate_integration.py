"""Integration tests for tier-1/tier-2 aggregators against a real datastore.

Skipped by default. Run with ``pixi run -e dev test-integration``. These
tests document the end-to-end contract; they are placeholders until a
fixture datastore and small CONUS fabric subset are checked in.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skip(reason="fixture datastore + mini-fabric not yet checked in")
def test_aggregate_era5_land_end_to_end():
    """aggregate_era5_land writes a 3-var NC with (time, hru_id) dims."""
    raise NotImplementedError


@pytest.mark.skip(reason="fixture datastore + mini-fabric not yet checked in")
def test_aggregate_gldas_end_to_end():
    """aggregate_gldas writes Qs_acc, Qsb_acc, runoff_total."""
    raise NotImplementedError


@pytest.mark.skip(reason="fixture datastore + mini-fabric not yet checked in")
def test_aggregate_merra2_end_to_end():
    raise NotImplementedError


@pytest.mark.skip(reason="fixture datastore + mini-fabric not yet checked in")
def test_aggregate_ncep_ncar_end_to_end():
    raise NotImplementedError


@pytest.mark.skip(reason="fixture datastore + mini-fabric not yet checked in")
def test_aggregate_nldas_mosaic_end_to_end():
    raise NotImplementedError


@pytest.mark.skip(reason="fixture datastore + mini-fabric not yet checked in")
def test_aggregate_nldas_noah_end_to_end():
    raise NotImplementedError


@pytest.mark.skip(reason="fixture datastore + mini-fabric not yet checked in")
def test_aggregate_watergap22d_end_to_end():
    raise NotImplementedError


@pytest.mark.skip(reason="fixture datastore + mini-fabric not yet checked in")
def test_aggregate_mod16a2_end_to_end():
    raise NotImplementedError


@pytest.mark.skip(reason="fixture datastore + mini-fabric not yet checked in")
def test_aggregate_mod10c1_end_to_end():
    """Should emit a warning when >10% of cells have zero valid area."""
    raise NotImplementedError
