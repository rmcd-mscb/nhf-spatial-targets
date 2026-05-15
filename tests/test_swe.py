"""Tests for the snow water equivalent calibration target."""

from __future__ import annotations

import pytest


def test_swe_build_is_stub():
    """The SWE target builder is a stub until implemented.

    Builder signature is unified across all targets on ``build(project)``;
    a sentinel ``object()`` is enough to reach the ``NotImplementedError``
    without needing a real Project.
    """
    from nhf_spatial_targets.targets.swe import build

    with pytest.raises(NotImplementedError):
        build(object())


def test_swe_variable_lists_four_sources():
    """SWE catalog variable carries all four SWE sources."""
    from nhf_spatial_targets import catalog

    v = catalog.variable("snow_water_equivalent")
    assert set(v["sources"]) == {
        "daymet",
        "snodas",
        "era5_land",
        "margulis_wus_sr",
    }
