"""Build snow water equivalent (SWE) calibration targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nhf_spatial_targets.workspace import Project

# Sources:  daymet, snodas, era5_land, margulis_wus_sr
# Method:   multi_source_minmax (per-HRU per-day nanmin/nanmax across sources)
# Variable: pkwater_equiv
# Timestep: daily


def build(project: "Project") -> None:
    """Build SWE target dataset."""
    raise NotImplementedError
