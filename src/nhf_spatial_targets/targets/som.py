"""Build soil moisture calibration targets from MERRA, NCEP, NLDAS."""

# Sources:  merra_land (or merra2), ncep_ncar, nldas_mosaic, nldas_noah
# Method:   normalized_minmax (per calendar month for monthly)
# Variable: soil_rechr
# Timestep: monthly and annual

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nhf_spatial_targets.workspace import Project


def build(project: "Project") -> None:
    """Build soil moisture target dataset."""
    raise NotImplementedError
