"""Build recharge calibration targets.

Sources (per catalog/variables.yml):
  - reitz2017: total_recharge (in/yr)
  - watergap22d: groundwater_recharge (kg m-2 s-1, → mm/yr)
  - era5_land: ssro summed monthly→annual (m water-eq, → mm/yr)

Method: each source aggregated to HRU, normalized 0-1 per HRU over
2000-2009, then per-HRU per-year lower/upper = min/max across the three
normalized sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nhf_spatial_targets.workspace import Project


def build(project: "Project") -> None:
    """Build recharge target dataset."""
    raise NotImplementedError
