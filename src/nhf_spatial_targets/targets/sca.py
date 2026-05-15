"""Build snow-covered area calibration targets from MOD10C1."""

# Sources:  mod10c1 (or v061)
# Method:   modis_ci (CI-based bounds, threshold 70%)
# Variable: snowcov_area
# Timestep: daily

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nhf_spatial_targets.workspace import Project


def build(project: "Project") -> None:
    """Build SCA target dataset."""
    raise NotImplementedError
