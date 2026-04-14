"""gdptools-based spatial aggregation to HRU fabric."""

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import update_manifest

__all__ = ["SourceAdapter", "update_manifest"]
