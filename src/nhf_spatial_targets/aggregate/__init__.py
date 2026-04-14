"""gdptools-based spatial aggregation to HRU fabric."""

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import (
    aggregate_source,
    update_manifest,
)

__all__ = ["SourceAdapter", "aggregate_source", "update_manifest"]
