"""Back-compat re-export of the decomposed target-builder helpers.

This module used to be the 1,557-line single file containing every shared
target-builder helper. It has been broken up by concern (PR #219) into:

- ``_shims``         — :class:`SourceShim` and the by-key / by-label /
                       validate_source_units indexers.
- ``_io``            — :func:`read_aggregated_source`,
                       :func:`reindex_to_month_start` /
                       :func:`reindex_to_day_start`,
                       :func:`compute_hru_*`, :func:`parse_period`,
                       :func:`check_hru_coords`.
- ``_combine``       — :func:`multi_source_nanminmax`,
                       :func:`build_n_sources_attrs`.
- ``_writers``       — :func:`write_target_nc`,
                       :func:`write_bounds_target`.
- ``_intermediates`` — :func:`iter_period_years`,
                       :func:`target_config_fingerprint`,
                       :func:`code_version_fingerprint`,
                       :func:`should_skip_year_build`,
                       :func:`prune_orphan_year_intermediates`,
                       :func:`stitch_year_chunks_to_target`, and the
                       ``INTERMEDIATE_*_ATTR`` constants.

This module re-exports every public (and a handful of private) symbol
so existing imports continue to work unchanged::

    from nhf_spatial_targets.targets._common import read_aggregated_source
    from nhf_spatial_targets.targets._common import (
        SourceShim, multi_source_nanminmax, write_bounds_target,
    )

Prefer the focused submodules for new code; this aggregator exists
purely to keep PR #219 a refactor (no test changes).
"""

from __future__ import annotations

from nhf_spatial_targets.targets._combine import (
    build_n_sources_attrs,
    multi_source_nanminmax,
)
from nhf_spatial_targets.targets._intermediates import (
    INTERMEDIATE_CODE_VERSION_ATTR,
    INTERMEDIATE_CONFIG_FINGERPRINT_ATTR,
    code_version_fingerprint,
    iter_period_years,
    prune_orphan_year_intermediates,
    should_skip_year_build,
    stitch_year_chunks_to_target,
    target_config_fingerprint,
)
from nhf_spatial_targets.targets._io import (
    _load_and_reproject_fabric,
    _read_chunk_hru,
    check_hru_coords,
    compute_hru_area_and_centroids,
    compute_hru_areas,
    compute_hru_centroids,
    parse_period,
    read_aggregated_source,
    reindex_to_day_start,
    reindex_to_month_start,
)
from nhf_spatial_targets.targets._shims import (
    SourceShim,
    shims_by_config_label,
    shims_by_key,
    validate_source_units,
)
from nhf_spatial_targets.targets._writers import (
    _target_encoding_without_chunks,
    write_bounds_target,
    write_target_nc,
)

__all__ = [
    # _shims
    "SourceShim",
    "shims_by_config_label",
    "shims_by_key",
    "validate_source_units",
    # _io
    "_read_chunk_hru",
    "_load_and_reproject_fabric",
    "check_hru_coords",
    "compute_hru_area_and_centroids",
    "compute_hru_areas",
    "compute_hru_centroids",
    "parse_period",
    "read_aggregated_source",
    "reindex_to_day_start",
    "reindex_to_month_start",
    # _combine
    "build_n_sources_attrs",
    "multi_source_nanminmax",
    # _writers
    "_target_encoding_without_chunks",
    "write_bounds_target",
    "write_target_nc",
    # _intermediates
    "INTERMEDIATE_CODE_VERSION_ATTR",
    "INTERMEDIATE_CONFIG_FINGERPRINT_ATTR",
    "code_version_fingerprint",
    "iter_period_years",
    "prune_orphan_year_intermediates",
    "should_skip_year_build",
    "stitch_year_chunks_to_target",
    "target_config_fingerprint",
]
