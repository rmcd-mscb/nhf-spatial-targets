# Lineage

`nhf_spatial_targets.release.lineage` holds the manifest building blocks shared
by every write-site (validate, consolidate, aggregate, target, publish): the
closed set of pipeline `STEP_KINDS`, the step/file record types, the shared
manifest skeleton (`_new_manifest_skeleton` + `atomic_write_manifest`), the
deterministic sort key (`step_sort_key`), and the `mtime`-derived timestamp
helper (`iso_from_mtime`) the [rebuild projection](rebuild-manifest.md) relies
on.

`manifest.json` is written **only** through this module's skeleton — a second
inline writer would recreate the two-skeleton drift #279 removed.

::: nhf_spatial_targets.release.lineage
    options:
      show_source: true
      heading_level: 2
