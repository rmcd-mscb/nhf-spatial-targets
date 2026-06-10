# Consolidate (CF metadata)

`nhf_spatial_targets.fetch.consolidate` is the **single entry point for CF-1.6
metadata** on every NetCDF the pipeline writes. `apply_cf_metadata` sets
`Conventions=CF-1.6`, variable `units` / `long_name` / `cell_methods` /
`grid_mapping`, coordinate `standard_name` / `units` / `axis`, and the WGS84
`crs` ancillary variable — all read from `catalog/sources.yml` so a unit
correction in the catalog flows through every NC on the next consolidate
(`CLAUDE.md` §Data & Catalog Conventions).

Per-source fetch/consolidate modules call this; none set CF attributes by hand.
Each has a test asserting the output NC carries the required CF-1.6 attribute
set.

::: nhf_spatial_targets.fetch.consolidate
    options:
      show_source: true
      heading_level: 2
