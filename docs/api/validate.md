# Validate

`nhf_spatial_targets.validate` runs the preflight checks behind `nhf-targets
validate`: it verifies the fabric, datastore, credentials, and catalog
references, then writes the two derived artifacts a project needs before any
fetch/agg/run — `fabric.json` (computed fabric metadata) and
`config.effective.yml` (the version/hash-stamped projection of `config.yml` ×
`defaults.py` × `fabric.json`).

This is the **config actuator**: there is no `rebuild-config` verb — re-running
`validate` is what regenerates `config.effective.yml` after a `config.yml` edit
(`CLAUDE.md` §Manifest & config durability). The publish gate is staleness-
checked against the hash this module stamps.

::: nhf_spatial_targets.validate
    options:
      show_source: true
      heading_level: 2
