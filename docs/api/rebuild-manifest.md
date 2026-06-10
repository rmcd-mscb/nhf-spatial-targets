# Rebuild manifest

`nhf_spatial_targets.rebuild_manifest` regenerates `manifest.json` as a
**deterministic projection** of (on-disk artifacts × current catalog ×
`fabric.json`). It is the authoritative manifest writer; the live capture during
`agg`/`run` is a fast path this projection can always reproduce
(`CLAUDE.md` §Manifest & config durability).

The rebuild path is a pure function of disk: no `datetime.now()` is reachable
from it (every timestamp comes from file `mtime` via
[`lineage.iso_from_mtime`](lineage.md)), JSON ordering is deterministic, and
every regenerated record is tagged `provenance: "reconstructed"`. Same disk +
catalog + code → byte-identical manifest. This module subsumes the former
`reconcile-manifest` (removed in #279).

::: nhf_spatial_targets.rebuild_manifest
    options:
      show_source: true
      heading_level: 2
