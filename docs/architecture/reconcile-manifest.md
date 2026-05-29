# reconcile-manifest (removed — superseded by `rebuild-manifest`)

> **`nhf-targets reconcile-manifest` no longer exists.** It was removed in
> issue #279 (PR-2) and replaced by `nhf-targets rebuild-manifest`.

## Why it was replaced

`reconcile-manifest` backfilled `manifest.json` from the datastore using a
per-source **hook registry** (`reconcile.py:_RECONCILERS`) that only ever
covered `era5_land` + `mod16a2_v061`; every other source returned `no-hook`.
That seam — the registry — was exactly the gap that left real manifests
holed.

`rebuild-manifest` replaces it with **one generic, catalog-keyed deterministic
projection**: `manifest.json` is regenerated as a pure function of
(datastore consolidated dirs × catalog) ∪ project `data/aggregated/` dirs ∪
`targets/` NCs ∪ `fabric.json`. No per-source hooks, so it covers every
source uniformly, and it is byte-identical on re-run.

## What to run instead

```bash
nhf-targets rebuild-manifest --project-dir <dir> [--compute-sha256] [--dry-run]
```

- **Sources** = the (datastore ∩ catalog) ∪ aggregated-dirs union. A dir whose
  name is not a catalog key (e.g. `era5_land_sd`) is recorded with
  `derived_variant: true` so a shipped NC is never orphaned.
- **Steps** = deterministic `consolidate` / `aggregate` / `target` / `validate`.
- **Identity fields** (`created_utc`, the `fabric` authorship block, any
  `release` block) are read-merged, never re-minted; `last_validated_utc` is
  not re-minted.
- **Honesty:** every regenerated record is tagged `provenance: "reconstructed"`.

See the design spec at
[docs/superpowers/specs/2026-05-29-durable-manifest-and-config-design.md](../superpowers/specs/2026-05-29-durable-manifest-and-config-design.md)
(Pillar 2) for the full rationale.
