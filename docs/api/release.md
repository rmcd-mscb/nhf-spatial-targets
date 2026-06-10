# Release package

`nhf_spatial_targets.release` is the ScienceBase data-release subsystem behind
the `nhf-targets release` command family. Its `__init__` re-exports the public
surface; the heavy lifting lives in focused submodules:

| Submodule | Responsibility |
|---|---|
| `release.config` | Load + validate the per-project `release.yml`; `scaffold_release_yml` |
| `release.build` | `build_all` — stage files, render metadata, checksum a payload offline |
| `release.checksums` | `compute_checksums` / `verify_csv` — `checksums.csv` + `SHA256SUMS` |
| `release.fgdc` / `release.iso` / `release.mcf` / `release.readme` | Metadata renderers (FGDC, ISO 19115, MCF, README) |
| `release.lineage` | Manifest step/record building blocks (see [Lineage](lineage.md)) |
| `release.dry_run` | Offline build + `mp`-validate + read-only ScienceBase diff |
| `release.registry` | The intent registry (local `sb_id` ↔ scope mapping) |
| `release.publish` | Idempotent create-vs-update to ScienceBase; the completeness/staleness publish gate |
| `release.sb_client` | Thin `SbSession` wrapper |
| `release.validate_xml` | Well-formedness check on rendered metadata XML |

For the operator-facing command sequence (`init` → `build` → `dry-run` →
`publish`), see the [Data release walkthrough](../data_release/walkthrough.md).

::: nhf_spatial_targets.release
    options:
      show_source: true
      heading_level: 2
