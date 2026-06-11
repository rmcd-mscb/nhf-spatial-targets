# Data release walkthrough

The `nhf-targets release` family publishes a project's calibration targets (and,
optionally, the consolidated source datasets) to a USGS ScienceBase data
release. It has seven subcommands; the first three are fully **offline**, the
rest talk to ScienceBase. All take `--project-dir` (`-d`).

For the institutional process around this (IPDS, DOI minting, review), see
[USGS data release process](usgs-process.md). For the Python surface, see the
[release package API](../api/release.md).

## The scopes

A release has three kinds of item, selected by `--scope`:

- **`fabric`** (default) — the per-project calibration-target child item.
- **`sources`** — one child item per consolidated source dataset.
- **`umbrella`** — the parent item that collects the children under one DOI.
- **`all`** — fabric + sources + umbrella together (where a command accepts it).

## Offline steps (write nothing to ScienceBase)

```bash
PROJ=/data/my-targets

# 1. Scaffold release/release.yml + release/build/. Edit release.yml to add
#    authors + ipds_number. Never clobbers an existing release.yml.
nhf-targets release init -d $PROJ

# 2. Build the payload on disk under release/build/: stage files (symlink by
#    default; --copy for no-symlink filesystems), render FGDC / ISO / README,
#    and write checksums for the selected scope.
nhf-targets release build -d $PROJ --scope fabric

# 3. (As needed) Recompute checksums.csv + SHA256SUMS for already-staged items
#    after editing a staged file — no re-render, no client.
nhf-targets release manifest -d $PROJ --scope fabric
```

## ScienceBase steps (need a token)

Authenticate with a ScienceBase token (`--token`, or `--env` to pick the
instance). Start with `auth-test`, then preview before any write.

```bash
# 4. Confirm the client authenticates and print the session identity.
nhf-targets release auth-test --token <sb-token>

# 5. Preview a release: build + mp-validate + read-only diff vs ScienceBase.
#    Writes nothing. Prints the dry-run summary.
nhf-targets release dry-run -d $PROJ --scope fabric --token <sb-token>

# 6. Read-only intent-vs-reality diff of the registry against ScienceBase.
nhf-targets release status -d $PROJ --scope all --token <sb-token>

# 7. Publish one scope (idempotent create-vs-update). Without --confirm this is
#    a dry-run preview; --confirm actually writes and logs to
#    release/last_publish_<scope>_<key>.log.
nhf-targets release publish -d $PROJ --scope fabric --confirm --token <sb-token>
```

!!! note "Publishing a single source child"
    Every subcommand spells the scope the same way (`--scope sources`); a
    *confirmed* `publish --scope sources` additionally needs `--source <key>`
    to name the one source child to publish (the singular `--scope source` /
    `--source-key` spellings were removed in #319).

## The publish gate

`publish` refuses to write when the on-disk `manifest.json` is incomplete or
drifts from the [rebuild projection](../api/rebuild-manifest.md), or when
`config.effective.yml` is stale relative to `config.yml` (you edited config
without re-running `validate`). Regenerate with
`nhf-targets maintenance rebuild-manifest` and `nhf-targets validate` rather
than overriding. `--allow-incomplete-sources`
is a deliberate, logged override for the source-completeness check only; the
schema / fabric / steps checks stay fatal.
