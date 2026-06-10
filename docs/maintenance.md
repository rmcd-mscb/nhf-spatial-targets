# Maintaining an existing project

A project directory (`config.yml`, `fabric.json`, `manifest.json`, the aggregated
and target NetCDFs) is a long-lived audit trail — you never delete it and you never
re-`init` it. But the pipeline keeps evolving: a `git pull` can bring a new config
key, a new source dataset, a new manifest field, or a storage-layout change. This
page is the **operator's** guide to bringing an existing project up to date without
losing hand-authored intent or already-fetched data.

> It is the mirror image of the **developer** checklists in `CLAUDE.md`
> ("Config schema additions", "When you add a source / variable / target / stage").
> Those tell the person *adding* an element what to change in the codebase; this tells
> the person *running* a project how to absorb that change afterward.

## The durability split (why catch-up is safe)

Project artifacts fall into two kinds with opposite rules. Catch-up only ever
regenerates the **derived** kind; it never touches your **intent**.

| Kind | Artifacts | Catch-up rule |
|---|---|---|
| **Intent** (hand-authored) | `config.yml`, `catalog/` | Edited only by *you*, by hand. The report-only commands below tell you what to paste; they never write to these files. |
| **Derived** (a projection) | `manifest.json`, `config.effective.yml`, `fabric.json` | Regenerated from disk + catalog + intent. Never hand-edited. `validate` and `rebuild-manifest` own these. |

## The maintenance commands

Five commands, plus `validate`. The verbs split into **report-only** (safe to run any
time; they only print) and **regenerating** (they rewrite a *derived* artifact, never
your intent):

| Command | Kind | Reads | Writes | Use when |
|---|---|---|---|---|
| `upgrade-config` | report-only | `config.yml`, defaults, catalog | nothing (prints stubs) | after a pull that may add optional config keys / targets / sources |
| `upgrade-manifest` | report-only | `manifest.json` | nothing (prints status) | to check whether the manifest schema is behind |
| `validate` | regenerates | `config.yml`, fabric, catalog | `fabric.json`, `config.effective.yml`, `manifest.json` | after **any** `config.yml` edit, and before any release |
| `rebuild-manifest` | regenerates | datastore × catalog, `data/aggregated/`, `targets/`, `fabric.json` | `manifest.json` | after fetching/aggregating new data, or to normalize a behind manifest |
| `rechunk` | regenerates | `data/aggregated/`, `targets/` | rewrites those NCs in place | one-time backfill of NetCDFs built before the chunked-encoding policy (#165) |

All five accept `--project-dir`/`-d`. The report-only pair **exit non-zero on drift**
(0 = in sync), so you can wire them into a scripted heartbeat. `rebuild-manifest` and
`rechunk` accept `--dry-run` to preview without writing.

```bash
pixi run upgrade-config    -- --project-dir /data/my-targets
pixi run upgrade-manifest  -- --project-dir /data/my-targets
pixi run rebuild-manifest  -- --project-dir /data/my-targets
pixi run rechunk           -- --project-dir /data/my-targets --dry-run
```

> **Naming caveat.** `upgrade-config` and `upgrade-manifest` only *report* — they do
> not upgrade anything. The actuator for the manifest is `rebuild-manifest`; the
> actuator for config is `validate` (there is no `rebuild-config`). Reconciling the
> verb names with this behavior is tracked in
> [issue #319](https://github.com/rmcd-mscb/nhf-spatial-targets/issues/319).

## The one ordering rule that bites

After you edit `config.yml` — including pasting a stub from `upgrade-config` — you
**must** re-run `validate` before you publish:

```bash
# edit config.yml ...
pixi run validate -- --project-dir /data/my-targets
```

`validate` regenerates `config.effective.yml`, a hash-stamped projection of your
`config.yml`. `release publish` has an **unconditionally fatal** preflight gate
(`_preflight_effective_config_current`, no override flag) that refuses to publish if
`config.effective.yml`'s recorded hash no longer matches the current `config.yml`
bytes. If you edit config and forget to re-validate, the failure surfaces only later,
at publish time, as a staleness error. Re-validating immediately after every config
edit keeps you clear of it.

## The three catch-up sequences

### (a) A new config key landed

A developer added an optional key to the init template and registered it in
`upgrade_config.OPTIONAL_CONFIG_FEATURES`. Existing projects don't have it.

```bash
pixi run upgrade-config -- --project-dir /data/my-targets   # prints the commented stub to paste
# paste the printed block into config.yml by hand
pixi run validate -- --project-dir /data/my-targets         # regenerates config.effective.yml
```

`upgrade-config` also prints two informational tables when relevant: **targets in the
defaults schema absent from your config** and **catalog sources available but not yet
in your `targets.*.sources[]`** (the latter only if you pin explicit source lists).
Neither affects the exit code — they are hints, not drift.

### (b) A new source dataset landed

A developer added a source to `catalog/sources.yml` plus its `fetch/` and `aggregate/`
modules. To fold it into an existing project's targets:

```bash
pixi run upgrade-config -- --project-dir /data/my-targets   # "available catalog sources" hint
# if you pin sources: add the new key to the relevant target's sources[] in config.yml, then:
pixi run validate -- --project-dir /data/my-targets         # re-stamp config.effective.yml
pixi run fetch-<src> -- --project-dir /data/my-targets      # download to the shared datastore
pixi run agg-<src>   -- --project-dir /data/my-targets      # aggregate to this fabric
pixi run rebuild-manifest -- --project-dir /data/my-targets # the new aggregated dir enters the projection
pixi run run-<target> -- --project-dir /data/my-targets     # rebuild the affected target(s)
```

If your project uses the *default* source set for a target (no pinned `sources[]`),
the new source is picked up automatically — skip the config edit and go straight to
`fetch` → `agg` → `rebuild-manifest` → `run`.

### (c) A new manifest field landed

A developer bumped `CURRENT_MANIFEST_SCHEMA_VERSION` (the manifest's top-level *shape*
changed).

```bash
pixi run upgrade-manifest  -- --project-dir /data/my-targets  # reports "schema vN; current is M"
pixi run rebuild-manifest  -- --project-dir /data/my-targets  # re-stamps to the current schema
```

`rebuild-manifest` is a deterministic projection of what is on disk: it regenerates
`sources` and `steps`, and **read-merges** identity fields (`created_utc`, the fabric
authorship block, any `release` config) from the existing manifest rather than
re-minting them. Same disk + catalog + code → byte-identical manifest.

## One-time: backfill old NetCDF encoding

Projects whose aggregated/target NetCDFs were built before the chunked-compressed
encoding policy (#165) can reclaim disk and gain consistent chunking with a one-time,
idempotent, value-preserving rewrite:

```bash
pixi run rechunk -- --project-dir /data/my-targets --dry-run   # preview: counts + candidate GB
pixi run rechunk -- --project-dir /data/my-targets             # rewrite in place
```

`rechunk` leaves the shared datastore's consolidated NCs and the daymet/ssebop
aggregated outputs untouched (intentionally unchunked, per #165). Restrict scope with
`--layer aggregated|target` or `--source <key>`. See
[Architecture · NetCDF encoding policy](architecture/nc-encoding-policy.md) for the
per-layer rationale.

## New project against an existing datastore

Pointing a brand-new fabric project at a datastore that another project already
populated is the same gap-fill: after `init` → edit `config.yml` → `validate`, the raw
downloads are already present, so you skip `fetch`, run `agg-*` (or `agg-all`) to build
this fabric's aggregated NCs, then `rebuild-manifest` to project them into
`manifest.json`. (`rebuild-manifest` subsumed the former `reconcile-manifest`, removed
in #279; see [Architecture · Reconcile manifest](architecture/reconcile-manifest.md).)
