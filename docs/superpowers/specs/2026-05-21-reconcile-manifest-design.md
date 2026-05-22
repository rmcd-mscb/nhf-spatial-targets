# reconcile-manifest — design

Issue: #160 — *cli: add reconcile-manifest to backfill provenance from existing datastore*

Date: 2026-05-21

## Problem

The project `manifest.json` is per-project state. `init` writes nothing,
`validate` writes a skeleton (`sources: {}`, `steps: []`), and only
`fetch <src>` / `agg <src>` runs *in that project* populate it. Nothing ever
scans the datastore.

That breaks down because the datastore is explicitly shared across projects
(e.g. `gfv2-spatial-targets` and `gfv11-spatial-targets` pointing at one
`nhf-datastore/`). A new project created against a datastore that already
holds 20 years of ERA5-Land and the full MODIS archive reports **zero**
sources fetched. Provenance disagrees with on-disk reality until the operator
re-runs `fetch <src>` for every source — which works (path-based completion
checks no-op the download, then merge file records) but is slow, noisy, and
easy to forget per source.

## Goal

A new CLI command that scans the configured datastore for each known source,
finds the consolidated NCs on disk, and merges file records into the
project's `manifest.json`:

```
nhf-targets reconcile-manifest --project-dir <dir> [--source <key> ...] [--dry-run] [--checksum]
```

## Resolved design decisions

- **Gap-fill only.** Reconcile adds records *only* for files/years not
  already present in the manifest. Existing records — especially true
  `fetch` records — are never touched, so genuine fetch provenance is never
  downgraded to `reconciled`. This makes reconcile idempotent by
  construction.
- **Checksums off by default.** Reconciled records carry
  `path`/`size_bytes`/a timestamp (from file mtime) and
  `provenance: reconciled`, but **no** `sha256` unless `--checksum` is
  passed. The default is a fast directory scan; computing sha256 over a
  multi-hundred-GB datastore is opt-in. Matches the issue's "sha256 if
  cheap" hedge.
- **Records mirror each module's native fetch schema; `provenance` is the
  disambiguator.** The acceptance says reconciled records should "match
  what `fetch <src>` writes today", and the issue separately says to use
  `downloaded_utc` from mtime. These conflict for ERA5-Land, whose fetch
  records use `consolidated_utc` (not `downloaded_utc`). Resolution: each
  hook populates **its module's own timestamp field** (`consolidated_utc`
  for era5_land, `downloaded_utc` for modis) from the file mtime, and the
  flat `provenance: "reconciled"` marker — present on every reconciled
  record and absent on fetch records — is what an auditor keys on. We do
  not invent a uniform timestamp field across modules.
- **Hook returns records; a single shared writer merges.** Each fetch module
  exposes a `reconcile(project, *, checksum=False) -> list[dict]` hook that
  only scans its on-disk layout and returns file records. One atomic
  read-merge-write in `reconcile.py` performs the gap-fill merge for every
  source, keeping the merge logic (the part bug #97 burned us on) in exactly
  one place rather than spread across modules.

## Architecture

### New module: `src/nhf_spatial_targets/reconcile.py`

- `reconcile_manifest(project, *, sources=None, dry_run=False, checksum=False) -> ReconcileReport`
  — orchestrator. Resolves the requested source keys (default: all sources
  in the registry), calls each registered hook, gap-fill-merges the returned
  records into `manifest.json` (unless `dry_run`), and returns a per-source
  summary.
- `_RECONCILERS: dict[str, str]` — registry mapping `source_key ->
  "module:func"`, imported **lazily** (fetch modules pull in
  earthaccess/cdsapi; we don't want that import cost for unrelated sources).
  A catalog source with no registry entry is reported as
  *"no reconcile hook (skipped)"* — not an error.
- **Shared atomic writer** — mirrors `aggregate/_driver.update_manifest`:
  `flock(LOCK_EX)` on a `.lock` file, read existing `manifest.json`, merge,
  write. Merge rules per source:
  - Dedupe by a stable identity = `record["year"]` if present else
    `record["path"]`.
  - **Append only** identities not already present in
    `manifest["sources"][source_key]["files"]`. Never mutate an existing
    record (preserves `fetch` provenance and any existing `sha256`).
  - Entry-level metadata is created only when the source entry is **absent**:
    minimal `{source_key, period (derived from the file years found),
    access_type + doi (from catalog), reconciled_utc}`. An existing entry's
    metadata is left untouched.

### `ReconcileReport`

A small dataclass (or list of per-source records) summarizing, per source:
`on_disk` (files found), `already_recorded` (skipped), `added` (new
reconciled records), and a `status` (`reconciled` / `no-op` /
`no-hook` / `empty`). Drives both the CLI table and test assertions.

### Per-source hook contract (colocated in the fetch module)

```python
def reconcile(project: Project, *, checksum: bool = False) -> list[dict]:
    """Scan the datastore for this source's consolidated NCs on disk and
    return file records (same schema fetch writes for this source),
    tagged provenance="reconciled" with downloaded_utc from file mtime.
    Returns [] when the datastore dir is absent or empty (the no-op case).
    """
```

The hook lives next to the module's existing `_update_manifest`. It knows
its own on-disk layout and record schema; it does **not** write the
manifest.

### Initial hooks

- **`era5_land`** — `reconcile` scans `project.raw_dir("era5_land")/{daily,
  monthly}/era5_land_{daily,monthly}_{year}.nc`. Records keyed by `year`:
  `{year, daily_path, monthly_path, consolidated_utc (max mtime of the
  pair), provenance: "reconciled"}`, plus `sha256_daily`/`sha256_monthly`
  when `--checksum`. (`consolidated_utc` is the field era5_land's fetch
  records already use.)
- **`mod16a2`** (via `modis.py`) — `reconcile` scans the per-year
  consolidated NCs. Records keyed by `year`: `{year, path, size_bytes,
  downloaded_utc (mtime), provenance: "reconciled"}`, plus `sha256` when
  `--checksum`. (`downloaded_utc` matches modis fetch records.)

`mod10c1` shares `modis.py` and becomes a one-line registry addition; it is
**out of scope for this PR** to keep the change to the two products the
acceptance criteria name.

### CLI

Top-level `@app.command(name="reconcile-manifest")` in `cli.py` (not nested
under `fetch_app`):

- `--project-dir` (required) — resolves the `Project`.
- `--source <key>` — repeatable; narrows scope. Default: all registered
  sources.
- `--dry-run` — compute and report what would be added, write nothing.
- `--checksum` — compute sha256 for each reconciled record.

Prints a per-source rich summary (`on disk` / `already recorded` / `added` /
`status`). In `--dry-run`, the table is identical but no write occurs and the
header notes "(dry run — no changes written)".

## Testing

`tests/test_reconcile_manifest.py`, modeled on
`tests/test_fetch_mwbm_climgrid.py`'s `_make_project(tmp_path)` fixture, with
tiny synthetic NCs written into the era5/modis on-disk layouts:

1. **Empty datastore → no-op.** No files on disk; manifest unchanged;
   report status `empty`/`no-op`.
2. **Pre-fetched datastore + empty manifest → full backfill.** All on-disk
   years appear as `provenance: reconciled` records.
3. **Pre-fetched datastore + partial manifest → idempotent gap-fill.** A
   pre-existing `fetch` record for one year is preserved (keeps
   `provenance: fetch`); only the missing years are added; running twice
   produces an identical manifest (no duplicates).
4. **`--dry-run` → reports without writing.** Report shows the additions; the
   on-disk `manifest.json` is byte-identical before and after.
5. **`--checksum`** — reconciled records carry a `sha256`; without it they do
   not.

## Documentation

- New `docs/architecture/reconcile-manifest.md`: when to use it ("new project
  against an existing datastore"), the `provenance: reconciled` vs `fetch`
  distinction, the gap-fill / never-clobber guarantee, and why it is **not**
  auto-run from `validate` (explicit operator action keeps the audit trail
  honest).
- One-line pointer from CLAUDE.md's *Projects & Datastore* section.

## Out of scope

- Re-deriving aggregation `steps` from `data/aggregated/` (aggregation is
  per-project; the project that ran the agg owns the step record).
- Auto-running reconcile from `validate`.
- `reconcile` hooks for sources beyond `era5_land` and `mod16a2`
  (`mod10c1` and the rest are follow-on registry additions).
