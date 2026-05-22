# reconcile-manifest

`nhf-targets reconcile-manifest --project-dir <dir> [--source <key> ...] [--dry-run] [--checksum]`

## When to use it

The datastore is shared across projects; `manifest.json` is per-project. A
new project created against a datastore another project already populated
starts with an empty manifest even though the NCs are on disk. `reconcile-manifest`
scans the datastore and backfills file records so the project's provenance
matches on-disk reality — without re-running `fetch` for every source.

## What it writes

For each source with a reconcile hook, it adds one `files` record per
consolidated NC found on disk that is not already recorded, tagged:

- `provenance: "reconciled"` — the marker that distinguishes a reconciled
  record from a true `fetch` record (which has no `provenance` key).
- a timestamp from the file's mtime (each source uses its native fetch
  field: `consolidated_utc` for era5_land, `downloaded_utc` for modis).
- a checksum only when `--checksum` is passed (off by default — reconcile is a
  fast directory scan; hashing a multi-hundred-GB datastore is opt-in).

The record *shape* follows each source's native fetch shape, so it is not
uniform across sources:

- single-file sources (modis) use `path` + `size_bytes` + `sha256`.
- the year-paired era5_land source uses `daily_path` / `monthly_path` and,
  under `--checksum`, `sha256_daily` / `sha256_monthly` (no single `path`/
  `sha256`). Records are deduped by `year` here, not `path`.

## Guarantees

- **Gap-fill only.** Existing records are never mutated. A true `fetch`
  record is never downgraded to `reconciled`. Re-running is idempotent.
- **Not auto-run from `validate`.** Reconcile is an explicit operator action
  so the audit trail stays honest — the manifest never silently claims
  provenance the operator didn't ask it to assert.

## Status column

The CLI summary table reports one status per source:

- `reconciled` — new on-disk records were appended.
- `no-op` — files found, but every one was already recorded (idempotent re-run).
- `empty` — the hook ran and found nothing on disk.
- `no-hook` — no reconcile hook is registered for this source (skipped).
- `error` — the hook or writer raised; logged and isolated so the rest of the
  run still completes (e.g. a file vanishing mid-scan on a shared datastore).

When a new source entry is created, its `period` is the **span** of reconciled
years (`min/max`), not a dense coverage guarantee — a gap year between the
endpoints is not implied to exist (this mirrors how `fetch` computes
`effective_period`).

## Coverage

Reconcile hooks ship for `era5_land` and `mod16a2_v061`. Sources without a
hook are reported as `no-hook` and skipped; adding one is a per-module
`reconcile(project, *, checksum=False) -> list[dict]` function plus a registry
line in `reconcile.py` (keep that `_RECONCILERS` dict and this list in sync).
