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
- `sha256` only when `--checksum` is passed (off by default — reconcile is a
  fast directory scan; hashing a multi-hundred-GB datastore is opt-in).

## Guarantees

- **Gap-fill only.** Existing records are never mutated. A true `fetch`
  record is never downgraded to `reconciled`. Re-running is idempotent.
- **Not auto-run from `validate`.** Reconcile is an explicit operator action
  so the audit trail stays honest — the manifest never silently claims
  provenance the operator didn't ask it to assert.

## Coverage

Reconcile hooks ship for `era5_land` and `mod16a2_v061`. Sources without a
hook are reported as `no-hook` and skipped; adding one is a per-module
`reconcile(project, *, checksum=False) -> list[dict]` function plus a registry
line in `reconcile.py`.
