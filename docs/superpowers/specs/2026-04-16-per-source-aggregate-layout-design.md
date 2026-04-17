# Per-source aggregate layout — eliminate `concat_years` OOM

**Issue:** [#53](https://github.com/rmcd-mscb/nhf-spatial-targets/issues/53)
**Status:** Design accepted; implementation pending
**Author:** Richard McDonald
**Date:** 2026-04-16

## Problem

`nhf-targets agg mod10c1` is OOM-killed in SLURM at 128 GB after all 26 per-year
intermediates have been successfully written. The crash is in the driver's
final `concat_years()` step, not aggregation itself.

`aggregate/_driver.py::concat_years` calls `.load()` on every per-year NetCDF
before `xr.concat`:

```python
for p in paths:
    with xr.open_dataset(p) as ds:
        loaded.append(ds.load())   # full copy into RAM per year
combined = xr.concat(loaded, dim=time_coord)
```

MOD10C1: 26 years × daily timesteps × 3 variables (`sca`, `ci`,
`valid_area_fraction`) × 361,471 HRUs. Materializing all 26 years at once
exhausts 128 GB. Three SLURM runs failed identically at this step
(15040449, 15074747, 15129548). MOD16A2 (array 7) has the same shape and is
at risk once its per-year runs complete.

## Goal

Eliminate the OOM permanently by removing the consolidated single-file
aggregated output. Per-year NetCDFs become the canonical aggregated output
for every source that uses `aggregate_source`. Memory is bounded to one
year at a time — the working-set size the driver already handles
successfully in `aggregate_year`.

## Non-goals

- Adding dask/streaming writes.
- Touching the fetch, target, or normalize layers.
- Fixing `targets/run.py` path conventions (independently broken; out of
  scope).
- Changing the aggregation math, weight-cache format, or gdptools calls.
- Any SLURM / HPC config changes.

## Design

### Architecture

The final consolidation step is removed. `aggregate_source` becomes:

1. Load fabric, enumerate source years, confirm grid invariants
   (unchanged).
2. **Legacy-layout migration shim** (new, idempotent): if
   `data/aggregated/_by_year/<key>_<year>_agg.nc` exists, move into
   `data/aggregated/<key>/<key>_<year>_agg.nc`; unlink any stale
   `data/aggregated/<key>_agg.nc`. If both legacy and new paths exist for
   the same year, leave the legacy file in place untouched (orphan, cleaned
   up by `rm -rf _by_year/`).
3. Per-year loop: `aggregate_year` writes the new per-source path. The
   adapter's `post_aggregate_hook` runs **per year**, before the atomic
   write. CF global attrs are attached in `aggregate_year` so each per-year
   file is independently CF-compliant.
4. **Year-coverage verification** (new): walk filenames in the per-source
   directory (no dataset opens), raise on duplicates or interior year gaps.
5. Update `manifest.json` with `output_files: list[str]` listing each
   per-year path relative to `project.workdir`. `period` is derived
   cheaply from min/max year in the filename set.

`concat_years()` is deleted. Its invariants (no duplicate time coords,
no interior year gaps) are preserved by `_verify_year_coverage` at the
filename level — no dataset opens required.

### Directory layout

Before:

```
data/aggregated/
  _by_year/
    <key>_<year>_agg.nc   # intermediate
  <key>_agg.nc            # consolidated canonical output
```

After:

```
data/aggregated/
  <key>/
    <key>_<year>_agg.nc   # canonical output (one file per year)
```

### Manifest entry shape

Before:

```json
"sources": {
  "mod10c1_v061": {
    "output_file": "data/aggregated/mod10c1_v061_agg.nc",
    ...
  }
}
```

After:

```json
"sources": {
  "mod10c1_v061": {
    "output_files": [
      "data/aggregated/mod10c1_v061/mod10c1_v061_2000_agg.nc",
      "...",
      "data/aggregated/mod10c1_v061/mod10c1_v061_2025_agg.nc"
    ],
    "period": "2000-01-01/2025-12-31",
    ...
  }
}
```

Breaking schema change. The manifest is per-project internal state — no
external consumers today. No compatibility shim; the next `aggregate`
run overwrites the entry cleanly.

### Component changes

- **`aggregate/_driver.py`**
  - Delete `concat_years()`.
  - Delete the post-concat `_attach_cf_global_attrs` + final
    `_atomic_write_netcdf` calls.
  - Move `_attach_cf_global_attrs` into `aggregate_year` (before the
    atomic write).
  - Change `per_year_output_path()` to return
    `data/aggregated/<key>/<key>_<year>_agg.nc`.
  - Add `_migrate_legacy_layout(project, source_key) -> None`: moves
    `_by_year/<key>_*.nc` into the per-source subdir; unlinks stale
    `<key>_agg.nc`; no-op on collisions; idempotent.
  - Add `_verify_year_coverage(per_source_dir, source_key) -> None`:
    filename scan; raises on duplicates or interior gaps with explicit
    `missing=[...]` wording.
  - `aggregate_source` calls the migration shim at the top and the
    coverage check before the manifest update.
  - `update_manifest` signature: `output_file: str` → `output_files:
    list[str]`. `period` argument derived from the per-year filenames in
    `aggregate_source`.
- **`aggregate/_adapter.py`** — no change. `post_aggregate_hook`
  signature stays `(Dataset) -> Dataset`; the driver simply calls it
  per-year instead of post-concat.
- **`aggregate/mod10c1.py`** — `_rename_and_warn` continues to rename
  `valid_mask` → `valid_area_fraction` and fire the
  low-valid-coverage warning; the warning now runs per-year and names
  the year in its message.

Other aggregators (`era5_land`, `gldas`, `merra2`, `mod16a2`,
`ncep_ncar`, `nldas_mosaic`, `nldas_noah`, `watergap22d`, `ssebop`)
all flow through `aggregate_source` and require no per-source code
changes.

### Error handling

- **Migration collisions** (legacy and new path both exist for one
  year): no-op. Existing per-year idempotency keeps the new path
  canonical.
- **Partial legacy state** (some years migrated, some not): migration
  moves what it finds; missing years are re-aggregated by the normal
  per-year loop.
- **Coverage gaps**: `_verify_year_coverage` raises `ValueError` with
  `missing=[...]` mirroring the deleted `concat_years` check. Runs
  after the per-year loop so aggregation failures surface as coverage
  gaps.
- **Duplicate years**: `enumerate_years()` already enforces one source
  file per year upstream. `_verify_year_coverage` adds a filename-level
  defense-in-depth check.
- **`post_aggregate_hook` failures**: per-year, so a bad year raises
  with the year in the exception note and does not poison siblings.
  Re-running resumes via existing `aggregate_year` idempotency.
- **Legacy manifest readers**: none exist outside the driver. Breaking
  change is intentional and clean.

## Testing

New and updated unit tests (no integration tests required — all new
logic is path arithmetic plus existing gdptools calls that already have
coverage).

- `tests/test_aggregate_driver.py`
  - Drop existing `concat_years` tests.
  - `test_per_year_output_path_new_layout` — confirms new path shape.
  - `test_migrate_legacy_layout_moves_by_year` — `_by_year/<key>_YYYY_agg.nc`
    fixture; assert files moved into `<key>/`.
  - `test_migrate_legacy_layout_removes_stale_consolidated` — `<key>_agg.nc`
    fixture; assert removed after migration.
  - `test_migrate_legacy_layout_idempotent` — run twice, second run is a
    no-op.
  - `test_migrate_legacy_layout_collision_is_noop` — both legacy and new
    paths exist; migration leaves both untouched.
  - `test_verify_year_coverage_ok` — contiguous years pass.
  - `test_verify_year_coverage_raises_on_gap` — interior gap raises
    with `missing=[...]`.
  - `test_verify_year_coverage_raises_on_duplicate` — duplicate filename
    raises.
  - `test_aggregate_source_writes_per_year_layout` — with mocked
    gdptools, assert manifest `output_files` populated and no
    consolidated file written.
  - `test_aggregate_year_per_year_cf_attrs` — open a per-year file and
    assert `Conventions`, `history`, `source` attrs present.
- `tests/test_mod10c1.py`
  - Update path expectations to the new layout.
  - Assert `post_aggregate_hook` fires per year (caplog or counter).
  - Assert low-valid-coverage warning fires per year and names the year
    in the log message.

## Migration & operational impact

- On the first post-upgrade aggregate run, `_migrate_legacy_layout`
  moves existing `_by_year/` files into the new `<key>/` subdir and
  unlinks any stale `<key>_agg.nc`. Zero user action required.
- **MOD10C1 specifically**: all 26 existing per-year intermediates are
  reused in place (moved, not re-aggregated). The fix avoids the
  `26 × 64-batch` re-aggregation cost that was the motivating concern.
- MOD16A2 (array 7), if its per-year files are already written, gets
  the same free migration.
- Old `data/aggregated/_by_year/` directories become empty after
  migration. Users can `rm -rf` them at leisure; the driver no longer
  writes to that path.

## Rollout

Single PR against `main`. Branch: `fix/53-per-source-aggregate-layout`.
CI must pass (`pixi run -e dev fmt && pixi run -e dev lint && pixi run
-e dev test`).

No SLURM reruns are required for sources whose per-year files are
already written — the migration moves them into the new layout on the
next aggregate invocation.
