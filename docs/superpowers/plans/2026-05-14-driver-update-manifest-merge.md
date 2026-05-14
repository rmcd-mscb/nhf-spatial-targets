# Fix `_driver.update_manifest` to merge at entry level (closes #119)

## Context

The shared
[src/nhf_spatial_targets/aggregate/_driver.py](src/nhf_spatial_targets/aggregate/_driver.py)
helper `update_manifest` builds a fresh entry dict and assigns it to
`manifest["sources"][source_key]` at line 80, replacing the whole
sub-entry. Every fetch module in this repo (era5_land, snodas,
margulis, modis, mwbm_climgrid, ncep_ncar, gldas, nldas, pangaea,
reitz2017, daymet) already does **entry-level read-merge-write**:

```python
entry = manifest["sources"].get(_SOURCE_KEY, {})
entry.update({...})
manifest["sources"][_SOURCE_KEY] = entry
```

The aggregator helper is the one place that diverges from this
convention. Daymet's fetch records nested provenance
(`sources.daymet.regions.<r>`); if the daymet aggregator called the
shared `update_manifest`, those keys would be wiped. PR #118
worked around this with a **local `_merge_manifest_entry` copy** in
`aggregate/daymet.py` and filed #119 for the central fix.

SNODAS has the same latent bug: `fetch/snodas.py` writes a
per-year `years` list that the SNODAS aggregator's
`update_manifest` call silently overwrites every run. The fix here
benefits both daymet (lets us delete the local copy) and SNODAS
(stops the silent overwrite).

Closes #119. References PR #117 (SNODAS), PR #118 (daymet),
umbrella #101.

## Approach

### 1. Three-line fix in `_driver.update_manifest`

[src/nhf_spatial_targets/aggregate/_driver.py:80](src/nhf_spatial_targets/aggregate/_driver.py#L80):

```python
# before
manifest["sources"][source_key] = entry

# after
existing = manifest["sources"].get(source_key, {})
existing.update(entry)
manifest["sources"][source_key] = existing
```

**Backward compatibility.** For sources whose fetch doesn't write
nested state, the existing entry contains only keys that the new
entry also writes (`source_key`, `access_type`, `period`,
`fabric_sha256`, `output_files`, `weight_files`, `timestamp`, plus
optional `doi`/`collection_id`/`short_name`/`version`). `dict.update`
overwrites those keys to the same effective values, so the call is a
no-op behavior change for ~10 of the 12 aggregators. For daymet and
SNODAS, fetch-side keys (`regions`, `years`, `doi`, `license`,
`access_url`, `spatial_extent`, `variables`) survive.

Update the function's docstring to mention the merge semantics
explicitly — current docstring says "Merge an aggregation provenance
entry" but isn't specific about entry-level vs source-level merge.

### 2. Delete daymet's local copy

[src/nhf_spatial_targets/aggregate/daymet.py](src/nhf_spatial_targets/aggregate/daymet.py):

- Delete `_merge_manifest_entry` (lines 52-117).
- Swap the call site at line 426 from `_merge_manifest_entry(...)` to
  `update_manifest(project=project, source_key=_SOURCE_KEY,
  access=access_with_doi, period=..., output_files=...,
  weight_files=...)`.
- Add `update_manifest` to the `_driver` import block (lines 28-37).
- Prune now-unused module imports: `os`, `tempfile`, and
  `from datetime import datetime, timezone`. Keep `json` (still
  used by `_resolve_zarr_path` at lines 163-164).

### 3. Regression test for the shared helper

Add a focused unit test in
[tests/test_aggregate_driver.py](tests/test_aggregate_driver.py)
alongside the existing
`test_update_manifest_preserves_existing_sources` (line 61) — same
pattern, different invariant:

```python
def test_update_manifest_preserves_pre_existing_entry_keys(...):
    """`update_manifest` must merge into the existing entry for
    `source_key`, not overwrite it. Guards against the bug from
    issue #119 where fetch-side keys (e.g. daymet's `regions` dict,
    snodas's `years` list) were silently wiped on every aggregator
    run."""
    # Pre-populate sources.foo with a fetch-style nested key.
    # Call update_manifest(source_key="foo", ...).
    # Assert: the pre-existing nested key survives; new aggregator
    # keys are present.
```

### 4. Existing daymet test stays as the integration guard

[tests/test_aggregate_daymet.py:359-399](tests/test_aggregate_daymet.py#L359-L399)
(`test_manifest_merge_preserves_fetch_regions`) was written in PR #118
to assert that fetch-side `regions` keys survive the manifest write.
The test goes through `aggregate_daymet` (public API), not the
private helper. After the swap, it exercises the shared
`update_manifest` path — same assertion, free integration coverage.
**No edit needed.**

## Files modified

- [src/nhf_spatial_targets/aggregate/_driver.py](src/nhf_spatial_targets/aggregate/_driver.py)
  — 3-line merge swap at L80; docstring tweak.
- [src/nhf_spatial_targets/aggregate/daymet.py](src/nhf_spatial_targets/aggregate/daymet.py)
  — delete `_merge_manifest_entry`; swap call site; prune unused
  imports; add `update_manifest` to existing `_driver` import.
- [tests/test_aggregate_driver.py](tests/test_aggregate_driver.py)
  — one new test (`test_update_manifest_preserves_pre_existing_entry_keys`).

## Files **not** modified

- `tests/test_aggregate_daymet.py` — existing manifest-merge test
  continues to pass through the new code path.
- All other aggregator modules — they use `update_manifest` already
  via `_driver.aggregate_source` and pick up the new merge for free.
- `aggregate/ssebop.py` — calls `update_manifest` directly
  ([line 297](src/nhf_spatial_targets/aggregate/ssebop.py#L297));
  picks up the merge behavior. No fetch-side keys at risk because
  ssebop is read-on-the-fly from STAC (no fetch module).
- Catalog, slurm scripts, notebooks, CLI, CLAUDE.md.

## Reused helpers

- `update_manifest` from
  [src/nhf_spatial_targets/aggregate/_driver.py](src/nhf_spatial_targets/aggregate/_driver.py)
  is the helper being fixed and re-used.
- Existing test fixtures in `tests/test_aggregate_driver.py` (e.g.
  `tmp_workdir` setup at line 17+) already cover the manifest-on-disk
  pattern.

## Verification

```bash
pixi run -e dev fmt && pixi run -e dev lint   # locally
git push                                       # let CI run pytest
```

CI must run, and these three tests must pass:

- `tests/test_aggregate_driver.py::test_update_manifest_preserves_pre_existing_entry_keys` (new)
- `tests/test_aggregate_driver.py::test_update_manifest_preserves_existing_sources` (existing cross-source guard)
- `tests/test_aggregate_daymet.py::test_manifest_merge_preserves_fetch_regions` (now exercises the shared helper through `aggregate_daymet`)

Manual cross-check after merge: on a project with a real
fetch-populated manifest, re-run `pixi run nhf-targets agg daymet
--project-dir <…> --period <…>`. Confirm `manifest.json`
`sources.daymet` still carries the fetch-side `regions` dict
alongside the new `output_files`/`weight_files`/`timestamp`. Same
check for SNODAS — `sources.snodas.years` should survive
post-aggregation (regression that the new path also fixes).

Per `feedback_skip_local_pytest_on_hpc` memory: don't run
`pixi run -e dev test` locally on this HPC; CI is the gate.

## Git workflow

- Branch from `main`: `fix/119-driver-update-manifest-merge`.
- PR title: `fix(driver): merge update_manifest at entry level (#119)`.
- PR body: closes #119; cite PR #117 (SNODAS, fixed-for-free) and
  PR #118 (daymet, deletes local copy).
- Copy this plan to
  `docs/superpowers/plans/2026-05-14-driver-update-manifest-merge.md`
  and include it in the PR (precedent: PR #117, PR #118).
