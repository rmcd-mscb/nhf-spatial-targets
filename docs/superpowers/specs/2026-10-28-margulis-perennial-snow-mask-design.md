# Margulis WUS-SR perennial-snow mask — design

Issue: #363. Branch: `feature/363-margulis-perennial-snow-mask`.
Related: #364 (`masked_mean` returns 0.0 for all-masked HRUs; this design
does not depend on it).

## Why

Margulis WUS-SR holds glacier and perennial-snow pixels at multi-metre SWE
through every summer. In Oregon, Mt Hood HRU 7305 sits at about 2.1–2.3 m
on Sep 30 every year, and 31 HRUs (156 km²) at Hood, Jefferson and the
Three Sisters hold more than 0.1 m on Sep 30 in every year sampled. In
Washington, Rainier and Baker pixels hold about 100 m and grow about
8 m/yr. A check on 2026-10-27 confirmed the Margulis-to-HRU mapping is
correct, so this is a trait of the product, not a pipeline bug.

PRMS cannot carry snow from one water year into the next, so at these HRUs
Margulis pulls the SWE target's upper bound to values the model can never
reach. An external user reported the effect as apparent mis-registration.

The fix removes perennial-snow pixels **before** they are averaged to HRUs.
At HRUs that are entirely perennial, Margulis drops out and `n_sources`
says so. Every other HRU keeps exactly its current value.

## Goals

- A fixed, per-pixel perennial-snow mask over the Margulis grid, derived
  from the whole record on disk.
- Margulis aggregated NCs exclude masked pixels from every daily HRU mean.
- The mask, its rule and its inputs are recorded from the datastore mask
  through the aggregated NC to the SWE target NC. A target can then be
  traced to the mask it used.
- The `inspect_aggregated_swe` notebook, and `inspect_target_swe` for
  `n_sources`, get figures and explicit checks that show the mask did what
  it should.

## Non-goals

- **No outlier screening in the multi-source ensemble.** Screening would
  also hide real disagreement between sources elsewhere. That rejection is
  the reason this design exists.
- No change to other SWE sources (UA SWE, SNODAS, ERA5-Land SD). If one of
  them also carries snow from year to year at glacier HRUs, that is a
  follow-up.
- No change to SCA; Margulis does not contribute to it.
- No per-year (time-varying) mask. One mask covers the whole record.
- No extra daily diagnostic variable in the aggregated NC. The notebook
  computes per-HRU masked-area fractions once, from the weight cache.
- **No Margulis on gfv2.** gfv2's SWE config leaves Margulis out on purpose
  and has never aggregated it, and this design does not add it.
- **Not the `masked_mean` 0.0 bug (#364).** That bug is separate, and this
  design deliberately avoids `masked_mean`.

## Anti-goals (would be a failure even if it "works")

- An HRU with no masked pixel in its footprint changes value. Its weights
  are untouched and the stat is the same, so its value must be
  **bit-identical** to today's.
- A fully perennial HRU comes out as 0.0 instead of NaN. This is the #364
  trap, and it would pull the SWE lower bound to zero at glaciers.
- Two projects that share the datastore end up with different masks.
- A stale pre-mask aggregated year, or a stale SWE intermediate, silently
  shadows the new rule.

## Decisions (Rich, 2026-10-28)

1. **Rule.** A pixel is *perennial* if its SWE is ≥ 0.1 m on Sep 30 (the
   last day of the water year) in **more than half** of the water years
   whose Sep 30 is on disk.
   - Today that is calendar files 1985–2020: 36 Sep 30 slices, so a pixel
     is perennial at ≥ 19 of 36.
2. **Fixed mask from a sweep of the whole record**, built before any
   aggregation reads it.
3. **Built automatically.** `agg margulis-wus-sr` builds the mask the first
   time it runs. It rebuilds the mask when the rule parameters or the set
   of input years change, the same self-healing approach the SWE
   intermediates use.
4. **Stay on `stat_method="mean"`; exclude masked pixels by dropping their
   rows from the weight table at load time.**
   - Rich first chose `masked_mean`. The design review then found that
     gdptools (0.3.13 and 0.4.0) returns 0.0, not NaN, for an all-NaN HRU
     under `masked_mean` (#364). Rich ruled for the weight filter instead.
   - gdptools' `mean` is `np.average(values, weights)`, which divides by
     the weights that remain, so dropping rows gives the mean over the
     seasonal pixels.
   - An HRU left with no weight rows takes the same path as an HRU the
     grid does not cover: gdptools emits the NC fill value, and the
     aggregation write boundary
     (`io_nc.mask_netcdf_default_fills`) turns it into NaN.
   - HRUs touching in-grid NaN pixels (lakes, coast) keep today's
     behaviour (NaN propagates). The earlier "edge effect" no longer
     exists.
   - The weight cache on disk is unchanged; the filter runs in memory.
5. **Notebook checks are required** (see "Verification"), not just figures.

## Constraints

- **The grid.**
  - The Margulis consolidated grid is fixed and does not depend on the
    fabric: `lat` 4050 (descending) × `lon` 5175, 31–49 N, 125–102 W, at
    225 cells/deg (about 0.2 km² per cell).
  - `SWE` is in metres and chunked one day per chunk (about 84 MB per
    slice).
  - Files are `<datastore>/margulis_wus_sr/daily/margulis_wus_sr_daily_<year>.nc`,
    one per calendar year. Sep 30 of calendar year Y is the end of WY Y.
  - Oregon and gfv2 share one datastore:
    `/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-datastore`.
- **Runs.**
  - Today Margulis aggregation runs as a single SLURM task (Oregon).
  - The CLI supports `--worker-index` / `--n-workers`, so several workers
    could reach "mask missing" at the same moment. Building the mask has
    to be safe under that race.
- **Weight cache.** Rows are `hru_id, i, j, wght`
  (`<project>/weights/margulis_wus_sr_batch<N>.csv`). `i`, `j` are
  gdptools grid indices.
- **Repo rules that apply here.**
  - Rule parameters live in `catalog/sources.yml`, read through
    `catalog.py`.
  - Every pipeline-written NetCDF is CF-compliant and written atomically.
  - `rebuild_manifest` stays a pure function of what is on disk: no
    `datetime.now()`.
  - Docs are synced in the same PR.

## How it fits

### Rule parameters — catalog

Add a block under `margulis_wus_sr` in `catalog/sources.yml`, for example:

```yaml
perennial_snow_mask:
  swe_threshold_m: 0.1        # SWE >= this on the check date counts as snow-held
  check_month_day: "09-30"    # last day of the water year
  min_year_fraction: 0.5      # perennial if held in MORE THAN this fraction of years
  notes: >
    Glacier / perennial-snow pixels removed before HRU aggregation (#363);
    PRMS cannot carry snow across water years.
```

The comparison is strict: held years / valid years > `min_year_fraction`.

### Mask artifact — datastore

- **Path.** `<datastore>/margulis_wus_sr/derived/perennial_snow_mask.nc`.
  The aggregator's `daily/...` glob does not match it.
- **Variables.** All on the consolidated grid's own `lat` / `lon` coords,
  copied exactly.
  - `perennial`: int8, 0/1, with CF `flag_values` / `flag_meanings`.
  - `n_years_held`: int16, the count of Sep 30 slices at or above the
    threshold.
  - `n_years_valid`: int16, the count of Sep 30 slices that are finite at
    the pixel. This is the denominator. A pixel with no valid years is not
    perennial.
- **Global attrs.**
  - The three rule parameters.
  - The sorted list of input years.
  - `mask_fingerprint`: a short sha256 of the rule parameters plus the
    input-year list.
  - The count and area of masked pixels.
- **Encoding.** CF-1.6, written atomically with an **explicit** encoding.
  `io_nc.build_encoding` raises for any layer without an HRU dimension,
  so it can't be used here.
- **Builder.** A new function, e.g.
  `aggregate/margulis_wus_sr.py:ensure_perennial_snow_mask(datastore) -> Path`.
  - It reads only the Sep 30 slice of each daily file: 36 slices of about
    84 MB each.
  - If the file exists and its `mask_fingerprint` matches the current
    catalog rule and the years on disk, it returns the path untouched.
  - Otherwise it builds the mask and writes it atomically (temp file plus
    rename). Concurrent workers then produce identical content, and
    whichever rename lands last does no harm.
  - A daily file without a Sep 30 timestep is left out of the year list,
    with an INFO log.

### Aggregation

- `aggregate_margulis_wus_sr` calls `ensure_perennial_snow_mask` before
  `aggregate_source`, and loads the mask's set of masked `(i, j)` cells.
- **Weight filter.** It installs a weight filter that removes rows whose
  `(i, j)` is masked from the loaded weight DataFrame before it reaches
  `AggGen`.
  - This needs a new optional hook in the shared driver, e.g. a
    `SourceAdapter.weight_filter: Callable[[pd.DataFrame], pd.DataFrame] | None`.
  - It is applied wherever weights are loaded or freshly generated, never
    written back to the cache.
  - Margulis is the only user.
- **Mask grid alignment is checked, not assumed.**
  - The builder must confirm how gdptools' `i` / `j` relate to the
    consolidated grid's `lat` / `lon` indices: full grid or a per-batch
    subset, and which is row and which is column. They must pin it with a
    test.
  - At run time, the mask's `lat` / `lon` must equal the year file's
    exactly. On any mismatch the run raises.
- `stat_method` stays `"mean"`. `pre_aggregate_hook` stays `None`; the
  mask works through the weights, not the pixel values.
- The per-year aggregated NC gets the mask's `mask_fingerprint` and rule
  parameters as global attrs.
- **Stale years self-heal.**
  - If an existing per-year aggregated NC lacks `mask_fingerprint` or
    carries a different one, the driver treats it as stale and
    re-aggregates that year, logging a WARNING, instead of skipping it.
  - Without this, today's idempotent skip in `aggregate/_driver.py`
    `aggregate_year` (around line 664) would keep the pre-mask glacier
    values forever.
  - The builder chooses the generic shape, e.g. an optional adapter field
    naming the expected attr and value.

### SWE target

- `targets/swe.py` stamps the Margulis `mask_fingerprint` on the SWE target
  NC (e.g. `margulis_wus_sr_perennial_mask_fingerprint`) whenever Margulis
  is a member.
- The SWE intermediates' cache fingerprint (`targets/_intermediates.py`)
  includes it, so pre-mask intermediates rebuild themselves instead of
  needing a manual `rm`. `__version__` is fixed at 0.1.0, so the version
  tag alone would not catch the change.
- *My call:* no new publish-gate check. The provenance chain is enough for
  now.

### Manifest / provenance

- `rebuild_manifest` already picks up `derived/perennial_snow_mask.nc` as a
  Margulis consolidate output, through its recursive search of
  `<datastore>/<key>/`.
- The new projection work is recording the fingerprint and rule
  parameters as Margulis aggregate-step params. Follow the existing ua_swe
  `depth_threshold_mm` pattern in `rebuild_manifest._aggregate_nc_params`.
- Add the matching case to `tests/test_rebuild_manifest.py`.
- No schema version bump: the top-level shape is unchanged.
- *My call:* the mask ships in the Margulis release source child as
  provenance. `release/payload.plan_source_child` already collects every
  `*.nc` under the source dir, so no filter is needed. It is a few MB.

### Docs (same PR)

- `docs/sources/margulis_wus_sr.md`:
  - what the mask removes and why, and the rule;
  - where the mask lives, how it rebuilds itself, and how to force a
    rebuild;
  - that edge and lake HRUs are unaffected.
- `docs/architecture/transformation-pipeline.md`: add the weight filter as
  the pattern for a **static** pixel mask under `mean`, and say why
  (#364).
- `aggregate/margulis_wus_sr.py` module docstring and the `SourceAdapter`
  docstring: the new `weight_filter` field.
- `catalog/sources.yml` Margulis `description` / `notes`.
- The "remains NaN" corrections for the `masked_mean` users belong to
  #364, not this PR.

## Verification

### Tests

Tests go in `tests/test_aggregate_margulis_wus_sr.py` and the driver's
test module, using synthetic tiny grids.

- **Mask rule.**
  - Held in exactly half of the years → not perennial; one more year →
    perennial.
  - NaN years are excluded from the denominator.
  - An all-NaN pixel → not perennial.
- **`ensure_perennial_snow_mask`.**
  - Reuses a matching mask without rewriting it (mtime unchanged).
  - Rebuilds when the threshold changes, and when a year is added.
  - Skips a daily file that has no Sep 30.
- **Weight filter.**
  - Drops exactly the masked `(i, j)` rows.
  - The `i` / `j` orientation matches the grid. Use an asymmetric mask so
    a transposed or flipped index fails.
  - The cache file on disk is unchanged.
- **End to end with real gdptools** on a three-HRU toy fabric:
  - a fully perennial HRU → **NaN** after the write boundary, not 0.0;
  - a half-perennial HRU → the weighted mean of its seasonal pixels;
  - an HRU with no perennial pixels → bit-identical to the unfiltered
    run.
- **Self-healing.**
  - A stale per-year NC (no fingerprint attr) is re-aggregated, not
    skipped.
  - SWE intermediates rebuild when the fingerprint changes.
- **Output metadata.** The aggregated and target NCs carry the fingerprint
  attrs. The mask NC carries the CF-1.6 attribute set.

### Notebook checks

These are assertions with printed summaries, not only plots.

`notebooks/aggregated/inspect_aggregated_swe.ipynb`:

1. **Mask maps.** Maps of `n_years_held / n_years_valid` and of the mask,
   over the whole Margulis domain and zoomed to Oregon's Cascades. Print
   the masked-pixel count and area per broad region: Cascades, Sierra,
   northern Rockies at least.
2. **Check: the rule holds.** Recompute the mask from the Sep 30 slices for
   a small window around Mt Hood, and assert it matches the stored mask.
3. **Check: no change away from masked pixels.**
   - Use the weight cache to find HRUs with no masked pixel in their
     footprint.
   - Compare their pre-mask and post-mask aggregated values for a sample of
     years, and assert they are **bit-identical**, including NaN in the
     same places.
4. **HRUs that touch the mask.**
   - Print how many there are, the distribution of masked-area fractions,
     and how many are fully masked.
   - **Check:** fully masked HRUs are NaN on every day, never 0.0.
   - If none are fully masked in this project, say so explicitly rather
     than passing silently.
5. **Reporter HRUs (Oregon 7305, 7346, 7306, 11845).**
   - Show July SWE before and after, and each HRU's masked-area fraction.
   - **Check:** for each partly masked reporter HRU on a sample day, the
     post-mask value equals the weighted mean of its unmasked pixels,
     recomputed from the weights and the raw grid. This is the
     non-vacuous check, since 7305 (about 2.2 m) is probably only partly
     perennial.

`notebooks/targets/inspect_target_swe.ipynb`:

6. Show `n_sources` before and after at the reporter HRUs, and count the
   HRUs whose `n_sources` dropped.
   - **Check:** `n_sources` dropped only at HRUs that are fully masked for
     Margulis.

The "before" side needs the pre-mask files. The rollout moves them aside
instead of deleting them, so the notebooks can read both.

## Rollout

Every `sbatch` needs Rich's OK first, and always uses `--account=impd`.

1. Merge the PR.
2. **Oregon.**
   1. Move `data/aggregated/margulis_wus_sr/` and the current
      `targets/swe_targets.nc` to
      `<project>/archive/margulis_premask_2026-10-28/`.
      - The design review confirmed that `archive/` is not walked by
        `rebuild_manifest`, the release payload, the target readers,
        `rechunk` or `relabel_id`.
   2. `agg margulis-wus-sr`. This builds the mask in the datastore.
      Weight caches stay valid because the grid is unchanged.
   3. `run-swe`. The intermediates rebuild themselves through the new
      fingerprint.
   4. `maintenance rebuild-manifest`, then `validate`.
   5. Run the notebook checks. The reply to the reporter waits for them.
3. **gfv2.** Run only `maintenance rebuild-manifest`. gfv2 does not use
   Margulis, but the new mask file in the shared datastore changes its
   consolidate step, and otherwise its publish gate would fail on drift.
4. Once Rich signs off on the comparison, the archived pre-mask files can
   be deleted. That is his decision.

## Left to the builder

- The generic shapes of the `weight_filter` hook and of the stale-year
  check in the driver.
- How the closure carries the mask into the weight filter.
- The exact name of the SWE target attr, and how the fingerprint enters
  the intermediates' cache key.
- How notebook check 3 finds the HRUs touching the mask from the weight
  cache.
- The region boundaries for the per-region counts in notebook check 1.

## Amendment (Rich, 2026-10-28, after final review)

This supersedes the fingerprint bullet in "Mask artifact — datastore".

- **Two keys.**
  - `mask_build_key(rule, years)` (rule plus sorted years; global attr
    `perennial_snow_mask_build_key`) decides when the mask NC is rebuilt.
  - `mask_content_fingerprint(rule, lat, lon, perennial)` (sha256 of the rule,
    grid sizes, `<f8` lat/lon and `|i1` flags in C order) is stored as
    `perennial_snow_mask_fingerprint` and is what aggregated NCs, the SWE
    target and the manifest carry. The years list stays on the mask NC for
    provenance but is not stamped downstream.
- **Why.** A rule-plus-years stamp re-aggregated every year in every project
  whenever any year was added or removed, even when no mask cell changed. It
  also left a rebuilt mask with changed cells under the same stamp. Stamping
  content re-aggregates exactly when the removed-pixel set or the rule changes.
- **Consequence.** Re-consolidating a year still needs a manual delete of the
  mask, that year's aggregated NC and `.swe_intermediates/`; after that, a
  changed mask re-aggregates all years automatically.
