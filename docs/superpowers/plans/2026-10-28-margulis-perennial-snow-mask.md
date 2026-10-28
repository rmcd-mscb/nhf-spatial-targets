# Margulis Perennial-Snow Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exclude Margulis WUS-SR perennial-snow (glacier) pixels from HRU
aggregation, so that fully glaciated HRUs drop Margulis (`n_sources` falls)
and every other HRU keeps its exact current value.

**Architecture:**
- A fixed, fabric-independent pixel mask is built once from the Sep 30 SWE
  of every water year and stored in the datastore.
- A new generic `SourceAdapter.weight_filter` drops masked cells from each
  batch's weight table in memory, under `stat_method="mean"`. Matching is
  done through the coordinates of the agg subset gdptools reads, never
  through raw indices.
- A mask fingerprint travels as an attr: aggregated per-year NC →
  manifest aggregate-step params → SWE target NC. It also drives
  self-healing of stale per-year aggregates and stale SWE intermediates.

**Tech Stack:** Python 3.11+, xarray, pandas, numpy, gdptools 0.3.13
(`UserCatData` / `AggGen`), netCDF4, pytest, pixi.

**Spec:** [docs/superpowers/specs/2026-10-28-margulis-perennial-snow-mask-design.md](../specs/2026-10-28-margulis-perennial-snow-mask-design.md)

## Global Constraints

- **Rule** (catalog, `catalog/sources.yml` → `margulis_wus_sr.perennial_snow_mask`):
  - `swe_threshold_m: 0.1`, `check_month_day: "09-30"`, `min_year_fraction: 0.5`.
  - A pixel is perennial iff `n_years_held / n_years_valid > 0.5`.
    `held` means SWE ≥ 0.1 on the check date; the comparison is strict.
  - A pixel with `n_years_valid == 0` is not perennial.
- **Mask path:** `<datastore>/margulis_wus_sr/derived/perennial_snow_mask.nc`.
- **Fingerprint:** the first 12 hex characters of the sha256 of the
  canonical JSON (`sort_keys=True`) of
  `{"swe_threshold_m", "check_month_day", "min_year_fraction", "years": sorted list[int]}`.
- **Attr names**, global attrs, used everywhere:
  - `perennial_snow_mask_fingerprint`
  - `perennial_snow_mask_swe_threshold_m`
  - `perennial_snow_mask_check_month_day`
  - `perennial_snow_mask_min_year_fraction`
  - the SWE target NC uses `margulis_wus_sr_perennial_snow_mask_fingerprint`.
- **Margulis stays `stat_method="mean"`.** Never `masked_mean`, because of
  the #364 all-NaN → 0.0 trap.
- **Weight caches on disk are never modified** by the filter.
- **Repo rules:**
  - Every NC is written atomically and is CF-1.6. The mask NC uses an
    **explicit** encoding (`io_nc.build_encoding` needs an HRU dim).
  - No `datetime.now()` reachable from `rebuild_manifest`.
  - Rule values are read via `nhf_spatial_targets.catalog`, never from
    YAML directly.
- **Testing on caldera:**
  - Run fmt and lint locally: `pixi run -e dev fmt && pixi run -e dev lint`.
  - Run only single targeted tests (`pixi run -e dev pytest <file>::<test> -v`).
    Never run the full suite locally; CI runs it.
  - Commit with `pixi run git commit`, staging files by explicit path.
- **No `sbatch`** without Rich's OK, and always with `--account=impd`.

## Review Focus

1. **A batch whose filtered weight table is empty** (every HRU in the batch
   fully glaciated). Expected: an all-NaN batch, not a gdptools crash on an
   empty groupby. Pinned in Task 2, `test_weight_filter_empty_batch_all_nan`.
2. **Mixed pre- and post-mask Margulis per-year NCs** (an interrupted
   re-aggregation). Expected: the SWE target build raises and names the
   years, rather than stamping one fingerprint. Pinned in Task 5,
   `test_margulis_mask_fingerprint_mixed_raises`.
3. **A mask grid that differs from a year file's grid** (e.g. a
   re-consolidated year with shifted coords). Expected: the filter raises
   `ValueError` naming the unmatched coordinate count, never a silent
   partial mask. Pinned in Task 3,
   `test_perennial_weight_filter_unmatched_coords_raise`.
4. **An HRU fully inside the Margulis grid but adjacent to in-grid NaN
   (lake / ocean) pixels and with no masked pixel.** Expected: the value is
   unchanged, NaN included. Pinned in Task 4,
   `test_end_to_end_unmasked_hru_bit_identical` (the toy grid includes a
   NaN pixel).
5. **A year file with no Sep 30 timestep, or a truncated last year.**
   Expected: it is left out of the year list (INFO log) and the
   fingerprint changes. Pinned in Task 3,
   `test_ensure_mask_skips_year_without_check_date`.

---

## File Structure

- **`src/nhf_spatial_targets/aggregate/_adapter.py`** (modify): new fields
  `weight_filter` and `stamp_attrs`.
- **`src/nhf_spatial_targets/aggregate/_driver.py`** (modify):
  - apply `weight_filter` inside `aggregate_variables_for_batch`;
  - stamp `stamp_attrs` onto every per-year NC;
  - treat an existing per-year NC whose stamped attrs mismatch as stale.
- **`src/nhf_spatial_targets/aggregate/margulis_wus_sr.py`** (modify): the
  mask rule, `ensure_perennial_snow_mask`, `make_perennial_weight_filter`,
  and wiring in `aggregate_margulis_wus_sr`.
- **`catalog/sources.yml`** (modify): the `perennial_snow_mask` block and
  prose.
- **`src/nhf_spatial_targets/targets/_adapter.py`**,
  **`targets/_driver.py`** and **`targets/_intermediates.py`** (modify): an
  optional `TargetAdapter.input_provenance` hook, folded into the cache
  fingerprint and the global attrs.
- **`src/nhf_spatial_targets/targets/swe.py`** (modify): the Margulis
  `input_provenance` implementation.
- **`src/nhf_spatial_targets/rebuild_manifest.py`** (modify): lift the
  Margulis mask attrs into the aggregate-step params.
- **Tests** (modify / extend): `tests/test_aggregate_driver.py`,
  `tests/test_aggregate_margulis_wus_sr.py`, `tests/test_targets_swe.py`,
  `tests/test_rebuild_manifest.py`.
- **Docs** (modify): `docs/sources/margulis_wus_sr.md`,
  `docs/architecture/transformation-pipeline.md`.
- **Notebooks** (modify):
  `notebooks/aggregated/inspect_aggregated_swe.ipynb`,
  `notebooks/targets/inspect_target_swe.ipynb`.

---

### Task 1: Driver — `stamp_attrs` and stale-year self-healing

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_adapter.py` (`SourceAdapter` fields + docstring)
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py:635-805` (`aggregate_year`)
- Test: `tests/test_aggregate_driver_per_year.py`

**Interfaces:**
- Produces:
  - `SourceAdapter.stamp_attrs: Mapping[str, str | float] | None = None`.
    These global attrs are written onto every per-year NC this adapter
    emits.
  - An existing per-year NC with **any** key in `stamp_attrs` missing, or
    with a value not equal to the expected one, is stale. `aggregate_year`
    logs a WARNING naming the key, the expected value and the found value,
    then re-aggregates and overwrites it atomically.
  - With `stamp_attrs=None`, behaviour is byte-for-byte unchanged.

- [ ] **Step 1: Write the failing tests** in `tests/test_aggregate_driver_per_year.py`. Reuse the existing toy-source fixtures in that file.

```python
def test_stamp_attrs_written_to_per_year_nc(...):
    # adapter = dataclasses.replace(toy_adapter, stamp_attrs={"perennial_snow_mask_fingerprint": "abc123def456",
    #                                                         "perennial_snow_mask_swe_threshold_m": 0.1})
    # run aggregate_year; open the output NC
    assert ds.attrs["perennial_snow_mask_fingerprint"] == "abc123def456"
    assert float(ds.attrs["perennial_snow_mask_swe_threshold_m"]) == 0.1

def test_existing_nc_without_stamp_is_reaggregated(..., caplog):
    # run once with stamp_attrs=None, then again with stamp_attrs={"perennial_snow_mask_fingerprint": "abc123def456"}
    assert reopened.attrs["perennial_snow_mask_fingerprint"] == "abc123def456"
    assert "stale" in caplog.text and "perennial_snow_mask_fingerprint" in caplog.text

def test_existing_nc_with_matching_stamp_is_skipped(...):
    # two runs with identical stamp_attrs; second run must not rewrite (mtime unchanged)
    assert out.stat().st_mtime_ns == first_mtime

def test_existing_nc_with_different_stamp_is_reaggregated(...):
    # first "aaaaaaaaaaaa", then "bbbbbbbbbbbb" -> rewritten with "bbbbbbbbbbbb"
```

- [ ] **Step 2: Run them and confirm they fail.** Run `pixi run -e dev pytest tests/test_aggregate_driver_per_year.py -k stamp -v`. Expected: FAIL, unexpected keyword `stamp_attrs`.
- [ ] **Step 3: Implement.**
  - Add the field after `post_aggregate_hook`.
  - In `aggregate_year`, the existing-file branch (around line 664) opens the file. If `adapter.stamp_attrs` is set and mismatches, warn and fall through to aggregation; otherwise keep the existing #309 warning and the skip.
  - Compare floats with `math.isclose(rel_tol=0, abs_tol=0)` after `float()`, and strings with `str()` equality.
  - Set `year_ds.attrs.update(stamp_attrs)` after `post_aggregate_hook` and before `_atomic_write_netcdf`. Confirm `_attach_cf_global_attrs` does not drop them; the first test proves it.
  - Document the field in the `SourceAdapter` docstring.
- [ ] **Step 4: Run the Step 2 command and confirm the tests pass.**
- [ ] **Step 5: Commit.** `pixi run git commit -m "feat(#363): SourceAdapter.stamp_attrs + stale per-year self-heal"`, after `git add` of the three files.

### Task 2: Driver — `weight_filter` resolved through agg-subset coordinates

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_adapter.py`
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py:956-1033` (`aggregate_variables_for_batch`) and its call in `aggregate_year` (around line 737)
- Test: `tests/test_aggregate_driver.py`

**Interfaces:**
- Produces:
  - `SourceAdapter.weight_filter: Callable[[pd.DataFrame, np.ndarray, np.ndarray], pd.DataFrame] | None = None`.
    It is called as `weight_filter(weights, cell_y, cell_x)`, where
    `cell_y[k]` / `cell_x[k]` are the y / x coordinate values of the source
    cell that weight row `k` refers to. It returns the subset of rows to
    keep, with the same columns.
  - `aggregate_variables_for_batch(..., weight_filter=None)`: a new
    keyword-only parameter. `aggregate_year` passes `adapter.weight_filter`.
  - `_weight_cell_coords(user_data: UserCatData, var: str, weights: pd.DataFrame, x_coord: str, y_coord: str) -> tuple[np.ndarray, np.ndarray]`.
    This private helper maps each row's `(i, j)` to coordinates using
    `user_data.get_source_subset(var)`, the same subset `AggGen` indexes
    as `mdata[:, i, j]`.
- Behaviour:
  - The filter runs once per batch, on the first variable's `user_data`,
    and the filtered table is used for every variable.
  - If the filtered table is empty, `AggGen` is not called. The function
    returns a dataset of all-NaN float values for every variable, with the
    batch's HRU ids on `id_col` and the time coords of the subset. It must
    merge and reindex exactly like a normal batch.
  - The cache CSV is never rewritten.

- [ ] **Step 1: Write the failing tests** in `tests/test_aggregate_driver.py`, using real gdptools on a synthetic 6 (lat, descending) × 8 (lon) grid. The grid must not start at the batch's corner, so subset offsets are non-zero. Use three HRU polygons, each covering whole cells, plus one NaN pixel inside HRU C.

```python
def test_weight_cell_coords_match_cell_centres():
    # for every weight row, (cell_y, cell_x) must equal the centre of a cell that intersects that HRU polygon
    # use an ASYMMETRIC layout (HRU A spans 1 row x 3 cols) so a swapped i/j or flipped lat fails
    assert set(zip(cy[a_rows], cx[a_rows])) == expected_cells_A

def test_weight_filter_drops_rows_and_means_survivors():
    # filter removes 2 of HRU B's 4 cells; SWE values known per cell
    np.testing.assert_allclose(out["SWE"].sel(hru=B), mean_of_surviving_cells_weighted)

def test_weight_filter_fully_filtered_hru_is_fill_then_nan():
    # all of HRU A's cells filtered -> after the driver's write-boundary mask, A is NaN on every day, never 0.0
    assert np.isnan(year_ds["SWE"].sel(hru=A)).all()

def test_weight_filter_none_is_unchanged():
    # weight_filter=None output is bit-identical to calling AggGen directly (existing path)
    xr.testing.assert_identical(with_none, baseline)

def test_weight_filter_empty_batch_all_nan():
    # filter returns weights.iloc[0:0]; result has all batch HRUs, all-NaN, and AggGen is not invoked
    assert np.isnan(out["SWE"]).all() and set(out[id_col].values) == set(batch_ids)

def test_weight_filter_does_not_touch_cache(tmp_path):
    # cache CSV bytes identical before/after an aggregate_year run with a filter
```

- [ ] **Step 2: Run them and confirm they fail.** Run `pixi run -e dev pytest tests/test_aggregate_driver.py -k "weight_filter or weight_cell_coords" -v`. Expected: FAIL, `weight_filter` unknown / `_weight_cell_coords` missing.
- [ ] **Step 3: Implement.**
  - Write the helper. Determine empirically, with the asymmetric test, which of `i` / `j` indexes y.
  - Add the parameter, the empty-table short-circuit and the field.
  - Extend the `stat_method` paragraph of the `SourceAdapter` docstring with one sentence: a **static** pixel mask under `mean` uses `weight_filter`, not `pre_aggregate_hook` + `masked_mean`, because of #364.
- [ ] **Step 4: Run the Step 2 command and confirm the tests pass.**
- [ ] **Step 5: Commit.** `feat(#363): SourceAdapter.weight_filter resolved via agg-subset coords`.

### Task 3: Margulis mask — catalog rule, builder, filter factory

**Files:**
- Modify: `catalog/sources.yml` (`margulis_wus_sr`: add the `perennial_snow_mask` block from the spec's "Rule parameters" section, update `description` / `notes`)
- Modify: `src/nhf_spatial_targets/aggregate/margulis_wus_sr.py`
- Test: `tests/test_aggregate_margulis_wus_sr.py`

**Interfaces:**
- Produces, in `aggregate/margulis_wus_sr.py`:
  - `MaskRule` (a frozen dataclass): `swe_threshold_m: float`,
    `check_month_day: str`, `min_year_fraction: float`, plus
    `@classmethod from_catalog() -> MaskRule`. It reads
    `catalog.source("margulis_wus_sr")["perennial_snow_mask"]` and raises
    `KeyError` / `ValueError` on a missing or invalid field.
  - `mask_fingerprint(rule: MaskRule, years: list[int]) -> str`, per the
    Global Constraints.
  - `classify_perennial(held: ArrayLike, valid: ArrayLike, min_year_fraction: float) -> np.ndarray`.
    It takes int counts (scalars, numpy or xarray) and returns an int8 0/1
    array of the broadcast shape.
  - `ensure_perennial_snow_mask(datastore: Path, rule: MaskRule | None = None) -> Path`.
    `rule` defaults to `MaskRule.from_catalog()`. It returns the mask path,
    and rebuilds the file only when its stored fingerprint differs from the
    current one.
  - `make_perennial_weight_filter(mask_path: Path) -> Callable[[pd.DataFrame, np.ndarray, np.ndarray], pd.DataFrame]`.
    This loads `perennial` into memory once.
  - `mask_stamp_attrs(mask_path: Path) -> dict[str, str | float]`. It
    returns the four `perennial_snow_mask_*` attrs read from the mask
    file.
- Mask NC contents:
  - `perennial` (int8, `flag_values=[0, 1]`,
    `flag_meanings="seasonal perennial"`), `n_years_held` (int16) and
    `n_years_valid` (int16), on the consolidated `lat` / `lon` coords,
    copied exactly.
  - A `crs` variable and coordinate attrs, following the consolidated
    file's pattern.
  - Global attrs: `Conventions="CF-1.6"`, the four `perennial_snow_mask_*`
    attrs, `perennial_snow_mask_years` (comma-separated, sorted),
    `perennial_snow_mask_n_cells` and `perennial_snow_mask_area_km2`.
  - The `history` timestamp comes from `max(mtime)` of the input files via
    `lineage.iso_from_mtime`, not `now()`, so a rebuild is deterministic.
- The builder:
  - Uses `ds["SWE"].sel(time=f"{year}-{check_month_day}")` per
    `daily/margulis_wus_sr_daily_*.nc`. A file with no such timestep is
    skipped with an INFO log.
  - Requires every year's `lat` / `lon` to be identical, and raises
    otherwise.
  - Writes atomically: a temp file in the same directory, then
    `os.replace`.
  - The filter raises `ValueError(f"... {n} weight rows reference cells absent from the perennial mask grid ...")`
    if any `(cell_y, cell_x)` is missing from the mask coords. Match through
    `pd.Index(mask.lat.values).get_indexer(cell_y)`; any `-1` counts as
    missing.

- [ ] **Step 1: Write the failing tests** in `tests/test_aggregate_margulis_wus_sr.py`, with a tiny synthetic datastore of 4 calendar-year daily files on a 3×4 grid.

```python
def test_classify_exactly_half_not_perennial():
    assert classify_perennial(held=2, valid=4, min_year_fraction=0.5) == 0
def test_classify_more_than_half_perennial():
    assert classify_perennial(held=3, valid=4, min_year_fraction=0.5) == 1
def test_classify_nan_years_excluded_from_denominator():
    # held=2 of valid=3 (4th year NaN at that pixel) -> perennial
def test_classify_all_nan_not_perennial():
    # valid=0 -> 0
def test_mask_rule_from_catalog_values():
    r = MaskRule.from_catalog()
    assert (r.swe_threshold_m, r.check_month_day, r.min_year_fraction) == (0.1, "09-30", 0.5)
def test_ensure_mask_builds_expected_cells_and_attrs(tmp_path): ...
def test_ensure_mask_reuses_matching_file(tmp_path):
    # second call: mtime_ns unchanged
def test_ensure_mask_rebuilds_on_threshold_change(tmp_path):
    # rule with swe_threshold_m=0.05 -> new fingerprint, file rewritten
def test_ensure_mask_rebuilds_when_year_added(tmp_path): ...
def test_ensure_mask_skips_year_without_check_date(tmp_path, caplog):
    # a 5th file ending 2020-06-30 -> not in perennial_snow_mask_years; INFO logged
def test_ensure_mask_cf_attrs(tmp_path):
    # Conventions CF-1.6; lat/lon standard_name/units/axis; flag_values/flag_meanings on `perennial`
def test_perennial_weight_filter_drops_masked_rows(tmp_path): ...
def test_perennial_weight_filter_unmatched_coords_raise(tmp_path):
    with pytest.raises(ValueError, match="absent from the perennial mask grid"): ...
def test_mask_fingerprint_stable_and_sensitive():
    assert mask_fingerprint(r, [1985, 1986]) == mask_fingerprint(r, [1986, 1985])
    assert mask_fingerprint(r, [1985]) != mask_fingerprint(r, [1985, 1986])
```

- [ ] **Step 2: Run them and confirm they fail.** Run `pixi run -e dev pytest tests/test_aggregate_margulis_wus_sr.py -v`. Expected: the new tests FAIL with ImportError.
- [ ] **Step 3: Implement** the catalog block and the module functions above.
- [ ] **Step 4: Run the Step 2 command and confirm the tests pass.** Also run `pixi run -e dev pytest tests/test_catalog.py -v`.
- [ ] **Step 5: Commit.** `feat(#363): Margulis perennial-snow mask builder + weight filter`.

### Task 4: Wire the mask into `aggregate_margulis_wus_sr`

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/margulis_wus_sr.py` (`aggregate_margulis_wus_sr`, module docstring)
- Test: `tests/test_aggregate_margulis_wus_sr.py`

**Interfaces:**
- Consumes:
  - from Task 1: `SourceAdapter.stamp_attrs`;
  - from Task 2: `SourceAdapter.weight_filter`;
  - from Task 3: `ensure_perennial_snow_mask`,
    `make_perennial_weight_filter`, `mask_stamp_attrs`.
- Behaviour:
  - `aggregate_margulis_wus_sr` resolves the datastore the same way the
    driver does (via the project loaded from `workdir`). It calls
    `ensure_perennial_snow_mask(datastore)`, then calls `aggregate_source`
    with
    `dataclasses.replace(ADAPTER, weight_filter=make_perennial_weight_filter(p), stamp_attrs=mask_stamp_attrs(p))`.
  - The module-level `ADAPTER` keeps `stat_method="mean"` and no filter,
    so importing it stays side-effect free.
  - The module docstring is rewritten to explain the mask, `mean` and the
    weight filter, and why `masked_mean` is avoided (#364).

- [ ] **Step 1: Write the failing end-to-end tests** with real gdptools: a synthetic datastore of 4 years on a 6×8 grid, and three HRUs.
  - A is fully perennial.
  - B is half perennial.
  - C has no perennial cells and contains one in-grid NaN pixel.

```python
def test_end_to_end_fully_perennial_hru_nan(tmp_project):
    assert np.isnan(ds["SWE"].sel({id_col: A})).all()          # never 0.0
def test_end_to_end_partial_hru_mean_of_seasonal(tmp_project):
    np.testing.assert_allclose(ds["SWE"].sel({id_col: B}), expected_seasonal_mean, rtol=1e-6)
def test_end_to_end_unmasked_hru_bit_identical(tmp_project):
    # same HRU C aggregated with plain ADAPTER (no mask) -> identical values incl. NaN positions
    np.testing.assert_array_equal(masked_C, unmasked_C)
def test_end_to_end_stamps_fingerprint(tmp_project):
    assert ds.attrs["perennial_snow_mask_fingerprint"] == mask_fingerprint(rule, years)
def test_end_to_end_premask_year_reaggregated(tmp_project):
    # pre-write a per-year NC without the attr -> rerun -> attr present
```

- [ ] **Step 2: Run them and confirm they fail.** Run `pixi run -e dev pytest tests/test_aggregate_margulis_wus_sr.py -k end_to_end -v`.
- [ ] **Step 3: Implement the wiring.**
- [ ] **Step 4: Run the Step 2 command and confirm the tests pass.**
- [ ] **Step 5: Commit.** `feat(#363): apply perennial-snow mask in Margulis aggregation`.

### Task 5: SWE target — record the fingerprint and include it in the cache key

**Files:**
- Modify: `src/nhf_spatial_targets/targets/_adapter.py` (`TargetAdapter`)
- Modify: `src/nhf_spatial_targets/targets/_driver.py` (around line 233, the attrs; around line 275, the fingerprint)
- Modify: `src/nhf_spatial_targets/targets/_intermediates.py:81` (`target_config_fingerprint`)
- Modify: `src/nhf_spatial_targets/targets/swe.py` (the `ADAPTER` declaration)
- Test: `tests/test_targets_swe.py`

**Interfaces:**
- Produces:
  - `TargetAdapter.input_provenance: Callable[[Project], dict[str, str]] | None = None`.
  - `target_config_fingerprint(project, target_name, input_provenance: Mapping[str, str] | None = None) -> str`.
    When the mapping is non-empty it is added to the hashed payload under
    the key `"input_provenance"`. When it is `None` or empty, the payload
    and the hash are **unchanged**, so existing intermediates of other
    targets are not invalidated.
  - The driver calls `adapter.input_provenance(project)` once per build.
    It passes the result to the fingerprint and merges it into the target
    NC's global attrs, in both the year-chunked and the single-shot paths.
  - `swe.margulis_input_provenance(project) -> dict[str, str]`:
    - It returns `{}` when `margulis_wus_sr` is not among the resolved
      SWE sources.
    - Otherwise it reads `perennial_snow_mask_fingerprint` from every
      Margulis per-year aggregated NC in the configured period. It returns
      `{"margulis_wus_sr_perennial_snow_mask_fingerprint": fp}` when all
      files agree.
    - It raises `ValueError` naming the years when any file lacks the attr
      ("pre-mask Margulis aggregate; re-run `agg margulis-wus-sr`") or when
      the files disagree.

- [ ] **Step 1: Write the failing tests** in `tests/test_targets_swe.py`.

```python
def test_margulis_provenance_empty_when_not_a_source(project_without_margulis):
    assert swe.margulis_input_provenance(p) == {}
def test_margulis_provenance_reads_fingerprint(project_with_stamped_margulis):
    assert swe.margulis_input_provenance(p) == {"margulis_wus_sr_perennial_snow_mask_fingerprint": "abc123def456"}
def test_margulis_mask_fingerprint_missing_raises(project_with_unstamped_margulis):
    with pytest.raises(ValueError, match="pre-mask Margulis aggregate"): ...
def test_margulis_mask_fingerprint_mixed_raises(project_mixed):
    with pytest.raises(ValueError, match="2004"): ...
def test_target_fingerprint_unchanged_without_provenance(project):
    assert target_config_fingerprint(p, "snow_water_equivalent") == target_config_fingerprint(p, "snow_water_equivalent", {})
def test_target_fingerprint_changes_with_mask(project):
    assert fp({"...": "aaa"}) != fp({"...": "bbb"})
def test_swe_target_nc_carries_mask_fingerprint(built_swe_target):
    assert ds.attrs["margulis_wus_sr_perennial_snow_mask_fingerprint"] == "abc123def456"
```

Update the existing SWE fixtures that write Margulis aggregates, so they
stamp the attr.

- [ ] **Step 2: Run them and confirm they fail.** Run `pixi run -e dev pytest tests/test_targets_swe.py -k "provenance or mask_fingerprint or target_fingerprint" -v`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the Step 2 command and confirm the tests pass.** Also run `pixi run -e dev pytest tests/test_targets_swe.py -v`, which is one file and allowed.
- [ ] **Step 5: Commit.** `feat(#363): SWE target records Margulis mask fingerprint; keys intermediates on it`.

### Task 6: Manifest projection — Margulis mask params

**Files:**
- Modify: `src/nhf_spatial_targets/rebuild_manifest.py:276-380` (generalize `_aggregate_nc_params`)
- Test: `tests/test_rebuild_manifest.py`, `tests/test_release_payload.py`

**Interfaces:**
- Behaviour:
  - For `source_key == "margulis_wus_sr"`, `_aggregate_nc_params` returns
    the four `perennial_snow_mask_*` global attrs, as `str` / `float`, from
    the first sorted per-year NC that carries
    `perennial_snow_mask_fingerprint`.
  - It returns `{}` when no file carries it.
  - A file with NetCDF magic that cannot be opened raises `ValueError`,
    mirroring `read_scf_threshold_stamp`.
  - ua_swe behaviour is unchanged.
  - The mask NC under `derived/` appears in the Margulis consolidate step's
    outputs. Pin this with a test, and fix the projection if the claim is
    false.

- [ ] **Step 1: Write the failing tests.**

```python
def test_margulis_aggregate_step_params_carry_mask(tmp_project_with_margulis):
    step = _step(manifest, "aggregate", "margulis_wus_sr")
    assert step["params"]["perennial_snow_mask_fingerprint"] == "abc123def456"
    assert step["params"]["perennial_snow_mask_swe_threshold_m"] == 0.1
def test_margulis_consolidate_outputs_include_mask(tmp_project_with_margulis):
    assert any(o["path"].endswith("derived/perennial_snow_mask.nc") for o in _step(manifest, "consolidate", "margulis_wus_sr")["outputs"])
def test_margulis_params_absent_when_unstamped(...):
    assert "perennial_snow_mask_fingerprint" not in step.get("params", {})
def test_rebuild_deterministic_with_mask(...):
    # two rebuilds -> byte-identical manifest.json
```

In `tests/test_release_payload.py`, pin the spec's call that the mask
ships in the Margulis source child. No payload code change is expected.

```python
def test_margulis_source_child_includes_perennial_mask(tmp_datastore):
    assert any(p.name == "perennial_snow_mask.nc" for p in plan_source_child(...).files)
```

- [ ] **Step 2: Run them and confirm they fail.** Run `pixi run -e dev pytest tests/test_rebuild_manifest.py -k margulis -v`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the Step 2 command and confirm the tests pass.**
- [ ] **Step 5: Commit.** `feat(#363): manifest records Margulis perennial-mask params`.

### Task 7: Docs and inspect notebooks

**Files:**
- Modify: `docs/sources/margulis_wus_sr.md`
- Modify: `docs/architecture/transformation-pipeline.md`
- Modify: `notebooks/aggregated/inspect_aggregated_swe.ipynb`
- Modify: `notebooks/targets/inspect_target_swe.ipynb`

**Interfaces:**
- Consumes:
  - the mask NC layout and attr names (Global Constraints);
  - the pre-mask archive path
    `<project>/archive/margulis_premask_2026-10-28/`, which holds
    `margulis_wus_sr/` and `swe_targets.nc`.

Steps:

- [ ] **Step 1: Docs.**
  - `margulis_wus_sr.md` covers:
    - the rule;
    - the mask path;
    - automatic rebuilds, and forcing one by deleting the mask file;
    - that HRUs which don't touch the mask are unchanged;
    - that fully glaciated HRUs drop Margulis;
    - that lake and coast HRUs are unaffected.
  - `transformation-pipeline.md` gets a short subsection, "Static pixel
    masks under `mean`: `weight_filter`", citing #363 and #364.
  - Run the Documentation Sync Gate scan from CLAUDE.md. Update CLAUDE.md
    only if a command or invariant changed (none is expected), and say so
    in the PR.
- [ ] **Step 2: `inspect_aggregated_swe.ipynb`.** Add a section, "Perennial-snow mask (#363)", that implements notebook checks 1–5 from the spec.
  - Each check is a cell that prints a summary and ends in `assert`.
  - Check 4 must print `"no fully masked HRUs in this project"` rather than pass silently when the count is zero.
  - The "before" data is read from the archive path. If the archive is absent, the checks are skipped with a printed message.
  - Read aggregated NCs with the fill mask applied (`io_nc.mask_netcdf_default_fills` or `targets._io.read_aggregated_source`).
  - Use bullet lists in markdown, not tables with an empty leading header column.
- [ ] **Step 3: `inspect_target_swe.ipynb`.** Add check 6: `n_sources` before and after at the reporter HRUs (Oregon `hru_id` 7305, 7346, 7306 and 11845), and an assertion that `n_sources` dropped only at HRUs fully masked for Margulis.
- [ ] **Step 4: Verify.**
  - Execute neither notebook against real data in CI.
  - Run `pixi run -e dev python -I -c "import json,sys; [json.load(open(p)) for p in sys.argv[1:]]" notebooks/aggregated/inspect_aggregated_swe.ipynb notebooks/targets/inspect_target_swe.ipynb` and confirm both files are valid JSON.
- [ ] **Step 5: Commit.** `docs(#363): Margulis perennial mask docs + notebook checks`.

### Task 8: Branch verification and PR

- [ ] **Step 1: Format and lint.** Run `pixi run -e dev fmt && pixi run -e dev lint`; it must be clean.
- [ ] **Step 2: Rebase and open the PR.**
  - `git fetch origin && git rebase origin/main`.
  - Push the branch and open a PR titled "feat(#363): mask Margulis perennial-snow pixels before HRU aggregation", with "Closes #363".
  - The body lists the spec's rollout steps as a post-merge checklist, notes #364, and notes that docs were scanned.
- [ ] **Step 3: Watch CI once,** with `gh pr checks --watch`. Fix any failures on the branch.
- [ ] **Step 4: Final review.** A whole-branch review by an **Opus** reviewer, focused on grid-index correctness, fill and NaN semantics, and provenance determinism.

## Post-merge rollout (not part of the PR; every `sbatch` needs Rich's OK)

The rollout is the spec's "Rollout" section:
- **Oregon:** archive the pre-mask files, run `agg margulis-wus-sr`, run
  `run-swe`, then `rebuild-manifest` and `validate`, then the notebook
  checks.
- **gfv2:** `rebuild-manifest` only.
