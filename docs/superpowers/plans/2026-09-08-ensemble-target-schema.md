# Ensemble Calibration Target Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Target NetCDFs retain per-source ensemble members as named variables and gain `ensemble_mean` / `ensemble_std`, and recharge / soil-moisture can normalize each source over its own complete-year period of record.

**Architecture:** Every target loader already builds a `dict[source_key, DataArray]` immediately before reducing it to bounds. That dict is the ensemble; it is carried through `SourceLoaderResult` to `write_bounds_target`, which emits the members and derives the two ensemble statistics from the same stack. The year-chunked path needs no new machinery because the stitcher is already generic over `data_vars` — only two hard-coded dtype maps name variables explicitly.

**Tech Stack:** Python ≥3.11, xarray, pandas, numpy, netCDF4, pytest, ruff, pixi.

**Spec:** `docs/superpowers/specs/2026-09-08-ensemble-target-schema-design.md`

**Issue:** [#338](https://github.com/rmcd-mscb/nhf-spatial-targets/issues/338)

## Global Constraints

- **Do NOT run the full pytest suite locally.** This is an HPC login node; the repo convention is `pixi run -e dev fmt && pixi run -e dev lint` locally, then push and let GitHub Actions run the suite.
- **Targeted test command.** `pixi run -e dev test` is `pytest tests/ -n auto ...`, so appending `-k` still collects the whole suite under xdist (~3.5 min). Every "run the test" step in this plan means the file-scoped form instead:

  ```bash
  pixi run -e dev pytest tests/<file>.py -k <pattern> -q
  ```

  That is ~2s of pytest (~55s wall, dominated by pixi env resolution). Where a step below writes `pixi run -e dev test -k <pattern>`, use the file-scoped form against the test file that task touches.
- **Always commit via `pixi run git commit`, never bare `git commit`.** A PreToolUse hook blocks the bare form.
- **Never commit to `main`.** Each PR below gets its own branch off `main`, named `<type>/338-<slug>`.
- Stage files explicitly by path. Never `git add -A` or `git add .`.
- `from __future__ import annotations` at the top of every module touched.
- Ruff line length 88.
- Type hints on all public functions; docstrings on public functions only.
- Config schema additions require all four of: `defaults.py:DEFAULTS`, `init_run.py:_CONFIG_TEMPLATE`, `tests/test_init_run.py`, `upgrade_config.py:OPTIONAL_CONFIG_FEATURES`.
- Docs sync gate applies to every PR: CLAUDE.md, `docs/architecture/transformation-pipeline.md`, `docs/references/calibration-target-recipes.md`, `docs/architecture/nc-encoding-policy.md`, SLURM headers.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/nhf_spatial_targets/targets/_combine.py` | multi-source reducers | add `ensemble_stats` |
| `src/nhf_spatial_targets/targets/_shims.py` | per-source contracts | add `label_members` |
| `src/nhf_spatial_targets/targets/_adapter.py` | `SourceLoaderResult` | add `members` field |
| `src/nhf_spatial_targets/targets/{aet,run,rch,som,swe}.py` | per-target loaders | populate `members` |
| `src/nhf_spatial_targets/targets/_driver.py` | pipeline | forward `members` + `emit_members` |
| `src/nhf_spatial_targets/targets/_writers.py` | NC emission | emit members + stats + attrs; dynamic dtypes |
| `src/nhf_spatial_targets/targets/_intermediates.py` | year stitch | dynamic dtypes |
| `src/nhf_spatial_targets/normalize/methods.py` | per-HRU transforms | add `complete_years_window` |
| `src/nhf_spatial_targets/defaults.py` | config schema | `emit_members` |
| `src/nhf_spatial_targets/init_run.py` | config template | `emit_members` stub |
| `src/nhf_spatial_targets/upgrade_config.py` | drift report | `emit_members` feature |
| `src/nhf_spatial_targets/validate.py` | preflight | accept `per_source_por` |

---

# PR 1 — Ensemble members and statistics

Branch: `feature/338-ensemble-members`

---

### Task 1: `ensemble_stats` reducer

**Files:**
- Modify: `src/nhf_spatial_targets/targets/_combine.py`
- Test: `tests/test_targets_common.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ensemble_stats(members: dict[str, xr.DataArray], n_sources: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]` returning `(mean, std)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_targets_common.py`:

```python
def _member(values: list[list[float]]) -> xr.DataArray:
    """Build a (time=2, nhm_id=3) float32 member array from nested lists."""
    return xr.DataArray(
        np.array(values, dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={
            "time": pd.date_range("2000-01-01", periods=2, freq="MS"),
            "nhm_id": [1, 2, 3],
        },
    )


def test_ensemble_stats_mean_and_std_across_three_members():
    from nhf_spatial_targets.targets._combine import (
        ensemble_stats,
        multi_source_nanminmax,
    )

    members = {
        "a": _member([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        "b": _member([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]]),
        "c": _member([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0]]),
    }
    _, _, n_sources = multi_source_nanminmax(members)
    mean, std = ensemble_stats(members, n_sources)

    # mean of (1, 3, 5) == 3; population std (ddof=0) of (1, 3, 5) == 1.632993
    assert np.allclose(mean.values[0], [3.0, 4.0, 5.0])
    assert np.allclose(std.values[0], [1.6329932, 1.6329932, 1.6329932])


def test_ensemble_stats_masks_std_where_fewer_than_two_sources():
    from nhf_spatial_targets.targets._combine import (
        ensemble_stats,
        multi_source_nanminmax,
    )

    nan = float("nan")
    members = {
        # HRU 1: three finite. HRU 2: two finite. HRU 3: one finite.
        "a": _member([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        "b": _member([[3.0, 4.0, nan], [3.0, 4.0, nan]]),
        "c": _member([[5.0, nan, nan], [5.0, nan, nan]]),
    }
    _, _, n_sources = multi_source_nanminmax(members)
    mean, std = ensemble_stats(members, n_sources)

    assert list(n_sources.values[0]) == [3, 2, 1]
    # Mean is defined wherever >=1 source is finite.
    assert np.allclose(mean.values[0], [3.0, 3.0, 3.0])
    # std defined at n>=2; population std of (2, 4) is 1.0. NaN at n == 1.
    assert np.isclose(std.values[0][0], 1.6329932)
    assert np.isclose(std.values[0][1], 1.0)
    assert np.isnan(std.values[0][2])


def test_ensemble_stats_all_nan_cell_yields_nan_mean_and_std():
    from nhf_spatial_targets.targets._combine import (
        ensemble_stats,
        multi_source_nanminmax,
    )

    nan = float("nan")
    members = {
        "a": _member([[nan, nan, nan], [nan, nan, nan]]),
        "b": _member([[nan, nan, nan], [nan, nan, nan]]),
    }
    _, _, n_sources = multi_source_nanminmax(members)
    mean, std = ensemble_stats(members, n_sources)

    assert int(n_sources.values[0][0]) == 0
    assert np.isnan(mean.values[0][0])
    assert np.isnan(std.values[0][0])


def test_ensemble_stats_rejects_empty_members():
    from nhf_spatial_targets.targets._combine import ensemble_stats

    dummy = _member([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="empty members dict"):
        ensemble_stats({}, dummy)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pixi run -e dev test -k ensemble_stats -v
```

Expected: FAIL with `ImportError: cannot import name 'ensemble_stats'`.

- [ ] **Step 3: Implement `ensemble_stats`**

Append to `src/nhf_spatial_targets/targets/_combine.py`:

```python
def ensemble_stats(
    members: dict[str, xr.DataArray],
    n_sources: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """NaN-aware ensemble mean and population standard deviation.

    Both are reduced over a stacked ``source`` dim with ``skipna=True``, so
    the mean is defined wherever at least one source is finite.

    ``std`` is masked to NaN wherever ``n_sources < 2``. With a single finite
    source the population standard deviation is exactly 0, which a downstream
    calibration weight would read as perfect inter-source agreement rather
    than as "only one source was available here". Masking makes the
    distinction explicit; ``n_sources`` remains the authoritative coverage
    diagnostic.

    Parameters
    ----------
    members
        Mapping from source key to per-source DataArray. All must share
        dims and coords (typically ``(time, id_col)``).
    n_sources
        Per-cell finite-source count from
        :func:`multi_source_nanminmax`, used for the ``std`` mask.

    Returns
    -------
    mean, std
        ``(time, id_col)`` arrays matching the members' shape.

    Raises
    ------
    ValueError
        If ``members`` is empty.
    """
    keys = list(members.keys())
    if not keys:
        raise ValueError("ensemble_stats: empty members dict")
    stacked = xr.concat([members[k] for k in keys], dim=xr.Variable("source", keys))
    mean = stacked.mean(dim="source", skipna=True)
    std = stacked.std(dim="source", skipna=True, ddof=0).where(n_sources >= 2)
    return mean, std
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pixi run -e dev test -k ensemble_stats -v
```

Expected: 4 passed.

- [ ] **Step 5: Format, lint, commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/targets/_combine.py tests/test_targets_common.py
pixi run git commit -m "feat(#338): add ensemble_stats reducer with n<2 std mask"
```

---

### Task 2: Carry members through `SourceLoaderResult`

No output change yet — this task only makes the ensemble reachable by the writer, so it can be reviewed purely as plumbing.

**Files:**
- Modify: `src/nhf_spatial_targets/targets/_adapter.py` (dataclass `SourceLoaderResult`, ends line 345)
- Modify: `src/nhf_spatial_targets/targets/_shims.py`
- Modify: `src/nhf_spatial_targets/targets/aet.py`, `run.py`, `rch.py`, `som.py`, `swe.py`
- Modify: `src/nhf_spatial_targets/targets/_driver.py` (`_apply_forced_zero`, ends line 141)
- Test: `tests/test_targets_common.py`

**Interfaces:**
- Consumes: `ensemble_stats` from Task 1 (not called yet).
- Produces: `SourceLoaderResult.members: dict[str, xr.DataArray] | None`; `_shims.label_members(members, shims) -> dict[str, xr.DataArray]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_targets_common.py`:

```python
def test_source_loader_result_members_defaults_to_none():
    from nhf_spatial_targets.targets._adapter import SourceLoaderResult

    da = _member([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    result = SourceLoaderResult(
        lower=da,
        upper=da,
        n_sources=da,
        n_sources_count=1,
        time_index=pd.date_range("2000-01-01", periods=2, freq="MS"),
        time_offset_unit=pd.offsets.MonthBegin(1),
        extra_attrs={},
    )
    assert result.members is None


def test_label_members_stamps_long_name_from_shim_description():
    from nhf_spatial_targets.targets._shims import SourceShim, label_members

    shims = {
        "era5_land": SourceShim(
            source_key="era5_land",
            aggregated_var="ro",
            description="ERA5-Land runoff (m/month -> mm/month)",
            to_common_units=lambda da: da,
        )
    }
    members = {"era5_land": _member([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])}
    labeled = label_members(members, shims)

    assert (
        labeled["era5_land"].attrs["long_name"]
        == "ERA5-Land runoff (m/month -> mm/month)"
    )
    # Same dict object semantics: keys preserved, values still DataArrays.
    assert list(labeled) == ["era5_land"]


def test_label_members_ignores_keys_absent_from_shims():
    from nhf_spatial_targets.targets._shims import label_members

    members = {"unknown_src": _member([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])}
    labeled = label_members(members, {})
    assert "long_name" not in labeled["unknown_src"].attrs
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pixi run -e dev test -k "members_defaults_to_none or label_members" -v
```

Expected: FAIL — `SourceLoaderResult` has no `members`, and `label_members` is not importable.

- [ ] **Step 3: Add the `members` field**

In `src/nhf_spatial_targets/targets/_adapter.py`, add to the `SourceLoaderResult` docstring's Attributes section, immediately after the `extras` entry:

```
    members
        Optional per-source contributions keyed by source key, in the
        order the target config lists them. This is the same dict the
        loader reduces to ``lower`` / ``upper``; carrying it through lets
        the writer emit the ensemble members and derive
        ``ensemble_mean`` / ``ensemble_std`` without recomputing.
        ``None`` means the target has no member decomposition (SCA, whose
        bounds are a CI interval rather than a member min/max).
```

Then add the field after `extras`:

```python
    extras: dict = field(default_factory=dict)
    members: "dict[str, xr.DataArray] | None" = None
```

- [ ] **Step 4: Add `label_members` to `_shims.py`**

Append to `src/nhf_spatial_targets/targets/_shims.py`:

```python
def label_members(
    members: dict[str, "xr.DataArray"],
    shims: dict[str, SourceShim],
) -> dict[str, "xr.DataArray"]:
    """Stamp each member's ``long_name`` from its shim ``description``.

    The target writer emits members as named data variables and reads
    ``long_name`` off each one. Setting it here keeps the human-readable
    source label in the single place that already owns it (the SHIMS
    registry) instead of duplicating a label map in the writer.

    Members whose key is absent from ``shims`` are left untouched.
    Returns the same dict for call-site convenience.
    """
    for key, da in members.items():
        shim = shims.get(key)
        if shim is not None:
            da.attrs["long_name"] = shim.description
    return members
```

Add `import xarray as xr` to the module's imports if it is not already present.

- [ ] **Step 5: Run the two new unit tests**

```bash
pixi run -e dev test -k "members_defaults_to_none or label_members" -v
```

Expected: 3 passed.

- [ ] **Step 6: Populate `members` in the four single-shot loaders**

In `src/nhf_spatial_targets/targets/aet.py`, replace:

```python
    lower, upper, n_sources = multi_source_nanminmax(sources_in_day)
```

with:

```python
    label_members(sources_in_day, shims)
    lower, upper, n_sources = multi_source_nanminmax(sources_in_day)
```

and add `members=sources_in_day,` to the `SourceLoaderResult(...)` call. Add `label_members` to the existing `from nhf_spatial_targets.targets._shims import (...)` block.

Apply the identical pattern to the other three:

| File | dict variable | `SourceLoaderResult` call |
|---|---|---|
| `run.py` | `sources_cfs` | add `members=sources_cfs,` |
| `rch.py` | `sources_normalized` | add `members=sources_normalized,` |
| `som.py` `_load_monthly` | `sources_monthly_norm` | add `members=sources_monthly_norm,` |
| `som.py` `_load_annual` | `sources_annual_norm` | add `members=sources_annual_norm,` |

In `som.py` both loaders build `shims = shims_by_key(SHIMS)` *after* the reduce; move that line above the `label_members` call in each.

- [ ] **Step 7: Populate `members` in the SWE year loader with constant schema**

In `src/nhf_spatial_targets/targets/swe.py:_load_year`, the `except OutsideCoverageError: ... continue` branch currently omits the source entirely, so the member set varies from year to year. The stitcher opens per-year files with `join="exact"`, so a ragged schema breaks the stitch. Replace the loop body's exception branch to insert an all-NaN member instead of skipping.

Replace:

```python
    year_sources: dict[str, xr.DataArray] = {}
    for src_label in sources:
        shim = shims[src_label]
        try:
            da_native = read_aggregated_source(
                project,
                shim.source_key,
                shim.aggregated_var,
                (year_start, year_end),
                chunks={"time": 365, id_col: -1},
            )
        except OutsideCoverageError:
            logger.info(
                "swe year %d: source '%s' has no data; contributes NaN",
                year,
                src_label,
            )
            continue
        check_hru_coords(da_native, fabric_hru_ids, id_col, src_label)
        da_mm = shim.to_common_units(da_native)
        da_in = mm_to_inches(da_mm)
        year_sources[src_label] = reindex_to_day_start(da_in, year_master_idx)

    if not year_sources:
```

with:

```python
    year_sources: dict[str, xr.DataArray] = {}
    covered: list[str] = []
    for src_label in sources:
        shim = shims[src_label]
        try:
            da_native = read_aggregated_source(
                project,
                shim.source_key,
                shim.aggregated_var,
                (year_start, year_end),
                chunks={"time": 365, id_col: -1},
            )
        except OutsideCoverageError:
            logger.info(
                "swe year %d: source '%s' has no data; contributes NaN",
                year,
                src_label,
            )
            # Emit an all-NaN member rather than omitting the key: the
            # per-year NCs are stitched with join="exact", so every year
            # must carry the same member variables (#338).
            year_sources[src_label] = xr.DataArray(
                np.full(
                    (len(year_master_idx), len(fabric_hru_ids)),
                    np.nan,
                    dtype=np.float32,
                ),
                dims=("time", id_col),
                coords={"time": year_master_idx, id_col: fabric_hru_ids},
            )
            continue
        check_hru_coords(da_native, fabric_hru_ids, id_col, src_label)
        da_mm = shim.to_common_units(da_native)
        da_in = mm_to_inches(da_mm)
        year_sources[src_label] = reindex_to_day_start(da_in, year_master_idx)
        covered.append(src_label)

    if not covered:
```

Then add `label_members(year_sources, shims)` before the `multi_source_nanminmax` call and `members=year_sources,` to the `SourceLoaderResult(...)` call. Ensure `numpy as np` and `xarray as xr` are imported in `swe.py`.

- [ ] **Step 8: Forward members through the SCA forced-zero rebuild**

In `src/nhf_spatial_targets/targets/_driver.py:_apply_forced_zero`, add `members=result.members,` to the `SourceLoaderResult(...)` it constructs, so the field is not silently dropped. SCA supplies `members=None`, so this is a pass-through today; it prevents a future member-supplying forced-zero target from losing its ensemble.

- [ ] **Step 9: Run the target test modules**

```bash
pixi run -e dev test -k "targets_aet or targets_run or targets_rch or targets_som or targets_swe" -v
```

Expected: PASS. Nothing about the written output changed, so no existing assertion should move.

- [ ] **Step 10: Format, lint, commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/targets/_adapter.py \
        src/nhf_spatial_targets/targets/_shims.py \
        src/nhf_spatial_targets/targets/_driver.py \
        src/nhf_spatial_targets/targets/aet.py \
        src/nhf_spatial_targets/targets/run.py \
        src/nhf_spatial_targets/targets/rch.py \
        src/nhf_spatial_targets/targets/som.py \
        src/nhf_spatial_targets/targets/swe.py \
        tests/test_targets_common.py
pixi run git commit -m "feat(#338): carry per-source members through SourceLoaderResult"
```

---

### Task 3: Emit members and statistics from the writer

**Files:**
- Modify: `src/nhf_spatial_targets/targets/_writers.py` (`write_target_nc` lines 66-162, `write_bounds_target` lines 165-370)
- Modify: `src/nhf_spatial_targets/targets/_intermediates.py:481` (dtype map)
- Test: `tests/test_targets_common.py`

**Interfaces:**
- Consumes: `ensemble_stats` (Task 1), `SourceLoaderResult.members` (Task 2).
- Produces: `write_bounds_target(..., members: dict[str, xr.DataArray] | None = None, emit_members: bool = False)`; output NC global attrs `source_keys` and `members_emitted`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_targets_common.py`:

```python
def _bounds_call_kwargs(tmp_path: Path, members, emit_members: bool) -> dict:
    """Minimal kwargs for a write_bounds_target call over 2 months x 3 HRUs."""
    time_index = pd.date_range("2000-01-01", periods=2, freq="MS")
    hru_meta = pd.DataFrame(
        {
            "centroid_lat": [45.0, 45.1, 45.2],
            "centroid_lon": [-120.0, -120.1, -120.2],
            "centroid_x": [0.0, 1.0, 2.0],
            "centroid_y": [0.0, 1.0, 2.0],
        },
        index=pd.Index([1, 2, 3], name="nhm_id"),
    )
    return {
        "time_index": time_index,
        "time_offset_unit": pd.offsets.MonthBegin(1),
        "bounds_units": "cfs",
        "bounds_long_name_kind": "monthly runoff",
        "cell_methods": "time: sum",
        "output_path": tmp_path / "runoff_targets.nc",
        "title": "test target",
        "nn_title": "test target (NN-filled)",
        "extra_global_attrs": {"source": "a; b"},
        "hru_meta": hru_meta,
        "nn_fill": False,
        "nn_max_candidates": 10,
        "id_col": "nhm_id",
        "members": members,
        "emit_members": emit_members,
    }


def test_write_bounds_target_emits_members_and_stats(tmp_path: Path):
    from nhf_spatial_targets.targets._combine import multi_source_nanminmax
    from nhf_spatial_targets.targets._writers import write_bounds_target

    workdir = make_minimal_project(tmp_path)
    project = load(workdir)
    members = {
        "era5_land": _member([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        "mwbm_climgrid": _member([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]]),
    }
    lower, upper, n_sources = multi_source_nanminmax(members)

    kwargs = _bounds_call_kwargs(tmp_path, members, emit_members=True)
    write_bounds_target(
        project=project,
        lower=lower,
        upper=upper,
        n_sources=n_sources,
        n_sources_count=2,
        **kwargs,
    )

    ds = xr.open_dataset(kwargs["output_path"])
    try:
        assert "era5_land" in ds.data_vars
        assert "mwbm_climgrid" in ds.data_vars
        assert "ensemble_mean" in ds.data_vars
        assert "ensemble_std" in ds.data_vars
        assert ds.attrs["source_keys"] == "era5_land,mwbm_climgrid"
        assert ds.attrs["members_emitted"] == "true"
        assert ds.attrs["Conventions"] == "CF-1.8"
        assert ds["era5_land"].attrs["units"] == "cfs"
        assert ds["ensemble_std"].attrs["ancillary_variables"] == "n_sources"
        # mean of (1, 3) == 2; population std == 1
        assert np.allclose(ds["ensemble_mean"].values[0], [2.0, 3.0, 4.0])
        assert np.allclose(ds["ensemble_std"].values[0], [1.0, 1.0, 1.0])
        # lower/upper unchanged by member emission
        assert np.allclose(ds["lower_bound"].values[0], [1.0, 2.0, 3.0])
        assert np.allclose(ds["upper_bound"].values[0], [3.0, 4.0, 5.0])
        assert ds["era5_land"].dtype == np.float32
    finally:
        ds.close()


def test_write_bounds_target_omits_members_when_disabled(tmp_path: Path):
    from nhf_spatial_targets.targets._combine import multi_source_nanminmax
    from nhf_spatial_targets.targets._writers import write_bounds_target

    workdir = make_minimal_project(tmp_path)
    project = load(workdir)
    members = {
        "era5_land": _member([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        "mwbm_climgrid": _member([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]]),
    }
    lower, upper, n_sources = multi_source_nanminmax(members)

    kwargs = _bounds_call_kwargs(tmp_path, members, emit_members=False)
    write_bounds_target(
        project=project,
        lower=lower,
        upper=upper,
        n_sources=n_sources,
        n_sources_count=2,
        **kwargs,
    )

    ds = xr.open_dataset(kwargs["output_path"])
    try:
        assert "era5_land" not in ds.data_vars
        assert "ensemble_mean" not in ds.data_vars
        assert "ensemble_std" not in ds.data_vars
        assert set(ds.data_vars) == {"lower_bound", "upper_bound", "n_sources", "crs"}
        # source_keys is provenance and is recorded either way.
        assert ds.attrs["source_keys"] == "era5_land,mwbm_climgrid"
        assert ds.attrs["members_emitted"] == "false"
    finally:
        ds.close()


def test_write_bounds_target_raises_when_emit_requested_without_members(
    tmp_path: Path,
):
    from nhf_spatial_targets.targets._writers import write_bounds_target

    workdir = make_minimal_project(tmp_path)
    project = load(workdir)
    da = _member([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    kwargs = _bounds_call_kwargs(tmp_path, None, emit_members=True)
    with pytest.raises(ValueError, match="emit_members is True but no members"):
        write_bounds_target(
            project=project,
            lower=da,
            upper=da,
            n_sources=da.astype("int8"),
            n_sources_count=1,
            **kwargs,
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pixi run -e dev test -k "write_bounds_target_emits_members or omits_members_when_disabled or emit_requested_without_members" -v
```

Expected: FAIL — `write_bounds_target() got an unexpected keyword argument 'members'`.

- [ ] **Step 3: Add the parameters and member assembly to `write_bounds_target`**

In `src/nhf_spatial_targets/targets/_writers.py`, add two keyword-only parameters to the signature, after `target_key`:

```python
    target_key: str | None = None,
    members: dict[str, xr.DataArray] | None = None,
    emit_members: bool = False,
```

Add to the docstring's Parameters section:

```
    members
        Per-source contributions keyed by source key, from
        ``SourceLoaderResult.members``. Recorded as the ``source_keys``
        global attr whenever present, and written as named data
        variables when ``emit_members`` is True.
    emit_members
        Whether to write the members and the derived ``ensemble_mean`` /
        ``ensemble_std`` variables. Purely an output switch: the bounds
        and ``n_sources`` are byte-identical either way. Raises when
        True and ``members`` is empty, rather than silently ignoring the
        operator's config.
```

Then, immediately after the `n_sources.attrs.update(build_n_sources_attrs(n_sources_count))` line and before the `ds = xr.Dataset(...)` construction, insert:

```python
    if emit_members and not members:
        raise ValueError(
            "write_bounds_target: emit_members is True but no members were "
            "supplied by the target's source_loader. Set emit_members=False "
            "for targets without a member decomposition (e.g. SCA, whose "
            "bounds are a CI interval rather than a member min/max)."
        )

    data_vars: dict[str, xr.DataArray] = {
        "lower_bound": lower,
        "upper_bound": upper,
        "n_sources": n_sources,
    }
    if emit_members:
        from nhf_spatial_targets.targets._combine import ensemble_stats

        mean, std = ensemble_stats(members, n_sources)
        mean.name = "ensemble_mean"
        std.name = "ensemble_std"
        mean.attrs = {
            "units": bounds_units,
            "long_name": f"ensemble mean of {bounds_long_name_kind}",
            "cell_methods": cell_methods,
            "coordinates": "centroid_lat centroid_lon",
            "ancillary_variables": "n_sources",
        }
        std.attrs = {
            "units": bounds_units,
            "long_name": (
                f"ensemble standard deviation of {bounds_long_name_kind} "
                "(population, NaN where n_sources < 2)"
            ),
            "cell_methods": cell_methods,
            "coordinates": "centroid_lat centroid_lon",
            "ancillary_variables": "n_sources",
        }
        data_vars["ensemble_mean"] = mean
        data_vars["ensemble_std"] = std
        for key, member_da in members.items():
            member = member_da.copy()
            member.name = key
            member.attrs = {
                "units": bounds_units,
                "long_name": member_da.attrs.get("long_name") or (
                    f"{key} contribution to {bounds_long_name_kind}"
                ),
                "cell_methods": cell_methods,
                "coordinates": "centroid_lat centroid_lon",
            }
            data_vars[key] = member

    extra_global_attrs = dict(extra_global_attrs)
    extra_global_attrs["members_emitted"] = "true" if emit_members else "false"
    if members:
        extra_global_attrs["source_keys"] = ",".join(members)
```

Change the Dataset construction from the explicit three-variable literal to:

```python
    ds = xr.Dataset(
        data_vars,
        coords={
```

leaving the `coords={...}` block unchanged.

**Do not add members to the NN-filled companion.** The `nn_fill` branch later in
`write_bounds_target` builds `filled_ds` from `nn_fill_bounds(ds_loaded, ...)`,
which fills `lower_bound` / `upper_bound` only. Leave that branch alone: per spec
§3.6 the companion carries just the filled bounds plus the `nn_filled` flag,
because "filling" an individual member would fabricate a source observation at an
HRU that source never covered.

- [ ] **Step 4: Make the dtype map and grid_mapping loop dynamic**

In `write_target_nc`, replace the fixed `grid_mapping` loop:

```python
    for _v in ("lower_bound", "upper_bound", "n_sources", "nn_filled"):
        if _v in ds.data_vars:
            ds[_v].attrs["grid_mapping"] = "crs"
```

with:

```python
    # Every data variable except the grid-mapping container itself points at
    # the crs variable. Iterating data_vars rather than a fixed name list
    # keeps ensemble members and statistics covered as the schema grows.
    for _v in ds.data_vars:
        if _v != "crs":
            ds[_v].attrs["grid_mapping"] = "crs"
```

Replace the `target_dtypes` block:

```python
    target_dtypes = {
        v: "float32" for v in ("lower_bound", "upper_bound") if v in ds.data_vars
    }
    target_dtypes.update(
        {v: "int8" for v in ("n_sources", "nn_filled") if v in ds.data_vars}
    )
```

with:

```python
    # int8 for the two flag diagnostics, float32 for every other data
    # variable (bounds, ensemble members, ensemble statistics). Derived from
    # data_vars rather than a fixed list so a new variable cannot silently
    # fall through to float64 on disk. `crs` is a 0-dim int32 grid-mapping
    # container minted above and carries no encoding.
    target_dtypes = {
        v: "int8" for v in ("n_sources", "nn_filled") if v in ds.data_vars
    }
    target_dtypes.update(
        {v: "float32" for v in ds.data_vars if v not in target_dtypes and v != "crs"}
    )
```

Bump the Conventions default in the same function:

```python
    ds.attrs.setdefault("Conventions", "CF-1.8")
```

- [ ] **Step 5: Apply the same dtype change to the stitcher**

In `src/nhf_spatial_targets/targets/_intermediates.py:stitch_year_chunks_to_target`, replace the `target_dtypes` block with the identical derivation, and bump its `ds.attrs.setdefault("Conventions", "CF-1.6")` to `"CF-1.8"`:

```python
    target_dtypes = {
        v: "int8" for v in ("n_sources", "nn_filled") if v in ds.data_vars
    }
    target_dtypes.update(
        {v: "float32" for v in ds.data_vars if v not in target_dtypes and v != "crs"}
    )
```

- [ ] **Step 6: Run the writer tests**

```bash
pixi run -e dev test -k "write_bounds_target or stitch_year_chunks" -v
```

Expected: PASS, including the three new tests.

- [ ] **Step 7: Update the CF-compliance test**

In `tests/test_cf_compliance.py`, any assertion of `Conventions == "CF-1.6"` on a **target** NC becomes `"CF-1.8"`. Leave consolidated / aggregated NC assertions at CF-1.6 — this bump is scoped to target outputs only. Run:

```bash
pixi run -e dev test -k cf_compliance -v
```

- [ ] **Step 8: Update the docs**

- `CLAUDE.md`: in the "Data & Catalog Conventions" bullet on CF compliance, note that **target** NCs are CF-1.8 while consolidated and aggregated NCs remain CF-1.6, and that target NCs may carry per-source ensemble member variables plus `ensemble_mean` / `ensemble_std`.
- `docs/architecture/nc-encoding-policy.md`: record that the target-layer dtype map is derived from `data_vars` (int8 for `n_sources` / `nn_filled`, float32 otherwise) rather than a fixed name list.
- `docs/architecture/transformation-pipeline.md`: add a short subsection stating that member emission is an output-only concern — members are always computed as the input to the bounds, and emitting them changes no computed value.

- [ ] **Step 9: Format, lint, commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/targets/_writers.py \
        src/nhf_spatial_targets/targets/_intermediates.py \
        tests/test_targets_common.py tests/test_cf_compliance.py \
        CLAUDE.md docs/architecture/nc-encoding-policy.md \
        docs/architecture/transformation-pipeline.md
pixi run git commit -m "feat(#338): emit ensemble members and mean/std from target writer"
```

---

### Task 4: `emit_members` config key

**Files:**
- Modify: `src/nhf_spatial_targets/defaults.py` (`DEFAULTS`, all six targets)
- Modify: `src/nhf_spatial_targets/init_run.py` (`_CONFIG_TEMPLATE`)
- Modify: `src/nhf_spatial_targets/upgrade_config.py` (`OPTIONAL_CONFIG_FEATURES`)
- Modify: `src/nhf_spatial_targets/targets/_driver.py` (both `write_bounds_target` call sites, lines ~236 and ~363)
- Test: `tests/test_defaults.py`, `tests/test_init_run.py`, `tests/test_upgrade_config.py`

**Interfaces:**
- Consumes: `write_bounds_target(..., emit_members=...)` from Task 3.
- Produces: config key `targets.<t>.emit_members`, read by the driver as `bool(target_cfg["emit_members"])`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_defaults.py`:

```python
def test_emit_members_defaults_true_for_bounds_targets():
    from nhf_spatial_targets.defaults import apply_defaults

    merged = apply_defaults({})
    for target in ("runoff", "aet", "recharge", "soil_moisture",
                   "snow_water_equivalent"):
        assert merged["targets"][target]["emit_members"] is True, target


def test_emit_members_defaults_false_for_snow_covered_area():
    from nhf_spatial_targets.defaults import apply_defaults

    merged = apply_defaults({})
    assert merged["targets"]["snow_covered_area"]["emit_members"] is False


def test_emit_members_is_a_known_key_for_the_unknown_key_linter():
    from nhf_spatial_targets.defaults import find_unknown_keys

    unknown = find_unknown_keys(
        {"targets": {"aet": {"emit_members": False}}}
    )
    assert unknown == []
```

Append to `tests/test_init_run.py`:

```python
def test_config_template_documents_emit_members(tmp_path):
    from nhf_spatial_targets.init_run import _CONFIG_TEMPLATE

    assert "emit_members" in _CONFIG_TEMPLATE
    # The comment must explain that this is an output switch, not a
    # science switch -- the operator-facing rationale, not just the key.
    assert "always computed" in _CONFIG_TEMPLATE
```

Append to `tests/test_upgrade_config.py`:

```python
def test_emit_members_is_tracked_as_an_optional_config_feature():
    from nhf_spatial_targets.upgrade_config import OPTIONAL_CONFIG_FEATURES

    names = {f.name for f in OPTIONAL_CONFIG_FEATURES}
    assert "targets.<target>.emit_members" in names
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pixi run -e dev test -k "emit_members" -v
```

Expected: FAIL — `KeyError: 'emit_members'` and the two membership assertions.

- [ ] **Step 3: Add the key to `DEFAULTS`**

In `src/nhf_spatial_targets/defaults.py`, add `"emit_members": True,` to each of `runoff`, `aet`, `recharge`, `soil_moisture`, `snow_water_equivalent`. For `snow_covered_area` add:

```python
            # SCA's bounds are a MOD10C1 CI interval, not a member min/max,
            # so emitted members would not reconstruct the bounds. Default
            # off; an operator can still enable it for diagnostics.
            "emit_members": False,
```

- [ ] **Step 4: Add the template stub**

In `src/nhf_spatial_targets/init_run.py:_CONFIG_TEMPLATE`, add this commented stub inside the `aet:` block (and reference it from the other target blocks with a one-line `# emit_members: true   # see aet: above`):

```
    # emit_members: true
    #   Whether to write the per-source ensemble members (one variable per
    #   source key) and the derived ensemble_mean / ensemble_std into the
    #   target NC. This is an OUTPUT switch, not a science switch: the
    #   members are always computed -- they are the input from which
    #   lower_bound / upper_bound / n_sources are derived -- so turning this
    #   off changes none of those values, it only stops them being written.
    #   Turn it off when file size matters: on the ~361k-HRU national fabric
    #   a daily SWE target grows from ~12 GB to ~36-48 GB with members. On a
    #   regional fabric the cost is negligible and true is the right choice.
    #   The output NC records the choice as the `members_emitted` global attr.
```

- [ ] **Step 5: Add the `OPTIONAL_CONFIG_FEATURES` entry**

Append to the list in `src/nhf_spatial_targets/upgrade_config.py`:

```python
    OptionalConfigFeature(
        name="targets.<target>.emit_members",
        detect=r"(?m)^\s*#?\s*emit_members\s*:",
        block=(
            "# Whether to write per-source ensemble members (one variable per\n"
            "# source key) plus ensemble_mean / ensemble_std into the target NC.\n"
            "# Output switch only: members are always computed, so turning this\n"
            "# off changes no bound value. Defaults true (false for\n"
            "# snow_covered_area). Turn off where file size matters -- daily SWE\n"
            "# on the national fabric grows from ~12 GB to ~36-48 GB.\n"
            "#\n"
            "#   emit_members: true\n"
        ),
        added="2026-09-08 (#338)",
        why=(
            "Retains the per-source ensemble alongside the bounds so a "
            "calibration consumer can see which source produced each bound."
        ),
    ),
```

- [ ] **Step 6: Wire the driver**

In `src/nhf_spatial_targets/targets/_driver.py`, add to **both** `write_bounds_target(...)` call sites (single-shot near line 236, per-year near line 363):

```python
        members=result.members,
        emit_members=bool(target_cfg["emit_members"]),
```

In the per-year call site the config dict is already in scope as `target_cfg`; confirm the local name before editing.

- [ ] **Step 7: Run the tests**

```bash
pixi run -e dev test -k "emit_members or defaults or init_run or upgrade_config" -v
```

Expected: PASS.

- [ ] **Step 8: Update CLAUDE.md**

Add `emit_members` to the config-schema discussion, noting it is per-target and defaults false for `snow_covered_area`.

- [ ] **Step 9: Format, lint, commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/defaults.py src/nhf_spatial_targets/init_run.py \
        src/nhf_spatial_targets/upgrade_config.py \
        src/nhf_spatial_targets/targets/_driver.py \
        tests/test_defaults.py tests/test_init_run.py tests/test_upgrade_config.py \
        CLAUDE.md
pixi run git commit -m "feat(#338): add per-target emit_members config key"
```

---

### Task 5: Year-chunked schema stability

**Files:**
- Test: `tests/test_targets_common.py`, `tests/test_targets_swe.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4. No new production code — this task proves the year-chunked path survives member emission.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_targets_common.py`. Note this extends the existing `_write_year_chunk_nc` helper defined around line 1244:

```python
def _write_year_chunk_with_members(
    path: Path,
    year: int,
    *,
    member_values: dict[str, float],
    hrus: list[int] | None = None,
) -> None:
    """Per-year intermediate carrying member variables as well as bounds.

    A NaN member value writes an all-NaN member, which is what
    `swe.py:_load_year` emits for a source with no data that year.
    """
    if hrus is None:
        hrus = [1, 2, 3]
    times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    shape = (len(times), len(hrus))
    data_vars = {
        "lower_bound": (("time", "nhm_id"), np.full(shape, 1.0, dtype=np.float32)),
        "upper_bound": (("time", "nhm_id"), np.full(shape, 2.0, dtype=np.float32)),
        "n_sources": (("time", "nhm_id"), np.full(shape, 2, dtype=np.int8)),
    }
    for key, value in member_values.items():
        data_vars[key] = (
            ("time", "nhm_id"),
            np.full(shape, value, dtype=np.float32),
        )
    ds = xr.Dataset(
        data_vars,
        coords={"time": times, "nhm_id": hrus},
        attrs={"Conventions": "CF-1.8", "title": f"chunk {year}", "year_chunk": year},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def test_stitch_preserves_member_variables_across_years(tmp_path: Path):
    """Members present in every year stitch into the canonical target."""
    from nhf_spatial_targets.targets._intermediates import stitch_year_chunks_to_target

    workdir = make_minimal_project(tmp_path)
    project = load(workdir)
    inter = tmp_path / "intermediates"
    # 2003: snodas has no data (all-NaN member). 2004: both sources present.
    _write_year_chunk_with_members(
        inter / "swe_targets_2003.nc",
        2003,
        member_values={"snodas": float("nan"), "era5_land": 1.5},
    )
    _write_year_chunk_with_members(
        inter / "swe_targets_2004.nc",
        2004,
        member_values={"snodas": 3.0, "era5_land": 1.5},
    )

    out = tmp_path / "swe_targets.nc"
    stitch_year_chunks_to_target(
        sorted(inter.glob("*.nc")),
        out,
        title="SWE",
        extra_global_attrs={"source_keys": "snodas,era5_land"},
        sort_dim="nhm_id",
        project=project,
    )

    ds = xr.open_dataset(out)
    try:
        assert "snodas" in ds.data_vars
        assert "era5_land" in ds.data_vars
        assert ds["snodas"].dtype == np.float32
        assert ds["era5_land"].dtype == np.float32
        # The absent-source year is honest NaN, not dropped or zero-filled.
        assert np.isnan(ds["snodas"].sel(time="2003-06-15").values).all()
        assert np.allclose(ds["snodas"].sel(time="2004-06-15").values, 3.0)
        # 2003 is a common year (365 days); 2004 is a leap year (366).
        assert len(ds.time) == 365 + 366
    finally:
        ds.close()
```

- [ ] **Step 2: Run the test to verify it fails or passes**

```bash
pixi run -e dev test -k stitch_preserves_member_variables -v
```

Expected: PASS if Task 3's dynamic dtype map is correct. If it FAILS with a dtype or encoding error, the dtype derivation in `_intermediates.py` is still name-based — fix it there, not in the test.

- [ ] **Step 3: Add the SWE loader constant-schema test**

Append to `tests/test_targets_swe.py`:

```python
def test_load_year_emits_all_nan_member_for_uncovered_source(tmp_path, monkeypatch):
    """A source with no data for a year still appears as an all-NaN member.

    The per-year NCs are stitched with join="exact", so the member set must
    not vary from year to year (#338).
    """
    import numpy as np
    import pandas as pd

    from nhf_spatial_targets.targets import swe as swe_mod
    from nhf_spatial_targets.targets._io import OutsideCoverageError

    def fake_read(project, source_key, var, period, chunks=None):
        if source_key == "snodas":
            raise OutsideCoverageError("no snodas for this year")
        times = pd.date_range("2003-01-01", "2003-12-31", freq="D")
        return xr.DataArray(
            np.full((len(times), 3), 100.0, dtype=np.float32),
            dims=("time", "nhm_id"),
            coords={"time": times, "nhm_id": [1, 2, 3]},
            attrs={"units": "mm"},
        )

    monkeypatch.setattr(swe_mod, "read_aggregated_source", fake_read)
    monkeypatch.setattr(swe_mod, "check_hru_coords", lambda *a, **k: None)
    monkeypatch.setattr(
        swe_mod, "_resolve_sources", lambda project: (["snodas", "era5_land"], [])
    )

    result = swe_mod._load_year(
        project=None,
        adapter=swe_mod.ADAPTER,
        period=("2003-01-01", "2003-12-31"),
        hru_meta=None,
        fabric_hru_ids=np.array([1, 2, 3]),
        id_col="nhm_id",
        year_context=(2003, "2003-01-01", "2003-12-31"),
    )

    assert set(result.members) == {"snodas", "era5_land"}
    assert np.isnan(result.members["snodas"].values).all()
    assert not np.isnan(result.members["era5_land"].values).any()
```

Adjust the monkeypatched names to match the actual import style in `swe.py` (the module imports `read_aggregated_source` and `check_hru_coords` by name, so patch them on `swe_mod`). If `_load_year` dereferences `project` before the source loop, pass a `make_minimal_project`-backed `Project` instead of `None`.

- [ ] **Step 4: Run the SWE test**

```bash
pixi run -e dev test -k "load_year_emits_all_nan_member" -v
```

Expected: PASS.

- [ ] **Step 5: Format, lint, commit, push, open PR**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add tests/test_targets_common.py tests/test_targets_swe.py
pixi run git commit -m "test(#338): year-chunked schema stability with ensemble members"
git push -u origin feature/338-ensemble-members
gh pr create --title "feat(#338): ensemble members and statistics in target NCs" \
  --body "Implements PR 1 of docs/superpowers/specs/2026-09-08-ensemble-target-schema-design.md.

Target NCs now retain per-source ensemble members as named variables and carry
ensemble_mean / ensemble_std, gated per-target by the new emit_members config
key (default true; false for snow_covered_area, whose bounds are a CI interval
rather than a member min/max). ensemble_std is NaN where n_sources < 2.

Conventions bumped to CF-1.8 on target NCs only; consolidated and aggregated
NCs remain CF-1.6.

Refs #338

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 6: Watch CI**

```bash
gh pr checks --watch
```

Do not run the full local suite. Fix any CI failure, push, and re-watch once.

---

# PR 2 — Per-source period-of-record normalization

Branch: `feature/338-per-source-por` (off `main`, after PR 1 merges)

---

### Task 6: `complete_years_window`

**Files:**
- Modify: `src/nhf_spatial_targets/normalize/methods.py`
- Test: `tests/test_normalize_methods.py`

**Interfaces:**
- Consumes: nothing from PR 1.
- Produces: `complete_years_window(da: xr.DataArray, cadence: str) -> tuple[str, str]` returning `(start, end)` ISO date strings, raising `ValueError` when no complete year exists.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_normalize_methods.py`:

```python
def _monthly_da(start: str, end: str) -> xr.DataArray:
    times = pd.date_range(start, end, freq="MS")
    return xr.DataArray(
        np.ones((len(times), 2), dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1, 2]},
    )


def _annual_da(start_year: int, end_year: int) -> xr.DataArray:
    times = pd.date_range(f"{start_year}-01-01", f"{end_year}-01-01", freq="YS")
    return xr.DataArray(
        np.ones((len(times), 2), dtype=np.float32),
        dims=("time", "nhm_id"),
        coords={"time": times, "nhm_id": [1, 2]},
    )


def test_complete_years_window_trims_partial_trailing_year_monthly():
    from nhf_spatial_targets.normalize.methods import complete_years_window

    # 1979-01 .. 2025-06: 2025 has only 6 months, so it is not a complete year.
    da = _monthly_da("1979-01-01", "2025-06-01")
    assert complete_years_window(da, "monthly") == ("1979-01-01", "2024-12-31")


def test_complete_years_window_trims_partial_leading_year_monthly():
    from nhf_spatial_targets.normalize.methods import complete_years_window

    da = _monthly_da("1979-07-01", "2020-12-01")
    assert complete_years_window(da, "monthly") == ("1980-01-01", "2020-12-31")


def test_complete_years_window_keeps_a_fully_covered_record():
    from nhf_spatial_targets.normalize.methods import complete_years_window

    da = _monthly_da("2000-01-01", "2013-12-01")
    assert complete_years_window(da, "monthly") == ("2000-01-01", "2013-12-31")


def test_complete_years_window_annual_cadence_needs_one_step_per_year():
    from nhf_spatial_targets.normalize.methods import complete_years_window

    da = _annual_da(2000, 2013)
    assert complete_years_window(da, "annual") == ("2000-01-01", "2013-12-31")


def test_complete_years_window_raises_when_no_complete_year():
    from nhf_spatial_targets.normalize.methods import complete_years_window

    da = _monthly_da("2000-03-01", "2000-09-01")
    with pytest.raises(ValueError, match="no complete calendar year"):
        complete_years_window(da, "monthly")
```

Add `import numpy as np`, `import pandas as pd`, `import pytest`, `import xarray as xr` to the test module if absent.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pixi run -e dev test -k complete_years_window -v
```

Expected: FAIL with `ImportError: cannot import name 'complete_years_window'`.

- [ ] **Step 3: Implement the helper**

Append to `src/nhf_spatial_targets/normalize/methods.py`:

```python
#: Timesteps a complete calendar year must contain, by cadence.
_STEPS_PER_YEAR = {"monthly": 12, "annual": 1}


def complete_years_window(da: xr.DataArray, cadence: str) -> tuple[str, str]:
    """Return ``(start, end)`` spanning only ``da``'s complete calendar years.

    This is the pipeline's definition of a source's **period of record**: a
    partial leading or trailing year is coverage, not record. A source whose
    monthly data runs 1979-01 .. 2025-06 has a POR of 1979-2024.

    The trim is load-bearing for annual-sum normalization. Summing a
    half-finished year yields a spuriously low annual total that becomes the
    per-HRU minimum, compressing every other year toward 1.0. It is applied
    uniformly to every cadence and reduction rather than only to sums, so
    that two sources' normalized series remain directly comparable.

    Parameters
    ----------
    da
        DataArray with a ``time`` dimension.
    cadence
        ``"monthly"`` or ``"annual"``. Determines how many timesteps a
        complete year must contain.

    Returns
    -------
    (start, end)
        ISO ``YYYY-MM-DD`` strings suitable for ``da.sel(time=slice(...))``.
        ``end`` is 31 December of the last complete year. The window is
        contiguous, so an incomplete year *interior* to the record is still
        spanned — the trim addresses ragged record ends, which is where
        every source in the catalog is actually ragged.

    Raises
    ------
    ValueError
        If ``cadence`` is unknown, ``da`` has no ``time`` dim, or no
        calendar year in ``da`` is complete.
    """
    if cadence not in _STEPS_PER_YEAR:
        raise ValueError(
            f"complete_years_window: unknown cadence {cadence!r}. "
            f"Expected one of {sorted(_STEPS_PER_YEAR)}."
        )
    if "time" not in da.dims:
        raise ValueError(
            f"complete_years_window: expected 'time' dim, got {tuple(da.dims)!r}."
        )
    required = _STEPS_PER_YEAR[cadence]
    years = pd.DatetimeIndex(da["time"].values).year
    counts = pd.Series(1, index=years).groupby(level=0).sum()
    complete = counts[counts >= required].index
    if len(complete) == 0:
        raise ValueError(
            "complete_years_window: no complete calendar year in the source "
            f"record (cadence={cadence!r} needs {required} timesteps per year; "
            f"observed per-year counts: {counts.to_dict()})."
        )
    return f"{int(complete.min())}-01-01", f"{int(complete.max())}-12-31"
```

Add `import pandas as pd` to the module imports if absent.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pixi run -e dev test -k complete_years_window -v
```

Expected: 5 passed.

- [ ] **Step 5: Format, lint, commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/normalize/methods.py tests/test_normalize_methods.py
pixi run git commit -m "feat(#338): add complete_years_window POR helper"
```

---

### Task 7: Recognize the `per_source_por` sentinel

**Files:**
- Modify: `src/nhf_spatial_targets/targets/_io.py:290` (`parse_period`)
- Modify: `src/nhf_spatial_targets/validate.py`
- Modify: `src/nhf_spatial_targets/init_run.py`, `src/nhf_spatial_targets/upgrade_config.py`
- Test: `tests/test_targets_common.py`, `tests/test_validate.py`

**Interfaces:**
- Consumes: nothing.
- Produces: module constant `PER_SOURCE_POR = "per_source_por"` in `targets/_io.py`; `parse_period` raises a sentinel-aware error rather than a generic one.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_targets_common.py`:

```python
def test_parse_period_rejects_the_sentinel_with_a_pointed_message():
    """parse_period must not be handed the sentinel -- callers branch first."""
    from nhf_spatial_targets.targets._io import PER_SOURCE_POR, parse_period

    with pytest.raises(ValueError, match="per_source_por is a sentinel"):
        parse_period(PER_SOURCE_POR)
```

Append to `tests/test_validate.py` (follow the module's existing project-fixture style):

```python
def test_validate_accepts_per_source_por_sentinel(tmp_path):
    """normalize_period: per_source_por must not be reported as malformed."""
    from nhf_spatial_targets.validate import _check_period_strings

    problems = _check_period_strings(
        {
            "targets": {
                "recharge": {
                    "enabled": True,
                    "period": "2000-01-01/2013-12-31",
                    "normalize_period": "per_source_por",
                }
            }
        }
    )
    assert problems == []
```

If `validate.py` has no `_check_period_strings`, locate the function that validates `period` / `normalize_period` strings and target that instead; the assertion is that the sentinel produces no problem entry.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pixi run -e dev test -k "per_source_por" -v
```

Expected: FAIL — `PER_SOURCE_POR` not importable.

- [ ] **Step 3: Add the sentinel and the guard**

In `src/nhf_spatial_targets/targets/_io.py`, above `parse_period`:

```python
#: Sentinel value for ``<target>.normalize_period`` meaning "normalize each
#: source over its own complete-year period of record" rather than over one
#: shared window. Handled by the recharge / soil-moisture builders before
#: they reach :func:`parse_period`.
PER_SOURCE_POR = "per_source_por"
```

Add to the top of `parse_period`'s body, before the `"/" not in period_str` check:

```python
    if period_str == PER_SOURCE_POR:
        raise ValueError(
            "parse_period: per_source_por is a sentinel, not a date range. "
            "The caller must branch on it and derive each source's window "
            "via normalize.methods.complete_years_window."
        )
```

- [ ] **Step 4: Teach `validate` the sentinel**

In `src/nhf_spatial_targets/validate.py`, wherever `normalize_period` is parsed or format-checked, short-circuit on the sentinel:

```python
from nhf_spatial_targets.targets._io import PER_SOURCE_POR

...
    if normalize_period == PER_SOURCE_POR:
        # Valid: each source derives its own window at build time.
        continue
```

- [ ] **Step 5: Document the sentinel in the config template and upgrade path**

In `init_run.py:_CONFIG_TEMPLATE`, under `recharge:` and `soil_moisture:`, add:

```
    # normalize_period accepts either an explicit "YYYY-MM-DD/YYYY-MM-DD"
    # window applied to every source, or the sentinel `per_source_por`,
    # which normalizes each source over its OWN complete-year period of
    # record. Partial leading/trailing years are trimmed first: an
    # unfinished year's annual sum would otherwise become that HRU's
    # minimum and compress every other year. The per-source windows are
    # recorded in the output NC as normalize_window_<source_key>.
    #   normalize_period: per_source_por
```

Add a matching `OptionalConfigFeature` entry named `targets.<target>.normalize_period: per_source_por` with `detect=r"(?m)^\s*#?\s*normalize_period\s*:"`, `added="2026-09-08 (#338)"`.

- [ ] **Step 6: Run the tests**

```bash
pixi run -e dev test -k "per_source_por or validate" -v
```

Expected: PASS.

- [ ] **Step 7: Format, lint, commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/targets/_io.py src/nhf_spatial_targets/validate.py \
        src/nhf_spatial_targets/init_run.py src/nhf_spatial_targets/upgrade_config.py \
        tests/test_targets_common.py tests/test_validate.py
pixi run git commit -m "feat(#338): recognize per_source_por normalize_period sentinel"
```

---

### Task 8: Recharge per-source POR normalization

**Files:**
- Modify: `src/nhf_spatial_targets/targets/rch.py` (loader, lines ~147-220)
- Test: `tests/test_targets_rch.py`

**Interfaces:**
- Consumes: `complete_years_window` (Task 6), `PER_SOURCE_POR` (Task 7).
- Produces: `extra_attrs["normalize_window_<source_key>"]` per source when the sentinel is active.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_targets_rch.py`, following the module's existing fixture style for synthesizing aggregated NCs:

```python
def test_recharge_per_source_por_uses_each_sources_own_complete_years(
    tmp_path, monkeypatch
):
    """Under the sentinel each source normalizes over its own trimmed record.

    reitz2017 covers 2000-2013 and era5_land 1998-2015, so the two sources
    must NOT share a window. The recorded per-source window attrs prove it.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    from nhf_spatial_targets.targets import rch as rch_mod

    def fake_read(project, source_key, var, period, chunks=None):
        spans = {
            "reitz2017": (2000, 2013),
            "era5_land": (1998, 2015),
        }
        y0, y1 = spans[source_key]
        if source_key == "era5_land":
            times = pd.date_range(f"{y0}-01-01", f"{y1}-12-01", freq="MS")
        else:
            times = pd.date_range(f"{y0}-01-01", f"{y1}-01-01", freq="YS")
        values = np.linspace(1.0, 2.0, len(times), dtype=np.float32)
        return xr.DataArray(
            np.repeat(values[:, None], 3, axis=1),
            dims=("time", "nhm_id"),
            coords={"time": times, "nhm_id": [1, 2, 3]},
            attrs={"units": "m"},
        )

    monkeypatch.setattr(rch_mod, "read_aggregated_source", fake_read)
    monkeypatch.setattr(rch_mod, "check_hru_coords", lambda *a, **k: None)

    result = rch_mod._load(
        project=_rch_project(tmp_path, normalize_period="per_source_por"),
        adapter=rch_mod.ADAPTER,
        period=("2000-01-01", "2013-12-31"),
        hru_meta=None,
        fabric_hru_ids=np.array([1, 2, 3]),
        id_col="nhm_id",
        year_context=None,
    )

    attrs = result.extra_attrs
    assert attrs["normalize_period"] == "per_source_por"
    assert attrs["normalize_window_reitz2017"] == "2000-01-01/2013-12-31"
    assert attrs["normalize_window_era5_land"] == "1998-01-01/2015-12-31"
```

Add a `_rch_project(tmp_path, normalize_period=...)` helper to the module if one does not already exist, building a project via `make_minimal_project` with a `targets.recharge` block whose `sources` are `["reitz2017", "era5_land"]`, `period` `"2000-01-01/2013-12-31"`, and the given `normalize_period`. Name the loader entry point to match `rch.py`'s actual loader function (check whether it is `_load` or another name before writing the call).

- [ ] **Step 2: Run the test to verify it fails**

```bash
pixi run -e dev test -k recharge_per_source_por -v
```

Expected: FAIL — the sentinel reaches `parse_period` and raises.

- [ ] **Step 3: Branch the recharge loader on the sentinel**

In `src/nhf_spatial_targets/targets/rch.py`, replace:

```python
    normalize_period = parse_period(rch_cfg["normalize_period"])
```

with:

```python
    raw_norm_period = rch_cfg["normalize_period"]
    per_source_por = raw_norm_period == PER_SOURCE_POR
    normalize_period = None if per_source_por else parse_period(raw_norm_period)
```

Update the log line to print `raw_norm_period` when the sentinel is active rather than dereferencing `normalize_period[0]` / `[1]`.

Inside the per-source loop, replace the read-window and window-slice logic:

```python
        read_start = min(period[0], normalize_period[0])
        read_end = max(period[1], normalize_period[1])
```

with:

```python
        if per_source_por:
            # Read the source's whole record so its own POR can be derived.
            read_start, read_end = "1900-01-01", "2200-12-31"
        else:
            read_start = min(period[0], normalize_period[0])
            read_end = max(period[1], normalize_period[1])
```

and replace the window computation:

```python
        window = da_annual_mm.sel(time=slice(normalize_period[0], normalize_period[1]))
```

with:

```python
        if per_source_por:
            win_start, win_end = complete_years_window(da_annual_mm, "annual")
            normalize_windows[src] = f"{win_start}/{win_end}"
        else:
            win_start, win_end = normalize_period
        window = da_annual_mm.sel(time=slice(win_start, win_end))
```

Declare `normalize_windows: dict[str, str] = {}` above the loop, and extend `extra_attrs`:

```python
    extra_attrs = {
        "source": "; ".join(shims[s].description for s in sources),
        "normalize_period": raw_norm_period,
    }
    for src, window_str in normalize_windows.items():
        extra_attrs[f"normalize_window_{src}"] = window_str
```

Add the imports:

```python
from nhf_spatial_targets.normalize.methods import (
    complete_years_window,
    normalize_0_1_over_window,
)
from nhf_spatial_targets.targets._io import PER_SOURCE_POR
```

Also update the existing empty-window `ValueError` message to name `raw_norm_period` rather than `rch_cfg["normalize_period"]`, so the sentinel case reads sensibly.

- [ ] **Step 4: Run the recharge tests**

```bash
pixi run -e dev test -k targets_rch -v
```

Expected: PASS, including the new test and every pre-existing explicit-window test.

- [ ] **Step 5: Format, lint, commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/targets/rch.py tests/test_targets_rch.py
pixi run git commit -m "feat(#338): per-source POR normalization for recharge"
```

---

### Task 9: Soil-moisture per-source POR normalization

**Files:**
- Modify: `src/nhf_spatial_targets/targets/som.py` (`_load_monthly` lines ~145-218, `_load_annual` lines ~220-278)
- Test: `tests/test_targets_som.py`

**Interfaces:**
- Consumes: `complete_years_window` (Task 6), `PER_SOURCE_POR` (Task 7).
- Produces: the same `normalize_window_<source_key>` attrs on both SOM variants.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_targets_som.py`, mirroring the module's existing fixture style:

```python
def test_som_monthly_per_source_por_records_per_source_windows(
    tmp_path, monkeypatch
):
    """Monthly SOM under the sentinel trims each source to complete years."""
    import numpy as np

    from nhf_spatial_targets.targets import som as som_mod

    result = som_mod._load_monthly(
        project=_som_project(tmp_path, normalize_period="per_source_por"),
        adapter=som_mod.ADAPTER_MONTHLY,
        period=("1980-01-01", "2020-12-31"),
        hru_meta=None,
        fabric_hru_ids=np.array([1, 2, 3]),
        id_col="nhm_id",
        year_context=None,
    )

    attrs = result.extra_attrs
    assert attrs["normalize_period"] == "per_source_por"
    assert attrs["normalize_method"] == "per_calendar_month"
    # merra2 fixture covers 1980-06 .. 2020-12, so 1980 is incomplete.
    assert attrs["normalize_window_merra2"] == "1981-01-01/2020-12-31"


def test_som_annual_per_source_por_records_per_source_windows(
    tmp_path, monkeypatch
):
    """Annual SOM under the sentinel trims on the annual-mean series."""
    import numpy as np

    from nhf_spatial_targets.targets import som as som_mod

    result = som_mod._load_annual(
        project=_som_project(tmp_path, normalize_period="per_source_por"),
        adapter=som_mod.ADAPTER_ANNUAL,
        period=("1980-01-01", "2020-12-31"),
        hru_meta=None,
        fabric_hru_ids=np.array([1, 2, 3]),
        id_col="nhm_id",
        year_context=None,
    )

    assert result.extra_attrs["normalize_period"] == "per_source_por"
    assert "normalize_window_merra2" in result.extra_attrs
```

Write `_som_project(tmp_path, normalize_period=...)` and the monkeypatched `read_aggregated_source` in the same style as Task 8's recharge helper: `sources` of `["merra2", "nldas_mosaic"]`, a `merra2` fixture spanning 1980-06 .. 2020-12 monthly and an `nldas_mosaic` fixture spanning 1979-01 .. 2020-12 monthly. Reuse the module's existing helper if one already exists.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pixi run -e dev test -k som_monthly_per_source_por -v
```

Expected: FAIL — the sentinel reaches `parse_period`.

- [ ] **Step 3: Branch `_load_monthly` on the sentinel**

In `som.py:_load_monthly`, replace:

```python
    raw_norm_period = som_cfg.get("normalize_period") or som_cfg["period"]
    normalize_period = parse_period(raw_norm_period)
```

with:

```python
    raw_norm_period = som_cfg.get("normalize_period") or som_cfg["period"]
    per_source_por = raw_norm_period == PER_SOURCE_POR
    normalize_period = None if per_source_por else parse_period(raw_norm_period)
```

Replace the normalization loop:

```python
    sources_monthly_norm: dict[str, xr.DataArray] = {}
    for src, da in sources_monthly.items():
        window = da.sel(time=slice(normalize_period[0], normalize_period[1]))
```

with:

```python
    sources_monthly_norm: dict[str, xr.DataArray] = {}
    normalize_windows: dict[str, str] = {}
    for src, da in sources_monthly.items():
        if per_source_por:
            win_start, win_end = complete_years_window(da, "monthly")
            normalize_windows[src] = f"{win_start}/{win_end}"
        else:
            win_start, win_end = normalize_period
        window = da.sel(time=slice(win_start, win_end))
```

Extend `extra_attrs` after it is built:

```python
    for src, window_str in normalize_windows.items():
        extra_attrs[f"normalize_window_{src}"] = window_str
```

Update the log line so it does not index `normalize_period` when the sentinel is active.

- [ ] **Step 4: Apply the same branch to `_load_annual`**

Same structure as `_load_monthly`, with one important difference: **derive the window from the MONTHLY series, not the annual one.**

`resample(time="YS").mean()` emits one step for a partial year just as it does for a complete one, so an incomplete trailing year survives as a mean over fewer months. Running `complete_years_window` on the annual series would therefore see one step per year and keep the partial year. The monthly series is where incompleteness is still visible.

Replace:

```python
    sources_annual_norm: dict[str, xr.DataArray] = {}
    for src, da in sources_annual.items():
        window = da.sel(time=slice(normalize_period[0], normalize_period[1]))
```

with:

```python
    sources_annual_norm: dict[str, xr.DataArray] = {}
    normalize_windows: dict[str, str] = {}
    for src, da in sources_annual.items():
        if per_source_por:
            # Derive from the pre-resample monthly series: an incomplete
            # year still has one annual step, so the annual series cannot
            # reveal that the year is partial.
            win_start, win_end = complete_years_window(
                sources_monthly[src], "monthly"
            )
            normalize_windows[src] = f"{win_start}/{win_end}"
        else:
            win_start, win_end = normalize_period
        window = da.sel(time=slice(win_start, win_end))
```

and extend `extra_attrs` the same way as in `_load_monthly`.

- [ ] **Step 5: Add the imports**

```python
from nhf_spatial_targets.normalize.methods import (
    complete_years_window,
    normalize_0_1_by_calendar_month_over_window,
    normalize_0_1_over_window,
)
from nhf_spatial_targets.targets._io import PER_SOURCE_POR
```

- [ ] **Step 6: Run the SOM tests**

```bash
pixi run -e dev test -k targets_som -v
```

Expected: PASS.

- [ ] **Step 7: Update the docs**

- `docs/architecture/transformation-pipeline.md`: in the per-HRU transforms section, document `per_source_por` and state that the trim narrows normalization windows only and never the emitted time axis.
- `docs/references/calibration-target-recipes.md`: note the sentinel for the recharge and soil-moisture recipes.
- `CLAUDE.md`: mention the sentinel alongside `normalize_period`.

- [ ] **Step 8: Format, lint, commit, push, open PR**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_spatial_targets/targets/som.py tests/test_targets_som.py \
        docs/architecture/transformation-pipeline.md \
        docs/references/calibration-target-recipes.md CLAUDE.md
pixi run git commit -m "feat(#338): per-source POR normalization for soil moisture"
git push -u origin feature/338-per-source-por
gh pr create --title "feat(#338): per-source period-of-record normalization" \
  --body "Implements PR 2 of docs/superpowers/specs/2026-09-08-ensemble-target-schema-design.md.

Adds the normalize_period: per_source_por sentinel. Each source is normalized
over its own complete-calendar-year period of record, with partial leading and
trailing years trimmed by normalize.methods.complete_years_window. The trim
applies uniformly to every cadence, not only the annual sums where it is
load-bearing, so two sources' normalized series stay comparable.

The trim narrows normalization windows only; the emitted time axis is still
driven by the target's configured period. Per-source windows are recorded as
normalize_window_<source_key> global attrs.

Refs #338

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
gh pr checks --watch
```

---

# PR 3 — Oregon project configuration

Branch: `chore/338-or-config` (off `main`, after PRs 1 and 2 merge)

---

### Task 10: Oregon config and SLURM memory

**Files:**
- Modify: `/caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets/config.yml` (outside the repo — an operator artifact, not version-controlled here)
- Modify: `slurm/project_or/run_or.slurm`, `slurm/project_gfv2/run_gfv2.slurm`

**Interfaces:**
- Consumes: `emit_members` (Task 4), `per_source_por` (Tasks 7-9).
- Produces: no code interfaces.

- [ ] **Step 1: Edit the Oregon config**

In `/caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets/config.yml`:

- `targets.recharge.sources`: comment out `watergap22d`, keeping the existing rationale comment.
- `targets.recharge.normalize_period`: `per_source_por`.
- `targets.soil_moisture.sources`: comment out `ncep_ncar`, keeping the rationale comment.
- `targets.soil_moisture.normalize_period`: `per_source_por`.
- `targets.snow_water_equivalent.sources`: comment out `daymet`, leaving `snodas`, `era5_land`, `margulis_wus_sr`, `ua_swe`. Add a comment: `# daymet dropped per colleagues' 06_Create_calibration_target_ensembles.ipynb (#338).`
- `targets.snow_covered_area.enabled`: `false`, with a comment that SCA is retained in the codebase and disabled for this fabric only.
- `nn_fill: false` on `runoff`, `aet`, `recharge`, `soil_moisture` (the snow targets are already false).
- Add `emit_members: true` explicitly to every enabled target, so the choice is visible in `config.effective.yml` rather than inherited.

- [ ] **Step 2: Re-validate the project**

```bash
pixi run validate -- --project-dir /caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets
```

Expected: no `unknown config key` warnings for `emit_members`, no malformed-period error for `per_source_por`, and a regenerated `config.effective.yml`. The publish staleness gate is fatal, so this step is mandatory after any config edit.

- [ ] **Step 3: Check config drift reporting**

```bash
pixi run check-config -- --project-dir /caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets
```

Expected: `emit_members` and the `per_source_por` sentinel both reported as in-sync (the entries added in Tasks 4 and 7).

- [ ] **Step 4: Bump SLURM memory**

In `slurm/project_or/run_or.slurm` and `slurm/project_gfv2/run_gfv2.slurm`, change `#SBATCH --mem=32G` to `#SBATCH --mem=64G` and add a header comment:

```bash
# --mem=64G: write_bounds_target calls ds.compute() (full materialization, no
# streaming) and nn_fill_bounds then makes a second full copy. With ensemble
# members emitted (#338) the gfv2 soil-moisture monthly target peaks near 11 GB;
# 64G leaves headroom without the in-code streaming refactor that --mem avoids.
```

- [ ] **Step 5: Rebuild the Oregon targets**

```bash
sbatch --account=impd slurm/project_or/run_or.slurm
```

`--account=impd` is mandatory — the `default` account has `MaxSubmit=0` and rejects every submission.

- [ ] **Step 6: Verify the rebuilt targets**

```bash
pixi run python -c "
import xarray as xr
for name in ('aet_targets', 'recharge_targets', 'swe_targets'):
    ds = xr.open_dataset(
        f'/caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets/targets/{name}.nc'
    )
    print(name, sorted(ds.data_vars))
    print('  source_keys:', ds.attrs.get('source_keys'))
    print('  members_emitted:', ds.attrs.get('members_emitted'))
    print('  Conventions:', ds.attrs.get('Conventions'))
    ds.close()
"
```

Expected: each file lists its member variables plus `ensemble_mean` / `ensemble_std`; `swe_targets` lists exactly `snodas`, `era5_land`, `margulis_wus_sr`, `ua_swe` (no `daymet`); `members_emitted` is `"true"`; `Conventions` is `CF-1.8`.

- [ ] **Step 7: Rebuild the manifest**

```bash
pixi run rebuild-manifest -- --project-dir /caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets
```

- [ ] **Step 8: Commit the SLURM changes and open the PR**

The project `config.yml` lives outside the repo and is not committed, so it will
not appear in this PR's diff. Paste the config diff into the PR body so the
change is reviewable. Versioning project intent artifacts properly is tracked
as [#343](https://github.com/rmcd-mscb/nhf-spatial-targets/issues/343); if that
has landed by the time this task runs, cite the project-repo commit SHA instead
of pasting the diff.

Only the SLURM scripts are version-controlled here.

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add slurm/project_or/run_or.slurm slurm/project_gfv2/run_gfv2.slurm
pixi run git commit -m "chore(#338): bump target-build SLURM memory for member emission"
git push -u origin chore/338-or-config
gh pr create --title "chore(#338): SLURM memory headroom for ensemble member emission" \
  --body "Implements PR 3 of docs/superpowers/specs/2026-09-08-ensemble-target-schema-design.md.

Bumps run_or.slurm and run_gfv2.slurm from 32G to 64G. write_bounds_target
materializes the full dataset and nn_fill_bounds copies it again; with members
emitted the gfv2 soil-moisture monthly target peaks near 11 GB.

The Oregon project config.yml lives outside the repo and is updated separately:
watergap22d and ncep_ncar dropped, SWE to snodas/era5_land/margulis_wus_sr/ua_swe
(daymet dropped), SCA disabled for this fabric, nn_fill off, per_source_por
normalization on recharge and soil moisture.

Refs #338

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
gh pr checks --watch
```

---

## Self-Review Notes

**Spec coverage:** §3.1 → Task 2. §3.2 → Tasks 1, 3. §3.3 → Tasks 2 (step 7), 5. §3.4 → Task 3. §3.5 → Task 3 (steps 4-5). §3.6 → no change needed; `write_bounds_target` builds the NN-filled dataset from `ds_loaded` via `nn_fill_bounds`, which operates on `lower_bound` / `upper_bound` only, so members are naturally absent from the companion. **Task 3 must not add members to `filled_ds`.** §4 → Tasks 6-9. §4.1 → Task 6. §5 → Tasks 4, 7. §6 → Task 10. §7 → no code (divergences are things we do *not* change; #341 tracks the notebook side). §8 → Task 10 SLURM bumps. §9 → tests distributed across all tasks. §10 → docs folded into Tasks 3, 4, 9. §11 → the three PR groupings.

**Known gap deliberately left to the implementer:** Tasks 8 and 9 reference `rch._load` / `som._load_monthly` / `som._load_annual` and the test modules' existing project fixtures. The exact loader entry-point name in `rch.py` and the fixture helpers in `tests/test_targets_rch.py` / `tests/test_targets_som.py` must be read before writing those tests; the plan says so at each site rather than guessing a name that may not exist.
