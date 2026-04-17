# Per-source aggregate layout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the `concat_years` OOM in `nhf-targets agg` by making per-year NetCDFs the canonical aggregated output, moved to `data/aggregated/<source_key>/`.

**Architecture:** Delete `concat_years()`. Move CF attrs + `post_aggregate_hook` into `aggregate_year` so each per-year file is independently usable. Add an idempotent legacy-layout migration shim and a filename-level year-coverage check. Manifest records `output_files: list[str]`.

**Tech Stack:** Python ≥3.11, xarray, pandas, numpy, geopandas, gdptools, pytest, ruff. Managed via pixi.

**Spec:** `docs/superpowers/specs/2026-04-16-per-source-aggregate-layout-design.md`

**Branch (already created):** `fix/53-per-source-aggregate-layout`

---

## File Structure

- **Modify** `src/nhf_spatial_targets/aggregate/_driver.py` — core changes: delete `concat_years`, retarget `per_year_output_path`, move CF attrs and `post_aggregate_hook` into `aggregate_year`, add `_migrate_legacy_layout` and `_verify_year_coverage`, add `_derive_period`, rework `update_manifest` and `aggregate_source`.
- **Modify** `src/nhf_spatial_targets/aggregate/mod10c1.py` — update `_rename_and_warn` to name the year in the warning log message (hook already runs per-year after the driver changes; no code move needed here because the hook input remains a single `xr.Dataset`).
- **Modify** `tests/test_aggregate_driver.py` — update existing tests that assert old path/manifest shape; keep unchanged tests as-is.
- **Modify** `tests/test_aggregate_driver_per_year.py` — delete `concat_years` tests; update `per_year_output_path` test; add migration + coverage tests.
- **Modify** `tests/test_aggregate_mod10c1.py` — update `_log_low_valid_coverage` tests to take a year argument.

No new files are introduced. All new functionality is private helpers inside `_driver.py`.

---

## Pre-flight

- [ ] **Step 0.1: Confirm branch is clean and up-to-date**

Run:
```bash
git status
git log --oneline -3
```

Expected: on branch `fix/53-per-source-aggregate-layout` with the two spec commits at the top (`docs: spec for per-source aggregate layout` and `docs: spec self-review`). Working tree clean.

- [ ] **Step 0.2: Confirm baseline tests pass**

Run:
```bash
pixi run -e dev test
```

Expected: all tests pass. If any fail on the untouched tree, stop and diagnose before continuing.

---

### Task 1: Retarget `per_year_output_path` to the new layout

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py:168-176`
- Modify: `tests/test_aggregate_driver_per_year.py:102-106`

- [ ] **Step 1.1: Update the existing test to assert the new path shape**

Replace `test_per_year_output_path` in `tests/test_aggregate_driver_per_year.py`:

```python
def test_per_year_output_path(project):
    from nhf_spatial_targets.aggregate._driver import per_year_output_path

    p = per_year_output_path(project, "foo", 2005)
    assert p == (
        project.workdir / "data" / "aggregated" / "foo" / "foo_2005_agg.nc"
    )
```

- [ ] **Step 1.2: Run test to verify it fails**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py::test_per_year_output_path -v
```

Expected: FAIL — the current path is `_by_year/foo_2005_agg.nc`.

- [ ] **Step 1.3: Update `per_year_output_path` in `_driver.py`**

Replace lines 168-176 with:

```python
def per_year_output_path(project: Project, source_key: str, year: int) -> Path:
    """Return the per-year aggregated NC path (canonical output)."""
    return (
        project.workdir
        / "data"
        / "aggregated"
        / source_key
        / f"{source_key}_{year}_agg.nc"
    )
```

Update the docstring on `aggregate_year` (lines 200-208) to replace "intermediate" language: reflect that the per-year NC is the canonical output.

- [ ] **Step 1.4: Run the single test to verify it passes**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py::test_per_year_output_path -v
```

Expected: PASS.

- [ ] **Step 1.5: Run the full `_driver_per_year` suite**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py -v
```

Expected: several other tests still fail (they reference the old path via `aggregate_year` writing to `_by_year/`). That's fine — later tasks will update them. All already-passing tests that don't touch the path must remain passing.

- [ ] **Step 1.6: Commit**

```bash
git add src/nhf_spatial_targets/aggregate/_driver.py tests/test_aggregate_driver_per_year.py
git commit -m "$(cat <<'EOF'
refactor: retarget per_year_output_path to data/aggregated/<source_key>/

First step of #53. Subsequent tasks move CF attrs and post_aggregate_hook
into aggregate_year, delete concat_years, and add migration + coverage
checks.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Move `_attach_cf_global_attrs` and `post_aggregate_hook` into `aggregate_year`

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py:192-289` (`aggregate_year`)

- [ ] **Step 2.1: Add a failing test that asserts the per-year file carries CF global attrs**

Append to `tests/test_aggregate_driver_per_year.py`:

```python
def test_aggregate_year_attaches_cf_global_attrs(project, tiny_batched_fabric):
    """Each per-year file must carry Conventions/history/source independent
    of any consolidation step."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import (
        aggregate_year,
        per_year_output_path,
    )

    src_dir = project.raw_dir("merra2")
    src_dir.mkdir(parents=True, exist_ok=True)
    src_file = src_dir / "src_2005_consolidated.nc"
    _write_nc(src_file, pd.date_range("2005-01-01", periods=12, freq="MS"))

    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=["v"]
    )
    fake_weights = pd.DataFrame({"i": [0], "j": [0], "wght": [1.0], "hru_id": [0]})
    fake_year_ds = xr.Dataset(
        {"v": (["time", "hru_id"], np.ones((1, 2)))},
        coords={
            "time": (
                "time",
                pd.date_range("2005-01-01", periods=1, freq="MS"),
                {"standard_name": "time"},
            ),
            "hru_id": [0, 1],
        },
    )
    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.catalog_source",
            return_value={"access": {"doi": "10.0/TEST"}},
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=fake_weights,
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            return_value=fake_year_ds,
        ),
    ):
        aggregate_year(
            adapter, project, 2005, src_file, tiny_batched_fabric, "hru_id"
        )

    with xr.open_dataset(per_year_output_path(project, "merra2", 2005)) as written:
        assert written.attrs["Conventions"] == "CF-1.6"
        assert written.attrs["source"] == "merra2"
        assert "aggregated to HRU fabric" in written.attrs["history"]
        assert written.attrs["source_doi"] == "10.0/TEST"
```

- [ ] **Step 2.2: Run the test to verify it fails**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py::test_aggregate_year_attaches_cf_global_attrs -v
```

Expected: FAIL — `aggregate_year` does not currently attach CF attrs (only `aggregate_source` does, on the combined dataset).

- [ ] **Step 2.3: Add a failing test that asserts `post_aggregate_hook` runs per-year**

Append to `tests/test_aggregate_driver_per_year.py`:

```python
def test_aggregate_year_runs_post_aggregate_hook(project, tiny_batched_fabric):
    """post_aggregate_hook must run inside aggregate_year on the per-year
    dataset (before the atomic write)."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import (
        aggregate_year,
        per_year_output_path,
    )

    src_dir = project.raw_dir("merra2")
    src_dir.mkdir(parents=True, exist_ok=True)
    src_file = src_dir / "src_2005_consolidated.nc"
    _write_nc(src_file, pd.date_range("2005-01-01", periods=12, freq="MS"))

    calls: list[int] = []

    def post_hook(ds):
        calls.append(1)
        return ds.rename({"v": "v_renamed"})

    adapter = SourceAdapter(
        source_key="merra2",
        output_name="merra2_agg.nc",
        variables=["v"],
        post_aggregate_hook=post_hook,
    )
    fake_weights = pd.DataFrame({"i": [0], "j": [0], "wght": [1.0], "hru_id": [0]})
    fake_year_ds = xr.Dataset(
        {"v": (["time", "hru_id"], np.ones((1, 2)))},
        coords={
            "time": (
                "time",
                pd.date_range("2005-01-01", periods=1, freq="MS"),
                {"standard_name": "time"},
            ),
            "hru_id": [0, 1],
        },
    )
    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.catalog_source",
            return_value={"access": {"type": "local_nc"}},
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=fake_weights,
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            return_value=fake_year_ds,
        ),
    ):
        aggregate_year(
            adapter, project, 2005, src_file, tiny_batched_fabric, "hru_id"
        )

    assert calls, "post_aggregate_hook was not invoked"
    with xr.open_dataset(per_year_output_path(project, "merra2", 2005)) as written:
        assert "v_renamed" in written.data_vars
        assert "v" not in written.data_vars
```

- [ ] **Step 2.4: Run the test to verify it fails**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py::test_aggregate_year_runs_post_aggregate_hook -v
```

Expected: FAIL — `post_aggregate_hook` is currently only called in `aggregate_source` on the combined dataset.

- [ ] **Step 2.5: Modify `aggregate_year` to attach CF attrs and run the hook**

In `src/nhf_spatial_targets/aggregate/_driver.py`, change `aggregate_year` to accept the catalog meta and apply both the hook and CF attrs before the atomic write. Full new body:

```python
def aggregate_year(
    adapter: SourceAdapter,
    project: Project,
    year: int,
    source_file: Path,
    fabric_batched: gpd.GeoDataFrame,
    id_col: str,
    *,
    catalog_meta: dict | None = None,
) -> Path:
    """Aggregate one year to HRU polygons; idempotent on the per-year NC.

    Returns the path of the per-year aggregated NC. If that path already
    exists, returns immediately without opening the source file. Otherwise
    opens the source file lazily, applies ``adapter.pre_aggregate_hook``
    if set, detects coords via CF attrs (respecting adapter overrides),
    runs the batch loop with ``period=(YYYY-01-01, YYYY-12-31)``,
    concatenates batches on ``id_col``, applies
    ``adapter.post_aggregate_hook`` if set, attaches CF-1.6 global attrs
    (using ``catalog_meta`` when provided; otherwise looked up from the
    catalog), and writes the per-year NC atomically.
    """
    out_path = per_year_output_path(project, adapter.source_key, year)
    if out_path.exists():
        logger.info(
            "%s: year %d: per-year NC exists, skipping (%s)",
            adapter.source_key,
            year,
            out_path,
        )
        return out_path

    logger.info(
        "%s: year %d: aggregating from %s",
        adapter.source_key,
        year,
        source_file.name,
    )
    period = (f"{year}-01-01", f"{year}-12-31")

    with xr.open_dataset(source_file) as raw:
        ds = raw
        if adapter.pre_aggregate_hook is not None:
            ds = adapter.pre_aggregate_hook(ds)

        grid_var = adapter.grid_variable or adapter.variables[0]
        x_coord, y_coord, time_coord = detect_coords(
            ds,
            grid_var,
            x_override=adapter.x_coord,
            y_override=adapter.y_coord,
            time_override=adapter.time_coord,
        )

        datasets: list[xr.Dataset] = []
        for bid in sorted(fabric_batched["batch_id"].unique()):
            batch_gdf = fabric_batched[fabric_batched["batch_id"] == bid].drop(
                columns=["batch_id"]
            )
            try:
                weights = compute_or_load_weights(
                    batch_gdf=batch_gdf,
                    source_ds=ds,
                    source_var=grid_var,
                    source_crs=adapter.source_crs,
                    x_coord=x_coord,
                    y_coord=y_coord,
                    time_coord=time_coord,
                    id_col=id_col,
                    source_key=adapter.source_key,
                    batch_id=int(bid),
                    workdir=project.workdir,
                    period=period,
                )
                batch_ds = aggregate_variables_for_batch(
                    batch_gdf=batch_gdf,
                    source_ds=ds,
                    variables=list(adapter.variables),
                    source_crs=adapter.source_crs,
                    x_coord=x_coord,
                    y_coord=y_coord,
                    time_coord=time_coord,
                    id_col=id_col,
                    weights=weights,
                    period=period,
                )
            except Exception as exc:
                exc.add_note(
                    f"{adapter.source_key}: aggregation failed for "
                    f"year={year} batch={int(bid)} "
                    f"source_file={source_file.name}"
                )
                raise
            datasets.append(batch_ds)

        year_ds = xr.concat(datasets, dim=id_col)

    if adapter.post_aggregate_hook is not None:
        year_ds = adapter.post_aggregate_hook(year_ds)

    meta = catalog_meta if catalog_meta is not None else catalog_source(adapter.source_key)
    _attach_cf_global_attrs(year_ds, adapter.source_key, meta)

    _atomic_write_netcdf(year_ds, out_path)
    logger.info("%s: year %d: wrote %s", adapter.source_key, year, out_path)
    return out_path
```

- [ ] **Step 2.6: Run the two new tests to verify they pass**

Run:
```bash
pixi run -e dev test -- \
  tests/test_aggregate_driver_per_year.py::test_aggregate_year_attaches_cf_global_attrs \
  tests/test_aggregate_driver_per_year.py::test_aggregate_year_runs_post_aggregate_hook -v
```

Expected: both PASS.

- [ ] **Step 2.7: Run the full driver_per_year and mod10c1 suites**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py tests/test_aggregate_mod10c1.py -v
```

Expected: at least the new tests pass. `concat_years` tests and any tests asserting the old `_by_year/` path still fail — that is expected and resolved in Task 3 / Task 5.

- [ ] **Step 2.8: Commit**

```bash
git add src/nhf_spatial_targets/aggregate/_driver.py tests/test_aggregate_driver_per_year.py
git commit -m "$(cat <<'EOF'
refactor: move CF attrs + post_aggregate_hook into aggregate_year

Each per-year NC is now independently CF-1.6 compliant. The post hook
runs once per year before the atomic write. Prepares for deletion of
concat_years in a follow-up task.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add `_migrate_legacy_layout` helper

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py` (add new helper after `_atomic_write_netcdf`)
- Modify: `tests/test_aggregate_driver_per_year.py` (add migration tests)

- [ ] **Step 3.1: Add a failing test: legacy `_by_year/` files are moved into the new subdir**

Append to `tests/test_aggregate_driver_per_year.py`:

```python
def test_migrate_legacy_layout_moves_by_year_files(project):
    from nhf_spatial_targets.aggregate._driver import _migrate_legacy_layout

    legacy_dir = project.workdir / "data" / "aggregated" / "_by_year"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    f2000 = legacy_dir / "mod10c1_v061_2000_agg.nc"
    f2001 = legacy_dir / "mod10c1_v061_2001_agg.nc"
    _write_year_intermediate(f2000, 2000)
    _write_year_intermediate(f2001, 2001)

    _migrate_legacy_layout(project, "mod10c1_v061")

    new_dir = project.workdir / "data" / "aggregated" / "mod10c1_v061"
    assert (new_dir / "mod10c1_v061_2000_agg.nc").exists()
    assert (new_dir / "mod10c1_v061_2001_agg.nc").exists()
    assert not f2000.exists()
    assert not f2001.exists()
```

- [ ] **Step 3.2: Add a failing test: stale consolidated file is unlinked**

```python
def test_migrate_legacy_layout_removes_stale_consolidated(project):
    from nhf_spatial_targets.aggregate._driver import _migrate_legacy_layout

    agg_dir = project.workdir / "data" / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    stale = agg_dir / "mod10c1_v061_agg.nc"
    stale.write_bytes(b"placeholder")

    _migrate_legacy_layout(project, "mod10c1_v061")

    assert not stale.exists()
```

- [ ] **Step 3.3: Add a failing test: idempotent (re-run is a no-op)**

```python
def test_migrate_legacy_layout_idempotent(project):
    from nhf_spatial_targets.aggregate._driver import _migrate_legacy_layout

    legacy_dir = project.workdir / "data" / "aggregated" / "_by_year"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    _write_year_intermediate(legacy_dir / "foo_2000_agg.nc", 2000)

    _migrate_legacy_layout(project, "foo")
    _migrate_legacy_layout(project, "foo")  # must not raise

    new_file = (
        project.workdir / "data" / "aggregated" / "foo" / "foo_2000_agg.nc"
    )
    assert new_file.exists()
```

- [ ] **Step 3.4: Add a failing test: collision is a no-op (both paths preserved)**

```python
def test_migrate_legacy_layout_collision_leaves_both(project):
    from nhf_spatial_targets.aggregate._driver import _migrate_legacy_layout

    legacy_dir = project.workdir / "data" / "aggregated" / "_by_year"
    new_dir = project.workdir / "data" / "aggregated" / "foo"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)
    legacy_file = legacy_dir / "foo_2000_agg.nc"
    new_file = new_dir / "foo_2000_agg.nc"
    _write_year_intermediate(legacy_file, 2000)
    _write_year_intermediate(new_file, 2000)

    _migrate_legacy_layout(project, "foo")

    # New-path file is canonical and untouched; legacy file is left in place.
    assert legacy_file.exists()
    assert new_file.exists()
```

- [ ] **Step 3.5: Run the four new tests to verify they fail**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py -k migrate_legacy_layout -v
```

Expected: all four FAIL with `ImportError` / `AttributeError` — helper does not exist yet.

- [ ] **Step 3.6: Implement `_migrate_legacy_layout` in `_driver.py`**

Insert after `_atomic_write_netcdf` (just before `aggregate_year`):

```python
def _migrate_legacy_layout(project: Project, source_key: str) -> None:
    """Migrate legacy aggregated layout into per-source subdirs.

    Idempotent. Moves ``data/aggregated/_by_year/<source_key>_*.nc`` into
    ``data/aggregated/<source_key>/`` and unlinks any stale
    ``data/aggregated/<source_key>_agg.nc``. If a target path already
    exists for a given year (collision), the legacy file is left in
    place — the new-path file is canonical.
    """
    agg_dir = project.aggregated_dir()
    legacy_dir = agg_dir / "_by_year"
    new_dir = agg_dir / source_key

    if legacy_dir.is_dir():
        new_dir.mkdir(parents=True, exist_ok=True)
        for legacy_file in sorted(legacy_dir.glob(f"{source_key}_*_agg.nc")):
            target = new_dir / legacy_file.name
            if target.exists():
                logger.info(
                    "%s: legacy %s collides with existing %s; "
                    "leaving both in place (new path is canonical)",
                    source_key,
                    legacy_file,
                    target,
                )
                continue
            legacy_file.rename(target)
            logger.info("%s: migrated %s -> %s", source_key, legacy_file, target)

    stale_consolidated = agg_dir / f"{source_key}_agg.nc"
    if stale_consolidated.is_file():
        stale_consolidated.unlink()
        logger.info(
            "%s: removed stale consolidated file %s",
            source_key,
            stale_consolidated,
        )
```

- [ ] **Step 3.7: Run the four new tests to verify they pass**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py -k migrate_legacy_layout -v
```

Expected: all four PASS.

- [ ] **Step 3.8: Commit**

```bash
git add src/nhf_spatial_targets/aggregate/_driver.py tests/test_aggregate_driver_per_year.py
git commit -m "$(cat <<'EOF'
feat: add _migrate_legacy_layout helper (idempotent)

Moves data/aggregated/_by_year/<key>_*.nc into data/aggregated/<key>/
and unlinks stale <key>_agg.nc. Collisions are a no-op; the new-path
file is canonical. Prepares the per-source directory convention
introduced by #53.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Add `_verify_year_coverage` helper (filename-level)

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py`
- Modify: `tests/test_aggregate_driver_per_year.py`

- [ ] **Step 4.1: Add failing test: contiguous years pass**

Append to `tests/test_aggregate_driver_per_year.py`:

```python
def test_verify_year_coverage_ok_on_contiguous(tmp_path):
    from nhf_spatial_targets.aggregate._driver import _verify_year_coverage

    d = tmp_path / "foo"
    d.mkdir()
    for y in (2000, 2001, 2002):
        (d / f"foo_{y}_agg.nc").write_bytes(b"")

    # Must not raise.
    _verify_year_coverage(d, "foo")
```

- [ ] **Step 4.2: Add failing test: interior gap raises with missing list**

```python
def test_verify_year_coverage_raises_on_interior_gap(tmp_path):
    from nhf_spatial_targets.aggregate._driver import _verify_year_coverage

    d = tmp_path / "foo"
    d.mkdir()
    for y in (2000, 2002, 2003):
        (d / f"foo_{y}_agg.nc").write_bytes(b"")

    with pytest.raises(ValueError, match=r"missing=\[2001\]"):
        _verify_year_coverage(d, "foo")
```

> Note on duplicates: filesystem uniqueness prevents two files named
> `foo_2000_agg.nc` in one directory, so the "duplicate year" case that
> the old `concat_years` caught on dataset-time-coord inspection is
> unreachable at the filename level. `_verify_year_coverage` documents
> this invariant in its docstring and does not test for it.

- [ ] **Step 4.3: Add failing test: empty directory raises**

```python
def test_verify_year_coverage_raises_on_empty(tmp_path):
    from nhf_spatial_targets.aggregate._driver import _verify_year_coverage

    d = tmp_path / "foo"
    d.mkdir()
    with pytest.raises(ValueError, match="no per-year"):
        _verify_year_coverage(d, "foo")
```

- [ ] **Step 4.4: Run the tests to verify they fail**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py -k verify_year_coverage -v
```

Expected: all FAIL — helper doesn't exist.

- [ ] **Step 4.5: Implement `_verify_year_coverage`**

Insert after `_migrate_legacy_layout` in `_driver.py`:

```python
_YEAR_FNAME_RE = re.compile(r"^(?P<key>.+)_(?P<year>\d{4})_agg\.nc$")


def _parse_year_from_filename(path: Path, source_key: str) -> int | None:
    """Return the year parsed from ``<source_key>_<YYYY>_agg.nc``, else None."""
    m = _YEAR_FNAME_RE.match(path.name)
    if m is None or m.group("key") != source_key:
        return None
    return int(m.group("year"))


def _verify_year_coverage(per_source_dir: Path, source_key: str) -> None:
    """Scan the per-source dir and verify contiguous year coverage.

    Filename-level check: parses ``<source_key>_<YYYY>_agg.nc`` matches.
    Raises ``ValueError`` if no matching files exist or if there is an
    interior gap between ``min_year`` and ``max_year``. Filesystem
    uniqueness guarantees no two files share the same filename, so
    duplicates are only possible via coincident filename shapes, which
    this function conservatively surfaces via the gap check as well.
    """
    years: list[int] = []
    for p in sorted(per_source_dir.glob(f"{source_key}_*_agg.nc")):
        y = _parse_year_from_filename(p, source_key)
        if y is not None:
            years.append(y)
    if not years:
        raise ValueError(
            f"{source_key}: no per-year aggregated files found in "
            f"{per_source_dir}"
        )
    expected = set(range(min(years), max(years) + 1))
    missing = sorted(expected - set(years))
    if missing:
        raise ValueError(
            f"{source_key}: year gap(s) in per-year aggregated files: "
            f"missing={missing}, covered={sorted(set(years))}"
        )
```

Also add `import re` at the top of `_driver.py` if it is not already imported.

- [ ] **Step 4.6: Run the tests to verify they pass**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py -k verify_year_coverage -v
```

Expected: all PASS.

- [ ] **Step 4.7: Commit**

```bash
git add src/nhf_spatial_targets/aggregate/_driver.py tests/test_aggregate_driver_per_year.py
git commit -m "$(cat <<'EOF'
feat: add _verify_year_coverage (filename-level gap detection)

Replaces concat_years's cross-year invariants with a cheap filename
scan. Raises on interior gaps or an empty per-source directory. No
dataset opens required.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Delete `concat_years` and its tests

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py` (remove function)
- Modify: `tests/test_aggregate_driver_per_year.py` (delete `concat_years` tests)

- [ ] **Step 5.1: Delete `concat_years` from `_driver.py`**

Remove the entire function (currently lines 292-324). Do not yet touch `aggregate_source` — Task 6 updates it to remove the call site.

Removing the function while `aggregate_source` still calls it will break tests. That's intentional: we verify the next task's test suite catches the regression, and we get a single coherent "concat_years is gone" commit.

- [ ] **Step 5.2: Delete all `concat_years` tests**

Remove these six tests from `tests/test_aggregate_driver_per_year.py`:

- `test_concat_years_orders_by_time`
- `test_concat_years_raises_on_duplicate_time`
- `test_concat_years_raises_on_year_gap`
- `test_concat_years_handles_cftime_calendar`
- `test_concat_years_cftime_year_gap_still_detected`
- `test_concat_years_detaches_from_disk`

Also remove the `_write_year_intermediate` helper if it is no longer referenced after Tasks 3/4 (they use it). Double-check: Task 3 tests reference it — keep it.

- [ ] **Step 5.3: Run tests to verify the broken state is exactly what we expect**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_driver_per_year.py tests/test_aggregate_driver.py -v
```

Expected: failures are limited to `aggregate_source` tests that exercise the full driver path (which still calls the now-missing `concat_years`). The migration and coverage tests should still pass.

- [ ] **Step 5.4: Commit**

```bash
git add src/nhf_spatial_targets/aggregate/_driver.py tests/test_aggregate_driver_per_year.py
git commit -m "$(cat <<'EOF'
refactor: delete concat_years (moves to per-source layout in #53)

concat_years loaded every per-year NC into RAM before xr.concat, causing
OOM on MOD10C1 at 128 GB. Its cross-year invariants now live in
_verify_year_coverage at the filename level. aggregate_source is
rewired in the next task to stop calling it.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Rework `update_manifest` and `aggregate_source` for the new contract

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py` (`update_manifest`, `aggregate_source`)
- Modify: `tests/test_aggregate_driver.py` (update `update_manifest` and `aggregate_source` tests)

- [ ] **Step 6.1: Update existing `update_manifest` tests to the new signature**

In `tests/test_aggregate_driver.py`, replace `test_update_manifest_writes_source_entry` body assertions:

```python
def test_update_manifest_writes_source_entry(project):
    update_manifest(
        project=project,
        source_key="foo",
        access={"type": "nasa_gesdisc", "short_name": "FOO"},
        period="2000-01-01/2009-12-31",
        output_files=[
            "data/aggregated/foo/foo_2000_agg.nc",
            "data/aggregated/foo/foo_2001_agg.nc",
        ],
        weight_files=["weights/foo_batch0.csv"],
    )
    manifest = json.loads((project.workdir / "manifest.json").read_text())
    entry = manifest["sources"]["foo"]
    assert entry["source_key"] == "foo"
    assert entry["access_type"] == "nasa_gesdisc"
    assert entry["short_name"] == "FOO"
    assert entry["period"] == "2000-01-01/2009-12-31"
    assert entry["fabric_sha256"] == "abc123"
    assert entry["output_files"] == [
        "data/aggregated/foo/foo_2000_agg.nc",
        "data/aggregated/foo/foo_2001_agg.nc",
    ]
    assert entry["weight_files"] == ["weights/foo_batch0.csv"]
    assert "timestamp" in entry
```

Also update `test_update_manifest_preserves_existing_sources` and `test_update_manifest_raises_on_corrupt_json` to pass `output_files=[...]` instead of `output_file=...`.

- [ ] **Step 6.2: Update `aggregate_source` integration tests**

Replace assertions in `test_aggregate_source_writes_multi_var_nc_and_manifest`:

```python
    # No consolidated file exists anymore.
    legacy_consolidated = tmp_path / "data" / "aggregated" / "merra2_agg.nc"
    assert not legacy_consolidated.exists()
    # Per-year file lives under data/aggregated/merra2/.
    per_year = (
        tmp_path / "data" / "aggregated" / "merra2" / "merra2_2000_agg.nc"
    )
    assert per_year.exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "merra2" in manifest["sources"]
    assert manifest["sources"]["merra2"]["output_files"] == [
        "data/aggregated/merra2/merra2_2000_agg.nc"
    ]
```

Also remove the `out` return-value assertion (`assert set(out.data_vars) == {"a", "b"}`) since `aggregate_source` will no longer return a concatenated Dataset — see step 6.4. Replace it with a no-op so the call-site assertion becomes:

```python
    aggregate_source(
        adapter,
        fabric_path=tiny_fabric,
        id_col="hru_id",
        workdir=tmp_path,
        batch_size=500,
    )
```

Same pattern in `test_aggregate_source_invokes_pre_aggregate_hook` and `test_aggregate_source_invokes_post_aggregate_hook`:

- Replace `out = aggregate_source(...)` with `aggregate_source(...)`.
- Replace assertions that read `out.data_vars` with assertions that open the written per-year file and check `data_vars`.
- For the post-hook test, the path to open is
  `tmp_path / "data" / "aggregated" / "merra2" / "merra2_2000_agg.nc"`.

- [ ] **Step 6.3: Run tests to verify they fail against the current code**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_driver.py -v
```

Expected: the updated tests FAIL — `update_manifest` still expects `output_file`, `aggregate_source` still returns a Dataset, etc.

- [ ] **Step 6.4: Update `update_manifest` and `aggregate_source`**

In `src/nhf_spatial_targets/aggregate/_driver.py`:

1. Change `update_manifest` signature and body:

```python
def update_manifest(
    project: Project,
    source_key: str,
    access: dict,
    period: str,
    output_files: list[str],
    weight_files: list[str],
) -> None:
    """Merge an aggregation provenance entry into ``manifest.json`` atomically.

    The manifest is keyed as ``sources[source_key]``; existing entries for
    other sources are preserved. ``period`` is stored as-is for provenance;
    ``fabric_sha256`` is read from ``fabric.json``. ``output_files`` lists
    each per-year NC relative to ``project.workdir``.
    """
    manifest_path = project.manifest_path
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {project.workdir} is corrupt: {exc}"
            ) from exc
    else:
        manifest = {"sources": {}, "steps": []}

    manifest.setdefault("sources", {})

    fabric_json = project.workdir / "fabric.json"
    fabric_sha = ""
    if fabric_json.exists():
        fabric_meta = json.loads(fabric_json.read_text())
        fabric_sha = fabric_meta.get("sha256", "")

    entry: dict = {
        "source_key": source_key,
        "access_type": access.get("type", ""),
        "period": period,
        "fabric_sha256": fabric_sha,
        "output_files": list(output_files),
        "weight_files": list(weight_files),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for extra_key in ("collection_id", "short_name", "version", "doi"):
        if extra_key in access:
            entry[extra_key] = access[extra_key]

    manifest["sources"][source_key] = entry

    tmp_fd, tmp_path = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp_path).replace(manifest_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    logger.info("Updated manifest.json with '%s' aggregation provenance", source_key)
```

2. Add a small helper for deriving the period by opening only the edge files:

```python
def _derive_period(per_year_paths: list[Path], time_coord: str) -> str:
    """Return ``'YYYY-MM-DD/YYYY-MM-DD'`` from the first/last per-year files.

    Opens only the first and last files (lazy, closed immediately) and reads
    the first/last ``time_coord`` values. Avoids opening intermediate files.
    """
    first, last = per_year_paths[0], per_year_paths[-1]
    with xr.open_dataset(first) as ds_first:
        t0 = str(ds_first[time_coord].values[0])[:10]
    with xr.open_dataset(last) as ds_last:
        t1 = str(ds_last[time_coord].values[-1])[:10]
    return f"{t0}/{t1}"
```

3. Replace `aggregate_source` with:

```python
def aggregate_source(
    adapter: SourceAdapter,
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
) -> None:
    """Aggregate a source to fabric HRU polygons; emit per-year NCs.

    Writes one NC per year to
    ``data/aggregated/<source_key>/<source_key>_<year>_agg.nc``. No
    consolidated single-file output is produced; per-year files are the
    canonical aggregated output. Idempotent: existing per-year files are
    preserved on restart. Legacy ``_by_year/`` files and stale
    ``<source_key>_agg.nc`` consolidated files are migrated via
    ``_migrate_legacy_layout`` at the top of the function.

    Variables declared by ``adapter.variables`` that are missing from the
    source NC cause ValueError before any year is aggregated — unless the
    adapter defines a ``pre_aggregate_hook`` (in which case the declared
    variables are constructed by the hook, not read from the raw NC).
    """
    workdir = Path(workdir)
    project = load_project(workdir)
    meta = catalog_source(adapter.source_key)

    _migrate_legacy_layout(project, adapter.source_key)

    raw_dir = project.raw_dir(adapter.source_key)
    files = sorted(raw_dir.glob(adapter.files_glob))
    if not files:
        raise FileNotFoundError(
            f"No NC matching '{adapter.files_glob}' found in {raw_dir}. "
            f"Run 'nhf-targets fetch {adapter.source_key}' first."
        )

    raw_grid_var = adapter.raw_grid_variable
    reference_grid: tuple | None = None
    for f in files:
        with xr.open_dataset(f) as peek:
            if adapter.pre_aggregate_hook is None:
                missing = [v for v in adapter.variables if v not in peek.data_vars]
                if missing:
                    raise ValueError(
                        f"{adapter.source_key}: variables {missing} missing from "
                        f"{f.name} (have {list(peek.data_vars)})"
                    )
            if raw_grid_var not in peek.data_vars:
                raise ValueError(
                    f"{adapter.source_key}: raw_grid_variable "
                    f"{raw_grid_var!r} missing from {f.name} "
                    f"(have {list(peek.data_vars)}). For adapters whose "
                    f"pre_aggregate_hook synthesizes declared variables, set "
                    f"SourceAdapter.raw_grid_variable to a variable that exists "
                    f"in the raw NC so the cross-year grid invariant can be "
                    f"enforced."
                )
            time_name = _find_time_coord_name(peek)
            shape = tuple(
                peek[raw_grid_var].sizes[d]
                for d in peek[raw_grid_var].dims
                if d != time_name
            )
            if reference_grid is None:
                reference_grid = shape
            elif shape != reference_grid:
                raise ValueError(
                    f"{adapter.source_key}: grid shape drift across source "
                    f"files — {f.name} has {shape}, expected {reference_grid}. "
                    f"Weight caches are reused across years and require a "
                    f"stable source grid."
                )

    year_files = enumerate_years(files)
    fabric_batched = load_and_batch_fabric(fabric_path, batch_size=batch_size)
    n_batches = int(fabric_batched["batch_id"].nunique())
    logger.info(
        "%s: %d years to aggregate across %d spatial batches",
        adapter.source_key,
        len(year_files),
        n_batches,
    )

    per_year_paths = [
        aggregate_year(
            adapter,
            project,
            year,
            path,
            fabric_batched,
            id_col,
            catalog_meta=meta,
        )
        for year, path in year_files
    ]

    per_source_dir = project.aggregated_dir() / adapter.source_key
    _verify_year_coverage(per_source_dir, adapter.source_key)

    with xr.open_dataset(per_year_paths[0]) as probe:
        time_coord = _find_time_coord_name(probe)
    if time_coord is None:
        raise ValueError(
            f"{adapter.source_key}: could not detect CF time coord on "
            f"per-year NC {per_year_paths[0].name}. Expected a "
            f"coord with axis='T' or standard_name='time'."
        )
    period = _derive_period(per_year_paths, time_coord)

    rel_output_files = [
        str(p.relative_to(project.workdir)) for p in per_year_paths
    ]
    update_manifest(
        project=project,
        source_key=adapter.source_key,
        access=meta.get("access", {}),
        period=period,
        output_files=rel_output_files,
        weight_files=[
            str(Path("weights") / f"{adapter.source_key}_batch{i}.csv")
            for i in range(n_batches)
        ],
    )
    logger.info(
        "%s: %d per-year NCs written to %s",
        adapter.source_key,
        len(per_year_paths),
        per_source_dir,
    )
```

Note: the function's return type changes from `xr.Dataset` to `None`. Callers should read per-year files directly; nothing in the repo currently needs the return value (grep confirms this in step 6.5).

- [ ] **Step 6.5: Verify no callers rely on the old return type**

Run:
```bash
grep -Rn "aggregate_source(" src/ tests/
```

Expected: only the adapter convenience functions in `aggregate/<source>.py` call it, each with `return aggregate_source(...)`. Update each adapter's wrapper function to drop the `return` since `aggregate_source` now returns `None`. Files to update:

- `src/nhf_spatial_targets/aggregate/era5_land.py`
- `src/nhf_spatial_targets/aggregate/gldas.py`
- `src/nhf_spatial_targets/aggregate/merra2.py`
- `src/nhf_spatial_targets/aggregate/mod10c1.py`
- `src/nhf_spatial_targets/aggregate/mod16a2.py`
- `src/nhf_spatial_targets/aggregate/ncep_ncar.py`
- `src/nhf_spatial_targets/aggregate/nldas_mosaic.py`
- `src/nhf_spatial_targets/aggregate/nldas_noah.py`
- `src/nhf_spatial_targets/aggregate/watergap22d.py`

For each: change the wrapper function's signature return annotation to `None` and replace `return aggregate_source(...)` with `aggregate_source(...)`.

For example, `mod10c1.py` becomes:

```python
def aggregate_mod10c1(
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
) -> None:
    """Aggregate MOD10C1 v061 daily SCA to HRU polygons with CI masking.

    The CI>0.70 filter is applied per-year inside the driver's per-year
    loop via ``pre_aggregate_hook``. Output is one NC per year at
    ``data/aggregated/mod10c1_v061/mod10c1_v061_<year>_agg.nc``; each
    carries ``sca``, ``ci``, and ``valid_area_fraction`` keyed on
    (time, HRU).
    """
    aggregate_source(ADAPTER, fabric_path, id_col, workdir, batch_size)
```

Apply the analogous change to every wrapper. If any wrapper was only `return aggregate_source(...)` and has no other logic, just change the `return` to a bare call.

- [ ] **Step 6.6: Verify no test expects a Dataset return value**

Run:
```bash
grep -Rn "out = aggregate_source" tests/
grep -Rn "result = aggregate_source" tests/
```

Expected: all existing `out =` / `result =` call sites are the three we already updated in step 6.2. If any remain, update them to drop the assignment and rewrite assertions to open the per-year NC from disk instead.

- [ ] **Step 6.7: Run the updated driver tests**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_driver.py tests/test_aggregate_driver_per_year.py -v
```

Expected: all PASS.

- [ ] **Step 6.8: Run the full test suite**

Run:
```bash
pixi run -e dev test
```

Expected: all PASS. `tests/test_aggregate_mod10c1.py::test_log_low_valid_coverage_*` continue to pass at this step — they call `_log_low_valid_coverage` directly and the signature has not changed yet. Task 7 updates both the function and the tests together.

- [ ] **Step 6.9: Commit**

```bash
git add src/nhf_spatial_targets/aggregate tests/test_aggregate_driver.py tests/test_aggregate_driver_per_year.py
git commit -m "$(cat <<'EOF'
feat(#53): per-source aggregated layout; drop consolidated output

aggregate_source now emits one NC per year under
data/aggregated/<source_key>/ and no longer produces a consolidated
<source_key>_agg.nc. Eliminates the concat_years OOM. manifest.json
records output_files: list[str]; period is derived from the first/last
per-year files without a full load. Legacy _by_year/ files and stale
consolidated files migrate automatically on next run.

Source wrappers return None. Downstream target builders must open the
per-year directory (out of scope for this PR).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Make `mod10c1._log_low_valid_coverage` per-year and name the year

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/mod10c1.py`
- Modify: `tests/test_aggregate_mod10c1.py`

- [ ] **Step 7.1: Update `_log_low_valid_coverage` tests to take a `year` argument**

Replace the two existing tests in `tests/test_aggregate_mod10c1.py`:

```python
def test_log_low_valid_coverage_warns_above_threshold(caplog):
    import logging

    from nhf_spatial_targets.aggregate.mod10c1 import _log_low_valid_coverage

    times = pd.date_range("2000-01-01", periods=10, freq="D")
    vaf_data = np.zeros((10, 10))
    vaf_data[0, 0] = 1.0  # 1 nonzero, 99 zero, 0 NaN -> 99% zero
    year_ds = xr.Dataset(
        {"valid_area_fraction": (["time", "hru_id"], vaf_data)},
        coords={"time": times, "hru_id": range(10)},
    )
    with caplog.at_level(
        logging.WARNING, logger="nhf_spatial_targets.aggregate.mod10c1"
    ):
        _log_low_valid_coverage(year_ds, year=2000)
    warnings = [rec.message for rec in caplog.records]
    assert any("zero valid-area" in m for m in warnings), warnings
    assert any("year=2000" in m for m in warnings), warnings


def test_log_low_valid_coverage_silent_below_threshold(caplog):
    import logging

    from nhf_spatial_targets.aggregate.mod10c1 import _log_low_valid_coverage

    times = pd.date_range("2000-01-01", periods=10, freq="D")
    vaf_data = np.ones((10, 10)) * 0.8
    vaf_data[0, 0] = 0.0
    year_ds = xr.Dataset(
        {"valid_area_fraction": (["time", "hru_id"], vaf_data)},
        coords={"time": times, "hru_id": range(10)},
    )
    with caplog.at_level(
        logging.WARNING, logger="nhf_spatial_targets.aggregate.mod10c1"
    ):
        _log_low_valid_coverage(year_ds, year=2000)
    assert not any(
        "zero valid-area" in rec.message for rec in caplog.records
    )
```

- [ ] **Step 7.2: Run the tests to verify they fail**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_mod10c1.py -k log_low_valid_coverage -v
```

Expected: FAIL — `_log_low_valid_coverage` has no `year` kwarg.

- [ ] **Step 7.3: Update `_log_low_valid_coverage` and `_rename_and_warn`**

In `src/nhf_spatial_targets/aggregate/mod10c1.py`:

```python
def _log_low_valid_coverage(year_ds: xr.Dataset, *, year: int) -> None:
    """Warn if > 10% of (HRU, time) cells for this year have zero valid area."""
    vaf = year_ds["valid_area_fraction"]
    n_total = int(vaf.notnull().sum())
    if n_total == 0:
        return
    n_zero = int(((vaf == 0) & vaf.notnull()).sum())
    zero_frac = n_zero / n_total
    if zero_frac > _LOW_COVERAGE_WARN_THRESHOLD:
        logger.warning(
            "mod10c1 year=%d: %.1f%% of (HRU, time) cells had zero "
            "valid-area after CI>%.2f filter (n=%d of %d finite). "
            "Downstream sca values are NaN for these cells.",
            year,
            zero_frac * 100,
            _CI_THRESHOLD,
            n_zero,
            n_total,
        )


def _rename_and_warn(year_ds: xr.Dataset) -> xr.Dataset:
    year_ds = year_ds.rename({"valid_mask": "valid_area_fraction"})
    year_ds["valid_area_fraction"].attrs = {
        "long_name": "fraction of HRU area that passed CI filter",
        "units": "1",
        "ci_threshold": _CI_THRESHOLD,
    }
    # Derive the year from the year_ds time coord so the warning names it.
    year = int(pd.DatetimeIndex(year_ds["time"].values).year[0])
    _log_low_valid_coverage(year_ds, year=year)
    return year_ds
```

Note: `_rename_and_warn` is wired to `post_aggregate_hook`, which (after Task 2) runs once per year inside `aggregate_year`, so it receives exactly one year's Dataset. Deriving `year` from the first timestep is safe.

- [ ] **Step 7.4: Run the tests to verify they pass**

Run:
```bash
pixi run -e dev test -- tests/test_aggregate_mod10c1.py -v
```

Expected: all PASS.

- [ ] **Step 7.5: Commit**

```bash
git add src/nhf_spatial_targets/aggregate/mod10c1.py tests/test_aggregate_mod10c1.py
git commit -m "$(cat <<'EOF'
refactor: mod10c1 low-valid-coverage warning is per-year and names the year

Since post_aggregate_hook now runs inside aggregate_year (once per year),
_log_low_valid_coverage takes the year explicitly and includes it in the
log message. The warning is more actionable — users see which year had
bad CI coverage, not a blended cross-year summary.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Full suite + lint/format gate

- [ ] **Step 8.1: Run formatter**

Run:
```bash
pixi run -e dev fmt
```

Expected: no diffs (or only whitespace). If the formatter reformats anything substantive, re-read the diff and confirm intent before committing.

- [ ] **Step 8.2: Run linter**

Run:
```bash
pixi run -e dev lint
```

Expected: clean. Fix any warnings introduced by the refactor before continuing.

- [ ] **Step 8.3: Run the full test suite**

Run:
```bash
pixi run -e dev test
```

Expected: all PASS.

- [ ] **Step 8.4: Commit format/lint fixes (if any)**

```bash
git status
# If changes are present:
git add -u
git commit -m "chore: ruff format/lint after #53 refactor

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Open the PR

- [ ] **Step 9.1: Push branch**

```bash
git push -u origin fix/53-per-source-aggregate-layout
```

- [ ] **Step 9.2: Open PR referencing #53**

```bash
gh pr create --title "Fix #53: per-source aggregate layout; drop concat_years OOM" --body "$(cat <<'EOF'
## Summary
- Deletes `concat_years()` — the source of the MOD10C1 OOM at 128 GB.
- `aggregate_source` now emits one NC per year under `data/aggregated/<source_key>/`.
- Idempotent legacy-layout migration shim moves existing `_by_year/` files into the per-source dir and removes stale consolidated outputs.
- `post_aggregate_hook` and CF global attrs move into `aggregate_year` so each per-year file is independently CF-1.6 compliant.
- `manifest.json` records `output_files: list[str]`; `period` is derived from first/last per-year files (two small opens, no full load).

Closes #53.

## Test plan
- [ ] `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
- [ ] On a real project with existing mod10c1 `_by_year/` files: rerun `pixi run nhf-targets agg mod10c1 --project-dir <project>` and confirm per-year files now live under `data/aggregated/mod10c1_v061/` with no re-aggregation cost.
- [ ] Confirm `manifest.json` entry for `mod10c1_v061` carries `output_files: [...]` and no `output_file` key.

Spec: `docs/superpowers/specs/2026-04-16-per-source-aggregate-layout-design.md`
Plan: `docs/superpowers/plans/2026-04-16-per-source-aggregate-layout.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 9.3: Report PR URL to the user**

After creation, `gh pr create` prints the PR URL. Share it with the user.

---

## Plan Self-Review Notes

- Every spec section maps to a task:
  - "Architecture" → Tasks 1–6
  - "Directory layout" → Task 1
  - "Manifest entry shape" → Task 6 (`update_manifest` + `aggregate_source`)
  - "Component changes" → Tasks 1–7
  - "Error handling" → Tasks 3 (migration), 4 (coverage)
  - "Testing" → Tasks 1–7 (every code change lands with a test first)
  - "Migration & operational impact" → Task 3 (shim) + Task 6 (wired into `aggregate_source`)
- No placeholders, no "similar to" references, no unresolved TBDs.
- Function signatures are consistent across tasks: `_migrate_legacy_layout(project, source_key)`, `_verify_year_coverage(per_source_dir, source_key)`, `_derive_period(per_year_paths, time_coord)`, `aggregate_year(..., *, catalog_meta=None)`, `aggregate_source(...) -> None`, `update_manifest(..., output_files, ...)`.
- Test names match the fixtures and helpers in the existing test files (`project`, `tiny_batched_fabric`, `tiny_fabric`, `_write_year_intermediate`, `_write_nc`, `_setup_aggregate_source_project`).
