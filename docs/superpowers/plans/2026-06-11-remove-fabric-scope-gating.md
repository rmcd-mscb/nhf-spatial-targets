# Remove fabric_scope / fabric.token Gating Implementation Plan (#309)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the hand-maintained `fabric_scope` catalog block and `fabric.token` config knob; replace the aggregation-stage token gate with a geometry-driven coverage guard, and rely on the existing NaN-aware multi-source combine at target stage.

**Architecture:** gdptools crashes (`max() iterable argument is empty` in `UserCatData`, from `resX=max(np.diff(...))` on an empty bbox subset) when a spatial batch has < 2 grid coords overlap — that is the only technical reason the token gate existed. We add `_covered_batch_ids()` to `aggregate/_driver.py` that classifies each fabric batch against the source grid bbox (mirroring gdptools' own 2×-max-resolution buffer), skip uncovered batches, and reindex each per-year NC to the full sorted fabric HRU set so uncovered HRUs are honest NaN. Zero total overlap → skip the source with an INFO log (geometry-driven replacement for the old skip). The SWE target's token filter, the catalog tokens/validator, and the config knob are then deleted; the NaN-aware combine (`multi_source_nanminmax`) already handles partial sources.

**Tech Stack:** Python 3.11+, xarray, geopandas, shapely, pyproj, gdptools 0.3.13, pytest via pixi.

**Key facts established during research (do not re-derive):**

- gdptools 0.3.13 has no graceful empty-intersection path — the guard is required (issue wrinkle 1 resolved: coverage guard).
- `targets/_io.py:check_hru_coords` requires aggregated NCs to carry the **exact** fabric HRU set → partial-coverage aggregation must reindex to the full fabric (NaN rows), not emit a subset.
- The Margulis fetch bbox is the **project fabric's buffered bbox** (`fabric.json:bbox_buffered`), not a hardcoded Oregon box → wrinkle 3 resolved: no fetch change needed; a CONUS project naturally fetches the full WUS domain.
- `defaults.find_unknown_keys` + `validate._report_defaults` already print `[warning] unknown config key: fabric.token` for leftover keys → no new migration machinery needed; existing projects degrade gracefully.
- `agg all` runs margulis like any other source after this change: missing raw → `FileNotFoundError` (consistent with every other unfetched source). Documented, intended.
- The release README fixture text (`tests/fixtures/release/expected_fabric_gfv2_readme.md:25`) is rendered verbatim from `catalog/variables.yml` `snow_water_equivalent.description` — both must change together.
- gfv2 SWE results change deliberately (margulis can now contribute on CONUS where aggregated); call out in PR body.
- Presentations under `docs/presentations/` are dated talk artifacts — intentionally NOT updated; say so in the PR's docs-sync note.

---

## File Structure

| File | Change |
|---|---|
| `src/nhf_spatial_targets/aggregate/_driver.py` | + `_covered_batch_ids()`; `aggregate_year` gains `covered_batch_ids` param + full-fabric reindex; `aggregate_source` zero-overlap skip + coverage diagnostic; − `_skip_for_fabric_scope` |
| `src/nhf_spatial_targets/targets/swe.py` | − `_filter_sources_by_fabric_scope`, token validation; `_resolve_sources` → 2-tuple; − `fabric_token` attr; docstrings |
| `src/nhf_spatial_targets/targets/_driver.py` | comment fix (line ~402) |
| `src/nhf_spatial_targets/catalog.py` | − `FABRIC_SCOPE_TOKENS`, − `validate_fabric_scope` |
| `src/nhf_spatial_targets/defaults.py` | − `fabric.token` default |
| `src/nhf_spatial_targets/init_run.py` | − token stub; comment fix line 157 |
| `src/nhf_spatial_targets/upgrade_config.py` | − `fabric.token` feature entry; docstring |
| `src/nhf_spatial_targets/cli/maintenance.py` | docstring example fix |
| `src/nhf_spatial_targets/cli/fetch.py` | margulis docstring fix |
| `src/nhf_spatial_targets/fetch/margulis_wus_sr.py` | − fabric_scope reads/warning/manifest fields; docstrings |
| `src/nhf_spatial_targets/aggregate/margulis_wus_sr.py` | docstring |
| `catalog/sources.yml` | − margulis `fabric_scope` block; prose |
| `catalog/variables.yml` | SWE description / range_notes / sources comment |
| `tests/test_aggregate_driver.py` | − 5 token tests + helper; + coverage-guard tests |
| `tests/test_targets_swe.py` | fixture de-tokenized; − 3 tests; edits; + partial-coverage test |
| `tests/test_targets_sca.py` | fixture de-tokenized |
| `tests/test_catalog.py` | − 7 tests; rename 1; comment fix |
| `tests/test_init_run.py` | − 1 test |
| `tests/test_upgrade_config.py` | − 1 test; set edit |
| `tests/test_aggregate_margulis_wus_sr.py` | − 1 test; rewrite 1 |
| `tests/test_fetch_margulis_wus_sr.py` | rewrite 1 test |
| `tests/fixtures/release/expected_fabric_gfv2_readme.md` | SWE description cell |
| `CLAUDE.md`, `README.md`, `CONTRIBUTING.md`, `docs/index.md`, `docs/sources/margulis_wus_sr.md`, `docs/references/calibration-target-recipes.md`, `docs/references/target-period-coverage.md`, `docs/maintenance.md` | prose sweep |

Run all tests/lint via pixi: `pixi run -e dev test`, targeted: `pixi run -e dev python -m pytest tests/<file>::<test> -v` (or `pixi run -e dev pytest ...` if that task alias exists — check `pixi task list`; elsewhere in this plan `pytest` means the working invocation).

---

### Task 0: Branch

- [ ] **Step 0.1:** `git checkout main && git pull --ff-only && git checkout -b refactor/309-remove-fabric-scope-gating`

---

### Task 1: Coverage classifier `_covered_batch_ids`

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py` (add helper near `_skip_for_fabric_scope`, ~line 1005)
- Test: `tests/test_aggregate_driver.py`

- [ ] **Step 1.1: Write the failing tests** — append to `tests/test_aggregate_driver.py` (a `# --- Coverage guard (#309) ---` section; reuse module imports `gpd`, `np`, `pd`, `xr`, `box` already at top of file — verify and add any missing):

```python
# ---------------------------------------------------------------------------
# Geometry-driven coverage guard (#309 — replaces the fabric_scope token gate)
# ---------------------------------------------------------------------------


def _grid_ds(lons, lats, var="a", lon_attrs=None):
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    return xr.Dataset(
        {var: (["time", "lat", "lon"], np.ones((12, len(lats), len(lons))))},
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "lat": ("lat", list(lats), {"standard_name": "latitude"}),
            "lon": ("lon", list(lons), lon_attrs or {"standard_name": "longitude"}),
        },
    )


def _write_grid_nc(tmp_path, lons, lats, var="a"):
    p = tmp_path / "grid.nc"
    _grid_ds(lons, lats, var=var).to_netcdf(p)
    return p


def _batched_fabric_gdf(fabric_path, batch_size=500):
    from nhf_spatial_targets.aggregate._driver import load_and_batch_fabric

    return load_and_batch_fabric(fabric_path, batch_size=batch_size)


def test_covered_batch_ids_full_overlap(tmp_path, tiny_fabric):
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import _covered_batch_ids

    nc = _write_grid_nc(tmp_path, lons=[0.5, 1.5], lats=[0.25, 0.75])
    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=["a"]
    )
    fabric = _batched_fabric_gdf(tiny_fabric)
    covered = _covered_batch_ids(nc, adapter, fabric)
    assert covered == set(fabric["batch_id"].unique())


def test_covered_batch_ids_zero_overlap(tmp_path, tiny_fabric):
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import _covered_batch_ids

    # Grid far away from the tiny_fabric (x 0-4, y 0-1).
    nc = _write_grid_nc(tmp_path, lons=[100.5, 101.5], lats=[50.25, 50.75])
    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=["a"]
    )
    covered = _covered_batch_ids(nc, adapter, _batched_fabric_gdf(tiny_fabric))
    assert covered == set()


def test_covered_batch_ids_partial_overlap_two_clusters(tmp_path):
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import _covered_batch_ids

    polys = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(10, 0, 11, 1), box(11, 0, 12, 1)]
    gdf = gpd.GeoDataFrame({"hru_id": [0, 1, 2, 3]}, geometry=polys, crs="EPSG:4326")
    fabric_path = tmp_path / "fabric2.gpkg"
    gdf.to_file(fabric_path, driver="GPKG")
    fabric = _batched_fabric_gdf(fabric_path, batch_size=2)
    assert fabric["batch_id"].nunique() == 2  # KD-tree split holds

    nc = _write_grid_nc(tmp_path, lons=[0.5, 1.5], lats=[0.25, 0.75])
    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=["a"]
    )
    covered = _covered_batch_ids(nc, adapter, fabric)
    # Exactly the batch containing HRUs 0/1 (x 0-2) is covered.
    covered_hrus = set(
        fabric.loc[fabric["batch_id"].isin(list(covered)), "hru_id"]
    )
    assert covered_hrus == {0, 1}


def test_covered_batch_ids_handles_0_360_longitudes(tmp_path):
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import _covered_batch_ids

    # Fabric at CONUS-style negative lons; grid published on 0-360.
    polys = [box(-124.5, 42.0, -123.5, 43.0)]
    gdf = gpd.GeoDataFrame({"hru_id": [0]}, geometry=polys, crs="EPSG:4326")
    fabric_path = tmp_path / "fabric_neg.gpkg"
    gdf.to_file(fabric_path, driver="GPKG")

    nc = _write_grid_nc(tmp_path, lons=[235.5, 236.5], lats=[42.25, 42.75])
    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=["a"]
    )
    fabric = _batched_fabric_gdf(fabric_path)
    assert _covered_batch_ids(nc, adapter, fabric) == set(
        fabric["batch_id"].unique()
    )
```

- [ ] **Step 1.2:** Run: `pixi run -e dev python -m pytest tests/test_aggregate_driver.py -k covered_batch_ids -v` — Expected: 4 FAIL with `ImportError`/`AttributeError` (`_covered_batch_ids` not defined).

- [ ] **Step 1.3: Implement** — in `src/nhf_spatial_targets/aggregate/_driver.py`, add directly above `_skip_for_fabric_scope` (verify `detect_coords` is already imported from `._coords` at module top — it is, used by `aggregate_year`):

```python
def _covered_batch_ids(
    source_file: Path,
    adapter: SourceAdapter,
    fabric_batched: gpd.GeoDataFrame,
) -> set[int]:
    """Classify which spatial batches overlap the source grid's bbox.

    Geometry-driven coverage guard (#309, replaces the retired catalog
    ``fabric_scope`` token gate). gdptools' ``UserCatData`` subsets the
    source grid to the target bbox buffered by twice the maximum cell
    size and crashes with ``max() iterable argument is empty`` when the
    subset spans fewer than two grid coords along either axis. A batch
    is therefore "covered" when at least two x and two y grid coords
    fall inside the batch bbox padded by the same 2×-max-resolution
    buffer gdptools applies (``_get_shp_bounds_w_buffer``).

    Coordinates are detected from ``adapter.raw_grid_variable`` on the
    raw (un-hooked) file — the same variable the cross-year grid-shape
    check uses, guaranteed present by ``aggregate_source``'s preflight.

    Longitude convention: for geographic source grids published on
    0–360 longitudes, negative batch-bound longitudes are shifted +360
    before comparison. Anti-meridian-crossing fabrics are out of scope
    (no current fabric crosses it).
    """
    from pyproj import CRS
    from shapely.geometry import box as _box

    with xr.open_dataset(source_file) as ds:
        x_coord, y_coord, _ = detect_coords(
            ds,
            adapter.raw_grid_variable,
            x_override=adapter.x_coord,
            y_override=adapter.y_coord,
            time_override=adapter.time_coord,
        )
        xs = np.asarray(ds[x_coord].values, dtype="float64")
        ys = np.asarray(ds[y_coord].values, dtype="float64")

    pad_x = 2.0 * float(np.max(np.abs(np.diff(xs)))) if xs.size > 1 else 0.0
    pad_y = 2.0 * float(np.max(np.abs(np.diff(ys)))) if ys.size > 1 else 0.0
    shift_lon = (
        CRS.from_user_input(adapter.source_crs).is_geographic
        and float(xs.max()) > 180.0
    )

    bids = [int(b) for b in sorted(fabric_batched["batch_id"].unique())]
    batch_boxes = gpd.GeoSeries(
        [
            _box(*fabric_batched.loc[fabric_batched["batch_id"] == b].total_bounds)
            for b in bids
        ],
        crs=fabric_batched.crs,
    ).to_crs(adapter.source_crs)

    covered: set[int] = set()
    for bid, geom in zip(bids, batch_boxes, strict=True):
        minx, miny, maxx, maxy = geom.bounds
        if shift_lon:
            minx = minx + 360.0 if minx < 0 else minx
            maxx = maxx + 360.0 if maxx < 0 else maxx
        n_x = int(np.count_nonzero((xs >= minx - pad_x) & (xs <= maxx + pad_x)))
        n_y = int(np.count_nonzero((ys >= miny - pad_y) & (ys <= maxy + pad_y)))
        if n_x >= 2 and n_y >= 2:
            covered.add(bid)
    return covered
```

If `adapter.x_coord`/`y_coord`/`time_coord` attribute names differ, read `SourceAdapter` in `aggregate/_adapter.py` and use its actual field names (they are the ones `aggregate_year` passes to `detect_coords`).

- [ ] **Step 1.4:** Run: `pixi run -e dev python -m pytest tests/test_aggregate_driver.py -k covered_batch_ids -v` — Expected: 4 PASS. If the two-cluster KD-tree assertion (`nunique() == 2`) fails, inspect `spatial_batch` semantics and adjust `batch_size`/polygon spacing until two batches form, keeping the covered/uncovered split.

- [ ] **Step 1.5: Commit**

```bash
git add tests/test_aggregate_driver.py src/nhf_spatial_targets/aggregate/_driver.py
pixi run git commit -m "feat(#309): add geometry-driven batch coverage classifier to agg driver"
```

---

### Task 2: `aggregate_year` partial-coverage NaN reindex

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py:633-756` (`aggregate_year`)
- Test: `tests/test_aggregate_driver.py`

- [ ] **Step 2.1: Write the failing tests.** These drive `aggregate_year` through `aggregate_source` with mocked weights/agg, like `test_aggregate_source_writes_multi_var_nc_and_manifest` (line 352) — copy its project-skeleton setup pattern:

```python
def _partial_coverage_project(tmp_path, lons, lats):
    """Two-cluster fabric (batches at x 0-2 and x 10-12) + grid NC."""
    polys = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(10, 0, 11, 1), box(11, 0, 12, 1)]
    gdf = gpd.GeoDataFrame({"hru_id": [0, 1, 2, 3]}, geometry=polys, crs="EPSG:4326")
    fabric_path = tmp_path / "fabric2.gpkg"
    gdf.to_file(fabric_path, driver="GPKG")

    datastore = tmp_path / "datastore"
    (datastore / "merra2").mkdir(parents=True)
    src_nc = datastore / "merra2" / "merra2_2000_consolidated.nc"
    _grid_ds(lons, lats).to_netcdf(src_nc)

    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": str(fabric_path), "id_col": "hru_id"},
                "datastore": str(datastore),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))
    (tmp_path / "data" / "aggregated").mkdir(parents=True)
    (tmp_path / "weights").mkdir()
    return fabric_path


def _fake_agg_for_batch(batch_gdf, **kwargs):
    """Echo the batch's own HRUs back, mimicking gdptools output."""
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    hrus = batch_gdf["hru_id"].to_list()
    return xr.Dataset(
        {"a": (["time", "hru_id"], np.ones((12, len(hrus))))},
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "hru_id": hrus,
        },
    )


def test_aggregate_source_partial_coverage_emits_full_fabric_with_nan(
    tmp_path, caplog
):
    """#309: uncovered batches are skipped; the per-year NC still carries
    every fabric HRU, with honest NaN rows at uncovered HRUs."""
    import logging

    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    fabric_path = _partial_coverage_project(
        tmp_path, lons=[0.5, 1.5], lats=[0.25, 0.75]
    )
    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=["a"]
    )
    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.catalog_source",
            return_value={"access": {"type": "local_nc"}},
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=_fake_weights(),
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            side_effect=lambda **kw: _fake_agg_for_batch(**kw),
        ),
        caplog.at_level(logging.INFO, logger="nhf_spatial_targets.aggregate._driver"),
    ):
        aggregate_source(
            adapter,
            fabric_path=fabric_path,
            id_col="hru_id",
            workdir=tmp_path,
            batch_size=2,
        )

    per_year = tmp_path / "data" / "aggregated" / "merra2" / "merra2_2000_agg.nc"
    with xr.open_dataset(per_year) as ds:
        out = ds.load()
    assert out["hru_id"].values.tolist() == [0, 1, 2, 3]
    vals = out["a"].values
    assert np.isfinite(vals[:, :2]).all()  # covered cluster
    assert np.isnan(vals[:, 2:]).all()  # uncovered cluster → honest NaN
    assert "partial fabric coverage" in caplog.text


def test_aggregate_source_zero_coverage_skips_cleanly(tmp_path, caplog):
    """#309: a source grid with no fabric overlap is skipped with an INFO
    log — no aggregated dir, no manifest entry, no gdptools crash."""
    import logging

    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    fabric_path = _partial_coverage_project(
        tmp_path, lons=[100.5, 101.5], lats=[50.25, 50.75]
    )
    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=["a"]
    )
    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.catalog_source",
            return_value={"access": {"type": "local_nc"}},
        ),
        caplog.at_level(logging.INFO, logger="nhf_spatial_targets.aggregate._driver"),
    ):
        aggregate_source(
            adapter,
            fabric_path=fabric_path,
            id_col="hru_id",
            workdir=tmp_path,
            batch_size=2,
        )

    assert not (tmp_path / "data" / "aggregated" / "merra2").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "merra2" not in manifest["sources"]
    assert "no spatial overlap" in caplog.text


def test_aggregate_year_rejects_hru_ids_absent_from_fabric(tmp_path):
    """The full-fabric reindex must fail loudly (not silently emit all-NaN)
    when aggregated ids don't match the fabric — e.g. id dtype drift or a
    stale weight cache from a different fabric."""
    from nhf_spatial_targets.aggregate._adapter import SourceAdapter
    from nhf_spatial_targets.aggregate._driver import aggregate_source

    fabric_path = _partial_coverage_project(
        tmp_path, lons=[0.5, 1.5], lats=[0.25, 0.75]
    )
    adapter = SourceAdapter(
        source_key="merra2", output_name="merra2_agg.nc", variables=["a"]
    )

    def _alien_ids(**kwargs):
        times = pd.date_range("2000-01-01", periods=12, freq="MS")
        return xr.Dataset(
            {"a": (["time", "hru_id"], np.ones((12, 1)))},
            coords={
                "time": ("time", times, {"standard_name": "time"}),
                "hru_id": [99],
            },
        )

    with (
        patch(
            "nhf_spatial_targets.aggregate._driver.catalog_source",
            return_value={"access": {"type": "local_nc"}},
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.compute_or_load_weights",
            return_value=_fake_weights(),
        ),
        patch(
            "nhf_spatial_targets.aggregate._driver.aggregate_variables_for_batch",
            side_effect=lambda **kw: _alien_ids(**kw),
        ),
        pytest.raises(ValueError, match="absent from the fabric"),
    ):
        aggregate_source(
            adapter,
            fabric_path=fabric_path,
            id_col="hru_id",
            workdir=tmp_path,
            batch_size=2,
        )
```

(Reuses `_grid_ds` from Task 1 and the file's existing `yaml`/`json`/`patch`/`_fake_weights` imports/helpers.)

- [ ] **Step 2.2:** Run: `pixi run -e dev python -m pytest tests/test_aggregate_driver.py -k "partial_coverage or zero_coverage or rejects_hru_ids" -v` — Expected: FAIL (no skip/reindex logic yet; zero-coverage case may crash inside mocked-out weight path or write all 4 batches finite).

- [ ] **Step 2.3: Implement `aggregate_year` changes.**
  1. Signature: add keyword-only param after `catalog_meta`:
     ```python
         *,
         catalog_meta: dict | None = None,
         covered_batch_ids: set[int] | None = None,
     ```
     (Note `catalog_meta` is currently keyword-only already via `*`; keep one `*`.) Docstring: add — `covered_batch_ids: when given (from _covered_batch_ids), batches not in the set are skipped and their HRUs become NaN rows via the full-fabric reindex; None aggregates every batch.`
  2. Top of the batch loop:
     ```python
             for bid in sorted(fabric_batched["batch_id"].unique()):
                 if covered_batch_ids is not None and int(bid) not in covered_batch_ids:
                     continue
     ```
  3. Guard before `xr.concat` (still inside the `with` block):
     ```python
             if not datasets:
                 raise ValueError(
                     f"{adapter.source_key}: year {year}: no spatial batch overlaps "
                     f"the source grid; aggregate_source should have skipped this "
                     f"source before the per-year loop."
                 )
     ```
  4. Replace the `# Canonical row order on emission ...` comment + `year_ds = year_ds.sortby(id_col)` with:
     ```python
         # Canonical row order + full-fabric HRU set on emission (#93, #309).
         # gdptools concatenates batches in iteration order (typically
         # VPU-grouped), and batches outside the source grid are skipped for
         # partial-coverage sources — reindexing to the sorted full-fabric id
         # set restores id_col-ascending order and inserts honest NaN rows
         # for HRUs the source does not cover.
         all_ids = np.sort(fabric_batched[id_col].to_numpy())
         extra = np.setdiff1d(year_ds[id_col].to_numpy(), all_ids)
         if extra.size:
             raise ValueError(
                 f"{adapter.source_key}: year {year}: aggregated output carries "
                 f"{extra.size} HRU id(s) absent from the fabric (e.g. "
                 f"{extra[:5].tolist()}) — id_col dtype mismatch or a stale "
                 f"weight cache from a different fabric."
             )
         year_ds = year_ds.reindex({id_col: all_ids})
     ```

- [ ] **Step 2.4: Implement `aggregate_source` wiring.** After the `n_batches` logging block (post `load_and_batch_fabric`, ~line 1226) insert:

```python
    # Geometry-driven coverage guard (#309, replaces the catalog
    # fabric_scope token gate). Zero overlap → skip the source for this
    # fabric; partial overlap → aggregate covered batches and emit NaN
    # rows for the rest (targets' NaN-aware combine reads them as "no
    # data here").
    covered = _covered_batch_ids(assigned_year_files[0][1], adapter, fabric_batched)
    if not covered:
        logger.info(
            "%s: skipping aggregation — the source grid has no spatial "
            "overlap with this fabric. Raw downloads remain reusable by "
            "projects whose fabric the source covers.",
            adapter.source_key,
        )
        return
    if len(covered) < n_batches:
        n_hrus = len(fabric_batched)
        n_covered_hrus = int(fabric_batched["batch_id"].isin(list(covered)).sum())
        logger.info(
            "%s: partial fabric coverage — source grid bbox overlaps %d of "
            "%d batches (%d of %d HRUs, %.1f%%); HRUs outside the source "
            "grid will be NaN in the aggregated output.",
            adapter.source_key,
            len(covered),
            n_batches,
            n_covered_hrus,
            n_hrus,
            100.0 * n_covered_hrus / n_hrus,
        )
```

and pass `covered_batch_ids=covered` in the `aggregate_year(...)` call (line ~1229).

- [ ] **Step 2.5:** Run: `pixi run -e dev python -m pytest tests/test_aggregate_driver.py -v` — Expected: the three new tests PASS; **older `_skip_for_fabric_scope` tests still pass** (gate not yet removed). If any pre-existing `aggregate_source` test fails inside `_covered_batch_ids` (synthetic grid not overlapping its fabric), fix the test's grid coords to overlap — do not weaken the guard.

- [ ] **Step 2.6: Commit**

```bash
git add tests/test_aggregate_driver.py src/nhf_spatial_targets/aggregate/_driver.py
pixi run git commit -m "feat(#309): partial-coverage aggregation — skip uncovered batches, reindex per-year NCs to full fabric"
```

---

### Task 3: Remove the agg-stage token gate

**Files:**
- Modify: `src/nhf_spatial_targets/aggregate/_driver.py` (− `_skip_for_fabric_scope` lines 1005-1037, − call at 1109-1110, docstring 1083-1087)
- Modify: `src/nhf_spatial_targets/aggregate/margulis_wus_sr.py` (docstring)
- Test: `tests/test_aggregate_driver.py`, `tests/test_aggregate_margulis_wus_sr.py`

- [ ] **Step 3.1: Update tests first.**
  - In `tests/test_aggregate_driver.py`: delete `_project_with_token` (line 1906) and the five tests `test_skip_for_fabric_scope_returns_false_when_scope_is_none`, `test_skip_for_fabric_scope_returns_false_when_token_in_scope`, `test_skip_for_fabric_scope_returns_true_when_no_token_set`, `test_skip_for_fabric_scope_validates_scope_via_catalog`, `test_aggregate_source_skips_before_reaching_raw_dir` (lines 1924-2026).
  - In `tests/test_aggregate_margulis_wus_sr.py`: delete `test_catalog_declares_fabric_scope_oregon` (line 64) and replace `test_aggregate_margulis_skips_non_or_fabric` (line 75) with (reuse its existing project-skeleton body, dropping `"token": None` from the config dict):

```python
def test_aggregate_margulis_missing_raw_raises(tmp_path):
    """#309 removed the fabric_scope token gate: margulis behaves like every
    other source — aggregating without fetched raw data raises
    FileNotFoundError pointing at the fetch command, instead of silently
    skipping based on a catalog token."""
    import json

    import yaml

    datastore = tmp_path / "datastore"
    datastore.mkdir()
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": str(tmp_path / "fake.gpkg"), "id_col": "hru_id"},
                "datastore": str(datastore),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))

    with pytest.raises(FileNotFoundError, match="fetch margulis-wus-sr"):
        aggregate_margulis_wus_sr(
            fabric_path=tmp_path / "fake.gpkg",
            id_col="hru_id",
            workdir=tmp_path,
        )
```

  (Match the existing call signature in that file — keep whatever kwargs the deleted test passed.)

- [ ] **Step 3.2: Remove the gate.** In `aggregate/_driver.py`:
  - Delete `_skip_for_fabric_scope` (whole function) and the `if _skip_for_fabric_scope(...)` block in `aggregate_source` (lines 1109-1110).
  - In `aggregate_source`'s docstring replace the paragraph `Sources whose catalog ``fabric_scope`` excludes ... through gdptools.` with:
    ```
    Spatial coverage is geometry-driven (#309): batches whose bbox does not
    overlap the source grid are skipped and their HRUs emitted as NaN rows;
    a source with zero fabric overlap is skipped entirely with an INFO log
    (see :func:`_covered_batch_ids`).
    ```
  - If `from nhf_spatial_targets import catalog` (or similar) becomes unused in `_driver.py`, remove the import (`pixi run -e dev lint` will flag it).
- [ ] **Step 3.3:** In `aggregate/margulis_wus_sr.py`, replace the final docstring paragraph (`The catalog declares ``fabric_scope.fabrics: [or]`` ... see ``targets/swe.py``.`) with:

```
The source covers only the Western US. The aggregation driver's
geometry-driven coverage guard (#309) skips fabric batches outside the
grid and emits honest NaN rows for them, so on a CONUS fabric the
aggregated NC carries data only at WUS HRUs; the SWE target's NaN-aware
combine uses it wherever it is finite.
```

- [ ] **Step 3.4:** Run: `pixi run -e dev python -m pytest tests/test_aggregate_driver.py tests/test_aggregate_margulis_wus_sr.py -v` — Expected: all PASS.
- [ ] **Step 3.5: Commit**

```bash
git add tests/test_aggregate_driver.py tests/test_aggregate_margulis_wus_sr.py src/nhf_spatial_targets/aggregate/_driver.py src/nhf_spatial_targets/aggregate/margulis_wus_sr.py
pixi run git commit -m "refactor(#309): drop _skip_for_fabric_scope token gate from agg driver"
```

---

### Task 4: De-tokenize the SWE target

**Files:**
- Modify: `src/nhf_spatial_targets/targets/swe.py`, `src/nhf_spatial_targets/targets/_driver.py:402`
- Test: `tests/test_targets_swe.py`, `tests/test_targets_sca.py`

- [ ] **Step 4.1: Update tests first** (`tests/test_targets_swe.py`):
  1. In `_make_swe_project`: delete the `fabric_token: str | None = None` parameter and the `"token": fabric_token,` line (103).
  2. Delete every `fabric_token=...` kwarg at all call sites (`grep -n "fabric_token" tests/test_targets_swe.py` must end empty).
  3. Delete tests: `test_fabric_scope_filter_keeps_scoped_source_when_token_matches`, `test_fabric_scope_filter_drops_scoped_source_when_token_mismatches`, `test_build_no_token_drops_margulis`, `test_build_invalid_fabric_token_raises`, plus the `# Fabric scope filter (logic-level)` section header.
  4. In `test_build_oregon_includes_margulis_in_source_attr`: delete `assert ds.attrs["fabric_token"] == "or"`.
  5. Docstring touch-ups where they mention fabric_token/fabric_scope (`test_build_oregon_without_margulis_succeeds_with_three_sources`, `test_build_ua_swe_participates_in_envelope_and_count` — for the latter change "Non-OR fabric (token unset) so Margulis is excluded; sources are" → "Margulis is omitted from the requested sources; the rest are").
  6. Add the partial-coverage end-to-end test:

```python
def test_build_partial_coverage_source_contributes_only_where_finite(tmp_path: Path):
    """#309: a partial-coverage source (NaN rows at uncovered HRUs, as the
    aggregation driver now emits) joins the bound only where finite; the
    bound falls back to the remaining sources elsewhere and n_sources
    drops by one at the uncovered HRU."""
    from nhf_spatial_targets.targets.swe import build
    from nhf_spatial_targets.workspace import load

    workdir = _make_swe_project(
        tmp_path, period="2003-12-15/2003-12-15", nn_fill=False
    )
    nc = (
        workdir
        / "data"
        / "aggregated"
        / "margulis_wus_sr"
        / "margulis_wus_sr_2003_agg.nc"
    )
    with xr.open_dataset(nc) as ds:
        patched = ds.load()
    patched["SWE"].values[:, 0] = np.nan  # HRU 0 "outside the source grid"
    nc.unlink()
    patched.to_netcdf(nc)

    project = load(workdir)
    build(project)
    with xr.open_dataset(project.targets_dir() / "swe_targets.nc") as out:
        n0 = out["n_sources"].isel(nhm_id=0).values
        nrest = out["n_sources"].isel(nhm_id=slice(1, None)).values
        assert (n0 == 3).all()
        assert (nrest == 4).all()
        # margulis (200 mm) sets the upper bound only where it has data;
        # at HRU 0 the bound falls back to era5 (100 mm).
        np.testing.assert_allclose(
            out["upper_bound"].isel(nhm_id=0).values, 100.0 / 25.4, rtol=1e-5
        )
        np.testing.assert_allclose(
            out["upper_bound"].isel(nhm_id=slice(1, None)).values,
            200.0 / 25.4,
            rtol=1e-5,
        )
```

  7. `tests/test_targets_sca.py`: delete the `"token": "or"` (or `fabric_token`) line from its project-config fixture.

- [ ] **Step 4.2:** Run: `pixi run -e dev python -m pytest tests/test_targets_swe.py tests/test_targets_sca.py -v` — Expected: FAIL (swe.py still validates tokens / filters / emits `fabric_token` attr).

- [ ] **Step 4.3: Implement `targets/swe.py`:**
  1. Module docstring: line 10 `(m water-eq, daily — Oregon fabric only)` → `(m water-eq, daily — Western US coverage)`. Replace the whole `**Fabric scope enforcement.** ...` paragraph (lines 38-48) with:
     ```
     **Partial spatial coverage.** Margulis WUS-SR covers only the Western
     US; the aggregation driver (#309) emits its aggregated NCs reindexed to
     the full fabric with honest NaN at uncovered HRUs. Because the combine
     is NaN-aware, the source contributes wherever it is finite and drops
     out elsewhere — no configuration needed. The former catalog
     ``fabric_scope`` / config ``fabric.token`` gate was removed in #309.
     ```
  2. SHIMS margulis description → `"Margulis WUS-SR SWE (m → mm, daily; WUS coverage)"`.
  3. Delete `_filter_sources_by_fabric_scope` and shrink the section header comment to `# Availability filter (single-pass before per-year loop)`.
  4. `_resolve_sources` becomes:
     ```python
     def _resolve_sources(project: Project) -> tuple[list[str], list[str]]:
         """Resolve the effective source list for the current build.

         Returns ``(effective_sources, requested_sources)``. Cached by the
         per-year loader's first call so the filter loop doesn't re-run
         every year (the inputs are config-derived and constant across years).
         """
         swe_cfg = project.target("snow_water_equivalent")
         requested = list(swe_cfg["sources"])
         shims = shims_by_config_label(SHIMS)

         validate_source_units(SHIMS, requested)

         sources = _filter_sources_by_availability(project, requested, shims)
         if not sources:
             raise ValueError(
                 f"snow_water_equivalent.sources={requested!r} resolved to "
                 f"zero sources after dropping unaggregated sources. Run "
                 f"'pixi run nhf-targets agg <source> --project-dir "
                 f"{project.workdir}' for at least one requested source "
                 f"before building the SWE target."
             )
         return sources, requested
     ```
     (Deletes: `fabric_cfg`/`fabric_token` reads, the `FABRIC_SCOPE_TOKENS` validation raise, the fabric_scope filter + its zero-sources raise.)
  5. `_load_year`: `sources, requested_sources = _resolve_sources(project)`; drop `"fabric_token": fabric_token or "",` from `extra_attrs`.
  6. `build`: `sources, requested = _resolve_sources(project)`; log message →
     ```python
         logger.info(
             "Building SWE target: %d sources (%s) [requested %d (%s)], "
             "period %s, fabric=%s",
             len(sources),
             ",".join(sources),
             len(requested),
             ",".join(requested),
             swe_cfg["period"],
             project.config["fabric"]["path"],
         )
     ```
     and in its docstring change `Source filtering (fabric_scope + availability) runs inside the loader` → `Source filtering (availability) runs inside the loader`.
  7. Remove the now-unused `from nhf_spatial_targets import catalog` import if nothing else in swe.py uses it (check: `_filter_sources_by_fabric_scope` was the only `catalog.` user — `grep -n "catalog\." src/nhf_spatial_targets/targets/swe.py`).
- [ ] **Step 4.4:** `targets/_driver.py` line ~402 comment: `and SWE's fabric_token / source list` → `and SWE's source list`.
- [ ] **Step 4.5:** Run: `pixi run -e dev python -m pytest tests/test_targets_swe.py tests/test_targets_sca.py -v` — Expected: all PASS.
- [ ] **Step 4.6: Commit**

```bash
git add tests/test_targets_swe.py tests/test_targets_sca.py src/nhf_spatial_targets/targets/swe.py src/nhf_spatial_targets/targets/_driver.py
pixi run git commit -m "refactor(#309): drop fabric.token filtering from SWE target — rely on NaN-aware combine"
```

---

### Task 5: Retire the config knob (defaults / init template / upgrade registry)

**Files:**
- Modify: `src/nhf_spatial_targets/defaults.py:33-37`, `src/nhf_spatial_targets/init_run.py:29-34,157`, `src/nhf_spatial_targets/upgrade_config.py:4,74-91`, `src/nhf_spatial_targets/cli/maintenance.py:212`
- Test: `tests/test_init_run.py`, `tests/test_upgrade_config.py`

- [ ] **Step 5.1: Update tests first.**
  - `tests/test_init_run.py`: delete `test_init_config_template_carries_fabric_token_stub` (line 105). In the valid-YAML test keep `assert "token" not in cfg["fabric"]` (still true and now trivially so) — fine to leave.
  - `tests/test_upgrade_config.py`: in `test_check_drift_reports_all_features_when_none_present` remove `"fabric.token",` from the expected set; delete `test_check_drift_detects_nested_fabric_token_commented`.
- [ ] **Step 5.2:** Run: `pixi run -e dev python -m pytest tests/test_init_run.py tests/test_upgrade_config.py -v` — Expected: `test_check_drift_reports_all_features_when_none_present` PASSES already (subset assert) but template-stub deletions make others fail only AFTER source edits — i.e. currently all PASS. Proceed to source edits (this is removal work; the "failing test" phase doesn't apply to deletions).
- [ ] **Step 5.3: Source edits.**
  - `defaults.py`: delete lines 33-37 (the comment + `"token": None,`).
  - `init_run.py`: delete template lines 29-34 (`# Optional fabric-scope token. ...` through `# token: or`); line 157 → `      - margulis_wus_sr   # Western US coverage; NaN elsewhere (#309)`.
  - `upgrade_config.py`: delete the entire `OptionalConfigFeature(name="fabric.token", ...)` entry (lines 75-91); module docstring line 4: `(e.g. ``fabric.token``, ``representative_points``)` → `(e.g. ``representative_points``)`.
  - `cli/maintenance.py` line 212 docstring: `(e.g. fabric.token, representative_points)` → `(e.g. representative_points)`.
- [ ] **Step 5.4:** Run: `pixi run -e dev python -m pytest tests/test_init_run.py tests/test_upgrade_config.py tests/test_defaults.py -v` (if `tests/test_defaults.py` exists; otherwise drop it) — Expected: PASS.
- [ ] **Step 5.5: Commit**

```bash
git add tests/test_init_run.py tests/test_upgrade_config.py src/nhf_spatial_targets/defaults.py src/nhf_spatial_targets/init_run.py src/nhf_spatial_targets/upgrade_config.py src/nhf_spatial_targets/cli/maintenance.py
pixi run git commit -m "refactor(#309): retire fabric.token config knob (defaults, init template, upgrade registry)"
```

---

### Task 6: Remove fabric_scope from the catalog layer

**Files:**
- Modify: `src/nhf_spatial_targets/catalog.py` (− lines 163-167, 189-234), `catalog/sources.yml`, `catalog/variables.yml`
- Test: `tests/test_catalog.py`
- Fixture: `tests/fixtures/release/expected_fabric_gfv2_readme.md`

- [ ] **Step 6.1: Update tests first** (`tests/test_catalog.py`):
  - Delete: `test_margulis_wus_sr_fabric_scope_oregon_only`, `test_fabric_scope_field_only_on_scoped_sources`, `test_every_fabric_scope_block_validates`, `test_validate_fabric_scope_rejects_unknown_token`, `test_validate_fabric_scope_rejects_non_list_fabrics`, `test_validate_fabric_scope_rejects_empty_fabrics`, `test_validate_fabric_scope_none_is_ok` (lines 259-315).
  - Add (defensive successor to the only-on-scoped-sources test):
    ```python
    def test_no_source_declares_fabric_scope():
        """fabric_scope was removed in #309 — partial fabric coverage is now
        geometry-driven at agg time. A reappearing block means someone
        resurrected the dead mechanism."""
        assert not {key for key, src in sources().items() if "fabric_scope" in src}
    ```
  - Rename `test_release_block_margulis_publishable_despite_fabric_scope` → `test_release_block_margulis_publishable`, docstring → `"""Margulis WUS-SR is publishable (partial fabric coverage is irrelevant to publication)."""`.
  - Line ~255 comment `# N7 — ... Fabric scoping lives in fabric_scope.` → `# N7 — spatial_extent is a pure source-extent string (partial fabric coverage is geometry-driven at agg time, #309).`
- [ ] **Step 6.2:** Run: `pixi run -e dev python -m pytest tests/test_catalog.py -v` — Expected: `test_no_source_declares_fabric_scope` FAILS (block still in sources.yml); deleted tests gone.
- [ ] **Step 6.3: Edit `catalog/sources.yml`** (margulis_wus_sr entry):
  - Description sentence (lines 851-854): `... Used as a SWE calibration source for the Oregon fabric only; for other fabrics the SWE target builder excludes this source (see fabric_scope).` → `... Used as a SWE calibration source; it contributes to the SWE bound at the HRUs its Western US domain covers and is NaN elsewhere (geometry-driven partial coverage, #309).`
  - Delete the whole `fabric_scope:` block (lines 877-889).
  - Release block comment + notes (lines 905-914) → :
    ```yaml
    release:
      # Spatial coverage is Western-US-only, but the consolidated NSIDC
      # source NCs are public-domain redistributable. The consolidated-source
      # child item in the ScienceBase release carries the data.
      publishable: true
      notes: >
        Western US spatial coverage; on fabrics extending beyond the WUS the
        aggregated/target outputs carry data only at covered HRUs (#309). The
        consolidated source data is public-domain and redistributable as part
        of the umbrella release for any consumer.
    ```
- [ ] **Step 6.4: Edit `catalog/variables.yml`** (`snow_water_equivalent`):
  - `description` →
    ```yaml
    description: >
      Daily basin snow water equivalent. Range is min/max across five
      independent SWE sources per HRU and time step. Margulis Western US
      Snow Reanalysis covers only the Western US; outside its domain the
      NaN-aware bound falls back to the remaining four sources (daymet,
      snodas, era5_land, ua_swe). UA SWE (NSIDC-0719) contributes calendar
      years 1982-2022 (re-windowed at consolidate from water years
      1982-2023), extending the bound well before the SNODAS 2003 start.
      Absolute values used (not normalized).
    ```
  - Sources list comment: `- margulis_wus_sr      # Oregon fabric only` → `- margulis_wus_sr      # Western US coverage; NaN elsewhere`.
  - `range_notes` tail: replace `Margulis WUS-SR is fabric-scoped to Oregon (see catalog/sources.yml fabric_scope); for non-OR fabrics it is excluded by the target builder so the bound reduces to a 4-source min/max (daymet, snodas, era5_sd, ua_swe).` with `Margulis WUS-SR covers only the Western US; at HRUs outside its domain it contributes NaN and the bound reduces to the remaining sources (the former fabric_scope gate was removed in #309).`
- [ ] **Step 6.5: Update the release README fixture.** In `tests/fixtures/release/expected_fabric_gfv2_readme.md` line 25, replace the `snow_water_equivalent` description cell with the new `variables.yml` description text **verbatim, newlines collapsed to single spaces** (the README builder renders the description as-is). Then run the README/release tests to confirm: `pixi run -e dev python -m pytest tests/ -k "readme" -v`. If the rendered text differs (e.g. trailing-space folding), regenerate the cell from the test's actual-vs-expected diff — the catalog text is authoritative.
- [ ] **Step 6.6: Remove the validator.** In `src/nhf_spatial_targets/catalog.py` delete the `FABRIC_SCOPE_TOKENS` constant + comment (lines 163-167) and `validate_fabric_scope` (lines 189-234). `grep -rn "FABRIC_SCOPE\|validate_fabric_scope" src/ tests/` must come back empty.
- [ ] **Step 6.7:** Run: `pixi run -e dev python -m pytest tests/test_catalog.py tests/ -k "catalog or readme or release" -v` — Expected: PASS.
- [ ] **Step 6.8: Dispatch the catalog-reviewer agent** on the `catalog/sources.yml` + `catalog/variables.yml` diff (project rule for catalog edits); address findings.
- [ ] **Step 6.9: Commit**

```bash
git add tests/test_catalog.py tests/fixtures/release/expected_fabric_gfv2_readme.md src/nhf_spatial_targets/catalog.py catalog/sources.yml catalog/variables.yml
pixi run git commit -m "refactor(#309): remove fabric_scope block, tokens, and validator from catalog layer"
```

---

### Task 7: Margulis fetch cleanup

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/margulis_wus_sr.py`, `src/nhf_spatial_targets/cli/fetch.py:868-873`
- Test: `tests/test_fetch_margulis_wus_sr.py`

- [ ] **Step 7.1: Update test first.** Replace `test_manifest_records_fabric_scope` (line 258) with:

```python
def test_manifest_records_search_bbox_and_variables(tmp_path, monkeypatch):
    """The manifest entry carries the fabric-buffered search bbox and the
    variable list (fabric_scope recording was removed in #309)."""
    workdir = _make_project(tmp_path)
    _stub_earthaccess(monkeypatch, granules_per_year=1)
    fetch_margulis_wus_sr(workdir=workdir, period="2000/2000")
    manifest = json.loads((workdir / "manifest.json").read_text())
    entry = manifest["sources"]["margulis_wus_sr"]
    assert "fabric_scope" not in entry
    assert entry["variables"] == ["SWE"]
    # Manifest field is `search_bbox` (the fabric-buffered search bbox),
    # distinct from SNODAS's fixed `bbox`.
    assert entry["search_bbox"] == [-124.2, 41.9, -115.8, 46.6]
```

- [ ] **Step 7.2:** Run: `pixi run -e dev python -m pytest tests/test_fetch_margulis_wus_sr.py -v` — Expected: the new test FAILS (`fabric_scope` still recorded).
- [ ] **Step 7.3: Implement `fetch/margulis_wus_sr.py`:**
  1. Module docstring lines 4-9: replace `This source is **fabric-scoped to Oregon only** ... can honour it.` with `The source covers the Western US; the CMR search bbox is the project fabric's buffered bbox, so a project fetches only what its fabric can use, while raw downloads stay in the shared datastore and remain usable by any project pointing at the same store.`
  2. `fetch_margulis_wus_sr` docstring: drop the `records the fabric scope from the catalog plus` phrasing (line 448-450 → `The flock-protected manifest entry records per-calendar-year ``daily_path`` / ``consolidated_utc`` / ``source_water_years`` for completed consolidations.`) and delete the `The source is fabric-scoped to Oregon via the catalog's ... Western US domain.` paragraph (452-456).
  3. Delete `fabric_scope = meta.get("fabric_scope", {})` (line 488), the operator-hint warning block including its comment and the `scope_fabrics` / `fabric_id` reads (lines 499-517).
  4. `_update_manifest` call (line 644) → `_update_manifest(workdir, period, meta, year_records, search_bbox)`; return dict: delete `"fabric_scope": fabric_scope,` (line 654).
  5. `_update_manifest` definition: drop the `fabric_scope: dict,` parameter (line 729); delete `"fabric_scope": fabric_scope,` from the entry update (line 767) and from `params` (line 798).
- [ ] **Step 7.4:** `cli/fetch.py` margulis docstring (lines 868-873): `Fabric-scoped to Oregon only (catalog `fabric_scope`); the scope is recorded in manifest.json but not enforced at fetch time. Fetch-only: ...` → `The CMR search bbox is the project fabric's buffered bbox (Western US coverage upstream). Fetch-only: ...`
- [ ] **Step 7.5:** Run: `pixi run -e dev python -m pytest tests/test_fetch_margulis_wus_sr.py -v` — Expected: PASS. Then dispatch the **provenance-reviewer** agent on the `fetch/margulis_wus_sr.py` + `aggregate/_driver.py` diff (project rule for fetch/pipeline provenance changes); address findings.
- [ ] **Step 7.6: Commit**

```bash
git add tests/test_fetch_margulis_wus_sr.py src/nhf_spatial_targets/fetch/margulis_wus_sr.py src/nhf_spatial_targets/cli/fetch.py
pixi run git commit -m "refactor(#309): drop fabric_scope recording from margulis fetch"
```

---

### Task 8: Docs sweep + final gate

**Files:** `CLAUDE.md`, `README.md`, `CONTRIBUTING.md`, `docs/index.md`, `docs/sources/margulis_wus_sr.md`, `docs/references/calibration-target-recipes.md`, `docs/references/target-period-coverage.md`, `docs/maintenance.md`

- [ ] **Step 8.1: CLAUDE.md.** Replace the Data & Catalog Conventions bullet `Fabric-restricted sources (e.g. Margulis WUS-SR for Oregon) carry an optional fabric_scope ... not at fetch time` with:

```
- Sources need not cover the whole fabric. The aggregation driver classifies
  each spatial batch against the source grid bbox
  (`aggregate/_driver.py:_covered_batch_ids`), skips non-overlapping batches,
  reindexes every per-year NC to the full fabric so uncovered HRUs are honest
  NaN, and logs a per-source HRU-coverage diagnostic. A source with zero
  fabric overlap is skipped with an INFO log. Target-stage multi-source
  combines are NaN-aware, so a partial source (e.g. Margulis WUS-SR, Western
  US only) contributes exactly where it has data. The former `fabric_scope` /
  `fabric.token` token gate was removed in #309; raw downloads remain
  fabric-independent and shared via the datastore
```

- [ ] **Step 8.2: README.md** (lines 18-22, 40, 427, 625, 662), **docs/index.md** (line 18), **CONTRIBUTING.md** (lines 51, 80, 223): rewrite each fabric_scope passage to the new model. Consistent phrasing: *"Margulis WUS-SR covers only the Western US; it contributes NaN-aware to the HRUs it covers and drops out elsewhere (#309 — the former fabric_scope/fabric.token gate is gone)."* CONTRIBUTING 80 (the "add a fabric_scope: block" instruction) and 223 (the "extend FABRIC_SCOPE_TOKENS" instruction) are deleted outright — partial coverage needs no catalog markup; replace 223's list item with "nothing — partial fabric coverage is geometry-driven at agg time (#309)" or drop the item if the list reads fine without it.
- [ ] **Step 8.3: docs/sources/margulis_wus_sr.md** (passages at lines 9-11, 22, 48, 66, 81): rewrite to the coverage model — fetch bbox = project fabric bbox; aggregation skips uncovered batches + NaN-fills; SWE combine NaN-aware; explicitly note the gfv2 consequence (margulis now contributes at WUS HRUs on a CONUS fabric once fetched+aggregated there) and that `agg margulis-wus-sr` / `agg all` now require fetched raw data (FileNotFoundError otherwise).
- [ ] **Step 8.4: docs/references/calibration-target-recipes.md** (lines 454, 490, 498-499, 575) and **target-period-coverage.md** (lines 45, 145, 154): replace fabric_scope/token mechanics with the coverage-guard description; delete the `fabric.token: oregon` ValueError recipe (the knob no longer exists).
- [ ] **Step 8.5: docs/maintenance.md**: add a short "Retired config keys" note: `fabric.token` was removed in #309; existing configs carrying it get a `[warning] unknown config key: fabric.token` from `validate` — delete the line; no behavior depends on it.
- [ ] **Step 8.6: Doc-sync scan:** `grep -rn "fabric_scope\|fabric\.token\|FABRIC_SCOPE" --include="*.py" --include="*.yml" --include="*.md" . | grep -v docs/superpowers | grep -v docs/presentations | grep -v .pixi` — Expected: no hits outside historical plan/spec documents (`docs/superpowers/**` and `docs/presentations/**` are dated artifacts, intentionally untouched).
- [ ] **Step 8.7: Full gate:** `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test` — Expected: clean format, zero lint errors, full suite PASS. Also `pixi run -e dev mkdocs build --strict` if the docs build task exists (CI runs docs-build; check `pixi task list` for the exact task name, e.g. `pixi run docs-build`).
- [ ] **Step 8.8: Commit**

```bash
git add CLAUDE.md README.md CONTRIBUTING.md docs/index.md docs/sources/margulis_wus_sr.md docs/references/calibration-target-recipes.md docs/references/target-period-coverage.md docs/maintenance.md
pixi run git commit -m "docs(#309): document geometry-driven partial coverage; retire fabric_scope/fabric.token prose"
```

- [ ] **Step 8.9: PR.** Rebase on origin/main, push, open PR titled `refactor(#309): remove fabric_scope/fabric.token gating — geometry-driven NaN-aware partial coverage` with body covering: the coverage guard (wrinkle 1), the seam/coverage diagnostic (wrinkle 2), fetch bbox already project-driven (wrinkle 3 — no change), the gfv2 SWE results-change consequence, the `agg all` behavior change for unfetched margulis, the leftover-`fabric.token` validate warning, and the docs-sync note (presentations intentionally untouched as dated artifacts). `Closes #309`.

---

## Self-Review (completed at plan time)

- **Spec coverage:** wrinkle 1 → Tasks 1-3 (guard + skip + NaN); wrinkle 2 → Task 2 diagnostic log + Task 8 docs; wrinkle 3 → resolved no-change (fetch bbox is fabric-driven), documented Task 8.3; every touchpoint file in the issue is covered by a task (cli/run.py checked — no references, no change needed).
- **Placeholder scan:** none; all steps carry code or exact text.
- **Type consistency:** `_covered_batch_ids(source_file, adapter, fabric_batched) -> set[int]` used identically in Tasks 1-2; `_resolve_sources` 2-tuple unpacked consistently in Task 4 steps 4.3.5-4.3.6.
