# ua_swe PR-A2 — Re-window Consolidator to Calendar Years — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ua_swe`'s consolidator emit per-**calendar-year** NetCDFs (`ua_swe_daily_<year>.nc`) instead of per-water-year ones, so the shared calendar-year aggregation driver stops crashing on the overlapping-year collision.

**Architecture:** Port the Margulis pattern (`fetch/margulis_wus_sr.py:consolidate_calendar_year_margulis_wus_sr`). Calendar year *X* is assembled from the Jan–Sep portion of WY *X* joined with the Oct–Dec portion of WY *X+1*. Extract the existing per-WY decode+reproject body into a reusable helper, then build calendar-year consolidation on top. Re-shard the fetch worker loop by calendar year (each worker downloads the two adjacent WY raws it needs, idempotently) so multi-worker SLURM runs have both raws on disk before consolidating. Drop partial edge years (1981, 2023) via a `FileNotFoundError` boundary rule.

**Tech Stack:** xarray, rioxarray (EPSG:4269→5070 reproject), pandas time decode, pytest. All commands via `pixi`.

**Spec:** `docs/superpowers/specs/2026-06-09-ua-swe-wiring-design.md` (PR-A2 section).

**Branch:** `feature/237-ua-swe-pr-a2-consolidate-rewindow` (off `main`). Per project norms, develop in a worktree; run `pixi install -e dev` once after creating it.

---

## File Structure

- **Modify** `src/nhf_spatial_targets/fetch/ua_swe.py`
  - Extract `_reproject_wy_to_dataset(wy, raw_path) -> xr.Dataset` from the body of `consolidate_water_year_ua_swe` (decode + sentinel-check + per-day reproject + rename → reprojected `swe`/`snow_depth` dataset, **pre-CF**).
  - Add `_calendar_year_slice(ds, calendar_year) -> xr.Dataset`.
  - Add `consolidate_calendar_year_ua_swe(calendar_year, raw_dir, daily_dir) -> Path`.
  - Replace `consolidate_water_year_ua_swe` (callers move to the CY function).
  - Add `_CONSOLIDATED_FILENAME_TEMPLATE = "ua_swe_daily_{year}.nc"` (replaces the `WY{wy}` template).
  - Re-shard `fetch_ua_swe`: assign **calendar years** to workers; per CY download WY *X* and *X+1* raws, then consolidate.
  - Rewrite `_update_manifest` records to be keyed by `calendar_year` (not `water_year`).
  - Rename `_assign_worker_water_years` → `_assign_worker_calendar_years` (same round-robin body).
- **Modify** `tests/test_ua_swe.py` — add calendar-year consolidation tests; update WY-assuming tests.
- **Modify** `catalog/sources.yml` — `ua_swe.access.notes` filename description.
- **Modify** `notebooks/consolidated/inspect_consolidated_swe.ipynb` and `inspect_consolidated_snow_covered_area.ipynb` — fix hardcoded `ua_swe_daily_WY{...}.nc` paths.
- **Create** `docs/sources/ua_swe.md`; **Modify** `docs/sources/index.md`.

---

## Task 1: Extract `_reproject_wy_to_dataset` helper (pure refactor)

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/ua_swe.py`
- Test: `tests/test_ua_swe.py`

- [ ] **Step 1: Write a characterization test for the helper**

Add to `tests/test_ua_swe.py` (reuse the existing synthetic-raw-NC fixture; if the file already builds a fake raw WY NC for `consolidate_water_year_ua_swe`, factor that into a `_make_raw_wy_nc(tmp_path, wy, n_days=...)` helper and call it here):

```python
def test_reproject_wy_to_dataset_shape_and_vars(tmp_path):
    from nhf_spatial_targets.fetch.ua_swe import _reproject_wy_to_dataset

    raw = _make_raw_wy_nc(tmp_path, wy=2000, n_days=366)  # WY2000 = Oct1999-Sep2000
    ds = _reproject_wy_to_dataset(2000, raw)

    assert set(ds.data_vars) == {"swe", "snow_depth"}
    assert ds["swe"].dims == ("time", "y", "x")
    # EPSG:5070 projected coords are metres, monotonic.
    assert "x" in ds.coords and "y" in ds.coords
    assert ds.sizes["time"] == 366
    # First timestamp is Oct 1 of the prior calendar year.
    assert pd.Timestamp(ds["time"].values[0]) == pd.Timestamp("1999-10-01")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run -e dev pytest tests/test_ua_swe.py::test_reproject_wy_to_dataset_shape_and_vars -v`
Expected: FAIL with `ImportError`/`AttributeError: _reproject_wy_to_dataset`.

- [ ] **Step 3: Extract the helper**

In `fetch/ua_swe.py`, cut the body of `consolidate_water_year_ua_swe` from the `ds_raw = xr.open_dataset(...)` line through the construction of `ds_reproj` (the `try/finally` block that decodes time, runs the sentinel `< -1.0` guard, does the per-day reproject, and builds `ds_reproj`) into a new module-level function. It returns the reprojected, renamed, **pre-CF** dataset:

```python
def _reproject_wy_to_dataset(wy: int, raw_path: Path) -> xr.Dataset:
    """Decode + pre-project one raw WY NC to an EPSG:5070 (time, y, x) dataset.

    Returns a dataset with ``swe`` / ``snow_depth`` (float32, native units),
    a decoded daily ``time`` axis (Oct 1 of WY-1 .. Sep 30 of WY), and
    projected ``y`` / ``x`` metre coords. CF metadata is NOT applied here —
    the caller applies it once after stitching (see
    :func:`consolidate_calendar_year_ua_swe`).
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"ua_swe raw NC not found: {raw_path}")
    ds_raw = xr.open_dataset(raw_path, decode_times=False)
    try:
        # ... existing time-decode, sentinel-check, per-day reproject body ...
        # (verbatim from the old consolidate_water_year_ua_swe), ending with:
        ds_reproj = xr.Dataset(
            {
                dst_name: xr.concat(days, dim="time")
                for dst_name, days in reprojected_per_var.items()
            }
        )
    finally:
        ds_raw.close()
    return ds_reproj
```

Leave `consolidate_water_year_ua_swe` temporarily calling the helper (it is removed in Task 3) so the suite stays green:

```python
def consolidate_water_year_ua_swe(wy: int, raw_path: Path, daily_dir: Path) -> Path:
    daily_dir.mkdir(parents=True, exist_ok=True)
    out_path = daily_dir / f"ua_swe_daily_WY{wy}.nc"
    if out_path.exists() and out_path.stat().st_mtime >= raw_path.stat().st_mtime:
        return out_path
    ds_reproj = _reproject_wy_to_dataset(wy, raw_path)
    ds_out = apply_cf_metadata(ds_reproj, _SOURCE_KEY, time_step="daily",
                               coord_type="projected")
    _atomic_write_dataset(ds_out, out_path, encoding=_build_consolidated_encoding(ds_out))
    return out_path
```

- [ ] **Step 4: Run the helper test + the existing WY-consolidate tests**

Run: `pixi run -e dev pytest tests/test_ua_swe.py -k "reproject or consolidate" -v`
Expected: PASS (helper test + any existing `consolidate_water_year` tests).

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/ua_swe.py tests/test_ua_swe.py
pixi run git commit -m "refactor(#237): extract _reproject_wy_to_dataset from ua_swe consolidator"
```

---

## Task 2: `_calendar_year_slice` + `consolidate_calendar_year_ua_swe`

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/ua_swe.py`
- Test: `tests/test_ua_swe.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_consolidate_calendar_year_spans_jan_to_dec(tmp_path):
    from nhf_spatial_targets.fetch.ua_swe import consolidate_calendar_year_ua_swe

    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    daily = tmp_path / "daily"
    _make_raw_wy_nc(raw_dir, wy=2000)   # Oct1999-Sep2000  -> file 4km_SWE_Depth_WY2000_v01.nc
    _make_raw_wy_nc(raw_dir, wy=2001)   # Oct2000-Sep2001  -> file 4km_SWE_Depth_WY2001_v01.nc

    out = consolidate_calendar_year_ua_swe(2000, raw_dir, daily)

    assert out.name == "ua_swe_daily_2000.nc"
    ds = xr.open_dataset(out)
    t = pd.DatetimeIndex(ds["time"].values)
    assert t.min() == pd.Timestamp("2000-01-01")
    assert t.max() == pd.Timestamp("2000-12-31")
    # No duplicate / gap across the Sep30 -> Oct1 seam.
    assert t.is_monotonic_increasing and t.is_unique
    assert (t[1:] - t[:-1]).max() == pd.Timedelta(days=1)
    ds.close()


def test_consolidate_calendar_year_boundary_raises(tmp_path):
    from nhf_spatial_targets.fetch.ua_swe import consolidate_calendar_year_ua_swe

    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    daily = tmp_path / "daily"
    _make_raw_wy_nc(raw_dir, wy=2023)   # only WY2023; WY2024 (for Oct-Dec 2023) absent
    with pytest.raises(FileNotFoundError):
        consolidate_calendar_year_ua_swe(2023, raw_dir, daily)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run -e dev pytest tests/test_ua_swe.py -k "calendar_year" -v`
Expected: FAIL with `AttributeError: consolidate_calendar_year_ua_swe`.

- [ ] **Step 3: Implement the slice helper + CY consolidator**

```python
def _calendar_year_slice(ds: xr.Dataset, calendar_year: int) -> xr.Dataset:
    """Slice a decoded WY dataset to whichever portion overlaps the calendar year."""
    return ds.sel(
        time=slice(
            pd.Timestamp(f"{calendar_year}-01-01"),
            pd.Timestamp(f"{calendar_year}-12-31"),
        )
    )


def consolidate_calendar_year_ua_swe(
    calendar_year: int, raw_dir: Path, daily_dir: Path
) -> Path:
    """Re-window two adjacent WY raws into one CF-1.6 calendar-year NC.

    Calendar year *X* = [Jan 1 – Sep 30 of X] from WY *X*
    (``4km_SWE_Depth_WY{X}_v01.nc``, which runs Oct *X-1* – Sep *X*)
    joined with [Oct 1 – Dec 31 of X] from WY *X+1*. Both raws must be
    present; a missing WY *X+1* (the archive boundary, e.g. CY 2023)
    raises ``FileNotFoundError`` and the caller skips the partial edge.

    Mirrors ``fetch/margulis_wus_sr.py:consolidate_calendar_year_margulis_wus_sr``.
    Output: ``<daily_dir>/ua_swe_daily_<calendar_year>.nc`` (EPSG:5070).
    mtime-idempotent against both contributing raws.
    """
    daily_dir.mkdir(parents=True, exist_ok=True)
    out_path = daily_dir / _CONSOLIDATED_FILENAME_TEMPLATE.format(year=calendar_year)

    raw_x = raw_dir / _WY_FILENAME_TEMPLATE.format(wy=calendar_year)
    raw_x1 = raw_dir / _WY_FILENAME_TEMPLATE.format(wy=calendar_year + 1)
    for p in (raw_x, raw_x1):
        if not p.exists():
            raise FileNotFoundError(
                f"ua_swe: calendar year {calendar_year} needs both "
                f"{raw_x.name} and {raw_x1.name}; missing {p.name}. "
                f"(Archive-boundary calendar years are dropped.)"
            )

    newest_raw = max(raw_x.stat().st_mtime, raw_x1.stat().st_mtime)
    if out_path.exists() and out_path.stat().st_mtime >= newest_raw:
        logger.info(
            "ua_swe: CY %d already consolidated and current (%s); skipping.",
            calendar_year, out_path.name,
        )
        return out_path

    ds_x = _reproject_wy_to_dataset(calendar_year, raw_x)
    ds_x1 = _reproject_wy_to_dataset(calendar_year + 1, raw_x1)
    try:
        jan_sep = _calendar_year_slice(ds_x, calendar_year)    # Jan1 .. Sep30
        oct_dec = _calendar_year_slice(ds_x1, calendar_year)   # Oct1 .. Dec31
        ds_cy = xr.concat([jan_sep, oct_dec], dim="time").sortby("time")
    finally:
        ds_x.close()
        ds_x1.close()

    ds_out = apply_cf_metadata(
        ds_cy, _SOURCE_KEY, time_step="daily", coord_type="projected"
    )
    _atomic_write_dataset(
        ds_out, out_path, encoding=_build_consolidated_encoding(ds_out)
    )
    return out_path
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e dev pytest tests/test_ua_swe.py -k "calendar_year" -v`
Expected: PASS (both new tests).

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/ua_swe.py tests/test_ua_swe.py
pixi run git commit -m "feat(#237): add calendar-year ua_swe consolidator (Margulis-style re-window)"
```

---

## Task 3: Swap the filename template + remove the WY consolidator

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/ua_swe.py`

- [ ] **Step 1: Update the constant**

Replace:

```python
_CONSOLIDATED_FILENAME_TEMPLATE = "ua_swe_daily_WY{wy}.nc"
```

with:

```python
_CONSOLIDATED_FILENAME_TEMPLATE = "ua_swe_daily_{year}.nc"
```

- [ ] **Step 2: Remove `consolidate_water_year_ua_swe`**

Delete the function (its only remaining caller, `fetch_ua_swe`, is rewired in Task 4). Keep `_reproject_wy_to_dataset`, `_build_consolidated_encoding`, `_atomic_write_dataset`.

- [ ] **Step 3: Delete/replace the now-obsolete WY-consolidate tests**

Remove any `tests/test_ua_swe.py` test that calls `consolidate_water_year_ua_swe` or asserts a `ua_swe_daily_WY*.nc` filename (the calendar-year tests from Task 2 supersede them).

- [ ] **Step 4: Run the module's tests**

Run: `pixi run -e dev pytest tests/test_ua_swe.py -v`
Expected: PASS, no reference to the removed function.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/ua_swe.py tests/test_ua_swe.py
pixi run git commit -m "refactor(#237): drop per-WY ua_swe consolidator + WY filename template"
```

---

## Task 4: Re-shard `fetch_ua_swe` by calendar year

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/ua_swe.py`
- Test: `tests/test_ua_swe.py`

**Why:** Consolidating CY *X* needs WY *X* and *X+1* raws on disk. Sharding by WY across SLURM-array workers cannot guarantee both adjacent raws land in one worker. Shard by **calendar year** instead: each worker downloads the (idempotent, shared) two WY raws for each CY it owns, then consolidates. Redundant boundary-WY downloads stat-check as `already_present`.

- [ ] **Step 1: Write the failing orchestration test**

```python
def test_fetch_ua_swe_consolidates_by_calendar_year(tmp_path, monkeypatch):
    # Stub the network: _download_file writes a synthetic raw WY NC and
    # returns "downloaded". Patch _earthaccess_session to a dummy.
    import nhf_spatial_targets.fetch.ua_swe as m

    def fake_download(session, url, out_path, **kw):
        wy = int(re.search(r"WY(\d{4})", url).group(1))
        _write_raw_wy_nc(out_path, wy)   # same body as _make_raw_wy_nc, given a path
        return "downloaded"

    monkeypatch.setattr(m, "_download_file", fake_download)
    monkeypatch.setattr(m, "_earthaccess_session", lambda: object())
    ws = _make_project(tmp_path)         # existing project fixture used elsewhere

    result = m.fetch_ua_swe(ws.workdir, "2000/2001")

    daily = ws.raw_dir("ua_swe") / "daily"
    assert (daily / "ua_swe_daily_2000.nc").exists()
    assert (daily / "ua_swe_daily_2001.nc").exists()
    # Records are keyed by calendar year now.
    assert {r["calendar_year"] for r in result["calendar_years"]} == {2000, 2001}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/test_ua_swe.py::test_fetch_ua_swe_consolidates_by_calendar_year -v`
Expected: FAIL (`KeyError: 'calendar_years'` or missing CY files).

- [ ] **Step 3: Rewire the worker loop**

In `fetch_ua_swe`, after the publisher-window validation:

```python
    # Full calendar years require WY X (Jan-Sep) AND WY X+1 (Oct-Dec).
    # Drop edge calendar years whose WY X+1 is unpublished.
    publishable_cys = [
        cy for cy in requested_cys if (cy + 1) <= publisher_wy_hi and cy >= publisher_wy_lo - 1
    ]
    assigned_cys = _assign_worker_calendar_years(
        publishable_cys, worker_index, n_workers
    )
```

Then replace the per-WY download+consolidate loop with a per-CY loop. For each assigned CY, download WY *X* and *X+1* (idempotent), then consolidate:

```python
    cy_records: list[dict] = []
    for cy in assigned_cys:
        wy_status: dict[int, str] = {}
        for wy in (cy, cy + 1):
            url = _wy_url(archive_url, wy)
            raw_path = raw_dir / _WY_FILENAME_TEMPLATE.format(wy=wy)
            wy_status[wy] = _download_file(session, url, raw_path)
        rec: dict = {
            "calendar_year": cy,
            "water_years": [cy, cy + 1],
            "wy_status": wy_status,
            "downloaded_utc": now_utc,
        }
        if all(s in ("downloaded", "already_present") for s in wy_status.values()):
            try:
                daily_path = consolidate_calendar_year_ua_swe(cy, raw_dir, daily_dir)
                rec["daily_path"] = str(daily_path)
                rec["consolidated_utc"] = datetime.now(timezone.utc).isoformat()
            except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
                logger.warning("ua_swe: CY %d consolidation failed: %s", cy, exc)
                rec["consolidate_error"] = str(exc)
        else:
            logger.warning(
                "ua_swe: CY %d skipped — WY download status %s", cy, wy_status
            )
        cy_records.append(rec)

    _update_manifest(workdir, period, meta, cy_records, archive_url, mask_record)
    return {
        "source_key": _SOURCE_KEY,
        "period": period,
        "archive_url": archive_url,
        "worker_index": worker_index,
        "n_workers": n_workers,
        "calendar_years": cy_records,
    }
```

Rename `_assign_worker_water_years` → `_assign_worker_calendar_years` (body unchanged — round-robin slice `items[worker_index::n_workers]`). Remove the old WY-assignment + early-return block that returned `{"water_years": []}`; the per-CY loop with an empty `assigned_cys` returns naturally with `cy_records == []`.

- [ ] **Step 4: Run the orchestration test + module suite**

Run: `pixi run -e dev pytest tests/test_ua_swe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/ua_swe.py tests/test_ua_swe.py
pixi run git commit -m "feat(#237): shard ua_swe fetch by calendar year; consolidate per-CY"
```

---

## Task 5: Manifest records keyed by calendar year

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/ua_swe.py` (`_update_manifest`)
- Test: `tests/test_ua_swe.py`

- [ ] **Step 1: Write the failing test**

```python
def test_manifest_records_calendar_years(tmp_path, monkeypatch):
    import nhf_spatial_targets.fetch.ua_swe as m
    # (reuse fake_download / project fixture from Task 4)
    ...
    m.fetch_ua_swe(ws.workdir, "2000/2000")
    manifest = json.loads(ws.manifest_path.read_text())
    entry = manifest["sources"]["ua_swe"]
    assert "calendar_years" in entry
    assert entry["calendar_years"][0]["calendar_year"] == 2000
    # consolidate step output is the CY NC.
    steps = [s for s in manifest["steps"] if s["source_key"] == "ua_swe"]
    assert any("ua_swe_daily_2000.nc" in o["path"]
               for s in steps for o in s["outputs"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/test_ua_swe.py::test_manifest_records_calendar_years -v`
Expected: FAIL (`water_years` key, not `calendar_years`).

- [ ] **Step 3: Update `_update_manifest`**

Change the signature param `wy_records` → `cy_records`, and inside `_do_update`:

```python
        existing_by_cy = {
            int(y["calendar_year"]): y for y in entry.get("calendar_years", [])
        }
        for rec in cy_records:
            existing_by_cy[int(rec["calendar_year"])] = rec
        merged_cys = [existing_by_cy[c] for c in sorted(existing_by_cy)]
        ...
        entry.update({
            ...
            "calendar_years": merged_cys,   # replaces "water_years"
            ...
        })
        ...
        step_outputs = [
            output_file_entry(Path(rec["daily_path"]))
            for rec in cy_records
            if rec.get("daily_path") and Path(rec["daily_path"]).exists()
        ]
        manifest["steps"].append(build_step_record(
            kind="consolidate",
            source_key=_SOURCE_KEY,
            outputs=step_outputs,
            params={
                "period": period,
                "calendar_years": [int(r["calendar_year"]) for r in cy_records],
                "archive_url": archive_url,
            },
            command=f"fetch {_SOURCE_KEY.replace('_', '-')}",
        ))
```

Update the trailing log line to `[r["calendar_year"] for r in cy_records]`.

- [ ] **Step 4: Run the test + module suite**

Run: `pixi run -e dev pytest tests/test_ua_swe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/ua_swe.py tests/test_ua_swe.py
pixi run git commit -m "feat(#237): ua_swe manifest records keyed by calendar year"
```

---

## Task 6: Catalog notes — describe calendar-year layout

**Files:**
- Modify: `catalog/sources.yml` (`ua_swe.access.notes`)
- Test: `tests/test_catalog.py` (load smoke-test must stay green)

- [ ] **Step 1: Edit the note**

In the `ua_swe.access.notes` block, change the sentence describing one NetCDF per water year named `ua_swe_daily_WY<YYYY>.nc` to: consolidated to one NetCDF **per calendar year** named `ua_swe_daily_<YYYY>.nc`, assembled from the Jan–Sep portion of WY *X* and the Oct–Dec portion of WY *X+1* (mirrors `margulis_wus_sr`). Note full coverage is CY 1982–2022 (partial edge years dropped). Leave the raw archive description (`4km_SWE_Depth_WY<YYYY>_v01.nc`) unchanged — the raws are still per-WY.

- [ ] **Step 2: Run catalog tests**

Run: `pixi run -e dev pytest tests/test_catalog.py -v`
Expected: PASS (YAML still parses; `ua_swe` still resolves).

- [ ] **Step 3: Commit**

```bash
git add catalog/sources.yml
pixi run git commit -m "docs(#237): ua_swe catalog notes describe calendar-year consolidation"
```

---

## Task 7: Fix the breaking notebook paths

**Files:**
- Modify: `notebooks/consolidated/inspect_consolidated_swe.ipynb`
- Modify: `notebooks/consolidated/inspect_consolidated_snow_covered_area.ipynb` (only if it carries the same hardcode)

- [ ] **Step 1: Find the hardcoded WY paths**

Run: `grep -n "ua_swe_daily_WY" notebooks/consolidated/*.ipynb`
Expected: the SWE notebook registry entry `DATASTORE / "ua_swe" / "daily" / f"ua_swe_daily_WY{TARGET_YEAR}.nc"` plus a markdown cell describing WY filenames.

- [ ] **Step 2: Edit the registry + markdown**

Change the path to `f"ua_swe_daily_{TARGET_YEAR}.nc"`. Update the markdown bullet that says the consolidated layout is per-water-year (`ua_swe_daily_WY<YYYY>.nc`, "for March 2010 the WY2010 file…") to per-calendar-year (`ua_swe_daily_<YYYY>.nc`, "the CY2010 file holds Jan–Dec 2010"). Repeat in the SCA notebook only if `grep` found it there.

- [ ] **Step 3: Sanity check (no execution required)**

Run: `grep -c "ua_swe_daily_WY" notebooks/consolidated/*.ipynb`
Expected: `0` in every file.

- [ ] **Step 4: Commit**

```bash
git add notebooks/consolidated/inspect_consolidated_swe.ipynb
# add the SCA notebook too iff it was edited
pixi run git commit -m "docs(#237): point consolidated SWE notebook at calendar-year ua_swe NCs"
```

---

## Task 8: Add the `docs/sources/ua_swe.md` source page

**Files:**
- Create: `docs/sources/ua_swe.md`
- Modify: `docs/sources/index.md`

- [ ] **Step 1: Read the template**

Run: `sed -n '1,60p' docs/sources/margulis_wus_sr.md` and `grep -n "margulis\|snodas" docs/sources/index.md` to match the existing per-source page structure and the index-row format.

- [ ] **Step 2: Write `docs/sources/ua_swe.md`**

Follow the `margulis_wus_sr.md` section layout. Cover: NSIDC-0719 (UA Daily 4-km Gridded SWE & Snow Depth), DOI `10.5067/0GGPB220EX6A`, CONUS, native NAD83 (EPSG:4269) ~4 km, **pre-projected to EPSG:5070 at consolidate time**, variables `swe` (kg m⁻²) / `snow_depth` (mm), the **calendar-year** consolidated layout (`ua_swe_daily_<year>.nc`, full coverage 1982–2022, assembled from two adjacent WY raws), its role as a 5th SWE source and 2nd SCA source, and the `snow_covered_fraction` depth-derived SCA proxy with its `depth_threshold_mm` knob (note: changing the threshold requires re-running `agg ua-swe`).

- [ ] **Step 3: Add the index row**

Add a `ua_swe` row to `docs/sources/index.md` matching the column layout of the existing `snodas` / `margulis_wus_sr` rows.

- [ ] **Step 4: Link check**

Run: `grep -n "ua_swe\|NSIDC-0719" docs/sources/index.md`
Expected: the new row present and pointing at `ua_swe.md`.

- [ ] **Step 5: Commit**

```bash
git add docs/sources/ua_swe.md docs/sources/index.md
pixi run git commit -m "docs(#237): add ua_swe source page + index row"
```

---

## Task 9: Full targeted verification + PR

**Files:** none (verification + PR).

- [ ] **Step 1: Run the touched-module tests**

Run: `pixi run -e dev pytest tests/test_ua_swe.py tests/test_catalog.py -v`
Expected: PASS.

- [ ] **Step 2: Lint + format**

Run: `pixi run -e dev fmt && pixi run -e dev lint`
Expected: clean (no diffs from fmt; lint passes).

- [ ] **Step 3: Push + open PR (let CI run the full suite — per the caldera test policy, do not run the whole pytest locally)**

```bash
git push -u origin feature/237-ua-swe-pr-a2-consolidate-rewindow
gh pr create --title "feat(#237): re-window ua_swe consolidator to calendar years (PR-A2)" \
  --body "$(cat <<'BODY'
Re-windows ua_swe consolidation from per-water-year to per-calendar-year
NCs (`ua_swe_daily_<year>.nc`), matching the Margulis pattern, so the
shared calendar-year aggregation driver stops crashing on the
overlapping-year collision. Shards the fetch by calendar year. Drops
partial edge years (full coverage 1982–2022). Fixes the breaking WY path
in the consolidated SWE notebook and adds the ua_swe source doc.

Part of #237 (PR-A2). Unblocks PR-B (aggregate).
BODY
)"
```

- [ ] **Step 4: Watch CI**

Run: `gh pr checks --watch`
Expected: green. Address failures before requesting review.

- [ ] **Step 5: Check off PR-A2 in #237** once merged.

---

## Self-Review notes

- **Spec coverage:** PR-A2 section items — new CY consolidator (T2), boundary-drop rule (T2/T4), fetch orchestration (T4), manifest CY records (T5), catalog notes (T6), breaking notebook fix (T7), `docs/sources/ua_swe.md` (T8) — all mapped.
- **Type/name consistency:** `_reproject_wy_to_dataset`, `_calendar_year_slice`, `consolidate_calendar_year_ua_swe`, `_assign_worker_calendar_years`, record key `calendar_year`, manifest key `calendar_years`, filename `ua_swe_daily_{year}.nc` used consistently across tasks.
- **Fixtures:** Tasks assume a `_make_raw_wy_nc(dir_or_path, wy, n_days=...)` synthetic-raw helper and a project fixture (`_make_project`). Task 1 Step 1 factors these from the existing `tests/test_ua_swe.py` setup; if the existing tests use different names, reuse those rather than inventing new ones.
- **Out of scope (later PRs):** aggregate adapter (PR-B), threshold provenance (PR-B2), SWE shim (PR-C), SCA refactor (PR-D).
