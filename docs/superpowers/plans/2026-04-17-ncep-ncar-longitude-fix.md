# NCEP/NCAR Longitude Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `consolidate_ncep_ncar()` to produce a consolidated NetCDF with monotonic -180..180 longitude so gdptools aggregation can slice the coordinate without KeyError.

**Architecture:** Add longitude normalization (wrap 0-360 → -180..180 + sort) inside `consolidate_ncep_ncar()` after `apply_cf_metadata()` and before `_write_netcdf()`. The CF `axis="X"` attribute (set by `apply_cf_metadata`) is used to detect the longitude coordinate name. A guard (`max > 180`) skips normalization for data already in -180..180.

**Tech Stack:** xarray, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-04-17-ncep-ncar-longitude-fix-design.md`

---

### Task 1: Write failing test for longitude normalization

**Files:**
- Modify: `tests/test_consolidate.py` (add new fixture + test after line ~393)

- [ ] **Step 1: Write test fixture and test**

Add a new fixture that creates synthetic NCEP/NCAR files with 0-360 longitude (matching the real T62 Gaussian grid convention), and a test asserting the consolidated output has -180..180 monotonic longitude.

Add after the `test_ncep_cf_metadata` test (~line 393):

```python
@pytest.fixture
def ncep_dir_0_360(tmp_path: Path) -> Path:
    """Create synthetic NCEP/NCAR monthly files with 0-360 longitude."""
    out = tmp_path / "ncep_ncar"
    out.mkdir(parents=True)

    lat = np.arange(-90, 91, 45.0)
    lon = np.arange(0, 360, 60.0)  # 0-360 convention

    for month in range(1, 4):
        time = np.array([f"2010-{month:02d}-15T00:00:00"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {
                "soilw": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )
        fname = f"soilw.0-10cm.gauss.2010-{month:02d}.monthly.nc"
        ds.to_netcdf(out / fname, format="NETCDF3_CLASSIC")

    return out


def test_ncep_longitude_normalization(ncep_dir_0_360):
    """NCEP/NCAR 0-360 longitude is normalized to -180..180 and sorted."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_ncep_ncar

    consolidate_ncep_ncar(source_dir=ncep_dir_0_360, variables=["soilw"])

    ds = xr.open_dataset(ncep_dir_0_360 / "ncep_ncar_consolidated.nc")
    lon_vals = ds.lon.values
    ds.close()

    # All values in -180..180
    assert float(lon_vals.min()) >= -180.0
    assert float(lon_vals.max()) <= 180.0

    # Monotonically increasing
    assert all(lon_vals[i] < lon_vals[i + 1] for i in range(len(lon_vals) - 1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev test -- tests/test_consolidate.py::test_ncep_longitude_normalization -v`

Expected: FAIL — the consolidated file will have 0-360 longitude values (not normalized) because `consolidate_ncep_ncar()` does not yet normalize. The `assert float(lon_vals.max()) <= 180.0` assertion should fail.

- [ ] **Step 3: Commit failing test**

```bash
git add tests/test_consolidate.py
git commit -m "test: add failing test for NCEP/NCAR 0-360 longitude normalization (#54)"
```

### Task 2: Implement longitude normalization

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py:1159-1161`

- [ ] **Step 1: Add normalization after `apply_cf_metadata()`**

In `consolidate_ncep_ncar()`, between the `apply_cf_metadata()` call (line 1159) and the `out_path` assignment (line 1161), add:

```python
        ds = apply_cf_metadata(ds, "ncep_ncar", "monthly")

        lon_name = next(
            (c for c in ds.coords if ds[c].attrs.get("axis") == "X"),
            "lon",
        )
        if float(ds[lon_name].max()) > 180:
            ds.coords[lon_name] = ((ds.coords[lon_name] + 180) % 360) - 180
            ds = ds.sortby(lon_name)

        out_path = source_dir / "ncep_ncar_consolidated.nc"
```

- [ ] **Step 2: Run the new test to verify it passes**

Run: `pixi run -e dev test -- tests/test_consolidate.py::test_ncep_longitude_normalization -v`

Expected: PASS

- [ ] **Step 3: Run the full NCEP/NCAR test suite to check for regressions**

Run: `pixi run -e dev test -- tests/test_consolidate.py -k ncep -v`

Expected: All NCEP/NCAR consolidation tests pass. The existing `ncep_dir` fixture uses -180..180 longitude, so the `max > 180` guard skips normalization and existing behavior is preserved.

- [ ] **Step 4: Run the full test suite**

Run: `pixi run -e dev test`

Expected: All tests pass.

- [ ] **Step 5: Run lint and format**

Run: `pixi run -e dev fmt && pixi run -e dev lint`

Expected: Clean.

- [ ] **Step 6: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate.py
git commit -m "fix: normalize NCEP/NCAR 0-360 longitude to -180..180 in consolidation (#54)"
```
