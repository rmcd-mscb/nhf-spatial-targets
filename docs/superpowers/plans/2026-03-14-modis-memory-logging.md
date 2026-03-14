# MODIS Memory & Logging Improvements — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor MOD16A2 fetch/consolidation to use a per-timestep pipeline that writes temp files instead of accumulating all timesteps in RAM, and add memory/progress logging throughout.

**Architecture:** `consolidate.py` gets two new public functions (`consolidate_mod16a2_timestep`, `consolidate_mod16a2_finalize`) that split the current monolithic `consolidate_mod16a2` into write-per-timestep + lazy-concat phases. `modis.py` orchestrates a per-timestep download→consolidate loop. A public `log_memory` helper in `consolidate.py` provides RSS logging at checkpoints and is imported by `modis.py`.

**Tech Stack:** Python 3.11+, xarray, rioxarray, dask, earthaccess, tqdm

**Spec:** `docs/superpowers/specs/2026-03-14-modis-memory-logging-design.md`

---

## Chunk 1: Memory logging helper and consolidation refactor

### Task 1: Add `log_memory` helper to consolidate.py

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py:1-18` (imports and module top)
- Test: `tests/test_consolidate_modis.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_consolidate_modis.py`, add:

```python
def testlog_memory_does_not_raise():
    """log_memory runs without error on any platform."""
    from nhf_spatial_targets.fetch.consolidate import log_memory

    # Should not raise regardless of platform
    log_memory("test checkpoint")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py::testlog_memory_does_not_raise -v`
Expected: FAIL with `ImportError: cannot import name 'log_memory'`

- [ ] **Step 3: Implement `log_memory`**

In `src/nhf_spatial_targets/fetch/consolidate.py`, add after the `logger = ...` line (line 18):

```python
def log_memory(label: str) -> None:
    """Log current RSS from /proc/self/status (Linux) or peak RSS as fallback."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    rss_gib = rss_kb / (1024**2)
                    logger.info("[memory] RSS=%.2f GiB — %s", rss_gib, label)
                    return
    except OSError:
        pass
    try:
        import resource

        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_gib = peak_kb / (1024**2)
        logger.info("[memory] peak RSS=%.2f GiB — %s", peak_gib, label)
    except (ImportError, OSError):
        logger.debug("[memory] cannot read RSS on this platform — %s", label)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py::testlog_memory_does_not_raise -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate_modis.py
git commit -m "feat: add log_memory helper for RSS checkpoint logging"
```

---

### Task 2: Add `consolidate_mod16a2_timestep` function

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py:427-530` (after existing `_mosaic_and_reproject_timestep`)
- Test: `tests/test_consolidate_modis.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_consolidate_modis.py`, add:

```python
from unittest.mock import patch as _patch


def _make_fake_mosaic(tile_paths, variable, resolution=0.04):
    """Return a synthetic DataArray mimicking _mosaic_and_reproject_timestep."""
    lat = np.linspace(25.0, 50.0, 4)
    lon = np.linspace(-125.0, -65.0, 6)
    data = np.random.rand(1, len(lat), len(lon)).astype(np.float32)
    da = xr.DataArray(data, dims=["band", "y", "x"])
    return da


def test_consolidate_mod16a2_timestep_writes_temp(mod16a2_run_dir: Path) -> None:
    """consolidate_mod16a2_timestep writes a temp NetCDF and returns its path."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2_timestep

    source_key = "mod16a2_v061"
    source_dir = mod16a2_run_dir / "data" / "raw" / source_key

    # Collect DOY 001 tiles
    tile_paths = sorted(source_dir.glob("MOD16A2GF.A2010001.*.hdf"))
    assert len(tile_paths) == 2  # h08v04 and h09v04

    with _patch(
        "nhf_spatial_targets.fetch.consolidate._mosaic_and_reproject_timestep",
        side_effect=_make_fake_mosaic,
    ):
        tmp_path = consolidate_mod16a2_timestep(
            tile_paths=tile_paths,
            variables=["ET_500m"],
            source_dir=source_dir,
            ydoy="2010001",
        )

    assert tmp_path.exists()
    assert tmp_path.name.startswith("_tmp_")
    assert "A2010001" in tmp_path.name
    assert tmp_path.suffix == ".nc"

    ds = xr.open_dataset(tmp_path)
    assert "time" in ds.dims
    assert len(ds.time) == 1
    assert "ET_500m" in ds.data_vars
    assert "lat" in ds.dims
    assert "lon" in ds.dims
    ds.close()

    # Clean up
    tmp_path.unlink()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py::test_consolidate_mod16a2_timestep_writes_temp -v`
Expected: FAIL with `ImportError: cannot import name 'consolidate_mod16a2_timestep'`

- [ ] **Step 3: Implement `consolidate_mod16a2_timestep`**

In `src/nhf_spatial_targets/fetch/consolidate.py`, add before the existing `consolidate_mod16a2` function (before line 427), and add `import os` to the imports at the top:

```python
def consolidate_mod16a2_timestep(
    tile_paths: list[Path],
    variables: list[str],
    source_dir: Path,
    ydoy: str,
    resolution: float = 0.04,
) -> Path:
    """Mosaic and reproject tiles for one timestep, write to a temp NetCDF.

    Parameters
    ----------
    tile_paths : list[Path]
        HDF tile files for a single MODIS timestep.
    variables : list[str]
        Variable names to extract.
    source_dir : Path
        Directory to write the temp file into.
    ydoy : str
        Seven-digit YYYYDDD token (e.g. "2010001").
    resolution : float
        Output resolution in degrees (default 0.04).

    Returns
    -------
    Path
        Path to the written temp NetCDF file.
    """
    timestamp = _time_from_modis_filename(tile_paths[0])

    var_arrays: dict[str, xr.DataArray] = {}
    for var in variables:
        da = _mosaic_and_reproject_timestep(tile_paths, var, resolution)
        if "band" in da.dims:
            da = da.squeeze("band", drop=True)
        rename_map = {}
        if "y" in da.dims:
            rename_map["y"] = "lat"
        if "x" in da.dims:
            rename_map["x"] = "lon"
        if rename_map:
            da = da.rename(rename_map)
        var_arrays[var] = da

    ds_step = xr.Dataset(var_arrays)
    ds_step = ds_step.expand_dims(time=[timestamp])

    tmp_path = source_dir / f"_tmp_{os.getpid()}_A{ydoy}.nc"
    try:
        ds_step.to_netcdf(tmp_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Failed to write temp file for timestep A{ydoy}. "
            f"Detail: {exc}"
        ) from exc
    finally:
        ds_step.close()
        for da in var_arrays.values():
            da.close()

    logger.info("Wrote temp file: %s", tmp_path.name)
    return tmp_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py::test_consolidate_mod16a2_timestep_writes_temp -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate_modis.py
git commit -m "feat: add consolidate_mod16a2_timestep for per-timestep temp writes"
```

---

### Task 3: Add `consolidate_mod16a2_finalize` function

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py`
- Test: `tests/test_consolidate_modis.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_consolidate_modis.py`, add:

```python
def test_consolidate_mod16a2_finalize_concats_and_cleans(tmp_path: Path) -> None:
    """finalize lazy-concats temp files, writes consolidated, cleans up temps."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2_finalize

    source_dir = tmp_path / "data" / "raw" / "mod16a2_v061"
    source_dir.mkdir(parents=True)

    lat = np.linspace(25.0, 50.0, 4)
    lon = np.linspace(-125.0, -65.0, 6)

    tmp_paths = []
    for doy in [1, 9]:
        ts = pd.Timestamp(year=2010, month=1, day=1) + pd.Timedelta(days=doy - 1)
        ds = xr.Dataset(
            {
                "ET_500m": (
                    ["time", "lat", "lon"],
                    np.random.rand(1, len(lat), len(lon)).astype(np.float32),
                ),
            },
            coords={"time": [ts], "lat": lat, "lon": lon},
        )
        p = source_dir / f"_tmp_99999_A2010{doy:03d}.nc"
        ds.to_netcdf(p)
        tmp_paths.append(p)

    out_path = source_dir / "mod16a2_v061_2010_consolidated.nc"
    result = consolidate_mod16a2_finalize(
        tmp_paths=tmp_paths,
        variables=["ET_500m"],
        out_path=out_path,
        run_dir=tmp_path,
    )

    # Final file exists
    assert out_path.exists()
    ds_out = xr.open_dataset(out_path)
    assert len(ds_out.time) == 2
    assert "ET_500m" in ds_out.data_vars
    assert pd.DatetimeIndex(ds_out.time.values).is_monotonic_increasing
    ds_out.close()

    # Temp files cleaned up
    for p in tmp_paths:
        assert not p.exists()

    # Provenance
    assert "consolidated_nc" in result
    assert result["n_files"] == 2


def test_consolidate_mod16a2_finalize_cleans_on_failure(tmp_path: Path) -> None:
    """Temp files are cleaned up even when the final write fails."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2_finalize

    source_dir = tmp_path / "data" / "raw" / "mod16a2_v061"
    source_dir.mkdir(parents=True)

    # Create a temp file that cannot be opened as NetCDF
    bad_tmp = source_dir / "_tmp_99999_A2010001.nc"
    bad_tmp.write_bytes(b"not-netcdf")

    out_path = source_dir / "mod16a2_v061_2010_consolidated.nc"
    with pytest.raises(RuntimeError):
        consolidate_mod16a2_finalize(
            tmp_paths=[bad_tmp],
            variables=["ET_500m"],
            out_path=out_path,
            run_dir=tmp_path,
        )

    # Temp file should be cleaned up
    assert not bad_tmp.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py::test_consolidate_mod16a2_finalize_concats_and_cleans tests/test_consolidate_modis.py::test_consolidate_mod16a2_finalize_cleans_on_failure -v`
Expected: FAIL with `ImportError: cannot import name 'consolidate_mod16a2_finalize'`

- [ ] **Step 3: Implement `consolidate_mod16a2_finalize`**

In `src/nhf_spatial_targets/fetch/consolidate.py`, add after `consolidate_mod16a2_timestep`:

```python
def consolidate_mod16a2_finalize(
    tmp_paths: list[Path],
    variables: list[str],
    out_path: Path,
    run_dir: Path,
) -> dict:
    """Lazy-concat per-timestep temp files into the final consolidated NetCDF.

    Parameters
    ----------
    tmp_paths : list[Path]
        Temp NetCDF files produced by ``consolidate_mod16a2_timestep``.
    variables : list[str]
        Variable names to validate.
    out_path : Path
        Path for the final consolidated file.
    run_dir : Path
        Run workspace root (for computing relative paths in provenance).

    Returns
    -------
    dict
        Provenance record.
    """
    def _cleanup_temps() -> None:
        for p in tmp_paths:
            if p.exists():
                p.unlink()
                logger.debug("Removed temp file: %s", p.name)

    logger.info(
        "Writing final consolidated file from %d timestep files", len(tmp_paths)
    )

    try:
        ds = xr.open_mfdataset(
            [str(p) for p in tmp_paths],
            combine="by_coords",
            chunks={},
        )
        ds = ds.sortby("time")
        _validate_variables(ds, variables)
        ds = ds[variables]
        _write_netcdf(ds, out_path)
        ds.close()
        logger.info("Wrote %s", out_path)
    except Exception as exc:
        _cleanup_temps()
        raise RuntimeError(
            f"Failed to finalize consolidated file {out_path}. "
            f"Temp files cleaned up. Detail: {exc}"
        ) from exc

    _cleanup_temps()
    log_memory(f"after writing {out_path.name}")

    return {
        "consolidated_nc": str(out_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(tmp_paths),
        "variables": variables,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py::test_consolidate_mod16a2_finalize_concats_and_cleans tests/test_consolidate_modis.py::test_consolidate_mod16a2_finalize_cleans_on_failure -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate_modis.py
git commit -m "feat: add consolidate_mod16a2_finalize for lazy concat of temp files"
```

---

### Task 4: Refactor `consolidate_mod16a2` as convenience wrapper

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py:427-530`
- Test: `tests/test_consolidate_modis.py`

- [ ] **Step 1: Run existing tests to confirm baseline**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py -v -k mod16a2`
Expected: all existing MOD16A2 tests PASS

- [ ] **Step 2: Refactor `consolidate_mod16a2`**

Replace the body of `consolidate_mod16a2` (keeping its signature and docstring) with:

```python
def consolidate_mod16a2(
    run_dir: Path,
    source_key: str,
    variables: list[str],
    year: int,
    resolution: float = 0.04,
) -> dict:
    """Merge per-granule MOD16A2 HDF files into a consolidated NetCDF.

    Convenience wrapper: groups tiles by timestep, calls
    ``consolidate_mod16a2_timestep`` for each, then
    ``consolidate_mod16a2_finalize`` to produce the final file.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory.
    source_key : str
        Source key (e.g. ``"mod16a2_v061"``).
    variables : list[str]
        Variable names to include.
    year : int
        Year to consolidate.
    resolution : float
        Output resolution in degrees (default 0.04 ≈ 4 km).

    Returns
    -------
    dict
        Provenance record.
    """
    from collections import defaultdict

    source_dir = run_dir / "data" / "raw" / source_key
    year_pattern = re.compile(rf"\.A({year}\d{{3}})\.")
    hdf_files = sorted(
        f for f in source_dir.glob("*.hdf") if year_pattern.search(f.name)
    )

    if not hdf_files:
        raise FileNotFoundError(
            f"No .hdf files for year {year} found in {source_dir}. "
            f"Run 'nhf-targets fetch {source_key.replace('_', '-')}' first."
        )

    # Clean up any stale temp files from a prior interrupted run (same PID)
    pid = os.getpid()
    for stale in source_dir.glob(f"_tmp_{pid}_*.nc"):
        logger.warning("Removing stale temp file: %s", stale.name)
        stale.unlink()

    # Group tiles by timestep (AYYYYDDD token)
    timestep_groups: dict[str, list[Path]] = defaultdict(list)
    for f in hdf_files:
        m = year_pattern.search(f.name)
        if m:
            timestep_groups[m.group(1)].append(f)

    logger.info(
        "Processing %d HDF files across %d time steps for %s year %d",
        len(hdf_files),
        len(timestep_groups),
        source_key,
        year,
    )

    sorted_ydoys = sorted(timestep_groups)
    n_steps = len(sorted_ydoys)
    tmp_paths: list[Path] = []

    for i, ydoy in enumerate(tqdm(sorted_ydoys, desc=f"Mosaicking {source_key} {year}"), 1):
        tile_paths = timestep_groups[ydoy]
        logger.info(
            "Consolidating timestep %d/%d (A%s): %d tiles",
            i, n_steps, ydoy, len(tile_paths),
        )
        tmp_path = consolidate_mod16a2_timestep(
            tile_paths=tile_paths,
            variables=variables,
            source_dir=source_dir,
            ydoy=ydoy,
            resolution=resolution,
        )
        tmp_paths.append(tmp_path)
        log_memory(f"after timestep {i}/{n_steps} (A{ydoy})")

    out_path = source_dir / f"{source_key}_{year}_consolidated.nc"
    return consolidate_mod16a2_finalize(
        tmp_paths=tmp_paths,
        variables=variables,
        out_path=out_path,
        run_dir=run_dir,
    )
```

- [ ] **Step 3: Run existing MOD16A2 tests to verify no regressions**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py -v -k mod16a2`
Expected: all PASS (same output, new internals)

- [ ] **Step 4: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py
git commit -m "refactor: consolidate_mod16a2 uses timestep+finalize pipeline"
```

---

## Chunk 2: Fetch pipeline refactor and progress logging

### Task 5: Add AYYYYDDD regex and granule grouping to modis.py

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/modis.py:32` (add new regex)
- Test: `tests/test_modis.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_modis.py`, add:

```python
def test_group_granules_by_timestep():
    """Granules are grouped by AYYYYDDD token."""
    from nhf_spatial_targets.fetch.modis import _group_granules_by_timestep

    granules = []
    for doy in [1, 1, 9, 9, 9]:
        g = _mock_granule(f"MOD16A2GF.A2010{doy:03d}.h08v04.061.2020256154955.hdf")
        # earthaccess granule data_links() returns download URLs containing filename
        g.data_links.return_value = [
            f"https://example.com/MOD16A2GF.A2010{doy:03d}.h08v04.061.2020256154955.hdf"
        ]
        granules.append(g)

    groups = _group_granules_by_timestep(granules)

    assert sorted(groups.keys()) == ["2010001", "2010009"]
    assert len(groups["2010001"]) == 2
    assert len(groups["2010009"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_modis.py::test_group_granules_by_timestep -v`
Expected: FAIL with `ImportError: cannot import name '_group_granules_by_timestep'`

- [ ] **Step 3: Implement**

In `src/nhf_spatial_targets/fetch/modis.py`, add after the existing `_MODIS_YEAR_RE` line:

```python
_MODIS_YDOY_RE = re.compile(r"\.A(\d{7})\.")


def _group_granules_by_timestep(
    granules: list,
) -> dict[str, list]:
    """Group earthaccess granules by AYYYYDDD token.

    Parameters
    ----------
    granules : list
        earthaccess granule objects. Each must have a ``data_links()``
        method returning URLs that contain the MODIS filename.

    Returns
    -------
    dict[str, list]
        Mapping from YYYYDDD token to list of granules.
    """
    from collections import defaultdict

    groups: dict[str, list] = defaultdict(list)
    for g in granules:
        links = g.data_links()
        if not links:
            logger.warning("Granule %s has no data links, skipping", g)
            continue
        filename = links[0].split("/")[-1]
        m = _MODIS_YDOY_RE.search(filename)
        if not m:
            logger.warning(
                "Cannot extract AYYYYDDD from granule URL: %s", links[0]
            )
            continue
        groups[m.group(1)].append(g)
    return dict(groups)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_modis.py::test_group_granules_by_timestep -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/modis.py tests/test_modis.py
git commit -m "feat: add _MODIS_YDOY_RE and _group_granules_by_timestep"
```

---

### Task 6: Refactor `fetch_mod16a2` to per-timestep pipeline

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/modis.py:229-377`
- Modify: `tests/test_modis.py` (update mock targets)

- [ ] **Step 1: Run existing tests to confirm baseline**

Run: `pixi run -e dev pytest tests/test_modis.py -v -k mod16a2`
Expected: all PASS

- [ ] **Step 2: Update imports in modis.py**

Replace the consolidation import block:

```python
from nhf_spatial_targets.fetch.consolidate import (
    consolidate_mod10c1,
    consolidate_mod16a2,
)
```

with:

```python
from nhf_spatial_targets.fetch.consolidate import (
    consolidate_mod10c1,
    consolidate_mod16a2,
    consolidate_mod16a2_finalize,
    consolidate_mod16a2_timestep,
    log_memory,
)
```

- [ ] **Step 3: Rewrite `fetch_mod16a2` with per-timestep pipeline**

Replace the `fetch_mod16a2` function body (lines 229-377) with:

```python
def fetch_mod16a2(run_dir: Path, period: str) -> dict:
    """Download MOD16A2 AET granules for the given period.

    Downloads and consolidates per-timestep to limit peak memory.
    Supports incremental download — years already recorded in
    ``manifest.json`` are skipped.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory. Reads ``fabric.json`` for bbox,
        writes files to ``data/raw/mod16a2_v061/``.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    source_key = _MOD16A2_SOURCE_KEY
    meta = _catalog.source(source_key)
    short_name = meta["access"]["short_name"]
    variables = meta["variables"]

    _check_superseded(meta, source_key)
    earthdata_login(run_dir)
    logger.info("Authenticated with NASA Earthdata")
    log_memory("after authentication")

    bbox = _read_fabric_bbox(run_dir)
    bbox_t = _bbox_tuple(bbox)

    # Determine which years need downloading
    all_years = years_in_period(period)
    already_have = _existing_years(run_dir, source_key)
    needed = [y for y in all_years if y not in already_have]

    output_dir = run_dir / "data" / "raw" / source_key
    output_dir.mkdir(parents=True, exist_ok=True)

    consolidated_ncs: dict[str, str] = {}

    if not needed:
        logger.info(
            "All %d years already downloaded, skipping to consolidation",
            len(all_years),
        )
    else:
        for year in needed:
            temporal = (f"{year}-01-01", f"{year}-12-31")
            logger.debug("bbox=%s, temporal=%s, year=%d", bbox_t, temporal, year)

            granules = earthaccess.search_data(
                short_name=short_name,
                bounding_box=bbox_t,
                temporal=temporal,
            )
            logger.info(
                "Found %d granules for %s year %d",
                len(granules),
                short_name,
                year,
            )
            log_memory(f"after search for year {year} ({len(granules)} granules)")

            if not granules:
                raise ValueError(
                    f"No granules found for {short_name} with "
                    f"bbox={bbox_t}, temporal={temporal}"
                )

            # Group granules by timestep for batched download
            ts_groups = _group_granules_by_timestep(granules)
            sorted_ydoys = sorted(ts_groups)
            n_steps = len(sorted_ydoys)
            logger.info(
                "Grouped into %d timesteps for year %d", n_steps, year
            )

            tmp_paths: list[Path] = []

            for i, ydoy in enumerate(sorted_ydoys, 1):
                batch = ts_groups[ydoy]
                logger.info(
                    "Downloading timestep %d/%d (A%s): %d granules",
                    i, n_steps, ydoy, len(batch),
                )

                downloaded = earthaccess.download(
                    batch,
                    local_path=str(output_dir),
                )

                if not downloaded:
                    raise RuntimeError(
                        f"earthaccess.download() returned no files for "
                        f"timestep A{ydoy} ({len(batch)} granules). "
                        f"Check network connectivity and Earthdata credentials."
                    )
                if len(downloaded) < len(batch):
                    logger.warning(
                        "Partial download for timestep A%s: got %d of %d granules.",
                        ydoy, len(downloaded), len(batch),
                    )

                log_memory(f"after downloading timestep {i}/{n_steps} (A{ydoy})")

                # Consolidate this timestep immediately
                tile_paths = [Path(f) for f in downloaded]
                tmp_path = consolidate_mod16a2_timestep(
                    tile_paths=tile_paths,
                    variables=variables,
                    source_dir=output_dir,
                    ydoy=ydoy,
                )
                tmp_paths.append(tmp_path)
                log_memory(
                    f"after consolidating timestep {i}/{n_steps} (A{ydoy})"
                )

            # Finalize: lazy-concat temp files into consolidated NetCDF
            out_path = output_dir / f"{source_key}_{year}_consolidated.nc"
            result = consolidate_mod16a2_finalize(
                tmp_paths=tmp_paths,
                variables=variables,
                out_path=out_path,
                run_dir=run_dir,
            )
            consolidated_ncs[str(year)] = result["consolidated_nc"]
            logger.info("Downloaded and consolidated year %d", year)

    # Re-consolidate any years that were already downloaded but not yet
    # consolidated in this run (e.g. prior download, no consolidated file)
    all_hdf_files = sorted(output_dir.glob("*.hdf"))
    years_on_disk = sorted({_year_from_path(p) for p in all_hdf_files})
    for year in years_on_disk:
        if str(year) not in consolidated_ncs:
            logger.info("Re-consolidating %s year %d", source_key, year)
            result = consolidate_mod16a2(
                run_dir, source_key, variables, year
            )
            consolidated_ncs[str(year)] = result["consolidated_nc"]

    # Build file inventory from all .hdf files on disk
    existing_timestamps = _existing_file_timestamps(run_dir, source_key)
    now_utc = datetime.now(timezone.utc).isoformat()

    files = []
    for p in all_hdf_files:
        rel = str(p.relative_to(run_dir))
        yr = _year_from_path(p)
        files.append(
            {
                "path": rel,
                "year": yr,
                "size_bytes": p.stat().st_size,
                "downloaded_utc": existing_timestamps.get(yr, now_utc),
            }
        )

    # Compute effective period from actual files on disk
    if years_on_disk:
        effective_period = f"{years_on_disk[0]}/{years_on_disk[-1]}"
    else:
        effective_period = period

    # Update manifest.json (merge, don't overwrite)
    _update_manifest(
        run_dir,
        source_key,
        effective_period,
        bbox,
        meta,
        files,
        consolidated_ncs,
    )

    return {
        "source_key": source_key,
        "access_url": meta["access"]["url"],
        "variables": variables,
        "period": effective_period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "files": files,
        "consolidated_ncs": consolidated_ncs,
    }
```

- [ ] **Step 4: Update test mocks**

In `tests/test_modis.py`, apply these changes:

**4a. Update `_mock_granule`** to support `data_links()` (required by `_group_granules_by_timestep`):

```python
def _mock_granule(name: str) -> MagicMock:
    """Create a mock granule object."""
    g = MagicMock()
    g.__str__ = lambda self: name
    g.data_links.return_value = [f"https://example.com/{name}"]
    return g
```

**4b. Add module-level mock return values** for the new functions:

```python
_MOCK_CONSOLIDATION_FINALIZE = {
    "consolidated_nc": "data/raw/mod16a2_v061/mod16a2_v061_2010_consolidated.nc",
    "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
    "n_files": 1,
    "variables": ["ET_500m", "ET_QC_500m"],
}
```

**4c. Update each test** that currently patches `consolidate_mod16a2`. The new `fetch_mod16a2` calls `consolidate_mod16a2_timestep` and `consolidate_mod16a2_finalize` during the download path, and `consolidate_mod16a2` only for re-consolidation of already-downloaded years. Each test needs the full decorator stack. Here are the complete updated test functions:

```python
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2",
    return_value=_MOCK_CONSOLIDATION,
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_finalize",
    return_value=_MOCK_CONSOLIDATION_FINALIZE,
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_timestep",
    return_value=Path("/tmp/fake_tmp.nc"),
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_search_params(
    mock_login, mock_search, mock_dl, mock_ts, mock_fin, mock_consolidate, run_dir
):
    """search_data called with correct short_name, bbox tuple, and temporal."""
    mock_search.return_value = [
        _mock_granule("MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf")
    ]
    mock_dl.return_value = _fake_download(run_dir)

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    fetch_mod16a2(run_dir=run_dir, period="2010/2010")

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["short_name"] == "MOD16A2GF"
    assert call_kwargs["bounding_box"] == (-125.1, 23.9, -65.9, 50.1)
    assert call_kwargs["temporal"] == ("2010-01-01", "2010-12-31")


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2",
    return_value=_MOCK_CONSOLIDATION,
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_finalize",
    return_value=_MOCK_CONSOLIDATION_FINALIZE,
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_timestep",
    return_value=Path("/tmp/fake_tmp.nc"),
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_provenance_record(
    mock_login, mock_search, mock_dl, mock_ts, mock_fin, mock_consolidate, run_dir
):
    """Returned dict has all required provenance keys."""
    mock_search.return_value = [
        _mock_granule("MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf")
    ]
    mock_dl.return_value = _fake_download(run_dir)

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    result = fetch_mod16a2(run_dir=run_dir, period="2010/2010")

    assert result["source_key"] == "mod16a2_v061"
    assert "access_url" in result
    assert result["variables"] == ["ET_500m", "ET_QC_500m"]
    assert result["period"] == "2010/2010"
    assert "bbox" in result
    assert "download_timestamp" in result
    assert isinstance(result["files"], list)
    assert len(result["files"]) == 1
    assert "path" in result["files"][0]
    assert "year" in result["files"][0]
    assert "size_bytes" in result["files"][0]
    assert isinstance(result["consolidated_ncs"], dict)
    assert not Path(result["files"][0]["path"]).is_absolute()


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2",
    return_value=_MOCK_CONSOLIDATION,
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_finalize",
    return_value=_MOCK_CONSOLIDATION_FINALIZE,
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_timestep",
    return_value=Path("/tmp/fake_tmp.nc"),
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_manifest_updated(
    mock_login, mock_search, mock_dl, mock_ts, mock_fin, mock_consolidate, run_dir
):
    """fetch_mod16a2 writes provenance to manifest.json."""
    mock_search.return_value = [
        _mock_granule("MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf")
    ]
    mock_dl.return_value = _fake_download(run_dir)

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    fetch_mod16a2(run_dir=run_dir, period="2010/2010")

    updated_manifest = json.loads((run_dir / "manifest.json").read_text())
    entry = updated_manifest["sources"]["mod16a2_v061"]
    assert entry["period"] == "2010/2010"
    assert len(entry["files"]) > 0
    assert "year" in entry["files"][0]
    assert "consolidated_ncs" in entry
    assert "variables" in entry
    assert entry["variables"] == ["ET_500m", "ET_QC_500m"]


@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2",
    return_value=_MOCK_CONSOLIDATION,
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_finalize",
    return_value={
        "consolidated_nc": "data/raw/mod16a2_v061/mod16a2_v061_2011_consolidated.nc",
        "last_consolidated_utc": "2026-01-01T00:00:00+00:00",
        "n_files": 1,
        "variables": ["ET_500m", "ET_QC_500m"],
    },
)
@patch(
    "nhf_spatial_targets.fetch.modis.consolidate_mod16a2_timestep",
    return_value=Path("/tmp/fake_tmp.nc"),
)
@patch("earthaccess.download")
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_incremental_skips_year(
    mock_login, mock_search, mock_dl, mock_ts, mock_fin, mock_consolidate, run_dir
):
    """Years already in manifest are not re-downloaded."""
    # Pre-populate manifest with 2010 already downloaded
    manifest = {
        "sources": {
            "mod16a2_v061": {
                "period": "2010/2010",
                "files": [
                    {
                        "path": "data/raw/mod16a2_v061/MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf",
                        "year": 2010,
                        "size_bytes": 100,
                        "downloaded_utc": "2026-01-01T00:00:00+00:00",
                    }
                ],
            }
        }
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    # Create the existing file on disk
    existing = (
        run_dir
        / "data"
        / "raw"
        / "mod16a2_v061"
        / "MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf"
    )
    existing.write_bytes(b"fake")

    # Also create a 2011 file that will be "downloaded"
    mock_search.return_value = [
        _mock_granule("MOD16A2GF.A2011001.h08v04.061.2020256154955.hdf")
    ]
    mock_dl.return_value = _fake_download(run_dir, year=2011)

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    fetch_mod16a2(run_dir=run_dir, period="2010/2011")

    # search_data should only be called for 2011, not 2010
    assert mock_search.call_count == 1
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["temporal"] == ("2011-01-01", "2011-12-31")
```

**4d. Update `test_mod16a2_empty_download_raises`** — the mock granule name must contain a valid MODIS filename for `_group_granules_by_timestep` to group it:

```python
@patch("earthaccess.download", return_value=[])
@patch("earthaccess.search_data")
@patch("nhf_spatial_targets.fetch.modis.earthdata_login")
def test_mod16a2_empty_download_raises(mock_login, mock_search, mock_dl, run_dir):
    """RuntimeError raised when download returns no files."""
    mock_search.return_value = [
        _mock_granule("MOD16A2GF.A2010001.h08v04.061.2020256154955.hdf")
    ]

    from nhf_spatial_targets.fetch.modis import fetch_mod16a2

    with pytest.raises(RuntimeError, match="returned no files"):
        fetch_mod16a2(run_dir=run_dir, period="2010/2010")
```

Note: all mock granule names now include the full MODIS filename so `_group_granules_by_timestep` can extract the AYYYYDDD token from `data_links()`.

- [ ] **Step 5: Run all MOD16A2 tests**

Run: `pixi run -e dev pytest tests/test_modis.py -v -k mod16a2`
Expected: all PASS

- [ ] **Step 6: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/nhf_spatial_targets/fetch/modis.py tests/test_modis.py
git commit -m "refactor: fetch_mod16a2 uses per-timestep download+consolidate pipeline"
```

---

### Task 7: Run full quality gate and verify

**Files:** None (verification only)

- [ ] **Step 1: Run fmt + lint + test**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: all PASS, no lint errors

- [ ] **Step 2: Verify no regressions in other modules**

Run: `pixi run -e dev pytest tests/ -v --tb=short`
Expected: all tests PASS

- [ ] **Step 3: Final commit if any formatting changes**

```bash
git add -u && git commit -m "style: formatting fixes from ruff"
```
