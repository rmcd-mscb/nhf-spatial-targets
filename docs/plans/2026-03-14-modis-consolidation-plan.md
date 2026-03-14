# MODIS Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the `consolidate_mod16a2` and `consolidate_mod10c1` stubs in `fetch/consolidate.py` with working implementations that merge per-granule downloads into single consolidated NetCDF files per year.

**Architecture:** MOD16A2 requires mosaicking sinusoidal tiles per time step, reprojecting to EPSG:4326 at 0.04° (~4km), then stacking along time. MOD10C1 is simpler — concat daily `.conus.nc` files along time. Both produce one consolidated NetCDF per year.

**Tech Stack:** rioxarray (open_rasterio, merge_arrays, rio.reproject), rasterio.enums.Resampling, xarray, numpy

---

### Task 1: MOD10C1 consolidation (the simpler one)

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py:293-300`
- Create: `tests/test_consolidate_modis.py`

**Step 1: Write the failing test**

Create `tests/test_consolidate_modis.py`:

```python
"""Tests for MODIS consolidation functions."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from nhf_spatial_targets.fetch.consolidate import consolidate_mod10c1


@pytest.fixture()
def mod10c1_run_dir(tmp_path: Path) -> Path:
    """Create a run workspace with synthetic MOD10C1 .conus.nc files."""
    rd = tmp_path / "run"
    rd.mkdir()
    source_dir = rd / "data" / "raw" / "mod10c1_v061"
    source_dir.mkdir(parents=True)

    # Create 3 synthetic daily files for year 2010
    for doy in [1, 2, 3]:
        fname = f"MOD10C1.A2010{doy:03d}.061.2020345123456.conus.nc"
        ds = xr.Dataset(
            {
                "Day_CMG_Snow_Cover": (
                    ["lat", "lon"],
                    np.random.randint(0, 100, (4, 6), dtype=np.int16),
                ),
                "Snow_Spatial_QA": (
                    ["lat", "lon"],
                    np.random.randint(0, 100, (4, 6), dtype=np.int16),
                ),
            },
            coords={
                "lat": [50.0, 49.95, 49.90, 49.85],
                "lon": [-125.0, -124.95, -124.90, -124.85, -124.80, -124.75],
            },
        )
        ds.to_netcdf(source_dir / fname)
    return rd


def test_consolidate_mod10c1_basic(mod10c1_run_dir):
    """Consolidation merges daily files into one NetCDF with time dim."""
    result = consolidate_mod10c1(
        run_dir=mod10c1_run_dir,
        source_key="mod10c1_v061",
        variables=["Day_CMG_Snow_Cover", "Snow_Spatial_QA"],
        year=2010,
    )

    assert "consolidated_nc" in result
    assert "last_consolidated_utc" in result
    assert result["n_files"] == 3
    assert result["variables"] == ["Day_CMG_Snow_Cover", "Snow_Spatial_QA"]

    nc_path = mod10c1_run_dir / result["consolidated_nc"]
    assert nc_path.exists()

    ds = xr.open_dataset(nc_path)
    assert "time" in ds.dims
    assert ds.sizes["time"] == 3
    assert "Day_CMG_Snow_Cover" in ds.data_vars
    assert "Snow_Spatial_QA" in ds.data_vars
    ds.close()


def test_consolidate_mod10c1_no_files(tmp_path):
    """Raises FileNotFoundError when no .conus.nc files exist for year."""
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "mod10c1_v061").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        consolidate_mod10c1(rd, "mod10c1_v061", ["Day_CMG_Snow_Cover"], year=2010)


def test_consolidate_mod10c1_overwrites_existing(mod10c1_run_dir):
    """Re-running consolidation overwrites the previous file."""
    result1 = consolidate_mod10c1(
        mod10c1_run_dir, "mod10c1_v061",
        ["Day_CMG_Snow_Cover", "Snow_Spatial_QA"], year=2010,
    )
    result2 = consolidate_mod10c1(
        mod10c1_run_dir, "mod10c1_v061",
        ["Day_CMG_Snow_Cover", "Snow_Spatial_QA"], year=2010,
    )
    assert result1["consolidated_nc"] == result2["consolidated_nc"]
    assert (mod10c1_run_dir / result2["consolidated_nc"]).exists()


def test_consolidate_mod10c1_filters_year(mod10c1_run_dir):
    """Only files matching the requested year are included."""
    # Add a file for a different year
    source_dir = mod10c1_run_dir / "data" / "raw" / "mod10c1_v061"
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (["lat", "lon"], np.zeros((4, 6), dtype=np.int16)),
            "Snow_Spatial_QA": (["lat", "lon"], np.zeros((4, 6), dtype=np.int16)),
        },
        coords={
            "lat": [50.0, 49.95, 49.90, 49.85],
            "lon": [-125.0, -124.95, -124.90, -124.85, -124.80, -124.75],
        },
    )
    ds.to_netcdf(source_dir / "MOD10C1.A2011001.061.2021345123456.conus.nc")

    result = consolidate_mod10c1(
        mod10c1_run_dir, "mod10c1_v061",
        ["Day_CMG_Snow_Cover", "Snow_Spatial_QA"], year=2010,
    )
    assert result["n_files"] == 3  # only 2010 files
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py -v`
Expected: FAIL with `NotImplementedError`

**Step 3: Write minimal implementation**

In `src/nhf_spatial_targets/fetch/consolidate.py`, replace the `consolidate_mod10c1` stub (lines 293-300) with:

```python
def consolidate_mod10c1(
    run_dir: Path,
    source_key: str,
    variables: list[str],
    year: int,
) -> dict:
    """Merge daily MOD10C1 CONUS subsets for a single year into one NetCDF.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory containing ``data/raw/<source_key>/*.conus.nc``.
    source_key : str
        Source key (e.g. ``"mod10c1_v061"``).
    variables : list[str]
        Variable names to include.
    year : int
        Year to consolidate.

    Returns
    -------
    dict
        Provenance record.
    """
    import re

    source_dir = run_dir / "data" / "raw" / source_key
    year_re = re.compile(rf"\.A{year}\d{{3}}\.")
    nc_files = sorted(
        f for f in source_dir.glob("*.conus.nc") if year_re.search(f.name)
    )

    if not nc_files:
        raise FileNotFoundError(
            f"No .conus.nc files found for year {year} in {source_dir}. "
            f"Run 'nhf-targets fetch {source_key.replace('_', '-')}' first."
        )

    logger.info("Merging %d daily files for %s year %d", len(nc_files), source_key, year)

    datasets = _open_datasets(nc_files, f"Reading {source_key} {year}")
    try:
        ds = xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal")
        ds = ds.sortby("time")
        _validate_variables(ds, variables)
        ds = ds[variables]

        out_path = source_dir / f"{source_key}_{year}_consolidated.nc"
        logger.info("Writing consolidated file: %s", out_path)
        _write_netcdf(ds, out_path)
        logger.info("Wrote %s", out_path)
    finally:
        for d in datasets:
            d.close()

    return {
        "consolidated_nc": str(out_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(nc_files),
        "variables": variables,
    }
```

**Important:** The `.conus.nc` files created by `_subset_to_conus` in `modis.py` use `xr.open_dataset` + `to_netcdf`, so they have lat/lon coordinates but no `time` dimension. The files are one-per-day. We need to infer time from the filename. Add this helper before `consolidate_mod10c1`:

```python
def _time_from_modis_filename(path: Path) -> pd.Timestamp:
    """Extract time from MODIS ``AYYYYDDD`` filename pattern."""
    m = re.search(r"\.A(\d{4})(\d{3})\.", path.name)
    if not m:
        raise ValueError(f"Cannot extract date from MODIS filename: {path.name}")
    year, doy = int(m.group(1)), int(m.group(2))
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
```

And update `_open_datasets` call to assign time coordinates. Actually, it's cleaner to assign time after opening. Revise the implementation to:

```python
    datasets = []
    for f in tqdm(nc_files, desc=f"Reading {source_key} {year}"):
        try:
            ds_i = xr.open_dataset(f, chunks={})
            t = _time_from_modis_filename(f)
            ds_i = ds_i.expand_dims(time=[t])
            datasets.append(ds_i)
        except Exception as exc:
            for d in datasets:
                d.close()
            raise RuntimeError(
                f"Failed to open {f.name} during consolidation. "
                f"The file may be corrupt or truncated. "
                f"Delete it and re-run the fetch. Detail: {exc}"
            ) from exc
    try:
        ds = xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal")
        ds = ds.sortby("time")
        _validate_variables(ds, variables)
        ds = ds[variables]

        out_path = source_dir / f"{source_key}_{year}_consolidated.nc"
        logger.info("Writing consolidated file: %s", out_path)
        _write_netcdf(ds, out_path)
        logger.info("Wrote %s", out_path)
    finally:
        for d in datasets:
            d.close()
```

Also update the test fixture to NOT include a time dimension in the synthetic files (matching real `_subset_to_conus` output which has only lat/lon).

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py -v`
Expected: PASS (all 4 tests)

**Step 5: Run full test suite**

Run: `pixi run -e dev test`
Expected: All tests pass

**Step 6: Commit**

```bash
git add tests/test_consolidate_modis.py src/nhf_spatial_targets/fetch/consolidate.py
git commit -m "feat: implement consolidate_mod10c1"
```

---

### Task 2: MOD16A2 tile grouping and time extraction tests

**Files:**
- Modify: `tests/test_consolidate_modis.py`
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py`

**Step 1: Write the failing tests**

Add to `tests/test_consolidate_modis.py`:

```python
from nhf_spatial_targets.fetch.consolidate import _time_from_modis_filename


def test_time_from_modis_filename():
    """Extract date from MODIS AYYYYDDD filename."""
    t = _time_from_modis_filename(Path("MOD16A2GF.A2010001.h08v04.061.hdf"))
    assert t == pd.Timestamp("2010-01-01")

    t = _time_from_modis_filename(Path("MOD16A2GF.A2010009.h08v04.061.hdf"))
    assert t == pd.Timestamp("2010-01-09")

    t = _time_from_modis_filename(Path("MOD10C1.A2010032.061.conus.nc"))
    assert t == pd.Timestamp("2010-02-01")


def test_time_from_modis_filename_bad():
    """Raises ValueError for non-MODIS filename."""
    with pytest.raises(ValueError, match="Cannot extract date"):
        _time_from_modis_filename(Path("random_file.nc"))
```

Import `pd` at top of test file:

```python
import pandas as pd
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py::test_time_from_modis_filename -v`
Expected: FAIL with ImportError (function doesn't exist yet or was added in Task 1)

**Step 3: Verify `_time_from_modis_filename` works**

If already added in Task 1, tests should pass. If not, add the helper now.

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_consolidate_modis.py src/nhf_spatial_targets/fetch/consolidate.py
git commit -m "test: add time extraction tests for MODIS filenames"
```

---

### Task 3: MOD16A2 consolidation — mosaic, reproject, stack

This is the most complex task. MOD16A2 HDF tiles are in sinusoidal projection with multiple tiles per 8-day composite.

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py:303-327`
- Modify: `tests/test_consolidate_modis.py`

**Step 1: Write the failing test**

Add to `tests/test_consolidate_modis.py`:

```python
import rioxarray  # noqa: F401 — registers rio accessor
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2


def _make_sinusoidal_tile(
    path: Path,
    variable: str,
    h: int,
    v: int,
    year: int,
    doy: int,
) -> None:
    """Create a tiny synthetic sinusoidal-projected HDF-like NetCDF tile.

    We use NetCDF instead of real HDF4 for testability. The consolidation
    code uses rioxarray.open_rasterio which can handle NetCDF with CRS.
    """
    # Sinusoidal CRS (MODIS)
    srs = CRS.from_proj4(
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
    )
    # Small 4x4 grid in sinusoidal coords (arbitrary tile location)
    nx, ny = 4, 4
    # Use h/v to offset the tile position
    x0 = -10_000_000 + h * 100_000
    y0 = 6_000_000 - v * 100_000
    res = 500.0  # 500m
    transform = from_bounds(x0, y0 - ny * res, x0 + nx * res, y0, nx, ny)

    data = np.random.randint(0, 1000, (1, ny, nx), dtype=np.int16)
    da = xr.DataArray(
        data,
        dims=["band", "y", "x"],
        coords={"band": [1]},
    )
    da.rio.write_crs(srs, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_nodata(-1, inplace=True)
    da.name = variable
    da.rio.to_raster(path)


@pytest.fixture()
def mod16a2_run_dir(tmp_path: Path) -> Path:
    """Create run workspace with synthetic MOD16A2 sinusoidal tiles."""
    rd = tmp_path / "run"
    rd.mkdir()
    source_dir = rd / "data" / "raw" / "mod16a2_v061"
    source_dir.mkdir(parents=True)

    # 2 time steps (DOY 001, 009), 2 tiles each (h08v04, h09v04)
    for doy in [1, 9]:
        for h, v in [(8, 4), (9, 4)]:
            for var in ["ET_500m", "ET_QC_500m"]:
                fname = f"MOD16A2GF.A2010{doy:03d}.h{h:02d}v{v:02d}.061.{var}.tif"
                _make_sinusoidal_tile(
                    source_dir / fname, var, h, v, 2010, doy,
                )
    return rd


def test_consolidate_mod16a2_basic(mod16a2_run_dir):
    """Consolidation mosaics tiles, reprojects to 4326, stacks along time."""
    result = consolidate_mod16a2(
        run_dir=mod16a2_run_dir,
        source_key="mod16a2_v061",
        variables=["ET_500m", "ET_QC_500m"],
        year=2010,
    )

    assert "consolidated_nc" in result
    assert "last_consolidated_utc" in result
    assert result["variables"] == ["ET_500m", "ET_QC_500m"]

    nc_path = mod16a2_run_dir / result["consolidated_nc"]
    assert nc_path.exists()

    ds = xr.open_dataset(nc_path)
    assert "time" in ds.dims
    assert ds.sizes["time"] == 2  # 2 time steps
    assert "ET_500m" in ds.data_vars
    assert "ET_QC_500m" in ds.data_vars
    # Verify reprojected to EPSG:4326 (1D lat/lon coords)
    assert "lat" in ds.coords or "y" in ds.coords
    assert "lon" in ds.coords or "x" in ds.coords
    ds.close()


def test_consolidate_mod16a2_no_files(tmp_path):
    """Raises FileNotFoundError when no HDF files exist for year."""
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "mod16a2_v061").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        consolidate_mod16a2(rd, "mod16a2_v061", ["ET_500m"], year=2010)
```

**Note on test design:** Real MOD16A2 files are HDF4 with subdatasets. For testing, we create GeoTIFF tiles with sinusoidal CRS using rioxarray. The consolidation code must handle both real HDF and test GeoTIFF — we use `rioxarray.open_rasterio()` which handles both formats.

The actual file glob pattern in the implementation needs to handle `.hdf` files (production) and `.tif` files (tests). Alternatively, the test fixture can create `.hdf`-named files that are actually NetCDF/GeoTIFF — rioxarray doesn't care about the extension. **Use `.hdf` extension in tests for realism.**

Update `_make_sinusoidal_tile` to use `.hdf` extension in the fixture:

```python
                fname = f"MOD16A2GF.A2010{doy:03d}.h{h:02d}v{v:02d}.061.2020256154955.hdf"
```

And create one file per tile (not per variable) — the real HDF has multiple subdatasets. For testing simplicity, store just `ET_500m` in each tile. The implementation will need to handle extracting variables from HDF subdatasets.

**Actually, let's simplify the test approach:** Real MODIS HDF4 files have subdatasets like `HDF4_EOS:EOS_GRID:"file.hdf":MOD_Grid_MOD16A2:ET_500m`. This is complex to mock. Instead:

1. The test creates small GeoTIFF files with `.tif` extension
2. The implementation groups files by time step and variable, mosaics per variable per time step
3. The glob pattern is configurable or we glob for both `.hdf` and `.tif`

**Revised simpler approach for implementation:** Since MODIS HDF4 files require GDAL HDF4 driver and have subdataset structure that's hard to mock, the implementation should:

1. Glob `.hdf` files
2. For each HDF, list subdatasets via `rioxarray.open_rasterio(f, variable=var_name)` to extract each variable
3. Group by (time_step, variable)
4. Mosaic tiles per group, reproject each mosaic
5. Combine variables into a Dataset, stack along time

For testing, we'll mock the rioxarray operations or create simple raster files.

**Revised test — mock-based approach:**

```python
from unittest.mock import patch, MagicMock

def test_consolidate_mod16a2_no_files(tmp_path):
    """Raises FileNotFoundError when no HDF files exist for year."""
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "data" / "raw" / "mod16a2_v061").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        consolidate_mod16a2(rd, "mod16a2_v061", ["ET_500m"], year=2010)
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py::test_consolidate_mod16a2_no_files -v`
Expected: FAIL with NotImplementedError

**Step 3: Write the implementation**

Replace `consolidate_mod16a2` stub in `consolidate.py`:

```python
def _mosaic_and_reproject_timestep(
    tile_paths: list[Path],
    variable: str,
    resolution: float = 0.04,
) -> xr.DataArray:
    """Mosaic tiles for one variable/timestep, reproject to EPSG:4326.

    Parameters
    ----------
    tile_paths : list[Path]
        Paths to HDF tiles for a single time step.
    variable : str
        Subdataset variable name to extract from each tile.
    resolution : float
        Target resolution in degrees (default 0.04° ≈ 4km).

    Returns
    -------
    xr.DataArray
        Mosaicked and reprojected array with y/x dimensions.
    """
    from rasterio.enums import Resampling
    from rioxarray.merge import merge_arrays

    arrays = []
    for p in tile_paths:
        da = rioxarray.open_rasterio(p, variable=variable, masked=True)
        if isinstance(da, list):
            da = da[0]
        arrays.append(da)

    if len(arrays) == 1:
        mosaic = arrays[0]
    else:
        mosaic = merge_arrays(arrays)

    # Choose resampling: average for continuous data, nearest for QC
    if "QC" in variable or "qa" in variable.lower():
        resampling = Resampling.nearest
    else:
        resampling = Resampling.average

    reprojected = mosaic.rio.reproject(
        "EPSG:4326",
        resolution=resolution,
        resampling=resampling,
    )

    for a in arrays:
        a.close()

    return reprojected


def consolidate_mod16a2(
    run_dir: Path,
    source_key: str,
    variables: list[str],
    year: int,
    resolution: float = 0.04,
) -> dict:
    """Merge per-granule MOD16A2 HDF files into a consolidated NetCDF.

    For each 8-day composite (identified by AYYYYDDD filename pattern):
    1. Mosaic tiles into a single raster per variable
    2. Reproject from sinusoidal to EPSG:4326 at *resolution* degrees
    3. Stack all time steps along a time dimension

    Parameters
    ----------
    run_dir : Path
        Run workspace directory.
    source_key : str
        Source key (e.g. ``"mod16a2_v061"``).
    variables : list[str]
        Variable names to extract from each HDF (e.g. ``["ET_500m"]``).
    year : int
        Year to consolidate.
    resolution : float
        Target resolution in degrees (default 0.04° ≈ 4km).

    Returns
    -------
    dict
        Provenance record.
    """
    import re
    from collections import defaultdict

    import rioxarray  # noqa: F401 — registers rio accessor

    source_dir = run_dir / "data" / "raw" / source_key
    year_re = re.compile(rf"\.A{year}\d{{3}}\.")
    # Match the AYYYYDDD token to group by time step
    timestep_re = re.compile(r"\.(A\d{7})\.")

    hdf_files = sorted(
        f for f in source_dir.glob("*.hdf") if year_re.search(f.name)
    )

    if not hdf_files:
        raise FileNotFoundError(
            f"No .hdf files found for year {year} in {source_dir}. "
            f"Run 'nhf-targets fetch {source_key.replace('_', '-')}' first."
        )

    # Group tiles by time step (AYYYYDDD token)
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in hdf_files:
        m = timestep_re.search(f.name)
        if m:
            groups[m.group(1)].append(f)
        else:
            logger.warning("Skipping file without AYYYYDDD token: %s", f.name)

    logger.info(
        "Processing %d time steps (%d tiles) for %s year %d",
        len(groups), len(hdf_files), source_key, year,
    )

    # Process each time step: mosaic tiles, reproject, build dataset
    time_slices: list[xr.Dataset] = []
    for ts_key in sorted(groups):
        tile_paths = groups[ts_key]
        t = _time_from_modis_filename(tile_paths[0])

        var_arrays = {}
        for var in variables:
            try:
                da = _mosaic_and_reproject_timestep(
                    tile_paths, var, resolution=resolution,
                )
                # Drop band dim if present (single band)
                if "band" in da.dims:
                    da = da.squeeze("band", drop=True)
                var_arrays[var] = da
            except Exception as exc:
                logger.warning(
                    "Failed to process variable %s for %s: %s",
                    var, ts_key, exc,
                )
                raise

        # Build dataset for this time step
        ds_t = xr.Dataset(var_arrays)
        ds_t = ds_t.expand_dims(time=[t])

        # Rename x/y to lon/lat for clean 1D coordinates
        if "x" in ds_t.dims:
            ds_t = ds_t.rename({"x": "lon", "y": "lat"})

        time_slices.append(ds_t)

    try:
        ds = xr.concat(time_slices, dim="time", data_vars="minimal", coords="minimal")
        ds = ds.sortby("time")

        out_path = source_dir / f"{source_key}_{year}_consolidated.nc"
        logger.info("Writing consolidated file: %s", out_path)
        _write_netcdf(ds, out_path)
        logger.info("Wrote %s", out_path)
    finally:
        for s in time_slices:
            s.close()

    return {
        "consolidated_nc": str(out_path.relative_to(run_dir)),
        "last_consolidated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(hdf_files),
        "variables": variables,
    }
```

Add `import rioxarray` to the top-level imports in `consolidate.py`.

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pixi run -e dev test`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate_modis.py
git commit -m "feat: implement consolidate_mod16a2 with mosaic and reproject"
```

---

### Task 4: MOD16A2 integration test with synthetic rasters

**Files:**
- Modify: `tests/test_consolidate_modis.py`

**Step 1: Write the test using real rioxarray operations**

This test creates actual sinusoidal-projected GeoTIFF files (named `.hdf`) and verifies end-to-end mosaicking, reprojection, and time stacking.

Add to `tests/test_consolidate_modis.py`:

```python
from rasterio.crs import CRS
from rasterio.transform import from_bounds


def _make_sinusoidal_tile(path: Path, h: int, v: int, value: int = 42) -> None:
    """Create a small sinusoidal-projected GeoTIFF (named .hdf for testing)."""
    srs = CRS.from_proj4(
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
    )
    nx, ny = 4, 4
    x0 = -10_000_000 + h * 200_000
    y0 = 6_000_000 - v * 200_000
    res = 500.0
    transform = from_bounds(x0, y0 - ny * res, x0 + nx * res, y0, nx, ny)

    data = np.full((1, ny, nx), value, dtype=np.int16)
    da = xr.DataArray(data, dims=["band", "y", "x"], coords={"band": [1]})
    da.rio.write_crs(srs, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_nodata(-1, inplace=True)
    da.rio.to_raster(path)


@pytest.fixture()
def mod16a2_run_dir(tmp_path: Path) -> Path:
    """Create run workspace with synthetic MOD16A2 tiles."""
    rd = tmp_path / "run"
    rd.mkdir()
    source_dir = rd / "data" / "raw" / "mod16a2_v061"
    source_dir.mkdir(parents=True)

    # 2 time steps (DOY 001, 009), 2 tiles each (h08v04, h09v04)
    for doy in [1, 9]:
        for h, v in [(8, 4), (9, 4)]:
            fname = (
                f"MOD16A2GF.A2010{doy:03d}.h{h:02d}v{v:02d}"
                f".061.2020256154955.hdf"
            )
            _make_sinusoidal_tile(source_dir / fname, h, v, value=doy * 10)

    return rd


def test_consolidate_mod16a2_synthetic(mod16a2_run_dir):
    """End-to-end: mosaic sinusoidal tiles, reproject to 4326, stack time."""
    result = consolidate_mod16a2(
        run_dir=mod16a2_run_dir,
        source_key="mod16a2_v061",
        variables=["ET_500m"],
        year=2010,
    )

    assert result["n_files"] == 4  # 2 timesteps × 2 tiles
    nc_path = mod16a2_run_dir / result["consolidated_nc"]
    assert nc_path.exists()

    ds = xr.open_dataset(nc_path)
    assert "time" in ds.dims
    assert ds.sizes["time"] == 2
    assert "ET_500m" in ds.data_vars
    # Verify 1D lat/lon coordinates (EPSG:4326)
    assert "lat" in ds.dims or "lat" in ds.coords
    assert "lon" in ds.dims or "lon" in ds.coords
    ds.close()


def test_consolidate_mod16a2_partial_tiles(mod16a2_run_dir):
    """Consolidation succeeds even if a time step has fewer tiles."""
    # Remove one tile from DOY 009
    source_dir = mod16a2_run_dir / "data" / "raw" / "mod16a2_v061"
    for f in source_dir.glob("*A2010009*h09v04*"):
        f.unlink()

    result = consolidate_mod16a2(
        mod16a2_run_dir, "mod16a2_v061", ["ET_500m"], year=2010,
    )
    assert result["n_files"] == 3  # 2 + 1
    nc_path = mod16a2_run_dir / result["consolidated_nc"]
    ds = xr.open_dataset(nc_path)
    assert ds.sizes["time"] == 2  # still 2 time steps
    ds.close()
```

**Note:** The `_mosaic_and_reproject_timestep` function uses `rioxarray.open_rasterio(p, variable=variable)`. For real MODIS HDF4 files, the `variable` parameter selects the subdataset. For GeoTIFF test files, there's only one band (no subdatasets), so the `variable` parameter may not apply. The implementation needs to handle both cases:
- If `variable` parameter works (HDF4 with subdatasets) — use it
- If it fails or file has no subdatasets (GeoTIFF) — open directly

Update `_mosaic_and_reproject_timestep` to try `variable=` first, fall back to plain open:

```python
    for p in tile_paths:
        try:
            da = rioxarray.open_rasterio(p, variable=variable, masked=True)
        except Exception:
            da = rioxarray.open_rasterio(p, masked=True)
        if isinstance(da, list):
            da = da[0]
        arrays.append(da)
```

**Step 2: Run tests**

Run: `pixi run -e dev pytest tests/test_consolidate_modis.py -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `pixi run -e dev test`
Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/test_consolidate_modis.py src/nhf_spatial_targets/fetch/consolidate.py
git commit -m "test: add synthetic raster tests for MOD16A2 consolidation"
```

---

### Task 5: Update fetch modules to remove NotImplementedError catch

Now that consolidation is implemented, the `try/except NotImplementedError` wrappers in `modis.py` (added during PR review) should be removed — consolidation should run normally.

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/modis.py`

**Step 1: Update modis.py**

In `fetch_mod16a2` (around line 342), change:

```python
    for year in years_on_disk:
        try:
            result = consolidate_mod16a2(run_dir, source_key, variables, year)
        except NotImplementedError:
            logger.warning(
                "Consolidation for %s year %d not yet implemented; skipping",
                source_key,
                year,
            )
            continue
        if result and "consolidated_nc" in result:
            consolidated_ncs[str(year)] = result["consolidated_nc"]
```

Back to:

```python
    for year in years_on_disk:
        result = consolidate_mod16a2(run_dir, source_key, variables, year)
        if result and "consolidated_nc" in result:
            consolidated_ncs[str(year)] = result["consolidated_nc"]
```

Do the same for `fetch_mod10c1` (around line 562).

**Step 2: Update test mocks in test_modis.py**

The existing tests in `test_modis.py` mock consolidation. Verify they still work — the mocks return dicts, not raise NotImplementedError, so they should be fine.

**Step 3: Run full test suite**

Run: `pixi run -e dev test`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/nhf_spatial_targets/fetch/modis.py
git commit -m "refactor: remove NotImplementedError catch now that consolidation is implemented"
```

---

### Task 6: Format, lint, final verification

**Step 1: Format and lint**

```bash
pixi run -e dev fmt && pixi run -e dev lint
```

Fix any issues.

**Step 2: Run full test suite**

```bash
pixi run -e dev test
```

Expected: All tests pass.

**Step 3: Final commit (if any formatting changes)**

```bash
git add -u
git commit -m "style: format and lint fixes"
```
