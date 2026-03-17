# CF-1.6 Compliance Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CF-1.6 compliant metadata to all consolidated NetCDF outputs via a shared `apply_cf_metadata()` helper, normalize coordinates to `lat`/`lon`, and add `time_bnds` for monthly data.

**Architecture:** A single `apply_cf_metadata()` function in `consolidate.py` handles all CF metadata: coordinate normalization, CRS variable, grid_mapping, variable attrs from catalog, coordinate attrs, time_bnds, and Conventions. Each consolidation function calls it before writing. Existing inline CF code in pangaea.py and reitz2017.py is replaced by the shared helper.

**Tech Stack:** xarray, numpy, pyproj, catalog/sources.yml, pytest

**Spec:** `docs/superpowers/specs/2026-03-17-cf-compliance-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `catalog/sources.yml` | Modify | Add `cf_units`, `long_name`, convert plain-string vars to dicts |
| `src/nhf_spatial_targets/fetch/consolidate.py` | Modify | Add `apply_cf_metadata()` helper; call it from 5 consolidation functions |
| `src/nhf_spatial_targets/fetch/pangaea.py` | Modify | Replace inline CF code in `_cf_fixup` with helper call |
| `src/nhf_spatial_targets/fetch/reitz2017.py` | Modify | Replace inline CF code in `_consolidate` with helper call (deferred if not on main) |
| `src/nhf_spatial_targets/fetch/modis.py` | Modify | Pass `source_key` to `consolidate_mod16a2_finalize` |
| `tests/test_consolidate.py` | Modify | Add `test_apply_cf_metadata` and CF assertions to existing tests |
| `tests/test_consolidate_modis.py` | Modify | Add CF assertions to MOD10C1 and MOD16A2 tests |
| `tests/test_pangaea.py` | Modify | Add CF assertions (lat/lon, time_bnds) |
| `tests/test_reitz2017.py` | Modify | Update CF assertions for lat/lon (deferred if not on main) |

---

## Chunk 1: Catalog Enrichment and Shared Helper

### Task 1: Enrich catalog/sources.yml with CF metadata

**Files:**
- Modify: `catalog/sources.yml`

This task adds `cf_units` and `long_name` to variable entries that need them, and converts plain-string variable lists to dict entries.

- [ ] **Step 1: Add `cf_units` to MERRA-2 variable dicts**

In `catalog/sources.yml`, add `cf_units: "1"` to each of the three MERRA-2 variable dicts (GWETTOP, GWETROOT, GWETPROF). The existing `units` field stays as human documentation.

```yaml
      - name: GWETTOP
        long_name: surface_soil_wetness
        layer_depth_m: "0.00-0.05"
        layer_depth_source: "dzsf in M2CONXLND; constant globally at 0.05m"
        units: dimensionless (fraction of saturation, 0-1)
        cf_units: "1"
        preferred: true
        notes: >
          Best variable for comparison with PRMS soil_rechr. Already
          dimensionless so no unit conversion before normalization.
```

Same pattern for GWETROOT and GWETPROF — add `cf_units: "1"` after the `units:` line.

- [ ] **Step 2: Convert MOD10C1 v061 variables to dicts**

Replace the plain-string variable list:
```yaml
    variables:
      - Day_CMG_Snow_Cover
      - Snow_Spatial_QA
```

With dict entries:
```yaml
    variables:
      - name: Day_CMG_Snow_Cover
        long_name: "daily snow-covered area"
        cf_units: "percent"
      - name: Snow_Spatial_QA
        long_name: "snow spatial QA confidence"
        cf_units: "percent"
```

- [ ] **Step 3: Convert MOD16A2 v061 variables to dicts**

Replace the plain-string variable list:
```yaml
    variables:
      - ET_500m
      - ET_QC_500m
```

With dict entries:
```yaml
    variables:
      - name: ET_500m
        long_name: "actual evapotranspiration"
        cf_units: "kg m-2"
        cell_methods: "time: sum"
      - name: ET_QC_500m
        long_name: "ET quality control"
        cf_units: "1"
```

- [ ] **Step 4: Update Reitz 2017 variables to match consolidated output**

Replace:
```yaml
    variables:
      - recharge
```

With:
```yaml
    variables:
      - name: total_recharge
        long_name: "Total recharge"
        cf_units: "inches/year"
      - name: eff_recharge
        long_name: "Effective recharge (base flow component)"
        cf_units: "inches/year"
```

- [ ] **Step 5: Run catalog tests**

Run: `pixi run -e dev test tests/test_catalog.py -v`
Expected: All catalog tests pass. The catalog loader accepts both dict and string variable entries.

- [ ] **Step 6: Commit**

```bash
git add catalog/sources.yml
git commit -m "catalog: add cf_units and long_name for CF-1.6 compliance (#27)"
```

---

### Task 2: Implement `apply_cf_metadata()` with unit test

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py`
- Modify: `tests/test_consolidate.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_consolidate.py`:

```python
def test_apply_cf_metadata_monthly():
    """apply_cf_metadata adds all CF-1.6 metadata for monthly data."""
    from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

    # Build a minimal dataset with y/x coords, spatial_ref, no CF metadata
    lat = np.arange(25.0, 50.0, 5.0)
    lon = np.arange(-125.0, -65.0, 10.0)
    time = pd.date_range("2010-01-15", periods=3, freq="MS")
    ds = xr.Dataset(
        {
            "SoilM_0_10cm": (
                ["time", "y", "x"],
                np.random.rand(3, len(lat), len(lon)).astype(np.float32),
            ),
            "spatial_ref": xr.DataArray(np.int32(0)),
        },
        coords={"time": time, "y": lat, "x": lon},
    )

    result = apply_cf_metadata(ds, "nldas_mosaic", "monthly")

    # Coordinates renamed to lat/lon
    assert "lat" in result.dims
    assert "lon" in result.dims
    assert "y" not in result.dims
    assert "x" not in result.dims

    # Dimension order
    assert result["SoilM_0_10cm"].dims == ("time", "lat", "lon")

    # CRS variable
    assert "crs" in result.data_vars
    assert result["crs"].attrs["grid_mapping_name"] == "latitude_longitude"
    assert result["crs"].attrs["semi_major_axis"] == pytest.approx(6378137.0)
    assert result["crs"].attrs["inverse_flattening"] == pytest.approx(298.257223563)
    assert "crs_wkt" in result["crs"].attrs

    # No spatial_ref
    assert "spatial_ref" not in result.data_vars
    assert "spatial_ref" not in result.coords

    # grid_mapping on data vars
    assert result["SoilM_0_10cm"].attrs["grid_mapping"] == "crs"

    # Variable metadata from catalog
    assert result["SoilM_0_10cm"].attrs["units"] == "kg/m2"
    assert result["SoilM_0_10cm"].attrs["long_name"] == "soil moisture 0-10 cm"

    # Coordinate attrs
    assert result.lat.attrs["standard_name"] == "latitude"
    assert result.lat.attrs["units"] == "degrees_north"
    assert result.lat.attrs["axis"] == "Y"
    assert result.lon.attrs["standard_name"] == "longitude"
    assert result.lon.attrs["units"] == "degrees_east"
    assert result.lon.attrs["axis"] == "X"
    assert result.time.attrs["standard_name"] == "time"
    assert result.time.attrs["axis"] == "T"

    # time_bnds for monthly
    assert "time_bnds" in result.data_vars
    assert result.time.attrs.get("bounds") == "time_bnds"

    # Conventions
    assert result.attrs["Conventions"] == "CF-1.6"


def test_apply_cf_metadata_daily_no_time_bnds():
    """apply_cf_metadata does not add time_bnds for daily data."""
    from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

    lat = np.linspace(25.0, 50.0, 4)
    lon = np.linspace(-125.0, -65.0, 6)
    time = pd.date_range("2010-01-01", periods=3, freq="D")
    ds = xr.Dataset(
        {
            "Day_CMG_Snow_Cover": (
                ["time", "lat", "lon"],
                np.random.rand(3, len(lat), len(lon)).astype(np.float32),
            ),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )

    result = apply_cf_metadata(ds, "mod10c1_v061", "daily")

    assert "time_bnds" not in result.data_vars
    assert "crs" in result.data_vars
    assert result.attrs["Conventions"] == "CF-1.6"


def test_apply_cf_metadata_skips_existing_time_bnds():
    """apply_cf_metadata skips time_bnds if already present (MERRA-2 case)."""
    from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

    lat = np.arange(-90, 91, 45.0)
    lon = np.arange(-180, 180, 60.0)
    time = pd.date_range("2010-01-15", periods=2, freq="MS")
    ds = xr.Dataset(
        {
            "GWETTOP": (
                ["time", "lat", "lon"],
                np.random.rand(2, len(lat), len(lon)).astype(np.float32),
            ),
            "time_bnds": (
                ["time", "nv"],
                np.array([[0, 31], [31, 59]], dtype="<i8"),
                {"units": "days since 1970-01-01", "calendar": "standard"},
            ),
        },
        coords={"time": time, "lat": lat, "lon": lon, "nv": [0, 1]},
    )

    result = apply_cf_metadata(ds, "merra2", "monthly")

    # Should keep existing time_bnds, not add a second one
    assert "time_bnds" in result.data_vars
    # Original values preserved
    np.testing.assert_array_equal(result["time_bnds"].values, np.array([[0, 31], [31, 59]]))


def test_apply_cf_metadata_custom_crs_wkt():
    """apply_cf_metadata uses pyproj to extract ellipsoid from custom CRS WKT."""
    from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

    lat = np.linspace(25.0, 50.0, 4)
    lon = np.linspace(-125.0, -65.0, 6)
    time = pd.date_range("2005-07-01", periods=2, freq="YS")
    ds = xr.Dataset(
        {
            "total_recharge": (
                ["time", "y", "x"],
                np.random.rand(2, len(lat), len(lon)).astype(np.float32),
            ),
        },
        coords={"time": time, "y": lat, "x": lon},
    )

    # NAD83 WKT
    from pyproj import CRS as _CRS
    nad83_wkt = _CRS.from_epsg(4269).to_wkt()

    result = apply_cf_metadata(ds, "reitz2017", "annual", crs_wkt=nad83_wkt)

    assert result["crs"].attrs["grid_mapping_name"] == "latitude_longitude"
    # NAD83 uses GRS 1980 ellipsoid
    assert result["crs"].attrs["inverse_flattening"] == pytest.approx(298.257222101)
    assert "NAD" in result["crs"].attrs["crs_wkt"]
    assert "time_bnds" not in result.data_vars
    # y/x renamed to lat/lon
    assert "lat" in result.dims
    assert "lon" in result.dims
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev test tests/test_consolidate.py::test_apply_cf_metadata_monthly -v`
Expected: FAIL with `ImportError` or `cannot import name 'apply_cf_metadata'`

- [ ] **Step 3: Implement `apply_cf_metadata()`**

Add to `src/nhf_spatial_targets/fetch/consolidate.py`, after the existing `_write_netcdf` function (around line 127):

```python
def apply_cf_metadata(
    ds: xr.Dataset,
    source_key: str,
    time_step: str = "monthly",
    crs_wkt: str | None = None,
) -> xr.Dataset:
    """Apply CF-1.6 compliant metadata to a consolidated dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to annotate. Callers must use the return value.
    source_key : str
        Catalog key for looking up variable metadata.
    time_step : str
        One of ``"monthly"``, ``"daily"``, ``"8-day"``, ``"annual"``.
        Controls whether ``time_bnds`` is added (monthly only).
    crs_wkt : str | None
        WKT string for the source CRS. Defaults to WGS84 when ``None``.

    Returns
    -------
    xr.Dataset
    """
    import nhf_spatial_targets.catalog as _catalog

    # 1. Normalize coordinates to lat/lon
    rename_map: dict[str, str] = {}
    for old, new in [("y", "lat"), ("x", "lon"), ("latitude", "lat"), ("longitude", "lon")]:
        if old in ds.dims and old != new:
            rename_map[old] = new
    if rename_map:
        ds = ds.rename(rename_map)

    # Ensure (time, lat, lon) dimension order
    dim_order = [d for d in ("time", "lat", "lon") if d in ds.dims]
    ds = ds.transpose(*dim_order)

    # 2. Drop spatial_ref if present
    if "spatial_ref" in ds:
        ds = ds.drop_vars("spatial_ref")
    # Also remove from coords if rioxarray left it there
    if "spatial_ref" in ds.coords:
        ds = ds.drop_vars("spatial_ref")

    # 3. Add CRS variable
    if crs_wkt is not None:
        from pyproj import CRS as _CRS

        src_crs = _CRS.from_wkt(crs_wkt)
        crs_attrs: dict = {"crs_wkt": crs_wkt}
        if src_crs.is_geographic:
            crs_attrs["grid_mapping_name"] = "latitude_longitude"
            ellipsoid = src_crs.ellipsoid
            crs_attrs["semi_major_axis"] = ellipsoid.semi_major_metre
            crs_attrs["inverse_flattening"] = ellipsoid.inverse_flattening
            crs_attrs["longitude_of_prime_meridian"] = 0.0
    else:
        # Default WGS84
        crs_attrs = {
            "grid_mapping_name": "latitude_longitude",
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
            "longitude_of_prime_meridian": 0.0,
            "crs_wkt": (
                'GEOGCS["WGS 84",'
                'DATUM["WGS_1984",'
                'SPHEROID["WGS 84",6378137,298.257223563]],'
                'PRIMEM["Greenwich",0],'
                'UNIT["degree",0.0174532925199433]]'
            ),
        }
    ds["crs"] = xr.DataArray(np.int32(0), attrs=crs_attrs)

    # 4. Set grid_mapping on data variables
    skip_vars = {"crs", "time_bnds"}
    for var in ds.data_vars:
        if var not in skip_vars:
            ds[var].attrs["grid_mapping"] = "crs"

    # 5. Set variable metadata from catalog
    try:
        meta = _catalog.source(source_key)
    except KeyError:
        logger.warning("Source '%s' not found in catalog; skipping variable metadata", source_key)
        meta = {}

    cat_vars = meta.get("variables", [])
    if cat_vars:
        # Build lookup: variable_name -> dict of attrs
        var_lookup: dict[str, dict] = {}
        for entry in cat_vars:
            if isinstance(entry, dict):
                name = entry.get("name", "")
                var_lookup[name] = entry
            else:
                var_lookup[str(entry)] = {}

        for var in ds.data_vars:
            if var in skip_vars:
                continue
            if var in var_lookup:
                entry = var_lookup[var]
                if "long_name" in entry:
                    ds[var].attrs["long_name"] = entry["long_name"]
                # cf_units takes precedence over units
                units = entry.get("cf_units") or entry.get("units")
                if units:
                    ds[var].attrs["units"] = units
                if "cell_methods" in entry:
                    ds[var].attrs["cell_methods"] = entry["cell_methods"]

    # 6. Set coordinate attributes
    if "lat" in ds.coords:
        ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"}
    if "lon" in ds.coords:
        ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}
    if "time" in ds.coords:
        ds.time.attrs.update({"standard_name": "time", "long_name": "time", "axis": "T"})

    # 7. Add time_bnds for monthly data
    if time_step == "monthly" and "time_bnds" not in ds:
        times = pd.DatetimeIndex(ds.time.values)
        epoch = pd.Timestamp("1970-01-01")
        bounds_list = []
        for t in times:
            m_start = t.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if t.month == 12:
                m_end = t.replace(
                    year=t.year + 1, month=1, day=1,
                    hour=0, minute=0, second=0, microsecond=0,
                )
            else:
                m_end = t.replace(
                    month=t.month + 1, day=1,
                    hour=0, minute=0, second=0, microsecond=0,
                )
            bounds_list.append([(m_start - epoch).days, (m_end - epoch).days])

        nv = np.array([0, 1])
        ds["time_bnds"] = xr.DataArray(
            np.array(bounds_list, dtype="<i8"),
            dims=["time", "nv"],
            attrs={"units": "days since 1970-01-01", "calendar": "standard"},
        )
        if "nv" not in ds.coords:
            ds = ds.assign_coords(nv=nv)
        ds.time.attrs["bounds"] = "time_bnds"

    # 8. Set Conventions
    ds.attrs["Conventions"] = "CF-1.6"
    ds.attrs.pop("conventions", None)  # remove stale lowercase variant

    return ds
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev test tests/test_consolidate.py -k "test_apply_cf" -v`
Expected: All 4 new tests PASS

- [ ] **Step 5: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All tests pass, no lint errors

- [ ] **Step 6: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate.py
git commit -m "feat: add apply_cf_metadata() shared helper (#27)"
```

---

## Chunk 2: Integrate into consolidate.py Functions

### Task 3: Integrate into `consolidate_merra2`

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py`
- Modify: `tests/test_consolidate.py`

- [ ] **Step 1: Add CF assertions to existing test**

Add to `test_global_attributes` in `tests/test_consolidate.py`, replacing the `CF-1.8` assertion:

```python
def test_global_attributes(merra2_dir):
    """Consolidated file has CF and provenance global attributes."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_merra2

    run_dir = merra2_dir.parent.parent.parent
    consolidate_merra2(run_dir=run_dir, variables=["GWETTOP"])

    ds = xr.open_dataset(merra2_dir / "merra2_consolidated.nc")
    # CF-1.6 compliance
    assert ds.attrs["Conventions"] == "CF-1.6"
    assert "crs" in ds.data_vars
    assert "spatial_ref" not in ds.data_vars
    assert ds["crs"].attrs["grid_mapping_name"] == "latitude_longitude"
    assert ds["GWETTOP"].attrs["grid_mapping"] == "crs"
    assert ds["GWETTOP"].attrs["units"] == "1"
    assert ds["GWETTOP"].attrs["long_name"] == "surface_soil_wetness"
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert ds.time.attrs["standard_name"] == "time"
    # Provenance attrs preserved
    assert "nhf-spatial-targets" in ds.attrs["history"]
    assert "M2TMNXLND" in ds.attrs["source"]
    assert "time_modification_note" in ds.attrs
    assert "references" in ds.attrs
    ds.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev test tests/test_consolidate.py::test_global_attributes -v`
Expected: FAIL — `assert ds.attrs["Conventions"] == "CF-1.6"` fails (currently `"CF-1.8"`)

- [ ] **Step 3: Add `apply_cf_metadata` call to `consolidate_merra2`**

In `consolidate_merra2`, after the `ds.attrs.update({...})` block (around line 252), replace `"Conventions": "CF-1.8"` with a call to `apply_cf_metadata`. The call goes after `_fix_time_merra2` (which creates `time_bnds`) and before `_write_netcdf`:

```python
        ds = _fix_time_merra2(ds)
        ds = apply_cf_metadata(ds, "merra2", "monthly")

        # Add provenance global attributes (after apply_cf_metadata sets Conventions)
        meta = _catalog.source("merra2")
        ds.attrs.update(
            {
                "history": (f"Consolidated by nhf-spatial-targets v{__version__}"),
                "source": (
                    f"NASA MERRA-2 {meta['access']['short_name']}"
                    f" v{meta['access'].get('version', 'unknown')}"
                ),
                "time_modification_note": (
                    "Original timestamps (YYYY-MM-01T00:30:00) shifted to mid-month "
                    "(15th) for consistency. See time_bnds for exact averaging periods."
                ),
                "references": meta["access"]["url"],
            }
        )
```

Remove `"Conventions": "CF-1.8"` from the existing `ds.attrs.update` dict. Keep the `import nhf_spatial_targets.catalog as _catalog` — the provenance block still uses `_catalog.source("merra2")` directly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev test tests/test_consolidate.py -v`
Expected: All tests pass including updated `test_global_attributes`

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate.py
git commit -m "feat: integrate apply_cf_metadata into consolidate_merra2 (#27)"
```

---

### Task 4: Integrate into `consolidate_nldas`

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py`
- Modify: `tests/test_consolidate.py`

- [ ] **Step 1: Add CF assertion test for NLDAS**

Add to `tests/test_consolidate.py`:

```python
def test_nldas_cf_metadata(nldas_dir):
    """NLDAS consolidated file has CF-1.6 metadata."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_nldas

    run_dir = nldas_dir.parent.parent.parent
    consolidate_nldas(
        run_dir=run_dir,
        source_key="nldas_mosaic",
        variables=["SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_200cm"],
    )

    ds = xr.open_dataset(nldas_dir / "nldas_mosaic_consolidated.nc")
    assert ds.attrs["Conventions"] == "CF-1.6"
    assert "crs" in ds.data_vars
    assert "spatial_ref" not in ds.data_vars
    assert ds["crs"].attrs["grid_mapping_name"] == "latitude_longitude"
    assert ds["SoilM_0_10cm"].attrs["grid_mapping"] == "crs"
    assert ds["SoilM_0_10cm"].attrs["units"] == "kg/m2"
    assert ds["SoilM_0_10cm"].attrs["long_name"] == "soil moisture 0-10 cm"
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert ds.time.attrs["standard_name"] == "time"
    assert "time_bnds" in ds.data_vars
    assert ds.time.attrs.get("bounds") == "time_bnds"
    ds.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev test tests/test_consolidate.py::test_nldas_cf_metadata -v`
Expected: FAIL

- [ ] **Step 3: Add `apply_cf_metadata` call to `consolidate_nldas`**

In `consolidate_nldas`, after `ds = ds[variables]` and before `_write_netcdf`:

```python
        ds = ds[variables]
        ds = apply_cf_metadata(ds, source_key, "monthly")

        out_path = source_dir / f"{source_key}_consolidated.nc"
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev test tests/test_consolidate.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate.py
git commit -m "feat: integrate apply_cf_metadata into consolidate_nldas (#27)"
```

---

### Task 5: Integrate into `consolidate_ncep_ncar`

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py`
- Modify: `tests/test_consolidate.py`

- [ ] **Step 1: Add CF assertion test**

Add to `tests/test_consolidate.py`:

```python
def test_ncep_cf_metadata(ncep_dir):
    """NCEP/NCAR consolidated file has CF-1.6 metadata."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_ncep_ncar

    run_dir = ncep_dir.parent.parent.parent
    consolidate_ncep_ncar(run_dir=run_dir, variables=["soilw"])

    ds = xr.open_dataset(ncep_dir / "ncep_ncar_consolidated.nc")
    assert ds.attrs["Conventions"] == "CF-1.6"
    assert "crs" in ds.data_vars
    assert "spatial_ref" not in ds.data_vars
    assert ds["soilw"].attrs["grid_mapping"] == "crs"
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert "time_bnds" in ds.data_vars
    ds.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev test tests/test_consolidate.py::test_ncep_cf_metadata -v`
Expected: FAIL

- [ ] **Step 3: Add `apply_cf_metadata` call to `consolidate_ncep_ncar`**

In `consolidate_ncep_ncar`, after `ds = ds[variables]` and before `_write_netcdf`:

```python
        ds = ds[variables]
        ds = apply_cf_metadata(ds, "ncep_ncar", "monthly")

        out_path = ncep_dir / "ncep_ncar_consolidated.nc"
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev test tests/test_consolidate.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate.py
git commit -m "feat: integrate apply_cf_metadata into consolidate_ncep_ncar (#27)"
```

---

### Task 6: Integrate into `consolidate_mod10c1`

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py`
- Modify: `tests/test_consolidate_modis.py`

- [ ] **Step 1: Add CF assertion test**

Add to `tests/test_consolidate_modis.py`:

```python
def test_consolidate_mod10c1_cf_metadata(mod10c1_run_dir: Path) -> None:
    """MOD10C1 consolidated file has CF-1.6 metadata."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod10c1

    source_key = "mod10c1_v061"
    consolidate_mod10c1(
        run_dir=mod10c1_run_dir,
        source_key=source_key,
        variables=["Day_CMG_Snow_Cover", "Snow_Spatial_QA"],
        year=2010,
    )

    out_path = (
        mod10c1_run_dir / "data" / "raw" / source_key
        / f"{source_key}_2010_consolidated.nc"
    )
    ds = xr.open_dataset(out_path)
    assert ds.attrs["Conventions"] == "CF-1.6"
    assert "crs" in ds.data_vars
    assert "spatial_ref" not in ds.data_vars
    assert ds["Day_CMG_Snow_Cover"].attrs["grid_mapping"] == "crs"
    assert ds["Day_CMG_Snow_Cover"].attrs["units"] == "percent"
    assert ds["Day_CMG_Snow_Cover"].attrs["long_name"] == "daily snow-covered area"
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert "time_bnds" not in ds.data_vars  # daily data, no time_bnds
    ds.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev test tests/test_consolidate_modis.py::test_consolidate_mod10c1_cf_metadata -v`
Expected: FAIL

- [ ] **Step 3: Add `apply_cf_metadata` call to `consolidate_mod10c1`**

In `consolidate_mod10c1`, after `ds_merged = ds_merged[variables]` and before `_write_netcdf`:

```python
        ds_merged = ds_merged[variables]
        ds_merged = apply_cf_metadata(ds_merged, source_key, "daily")

        out_path = source_dir / f"{source_key}_{year}_consolidated.nc"
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev test tests/test_consolidate_modis.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py tests/test_consolidate_modis.py
git commit -m "feat: integrate apply_cf_metadata into consolidate_mod10c1 (#27)"
```

---

### Task 7: Integrate into `consolidate_mod16a2_finalize`

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/consolidate.py`
- Modify: `src/nhf_spatial_targets/fetch/modis.py`
- Modify: `tests/test_consolidate_modis.py`

This task is more involved: add `source_key` parameter to `consolidate_mod16a2_finalize`, remove inline `spatial_ref` handling, add `apply_cf_metadata` call, and update callers.

- [ ] **Step 1: Add CF assertion test**

Add to `tests/test_consolidate_modis.py`:

```python
def test_consolidate_mod16a2_cf_metadata(mod16a2_run_dir: Path) -> None:
    """MOD16A2 consolidated file has CF-1.6 metadata with crs (not spatial_ref)."""
    from nhf_spatial_targets.fetch.consolidate import consolidate_mod16a2

    source_key = "mod16a2_v061"
    result = consolidate_mod16a2(
        run_dir=mod16a2_run_dir,
        source_key=source_key,
        variables=["ET_500m"],
        year=2010,
        bbox=_TEST_BBOX,
    )

    out_path = mod16a2_run_dir / result["consolidated_nc"]
    ds = xr.open_dataset(out_path)
    assert ds.attrs["Conventions"] == "CF-1.6"
    assert "crs" in ds.data_vars
    assert "spatial_ref" not in ds.data_vars
    assert "spatial_ref" not in ds.coords
    assert ds["ET_500m"].attrs["grid_mapping"] == "crs"
    assert ds["ET_500m"].attrs["units"] == "kg m-2"
    assert ds["ET_500m"].attrs["long_name"] == "actual evapotranspiration"
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    ds.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev test tests/test_consolidate_modis.py::test_consolidate_mod16a2_cf_metadata -v`
Expected: FAIL

- [ ] **Step 3: Add `source_key` parameter to `consolidate_mod16a2_finalize`**

Update the function signature:

```python
def consolidate_mod16a2_finalize(
    tmp_paths: list[Path],
    variables: list[str],
    out_path: Path,
    run_dir: Path,
    source_key: str = "mod16a2_v061",
    keep_tmp: bool = False,
) -> dict:
```

Replace the `spatial_ref` handling block (lines 643-651) with:

```python
            ds = ds.sortby("time")
            _validate_variables(ds, variables)
            ds = ds[variables]
            ds = apply_cf_metadata(ds, source_key, "8-day")
            _write_netcdf(ds, out_path)
```

- [ ] **Step 4: Pass `source_key` from `consolidate_mod16a2`**

In `consolidate_mod16a2`, update the call to `consolidate_mod16a2_finalize`:

```python
    result = consolidate_mod16a2_finalize(
        tmp_paths=tmp_paths,
        variables=variables,
        out_path=out_path,
        run_dir=run_dir,
        source_key=source_key,
    )
```

- [ ] **Step 5: Pass `source_key` from `modis.py` direct call**

In `src/nhf_spatial_targets/fetch/modis.py`, find the direct call to `consolidate_mod16a2_finalize` (around line 497) and add `source_key=source_key`:

```python
                result = consolidate_mod16a2_finalize(
                    tmp_paths=tmp_paths,
                    variables=variables,
                    out_path=out_path,
                    run_dir=run_dir,
                    source_key=source_key,
                )
```

- [ ] **Step 6: Update existing `test_consolidate_mod16a2_finalize_concats_and_cleans` test**

The existing test calls `consolidate_mod16a2_finalize` without `source_key`. Since we added a default, the existing test should still work. However, verify by running.

- [ ] **Step 7: Run tests**

Run: `pixi run -e dev test tests/test_consolidate_modis.py -v`
Expected: All pass

- [ ] **Step 8: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All 200+ tests pass

- [ ] **Step 9: Commit**

```bash
git add src/nhf_spatial_targets/fetch/consolidate.py src/nhf_spatial_targets/fetch/modis.py tests/test_consolidate_modis.py
git commit -m "feat: integrate apply_cf_metadata into consolidate_mod16a2 (#27)"
```

---

## Chunk 3: Refactor pangaea.py and reitz2017.py

### Task 8: Refactor `pangaea.py` to use shared helper

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/pangaea.py`
- Modify: `tests/test_pangaea.py`

- [ ] **Step 1: Add CF assertion test for lat/lon and time_bnds**

Add to `tests/test_pangaea.py`:

```python
def test_cf_fixup_coordinate_metadata(tmp_path: Path):
    """CF fix-up sets coordinate attrs and time_bnds for monthly data."""
    from nhf_spatial_targets.fetch.pangaea import _cf_fixup

    raw = _make_watergap_nc(tmp_path / "raw.nc4", n_times=3)
    fixed = _cf_fixup(raw, tmp_path / "fixed.nc")

    ds = xr.open_dataset(fixed)
    # Coordinate names
    assert "lat" in ds.dims
    assert "lon" in ds.dims
    # Coordinate attrs
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lat.attrs["units"] == "degrees_north"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert ds.lon.attrs["units"] == "degrees_east"
    assert ds.time.attrs["standard_name"] == "time"
    # time_bnds for monthly data
    assert "time_bnds" in ds.data_vars
    assert ds.time.attrs.get("bounds") == "time_bnds"
    # No spatial_ref
    assert "spatial_ref" not in ds.data_vars
    assert "spatial_ref" not in ds.coords
    ds.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev test tests/test_pangaea.py::test_cf_fixup_coordinate_metadata -v`
Expected: FAIL — coordinate attrs and time_bnds not yet added

- [ ] **Step 3: Replace inline CF code in `_cf_fixup` with helper call**

In `src/nhf_spatial_targets/fetch/pangaea.py`, replace the CRS/grid_mapping/Conventions block in `_cf_fixup` (lines 65-93) with:

```python
        # --- Apply CF-1.6 metadata via shared helper ---
        from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

        ds = apply_cf_metadata(ds, "watergap22d", "monthly")
```

Keep the time reconstruction logic above it (lines 51-63) unchanged. Remove the inline CRS creation, grid_mapping loop, and Conventions lines.

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev test tests/test_pangaea.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/nhf_spatial_targets/fetch/pangaea.py tests/test_pangaea.py
git commit -m "refactor: replace inline CF code in pangaea.py with shared helper (#27)"
```

---

### Task 9: Refactor `reitz2017.py` to use shared helper (conditional)

**Precondition:** `reitz2017.py` must exist on the current branch. If not yet merged from `feature/reitz2017-fetch`, skip this task.

**Files:**
- Modify: `src/nhf_spatial_targets/fetch/reitz2017.py`
- Modify: `tests/test_reitz2017.py`

- [ ] **Step 1: Check if reitz2017.py exists**

Run: `ls src/nhf_spatial_targets/fetch/reitz2017.py`
If file not found, skip this entire task.

- [ ] **Step 2: Update test assertions for lat/lon**

In `tests/test_reitz2017.py`, find CF assertions that check for `y`/`x` and update to `lat`/`lon`:

```python
    # Replace:  assert ds.y.attrs["standard_name"] == "latitude"
    # With:     assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lat.attrs["units"] == "degrees_north"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert ds.lon.attrs["units"] == "degrees_east"
```

- [ ] **Step 3: Replace inline CF code in `_consolidate` with helper call**

In `reitz2017.py`, replace the entire CF metadata section (from `# Drop rioxarray's spatial_ref` through `ds.attrs["Conventions"] = "CF-1.6"`) with:

```python
    from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

    ds = apply_cf_metadata(ds, "reitz2017", "annual", crs_wkt=src_crs_wkt)
```

Keep the `src_crs_wkt` capture from GeoTIFFs and the existing variable stacking code.

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev test tests/test_reitz2017.py -v`
Expected: All pass

- [ ] **Step 5: Run full suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/nhf_spatial_targets/fetch/reitz2017.py tests/test_reitz2017.py
git commit -m "refactor: replace inline CF code in reitz2017.py with shared helper (#27)"
```

---

### Task 10: Final verification

- [ ] **Step 1: Run full test suite**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: All tests pass, no lint errors

- [ ] **Step 2: Verify no remaining inline CF code**

Search for old patterns that should no longer exist:

```bash
pixi run -e dev python -c "
import subprocess
# Check for inline Conventions settings outside consolidate.py
result = subprocess.run(
    ['grep', '-rn', 'Conventions.*CF', 'src/nhf_spatial_targets/fetch/'],
    capture_output=True, text=True
)
print(result.stdout)
"
```

Expected: Only `consolidate.py:apply_cf_metadata` should set Conventions. No hits in `pangaea.py` or `reitz2017.py`.

- [ ] **Step 3: Verify no remaining spatial_ref usage**

```bash
pixi run -e dev python -c "
import subprocess
result = subprocess.run(
    ['grep', '-rn', 'spatial_ref', 'src/nhf_spatial_targets/fetch/consolidate.py'],
    capture_output=True, text=True
)
print(result.stdout or 'No spatial_ref references - good!')
"
```

Expected: No hits (helper drops `spatial_ref`, doesn't create it)
