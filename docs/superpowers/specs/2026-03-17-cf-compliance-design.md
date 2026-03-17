# CF-1.6 Compliance for All Consolidated NetCDF Outputs

**Goal:** Add CF-1.6 compliant metadata to all consolidated NetCDF outputs via a shared helper, normalize coordinates to `lat`/`lon`, and add `time_bnds` for monthly data.

**Issue:** #27

**Architecture:** A single `apply_cf_metadata()` function in `consolidate.py` that all 7 consolidation paths call before writing. The helper reads variable metadata from the catalog, normalizes coordinates, adds CRS/grid_mapping/coordinate attrs, and sets Conventions. Modules that currently have inline CF code (pangaea.py, reitz2017.py) are refactored to use the shared helper.

**Tech Stack:** xarray, numpy, pyproj (for CRS WKT), catalog/sources.yml

**Note:** `reitz2017.py` exists on branch `feature/reitz2017-fetch` (PR pending). If not yet merged when implementation begins, the reitz2017 refactoring task is deferred until after merge.

---

## 1. Shared Helper: `apply_cf_metadata()`

### Location

`src/nhf_spatial_targets/fetch/consolidate.py`

### Signature

```python
def apply_cf_metadata(
    ds: xr.Dataset,
    source_key: str,
    time_step: str = "monthly",
    crs_wkt: str | None = None,
) -> xr.Dataset:
```

### Parameters

- `ds` — Dataset to modify. Some xarray operations (transpose, drop_vars, assign_coords) return new objects, so callers must always use the return value: `ds = apply_cf_metadata(ds, ...)`.
- `source_key` — Catalog key (e.g. `"merra2"`, `"nldas_mosaic"`). Used to look up variable `units`/`long_name` from `catalog/sources.yml`.
- `time_step` — One of `"monthly"`, `"daily"`, `"8-day"`, `"annual"`. Controls whether `time_bnds` is added (monthly only).
- `crs_wkt` — Optional WKT string for the source CRS. When `None`, defaults to WGS84 (EPSG:4326). When provided (e.g., by reitz2017 for NAD83), uses `pyproj.CRS.from_wkt(crs_wkt)` to derive ellipsoid params.

### Behavior

1. **Normalize coordinates to `lat`/`lon`:**
   - Detect `y`/`x`, `latitude`/`longitude`, or `lat`/`lon` and rename to `lat`/`lon`.
   - Ensure dimension order is `(time, lat, lon)` via `ds.transpose()`.

2. **Drop `spatial_ref` if present** (rioxarray artifact that doesn't survive NetCDF round-trips).

3. **Add CRS variable:**
   - All modules produce WGS84 (EPSG:4326) geographic data after download/reprojection.
   - Exception: Reitz 2017 source GeoTIFFs are NAD83 (EPSG:4269) — pass `crs_wkt` to preserve source datum.
   - When `crs_wkt` is `None`: uses hardcoded WGS84 WKT and ellipsoid params (semi_major_axis=6378137.0, inverse_flattening=298.257223563).
   - When `crs_wkt` is provided: uses `pyproj.CRS.from_wkt(crs_wkt)` to extract ellipsoid params.
   - Creates `crs` DataArray with `grid_mapping_name`, `semi_major_axis`, `inverse_flattening`, `longitude_of_prime_meridian`, `crs_wkt`.

4. **Set `grid_mapping = "crs"`** on all data variables (excluding `crs`, `time_bnds`).

5. **Set variable metadata from catalog:**
   - Calls `catalog.source(source_key)` and iterates the `variables` list.
   - If the source has no `variables` key, logs a warning and skips this step.
   - For dict entries: matches on `name` field against dataset data var names.
   - For plain string entries: matches on the string value.
   - Applies `long_name`, `cf_units` (falling back to `units`), and `cell_methods` (when present) to matching data vars.
   - The `units` field in the catalog is human-readable documentation (may include parenthetical annotations). The `cf_units` field is the machine-readable UDUNITS-2 compatible string used for CF metadata. When both exist, `cf_units` takes precedence.
   - Silently skips data vars not found in catalog (e.g., `crs`, `time_bnds`).

6. **Set coordinate attributes:**
   - `lat`: `standard_name="latitude"`, `units="degrees_north"`, `axis="Y"`
   - `lon`: `standard_name="longitude"`, `units="degrees_east"`, `axis="X"`
   - `time`: `standard_name="time"`, `long_name="time"`, `axis="T"`

7. **Add `time_bnds` for monthly data** (when `time_step == "monthly"`):
   - Skip if `time_bnds` already exists in the dataset (MERRA-2 creates them in `_fix_time_merra2`).
   - Build bounds as `[first-of-month, first-of-next-month)` in days since 1970-01-01.
   - Use `nv` as the bounds dimension name (matching MERRA-2's existing convention).
   - Add `bounds="time_bnds"` attr to time coordinate.

8. **Set `Conventions = "CF-1.6"`**. Remove any stale `conventions` (lowercase) attr.

---

## 2. Integration into Each Module

### `consolidate.py` — 5 functions

| Function | Call | Notes |
|----------|------|-------|
| `consolidate_merra2` | `ds = apply_cf_metadata(ds, "merra2", "monthly")` | Remove inline `Conventions: "CF-1.8"`. Keep `_fix_time_merra2` (helper skips `time_bnds` since they already exist). |
| `consolidate_nldas` | `ds = apply_cf_metadata(ds, source_key, "monthly")` | `source_key` is already a parameter (`"nldas_mosaic"` or `"nldas_noah"`). |
| `consolidate_ncep_ncar` | `ds = apply_cf_metadata(ds, "ncep_ncar", "monthly")` | |
| `consolidate_mod10c1` | `ds = apply_cf_metadata(ds, source_key, "daily")` | No `time_bnds` for daily. |
| `consolidate_mod16a2_finalize` | `ds = apply_cf_metadata(ds, source_key, "8-day")` | Delete lines 643-651 (the `spatial_ref` keep-list and `grid_mapping = "spatial_ref"` restoration block). The helper drops `spatial_ref`, creates `crs`, and sets `grid_mapping = "crs"`. Requires passing `source_key` through (currently not a parameter). |

**`consolidate_mod16a2_finalize` change:** Add `source_key: str` parameter. Update callers (`consolidate_mod16a2` and `modis.py:fetch_mod16a2`) to pass it.

### `pangaea.py` — refactor `_cf_fixup`

Replace inline CRS variable creation, `grid_mapping` loop, and `Conventions` setting with:

```python
ds = apply_cf_metadata(ds, "watergap22d", "monthly")
```

Keep the PANGAEA-specific time reconstruction logic (months-since-1901 decoding) — that stays in `_cf_fixup`.

### `reitz2017.py` — refactor `_consolidate`

Replace inline CRS, grid_mapping, coordinate attrs, variable attrs, and Conventions code with:

```python
ds = apply_cf_metadata(ds, "reitz2017", "annual", crs_wkt=src_crs_wkt)
```

Keep the existing `src_crs_wkt` capture from GeoTIFFs. The helper renames `y`/`x` to `lat`/`lon`.

---

## 3. Catalog Enrichment

### Changes to `catalog/sources.yml`

Convert plain-string variable entries to dicts with `name`, `long_name`, and `cf_units`. Add `cf_units` where `units` has parenthetical annotations.

**MOD10C1 v061:**
```yaml
variables:
  - name: Day_CMG_Snow_Cover
    long_name: "daily snow-covered area"
    cf_units: "percent"
  - name: Snow_Spatial_QA
    long_name: "snow spatial QA confidence"
    cf_units: "percent"
```

**MOD16A2 v061:**
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

Note: MOD16A2 ET is accumulated over the 8-day compositing period. The CF-valid unit is `kg m-2` (mass per area) with `cell_methods: "time: sum"` to indicate temporal accumulation.

**Reitz 2017** — update variables to match the dataset variable names used in consolidated output. The current catalog has `variables: [recharge]` which is the calibration variable name from `variables.yml`. The consolidated NetCDF contains two data variables: `total_recharge` and `eff_recharge`. The catalog should reflect the actual file contents. Downstream code in `variables.yml` references `reitz2017` by source key, not by individual variable names, so this change does not break the pipeline.

```yaml
variables:
  - name: total_recharge
    long_name: "Total recharge"
    cf_units: "inches/year"
  - name: eff_recharge
    long_name: "Effective recharge (base flow component)"
    cf_units: "inches/year"
```

**MERRA-2** — add `cf_units` to existing variable dicts:
```yaml
- name: GWETTOP
  cf_units: "1"
  # ... existing fields unchanged
```

(Same for GWETROOT, GWETPROF.)

**NCEP/NCAR** — already has `units: kg/m2` which is clean CF. No change needed.

**NLDAS MOSAIC/NOAH** — already has `units: kg/m2` which is clean CF. No change needed.

**WaterGAP 2.2d** — already has `units: "kg m-2 s-1"` which is clean CF. No change needed.

---

## 4. Testing

### Unit test for the helper (`test_consolidate.py`)

`test_apply_cf_metadata` — construct a minimal `xr.Dataset` with `y`/`x` coordinates, no CF metadata, and a mock source key. Call `apply_cf_metadata`. Verify:
- Coordinates renamed to `lat`/`lon`
- Dimension order is `(time, lat, lon)`
- `crs` variable present with expected attrs
- `spatial_ref` absent
- `grid_mapping = "crs"` on data vars
- `Conventions = "CF-1.6"`
- Coordinate attrs set correctly
- `time_bnds` present for monthly, absent for daily/annual

### Existing consolidation tests — add CF assertions

Each module's existing test that exercises consolidation gets additional assertions:
- `test_consolidate.py`: MERRA-2, NLDAS, NCEP/NCAR, MOD10C1, MOD16A2 consolidation tests
- `test_reitz2017.py`: Update existing CF assertions, verify `lat`/`lon` instead of `y`/`x`
- `test_pangaea.py`: Add CF assertions, verify `lat`/`lon`, verify `time_bnds`

### What tests verify per module

| Check | MERRA-2 | NLDAS | NCEP | MOD10C1 | MOD16A2 | WaterGAP | Reitz |
|-------|---------|-------|------|---------|---------|----------|-------|
| `crs` variable | X | X | X | X | X | X | X |
| No `spatial_ref` | X | X | X | X | X | X | X |
| `grid_mapping` on vars | X | X | X | X | X | X | X |
| `units` on vars | X | X | X | X | X | X | X |
| `long_name` on vars | X | X | X | X | X | X | X |
| `lat`/`lon` names | X | X | X | X | X | X | X |
| Coordinate attrs | X | X | X | X | X | X | X |
| `Conventions = "CF-1.6"` | X | X | X | X | X | X | X |
| `time_bnds` | X | X | X | | | X | |

**Note on timestamps:** The `apply_cf_metadata` helper does not modify timestamp values — only MERRA-2's `_fix_time_merra2` shifts timestamps to mid-month. WaterGAP timestamps remain at 1st-of-month (as reconstructed from PANGAEA's offset encoding). Standardizing timestamp conventions across all monthly data is out of scope for this issue.

---

## 5. Scope Boundaries

**In scope:**
- Shared `apply_cf_metadata()` in `consolidate.py`
- Refactor all 7 consolidation paths to use it
- Catalog enrichment (`cf_units`, `long_name`, convert plain strings to dicts)
- Coordinate normalization to `lat`/`lon` with `(time, lat, lon)` ordering
- `time_bnds` for all monthly datasets
- Standardize on CF-1.6 (MERRA-2 changes from 1.8)
- Test updates for all modules

**Out of scope:**
- `cfchecks` or formal CF validation tooling
- Download/fetch logic changes
- NetCDF encoding or compression changes
- Aggregation or downstream pipeline changes
