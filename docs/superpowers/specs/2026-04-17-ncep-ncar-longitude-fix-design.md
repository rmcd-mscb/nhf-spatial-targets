# NCEP/NCAR Non-Monotonic Longitude Fix

**Date:** 2026-04-17
**Issue:** #54
**Status:** Design

## Problem

`nhf-targets agg ncep-ncar` fails with `KeyError: "Cannot get left slice bound for non-monotonic index"` because the NCEP/NCAR T62 Gaussian grid uses 0-360 longitude convention. When gdptools rotates values >180 by subtracting 360, the coordinate becomes non-monotonic (values wrap from ~358.125 → -1.875 at the seam), and xarray's `.sel(lon=slice(...))` raises a KeyError.

A secondary gdptools bug (in-place mutation of shared dataset coordinates) means the rotation corrupts the dataset for subsequent spatial batches even if the first batch succeeds. That bug will be reported upstream separately.

## Fix

Normalize longitude to -180..180 and sort in `consolidate_ncep_ncar()` so the consolidated file is correct on disk for all consumers.

### Location

`src/nhf_spatial_targets/fetch/consolidate.py`, inside `consolidate_ncep_ncar()`, after the `apply_cf_metadata()` call (line 1159) and before `_write_netcdf()` (line 1163).

### Change

```python
ds = apply_cf_metadata(ds, "ncep_ncar", "monthly")

# Normalize 0-360 longitude to -180..180 and sort for monotonic indexing
lon_name = next(
    (c for c in ds.coords if ds[c].attrs.get("axis") == "X"),
    "lon",
)
if float(ds[lon_name].max()) > 180:
    ds.coords[lon_name] = ((ds.coords[lon_name] + 180) % 360) - 180
    ds = ds.sortby(lon_name)

out_path = source_dir / "ncep_ncar_consolidated.nc"
```

The CF-detected coordinate name lookup via `axis="X"` is consistent with how `apply_cf_metadata()` sets the attribute. The guard `max > 180` avoids a no-op sort on data already in -180..180.

### Why not fix in the aggregation adapter?

- Leaves a known-bad file on disk that would trip up any non-gdptools consumer.
- The GLDAS fetch module already normalizes longitude at consolidation time (`fetch/gldas.py:90-94`), establishing a precedent.
- The consolidated file is the contract between fetch and aggregation; it should be self-consistent.

### Why not generalize in `apply_cf_metadata()`?

- Only NCEP/NCAR uses 0-360 in this project. Generalizing risks unintended effects on future sources that may legitimately use 0-360.
- If more sources need this, it can be promoted to `apply_cf_metadata()` later.

## Testing

Add a test in `tests/test_consolidate.py` that:

1. Creates synthetic NCEP/NCAR monthly files with 0-360 longitude (e.g., `np.arange(0, 360, 1.875)`)
2. Runs `consolidate_ncep_ncar()`
3. Asserts the output longitude is in -180..180 and monotonically increasing

## Scope

- 2 production lines + coordinate lookup in `consolidate.py`
- 1 new test in `test_consolidate.py`
- No changes to aggregation adapters, `_driver.py`, or `apply_cf_metadata()`

## Files Modified

| File | Change |
|------|--------|
| `src/nhf_spatial_targets/fetch/consolidate.py` | Normalize longitude after `apply_cf_metadata()` in `consolidate_ncep_ncar()` |
| `tests/test_consolidate.py` | Add test for 0-360 → -180..180 normalization |
