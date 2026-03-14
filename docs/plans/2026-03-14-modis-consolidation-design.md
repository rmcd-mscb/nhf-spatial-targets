# MODIS Consolidation Design

**Goal:** Implement `consolidate_mod16a2` and `consolidate_mod10c1` — the two stubs in `fetch/consolidate.py` — to merge per-granule downloads into single consolidated NetCDF files per year.

## MOD16A2 (AET)

**Input:** `data/raw/mod16a2_v061/*.hdf` — sinusoidal-projected 500m tiles, multiple tiles per 8-day composite, ~46 composites/year.

**Pipeline per year:**

1. Glob HDF files, filter to requested year via `AYYYYDDD` filename pattern.
2. Group tiles by time step (same AYYYYDDD token = same 8-day composite).
3. For each time step:
   - Open tiles with rioxarray.
   - Mosaic into a single raster using `rasterio.merge`.
   - Reproject to EPSG:4326 at 0.04° resolution (~4km).
   - Resampling: **average** for `ET_500m`, **nearest neighbor** for `ET_QC_500m`.
4. Stack all time steps along a `time` dimension.
5. Select requested variables.
6. Write to `data/raw/mod16a2_v061/mod16a2_v061_{year}_consolidated.nc`.

**Output size:** ~500 MB/year (vs ~30 GB at native 500m).

**Rationale for 0.04° / ~4km:** Matches the typical scale of climate forcing used in NHM. 500m is overkill for area-weighted HRU aggregation. gdptools handles CRS reprojection internally via `to_crs()`, but clean 1D lat/lon coordinates make the consolidated files directly inspectable.

## MOD10C1 (Snow Cover)

**Input:** `data/raw/mod10c1_v061/*.conus.nc` — already subsetted to CONUS at 0.05° lat/lon, one file per day, ~365 files/year.

**Pipeline per year:**

1. Glob `.conus.nc` files, filter to requested year.
2. Open all files, concat along time, sort by time.
3. Select requested variables (`Day_CMG_Snow_Cover`, `Snow_Spatial_QA`).
4. Write to `data/raw/mod10c1_v061/mod10c1_v061_{year}_consolidated.nc`.

No mosaicking or reprojection needed. Daily resolution preserved (no temporal aggregation).

## Error Handling

- **Missing tiles for a time step:** Log warning, mosaic available tiles (partial coverage preferred over skipping).
- **Corrupt HDF files:** Use existing `_open_datasets` pattern — clean up and raise with clear message.
- **Existing consolidated file:** Overwrite (reconsolidation is idempotent).
- **No files for requested year:** Raise `FileNotFoundError`.
- **Empty year after filtering:** Return `None` so caller skips gracefully.

## Testing

- **MOD16A2:** Synthetic sinusoidal-projected arrays, verify grouping by time step, verify output has 1D lat/lon coords at 0.04°.
- **MOD10C1:** Synthetic `.conus.nc` files with time/lat/lon, verify concat and variable selection.
- **Both:** Provenance dict structure, overwrite behavior, error cases.
- **Integration tests:** Deferred to full integration test session with real data.

## Dependencies

- `rioxarray` — CRS-aware xarray operations
- `rasterio.merge` — tile mosaicking
- Both available via gdptools dependency chain (no new dependencies needed).
