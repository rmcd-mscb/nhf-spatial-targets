"""Spatial aggregation of gridded source data to HRU polygons via gdptools."""

# TODO: implement area-weighted aggregation using gdptools
# gdptools: https://github.com/rmcd-mscb/gdptools
#
# General pattern:
#   1. Load HRU fabric GeoDataFrame (GeoPackage)
#   2. Load source raster/NetCDF as xarray Dataset
#   3. Use gdptools WeightGen to compute area-weighted intersections
#   4. Apply weights to aggregate source variable to HRU polygons
#   5. Return xarray Dataset indexed by nhm_id and time


def aggregate_to_fabric(
    source_ds,         # xarray.Dataset: gridded source data
    fabric_path: str,  # path to GeoPackage fabric
    id_col: str,       # HRU ID column name
    variable: str,     # variable name in source_ds
    method: str = "area_weighted",
) -> object:           # xarray.Dataset indexed by id_col
    """Aggregate gridded source data to HRU polygons."""
    raise NotImplementedError
