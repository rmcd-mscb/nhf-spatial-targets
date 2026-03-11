"""Build soil moisture calibration targets from MERRA, NCEP, NLDAS."""

# Sources:  merra_land (or merra2), ncep_ncar, nldas_mosaic, nldas_noah
# Method:   normalized_minmax (per calendar month for monthly)
# Variable: soil_rechr
# Timestep: monthly and annual


def build(config: dict, fabric_path: str, output_path: str) -> None:
    """Build soil moisture target dataset."""
    raise NotImplementedError
