"""Fetch MERRA-2 soil moisture (replacement for MERRA-Land)."""

# TODO: implement via NASA GES DISC
# Product: M2TMNXLND (tavg1_2d_lnd_Nx, monthly mean)
# URL: https://disc.gsfc.nasa.gov/datasets/M2TMNXLND_5.12.4/summary
# Access via earthaccess or OPeNDAP
#
# Key variable: SFMC (surface soil moisture, kg/m2)
# Verify appropriate layer depth vs PRMS soil_rechr


def fetch_merra2_soilm(period: str, output_dir: str) -> None:
    """Download MERRA-2 monthly soil moisture for the given period."""
    raise NotImplementedError
