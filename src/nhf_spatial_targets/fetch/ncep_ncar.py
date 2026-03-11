"""Fetch NCEP/NCAR Reanalysis soil moisture."""

# TODO: implement via NOAA PSL
# URL: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html
# Variable: soilw (0-10cm layer) — verify exact variable name
# Access: direct download of NetCDF files from PSL FTP/HTTPS


def fetch_ncep_ncar_soilm(period: str, output_dir: str) -> None:
    """Download NCEP/NCAR monthly soil moisture for the given period."""
    raise NotImplementedError
