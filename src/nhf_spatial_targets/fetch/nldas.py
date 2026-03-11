"""Fetch NLDAS-2 land surface model outputs (MOSAIC and NOAH)."""

# TODO: implement via NASA GES DISC / Hydrology Data and Information Services Center
# NLDAS-2 data: https://ldas.gsfc.nasa.gov/nldas/nldas-2-model-data
# Access via: https://disc.gsfc.nasa.gov/ (requires NASA Earthdata login)
#
# Key variables:
#   MOSAIC: verify upper-layer soil moisture variable name
#   NOAH:   verify upper-layer soil moisture variable name


def fetch_nldas_mosaic(period: str, output_dir: str) -> None:
    """Download NLDAS-2 MOSAIC monthly soil moisture for the given period."""
    raise NotImplementedError


def fetch_nldas_noah(period: str, output_dir: str) -> None:
    """Download NLDAS-2 NOAH monthly soil moisture for the given period."""
    raise NotImplementedError
