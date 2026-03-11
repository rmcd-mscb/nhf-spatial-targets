"""Fetch MODIS products: MOD16A2 (AET) and MOD10C1 (SCA)."""

# TODO: implement via earthaccess or pydap
# earthaccess is the recommended NASA EDL access library:
#   https://github.com/nsidc/earthaccess
#
# MOD16A2 v061: https://lpdaac.usgs.gov/products/mod16a2v061/
# MOD10C1 v061: https://nsidc.org/data/mod10c1/versions/61


def fetch_mod16a2(period: str, output_dir: str) -> None:
    """Download MOD16A2 AET tiles for CONUS for the given period."""
    raise NotImplementedError


def fetch_mod10c1(period: str, output_dir: str) -> None:
    """Download MOD10C1 daily snow cover CMG files for the given period."""
    raise NotImplementedError
