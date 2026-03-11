"""Fetch SSEBop actual evapotranspiration."""

# TODO: verify current SSEBop product version and access endpoint
# Historical access: https://earlywarning.usgs.gov/ssebop/
# May also be available via ScienceBase or USGS Water Resources data portal


def fetch_ssebop(period: str, output_dir: str) -> None:
    """Download SSEBop monthly AET for the given period."""
    raise NotImplementedError
