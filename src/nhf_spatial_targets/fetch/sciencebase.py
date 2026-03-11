"""Fetch datasets hosted on USGS ScienceBase (MWBM, Reitz recharge)."""

# TODO: implement via sciencebasepy
#   pip install sciencebasepy
#   https://github.com/usgs/sciencebasepy


def fetch_mwbm(sb_item_id: str, output_dir: str) -> None:
    """Download NHM-MWBM runoff + uncertainty output from ScienceBase."""
    raise NotImplementedError


def fetch_reitz2017(sb_item_id: str, output_dir: str) -> None:
    """Download Reitz et al. (2017) recharge estimates from ScienceBase."""
    raise NotImplementedError
