"""Build runoff calibration targets from NHM-MWBM output."""

# Sources:  nhm_mwbm
# Method:   mwbm_uncertainty (pre-computed bounds from Bock et al. 2018)
# Variable: basin_cfs
# Timestep: monthly


def build(config: dict, fabric_path: str, output_path: str) -> None:
    """Build runoff target dataset."""
    raise NotImplementedError
