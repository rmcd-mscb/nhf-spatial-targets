"""Build snow-covered area calibration targets from MOD10C1."""

# Sources:  mod10c1 (or v061)
# Method:   modis_ci (CI-based bounds, threshold 70%)
# Variable: snowcov_area
# Timestep: daily


def build(config: dict, fabric_path: str, output_path: str) -> None:
    """Build SCA target dataset."""
    raise NotImplementedError
