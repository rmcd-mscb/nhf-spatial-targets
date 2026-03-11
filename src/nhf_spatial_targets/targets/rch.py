"""Build recharge calibration targets from Reitz 2017 and WaterGAP 2.2a."""

# Sources:  reitz2017, watergap22a
# Method:   normalized_minmax
# Variable: recharge
# Timestep: annual


def build(config: dict, fabric_path: str, output_path: str) -> None:
    """Build recharge target dataset."""
    raise NotImplementedError
