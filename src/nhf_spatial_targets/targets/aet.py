"""Build AET calibration targets from MOD16A2 and SSEBop."""

# Sources:  mod16a2_v061, ssebop
# Method:   multi_source_minmax
# Variable: hru_actet
# Timestep: monthly


def build(config: dict, fabric_path: str, output_path: str) -> None:
    """Build AET target dataset."""
    raise NotImplementedError
