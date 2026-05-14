"""Build snow water equivalent (SWE) calibration targets."""

from __future__ import annotations

# Sources:  daymet, snodas, era5_land, margulis_wus_sr
# Method:   multi_source_minmax (per-HRU per-day nanmin/nanmax across sources)
# Variable: pkwater_equiv
# Timestep: daily


def build(config: dict, fabric_path: str, output_path: str) -> None:
    """Build SWE target dataset."""
    raise NotImplementedError
