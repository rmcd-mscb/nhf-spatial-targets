"""Logic for 'nhf-targets init' — create a workspace skeleton."""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_TEMPLATE = """\
# nhf-spatial-targets workspace configuration
# Edit this file, then run: nhf-targets validate --workdir <this-dir>

# ---------------------------------------------------------------------------
# Fabric
# ---------------------------------------------------------------------------
fabric:
  path: /path/to/fabric.gpkg        # absolute path to the HRU fabric
  id_col: nhm_id
  crs: EPSG:4326
  buffer_deg: 0.1                    # degrees to buffer bbox for downloads

# ---------------------------------------------------------------------------
# Datastore — shared directory for raw source downloads
# ---------------------------------------------------------------------------
datastore: /path/to/datastore        # absolute path; may be shared across runs

# Optional Unix directory permissions (octal, e.g. "0o2775" for group-sticky)
dir_mode: null

# ---------------------------------------------------------------------------
# Spatial aggregation (gdptools)
# ---------------------------------------------------------------------------
aggregation:
  engine: gdptools
  method: area_weighted

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
output:
  dir: outputs
  format: netcdf
  compress: true

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
targets:

  runoff:
    enabled: true
    sources:
      - nhm_mwbm
    time_step: monthly
    period: "1982-01-01/2010-12-31"
    prms_variable: basin_cfs
    range_method: mwbm_uncertainty
    output_file: runoff_targets.nc

  aet:
    enabled: true
    sources:
      - nhm_mwbm
      - mod16a2_v061
      - ssebop
    time_step: monthly
    period: "2000-01-01/2010-12-31"
    prms_variable: hru_actet
    range_method: multi_source_minmax
    output_file: aet_targets.nc

  recharge:
    enabled: true
    sources:
      - reitz2017
      - watergap22d
    time_step: annual
    period: "2000-01-01/2009-12-31"
    prms_variable: recharge
    range_method: normalized_minmax
    normalize: true
    normalize_period: "2000-01-01/2009-12-31"
    output_file: recharge_targets.nc

  soil_moisture:
    enabled: true
    sources:
      - merra2
      - ncep_ncar
      - nldas_mosaic
      - nldas_noah
    time_step:
      - monthly
      - annual
    period: "1982-01-01/2010-12-31"
    prms_variable: soil_rechr
    range_method: normalized_minmax
    normalize: true
    normalize_by: calendar_month
    output_file: soil_moisture_targets.nc

  snow_covered_area:
    enabled: true
    sources:
      - mod10c1_v061
    time_step: daily
    period: "2000-01-01/2010-12-31"
    prms_variable: snowcov_area
    range_method: modis_ci
    ci_threshold: 0.70
    output_file: sca_targets.nc
"""

_CREDENTIALS_TEMPLATE = {
    "nasa_earthdata": {
        "_comment": "NASA Earthdata login — https://urs.earthdata.nasa.gov",
        "username": "",
        "password": "",
    },
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def init_workspace(workdir: Path) -> Path:
    """Create a new workspace skeleton under *workdir*.

    Parameters
    ----------
    workdir : Path
        Directory to create.  Must not already exist.

    Returns
    -------
    Path to the newly created workspace directory (resolved).
    """
    workdir = workdir.resolve()

    if workdir.exists():
        raise FileExistsError(
            f"Workspace already exists: {workdir}\n"
            "Choose a different --workdir path, or remove the existing directory."
        )

    # Directory skeleton
    (workdir / "data" / "aggregated").mkdir(parents=True)
    (workdir / "targets").mkdir(parents=True)
    (workdir / "logs").mkdir(parents=True)

    # Config template
    (workdir / "config.yml").write_text(_CONFIG_TEMPLATE)

    # Credentials template
    (workdir / ".credentials.yml").write_text(
        "# nhf-spatial-targets workspace credentials\n"
        "# Fill in before running 'nhf-targets validate'.\n"
        "# This file is gitignored — do not commit it.\n\n"
        + yaml.dump(_CREDENTIALS_TEMPLATE, default_flow_style=False, sort_keys=False)
    )

    return workdir
