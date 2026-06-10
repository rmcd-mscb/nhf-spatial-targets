"""Logic for 'nhf-targets init' — create a project skeleton."""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_TEMPLATE = """\
# nhf-spatial-targets project configuration
# Edit this file, then run: nhf-targets validate --project-dir <this-dir>

# ---------------------------------------------------------------------------
# Fabric
# ---------------------------------------------------------------------------
fabric:
  path: /path/to/fabric.gpkg        # absolute path to the HRU fabric
  id_col: nhm_id
  crs: EPSG:4326
  buffer_deg: 0.1                    # degrees to buffer bbox for downloads
  # Equal-area CRS used for HRU area + NN-fill distances. EPSG:5070 is
  # CONUS Albers — override for AK / HI / PR (e.g. EPSG:3338 for Alaska).
  area_crs: "EPSG:5070"
  # Optional fabric-scope token. Sources tagged with `fabric_scope` in
  # catalog/sources.yml (e.g. margulis_wus_sr -> [or]) are silently skipped
  # at both agg and target stages when this is unset. Set to one of
  # catalog.FABRIC_SCOPE_TOKENS (currently {"or"}) on fabric-restricted
  # projects to opt in.
  # token: or
  # KD-tree partition target HRUs/batch. The same partition is reused
  # across every source's aggregation. Override per-run with CLI
  # --batch-size. Changing this value invalidates any cached weight
  # CSVs in <workdir>/weights/ — the SHA-256 batch-HRU fingerprint
  # forces recompute.
  batch_size: 500

# ---------------------------------------------------------------------------
# Datastore — shared directory for raw source downloads
# ---------------------------------------------------------------------------
datastore: /path/to/datastore        # absolute path; may be shared across runs

# Optional Unix directory permissions (octal string, e.g. "2775" for setgid + group rwx)
# Ignored on Windows.
dir_mode: "2775"

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
      - era5_land
      - gldas_noah_v21_monthly
      - mwbm_climgrid
    time_step: monthly
    period: "2000-01-01/2010-12-31"
    prms_variable: basin_cfs
    range_method: multi_source_minmax
    output_file: runoff_targets.nc
    nn_fill: true
    nn_max_candidates: 10
    chunk_months: 12

  aet:
    enabled: true
    sources:
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
      - era5_land
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
      - ua_swe            # UA NSIDC-0719, CONUS, snow_covered_fraction (WY 1982+)
    time_step: daily
    period: "2000-01-01/2010-12-31"
    prms_variable: snowcov_area
    range_method: modis_ci
    ci_threshold: 0.70
    # Pixel snow-depth cutoff (mm) for ua_swe's snow_covered_fraction binary.
    # Consumed at the ua_swe AGGREGATE stage (changing it requires re-aggregating
    # ua_swe; the publish gate refuses a stamp that disagrees with this value).
    depth_threshold_mm: 1.0
    # true: zero the COMBINED bound in Jul/Aug (calcSCA). false: zero only the
    # mod10c1 contribution pre-combine, so ua_swe's summer alpine fraction
    # survives into the combined upper bound.
    forced_zero_combined: true
    # 1: keep a bound wherever >=1 source is finite (a single source's [v, v] is
    # allowed). 2: NaN the combined bound where fewer than 2 sources are finite.
    min_sources_for_bound: 1
    output_file: sca_targets.nc

  snow_water_equivalent:
    enabled: true
    sources:
      - daymet
      - snodas
      - era5_land
      - margulis_wus_sr   # Oregon fabric only (fabric_scope)
      - ua_swe            # UA NSIDC-0719, CONUS, extends bound to 1982 (CY 1982-2022)
    time_step: daily
    period: "2003-01-01/2010-12-31"
    prms_variable: pkwater_equiv
    range_method: multi_source_minmax
    output_file: swe_targets.nc
    nn_fill: true
    nn_max_candidates: 10

# ---------------------------------------------------------------------------
# Inspection-notebook overrides (optional)
# ---------------------------------------------------------------------------
# Per-target representative HRU points used by the inspect_* notebooks'
# time-series cells. When unset, each notebook falls back to its hardcoded
# CONUS-spanning defaults (Olympic / Iowa / Phoenix / Appalachians) — fine
# for gfv2, but those points lie outside regional fabrics (e.g. Oregon's PNW
# extent) and lookup_hrus_by_points will raise. Set per-target lists of
# `"label": [lon, lat]` inside the fabric to override. YAML anchors keep
# one set across all targets if desired.
#
# representative_points:
#   aet: &rep_pts
#     "Region A": [-121.7, 45.4]
#     "Region B": [-123.0, 44.6]
#     "Region C": [-118.6, 42.7]
#     "Region D": [-123.5, 45.0]
#   runoff: *rep_pts
#   recharge: *rep_pts
#   soil_moisture: *rep_pts
#   snow_covered_area: *rep_pts
#   swe: *rep_pts

# ---------------------------------------------------------------------------
# ScienceBase data release (optional)
# ---------------------------------------------------------------------------
# Per-project metadata for the `nhf-targets release` workflow. Most
# defaults live in catalog/release_defaults.yml (committed repo-wide).
# This block carries the per-project bits: authors, IPDS number, DOI
# (filled after ScienceBase staff mints it post-IPDS approval), and an
# optional fabric-label override. Leave commented until ready to publish.
# See docs/data_release/ for the full workflow.
#
# release:
#   authors:
#     - given: Jane
#       family: Doe
#       orcid: 0000-0000-0000-0000
#       affiliation: U.S. Geological Survey
#   ipds_number: IP-XXXXXX
#   doi: null               # populated after SB staff mints DOI
#   abstract_notes: null    # optional paragraph appended to fabric abstract
#   fabric_label: null      # human-readable name; defaults to Path(fabric.path).stem
"""

_CREDENTIALS_TEMPLATE = {
    "nasa_earthdata": {
        "_comment": "NASA Earthdata login — https://urs.earthdata.nasa.gov",
        "username": "",
        "password": "",
    },
    "cds": {
        "_comment": "Copernicus CDS API — https://cds.climate.copernicus.eu",
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "<uid>:<key>",
    },
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def init_project(workdir: Path) -> Path:
    """Create a new project skeleton under *workdir*.

    Parameters
    ----------
    workdir : Path
        Directory to create.  Must not already exist.

    Returns
    -------
    Path to the newly created project directory (resolved).
    """
    workdir = workdir.resolve()

    if workdir.exists():
        raise FileExistsError(
            f"Project already exists: {workdir}\n"
            "Choose a different --project-dir path, or remove the existing directory."
        )

    # Directory skeleton
    (workdir / "data" / "aggregated").mkdir(parents=True)
    (workdir / "targets").mkdir(parents=True)
    (workdir / "logs").mkdir(parents=True)

    # Config template
    (workdir / "config.yml").write_text(_CONFIG_TEMPLATE)

    # Credentials template
    (workdir / ".credentials.yml").write_text(
        "# nhf-spatial-targets project credentials\n"
        "# Fill in before running 'nhf-targets validate'.\n"
        "# This file is gitignored — do not commit it.\n\n"
        + yaml.dump(_CREDENTIALS_TEMPLATE, default_flow_style=False, sort_keys=False)
    )

    return workdir
