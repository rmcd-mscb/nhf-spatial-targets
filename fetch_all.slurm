#!/bin/bash
#SBATCH --job-name=nhf-fetch
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-13
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fetch_%a_%A.out
#SBATCH --error=logs/fetch_%a_%A.err

# NHF Spatial Targets — parallel fetch array
# Submits one SLURM job per source dataset (14 total).
# Each job downloads independently; they can run concurrently.
#
# Usage (REPO_DIR and PROJECT_DIR are overridable via environment):
#   mkdir -p logs
#   export REPO_DIR=/path/to/nhf-spatial-targets
#   export PROJECT_DIR=/path/to/your/project
#   sbatch fetch_all.slurm
#
# To run a single source only (e.g. index 2 = merra2):
#   sbatch --array=2 fetch_all.slurm
#
# Force re-fetch (bypasses the manifest-based year-skip; the entry in
# manifest.json is overwritten on completion). Only applies to sources
# that record per-year completion in the manifest — currently
# fetch-mod16a2 (index 8) and fetch-mod10c1 (index 9). Other sources
# fetch all-or-nothing per run, so FORCE is ignored for them with a
# log line.
#
#   export FORCE=1
#   sbatch --array=8 fetch_all.slurm   # MOD16A2 only, force re-fetch
#
# Use this after a pipeline change has invalidated the existing
# consolidated NCs (e.g. PR #88's MOD16A2 fill-mask fix).
#
# Daymet (index 11) is verify-only — it does not download. The fetch
# expects the three regional zarrs (NA/HI/PR) pre-staged on a shared
# filesystem. Configure ONE of:
#   - `daymet_root: /path/to/zarr/parent` in <project>/config.yml, OR
#   - DAYMET_ROOT env var, which this script forwards as --source-path.
# Missing zarrs raise FileNotFoundError per region. See
# docs/sources/daymet.md.
#
# Margulis WUS-SR (index 13) is fabric-scoped to Oregon. Non-Oregon
# projects can still fetch it (raw downloads stay reusable across
# projects sharing a datastore) but expect zero granules per year and
# a one-line warning. See docs/sources/margulis_wus_sr.md.

set -euo pipefail

# Repo and project directories — override via environment, or edit these
# defaults for your site. The repo default mirrors the USGS Hovenweep path;
# $PROJECT_DIR must be a project directory created by `nhf-targets init`.
REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"

cd "$REPO_DIR"

# Map array index → pixi task name
FETCH_TASKS=(
    "fetch-era5-land"      # 0  — 1979–present at 0.1°, hourly (CDS)
    "fetch-gldas"          # 1  — 2000–present at 0.25°, monthly (NASA GES DISC)
    "fetch-merra2"         # 2  — 1980–present at 0.5×0.625°, monthly (NASA GES DISC)
    "fetch-nldas-mosaic"   # 3  — 1979–present at 0.125°, monthly (NASA GES DISC)
    "fetch-nldas-noah"     # 4  — 1979–present at 0.125°, monthly (NASA GES DISC)
    "fetch-ncep-ncar"      # 5  — 1948–present at ~1.875°, daily→monthly (NOAA PSL)
    "fetch-watergap22d"    # 6  — 1901–2016 at 0.5°, monthly (PANGAEA) — ends 2016
    "fetch-reitz2017"      # 7  — 2000–2013 at 800m, annual (ScienceBase) — ends 2013
    "fetch-mod16a2"        # 8  — 2000–present at 500m, 8-day (LP DAAC/earthaccess)
    "fetch-mod10c1"        # 9  — 2000–present at 0.05°, daily (NSIDC/earthaccess)
    "fetch-mwbm-climgrid"  # 10 — 1900–2020 at 2.5 arcmin, monthly (ScienceBase, MANUAL DOWNLOAD; see docs/sources/mwbm_climgrid.md) — ends 2020
    "fetch-daymet"         # 11 — 1980–2024 at 1 km, daily (ORNL DAAC zarr, MANUAL STAGING; see docs/sources/daymet.md)
    "fetch-snodas"         # 12 — 2003–present at 1 km, daily (NSIDC G02158 via earthaccess)
    "fetch-margulis-wus-sr" # 13 — 1985–2021 at 90 m, daily (NSIDC-0719 via earthaccess) — Oregon-fabric only
)

# Map array index → fetch period (YYYY/YYYY)
# Periods chosen to cover the NHM model run (1979–2025) as fully as each
# source allows.  Sources with fixed ends (WaterGAP 2.2d: 2016,
# Reitz 2017: 2013, MODIS: 2000) are clamped to their available range.
FETCH_PERIODS=(
    "1979/2025"   # 0  era5-land
    "2000/2025"   # 1  gldas       (v2.1 starts 2000)
    "1980/2025"   # 2  merra2      (starts 1980)
    "1979/2025"   # 3  nldas-mosaic
    "1979/2025"   # 4  nldas-noah
    "1979/2025"   # 5  ncep-ncar   (product starts 1948; clamp to model start)
    "1979/2016"   # 6  watergap22d (dataset ends 2016)
    "2000/2013"   # 7  reitz2017   (dataset ends 2013)
    "2000/2025"   # 8  mod16a2     (MODIS starts 2000)
    "2000/2025"   # 9  mod10c1     (MODIS starts 2000)
    "1979/2020"   # 10 mwbm-climgrid (publisher ends 2020; 1895-1899 spinup excluded)
    "1980/2024"   # 11 daymet      (publisher ends 2024; V4 R1)
    "2003/2025"   # 12 snodas      (daily archive starts 2003-09-30)
    "1985/2021"   # 13 margulis-wus-sr (water years 1985-2021)
)

TASK="${FETCH_TASKS[$SLURM_ARRAY_TASK_ID]}"
PERIOD="${FETCH_PERIODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Array task $SLURM_ARRAY_TASK_ID: $TASK  period=$PERIOD ==="
echo "=== Start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:  $(hostname) ==="

# Decide whether to forward --force. Only fetch-mod16a2 / fetch-mod10c1
# implement the flag (they're the sources with manifest-based per-year
# skip logic). For everything else, log and ignore so a single FORCE=1
# can be used with the full array without erroring on un-flagged sources.
EXTRA_ARGS=()
case "${FORCE:-}" in
    1|true|TRUE|yes)
        case "$TASK" in
            fetch-mod16a2|fetch-mod10c1)
                EXTRA_ARGS+=("--force")
                echo "=== FORCE=$FORCE: appending --force to $TASK ==="
                ;;
            *)
                echo "=== FORCE=$FORCE: ignored for $TASK (no --force support) ==="
                ;;
        esac
        ;;
esac

# fetch-daymet is verify-only and needs to locate the staged regional
# zarrs. Two ways: `daymet_root:` in <project>/config.yml (preferred)
# or pass the directory via DAYMET_ROOT env → --source-path here.
# If neither is set, the fetch raises FileNotFoundError with operator
# instructions.
if [ "$TASK" = "fetch-daymet" ] && [ -n "${DAYMET_ROOT:-}" ]; then
    EXTRA_ARGS+=("--source-path" "$DAYMET_ROOT")
    echo "=== DAYMET_ROOT=$DAYMET_ROOT: appending --source-path to $TASK ==="
fi

pixi run "$TASK" -- --project-dir "$PROJECT_DIR" --period "$PERIOD" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "=== Done:  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
