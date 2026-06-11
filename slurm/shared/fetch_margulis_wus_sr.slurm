#!/bin/bash
#SBATCH --job-name=nhf-margulis-wus-sr
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00      # NSIDC-0719 is per-water-year per-tile; ~37 years total
#SBATCH --output=logs/margulis_wus_sr_%j.out
#SBATCH --error=logs/margulis_wus_sr_%j.err

# NHF Spatial Targets — Margulis Western US Snow Reanalysis (NSIDC-0719) fetch
#
# Single-task job (no --array): fetch_margulis_wus_sr does not support
# worker sharding today and the granule volume is modest enough that
# serial year-by-year iteration finishes well within the 24h wall clock.
# Years already recorded with n_granules > 0 are skipped at startup
# (manifest fast path), so resuming after a partial run requires no
# flag changes.
#
# This source covers only the Western US (#309 removed the old
# fabric_scope opt-in, so this is now a fabric-independent shared
# fetcher like fetch_ua_swe / fetch_snodas). The CMR search bbox is the
# project fabric's buffered bbox — raw data lands in the shared
# datastore and remains usable by any project pointed at the same
# datastore; fabrics outside the WUS get zero granules per year. A CONUS
# project (e.g. gfv2) wants this too, since Margulis now contributes to
# its Western-US HRUs.
#
# Per-water-year per-tile granules are consolidated into per-calendar-year
# NetCDFs by the consolidator that runs after each year's downloads
# complete. See docs/sources/margulis_wus_sr.md.
#
# Usage:
#   mkdir -p logs
#   export PROJECT_DIR=/path/to/your/project
#   sbatch slurm/shared/fetch_margulis_wus_sr.slurm
#
# To fetch a subset of years only:
#   PERIOD=2000/2010 sbatch slurm/shared/fetch_margulis_wus_sr.slurm

set -euo pipefail

# Flush Python stdout per-line so slurm logs (--output=...) update in
# real time instead of dumping batches when the 8 KB block buffer fills.
# Slow workloads can otherwise look frozen for tens of minutes.
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR to the target project dir, e.g. PROJECT_DIR=/caldera/.../gfv2-spatial-targets}"
PERIOD="${PERIOD:-1985/2021}"

echo "=== Margulis WUS-SR fetch ==="
echo "=== Period:  ${PERIOD} ==="
echo "=== Project: ${PROJECT_DIR} ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:    $(hostname) ==="

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} not found" >&2; exit 1; }

pixi run fetch-margulis-wus-sr -- \
    --project-dir "${PROJECT_DIR}" \
    --period      "${PERIOD}"

echo "=== Done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
