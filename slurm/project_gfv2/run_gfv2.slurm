#!/bin/bash
#SBATCH --job-name=nhf-run-gfv2
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/run_%a_%A.out
#SBATCH --error=logs/run_%a_%A.err

# gfv2 target-build array. Thin wrapper over the shared target dispatch; only
# PROJECT_DIR differs from slurm/project_or/run_or.slurm. Run the agg array
# (slurm/project_gfv2/agg_all_gfv2.slurm) + agg_ssebop + agg_daymet first.
#
# Usage (from the repo root so logs/ resolves there):
#   mkdir -p logs
#   sbatch slurm/project_gfv2/run_gfv2.slurm
#   sbatch --array=0 slurm/project_gfv2/run_gfv2.slurm     # runoff only

set -euo pipefail

# Group-writable output on the shared HPC filesystem — SLURM does not
# inherit the login-shell umask. Belt-and-suspenders with os.umask(0o002)
# in the CLI (see io_nc.apply_umask_mode for why umask alone is not enough
# for the mkstemp-based atomic NetCDF writers).
umask 002

PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
REPO_DIR="${REPO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
[ -n "$REPO_DIR" ] || {
    echo "ERROR: could not resolve REPO_DIR; set REPO_DIR=/path/to/repo and resubmit" >&2
    exit 1
}

# Guard the wrong-repo case: $SLURM_SUBMIT_DIR may sit inside a *different* git
# checkout than the one holding these scripts. Require the shared body to exist
# and the tree to identify as nhf-spatial-targets before sourcing, so a stale
# clone fails loudly instead of silently running the wrong code.
BODY="$REPO_DIR/slurm/shared/run_all.body.sh"
if [ ! -f "$BODY" ] || ! grep -q 'name = "nhf-spatial-targets"' "$REPO_DIR/pyproject.toml" 2>/dev/null; then
    echo "ERROR: REPO_DIR=$REPO_DIR (resolved from ${SLURM_SUBMIT_DIR:-$PWD}) is not an nhf-spatial-targets checkout with $BODY. Set REPO_DIR=/path/to/nhf-spatial-targets and resubmit." >&2
    exit 1
fi

source "$BODY"
