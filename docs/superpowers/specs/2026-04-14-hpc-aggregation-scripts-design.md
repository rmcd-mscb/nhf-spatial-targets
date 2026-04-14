# HPC Aggregation Scripts — Design

**Date:** 2026-04-14
**Status:** approved, ready for implementation plan
**Scope:** Add SLURM batch scripts and matching pixi tasks so the 10 aggregation
sources added in PR #47 can be submitted as HPC array jobs, mirroring the
existing `fetch_all.slurm` pattern.

## Goals

1. Run the 9 tier-1/tier-2 aggregators in parallel on SLURM as a single array
   job, with per-index failure isolation and env-overridable project paths.
2. Run the remote-STAC SSEBop aggregator as a separate single job (its I/O
   profile differs from local-NC sources).
3. Expose a pixi task per source so both the SLURM scripts and workstation
   users share a single, discoverable command surface.
4. Document the PC and HPC invocation paths in the README so operators don't
   have to read SLURM files to learn the entry points.

## Non-goals

- Auto-chaining aggregation after fetch via `--dependency=afterok`. Scripts
  are independent; operators submit them in sequence manually.
- Per-source differentiated resource profiles. Single 128 GB / 24 h / 1 CPU
  tier for all nine indices; tune later from real `sacct` metrics.
- Source-code changes to `src/nhf_spatial_targets/`. The aggregation API
  built in PR #47 already exposes everything the scripts need.

## Architecture

Two SLURM scripts at the repo root, alongside the existing `fetch_all.slurm`:

### `agg_all.slurm` — 9-element array for local-NC aggregators

| Array idx | pixi task | Source description |
|---|---|---|
| 0 | `agg-era5-land` | ERA5-Land (0.1° monthly, runoff) |
| 1 | `agg-gldas` | GLDAS-2.1 NOAH (0.25° monthly, runoff) |
| 2 | `agg-merra2` | MERRA-2 (~0.5° monthly, soil wetness) |
| 3 | `agg-ncep-ncar` | NCEP/NCAR (~1.9° monthly, soil moisture) |
| 4 | `agg-nldas-mosaic` | NLDAS-2 MOSAIC (0.125° monthly, soil moisture) |
| 5 | `agg-nldas-noah` | NLDAS-2 NOAH (0.125° monthly, soil moisture) |
| 6 | `agg-watergap22d` | WaterGAP 2.2d (0.5° monthly, recharge) |
| 7 | `agg-mod16a2` | MOD16A2 v061 (500m 8-day AET) — memory-heavy |
| 8 | `agg-mod10c1` | MOD10C1 v061 (0.05° daily SCA) — memory-heavy |

**SBATCH directives:**

```
#SBATCH --job-name=nhf-agg
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/agg_%a_%A.out
#SBATCH --error=logs/agg_%a_%A.err
```

**Env-overridable parameters** (same idiom as `fetch_all.slurm`):

- `REPO_DIR` — default `/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets`
- `PROJECT_DIR` — default `/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets`
- `BATCH_SIZE` — default `10000`, sized for the 128 GB allocation. CLI/library
  default remains `500` (appropriate for workstation runs).

**Body:**

```bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
BATCH_SIZE="${BATCH_SIZE:-10000}"

AGG_TASKS=(
    "agg-era5-land"     # 0
    "agg-gldas"         # 1
    "agg-merra2"        # 2
    "agg-ncep-ncar"     # 3
    "agg-nldas-mosaic"  # 4
    "agg-nldas-noah"    # 5
    "agg-watergap22d"   # 6
    "agg-mod16a2"       # 7
    "agg-mod10c1"       # 8
)

cd "$REPO_DIR"

TASK="${AGG_TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Array task $SLURM_ARRAY_TASK_ID: $TASK  batch_size=$BATCH_SIZE ==="
echo "=== Start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:  $(hostname) ==="

pixi run "$TASK" -- --project-dir "$PROJECT_DIR" --batch-size "$BATCH_SIZE"

echo "=== Done:  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
```

**Failure semantics:** SLURM array indices fail independently. A MOD10C1 OOM
on index 8 does not affect the other eight. Re-run a single failed source
with `sbatch --array=8 --mem=256G agg_all.slurm`.

**Usage examples** in the header comment:

```
#   sbatch agg_all.slurm                    # all 9 sources
#   sbatch --array=7-8 agg_all.slurm        # just the MODIS two
#   sbatch --array=8 --mem=256G agg_all.slurm   # MOD10C1 with more memory
#   BATCH_SIZE=2500 sbatch agg_all.slurm    # smaller spatial batches
#   PROJECT_DIR=/path sbatch agg_all.slurm  # explicit project
```

### `agg_ssebop.slurm` — single job for remote STAC aggregator

SSEBop reads from the USGS NHGF STAC Zarr store rather than a local
consolidated NetCDF, so its I/O profile differs from the nine local-NC
aggregators. It also requires a `--period` argument (the remote query span).

**SBATCH directives:**

```
#SBATCH --job-name=nhf-agg-ssebop
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/agg_ssebop_%j.out
#SBATCH --error=logs/agg_ssebop_%j.err
```

**Env-overridable parameters:**

- `REPO_DIR`, `PROJECT_DIR` — same defaults as `agg_all.slurm`.
- `PERIOD` — default `2000/2023` (SSEBop catalog availability).
- `BATCH_SIZE` — default `10000`.

**Body:**

```bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
PERIOD="${PERIOD:-2000/2023}"
BATCH_SIZE="${BATCH_SIZE:-10000}"

cd "$REPO_DIR"

echo "=== agg-ssebop  period=$PERIOD  batch_size=$BATCH_SIZE ==="
echo "=== Start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:  $(hostname) ==="

pixi run agg-ssebop -- \
    --project-dir "$PROJECT_DIR" \
    --period "$PERIOD" \
    --batch-size "$BATCH_SIZE"

echo "=== Done:  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
```

**Usage examples:**

```
#   sbatch agg_ssebop.slurm
#   PERIOD=2010/2020 sbatch agg_ssebop.slurm
```

## Pixi task additions

Add ten new tasks to `pixi.toml` in the `[tool.pixi.tasks]` block, grouped
with the existing `fetch-*` and `agg-ssebop` entries:

```toml
agg-era5-land    = { cmd = "nhf-targets agg era5-land",    description = "Aggregate ERA5-Land runoff to HRU fabric" }
agg-gldas        = { cmd = "nhf-targets agg gldas",        description = "Aggregate GLDAS-2.1 NOAH runoff to HRU fabric" }
agg-merra2       = { cmd = "nhf-targets agg merra2",       description = "Aggregate MERRA-2 soil wetness to HRU fabric" }
agg-ncep-ncar    = { cmd = "nhf-targets agg ncep-ncar",    description = "Aggregate NCEP/NCAR soil moisture to HRU fabric" }
agg-nldas-mosaic = { cmd = "nhf-targets agg nldas-mosaic", description = "Aggregate NLDAS-2 MOSAIC soil moisture to HRU fabric" }
agg-nldas-noah   = { cmd = "nhf-targets agg nldas-noah",   description = "Aggregate NLDAS-2 NOAH soil moisture to HRU fabric" }
agg-watergap22d  = { cmd = "nhf-targets agg watergap22d",  description = "Aggregate WaterGAP 2.2d recharge to HRU fabric" }
agg-mod16a2      = { cmd = "nhf-targets agg mod16a2",      description = "Aggregate MOD16A2 v061 AET to HRU fabric" }
agg-mod10c1      = { cmd = "nhf-targets agg mod10c1",      description = "Aggregate MOD10C1 v061 SCA to HRU fabric" }
agg-all          = { cmd = "nhf-targets agg all",          description = "Aggregate all tier-1/2 sources to HRU fabric" }
```

Existing `agg-ssebop` is unchanged.

## README update

Insert a new section immediately after the existing "Running Fetches: PC vs
HPC" section:

```markdown
### Running Aggregation: PC vs HPC

#### On a PC / workstation

    pixi run agg-era5-land -- --project-dir /path/to/project
    pixi run agg-mod10c1   -- --project-dir /path/to/project

Or all nine tier-1/2 sources sequentially:

    pixi run agg-all -- --project-dir /path/to/project

SSEBop (remote STAC) takes a period:

    pixi run agg-ssebop -- --project-dir /path/to/project --period 2000/2023

#### On HPC (SLURM)

The nine local-NC aggregators run as a SLURM array via `agg_all.slurm`;
SSEBop (remote STAC) is a separate single job in `agg_ssebop.slurm`.

Prerequisites — datastore already hydrated (see fetch section) and
`nhf-targets validate` has produced `fabric.json`.

    export PROJECT_DIR=/path/to/project
    mkdir -p logs

    # All 9 local-NC sources in parallel:
    sbatch agg_all.slurm

    # Rerun a single source by index (e.g. MOD10C1 at 8):
    sbatch --array=8 agg_all.slurm

    # Bump memory for a MODIS rerun:
    sbatch --array=7-8 --mem=256G agg_all.slurm

    # SSEBop:
    sbatch agg_ssebop.slurm

    # Monitor:
    squeue -u $USER
```

Include the index→source table from this spec (transposed into a short
markdown table) inside the HPC subsection so operators don't have to read the
script to learn the mapping.

Override `BATCH_SIZE` (default 10000, tuned for the 128 GB allocation) with
`BATCH_SIZE=2500 sbatch agg_all.slurm` if a source hits OOM.

## Open risks

- **MOD16A2 / MOD10C1 may exceed 128 GB.** MOD10C1 at 0.05° daily across
  25 years is a large lazy stack; after our recent `.load()` refactor, the
  whole opened NC lives in memory. If the first HPC run OOMs at index 7 or 8,
  the documented remediation is `--mem=256G` on a rerun. A follow-up can
  differentiate resources by index if this becomes a regular pain point.
- **SSEBop STAC network variability.** The SSEBop aggregator fetches from a
  remote Zarr store; transient S3 failures translate to job failure. Retry
  is the operator's responsibility — `sbatch agg_ssebop.slurm` again.
- **Hovenweep path defaults.** The `REPO_DIR` default hard-codes a USGS
  Hovenweep path. Users on other clusters must set `REPO_DIR` explicitly or
  edit the default — documented in the script header comment.
