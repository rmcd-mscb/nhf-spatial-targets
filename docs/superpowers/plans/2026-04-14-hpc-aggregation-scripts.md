# HPC Aggregation Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SLURM batch scripts (`agg_all.slurm`, `agg_ssebop.slurm`) and matching pixi tasks so the ten aggregation sources from PR #47 can be submitted as HPC array/single jobs on USGS Hovenweep.

**Architecture:** Four deliverables: (1) ten new `agg-*` pixi tasks in `pixi.toml`, (2) `agg_all.slurm` 9-element array job for local-NC aggregators, (3) `agg_ssebop.slurm` single job for the remote STAC aggregator, (4) a "Running Aggregation: PC vs HPC" README section parallel to the existing fetch docs. No Python source changes — the aggregation API is already in place.

**Tech Stack:** Bash, SLURM (`#SBATCH`), pixi, TOML, Markdown.

**Spec:** `docs/superpowers/specs/2026-04-14-hpc-aggregation-scripts-design.md`

**Reference for style:** `fetch_all.slurm` (existing), `pixi.toml` `[tool.pixi.tasks]` block (existing `fetch-*` entries).

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `agg_all.slurm` | SLURM array job (0-8) wrapping nine `agg-*` pixi tasks |
| `agg_ssebop.slurm` | SLURM single job for `agg-ssebop` with `--period` |

### Modified files

| Path | Change |
|---|---|
| `pixi.toml` | Add ten `agg-*` task lines under `[tool.pixi.tasks]` |
| `README.md` | Insert "Running Aggregation: PC vs HPC" section after the fetch section |

---

## Task 1: Add pixi `agg-*` tasks

**Files:**
- Modify: `pixi.toml` (insert after the existing `agg-ssebop` line in `[tool.pixi.tasks]`)

**Context:** The slurm scripts (added in later tasks) call `pixi run "$TASK"`. Today only `agg-ssebop` exists. Ten new lines mirror the existing `fetch-*` style.

- [ ] **Step 1: Locate the existing aggregation task line**

Run: `grep -n "agg-ssebop" pixi.toml`
Expected: a single line like `agg-ssebop = { cmd = "nhf-targets agg ssebop", description = "..." }`.

- [ ] **Step 2: Add the ten new tasks**

Insert immediately after the existing `agg-ssebop = ...` line in `pixi.toml`:

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

- [ ] **Step 3: Verify pixi parses the file and sees the new tasks**

Run: `pixi task list 2>&1 | grep -E '^(agg-era5-land|agg-gldas|agg-merra2|agg-ncep-ncar|agg-nldas-mosaic|agg-nldas-noah|agg-watergap22d|agg-mod16a2|agg-mod10c1|agg-all)\b'`
Expected: ten lines, one per new task. If the grep returns nothing, pixi rejected the TOML or the regex doesn't match pixi's output format — investigate before proceeding.

- [ ] **Step 4: Verify the CLI invocation for one new task resolves without error**

Run: `pixi run agg-era5-land -- --help`
Expected: the `nhf-targets agg era5-land` help text appears with a non-zero `--project-dir` option description. Any other outcome means the task wiring is broken.

- [ ] **Step 5: Commit**

```bash
git add pixi.toml
git commit -m "feat(pixi): add agg-* tasks for every aggregation source"
```

---

## Task 2: Create `agg_all.slurm`

**Files:**
- Create: `agg_all.slurm` (repo root)

**Context:** 9-element SLURM array; each index maps to one of the new pixi `agg-*` tasks. Style mirrors `fetch_all.slurm` exactly (same `#SBATCH` block shape, same env-override idiom, same logs dir).

- [ ] **Step 1: Create the script**

Write `agg_all.slurm` (repo root):

```bash
#!/bin/bash
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

# NHF Spatial Targets — parallel aggregation array (tier-1 + tier-2)
# Submits one SLURM job per aggregation source (9 total). SSEBop is the
# remote-STAC aggregator and has its own separate script (agg_ssebop.slurm).
# Each job aggregates independently; they can run concurrently on separate
# compute nodes.
#
# Prerequisites:
#   - Datastore already hydrated (see fetch_all.slurm)
#   - `nhf-targets validate` has produced $PROJECT_DIR/fabric.json
#
# Usage (REPO_DIR and PROJECT_DIR are overridable via environment):
#   mkdir -p logs
#   export PROJECT_DIR=/path/to/your/project
#   sbatch agg_all.slurm
#
# Run a single source by index (e.g. MOD10C1 = 8):
#   sbatch --array=8 agg_all.slurm
#
# Bump memory for a MODIS rerun:
#   sbatch --array=7-8 --mem=256G agg_all.slurm
#
# Override spatial batch size (default 10000 HRUs/batch, tuned for 128 GB):
#   BATCH_SIZE=2500 sbatch agg_all.slurm

set -euo pipefail

# Repo and project directories — override via environment, or edit these
# defaults for your site. The repo default mirrors the USGS Hovenweep path;
# $PROJECT_DIR must be a project directory created by `nhf-targets init`.
REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"
BATCH_SIZE="${BATCH_SIZE:-10000}"

cd "$REPO_DIR"

# Map array index -> pixi task name
AGG_TASKS=(
    "agg-era5-land"     # 0 — ERA5-Land      (0.1°  monthly, runoff)
    "agg-gldas"         # 1 — GLDAS-2.1 NOAH (0.25° monthly, runoff)
    "agg-merra2"        # 2 — MERRA-2        (~0.5°  monthly, soil wetness)
    "agg-ncep-ncar"     # 3 — NCEP/NCAR      (~1.9° monthly, soil moisture)
    "agg-nldas-mosaic"  # 4 — NLDAS-2 MOSAIC (0.125° monthly, soil moisture)
    "agg-nldas-noah"    # 5 — NLDAS-2 NOAH   (0.125° monthly, soil moisture)
    "agg-watergap22d"   # 6 — WaterGAP 2.2d  (0.5° monthly, recharge)
    "agg-mod16a2"       # 7 — MOD16A2 v061   (500m 8-day AET)  — memory-heavy
    "agg-mod10c1"       # 8 — MOD10C1 v061   (0.05° daily SCA) — memory-heavy
)

TASK="${AGG_TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Array task $SLURM_ARRAY_TASK_ID: $TASK  batch_size=$BATCH_SIZE ==="
echo "=== Start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:  $(hostname) ==="

pixi run "$TASK" -- --project-dir "$PROJECT_DIR" --batch-size "$BATCH_SIZE"

echo "=== Done:  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
```

- [ ] **Step 2: Bash syntax check**

Run: `bash -n agg_all.slurm`
Expected: exit 0 with no output.

- [ ] **Step 3: Dry-run the array-index → task expansion (no SLURM)**

Run:
```bash
bash -c '
AGG_TASKS=(
    "agg-era5-land" "agg-gldas" "agg-merra2" "agg-ncep-ncar"
    "agg-nldas-mosaic" "agg-nldas-noah" "agg-watergap22d"
    "agg-mod16a2" "agg-mod10c1"
)
for i in 0 1 2 3 4 5 6 7 8; do
    echo "$i -> ${AGG_TASKS[$i]}"
done
'
```
Expected: nine lines mapping indices 0-8 onto the pixi task names in the same order as the `AGG_TASKS` array in the script. Confirm no off-by-one.

- [ ] **Step 4: Commit**

```bash
git add agg_all.slurm
git commit -m "feat(hpc): SLURM array job for tier-1/2 aggregation"
```

---

## Task 3: Create `agg_ssebop.slurm`

**Files:**
- Create: `agg_ssebop.slurm` (repo root)

**Context:** SSEBop is remote STAC/Zarr, not a local NC, and takes `--period`. Single SLURM job (not an array).

- [ ] **Step 1: Create the script**

Write `agg_ssebop.slurm` (repo root):

```bash
#!/bin/bash
#SBATCH --job-name=nhf-agg-ssebop
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/agg_ssebop_%j.out
#SBATCH --error=logs/agg_ssebop_%j.err

# NHF Spatial Targets — SSEBop aggregation (remote STAC)
# SSEBop reads from the USGS NHGF STAC Zarr store rather than a local
# consolidated NC, so it's kept separate from agg_all.slurm. Single job,
# not an array.
#
# Prerequisites:
#   - Project is initialised and `nhf-targets validate` has produced
#     $PROJECT_DIR/fabric.json
#
# Usage:
#   mkdir -p logs
#   export PROJECT_DIR=/path/to/your/project
#   sbatch agg_ssebop.slurm
#
# Override the period (default is SSEBop catalog availability 2000/2023):
#   PERIOD=2010/2020 sbatch agg_ssebop.slurm
#
# Override spatial batch size (default 10000, tuned for 128 GB):
#   BATCH_SIZE=2500 sbatch agg_ssebop.slurm

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

- [ ] **Step 2: Bash syntax check**

Run: `bash -n agg_ssebop.slurm`
Expected: exit 0 with no output.

- [ ] **Step 3: Confirm the period default matches the catalog**

Run: `grep -A1 '^  ssebop:' catalog/sources.yml | head -20 ; grep 'period:' catalog/sources.yml | head -30`
Expected: somewhere in the SSEBop block, a line resembling `period: "2000/2023"`. If it differs (e.g. the catalog has been extended to 2024), update the script's `PERIOD="${PERIOD:-2000/2023}"` default to match exactly.

- [ ] **Step 4: Commit**

```bash
git add agg_ssebop.slurm
git commit -m "feat(hpc): SLURM job for SSEBop STAC aggregation"
```

---

## Task 4: README "Running Aggregation: PC vs HPC" section

**Files:**
- Modify: `README.md` (insert a new subsection after "Running Fetches: PC vs HPC" and before the existing `## Aggregation` heading)

**Context:** The existing README already has a `## Aggregation` section that only covers SSEBop. The new subsection lives earlier in the document, structurally parallel to `### Running Fetches: PC vs HPC`, and covers both `agg_all.slurm` and `agg_ssebop.slurm`.

- [ ] **Step 1: Find the insertion point**

Run: `grep -n "## Aggregation" README.md`
Expected: one line with a line number (call it `L_AGG`). The new `### Running Aggregation: PC vs HPC` subsection goes *immediately before* `## Aggregation`. Confirm with `sed -n "$((L_AGG-5)),$((L_AGG+3))p" README.md` that the line above `## Aggregation` is the end of the fetch HPC section (typically a paragraph ending in "...adjusted for your cluster.").

- [ ] **Step 2: Insert the new subsection**

Insert *before* the `## Aggregation` heading line:

````markdown
### Running Aggregation: PC vs HPC

#### On a PC / workstation

Run aggregators individually or all at once:

```bash
# One source at a time
pixi run agg-era5-land -- --project-dir /data/gfv11-targets
pixi run agg-mod10c1   -- --project-dir /data/gfv11-targets

# All nine tier-1/2 sources sequentially
pixi run agg-all -- --project-dir /data/gfv11-targets

# SSEBop (remote STAC, takes a period)
pixi run agg-ssebop -- --project-dir /data/gfv11-targets --period 2000/2023
```

Each aggregator writes one NetCDF per source to `$PROJECT_DIR/data/aggregated/` and caches per-batch weights under `$PROJECT_DIR/weights/`.

#### On HPC (SLURM)

Two scripts at the repo root:

- [`agg_all.slurm`](agg_all.slurm) — 9-element array for local-NC aggregators
- [`agg_ssebop.slurm`](agg_ssebop.slurm) — single job for SSEBop (remote STAC)

**Prerequisites:**

1. Datastore hydrated (see fetch section above)
2. `pixi run validate -- --project-dir <dir>` completed (writes `fabric.json`)
3. `PROJECT_DIR` set via environment or edited inside the scripts

```bash
# From the repo root:
mkdir -p logs
export PROJECT_DIR=/path/to/gfv11-targets

# All 9 local-NC sources in parallel:
sbatch agg_all.slurm

# Rerun a single source by index (e.g. MOD10C1 at 8):
sbatch --array=8 agg_all.slurm

# Bump memory for a MODIS rerun that OOMed:
sbatch --array=7-8 --mem=256G agg_all.slurm

# SSEBop (remote STAC, separate script):
sbatch agg_ssebop.slurm

# Monitor:
squeue -u $USER

# Inspect logs (format: logs/agg_<arrayindex>_<jobid>.out/err):
tail -f logs/agg_8_*.out   # MOD10C1 live log
cat  logs/agg_7_*.err      # MOD16A2 error output
```

Array index → source mapping for `agg_all.slurm`:

| Index | Source | Notes |
|---|---|---|
| 0 | ERA5-Land | 0.1° monthly, runoff (ro, sro, ssro) |
| 1 | GLDAS-2.1 NOAH | 0.25° monthly, runoff (Qs + Qsb) |
| 2 | MERRA-2 | ~0.5° monthly, soil wetness |
| 3 | NCEP/NCAR | ~1.9° monthly, soil moisture |
| 4 | NLDAS-2 MOSAIC | 0.125° monthly, soil moisture (3 layers) |
| 5 | NLDAS-2 NOAH | 0.125° monthly, soil moisture (4 layers) |
| 6 | WaterGAP 2.2d | 0.5° monthly, diffuse recharge |
| 7 | MOD16A2 v061 | 500m 8-day AET (sinusoidal) — memory-heavy |
| 8 | MOD10C1 v061 | 0.05° daily SCA (CI-masked) — memory-heavy |

All nine jobs are CPU/memory-bound (not network I/O); the script allocates 1 CPU and 128 GB RAM per task with a 24-hour wall-clock limit. The 128 GB figure is sized for MOD10C1's daily 2000-present stack after the aggregator's in-memory `.load()`. Override `BATCH_SIZE` (default 10000 HRUs/batch, tuned for 128 GB) with `BATCH_SIZE=2500 sbatch agg_all.slurm` if a source OOMs. SLURM directives (`--account`, `--partition`) at the top of each script may need adjustment for non-Hovenweep clusters.

````

- [ ] **Step 3: Verify the rendering**

Run: `grep -n "Running Aggregation" README.md`
Expected: one match. Follow-up: `sed -n "$(grep -n "Running Aggregation" README.md | cut -d: -f1),+60p" README.md | head -80` — spot-check that the table renders cleanly and the code fences open/close correctly.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add Running Aggregation: PC vs HPC section"
```

---

## Final check

- [ ] **Full verification**

Run:
```bash
pixi task list | grep -cE '^agg-' | head      # count agg tasks
bash -n agg_all.slurm && bash -n agg_ssebop.slurm
pixi run -e dev fmt-check && pixi run -e dev lint && pixi run -e dev test
grep -n "Running Aggregation" README.md
```
Expected:
- `agg-` task count is 11 (the 10 new tasks + the pre-existing `agg-ssebop`).
- Both bash syntax checks exit 0 silently.
- Test suite still 402 passing + 14 deselected (no source code changed).
- README grep returns exactly one hit.

- [ ] **Optional dry-run on an HPC node** (out of scope for this plan, but documented as a follow-up)

If operator access to Hovenweep: submit `sbatch --array=6 agg_all.slurm` against a small project (WaterGAP 2.2d is the smallest: single variable, 0.5° monthly 1901-2016). Verify `data/aggregated/watergap22d_agg.nc` appears, `manifest.json` updates, and `logs/agg_6_*.out` contains the Start / Done markers.

---

## Notes for the implementer

- The two slurm scripts hardcode `--account=impd` and `--partition=cpu` (Hovenweep). Operators on other clusters override via `sbatch --account=... --partition=...` at submit time or edit the defaults.
- `REPO_DIR` default mirrors the Hovenweep site path. `PROJECT_DIR` default is the canonical GF v2 project — users typically override it.
- `BATCH_SIZE` env default in the SLURM scripts is 10000 (tuned for 128 GB). The underlying CLI/library default remains 500; a workstation user running `pixi run agg-mod10c1 -- --project-dir ...` without an explicit `--batch-size` stays safe on typical 32-64 GB laptops.
- `logs/` directory must exist before submission (`mkdir -p logs`) — SLURM will fail the job start otherwise.
