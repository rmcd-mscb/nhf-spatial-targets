# Runoff Target Builder — Design

**Date:** 2026-05-05
**Status:** Approved (brainstorming complete; awaiting implementation plan)
**Scope:** Implement the runoff calibration-target builder using all three configured sources (ERA5-Land, GLDAS-2.1 NOAH, MWBM ClimGrid) with period-union semantics, optional NN-fill post-processing, and CF-1.6 compliant output. Add a project-wide config-defaults layer so existing project directories continue to work after new keys are introduced. Update related docs and reference configs. Add an HPC `run_all.slurm` script.

---

## Goals

1. Replace the existing partial `targets/run.py` (which is misaligned with the per-year aggregation layout established in PR #51 and only consumes 2 of 3 sources) with a working builder that:
   - Reads per-year aggregated NCs at `<project>/data/aggregated/<source_key>/<source_key>_<YYYY>_agg.nc`.
   - Accepts a project-config `period: "<min>/<max>"` and uses every source's available data within that period (union semantics, not intersection).
   - Combines sources via NaN-aware min/max so a bound is defined whenever ≥1 source is finite at a given (HRU, time).
   - Writes a CF-1.6 compliant NetCDF with dimensions `(time, <fabric.id_col>)`, ancillary lat/lon centroid coords, and `time_bnds`.
   - Optionally produces a parallel `*_nn_filled.nc` post-processing artifact that fills cells where every source was NaN, using a nearest-finite-neighbor walk in equal-area space.

2. Introduce a project-config defaults layer so old `config.yml` files don't break when new keys are introduced. Apply to all five targets (runoff + the four still-stub targets).

3. Update existing docs (`calibration-target-recipes.md`, `processing-steps-reference.md`) and reference configs to be consistent with the new design and the all-three-sources reality.

4. Add `run_all.slurm` parallel to `agg_all.slurm` and `fetch_all.slurm` for HPC integration.

## Non-goals

- Implementing AET, recharge, soil-moisture, or SCA target builders. They remain stubs; only the shared infrastructure (`targets/_common.py`, defaults layer, slurm script, output writer) is built and tested in this work.
- Any change to `aggregate/` or `fetch/` modules.
- A `cfchecker`-style external CF-validator step (xarray round-trip is sufficient per project preference).
- Backwards compatibility shims for the old `<source_key>/<var>.nc` aggregated layout (already gone since PR #51).

---

## Architecture

### Module map

**New / rewritten under `src/nhf_spatial_targets/`:**

- `targets/run.py` — rewritten. Declares the three runoff contributors and their per-source unit-conversion shims; orchestrates by calling helpers in `targets/_common.py`. Retains the existing unit-conversion functions (`era5_to_mm_per_month`, `gldas_to_mm_per_month`, `mm_per_month_to_cfs`) and adds an MWBM shim (no conversion — MWBM `runoff` is already `mm/month`; the shim verifies units against the catalog).
- `targets/_common.py` — new. Shared multi-source-minmax target machinery:
  - `read_aggregated_source(project, source_key, var, period) -> xr.DataArray` — opens per-year NCs at `<project>/data/aggregated/<source_key>/<source_key>_*_agg.nc` via `xr.open_mfdataset` with `chunks={"time": runoff.chunk_months, id_col: -1}`, slices to `[period_min, period_max]`. Raises if no files match or if the period falls entirely outside the source's coverage.
  - `reindex_to_month_start(da, master_index) -> xr.DataArray` — maps any monthly-source `time` coord (EOM / SOM / mid-month) onto the master `freq="MS"` index. Implemented as `da.assign_coords(time=da.time.dt.to_period("M").dt.to_timestamp())` then `.reindex(time=master_index)`.
  - `multi_source_nanminmax(sources_dict) -> (lower, upper, n_sources)` — `xr.concat([...], dim="source").min(dim="source", skipna=True)` and `.max(...)`; `n_sources` is `xr.where(stacked.notnull(), 1, 0).sum(dim="source")` cast to int8.
  - `compute_hru_area_and_centroids(project) -> pandas.DataFrame` — opens fabric, reprojects geometries to `project.area_crs`, computes `area_m2` and centroid `(x, y)` in equal-area; reprojects centroids to EPSG:4326 for `centroid_lat` / `centroid_lon`. Always recomputes from geometry (no fabric-column fallback).
  - `write_target_nc(ds, output_path, title, extra_global_attrs) -> None` — atomic write (tempfile + rename), CF-1.6 conventions, sets `time_bnds`, `_FillValue=NaN` for floats, `_FillValue=-1` for `n_sources`/`nn_filled` int8s, zlib compression complevel 4, time encoded as `days since <epoch>` with `calendar="proleptic_gregorian"`.
- `normalize/methods.py` — implement `nn_fill_bounds(ds, centroids_xy, max_candidates=10) -> (filled_ds, nn_filled_diag)`. Uses `scipy.spatial.cKDTree` on `centroids_xy` (HRU centroids in equal-area CRS). For each HRU that is NaN in *both* `lower_bound` and `upper_bound` at any time step: walk neighbors in increasing-distance order until finding a donor that is finite at that time step, or until `max_candidates` exhausted (in which case the cell stays NaN). Vectorized over HRUs that ever go NaN; per-time-step inner loop. Returns the filled dataset plus an int8 `nn_filled(time, id_col)` flag variable.
- `workspace.py` — extend `load()` to merge user config over `defaults.DEFAULTS`. Surface a `Project.target(name) -> dict` method that returns the resolved per-target sub-dict. New `Project.area_crs` property reads `fabric.area_crs` from the merged config.
- `defaults.py` — new. Single `DEFAULTS` dict mirroring the config schema with default values for every settable key across all five targets and project-level fields. `None` sentinel means "no default — required." Single source of truth for migration safety.
- `validate.py` — extend to (i) report which keys took defaults to stderr, (ii) warn about user-set keys not present in `DEFAULTS` (likely typos), (iii) write `<project>/config.effective.yml` (mode `0o444`) with the merged config and a leading `# AUTOGENERATED` comment, (iv) keep all existing fabric/credential/catalog checks.
- `cli.py` — add `run-runoff` subcommand and wire `--target runoff` for the existing `run` command. The flag form is what the new pixi tasks call.

**New tests under `tests/`:**

- `tests/test_targets_common.py` — covers every helper in `targets/_common.py` with synthetic per-year aggregated NCs and a synthetic 3–5 polygon GeoPackage built in-test.
- `tests/test_targets_run.py` — rewritten end-to-end build with three synthetic sources, including a deliberate partial-period source to verify period-union semantics.
- `tests/test_normalize_nn_fill.py` — donor walk, finite-only constraint, max-candidates cap, time-step independence.
- `tests/test_workspace_defaults.py` — merge semantics, list-replace not list-merge, unknown-key warning, missing-required-key error, `config.effective.yml` round-trip.

**Updated config / docs / scripts:**

- `config/pipeline.yml` (REFERENCE ONLY) — runoff block updated to include `mwbm_climgrid`, `nn_fill: true`, `nn_max_candidates: 10`; project-level `fabric.area_crs: "EPSG:5070"`.
- `src/nhf_spatial_targets/init_run.py` `_CONFIG_TEMPLATE` — same updates as the reference config.
- `docs/references/calibration-target-recipes.md` § 1 Runoff — rewritten per Section 7 below.
- `docs/references/processing-steps-reference.md` § 1 Runoff and § Cross-cutting conventions — minor edits per Section 7 below.
- `catalog/variables.yml` `runoff:` block `range_notes` — clarify NaN-aware semantics.
- `CLAUDE.md` — append `pixi run run-runoff` to example commands; expand the NN-fill placement bullet under Aggregation Transformation Policy.
- `pixi.toml` — add `run-runoff`, `run-aet`, `run-recharge`, `run-soil-moisture`, `run-snow-covered-area` task aliases.
- `run_all.slurm` — new at repo root, mirroring `agg_all.slurm` (5-element array, 32 GB / 4 h defaults).

### Data flow (one runoff invocation)

```
1. CLI nhf-targets run --target runoff --project-dir <dir>
   → workspace.load(project_dir):
       - read config.yml
       - deep-merge over defaults.DEFAULTS (user wins; lists replace wholesale)
       - check required keys (datastore, fabric.path, target.period when enabled)
       - load fabric.json
       - return Project + resolved per-target config

2. targets.run.build(project, runoff_cfg):
   a. hru_meta = compute_hru_area_and_centroids(project)
        - open fabric (geopandas)
        - reproject to project.area_crs (e.g. EPSG:5070 for CONUS)
        - area_m2 = geometries.area
        - centroid_x, centroid_y = geometries.centroid (in area_crs)
        - centroid_lat, centroid_lon = those centroids reprojected to EPSG:4326
        - return DataFrame indexed by id_col with columns
          [area_m2, centroid_x, centroid_y, centroid_lat, centroid_lon]

   b. master_idx = pd.date_range(period_min, period_max, freq="MS")

   c. sources_cfs = {}
      for src in runoff_cfg.sources:
          var = catalog.runoff_variable_for(src)   # ro / runoff_total / runoff
          da_native = read_aggregated_source(project, src, var, period)
          da_mm = source_unit_shim[src](da_native)         # → mm/month
          da_cfs = mm_per_month_to_cfs(da_mm, hru_meta.area_m2)
          sources_cfs[src] = reindex_to_month_start(da_cfs, master_idx)

   d. lower, upper, n_sources = multi_source_nanminmax(sources_cfs)

   e. ds = assemble Dataset:
          data_vars = {lower_bound, upper_bound, n_sources}
          coords    = {time, time_bnds, <id_col>, centroid_lat, centroid_lon}
          attrs     = CF-1.6 + provenance (sources, period, fabric_sha256, software_version, area_crs)

   f. write_target_nc(ds, project.targets_dir / "runoff_targets.nc", title=...)

3. if runoff_cfg.nn_fill:
       centroids_xy = hru_meta[["centroid_x", "centroid_y"]].values
       filled_ds = nn_fill_bounds(ds, centroids_xy, runoff_cfg.nn_max_candidates)
       write_target_nc(filled_ds, project.targets_dir / "runoff_targets_nn_filled.nc",
                       title="NHM runoff calibration target (NN-filled)")

4. update manifest.json with both output files (paths + sha256 + period + sources)
```

### Output NetCDF schema

```
dimensions:
  time = N        (monthly, freq="MS")
  <id_col> = M    (e.g. nhm_id)
  nv = 2          (for time_bnds)

coordinates:
  time(time)             datetime64[ns]
                         attrs: standard_name="time", long_name="time at month start",
                                bounds="time_bnds", axis="T"
  time_bnds(time, nv)    datetime64[ns]   -- [month_start, next_month_start)
  <id_col>(<id_col>)     int64
                         attrs: long_name="HRU identifier", cf_role="timeseries_id"

ancillary coords (on <id_col>):
  centroid_lat(<id_col>) float64, units="degrees_north", standard_name="latitude"
  centroid_lon(<id_col>) float64, units="degrees_east",  standard_name="longitude"

data variables:
  lower_bound(time, <id_col>) float32, units="cfs",
                              long_name="lower bound of monthly runoff (NaN-aware min across sources)",
                              cell_methods="time: sum",
                              coordinates="centroid_lat centroid_lon"
  upper_bound(time, <id_col>) float32, units="cfs",
                              long_name="upper bound of monthly runoff (NaN-aware max across sources)",
                              cell_methods="time: sum",
                              coordinates="centroid_lat centroid_lon"
  n_sources(time, <id_col>)   int8, units="1",
                              long_name="number of finite source contributions at this HRU/time",
                              flag_values=[0,1,2,3], flag_meanings="none one two three",
                              coordinates="centroid_lat centroid_lon"

global attrs:
  Conventions = "CF-1.6"
  title = "NHM runoff calibration target (lower/upper bounds in cfs)"
  institution = "USGS"
  references = "Hay et al. 2022, doi:10.3133/tm6B10"
  history = "<ISO timestamp> created by nhf_spatial_targets vX.Y.Z"
  source = "ERA5-Land ro; GLDAS-2.1 NOAH Qs_acc+Qsb_acc; MWBM ClimGrid runoff"
  fabric = "<fabric.path basename>"
  fabric_sha256 = "<from fabric.json>"
  period = "<period_min>/<period_max>"
  software_version = "<package version>"
  area_crs = "EPSG:5070"

encoding:
  zlib=True, complevel=4 on lower_bound, upper_bound, n_sources
  _FillValue = NaN (float32 vars), -1 (n_sources int8)
  time uses CF "days since <epoch>" with calendar="proleptic_gregorian"
```

The `*_nn_filled.nc` file is identical except:

- Adds `nn_filled(time, <id_col>)` int8 with `flag_values=[0,1]`, `flag_meanings="not_filled filled"`.
- `lower_bound` / `upper_bound` have NaNs replaced where `nn_filled == 1`.
- Global attrs add `nn_fill_max_candidates` and `nn_fill_distance_crs`.
- Title prepended with "(NN-filled)".

### Defaults layer

`src/nhf_spatial_targets/defaults.py` — sketch:

```python
DEFAULTS = {
    "fabric": {
        "id_col": "nhm_id",
        "crs": "EPSG:4326",
        "buffer_deg": 0.1,
        "area_crs": "EPSG:5070",   # CONUS Albers; override for AK/HI/PR
    },
    "datastore": None,             # required
    "dir_mode": "2775",
    "aggregation": {"engine": "gdptools", "method": "area_weighted"},
    "output": {"dir": "outputs", "format": "netcdf", "compress": True},
    "targets": {
        "runoff": {
            "enabled": True,
            "sources": ["era5_land", "gldas_noah_v21_monthly", "mwbm_climgrid"],
            "time_step": "monthly",
            "period": None,        # required when enabled
            "prms_variable": "basin_cfs",
            "range_method": "multi_source_minmax",
            "output_file": "runoff_targets.nc",
            "nn_fill": True,
            "nn_max_candidates": 10,
            "chunk_months": 12,
        },
        "aet": { ... full default block including 3 sources ... },
        "recharge": { ... },
        "soil_moisture": { ... },
        "snow_covered_area": { ... },
    },
}
```

Merge semantics:

- `_deep_merge(defaults, user)` — user wins at every leaf; nested dicts recurse; **lists are replaced wholesale, not merged** (so `targets.runoff.sources: [era5_land]` in user config means just that one source — no surprise inclusion of the default three).
- `Project.target(name)` returns the merged per-target sub-dict.
- Required-key check happens after merge; missing required keys raise with the dotted path.

`validate` behavior:

1. Run the merge.
2. For every key the user didn't set that took a default, print to stderr:
   `[defaults] runoff.nn_fill not set; using default: True`
3. For every user-set key not present in `DEFAULTS`, print to stderr:
   `[warning] unknown config key: runoff.nn_fil (typo? not in defaults schema)`
4. Required-keys check; raise on missing.
5. Existing fabric / credential / catalog checks run unchanged.
6. Write `<project>/config.effective.yml` (mode `0o444`) with the fully-merged config and leading `# AUTOGENERATED by nhf-targets validate. Do not edit; edit config.yml.` comment. Always written, every run.
7. Existing `fabric.json` write happens after.

Migration path for an existing HPC project: `nhf-targets validate --project-dir <hpc-path>` shows which new keys took defaults; the user decides whether to override (e.g. set non-CONUS `area_crs`, disable `nn_fill`); old `config.yml` files just work.

### Error handling

**Boundary errors (fail fast, clear message):**

- Missing required key (e.g. `datastore`, `fabric.path`, `targets.runoff.period`) → `ValueError` with the dotted path.
- Unknown config key → warn to stderr, continue.
- Invalid `area_crs` (un-parseable EPSG) → raise on first use during area compute.
- Invalid `period` format → raise during load with example `YYYY-MM-DD/YYYY-MM-DD`.
- Source listed in `runoff.sources` not in `catalog/sources.yml` → raise with the bad key + valid runoff-eligible source keys.
- Source listed but its aggregated dir is empty → raise with the path checked + the `pixi run nhf-targets agg <src>` command to fix it.

**Trust-internal errors (let propagate):**

- `xr.open_mfdataset` raising on corrupt NCs.
- Fabric file unreadable.
- Disk full on write — atomic-write `try/except BaseException` un-links tempfile and re-raises.

**Substantive runtime conditions (warn but proceed):**

- A source's per-year NCs cover only part of the requested `period` → log INFO with the actual coverage range. (Designed-for case: MWBM ends 2020; user requests through 2024.)
- A source contributes zero finite values for the entire run → log WARNING; continue. Output's `n_sources` shows the audit signal.
- All sources NaN at some (HRU, time) cells → log INFO with count + percentage.
- NN-fill exhausts `nn_max_candidates` for some cells → log WARNING with count; cells stay NaN.
- Time coverage of a source extends *beyond* the requested period → silently slice.

**No silent-failure traps:**

- No bare `try/except: pass`. No broad `except Exception`.
- HRU coord mismatch between sources after reindex (fabric ID set differs) → raise. Different sources aggregated against different fabrics is a real bug; silently dropping HRUs would corrupt the target.

### Memory profile

Target: 16 GB laptop should comfortably build runoff for gfv2.0 (~250k HRUs × 25 years × 3 sources, NN-fill on).

- Source NCs read lazily via `xr.open_mfdataset(..., chunks={"time": chunk_months, id_col: -1})`; reindex/unit-conversion stay lazy.
- `xr.concat` on `source` dim and `.min/max(skipna=True)` are dask-native.
- `to_netcdf()` triggers compute and streams chunk-by-chunk; peak resident memory ≈ a few chunk-windows.
- `cKDTree` on M centroids ≈ `M × 2 × 8` bytes = ~4 MB for 250k HRUs.
- `compute_hru_area_and_centroids` DataFrame: M rows × 5 floats ≈ 12 MB.

Worst-case peak ≈ 500 MB resident, ~1 GB if dask happens to hold a couple chunk-windows. Comfortable on 16 GB. `runoff.chunk_months` (default 12) is the primary tunable for memory-constrained machines.

### Testing

Coverage rule (per CLAUDE.md): every new module gets a test file before the PR is opened. Bug-fix-style: a test covering the period-union semantics should fail against today's `targets/run.py` (which is broken vs. the per-year layout).

Test files (all in-process, no network, no real source data — runs under `pixi run -e dev test`, not `test-integration`):

- `tests/test_targets_common.py` — `read_aggregated_source` (full overlap, partial overlap, period-outside-range raises); `reindex_to_month_start` (EOM/SOM/mid-month all map to same MS slot); `multi_source_nanminmax` (3-source synthetic; n=3 / n=2 / n=0 cells); `compute_hru_area_and_centroids` (3-polygon synthetic GeoPackage, areas within 1% of analytical truth in EPSG:5070, centroid CRS round-trip); `write_target_nc` (CF-1.6 round-trip via xarray, `time_bnds` shape, attrs preserved).

- `tests/test_targets_run.py` — fixture: 3 synthetic per-year aggregated NCs (era5_land, gldas, mwbm) for 5 HRUs × 24 months with deliberate value patterns. Tests: period-union (mwbm covers only first year → second year `n_sources=2`); unit-chain (known mm/month + area → cfs match to 1e-6); NN-fill on/off (both files exist, filled has fewer NaNs); HRU-mismatch raises; output schema matches Section "Output NetCDF schema."

- `tests/test_normalize_nn_fill.py` — hand-built 5-HRU centroid layout; donor is geometrically nearest finite HRU; donor walk when nearest is itself NaN; `max_candidates` cap leaves cells NaN; per-time-step independence (NaN at t=0 doesn't pollute t=1).

- `tests/test_workspace_defaults.py` — minimal user config resolves all five targets; user override on `runoff.sources` is list-replace (not merge); unknown key warns; missing required key raises with dotted path; `config.effective.yml` round-trips and is mode `0o444`.

- `tests/test_validate_runoff_config.py` — small, runs in default test env: bad `area_crs` raises on first area compute; source-not-in-catalog raises with helpful message.

CF-compliance: xarray round-trip + assert attrs is sufficient (no external `cfchecker`).

### CLI surface

- `pixi run run-runoff -- --project-dir <dir>` — single-target invocation (matches `run-aet` convention named in CLAUDE.md).
- `pixi run nhf-targets run --target runoff --project-dir <dir>` — same effect via the underlying CLI.
- `pixi run nhf-targets run --project-dir <dir>` — builds all enabled targets in order (runoff → aet → recharge → soil_moisture → snow_covered_area). Stub targets log a `WARNING: target not yet implemented; skipping` and exit normally so bulk runs don't fail noisily.

### HPC integration: `run_all.slurm`

Mirrors `agg_all.slurm` pattern: 5-element array, one slurm task per target, 32 GB / 4 h defaults. Stub targets (AET, RCH, SOM, SCA) log "not yet implemented; skipping" and exit 0 so the array doesn't generate spurious failures. As each target lands, the skip becomes a real build with no slurm-script change.

```bash
#!/bin/bash
#SBATCH --job-name=nhf-run
#SBATCH --account=impd
#SBATCH --partition=cpu
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/run_%a_%A.out
#SBATCH --error=logs/run_%a_%A.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets}"
PROJECT_DIR="${PROJECT_DIR:-/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets}"

cd "$REPO_DIR" || { echo "ERROR: REPO_DIR=$REPO_DIR not found" >&2; exit 1; }

RUN_TASKS=(
    "run-runoff"             # 0
    "run-aet"                # 1 (stub)
    "run-recharge"           # 2 (stub)
    "run-soil-moisture"      # 3 (stub)
    "run-snow-covered-area"  # 4 (stub)
)

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#RUN_TASKS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range" >&2
    exit 2
fi

TASK="${RUN_TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Array task $SLURM_ARRAY_TASK_ID: $TASK ==="
echo "=== Project: $PROJECT_DIR ==="
echo "=== Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Host:    $(hostname) ==="

pixi run "$TASK" -- --project-dir "$PROJECT_DIR"

echo "=== Done:   $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
```

---

## Doc, catalog & config updates

### `docs/references/calibration-target-recipes.md` § 1 Runoff — rewrite

- Add MWBM ClimGrid `runoff` as the third source (currently lists only ERA5+GLDAS).
- Add MWBM unit row: already mm/month, no conversion (cite catalog as authoritative).
- Update the combination rule:
  `lower = nanmin(era5_cfs, gldas_cfs, mwbm_cfs)`,
  `upper = nanmax(...)`,
  with explicit note that `nanmin`/`nanmax` semantics (equivalent to `xr.concat(...).min(skipna=True)`) means a bound is defined whenever ≥1 source is finite. Cross-link to the existing "Aggregation, masked_mean, and target-time NN-fill" section.
- New "Period semantics" sub-block: requested period drives a master month-start time index; each source contributes for `[period_min, period_max] ∩ source_native_range`; gaps outside a source's native range simply don't contribute (NaN, doesn't poison the bound).
- New "Time canonicalization" sub-block pointing to the EOM/SOM canonicalization rule and `time_bnds`.
- New "NN-fill" sub-block: optional, default-on, post-processing artifact written to `*_nn_filled.nc`; only fills cells where all sources are NaN; donor walk in equal-area CRS; `nn_filled` diagnostic.
- New "Project config keys" sub-block listing `runoff.sources`, `runoff.period`, `runoff.nn_fill`, `runoff.nn_max_candidates`, `runoff.chunk_months`, `fabric.area_crs` with defaults, pointing readers at `defaults.py` as the authoritative source.

### `docs/references/processing-steps-reference.md` § 1 Runoff — minor edits

- "Cross-cutting conventions → NaN HRU fill" block currently says NN-fill is in `normalize/` and runs "before any combination or normalization" — update to match the agreed design: fill is a **target-stage post-processing step on the bounds**, runs *after* combination, writes a separate `*_nn_filled.nc`. (This contradicts the calibration-target-recipes version of the same rule; both should land consistent in this PR.)
- Add a one-line note that `n_sources(time, id_col)` and `nn_filled(time, id_col)` diagnostics are written to the output.
- Verify per-source bullets still match (they do — ERA5/GLDAS/MWBM all named, unit conversions correct).

### `catalog/variables.yml` `runoff:` block

- Update `range_notes` to say `lower_bound = nanmin(era5, gldas, mwbm)` (currently `min(...)` which is ambiguous — could imply NaN-propagating). Same for upper_bound.

### `catalog/sources.yml`

No changes needed for runoff sources; all three are already documented with `status: current`.

### `config/pipeline.yml` (REFERENCE ONLY) and `init_run.py` `_CONFIG_TEMPLATE`

Both have the runoff block with only 2 sources today. Update both to match the new defaults:

```yaml
fabric:
  path: /path/to/fabric.gpkg
  id_col: nhm_id
  crs: EPSG:4326
  buffer_deg: 0.1
  area_crs: "EPSG:5070"   # override for AK/HI/PR
# ...
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
```

### `CLAUDE.md`

- Append `pixi run run-runoff -- --project-dir /data/nhf-runs/my-run` to the example command block at the top.
- Expand the NN-fill placement bullet under Aggregation Transformation Policy: "NN-fill, when applied, is a target-stage post-processing step on the bounds; produces a separate `*_nn_filled.nc` artifact alongside the honest-NaN NC."

### `pixi.toml`

Add task aliases:
```toml
run-runoff = "nhf-targets run --target runoff --project-dir"
run-aet = "nhf-targets run --target aet --project-dir"
run-recharge = "nhf-targets run --target recharge --project-dir"
run-soil-moisture = "nhf-targets run --target soil_moisture --project-dir"
run-snow-covered-area = "nhf-targets run --target snow_covered_area --project-dir"
```

---

## Build sequence (for the implementation plan to expand)

1. `defaults.py` + `workspace.py` merge layer + `tests/test_workspace_defaults.py`. Lands first because everything else reads through it.
2. `validate.py` updates (defaults diff report, unknown-key warning, `config.effective.yml` write) + tests.
3. `targets/_common.py` (read, reindex, combine, area/centroids, write) + `tests/test_targets_common.py`.
4. `normalize/nn_fill_bounds` + `tests/test_normalize_nn_fill.py`.
5. `targets/run.py` rewrite + `tests/test_targets_run.py`.
6. `cli.py` `run-runoff` subcommand + `--target` flag + integration smoke test.
7. `pixi.toml` task aliases + `run_all.slurm` + manual HPC dry-run.
8. Doc/catalog/config updates: `calibration-target-recipes.md`, `processing-steps-reference.md`, `catalog/variables.yml`, `config/pipeline.yml`, `init_run.py` template, `CLAUDE.md`.
9. Stub-target `WARNING: not yet implemented; skipping` behavior in the four other target builders so `nhf-targets run` (no `--target`) and `run_all.slurm` don't fail.

Each step lands as a passing-tests checkpoint before the next begins.

---

## Risks & open questions

- **MWBM time convention is unverified.** Catalog says monthly; assumed start-of-month based on the CF NetCDF distribution but the `reindex_to_month_start` helper handles all three conventions safely, so this is robust to the assumption being wrong. Confirm during implementation by reading the actual MWBM aggregated NC.
- **Period-outside-source-range raise.** A source whose entire native range is outside `[period_min, period_max]` will raise from `read_aggregated_source`. This is intentional (the user listed a source that contributes nothing). Could be downgraded to "warn and skip" later if it gets in the way; design starts strict.
- **`nn_fill` default-on means slower default builds and more disk.** Trade-off accepted: the second file is purely additive and helps downstream PRMS calibration; experienced users can disable with `runoff.nn_fill: false`.
- **Stub-target skip-on-not-implemented changes a `NotImplementedError` raise to a log line.** This is a behavior change; document it in CLAUDE.md and the relevant target stubs so it's not surprising.
