# MODIS Fetch/Consolidate — Logging & Memory Efficiency

**Date:** 2026-03-14
**Status:** Approved
**Scope:** `src/nhf_spatial_targets/fetch/modis.py`, `src/nhf_spatial_targets/fetch/consolidate.py`

## Problem

Running `nhf-targets fetch mod16a2` for a single year (2001) finds 1104 granules (~24 CONUS tiles x 46 eight-day periods). The current implementation:

1. Downloads all 1104 granules in a single `earthaccess.download()` call with no per-batch progress logging.
2. Consolidation (`consolidate_mod16a2`) accumulates all ~46 mosaicked/reprojected timestep datasets in a Python list before `xr.concat` and writing — peak RAM holds every timestep simultaneously.
3. When the process OOMs, the kernel kills it with no diagnostic output — no logging indicates where it failed or how much memory was in use.

## Design

### Core change: per-timestep pipeline

Shift from **download-all → consolidate-all** to **per-timestep download → consolidate → write-temp**, then lazy-concat temp files into the final output.

### Responsibility boundary

`modis.py` owns the per-timestep download loop and orchestrates the pipeline. `consolidate.py` owns all consolidation logic. The per-timestep loop in `fetch_mod16a2` calls `consolidate_mod16a2_timestep` (new public function in consolidate.py) after each download batch, then calls `consolidate_mod16a2_finalize` (new public function) to lazy-concat and write the final file. The existing `consolidate_mod16a2` becomes a convenience wrapper that calls both (for standalone re-consolidation from existing HDF files).

### `modis.py` changes (fetch_mod16a2)

1. **Search granules** — unchanged (one `earthaccess.search_data` per year).
2. **Group granules by timestep** — parse AYYYYDDD token from granule metadata/URLs before downloading. Produces ~46 groups of ~24 tiles each.
3. **Download per-timestep batch** — call `earthaccess.download()` once per timestep group (~24 granules). Log: `"Downloading timestep 12/46 (A2001089): 24 granules"`.
4. **Consolidate each timestep immediately** — after downloading a batch, call `consolidate_mod16a2_timestep()` which writes a temp NetCDF and returns its path.
5. **Finalize** — after all timesteps, call `consolidate_mod16a2_finalize()` with the list of temp file paths to produce the final consolidated file.
6. **Log memory at checkpoints** — after each download batch and after each timestep consolidation.

#### Granule grouping

earthaccess granule objects expose metadata including the granule filename. Extract the AYYYYDDD token from each granule's filename to group them by timestep before downloading. This uses the same `_MODIS_YEAR_RE`-style regex already in the module.

### `consolidate.py` changes

#### New function: `consolidate_mod16a2_timestep`

Public function. Extracts the current per-timestep logic (mosaic tiles, reproject, squeeze/rename dims) into a function that:

1. Takes a list of HDF tile paths for one timestep, the variable list, and the source directory.
2. Mosaics and reprojects (existing `_mosaic_and_reproject_timestep` — unchanged).
3. Writes the result to a temp NetCDF with a unique name (see temp file convention below).
4. Returns the temp file path.
5. Closes all intermediate arrays before returning.

#### New function: `consolidate_mod16a2_finalize`

Public function. Takes a list of temp file paths and the output path:

1. Opens all temp files via `xr.open_mfdataset(tmp_paths, combine="by_coords", chunks={})` — lazy, dask-backed. The `combine="by_coords"` strategy works because each temp file has a single time coordinate value.
2. Sorts by time.
3. Validates requested variables are present.
4. Writes final `{source_key}_{year}_consolidated.nc` with `ProgressBar()`.
5. Cleans up all temp files on success.
6. On failure during the final write, cleans up all temp files before raising.

#### Refactored `consolidate_mod16a2`

Becomes a convenience wrapper for standalone re-consolidation:

1. Groups HDF files by timestep (existing logic).
2. Calls `consolidate_mod16a2_timestep` for each group.
3. Calls `consolidate_mod16a2_finalize` with the collected temp paths.

#### Temp file convention

- Format: `_tmp_{PID}_A{YYYYDDD}.nc` (e.g. `_tmp_12345_A2001001.nc`)
- Location: same directory as HDF files (`data/raw/mod16a2_v061/`)
- PID suffix prevents collisions between concurrent processes
- `_tmp_` prefix makes them identifiable for manual cleanup if the process dies
- Stale temp file cleanup at start of consolidation only removes files matching the current PID (not files from other processes)

### Logging additions

All logging uses the existing `logger` (module-level `logging.getLogger(__name__)`).

#### Memory reporting helper

```python
def _log_memory(label: str) -> None:
    """Log current RSS from /proc/self/status (Linux) or peak RSS as fallback."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    rss_gib = rss_kb / (1024 ** 2)
                    logger.info("[memory] RSS=%.2f GiB — %s", rss_gib, label)
                    return
    except OSError:
        pass
    # Fallback: peak RSS via resource (Unix only; reports high-water mark)
    try:
        import resource
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_gib = peak_kb / (1024 ** 2)
        logger.info("[memory] peak RSS=%.2f GiB — %s", peak_gib, label)
    except (ImportError, OSError):
        logger.debug("[memory] cannot read RSS on this platform — %s", label)
```

Uses `/proc/self/status` `VmRSS` for current (not peak) RSS on Linux. Falls back to `resource.getrusage` peak RSS on other Unix systems. Silently degrades on unsupported platforms.

Called at these checkpoints:
- After earthdata authentication
- After granule search (with count)
- After each timestep download batch
- After each timestep consolidation (temp file written)
- After final consolidated file write

#### Progress logging

- Per-timestep download: `INFO "Downloading timestep 12/46 (A2001089): 24 granules"`
- Per-timestep consolidation: tqdm bar (already used in consolidate.py) plus `INFO "Consolidated timestep 12/46 → _tmp_12345_A2001089.nc"`
- Final write: `INFO "Writing final consolidated file from 46 timestep files"`

#### Debug-level detail

- Per-file download paths at DEBUG
- Per-tile mosaic/reproject at DEBUG (already partially exists)

### Disk space

The per-timestep approach writes ~46 temp NetCDF files per year alongside the original HDF files. Each temp file is a single CONUS-extent timestep at 0.04° resolution (~1650x800 pixels, 2 variables, float32) — roughly 4 MB per file, ~200 MB total for a year. This is small relative to the ~5-10 GB of source HDF tiles and is cleaned up after the final write.

### What does not change

- `fetch_mod10c1` — MOD10C1 is global CMG (one file per day, already small after CONUS subset), no tiling/memory issue.
- `_mosaic_and_reproject_timestep` internals — already closes tile arrays after mosaic.
- Manifest update logic, provenance dict structure, CLI layer.
- The return dict from `fetch_mod16a2` keeps the same shape (file inventory still built from all HDF files on disk after all downloads complete).
- Unit tests for provenance, manifest updates, incremental skipping.

### Error handling

- **Download failure**: log which timestep batch failed (timestep index, AYYYYDDD token, granule count), raise with context.
- **Per-timestep consolidation failure**: clean up the temp file for the failed timestep (if partially written), log which timestep failed, raise with context.
- **Final write failure**: clean up all temp files, log error, raise with context.
- **Partial download within a batch**: existing warning logic (`len(downloaded) < len(granules)`) applies per-batch instead of per-year.
- **Stale temp files**: at the start of consolidation, remove any `_tmp_{current_PID}_*.nc` files (from a prior interrupted call in the same process). Files from other PIDs are left alone.

### Testing

- Existing unit tests in `test_modis.py` and `test_consolidate_modis.py` need updates to reflect the per-timestep pipeline.
- Add a unit test for granule grouping by timestep.
- Add a unit test for `consolidate_mod16a2_timestep` — mock `_mosaic_and_reproject_timestep`, verify temp file is written and path returned.
- Add a unit test for `consolidate_mod16a2_finalize` — create synthetic temp NetCDFs, verify final concat and cleanup.
- Add a unit test for temp file cleanup on failure (simulate error during finalize, verify temps are removed).
- Add a regression test: verify the refactored pipeline produces identical output to the original for a small synthetic dataset.
- Integration test pattern unchanged — small bbox, single year.
