# NetCDF encoding policy (chunking + compression)

This document is the architectural reference for **how** the pipeline chunks
and compresses the NetCDFs it writes. It exists because the pipeline emits NCs
at three stages — consolidate, aggregate, target — and naive (or absent)
encoding choices silently cost disk and make the dominant calibration read
pattern slow.

If you are writing a new module that calls `to_netcdf`, this is the file that
tells you to route through `io_nc` instead, and why.

## TL;DR

- **Never call `ds.to_netcdf(...)` directly.** Use
  [`io_nc.build_encoding`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/io_nc.py) to build the
  encoding dict and `io_nc.atomic_to_netcdf` to write it. One module owns the
  policy so a future codec change is a one-file edit.
- **Chunk `(time, hru)` data variables `(timesteps_per_file, chunk_hru)`**,
  where `chunk_hru = ceil(target_chunk_bytes / (timesteps_per_file × dtype_bytes))`
  capped at the HRU count, with `target_chunk_bytes = 1 MiB`. This puts a
  single HRU's full time series in ~one HDF5 chunk — the read calibration
  does on every iteration.
- **Compress** with `zlib` `complevel=4`. `shuffle=True` for integer dtypes,
  **explicitly `False` for floats** (netCDF4 defaults shuffle to `True` under
  zlib, so the key must be set, not omitted).
- **Preserve native dtypes** in the aggregated layer (no float64→float32
  downcast — that is a values change, not a storage change).
- **Pin time** to `units="days since 1970-01-01 00:00:00"`,
  `calendar="proleptic_gregorian"`, `dtype="float64"` on `time`/`time_bnds`,
  on every layer.
- **Backfill existing files** with `nhf-targets rechunk` rather than
  re-aggregating.

## The principle

> **HDF5 cannot read part of a chunk.** To return any slice of a chunked
> variable, the whole chunk that the slice touches must be read (and, if
> compressed, decompressed). So the chunk *shape* decides what each access
> pattern costs. The calibration loop reads one HRU's full time series over
> and over; chunking the time axis whole and the HRU axis narrow makes that
> read one ~1 MiB chunk instead of the entire file.

The corollary, and the one tradeoff to keep in mind: a **spatial snapshot**
(one timestep, all HRUs — what the inspect notebooks do) now touches *every*
chunk along the HRU dimension. That is unavoidable under a
calibration-optimized layout and is accepted: the calibration time-series read
is the dominant, repeated pattern; the snapshot read is occasional.

## Per-layer policy

| Layer | Writer | `build_encoding(layer=...)` | Notes |
|---|---|---|---|
| **Consolidated** (`<datastore>/<src>/`) | `fetch/consolidate.py` | `"consolidated"` — **seam owned by issue #158** (per-source spatial tiling of `(time, y, x)` grids). `build_encoding` raises `NotImplementedError` until #158 lands. | Different shape (gridded, not HRU), different policy (tile `y`/`x`). |
| **Aggregated** (`<project>/data/aggregated/<src>/`) | `aggregate/_driver.py::_atomic_write_netcdf` | `"aggregated"`, `hru_dim=id_col`, `timesteps_per_file=ds.sizes["time"]`. Native dtype preserved. | daymet (zarr) + ssebop (STAC) are **left as-is** — already chunked, remote-sourced. The `crs` grid-mapping container is skipped (never compressed/filled). |
| **Target** (`<project>/targets/`) | `targets/_writers.py::write_target_nc` | `"target"`, `hru_dim=sort_dim`. float32 bounds, int8 diagnostics. | The streaming `stitch_year_chunks_to_target` (daily SWE) is a separate concern — see ST3b. |

### `_FillValue` policy (dtype-driven, in `io_nc._fill_value_for`)

- floating dtypes → `NaN`
- `int16` (packed) → `-9999` (the SNODAS/aggregate sentinel)
- all other integers (notably the `int8` diagnostics) → no fill value

This matches the pre-#165 `targets/_writers.py` writer for the float and int8
cases; the int16 path is the aggregate-layer convention.

## Reading aggregated NCs

`targets/_io.py::read_aggregated_source` reads per-year aggregated files
via `open_mfdataset`. Its default dask chunks are `{"time": -1, id_col:
chunk_hru}` — **full time per file, HRU dim chunked to ~256 MiB**. This is
deliberately *not* the on-disk 1 MiB chunk size: it trades chunk granularity
for a smaller dask graph on a full-source read.

The reason `time` must not be sub-chunked: `open_mfdataset` chunks per file
along the concat dim, and a file's on-disk time chunk spans its whole time
axis. A dask time chunk smaller than the file (the old `{"time": 12}` default)
would force the compressor to re-inflate every on-disk chunk on every time
slab — ~30× read amplification on a daily source. Full-time dask chunks read
each on-disk chunk exactly once.

## Migration: `nhf-targets rechunk`

The writers only chunk NCs they newly write. Projects built before this work
hold contiguous, uncompressed NCs. Backfill them in place:

```bash
nhf-targets rechunk --project-dir <dir> [--layer aggregated|target] [--source <key>] [--dry-run]
```

It is **idempotent** (skips files already fully chunked), **atomic**
(`<file>.rechunk.tmp` → rename), **value-preserving** (every variable —
data + coords — compared decoded and NaN-aware before the rename; a mismatch
aborts that file untouched), and **per-file isolated** (one bad file is
reported and the run continues; the CLI exits non-zero if any failed). It never
touches the datastore's consolidated NCs, and skips daymet/ssebop. Prefer it
over re-aggregating (which is multi-day SLURM time to produce identical data in
a different layout).

## What the win actually is

Measured on real gfv2 aggregated NCs (rewritten with this policy,
bit-identity confirmed, float64 preserved):

| Source | Cadence / dtype | Before | After | Reduction |
|---|---|---|---|---|
| snodas | daily, sparse float64 | 1060 MB | 158 MB | **85%** |
| era5_land | monthly, dense float64 | 114 MB | 98 MB | 14% |

The dominant, source-dependent win is **disk footprint** — large for
sparse/NaN-heavy daily sources, modest for dense monthly data. Per-HRU
single-column read latency improves only modestly (~1.2–2.2×, not 10×):
zlib decompression of the chunk offsets the chunk-locality gain for the
single-HRU pattern. Disk pressure on the shared filesystem was the motivating
concern, and that is where the policy pays off.

## Cross-references

- [`io_nc.py`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/io_nc.py) — `build_encoding` /
  `atomic_to_netcdf` (the single policy home).
- [`rechunk.py`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/rechunk.py) — the backfill CLI.
- [transformation-pipeline.md](transformation-pipeline.md) — *where* transforms
  live (this doc is *how* NCs are encoded); canonical `id_col`-ascending row
  order (issue #93) is preserved through all of this.
- Issue #165 (this work: aggregated + target layers) and issue #158
  (consolidated layer tiling).
