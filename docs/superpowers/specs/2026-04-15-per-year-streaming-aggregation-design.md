# Per-Year Streaming Aggregation — Design

**Date:** 2026-04-15
**Status:** Proposed
**Related:** PR branch `fix/modis-aggregator-open-consolidated-nc`; builds on commits `d178653`, `fb323c5`, `587f457`.

## Problem

`aggregate_source` in `src/nhf_spatial_targets/aggregate/_driver.py` opens every `*_consolidated.nc` under `project.raw_dir(source_key)`, concatenates them into a single in-memory `xr.Dataset`, then loops spatial batches over the full time range. For MOD16A2 v061 and MOD10C1 v061 (per-year consolidated NCs across 2000–2023) this exceeds available memory on the HPC nodes used by `pixi run nhf-targets agg …` job arrays (array=7 reproduced the failure).

Adapters also hard-code coordinate names (`x_coord="x"`, `y_coord="lat"`, etc.). The MOD10C1 consolidated NCs carry `['time','lat','lon']` but the adapter assumed `time,y,x`. Coordinate resolution must come from the dataset, not the adapter.

## Goals

1. Stream aggregation year-by-year so peak memory is one year of source grid data plus one batch's weights and results.
2. Detect coordinate names from CF metadata on the opened dataset; fail loudly when detection fails.
3. Support restart: a killed job resumes at the first year without a per-year intermediate output, without recomputing earlier years.
4. Apply uniformly to every adapter — including single-file sources (MERRA-2, WaterGAP, NCEP-NCAR), which benefit via gdptools' lazy period clip.
5. Produce a single CF-1.6 compliant `data/aggregated/<source_key>_agg.nc` identical in schema to today's output.

## Non-Goals

- Changing the fetch layer or consolidated NC filenames.
- Changing the weight cache layout (`weights/<source_key>_batch<id>.csv`).
- Changing the manifest schema or CLI surface.
- Parallelizing across years within a single process (the HPC array scripts already shard by source; per-year parallelism is out of scope).

## Architecture

Four layers, all in `src/nhf_spatial_targets/aggregate/`:

1. **Coordinate detection** — new module `_coords.py`.
2. **Year enumeration** — `_driver.py`, reads `time` from each consolidated NC lazily.
3. **Per-year aggregation with intermediate output** — `_driver.py`.
4. **Final concat + manifest update** — `_driver.py`.

MOD10C1's bespoke masking logic moves into adapter hooks (`pre_aggregate_hook`, `post_aggregate_hook`), so `aggregate/mod10c1.py` collapses to an adapter definition + two small helper functions.

## Components

### `aggregate/_coords.py` (new)

```python
def detect_coords(
    ds: xr.Dataset,
    var: str,
    x_override: str | None = None,
    y_override: str | None = None,
    time_override: str | None = None,
) -> tuple[str, str, str]:
    """Resolve (x_coord, y_coord, time_coord) for ``ds[var]``.

    Resolution order per axis:
      1. Explicit override, if given and present in ``ds[var].dims``.
      2. Coordinate with CF ``axis`` attribute matching 'X' / 'Y' / 'T'.
      3. Coordinate with CF ``standard_name`` in:
           - X: 'longitude', 'projection_x_coordinate'
           - Y: 'latitude',  'projection_y_coordinate'
           - T: 'time'
    Raises ValueError listing ``ds[var].dims`` and their attrs if unresolved.
    No name-based fallback. Consolidated NCs are CF-1.6 per the project's
    fetch-layer conventions.
    """
```

### `aggregate/_adapter.py` (changed)

`SourceAdapter` changes:

- `x_coord`, `y_coord`, `time_coord` become `str | None = None`. The `__post_init__` CRS/catalog validation stays; coord validation against a variable is deferred to `detect_coords` at run time.
- New field `pre_aggregate_hook: Callable[[xr.Dataset], xr.Dataset] | None = None` — applied to each year's opened dataset before weight/aggregation calls. Used by MOD10C1 for CI masking.
- New field `post_aggregate_hook: Callable[[xr.Dataset], xr.Dataset] | None = None` — applied to the final concatenated dataset before the terminal atomic write. Used by MOD10C1 for the `valid_mask` → `valid_area_fraction` rename plus its low-coverage warning.
- `open_hook` semantics change: it must return a **lazy** Dataset (no `.load()`) or a mapping of year → lazy Dataset. The default hook globs `*_consolidated.nc` in `project.raw_dir(source_key)` and opens each lazily, keeping them independent (no concat).

### `aggregate/_driver.py` (changed)

New functions:

```python
def enumerate_years(
    files: list[Path], time_coord_hint: str | None = None
) -> list[tuple[int, Path]]:
    """Open each NC lazily, read the time coord, map covering years → file.

    Returns a sorted list of (year, file_path) with one entry per distinct
    year across all files. A single-file source may produce many tuples
    pointing at the same file; a per-year source produces one tuple per file.

    Raises ValueError if two files cover the same year (stale fetch) or if
    the time coord cannot be located (passes time_coord_hint through to
    detect_coords when provided; otherwise uses CF axis/standard_name only).
    """

def per_year_output_path(project: Project, source_key: str, year: int) -> Path:
    """Return <project>/data/aggregated/_by_year/<source_key>_<year>_agg.nc."""

def aggregate_year(
    adapter: SourceAdapter,
    project: Project,
    year: int,
    source_file: Path,
    fabric_batched: gpd.GeoDataFrame,
    id_col: str,
) -> Path:
    """Aggregate one year and atomically write the per-year intermediate.

    Idempotent: returns the per-year path immediately if it exists. Else:
      - open source_file lazily
      - apply adapter.pre_aggregate_hook
      - detect coords via _coords.detect_coords
      - for each spatial batch: compute-or-load weights, aggregate vars,
        collect per-batch xr.Dataset
      - concat on id_col
      - atomic write (tempfile + rename) to per_year_output_path
      - close source file, return path
    """

def concat_years(paths: list[Path], time_coord: str) -> xr.Dataset:
    """Open all per-year NCs lazily, concat on time, validate monotonic + unique."""
```

Refactored `aggregate_source`:

```python
def aggregate_source(adapter, fabric_path, id_col, workdir, batch_size=500):
    project = load_project(workdir)
    meta = catalog_source(adapter.source_key)

    files = sorted(project.raw_dir(adapter.source_key).glob("*_consolidated.nc"))
    if not files:
        raise FileNotFoundError(...)

    year_files = enumerate_years(files, time_coord_hint=adapter.time_coord)
    fabric_batched = load_and_batch_fabric(fabric_path, batch_size=batch_size)

    per_year_paths = [
        aggregate_year(adapter, project, year, path, fabric_batched, id_col)
        for year, path in year_files
    ]

    combined_lazy = concat_years(per_year_paths, time_coord=<resolved>)
    if adapter.post_aggregate_hook is not None:
        combined_lazy = adapter.post_aggregate_hook(combined_lazy)

    combined = combined_lazy.load()
    _attach_cf_global_attrs(combined, adapter.source_key, meta)

    output_path = project.aggregated_dir() / adapter.output_name
    _atomic_write_netcdf(combined, output_path)

    update_manifest(project, adapter.source_key, meta.get("access", {}),
                    period=<t0/t1 from combined>,
                    output_file=str(Path("data") / "aggregated" / adapter.output_name),
                    weight_files=[...])
    return combined
```

Helpers `compute_or_load_weights` and `aggregate_variables_for_batch` gain an explicit `period: tuple[str, str]` parameter and stop deriving period from `source_ds[time_coord]`. For per-year aggregation, callers pass `(f"{year}-01-01", f"{year}-12-31")`; gdptools' `UserCatData` applies the lazy time clip internally.

### `aggregate/mod10c1.py` (changed)

Collapses to:

```python
ADAPTER = SourceAdapter(
    source_key="mod10c1_v061",
    output_name="mod10c1_agg.nc",
    variables=("sca", "ci", "valid_mask"),
    grid_variable="sca",
    source_crs="EPSG:4326",
    pre_aggregate_hook=build_masked_source,
    post_aggregate_hook=_rename_and_warn,
)

def aggregate_mod10c1(fabric_path, id_col, workdir, batch_size=500):
    return aggregate_source(ADAPTER, fabric_path, id_col, workdir, batch_size)
```

`build_masked_source` is unchanged. `_rename_and_warn` wraps the existing
`valid_mask → valid_area_fraction` rename plus `_log_low_valid_coverage`.
Custom `_open` hook and the bespoke batch loop are deleted.

### `aggregate/mod16a2.py` (changed)

Unchanged in spirit — still an adapter. The sinusoidal PROJ override stays
as-is. `x_coord`/`y_coord` can be dropped (CF detection resolves them from
the consolidated NC's `projection_x_coordinate` / `projection_y_coordinate`
attrs). Keep them as explicit overrides for safety until the refactor is
verified against a full run.

### Other adapters

No code changes required — all use `aggregate_source` and inherit the
per-year pipeline automatically. Single-file sources (MERRA-2, WaterGAP,
NCEP-NCAR) run through `enumerate_years` producing many `(year, same_file)`
tuples; gdptools' period clip ensures only one year is read per iteration.

## Data Flow

```
project.raw_dir(source_key)/*_consolidated.nc
      │
      ▼  enumerate_years (lazy open → read time → close)
[(2000, f0), (2001, f1), … (2023, fN)]    # or many tuples → same file for single-NC sources
      │
      ▼  for each year (skip if _by_year/<src>_<year>_agg.nc exists)
         ├─ open source_file lazily
         ├─ ds = adapter.pre_aggregate_hook(ds)           # optional
         ├─ (x,y,t) = detect_coords(ds, adapter.grid_variable, overrides)
         ├─ for each spatial batch:
         │     weights = compute_or_load_weights(..., period=(y-01-01, y-12-31))
         │     per-batch ds = aggregate_variables_for_batch(..., period=...)
         ├─ concat batches on id_col
         ├─ atomic write → _by_year/<src>_<year>_agg.nc
         └─ close source_file
      │
      ▼  concat_years (lazy open _by_year/*, concat on time)
      ▼  adapter.post_aggregate_hook (optional)
      ▼  attach CF-1.6 global attrs
      ▼  atomic write → data/aggregated/<src>_agg.nc
      ▼  update_manifest
```

## Failure Modes

| Condition | Behavior |
|---|---|
| No `*_consolidated.nc` in raw_dir | `FileNotFoundError` pointing at `nhf-targets fetch <source>`. |
| Two files cover same year | `ValueError` from `enumerate_years` listing both filenames. |
| `time` coord not resolvable | `ValueError` from `detect_coords` listing dims + attrs. |
| X/Y coord not resolvable | `ValueError` from `detect_coords` listing dims + attrs. |
| Per-year intermediate exists but is corrupt | `ValueError` — operator deletes and retries (no silent re-aggregate). |
| Duplicate / non-monotonic time after `concat_years` | `ValueError` listing offending years. |
| Atomic-write tempfile survives crash | Replaced on next successful write (existing tempfile+rename pattern). |
| Final `<src>_agg.nc` already exists | Overwritten atomically; manifest updated. |
| User wants a full rebuild | `rm -rf <project>/data/aggregated/_by_year/<src>_*` before rerunning. |

## Restart Semantics

- Rerunning `nhf-targets agg <source>` is idempotent: `aggregate_year` skips years whose per-year NC already exists, so a killed job resumes at the first missing year.
- The weight cache is unchanged and remains restart-safe — grid geometry is year-independent.
- Per-year intermediates are **preserved** after the final concat (audit trail; small size: HRU × days_in_year). A future `--clean-intermediates` flag may be added; out of scope here.

## Memory Model

Per-year peak state:

- One year of lazily-opened source grid (gdptools clips via `period`).
- One batch's weight table (already cached on disk).
- One batch's aggregated Dataset (HRU_batch × time_year × variables).

All previous-year state is closed before the next year begins. Single-file sources benefit identically because `UserCatData(period=...)` triggers a lazy time subset.

## CF Compliance

- Per-year intermediates inherit variable attrs from the adapter / `pre_aggregate_hook`.
- Final concat preserves those attrs; `_attach_cf_global_attrs` sets top-level `Conventions: "CF-1.6"`, `history` (ISO8601 of the concat), `institution`, `source`, and `source_doi` from `catalog_source(source_key).access`.
- `to_netcdf` writes with `format="NETCDF4"` (default) and no special encoding changes from today.

## Testing

New tests under `tests/`:

- `test_coords.py` — `detect_coords` resolution order, override precedence, CF `axis` vs `standard_name`, error messages on absent metadata.
- `test_driver_per_year.py` — `enumerate_years` on per-year and single-file inputs (including duplicate-year error), `aggregate_year` idempotency (rerun does not re-open source), `concat_years` monotonic/unique check.
- `test_driver_restart.py` — pre-populate `_by_year/<src>_<year>_agg.nc` for a subset of years and assert those years are not re-aggregated on rerun.

Existing tests updated:

- `tests/test_aggregate_driver.py` — adjust fixtures to the per-year interface; assert `period` is passed through to gdptools.
- `tests/test_aggregate_mod10c1.py` — verify the adapter + hooks path produces the same `sca` / `ci` / `valid_area_fraction` variables as the current implementation on a small synthetic fixture.

Integration tests (`@pytest.mark.integration`) for real MOD16A2 / MOD10C1 data remain out of scope for unit test suites; the HPC array scripts are the integration harness.

## Migration

- No manifest schema change. Existing `data/aggregated/<src>_agg.nc` outputs remain valid; rerunning will overwrite them via the new path.
- Existing `weights/<src>_batch<id>.csv` caches are reused unchanged.
- New directory `data/aggregated/_by_year/` appears on first run.

## Open Questions

None at brainstorm time. Any ambiguity surfaced during implementation should be recorded in the implementation plan, not decided silently.
