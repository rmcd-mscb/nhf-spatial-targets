# Daymet SWE aggregator (NA region) — PR 2 of 2 for issue #101

## Context

Issue #101 (umbrella) tracks SWE-category aggregate + target wiring.
PR #117 landed the SNODAS aggregator. This plan tackles **Daymet
(NA region only)** — the second of the four SWE sources.

Daymet's source is not a per-year NetCDF — it's a single multi-year
**zarr store** per region (`{daymet_root}/daymet_{na,hi,pr}.zarr`),
verified by `fetch.daymet` and recorded in `manifest.json`. The data
is on a **Lambert Conformal Conic** projected grid (not WGS84) with
projected `x`, `y` coords in metres. None of that fits the standard
`SourceAdapter`/`aggregate_source` machinery, which assumes per-year
NCs and CF-tagged lat/lon coords.

Per [src/nhf_spatial_targets/aggregate/ssebop.py](src/nhf_spatial_targets/aggregate/ssebop.py)
— the existing precedent for sources that don't fit the driver — the
solution is a **custom loop** that reuses the driver's helpers
(`compute_or_load_weights`, `aggregate_variables_for_batch`,
`_atomic_write_netcdf`, `_attach_cf_global_attrs`,
`_migrate_legacy_layout`, `_verify_year_coverage`,
`load_and_batch_fabric`, `update_manifest`) without going through
`aggregate_source`. Daymet's loop opens a zarr instead of a STAC
collection but is otherwise structurally identical.

Decisions locked in:
- **NA region only** for this PR. A `--region` flag is exposed now
  (default `"na"`), raising `NotImplementedError` for `hi`/`pr` until
  those fabrics need them.
- **Future-proof output naming**:
  `data/aggregated/daymet/daymet_<region>_<year>_agg.nc`. The region
  segment is in the filename so HI/PR can land later without renaming
  existing NA outputs.
- **Standalone slurm script** (`agg_daymet.slurm`), not added to
  `agg all` or `agg_all.slurm`. `agg daymet` requires `--period` (like
  ssebop), and `agg all` doesn't forward periods. Update the
  `agg_all_cmd` docstring to mention daymet alongside ssebop as
  excluded.

The SWE target builder (`targets/swe.py`) and the remaining SWE
sources (ERA5-Land `sd`, Margulis WUS-SR) stay tracked under #101 for
separate PRs after this one.

## Approach

### `aggregate/daymet.py` (new)

Function signature mirroring ssebop's, with a region kwarg added:

```python
def aggregate_daymet(
    fabric_path: str | Path,
    id_col: str,
    period: str,                  # required: 'YYYY/YYYY'
    workdir: str | Path,
    batch_size: int = 500,
    region: str = "na",
) -> None:
```

Body, in order:

1. **Region guard.** `if region != "na": raise NotImplementedError(...)`
   with a clear message pointing to issue #101's HI/PR follow-ups.
2. **Resolve zarr path** from `manifest.json["sources"]["daymet"]["regions"][region]["path"]`
   (written by `fetch.daymet._update_manifest` at
   [src/nhf_spatial_targets/fetch/daymet.py:399-412](src/nhf_spatial_targets/fetch/daymet.py#L399-L412)).
   Fail with a clear "run `nhf-targets fetch daymet` first" if absent.
3. **Open** with `xr.open_zarr(zroot, consolidated=False)` (matches
   the existing fetch-side open at
   [src/nhf_spatial_targets/fetch/daymet.py:175](src/nhf_spatial_targets/fetch/daymet.py#L175)).
   Select only `swe` (drop the other five vars) so gdptools doesn't
   chunk-walk them.
4. **Derive `source_crs`** from the zarr's `lambert_conformal_conic`
   grid-mapping variable via `pyproj.CRS.from_cf(grid_mapping.attrs)`.
   Convert to a WKT string for gdptools `WeightGen`. Fail with a clear
   error if the grid-mapping variable is missing.
5. **Validate period** (`'YYYY/YYYY'`, both ints, start ≤ end, within
   the zarr's actual time coord range). Reuse
   `nhf_spatial_targets.fetch._period.parse_period` for the format
   check (already used by `aggregate_source`); add the in-range check
   here.
6. **Call `_migrate_legacy_layout(project, "daymet")`** for symmetry
   with the standard driver — harmless no-op on first run.
7. **Load + batch fabric** via `load_and_batch_fabric`.
8. **Year loop** (`for year in range(start, end + 1)`):
   - Compute output path via a local helper:
     `_daymet_region_year_path(project, region, year)` →
     `<project>/data/aggregated/daymet/daymet_<region>_<year>_agg.nc`.
   - Skip-if-exists idempotency check (mirrors
     [aggregate/ssebop.py:226-249](src/nhf_spatial_targets/aggregate/ssebop.py#L226-L249);
     reject zero-byte stubs).
   - Slice the zarr's `swe` to that calendar year:
     `ds_year = swe.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))`.
   - Per-batch loop (mirrors ssebop):
     - `compute_or_load_weights(...)` keyed by `source_key=f"daymet_{region}"`
       so caches are segregated per region (matters for HI/PR later).
     - `aggregate_variables_for_batch(..., variables=["swe"],
       x_coord="x", y_coord="y", time_coord="time",
       source_crs=lcc_wkt, stat_method="mean")`.
   - `xr.concat(batches, dim=id_col)` → `year_ds`.
   - `year_ds.sortby(id_col)` to enforce canonical row order (issue #93).
   - `_attach_cf_global_attrs(year_ds, "daymet", meta)`. Add a
     `daymet_region` global attr ("na"/"hi"/"pr") in the same pass.
   - `_atomic_write_netcdf(year_ds, out_path)`.
9. **`_verify_year_coverage`** on the per-region file set. The helper
   matches `<source_key>_*_agg.nc`; pass `source_key=f"daymet_{region}"`
   so it sees only the region's files.
10. **`update_manifest`** keyed at `sources.daymet`, listing the
    per-(region,year) NCs and the weight files. Use the existing
    flat structure for this PR; refactor to nested `regions.<r>.{...}`
    when HI/PR land.

### CLI: `agg daymet`

Mirror `agg_ssebop_cmd` exactly (custom wrapper, not `_run_tier_agg`,
period required). Add a `--region` Annotated parameter defaulting to
`"na"`. Read `fabric.path` and `fabric.id_col` from `config.yml` and
forward to `aggregate_daymet`.

Update [src/nhf_spatial_targets/cli.py](src/nhf_spatial_targets/cli.py):
- Import `aggregate_daymet`.
- Add `@agg_app.command(name="daymet") def agg_daymet_cmd(...)`.
- Update `agg_all_cmd`'s docstring to mention daymet excluded
  alongside ssebop. Do **not** add daymet to the `sources` list inside
  `agg_all_cmd` — it would block on missing `--period`.

### Tests: `tests/test_aggregate_daymet.py` (new)

Mirror the structure of
[tests/test_aggregate_ssebop.py](tests/test_aggregate_ssebop.py) but
tighter — daymet's surface area is smaller (single variable, single
region in this PR). Cover:

1. Function signature: `period` and `region` keyword params present;
   `region` defaults to `"na"`.
2. Region guard: `region="hi"` raises `NotImplementedError` with a
   message referencing issue #101.
3. Missing-zarr-in-manifest: clear `FileNotFoundError` pointing the
   user at `nhf-targets fetch daymet`.
4. End-to-end smoke with mocks for `xr.open_zarr`, `WeightGen`, and
   `AggGen` (mirroring ssebop's mock pattern at
   [tests/test_aggregate_ssebop.py:64-67](tests/test_aggregate_ssebop.py#L64-L67)):
   - 2-year synthetic LCC zarr (4×4 grid, single `swe` var,
     `lambert_conformal_conic` grid mapping)
   - 3-HRU fabric
   - Run `aggregate_daymet(..., period="2010/2011", region="na")`
   - Assert per-year NCs exist at `daymet_na_<year>_agg.nc`
   - Assert `id_col` is monotonic ascending
   - Assert CF-1.6 globals + `daymet_region: "na"` attr
   - Assert weight cache file is `weights/daymet_na_batch0.csv` (not
     `daymet_batch0.csv`), proving region segregation
   - Assert `manifest.json` has `sources.daymet` with the per-year
     output paths
5. CLI registration: `cli.aggregate_daymet is aggregate_daymet`.
6. CLI `--region` plumbing: invoking the cyclopts app with
   `agg daymet --region hi ...` surfaces the `NotImplementedError`
   through `_run_tier_agg`-equivalent error handling (or
   `SystemExit(1)` via the custom wrapper).

### Inspection notebook: `notebooks/aggregated/inspect_aggregated_daymet.ipynb` (new)

Mirror
[notebooks/aggregated/inspect_aggregated_snodas.ipynb](notebooks/aggregated/inspect_aggregated_snodas.ipynb)
since both are daily SWE in `kg m-2`. Differences to surface:
- File discovery: glob `daymet_na_*_agg.nc` rather than
  `daymet_*_agg.nc` (the helper `discover_aggregated` in
  [notebooks/aggregated/_helpers.py:88-99](notebooks/aggregated/_helpers.py#L88-L99)
  globs `<source_key>_*_agg.nc`; we'll need either to widen its glob
  or to pass `source_key="daymet"` and post-filter. Decision deferred
  to implementation — likely add a `region` parameter to the helper).
- TARGET_YEAR around 2010 (good Daymet coverage).
- Same `kg m-2 → inches` PRMS-units rescale (× 1/25.4).
- Magnitude validation: Daymet CONUS-mean Feb SWE lands in similar
  order of magnitude to SNODAS (tens of mm); inter-source differences
  in mountain regions are expected and informative.

### Slurm: `agg_daymet.slurm` (new, standalone)

Pattern after `agg_ssebop.slurm` if it exists; otherwise build from
`agg_all.slurm`'s shape. Single-task script (no array), required
`--period` env var (e.g. `PERIOD=1980/2024`), `--mem=128G` default
(daymet is daily 1km LCC × 45 years × 1 var — comparable to MOD16A2
in memory weight). Document `--region` as an env var override
defaulting to `na`.

```bash
PROJECT_DIR=/path/to/project PERIOD=1980/2024 sbatch agg_daymet.slurm
```

### Pixi task: `pixi.toml`

Add `agg-daymet = { cmd = "nhf-targets agg daymet", description = "..." }`
between `agg-mod10c1` and `agg-snodas` (alphabetical within the
SWE-source cluster).

### CLAUDE.md

Add `pixi run nhf-targets agg daymet --project-dir ... --period YYYY/YYYY`
to the command list.

## Files to add / modify

**New:**

- [src/nhf_spatial_targets/aggregate/daymet.py](src/nhf_spatial_targets/aggregate/daymet.py)
- [tests/test_aggregate_daymet.py](tests/test_aggregate_daymet.py)
- [notebooks/aggregated/inspect_aggregated_daymet.ipynb](notebooks/aggregated/inspect_aggregated_daymet.ipynb)
- [agg_daymet.slurm](agg_daymet.slurm)
- [docs/superpowers/plans/2026-05-13-daymet-aggregate.md](docs/superpowers/plans/2026-05-13-daymet-aggregate.md)
  (copy of this plan)

**Modified:**

- [src/nhf_spatial_targets/cli.py](src/nhf_spatial_targets/cli.py)
  — import + `agg daymet` command + update `agg_all_cmd` docstring.
- [pixi.toml](pixi.toml) — new `agg-daymet` task.
- [CLAUDE.md](CLAUDE.md) — command-list entry.
- [notebooks/aggregated/_helpers.py](notebooks/aggregated/_helpers.py)
  — optional: extend `discover_aggregated` / `open_year` /
  `open_year_range` to accept an optional `region` parameter so the
  inspection notebook can target `daymet_na_*` cleanly. Smallest
  change: add `region: str | None = None` defaulting to `None` and
  splice into the glob. Keep backward compatibility for existing
  notebooks (none pass region).

**Not changed:**

- `catalog/sources.yml` — the daymet entry is complete.
- `aggregate/_driver.py` / `_adapter.py` — no driver changes needed;
  daymet uses its existing helpers.
- `defaults.py` — SWE target block lands with `targets/swe.py`
  (separate PR).

## Reused helpers

From [src/nhf_spatial_targets/aggregate/_driver.py](src/nhf_spatial_targets/aggregate/_driver.py):

- `WEIGHT_GEN_CRS` (line 94) — EPSG:5070 target for WeightGen.
- `load_and_batch_fabric` (line 97) — fabric loader with KD-tree
  spatial batching.
- `compute_or_load_weights` (line 474) — per-batch weight cache with
  fingerprinting.
- `aggregate_variables_for_batch` (line 592) — gdptools AggGen
  wrapper.
- `_atomic_write_netcdf` (line 201) — tempfile + rename.
- `_attach_cf_global_attrs` (line 672) — CF-1.6 globals.
- `_migrate_legacy_layout` (line 214) — clean up pre-per-year files.
- `_verify_year_coverage` (line 271) — contiguous-year check.
- `update_manifest` (line 29) — read-merge-write manifest entry.

From [src/nhf_spatial_targets/fetch/_period.py](src/nhf_spatial_targets/fetch/_period.py):

- `parse_period` — `'YYYY/YYYY'` format validation.

From `pyproj`:

- `CRS.from_cf(attrs_dict)` → WKT — decodes the
  `lambert_conformal_conic` grid-mapping variable into a CRS without
  hardcoding Daymet's proj string.

## Verification

End-to-end on a project that has run `nhf-targets fetch daymet`:

```bash
pixi run nhf-targets agg daymet \
    --project-dir <project-dir> \
    --period 2010/2011
```

Expected: two NCs at
`<project>/data/aggregated/daymet/daymet_na_2010_agg.nc` and
`daymet_na_2011_agg.nc`; a re-run is a no-op; `manifest.json` has
`sources.daymet` with those output paths and the period.

Then in the inspect notebook: open the agg NC, confirm HRU dim
`id_col` ascending, plot a winter day's HRU choropleth in both
`kg m-2` and inches, validate the CONUS-area-weighted Feb mean
against the SNODAS Feb mean from PR #117 (they should be in similar
orders of magnitude; mountain-HRU differences are expected and
informative for the eventual SWE target's multi-source min/max).

Automated:

```bash
pixi run -e dev fmt && pixi run -e dev lint   # locally
git push                                       # let CI run pytest
```

Per `feedback_skip_local_pytest_on_hpc` memory, do not run
`pixi run -e dev test` locally on this HPC; CI handles it.

## Git workflow

- Branch from `main`: `feature/101-daymet-aggregate`.
- PR title: `feat(daymet): aggregate daily SWE to HRU polygons (NA, #101)`.
- Open with "Refs #101 — Daymet NA aggregate" in the body.
- Copy this plan into the repo at
  `docs/superpowers/plans/2026-05-13-daymet-aggregate.md` and include
  it in the PR (precedent: PR #117).

## What's NOT in this PR (deferred)

- Daymet HI / PR regions — separate PR (or PRs) when those fabrics
  need them. The `--region` flag and zarr-resolution code are
  region-parameterised; HI/PR will only need the
  `NotImplementedError` guard relaxed and (probably) a
  manifest-schema refactor to nest by region.
- ERA5-Land `sd` extension, Margulis WUS-SR aggregator,
  `targets/swe.py`, `defaults.py` SWE block — separate PRs under #101.
