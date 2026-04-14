# Remaining Aggregation Methods — Design

**Date:** 2026-04-14
**Status:** approved, ready for implementation plan
**Scope:** Build tier-1 (shared-driver) and tier-2 (bespoke) spatial aggregation
modules for every catalog source that does not yet have one, excluding SSEBop
(already done) and Reitz2017 (deferred pending possible ERA5-Land replacement).

## Goals

1. Aggregate every remaining gridded source to the project's HRU fabric using
   `gdptools`, writing one NetCDF per source into `data/aggregated/`.
2. Eliminate the ~200 LOC of boilerplate that a copy-of-`ssebop.py` per source
   would produce, without forcing genuinely-different sources into an awkward
   shared shape.
3. Preserve the provenance record in `manifest.json` with a single source of
   truth for the manifest schema.
4. Leave SSEBop (remote STAC/Zarr) unchanged beyond an optional manifest-helper
   refactor.

## Non-goals

- Cross-source weight sharing (weights keyed per source, per `ssebop.py`).
- Re-implementing reprojection. gdptools handles source→equal-area internally
  as long as source CRS is declared.
- Changes to any fetch module, target builder, or catalog YAML.
- Reitz2017 aggregation (deferred).

## Architecture: two-tier

### Tier 1 — shared driver + thin adapters

Seven sources whose consolidated NetCDFs in the datastore are well-behaved
lat/lon (or lat/lon-ish) monthly grids:

| Source | Variables to aggregate |
|---|---|
| `era5_land` | `ro`, `sro`, `ssro` |
| `gldas_noah_v21_monthly` | `Qs_acc`, `Qsb_acc`, `runoff_total` (derived) |
| `merra2` | `GWETTOP`, `GWETROOT`, `GWETPROF` |
| `ncep_ncar` | `soilw_0_10cm`, `soilw_10_200cm` |
| `nldas_mosaic` | `SoilM_0_10cm`, `SoilM_10_40cm`, `SoilM_40_200cm` |
| `nldas_noah` | `SoilM_0_10cm`, `SoilM_10_40cm`, `SoilM_40_100cm`, `SoilM_100_200cm` |
| `watergap22d` | `qrdif` |

### Tier 2 — bespoke modules using shared helpers

Two sources whose shape does not fit tier-1 declaratively, but which reuse the
driver's batching, weight-cache, and manifest helpers:

| Source | Reason for bespoke handling |
|---|---|
| `mod16a2_v061` | Native sinusoidal MODIS projection; adapter declares a non-4326 `source_crs` and opens the sinusoidal tile stack with CRS attribute set. gdptools reprojects into EPSG:5070 for weighting. |
| `mod10c1_v061` | CI-based masking at source grid cells (`ci > 0.70`); writes three derived HRU variables (`sca`, `ci`, `valid_area_fraction`). |

SSEBop stays as-is. Reitz2017 is out of scope.

## Module layout

```
src/nhf_spatial_targets/aggregate/
  __init__.py
  _driver.py          # NEW: shared engine + manifest helper
  _adapter.py         # NEW: SourceAdapter dataclass
  batching.py         # unchanged
  ssebop.py           # unchanged (may call extracted manifest helper)
  era5_land.py        # NEW (tier-1 adapter + aggregate_era5_land)
  gldas.py            # NEW (tier-1, derived runoff_total)
  merra2.py           # NEW (tier-1)
  ncep_ncar.py        # NEW (tier-1)
  nldas_mosaic.py     # NEW (tier-1)
  nldas_noah.py       # NEW (tier-1)
  watergap22d.py      # NEW (tier-1)
  mod16a2.py          # NEW (tier-2, sinusoidal CRS)
  mod10c1.py          # NEW (tier-2, CI masking + valid_area)
```

Each tier-1 and tier-2 module exposes
`aggregate_<source>(fabric_path, id_col, workdir, batch_size)` (no period
argument). `aggregate_ssebop` keeps its existing `period` parameter since
it drives a remote Zarr query.

## Tier-1 contract

### `SourceAdapter` dataclass

```python
@dataclass(frozen=True)
class SourceAdapter:
    source_key: str                    # catalog key
    output_name: str                   # e.g. "era5_land_agg.nc"
    variables: list[str]               # vars to aggregate after open_hook
    x_coord: str = "lon"
    y_coord: str = "lat"
    time_coord: str = "time"
    source_crs: str = "EPSG:4326"
    open_hook: Callable[[Project], xr.Dataset] | None = None
```

The `open_hook` receives the resolved `Project` and returns an `xr.Dataset`
with CRS set and any derived variables materialised. Default behaviour (when
hook is None): open the single consolidated NC under
`project.raw_dir(source_key)`.

### Driver entry point

```python
def aggregate_source(
    adapter: SourceAdapter,
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
) -> xr.Dataset
```

Responsibilities (generalised from `ssebop.py`):

1. Load project, open source via `adapter.open_hook`.
2. Validate every `adapter.variables` name present. **Do not clip time** —
   aggregate the source's full available period. Period-of-interest clipping
   is the responsibility of downstream target builders.
3. Load fabric (GeoPackage/Parquet), run `spatial_batch(gdf, batch_size)`.
4. Per batch: load or compute weights with `WeightGen` (weight_gen_crs=5070),
   cache at `weights/<source_key>_batch<id>.csv`; run `AggGen` with
   `stat_method="masked_mean"` across `adapter.variables`.
5. Concat batch results on `id_col`, write `data/aggregated/<output_name>`
   with all aggregated variables.
6. Update `manifest.json` atomically (helper in `_driver.py`) under
   `sources[source_key]` with `access_type`, `period`, `fabric_sha256`,
   `output_file`, `weight_files`, and UTC `timestamp`.

### Example adapter (GLDAS, illustrative)

```python
def _open(project: Project) -> xr.Dataset:
    nc = project.raw_dir("gldas_noah_v21_monthly") / "gldas_noah_v21_monthly.nc"
    ds = xr.open_dataset(nc)
    ds["runoff_total"] = ds["Qs_acc"] + ds["Qsb_acc"]
    ds["runoff_total"].attrs = {
        "long_name": "total runoff (Qs_acc + Qsb_acc)",
        "units": "kg m-2",
    }
    return ds

ADAPTER = SourceAdapter(
    source_key="gldas_noah_v21_monthly",
    output_name="gldas_agg.nc",
    variables=["Qs_acc", "Qsb_acc", "runoff_total"],
    open_hook=_open,
)

def aggregate_gldas(fabric_path, id_col, workdir, batch_size=500):
    return aggregate_source(
        ADAPTER, fabric_path, id_col, workdir, batch_size,
    )
```

## Tier-2 contract

Tier-2 modules do not use `SourceAdapter` declaratively; they implement their
own `aggregate_<source>()` but import three helpers from `_driver.py`:

- `load_and_batch_fabric(fabric_path, batch_size)` → batched GeoDataFrame
- `compute_or_load_weights(batch_gdf, source_data, source_key, batch_id, workdir)`
- `update_manifest(project, source_key, access, output_file, weight_files)`
  (period derived from the aggregated dataset's time coord, recorded for
  provenance)

### MOD16A2 (`mod16a2.py`)

- `open_hook` returns the sinusoidal tile stack as one xarray Dataset with the
  sinusoidal CRS attribute set on the dataset.
- `source_crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m"`.
- Variable: `ET_500m`. 8-day cadence passes straight through — monthly
  resampling lives in `targets/aet.py`, not here.
- Output: `data/aggregated/mod16a2_agg.nc`.

### MOD10C1 (`mod10c1.py`)

Masking happens at source grid cells, before aggregation:

1. Open consolidated NC with `Day_CMG_Snow_Cover` and `Snow_Spatial_QA`.
2. Build three source DataArrays:
   - `sca = Day_CMG_Snow_Cover / 100.0` where `Snow_Spatial_QA/100 > 0.70`,
     else NaN.
   - `ci  = Snow_Spatial_QA / 100.0` (passed through; no mask).
   - `valid_mask = 1.0` where `Snow_Spatial_QA/100 > 0.70`, else `0.0`
     (float, pre-NaN propagation).
3. Aggregate all three to HRUs. `valid_mask` aggregated as
   `masked_mean` becomes the fraction of each HRU's area that passed the CI
   filter on that day — stored as `valid_area_fraction`.
4. Output variables: `sca`, `ci`, `valid_area_fraction`. Target builder
   (`targets/sca.py`) can drop HRU-days with low `valid_area_fraction`.
5. Output: `data/aggregated/mod10c1_agg.nc`.

## CLI

Extend `agg_app` in `cli.py` with one subcommand per source, each structurally
a copy of `agg_ssebop_cmd`:

```
nhf-targets agg era5-land     --project-dir <dir>
nhf-targets agg gldas         --project-dir <dir>
nhf-targets agg merra2        --project-dir <dir>
nhf-targets agg ncep-ncar     --project-dir <dir>
nhf-targets agg nldas-mosaic  --project-dir <dir>
nhf-targets agg nldas-noah    --project-dir <dir>
nhf-targets agg watergap22d   --project-dir <dir>
nhf-targets agg mod16a2       --project-dir <dir>
nhf-targets agg mod10c1       --project-dir <dir>
nhf-targets agg all           --project-dir <dir>
```

No `--period` flag. Each aggregator processes the full temporal range
present in the consolidated source NetCDF. Target builders apply the
period-of-interest window downstream.

Note: `agg ssebop` retains `--period` because SSEBop is a remote Zarr
store (not a local consolidated NC) and the period controls what is
pulled from the store. Its signature is unchanged by this work.

`agg all` iterates registered sources and runs them sequentially,
stopping on the first failure. SSEBop included in `agg all` (with its
default/config-driven period).

## Testing

- `tests/test_aggregate_driver.py` — unit tests with a synthetic 3×3 source
  NetCDF and a 2-polygon fabric. Verify: weights cached on first run and
  reloaded on second, multi-var output structure, manifest updated atomically,
  period clipping.
- `tests/test_aggregate_<source>.py` per new source — minimal unit test that
  the adapter's `_open` hook handles the expected consolidated-NC shape (small
  fixture or mocked xarray). For GLDAS, assert `runoff_total = Qs_acc +
  Qsb_acc`.
- `tests/test_aggregate_mod10c1.py` — dedicated tier-2 test for CI masking:
  a synthetic 2×2×2 (time × y × x) grid with known CI values verifies that
  cells with `ci <= 0.70` become NaN in `sca`, that `valid_area_fraction` at
  the HRU level equals the correct area-weighted mean of the 0/1 mask, and
  that `ci` passes through untouched.
- `tests/test_aggregate_mod16a2.py` — verifies the sinusoidal CRS propagates
  into gdptools (mock the `WeightGen` call and assert the source dataset's
  CRS attribute).
- Integration tests marked `pytest.mark.integration`, skipped by default,
  run via `pixi run -e dev test-integration` against a real datastore.

## Manifest helper extraction (minor refactor)

Lift `_update_manifest` out of `src/nhf_spatial_targets/aggregate/ssebop.py`
into `_driver.py` as `update_manifest(project, source_key, access,
output_file, weight_files, period)`. The `period` field is still written to
the manifest for provenance but is now supplied by the caller — derived from
the aggregated dataset's time coord for tier-1/tier-2, and from the CLI arg
for SSEBop. Update `ssebop.py` to call the shared helper. Manifest schema
stays exactly as it is today.

## Open questions / risks

- **NetCDF variable coord names after consolidation.** Sources are
  consolidated into CF-1.6 NCs by existing fetch modules; most should use
  `lat`/`lon`, but `ncep_ncar` (Gaussian grid) and MERRA-2 coord names must
  be verified against the actual written NCs before finalising the adapter's
  `x_coord`/`y_coord`. Mitigation: tier-1 adapters inspect the consolidated
  NC on first implementation; values get pinned in the adapter.
- **Period handling.** Aggregators do not accept `--period`; each processes
  the full temporal range present in the consolidated source NetCDF.
  Period-of-interest clipping (e.g. the 2000-2009 recharge normalization
  window, the AET 2000-2010 window) happens inside the target builders
  when they read the aggregated NCs. This keeps aggregated outputs
  reusable across targets with different period requirements.
- **Weight-cache invalidation.** If the fabric changes (different
  `fabric.sha256`), cached weights become stale. Current `ssebop.py` does not
  invalidate on fabric change. Out of scope here — carried over as-is; could
  be revisited later by including `fabric_sha256[:8]` in the weight filename.

## Implementation sequencing hint

1. Extract `update_manifest` into `_driver.py`; update `ssebop.py`. Tests
   still pass.
2. Add `SourceAdapter` + `aggregate_source` in `_driver.py`/`_adapter.py`
   with driver-level tests against a synthetic adapter.
3. Implement tier-1 adapters in one PR per source (or grouped by variable
   family: runoff, soil moisture, recharge). Each comes with its own test
   file and CLI subcommand.
4. Implement tier-2 modules (`mod16a2.py`, `mod10c1.py`) last.
5. Finally, add `agg all` to iterate registered aggregators.
