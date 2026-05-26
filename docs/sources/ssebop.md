# SSEBop AET (USGS NHGF STAC, aggregator-only)

USGS [Operational Simplified Surface Energy Balance (SSEBop)](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-data-citation)
monthly actual evapotranspiration (Senay et al., 2013;
doi:10.5066/P9L2YMV). Used as one of three sources for the AET (`aet`)
calibration target range.

**This source has no fetch step.** SSEBop is read on the fly from the
USGS NHGF STAC Zarr store via [gdptools](https://github.com/usgs-makerspace/gdptools)
during aggregation. There is no consolidated NC in the datastore;
`<datastore>/ssebop/` does not exist as a directory in this pipeline.
The aggregator opens the remote Zarr per batch per year and writes
per-year aggregated NCs directly.

## Access path — USGS NHGF STAC catalog

| Field | Value |
| ----- | ----- |
| Collection ID | `ssebopeta_monthly` |
| STAC endpoint | <https://usgs.osn.mghpcc.org/> |
| Zarr store | `s3://mdmf/gdp/ssebopeta_monthly.zarr/` |
| Period | 2000-2023 (monthly) |
| Spatial extent | CONUS, ~1 km |
| Source variable | `et` (catalog `cf_units: mm` with `cell_methods: time: sum`) |

No authentication. The STAC collection is resolved at aggregator
start via `gdptools.helpers.get_stac_collection("ssebopeta_monthly")`,
which fails fast with an explanatory note if the NHGF STAC endpoint
is unreachable.

The source-level `units: mm/month` (preserved in the catalog for
backwards readability) is recovered by combining `cf_units: mm` with
the time-axis aggregation declared in `cell_methods: time: sum`. The
`aet` target builder applies the linear conversion to inches/day
downstream (CLAUDE.md: linear conversions live in `targets/`).

## On-disk layout — no fetch artefact

```
<project>/data/aggregated/ssebop/
  ssebop_<year>_agg.nc          # one per year in --period
<project>/weights/
  ssebop_batch<i>.csv           # per-batch weight cache
  ssebop_batch<i>.csv.meta      # fingerprint sidecar
  ssebop_batch<i>.csv.fp        # fingerprint canonical-form
```

No `<datastore>/ssebop/` directory; the source is remote.

## Procedure — aggregation only, `--period` required

```bash
pixi run nhf-targets agg ssebop \
    --project-dir <project> \
    --period 2000/2023
```

`--period` is **required** (unlike file-based sources, where the
aggregator infers years from the consolidated NC). Months outside the
publisher window (2000-2023) raise on STAC read.

Or via the dedicated SLURM script
([`slurm/shared/agg_ssebop.slurm`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/slurm/shared/agg_ssebop.slurm),
not slotted into the `agg_all_<fabric>.slurm` array because the array
doesn't forward periods):

```bash
export PROJECT_DIR=/path/to/project
sbatch slurm/shared/agg_ssebop.slurm
```

## Aggregator wiring — custom STAC path

`aggregate/ssebop.py` is **not** a thin `SourceAdapter`. It re-uses
the driver's helpers (`_atomic_write_netcdf`, `_batch_fingerprint`,
`compute_or_load_weights`, `load_and_batch_fabric`, `update_manifest`,
`_verify_year_coverage`) but reads the remote Zarr directly via
`NHGFStacZarrData → WeightGen / AggGen`. Highlights:

- **Weight cache uses STAC collection ID as the source-identity arm.**
  The fingerprint is keyed on `f"stac:{collection_id}"` rather than a
  source CRS, so a future SSEBop migration to a different
  collection invalidates the cache.
- **Per-year idempotent.** Existing per-year NCs are skipped; zero-byte
  stubs (left by a non-atomic write or SIGKILL) are unlinked and
  re-aggregated.
- **SLURM-array sharding.** `--worker-index` / `--n-workers` select a
  round-robin slice of years (issue #156). The contiguous-year check
  in `_verify_year_coverage` is skipped when `n_workers > 1` so
  sibling workers don't false-positive each other's gaps; the manifest
  update merges file lists across workers via flock-protected
  `update_manifest`.
- **`stat_method="mean"`.** No per-pixel masking happens upstream; the
  remote Zarr already has flagged pixels as NaN, so the default
  NaN-propagating mean is correct (HRUs at the CONUS edge come out
  honestly NaN).

## HPC memory/time notes

- Weight gen on a typical CONUS fabric runs ~15-30 s per batch on the
  first year, with the per-batch cache reused across all subsequent
  years. The cache is keyed on (HRU fingerprint, STAC collection); a
  fabric swap or batch-size change invalidates it.
- Aggregation itself is fast (per-year STAC read is on the order of a
  minute for CONUS at 1 km monthly). Wall time is dominated by weight
  generation on first year.
- Single-job 128 GB / 24 h SLURM allocation is comfortable.

## Troubleshooting

- `Failed to resolve STAC collection 'ssebopeta_monthly'` — the
  NHGF STAC endpoint is unreachable. Verify
  <https://usgs.osn.mghpcc.org/> resolves and reachability via
  `curl -I https://usgs.osn.mghpcc.org/` then retry.
- `ssebop: --period must be 'YYYY/YYYY'` — pass `--period` explicitly;
  unlike file-based sources, ssebop cannot infer years from disk.
- Weight cache "stale batch fingerprint" warning followed by a
  recompute — fabric or batch size changed since the cache was
  written. This is normal; the recomputed cache is then re-used.
- Catalog reference: see [`catalog/sources.yml`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/catalog/sources.yml)
  `ssebop:` block for the STAC collection ID and DOI.
