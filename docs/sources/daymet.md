# Daymet V4 R1 (operator-staged zarr)

The ORNL DAAC Daymet V4 R1 daily 1 km gridded surface weather product
(Thornton et al., 2022; doi:10.3334/ORNLDAAC/2129) is distributed as
three regional zarr stores covering North America (`na`), Hawaii
(`hi`), and Puerto Rico (`pr`). Each regional zarr exposes
`prcp, srad, swe, tmax, tmin, vp` plus the `lambert_conformal_conic`
CRS spec. Only `swe` is consumed by the SWE calibration target;
the other variables are documented for completeness.

The publisher distributes the zarrs in hundreds-of-GB per region.
We do **not** download them — the operator pre-stages the zarrs on a
shared filesystem and `nhf-targets fetch daymet` fingerprints each
region (sha256 over `.zgroup` / `.zarray` / `.zattrs` / `.zmetadata`
files plus the directory's total byte count) and writes a per-region
manifest entry. The fingerprint design and its residual gaps are
documented in [`src/nhf_spatial_targets/fetch/daymet.py`](../../src/nhf_spatial_targets/fetch/daymet.py).

## Procedure

1. Obtain the three regional zarrs from the publisher
   <https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=2129> and stage
   them on a shared filesystem location, e.g.:

   ```
   /<shared>/daymet/source/zarr/
     daymet_na.zarr/
     daymet_hi.zarr/
     daymet_pr.zarr/
   ```

   Operators on the Hovenweep HPC cluster can point at the existing
   `/caldera/hovenweep/projects/usgs/water/impd/nhgf/data_creation/daymet/source/zarr`
   tree rather than re-staging.

2. Configure the project to find the zarrs. Either:

   - Add `daymet_root: /<shared>/daymet/source/zarr` to `config.yml`, **or**
   - Pass `--source-path /<shared>/daymet/source/zarr` on the CLI.

   The `--source-path` flag wins if both are present.

3. Run the fetch:

   ```bash
   nhf-targets fetch daymet --project-dir <project> --period 1980/2024
   ```

   Optional flags:

   - `--region {na|hi|pr|all}` (default `all`). Use `--region na` for
     CONUS-only fabrics.

   The command verifies each present zarr opens cleanly, asserts the
   required variables are exposed, fingerprints the structural
   metadata, and records per-region entries under
   `manifest.json` → `sources.daymet.regions.{na,hi,pr}`. Subsequent
   runs against unchanged zarrs are a no-op (matching fingerprint =
   skip re-validation).

4. Parallel runs across regions are safe: the manifest update takes an
   exclusive `flock` on a `manifest.json.lock` sibling, so
   `fetch daymet --region na` and `fetch daymet --region hi` can run
   concurrently without losing each other's records.

## Troubleshooting

- `FileNotFoundError: Daymet zarr root is not configured` — either
  `--source-path` was not passed or `daymet_root:` is missing from
  `config.yml`. Fix one and re-run.

- `RuntimeError: <zroot> is missing required variables [...]` — the
  staged zarr is incomplete. Re-stage from the publisher.

- `RuntimeError: <zroot> is being modified concurrently` — another
  process is writing to the zarr while fetch tries to fingerprint it.
  Wait for the writer to finish, then re-run.

- Fingerprint residual risk: a chunk file overwritten in place with
  identical compressed length but different bytes is invisible to the
  structural-metadata sha256 + directory byte count. Re-stage from
  the publisher if chunk-level tampering is suspected.

## Daymet aggregator (PR #118)

`aggregate/daymet.py` is a **custom-loop aggregator** (not a thin
`SourceAdapter`) — Daymet's source is a single multi-year zarr per
region on a Lambert Conformal Conic projected grid, not per-year
NetCDFs in WGS84, so the standard `aggregate_source` driver doesn't
apply. The custom loop reuses the driver's helpers
(`compute_or_load_weights`, `aggregate_variables_for_batch`,
`_atomic_write_netcdf`, `_attach_cf_global_attrs`,
`_migrate_legacy_layout`, `_verify_year_coverage`,
`load_and_batch_fabric`, `update_manifest`) but opens the zarr
directly and decodes the source CRS from the
`lambert_conformal_conic` grid-mapping variable via
`pyproj.CRS.from_cf`.

Run it via:

```bash
pixi run nhf-targets agg daymet \
    --project-dir <project> \
    --period 1980/2024 \
    --region na           # default; hi/pr raise NotImplementedError
```

`--period` is required (like ssebop); the requested window is hard-
failed against the open zarr's actual time-coord range so a typo
like `1900/1979` surfaces in milliseconds rather than as an opaque
gdptools error mid-loop.

Output: `<project>/data/aggregated/daymet/daymet_<region>_<year>_agg.nc`,
one per (region, year). Region is **encoded in the filename** so HI
and PR can land alongside NA without renaming existing NA outputs.
HRU dim `id_col` ascending (issue #93), CF-1.6 globals plus a
`daymet_region: "na"` attr.

### Region status

| Region | Status | Notes |
|---|---|---|
| `na` | implemented | CONUS + Canada + Mexico fabric. |
| `hi` | **not wired** | `NotImplementedError`. Add to `_SUPPORTED_REGIONS` when a Hawaii fabric needs it. |
| `pr` | **not wired** | Same — when a Puerto Rico fabric needs it. |

### Weight cache + manifest layout

Weights are cached per `(region, batch_id)` at
`<project>/weights/daymet_<region>_batch<i>.csv`. The synthetic key
`daymet_<region>` is also used by `_verify_year_coverage` to glob
only the region's files. Manifest indexing stays
`sources.daymet` (catalog source_key) — the aggregator merges its
fields into the existing entry alongside the fetch-side
`regions: {na: {...}}` dict via `_driver.update_manifest`'s
entry-level read-merge-write (PR #120 closes #119).

### Performance

Daymet's native LCC grid is well-aligned with the EPSG:5070
weight-gen CRS used by gdptools. Weight gen runs about ~10-15 s per
5648-HRU batch at the CONUS fabric, vs ~60 s for the same batch
against SNODAS's WGS84 grid — see
[CRS-cost note in the transformation-pipeline doc](../architecture/transformation-pipeline.md#operational-cost-source-crs-vs-aggregator-cost).
Daily aggregation runs at ~3-4 s per batch once weights are cached;
the cache is reused across all years.

### SLURM wiring (`agg_daymet.slurm`, standalone)

Daymet has its own slurm script rather than slotting into
`agg_all.slurm` because `agg daymet` requires `--period` and `agg
all` doesn't forward periods. Run via:

```bash
PROJECT_DIR=/path/to/project PERIOD=1980/2024 sbatch agg_daymet.slurm
```

`REGION` defaults to `na`; override with `REGION=na sbatch agg_daymet.slurm`
once HI/PR are wired up.
