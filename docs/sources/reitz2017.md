# Reitz 2017 Empirical Recharge (USGS ScienceBase)

[Reitz et al. (2017)](https://www.sciencebase.gov/catalog/item/56c49126e4b0946c65219231)
annual empirical regression-based estimates of groundwater recharge for
CONUS (doi:10.5066/F7PN93P0; the publication also covers runoff and
ET, but only **total recharge** and **effective recharge** are used
here). Used as one of two sources for the recharge (`rch`) calibration
target range alongside [WaterGAP 2.2d](watergap22d.md).

The pipeline downloads two variables, one annual GeoTIFF per
year-variable, for 2000-2013:

| Variable | File variable | Long name | Units (after correction) |
| -------- | ------------- | --------- | ----------------------- |
| `total_recharge` | `TotalRecharge` | total groundwater recharge | `m yr-1` |
| `eff_recharge` | `EffRecharge` | effective groundwater recharge (base flow) | `m yr-1` |

## The "no embedded units" gotcha

> The ScienceBase release of these GeoTIFFs has **no embedded units
> metadata** — no `Band.GetUnitType`, no file/band metadata, no PAM
> `.aux.xml` sidecar. The ScienceBase landing page documents the
> values as **m/year**, and CONUS-wide mean (~122 mm/yr) plus
> spot-checks at known-high-recharge locations are consistent with
> that interpretation.
>
> Earlier catalog versions declared inches/year, which produced values
> ~8× too low. Corrected to `m yr-1` (see PR #68 review). The
> catalog `units:` field is authoritative.

This is a worked example of "validate magnitudes against published
means before trusting catalog metadata" (see the
`feedback_validate_magnitudes` memo). A CONUS-mean >30% off published
value is a smoking gun for a missed conversion factor.

## Access path — sciencebasepy

No NASA/CDS credentials needed. The fetch module
[`fetch/reitz2017.py`](../../src/nhf_spatial_targets/fetch/reitz2017.py)
uses [`sciencebasepy`](https://github.com/DOI-USGS/sciencebasepy) to:

1. Connect to ScienceBase item
   [`55d383a9e4b0518e35468e58`](https://www.sciencebase.gov/catalog/item/55d383a9e4b0518e35468e58)
   (the child item of the catalog `item_id` parent that holds the
   actual files).
2. Build a filename → file-info lookup from `sb.get_item_file_info(item)`.
3. For each year in `period`, download two zipped GeoTIFFs:
   `TotalRecharge_<year>.zip` and `EffRecharge_<year>.zip`.
4. Extract the single `.tif` from each (renaming `RC_<year>.tif` /
   `RC_eff_<year>.tif` to match the
   `TotalRecharge_<year>.tif` / `EffRecharge_<year>.tif` convention).
5. Delete the zip after extraction; keep the GeoTIFF on disk.

After all year-variable GeoTIFFs are present, `_consolidate` stacks
them along a `time` dimension (mid-year July 1 timestamps) and writes
a single CF-1.6 consolidated NetCDF.

## On-disk layout

```
<datastore>/reitz2017/
  TotalRecharge_<year>.tif       # one per year in 2000-2013
  EffRecharge_<year>.tif
  reitz2017_consolidated.nc       # combined CF-1.6 NC (consumed by aggregator)
```

## Period gating

The fetch hard-rejects years outside 2000-2013 with a clear error
because the publisher's archive ends at 2013. `nhf-targets fetch reitz2017
--period 1979/2025` will fail loudly before any HTTP work.

## Procedure

```bash
nhf-targets fetch reitz2017 --project-dir <project> --period 2000/2013
```

Or via the general SLURM fetch array, index 7
([`slurm/shared/fetch_all.slurm`](../../slurm/shared/fetch_all.slurm)).

The fetcher is **incremental at the year-variable pair level**:
existing GeoTIFFs on disk are skipped. The consolidator regenerates the
combined NetCDF from whatever's on disk, so removing a year's tif and
re-running rebuilds without that year.

## CRS gotcha — Reitz GeoTIFFs declare NAD83 (EPSG:4269)

`aggregate/reitz2017.py` overrides the default `EPSG:4326` with
`source_crs="EPSG:4269"` (NAD83 geographic). The CRS is preserved
through `_consolidate → apply_cf_metadata` and read from the on-disk
GeoTIFF's WKT (`rio.crs.to_wkt()`) — this is the only catalog source
that uses NAD83 rather than WGS84 natively.

## HPC memory/time notes

- 14 years × 2 variables = 28 small GeoTIFFs at ~tens of MB each.
  Total raw ~2-4 GB (see [README.md](../../README.md) §Datastore
  Storage Estimates).
- The general 128 GB / 24 h fetch SLURM allocation is overkill;
  full-period runs finish in 5-10 minutes.

## Aggregator wiring

`aggregate/reitz2017.py` is a thin `SourceAdapter` (annual cadence,
`stat_method="mean"`). Per-year aggregated NCs land at
`<project>/data/aggregated/reitz2017/reitz2017_<year>_agg.nc`, in
native `m yr-1`. The `rch` target builder normalizes to [0, 1] and
combines with WaterGAP 2.2d.

## Troubleshooting

- `RuntimeError: File '<zip>' not found in ScienceBase item ...` —
  the child item's file listing has changed. Browse to
  <https://www.sciencebase.gov/catalog/item/55d383a9e4b0518e35468e58>
  and verify the expected `TotalRecharge_YYYY.zip` /
  `EffRecharge_YYYY.zip` files exist; update `file_patterns` in
  the catalog if renamed.
- `ValueError: Year YYYY is outside the Reitz 2017 data range
  (2000-2013)` — adjust `--period` to fit the publisher window.
- A CONUS-mean recharge plot that looks like 1/8 of expected
  magnitude — the `units` field in the catalog drifted from `m yr-1`
  back to `in/yr` or similar. Re-confirm against `catalog.source("reitz2017")`
  and re-fetch + re-aggregate.
- Catalog reference: see [`catalog/sources.yml`](../../catalog/sources.yml)
  `reitz2017:` block for the units-correction notes and ScienceBase
  links.
