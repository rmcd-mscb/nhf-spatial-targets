# UA Daily 4-km Gridded SWE and Snow Depth (NSIDC-0719)

The University of Arizona Daily 4-km Gridded SWE and Snow Depth is a
daily CONUS-wide SWE and snow depth reanalysis produced by assimilating
SNOTEL and COOP in-situ measurements with PRISM-modeled precipitation
fields (Broxton, Zeng & Dawson, University of Arizona). NSIDC
distributes it as collection
[NSIDC-0719 v1](https://nsidc.org/data/nsidc-0719/versions/1)
(doi:[10.5067/0GGPB220EX6A](https://doi.org/10.5067/0GGPB220EX6A)).
Water-year coverage is WY 1982–2023 (1981-10-01 through 2023-09-30).

This source serves as the **5th SWE source** in the pipeline (alongside
Daymet, SNODAS, ERA5-Land `sd`, and Margulis WUS-SR) and as the **2nd
SCA source** (alongside MOD10C1 v061) — adding decades of pre-MODIS SCA
coverage via a depth-derived binary snow-cover indicator. The SWE target
shim (PR-C) and the multi-source SCA wiring (PR-D) both land under
umbrella issue #237; the aggregate adapter lands in PR-B.

## Provider, license, and citation

**Provider:** NASA National Snow and Ice Data Center Distributed Active
Archive Center (NSIDC DAAC), Boulder, CO.

**License:** Public domain (NASA NSIDC product). No use restrictions;
NSIDC requests data citation per their standard terms.

**Recommended citation:**

> Broxton, P., Zeng, X., and Dawson, N. (2019). *Daily 4 km Gridded SWE
> and Snow Depth from Assimilated In-Situ and Modeled Data over the
> Conterminous US, Version 1.* Boulder, Colorado USA: NASA National Snow
> and Ice Data Center DAAC.
> doi:[10.5067/0GGPB220EX6A](https://doi.org/10.5067/0GGPB220EX6A)

The FGDC `useconst` metadata field for the release child carrying this
source should carry an attribution line: cite Broxton, Zeng & Dawson
(2019) and acknowledge the NASA NSIDC DAAC.

## Access pathway

> **Access type: `nsidc_https`** — per-WY filenames are deterministic at
> the NSIDC archive root. The fetch module constructs URLs directly
> rather than querying CMR. Same pattern as SNODAS (see issue #107 for
> the CMR-stub failure that motivated this approach).

The raw archive lives at:

```
https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0719_SWE_Snow_Depth_v1/
```

Files are named `4km_SWE_Depth_WY<YYYY>_v01.nc` for WY 1982 through
WY 2023 (42 files, ~90 MB each), plus one CONUS land/water/ice mask
`SWE_Mask_v01.nc` at the archive root.

**Authentication:** NASA Earthdata Login. Credentials are materialized
into `~/.netrc` by
`nhf-targets materialize-credentials --project-dir <project>`. The
fetch module opens an earthaccess HTTPS auth session
(`earthaccess.login(strategy='netrc').get_session()`) and streams each
file via `session.get(url)`.

## Prerequisites

- NASA Earthdata account with credentials materialized into `~/.netrc`
  (`nhf-targets materialize-credentials --project-dir <project>`).
- Approximately 4 GB of free space in `<datastore>/ua_swe/raw/` for
  the full WY 1982–2023 archive.

## Procedure

```bash
nhf-targets fetch ua-swe \
    --project-dir <project> \
    [--period 1982/2022]
```

The command:

1. Logs in to NASA EDL via `earthaccess.login(strategy='netrc')`.
2. Reads the publisher window from `sources.yml[ua_swe].period` and
   rejects out-of-range years.
3. Pre-filters water years against the existing manifest — WYs already
   present on disk with non-zero size are skipped on re-runs.
4. For each pending WY, constructs the URL
   `<archive_url>/4km_SWE_Depth_WY<YYYY>_v01.nc` and streams it into
   `<datastore>/ua_swe/raw/4km_SWE_Depth_WY<YYYY>_v01.nc`.
5. Consolidates the per-WY raws into per-calendar-year NCs (see
   "On-disk layout" below).

## On-disk layout

```
<datastore>/ua_swe/
  raw/
    4km_SWE_Depth_WY1982_v01.nc   # raw per-WY native files (NAD83 lat/lon)
    4km_SWE_Depth_WY1983_v01.nc
    ...
    4km_SWE_Depth_WY2023_v01.nc
    SWE_Mask_v01.nc               # CONUS land/water/ice mask
  daily/
    ua_swe_daily_1982.nc          # consolidated, per-calendar-year, EPSG:5070
    ua_swe_daily_1983.nc
    ...
    ua_swe_daily_2022.nc          # full coverage CY 1982-2022
```

**Consolidated per-calendar-year NCs** (`<datastore>/ua_swe/daily/`):

The consolidator re-windows the per-WY raws into one CF-1.6
calendar-year NetCDF per calendar year, mirroring the
`margulis_wus_sr` pattern. Each `ua_swe_daily_<YYYY>.nc` holds Jan 1
through Dec 31 of calendar year `YYYY`, assembled from:

- **Jan–Sep of WY `YYYY`** (from `4km_SWE_Depth_WY<YYYY>_v01.nc`)
- **Oct–Dec of WY `YYYY+1`** (from `4km_SWE_Depth_WY<YYYY+1>_v01.nc`)

Full calendar-year coverage is **CY 1982–2022** (41 files). The partial
edge years 1981 (Oct–Dec only) and 2023 (Jan–Sep only) are dropped —
they lack a complete calendar year and would produce a ragged time axis.

Each year file carries:

| Variable | Type | Units | cell_methods |
| --- | --- | --- | --- |
| `swe` | float32 | `kg m-2` | `time: point` |
| `snow_depth` | float32 | `mm` | `time: point` |
| `crs` | int (scalar) | — | EPSG:5070 WGS84/CONUS Albers grid mapping |

Global attributes: `Conventions=CF-1.6`, `title`, `institution`,
`source`, `references`, `frequency=day`.

**Pre-projection to EPSG:5070.** The native source grid is NAD83
EPSG:4269 lat/lon (~4 km). At consolidate time each day is reprojected
to EPSG:5070 NAD83/CONUS Albers using nearest-neighbour resampling
(destination grid locked from day 0 of each WY). This matches the
aggregator's `WEIGHT_GEN_CRS` and lets gdptools skip the reprojection
step during weight generation (~5–8× speedup vs. WGS84-native sources
at ~1 km; measured for SNODAS). Nearest-neighbour resampling preserves
NaN fill without bleed.

## Per-variable units

Catalog (`catalog/sources.yml`) is authoritative; never read units off
on-disk NetCDF attrs.

| Variable | Catalog `cf_units` | Notes |
| --- | --- | --- |
| `swe` | `kg m-2` | mm water-equivalent at density 1000 kg m-3. |
| `snow_depth` | `mm` | mm of snow thickness (not water equivalent). Used both as a standalone aggregated variable and as the basis for `snow_covered_fraction` in the PR-B aggregate hook. |

## Known quirks

- **Fill is NaN, no integer sentinel.** The source ships float32 with
  NaN fill — no -9999 or other integer sentinel. The consolidator
  enforces `min >= -1.0` as a guard against a future format change that
  might introduce a -9999 sentinel; values below that abort with a
  `ValueError`.

- **Partial edge calendar years dropped.** CY 1981 (Oct–Dec only, from
  WY 1982) and CY 2023 (Jan–Sep only, from WY 2023) lack a complete
  calendar year and are not written. Full coverage is CY 1982–2022.

- **Time is float32 with no `units` attribute in the raw files.** The
  consolidator decodes explicitly using `days since 1900-01-01`, the
  standard for this archive.

- **`short_name` is `NSIDC-0719` (recorded for provenance).** CMR may
  carry a collection record, but — per the SNODAS precedent (issue
  #107) — NSIDC HTTPS-only archives often have granule-less CMR stubs.
  The fetch path constructs URLs directly without calling
  `earthaccess.search_data`; `short_name` is not used by the fetch path.

- **UA SWE SCA wiring lands in PR-D (umbrella #237).** The
  `snow_covered_fraction` variable (a depth-derived 0/1 binary
  area-weighted to HRU scale) is emitted by `aggregate/ua_swe.py`'s
  `pre_aggregate_hook` (PR-B). The multi-source SCA combination in
  `targets/sca.py` (PR-D) then pairs it with MOD10C1. Both are deferred
  from PR-A2; the catalog entry's `snow_covered_fraction` scaffolding
  block documents the intended variable but is not enforced by anything
  in the consolidation layer.

## HPC notes

- **Memory:** each raw WY NC is ~90 MB on disk; the consolidator holds
  two WY datasets in memory simultaneously (WY X for Jan–Sep, WY X+1
  for Oct–Dec) plus the output CY array. Peak RSS per year is well under
  4 GB. No SLURM `--mem` bump required for this source.

- **Runtime:** per-day reprojection (NN resampling from EPSG:4269 to
  EPSG:5070, ~621×1405 → ~802×1488) is the dominant cost; expect
  ~5–10 minutes per WY on hovenweep.

- **CRS cost at aggregation:** EPSG:5070-native source means gdptools
  can reuse the same weight file across all calendar years (source grid
  is locked at consolidate time). Weight generation is a one-time
  ~5–10-minute cost per fabric; subsequent years are fast no-ops.

## Troubleshooting

- `RuntimeError: earthaccess login failed` — Earthdata credentials are
  missing or wrong. Re-run
  `nhf-targets materialize-credentials --project-dir <project>` after
  editing `.credentials.yml`.

- `ValueError: min SWE value below -1.0` — the raw file contains values
  below the guard threshold. This may indicate a format change that
  introduced an integer fill sentinel. Inspect the raw NC with
  `xr.open_dataset` and check `ds["SWE"].min()`.

- `FileNotFoundError: 4km_SWE_Depth_WY<YYYY>_v01.nc` — raw WY file is
  missing. Re-run `nhf-targets fetch ua-swe` to download the missing WY.
  Note that WY 1982 requires both WY 1982 and WY 1983 raw files to
  produce CY 1982; similarly, CY 2022 requires WY 2022 and WY 2023.
