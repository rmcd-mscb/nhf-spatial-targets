# WaterGAP 2.2d Diffuse Groundwater Recharge (PANGAEA)

WaterGAP 2.2d global hydrological model output (Müller Schmied et al.,
2021; doi:10.5194/gmd-14-1037-2021), distributed via PANGAEA
(doi:10.1594/PANGAEA.918447). Used as the second source for the
recharge (`rch`) calibration target range alongside
[Reitz 2017](reitz2017.md); the `rch` target is a `normalized_minmax`
two-source target, so absolute magnitudes wash out.

This is the **open-access successor to WaterGAP 2.2a**, which was
registration-gated at the time of TM 6-B10. The 2.2a entry is
preserved in the catalog with `status: superseded_registration_required`
for provenance only.

The pipeline downloads one variable from one file:

| Variable | File variable | Long name | Native units |
| -------- | ------------- | --------- | ------------ |
| `groundwater_recharge` | `qrdif` | diffuse groundwater recharge | `kg m-2 s-1` |

The publisher's NC4 (`watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4`)
is **a single ~few-hundred-MB global monthly file covering 1901-2016**;
spatial subsetting happens at aggregation time, not at fetch.

## Access path — pangaeapy

No NASA/CDS credentials needed. The fetch module
[`fetch/pangaea.py`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/fetch/pangaea.py)
uses [pangaeapy](https://github.com/pangaea-data-publisher/pangaeapy)
to download the file by row index in the PanDataSet table (catalog
`file_index: 30`). The download is sanity-checked against the
expected filename stem before being moved into the datastore — a
publisher reorganisation would surface as a clear `RuntimeError`
rather than silently substituting a different file.

## CF compliance fix-up

PANGAEA's NC4 ships with **CF-non-conformant metadata** for this
pipeline's purposes — most notably a `time` axis encoded as
`months since 1901-01-01` (integer offsets, not CF-conformant
calendar units). The fetch module reconstructs the time coord as
`datetime64`, then runs `apply_cf_metadata` to add the WGS84 grid
mapping, coordinate axis attrs, and `Conventions=CF-1.6` global, and
writes the corrected file as
`<datastore>/watergap22d/watergap22d_qrdif_cf.nc`. The raw NC4 is
preserved on disk for forensic reference.

## On-disk layout

```
<datastore>/watergap22d/
  watergap_22d_WFDEI-GPCC_histsoc_qrdif_monthly_1901_2016.nc4   # raw
  watergap22d_qrdif_cf.nc                                          # CF-corrected (consumed by aggregator)
  .pangaea_cache/                                                  # pangaeapy cache (ok to delete)
```

## Procedure

```bash
nhf-targets fetch watergap22d --project-dir <project> --period 1979/2016
```

Or via the general SLURM fetch array, index 6
([`slurm/shared/fetch_all.slurm`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/slurm/shared/fetch_all.slurm)).

Note the **publisher dataset ends 2016**; `period` end-years past
2016 are silently truncated at aggregation time (the file simply has
no data past 2016-12). Multi-source recharge runs targeting 2017+ will
need WaterGAP 2.2e or a different second source.

The `period` argument is **provenance-only** for this fetcher — the
file itself covers the full 1901-2016 range and is not subsettable at
download time. The aggregator's `--period` is what actually clips
output.

## HPC memory/time notes

- The file is ~ few hundred MB; the fetch is fast (single connection
  to PANGAEA, no chunking). The general 128 GB / 24 h fetch SLURM
  allocation is overkill — the operation runs in minutes.
- Re-runs are idempotent: if `watergap22d_qrdif_cf.nc` already exists,
  the fetcher skips re-download and re-fix-up.

## Aggregator wiring

`aggregate/watergap22d.py` is a thin `SourceAdapter` (monthly cadence,
`stat_method="mean"`, `files_glob="*_cf.nc"` so the raw NC4 is not
accidentally re-aggregated). The aggregator's per-year emission walks
the monthly time axis and writes
`<project>/data/aggregated/watergap22d/watergap22d_<year>_agg.nc`,
one per year, in native `kg m-2 s-1`. The `rch` target builder
applies the linear conversion to mm/yr downstream.

## Grid resolution and choropleth quirk

WaterGAP 2.2d is on a **0.5° global grid** (~55 km). On
intermountain-west fabrics with many small HRUs per source cell, the
aggregated choropleth looks visibly blocky (see
`project_coarse_source_blockiness` memo). This is the visual
evidence for the gfv2 fabric excluding WaterGAP from the `rch` target;
per-fabric decision, not a pipeline bug.

## Troubleshooting

- `RuntimeError: PANGAEA file index 30 contains '<name>', expected '<stem>'`
  — PANGAEA reorganised the dataset listing. Inspect
  `pan_ds.data` from a Python session and update `file_index` in
  `catalog/sources.yml`.
- `RuntimeError: Failed to connect to PANGAEA dataset 918447` — check
  <https://doi.pangaea.de/10.1594/PANGAEA.918447> in a browser; the
  server is occasionally offline.
- Catalog reference: see [`catalog/sources.yml`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/catalog/sources.yml)
  `watergap22d:` block for the file index and DOI.
