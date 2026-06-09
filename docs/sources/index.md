# Sources

This pipeline ingests **16 active gridded sources**, plus one (SSEBop) accessed remotely via STAC at aggregation time without a local fetch. Each source has its own operator notes covering access pathway, authentication quirks, on-disk layout, known gaps, and HPC tuning observations.

## Source-by-target map

| Target | Sources |
|---|---|
| Runoff | [ERA5-Land](era5_land.md) · [GLDAS](gldas.md) · [MWBM ClimGrid](mwbm_climgrid.md) |
| AET | [MOD16A2 v061](mod16a2_v061.md) · [SSEBop](ssebop.md) · [MWBM ClimGrid](mwbm_climgrid.md) |
| Recharge | [Reitz 2017](reitz2017.md) · [WaterGAP 2.2d](watergap22d.md) · [ERA5-Land](era5_land.md) |
| Soil moisture | [MERRA-2](merra2.md) · [NCEP/NCAR](ncep_ncar.md) · [NLDAS-MOSAIC](nldas_mosaic.md) · [NLDAS-NOAH](nldas_noah.md) |
| Snow cover (SCA) | [MOD10C1 v061](mod10c1_v061.md) · [UA SWE](ua_swe.md) |
| SWE | [Daymet](daymet.md) · [SNODAS](snodas.md) · [ERA5-Land](era5_land.md) · [Margulis WUS-SR](margulis_wus_sr.md) · [UA SWE](ua_swe.md) |

## Per-source docs

Each source page documents:

- **Provider, license, citation** — upstream landing page, DOI, and any account / acceptance prerequisites
- **Access pathway** — earthaccess, CDS API, HTTPS, sciencebasepy, pangaeapy, etc.
- **On-disk layout** — what lives under `<datastore>/<source_key>/` after a successful fetch
- **Per-variable units** — cross-referenced to `catalog/sources.yml` `cf_units` (catalog is the single source of truth; never read units off on-disk NetCDF attrs)
- **Known quirks** — flag values to mask, archive gaps, unit-conversion gotchas, PR# references for past fixes
- **HPC notes** — memory ceilings, throttling rules, parallelism strategies observed on caldera-hovenweep

When the on-disk NC attrs contradict the operator doc, **the catalog is authoritative**. See [`docs/architecture/transformation-pipeline.md`](../architecture/transformation-pipeline.md) for the rule.

## Adding a new source

See [Contributing · Adding a new source](../contributing.md#adding-a-new-source) for the file-by-file checklist.
