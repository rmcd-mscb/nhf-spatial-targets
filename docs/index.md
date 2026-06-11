# nhf-spatial-targets

Curated calibration target datasets for the [USGS National Hydrologic Model (NHM)](https://www.usgs.gov/mission-areas/water-resources/science/national-hydrologic-model-infrastructure).

This pipeline builds the baseline calibration targets documented in [Hay and others (2022), USGS TM 6-B10](https://doi.org/10.3133/tm6B10) by spatially aggregating gridded source datasets to an NHM HRU fabric using [gdptools](https://github.com/rmcd-mscb/gdptools). Where original sources have been retired (MOD16A2 v006, MOD10C1 v006, MERRA-Land, NHM-MWBM, WaterGAP 2.2a) the catalog substitutes the modern replacement — see [Known gaps resolved](references/known-gaps-resolved.md).

## Six calibration targets

| Target | PRMS variable | Sources | Method | Cadence |
|---|---|---|---|---|
| Runoff | `basin_cfs` | ERA5-Land · GLDAS-NOAH · MWBM ClimGrid | NaN-aware multi-source min/max | Monthly |
| AET | `hru_actet` | MOD16A2 v061 · SSEBop · MWBM ClimGrid | Multi-source min/max | Monthly |
| Recharge | `recharge` | Reitz 2017 · WaterGAP 2.2d · ERA5-Land | Normalized min/max | Annual |
| Soil moisture | `soil_rechr` | MERRA-2 · NCEP/NCAR · NLDAS-MOSAIC · NLDAS-NOAH | Normalized min/max per calendar month | Monthly + annual |
| Snow cover | `snowcov_area` | MOD10C1 v061 · UA SWE² | NaN-aware bound: MODIS CI-interval (CI ≥ 70 %) ∪ depth-derived fraction | Daily |
| SWE | `pkwater_equiv` | Daymet · SNODAS · ERA5-Land · UA SWE² · Margulis WUS-SR¹ | NaN-aware multi-source min/max | Daily |

¹ Margulis Western US Snow Reanalysis covers only the **Western US**; it contributes NaN-aware to the HRUs it covers and drops out elsewhere (#309 — the former fabric_scope/fabric.token gate is gone). Outside its domain the SWE bound falls back to the remaining four sources (Daymet, SNODAS, ERA5-Land, UA SWE); raw downloads remain reusable across any project sharing the datastore.

² UA Daily 4-km SWE (NSIDC-0719; Broxton, Zeng & Dawson 2019) is CONUS-wide for calendar years 1982–2022 and is the only SWE source reaching before SNODAS's 2003 start. It is a fifth SWE source and the second SCA source (its depth-derived snow-covered fraction extends the CI-bound back to 1982); see [#237](https://github.com/rmcd-mscb/nhf-spatial-targets/issues/237).

## Where to start

- **First-time setup, day-to-day commands:** [Getting started](getting-started.md)
- **What lives on disk where, and why:** [Architecture · Transformation pipeline](architecture/transformation-pipeline.md)
- **Per-source operator notes (gotchas, units, archive gaps):** [Sources overview](sources/index.md)
- **Adding a new source / target / fabric:** [Contributing guide](contributing.md)
- **Project conventions (pre-commit gate, manifest concurrency, test discipline):** [Conventions](conventions.md)
- **Python patterns that show up everywhere (`from __future__ import annotations`, atomic writes, fingerprint caches):** [Architecture · Python patterns](architecture/python-patterns.md)

## How this site is built

These pages are rendered locally via MkDocs + Material. To run the site on your laptop:

```bash
pixi install -e docs
pixi run docs-serve            # http://127.0.0.1:8000 with hot reload
```

The API reference pages under [API reference](api/index.md) are auto-generated from the numpy-style docstrings in `src/nhf_spatial_targets/` via [mkdocstrings](https://mkdocstrings.github.io/) — they update live as docstrings change.

Publishing to GitHub Pages is not enabled. See [issue #233](https://github.com/rmcd-mscb/nhf-spatial-targets/issues/233) for the local-only-for-now scope decision.
