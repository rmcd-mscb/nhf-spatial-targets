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
| Snow cover | `snowcov_area` | MOD10C1 v061 | MODIS CI-bounded (CI > 70 %) | Daily |
| SWE | `pkwater_equiv` | Daymet · SNODAS · ERA5-Land · Margulis WUS-SR¹ | NaN-aware multi-source min/max | Daily |

¹ Margulis Western US Snow Reanalysis (NSIDC-0719) is **fabric-scoped to Oregon** via `catalog/sources.yml → margulis_wus_sr.fabric_scope`. Non-Oregon projects reduce the SWE bound to the remaining three sources at target-build time; raw downloads remain reusable across any project sharing the datastore.

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
