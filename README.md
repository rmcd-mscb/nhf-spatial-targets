# nhf-spatial-targets

Curated calibration target datasets for the [National Hydrologic Model (NHM)](https://www.usgs.gov/mission-areas/water-resources/science/national-hydrologic-model-infrastructure).

Builds the five baseline calibration targets documented in [Hay and others (2022), USGS TM 6-B10](https://doi.org/10.3133/tm6B10) by spatially aggregating gridded source datasets to an NHM HRU fabric using [gdptools](https://github.com/rmcd-mscb/gdptools).

## Calibration Targets

| Target | PRMS Variable | Sources | Method | Time Step |
|---|---|---|---|---|
| Runoff | `basin_cfs` | NHM-MWBM | MWBM uncertainty | Monthly |
| AET | `hru_actet` | MWBM + MOD16A2 + SSEBop | Multi-source min/max | Monthly |
| Recharge | `recharge` | Reitz 2017 + WaterGAP 2.2a | Normalized min/max | Annual |
| Soil Moisture | `soil_rechr` | MERRA-2 + NCEP/NCAR + NLDAS-MOSAIC + NLDAS-NOAH | Normalized min/max | Monthly + Annual |
| Snow Cover | `snowcov_area` | MOD10C1 | MODIS CI bounds | Daily |

## Quick Start

```bash
# Install with pixi
pixi install

# Run all targets
pixi run run

# Run a single target
pixi run run-aet

# Inspect the catalog
pixi run catalog-sources
pixi run catalog-variables
```

## Configuration

Edit `config/pipeline.yml` to set:
- Fabric path and HRU ID column
- Period of record
- Which targets to enable
- Source dataset versions (e.g. MOD16A2 v006 vs v061)

## Repository Structure

```
nhf-spatial-targets/
├── catalog/
│   ├── sources.yml          # data source registry with access info and gaps
│   └── variables.yml        # calibration target definitions and range methods
├── config/
│   └── pipeline.yml         # run configuration
├── src/nhf_spatial_targets/
│   ├── fetch/               # per-source download modules
│   ├── aggregate/           # gdptools spatial aggregation
│   ├── normalize/           # normalization and CI bound construction
│   ├── targets/             # per-variable target builders
│   ├── catalog.py           # catalog interface
│   └── cli.py               # nhf-targets CLI
├── tests/
├── pixi.toml
└── pyproject.toml
```

## Known Gaps

See `catalog/sources.yml` for per-source notes. Key items to resolve before a production run:

- **MWBM**: Locate exact ScienceBase item ID for HRU-level uncertainty bounds
- **MOD16A2 / MOD10C1**: Update from v006 to v061
- **MERRA-Land**: Replace with MERRA-2 (already scaffolded in sources.yml)
- **WaterGAP 2.2a**: Registration-gated; assess substitute or ScienceBase copy
- **SCA bounds formula**: Verify `sca × ci` construction against `PRMSobjfun.f`
- **Recharge normalization window**: Confirm 2000–2009 vs 1990–1999 (Appendix 1 discrepancy)

## References

- Hay, L.E., and others, 2022, USGS TM 6-B10
- Bock, A.R., and others, 2016/2018 — NHM-MWBM
- Reitz, M., and others, 2017 — Recharge estimates
