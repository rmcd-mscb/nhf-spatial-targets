# Calibration targets for the gfv2 hydrologic fabric, U.S. Geological Survey National Hydrologic Model

This child item provides calibration-target datasets for the gfv2 hydrologic fabric (109951 hydrologic response units) of the USGS National Hydrologic Model. Targets: runoff, aet, snow_water_equivalent. Participating sources: daymet, era5_land, gldas_noah_v21_monthly, snodas. Project-specific note appended to the fabric abstract.

## Purpose

Provide calibration targets aggregated to the gfv2 fabric for parameter estimation of the USGS National Hydrologic Model.

## Citation

- Doe, Jane (U.S. Geological Survey) [ORCID: 0000-0000-0000-0000]
- Roe, John

## Extent

- Temporal: 2000-01-01 to present
- Bounding coordinates (W, S, E, N): -124.7, 24.5, -66.95, 49.4

## Calibration targets

| Target | Description | Units | Range method |
| --- | --- | --- | --- |
| runoff | Monthly basin mean runoff. Range is min/max across two reanalysis sources per HRU and time step. | cfs | multi_source_minmax |
| aet | Monthly actual evapotranspiration. Range is min/max across three independent source datasets per HRU and time step. Absolute values used (not normalized); range reflects inter-product spread. | inches/day | multi_source_minmax |
| snow_water_equivalent | Daily basin snow water equivalent. Range is min/max across five independent SWE sources per HRU and time step. Margulis Western US Snow Reanalysis is fabric-scoped to Oregon only (see fabric_scope in catalog/sources.yml); non-Oregon fabrics are bounded by the remaining four sources (daymet, snodas, era5_land, ua_swe). UA SWE (NSIDC-0719) reaches back to water year 1982, extending the bound well before the SNODAS 2003 start. Absolute values used (not normalized). | inches | multi_source_minmax |

## Files

- `fabric.gpkg` — Hydrologic fabric GeoPackage.
- `aggregated/era5_land/era5_land_2003.nc` — Source dataset area-weighted to the fabric (CF-1.6 NetCDF).
- `aggregated/snodas/snodas_2003.nc` — Source dataset area-weighted to the fabric (CF-1.6 NetCDF).
- `targets/runoff_targets.nc` — Calibration-target dataset (CF-1.6 NetCDF).
- `targets/swe_targets.nc` — Calibration-target dataset (CF-1.6 NetCDF).
- `manifest.json` — Provenance manifest.

## Use constraints

None. Users are advised to read the dataset's metadata thoroughly to understand appropriate use and data limitations.

Participating source dataset licenses: public domain (NASA / ORNL DAAC); Copernicus license (free, attribution); public domain (NASA); public domain (NSIDC / NOAA).

These data are distributed by the Oak Ridge National Laboratory Distributed Active Archive Center (ORNL DAAC), Oak Ridge, Tennessee, USA.

This dataset was generated using Copernicus Climate Change Service (C3S) information (ECMWF ERA5-Land). Neither the European Commission nor ECMWF is responsible for any use of the Copernicus information or data it contains.

These data were obtained from the NASA Earthdata system. We acknowledge the use of data products from the NASA Earth science data systems.

These data are distributed by the NASA National Snow and Ice Data Center Distributed Active Archive Center (NSIDC DAAC).

## Contact

- Point of contact: U.S. Geological Survey (gs-w-nhm_calibration@usgs.gov)
- Distributed by: U.S. Geological Survey - ScienceBase (sciencebase@usgs.gov)

## Provenance

Each gridded source dataset was area-weighted to the gfv2 fabric with gdptools, then combined into per-target ranges by the nhf-spatial-targets pipeline.

5 process step(s) recorded; see `fgdc.xml` and the bundled `manifest.json` for the full lineage.
