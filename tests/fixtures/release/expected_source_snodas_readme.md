# SNODAS Snow Data Assimilation System Data Products (NSIDC G02158) (consolidated for NHF)

NOAA NOHRSC SNODAS daily snow products distributed by NSIDC (collection G02158). Daily 1 km CONUS analysis combining satellite observations, ground stations, and an energy-balance snow model. Used as one source for the SWE calibration target. These data were consolidated to CF-1.6 NetCDF as part of the USGS National Hydrologic Model calibration-targets workflow.

## Purpose

Document and distribute the consolidated form of the SNODAS Snow Data Assimilation System Data Products (NSIDC G02158) dataset as used in the USGS National Hydrologic Model workflow.

## Source citation

- National Operational Hydrologic Remote Sensing Center, 2004, doi:10.7265/N5TB14TC

DOI: https://doi.org/10.7265/N5TB14TC

## Extent

- Temporal: 2003-01-01 to present

## Variables

| Name | Description | Units | Cell methods |
| --- | --- | --- | --- |
| swe | snow water equivalent | kg m-2 | time: point |

## Files

- `daily/snodas_2003.nc` — Consolidated CF-1.6 NetCDF.

## Use constraints

None. Users are advised to read the dataset's metadata thoroughly to understand appropriate use and data limitations.

Source dataset license: public domain (NSIDC / NOAA).

These data are distributed by the NASA National Snow and Ice Data Center Distributed Active Archive Center (NSIDC DAAC).

## Contact

- Point of contact: U.S. Geological Survey (gs-w-nhm_calibration@usgs.gov)
- Distributed by: U.S. Geological Survey - ScienceBase (sciencebase@usgs.gov)

## Provenance

Source granules for SNODAS Snow Data Assimilation System Data Products (NSIDC G02158) were downloaded and consolidated to CF-1.6 NetCDF by the nhf-spatial-targets pipeline.

1 process step(s) recorded; see `fgdc.xml` and the bundled `manifest.json` for the full lineage.
