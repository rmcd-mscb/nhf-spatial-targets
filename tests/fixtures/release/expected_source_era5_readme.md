# ECMWF ERA5-Land Reanalysis (consolidated for NHF)

ECMWF ERA5-Land hourly reanalysis. Total runoff (ro), surface runoff (sro), sub-surface runoff (ssro), and snow depth water equivalent (sd) downloaded for CONUS plus contributing watersheds (Canada/ Mexico). Used as a source for the runoff calibration target (ro), the recharge calibration target (ssro, as drainage proxy), and the snow water equivalent calibration target (sd). These data were consolidated to CF-1.6 NetCDF as part of the USGS National Hydrologic Model calibration-targets workflow.

## Purpose

Document and distribute the consolidated form of the ECMWF ERA5-Land Reanalysis dataset as used in the USGS National Hydrologic Model workflow.

## Source citation

- Muñoz-Sabater, J., and others, 2021, doi:10.5194/essd-13-4349-2021

## Extent

- Temporal: 1979-01-01 to present
- Bounding coordinates (W, S, E, N): -125.0, 24.7, -66.0, 53.0

## Variables

| Name | Description | Units | Cell methods |
| --- | --- | --- | --- |
| ro | total runoff | m | time: sum |
| sro | surface runoff | m | time: sum |
| ssro | sub-surface runoff | m | time: sum |
| sd | snow depth water equivalent | m | time: point |

## Files

- `monthly/era5_land_monthly_2003.nc` — Consolidated CF-1.6 NetCDF.
- `daily/era5_land_daily_2003.nc` — Consolidated CF-1.6 NetCDF.

## Use constraints

None. Users are advised to read the dataset's metadata thoroughly to understand appropriate use and data limitations.

Source dataset license: Copernicus license (free, attribution).

This dataset was generated using Copernicus Climate Change Service (C3S) information (ECMWF ERA5-Land). Neither the European Commission nor ECMWF is responsible for any use of the Copernicus information or data it contains.

## Contact

- Point of contact: U.S. Geological Survey (gs-w-nhm_calibration@usgs.gov)
- Distributed by: U.S. Geological Survey - ScienceBase (sciencebase@usgs.gov)

## Provenance

Source granules for ECMWF ERA5-Land Reanalysis were downloaded and consolidated to CF-1.6 NetCDF by the nhf-spatial-targets pipeline.

2 process step(s) recorded; see `fgdc.xml` and the bundled `manifest.json` for the full lineage.
