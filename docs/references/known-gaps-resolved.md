# Known Gaps — Resolved

These items were previously tracked in CLAUDE.md as open gaps and have since
been confirmed or fixed. Kept here for historical reference.

## Resolved

- Reitz 2017 ScienceBase item ID — confirmed: `56c49126e4b0946c65219231`, doi:10.5066/F7PN93P0
- Runoff source replacement — NHM-MWBM removed; replaced by ERA5-Land (CDS) + GLDAS-2.1 NOAH monthly. ERA5-Land ssro also added as third recharge source. Closes issue #41.
- Recharge normalization window — confirmed **2000-2009** from TM 6-B10 body text
- MOD16A2 / MOD10C1 v006 → v061: both decommissioned; use v061 in all new runs
- MERRA-2 variable — use `GWETTOP` (0-0.05m, dimensionless); product M2TMNXLND
- MERRA-2 layer depths — dzsf=0.05m (constant globally), dzrz=1.00m (per GMAO FAQ), dzpr=spatially varying (surface to bedrock, ~1.3-8.5m). Thicknesses in M2CONXLND collection.
- NLDAS NOAH variable names — confirmed from file inspection: SoilM_0_10cm, SoilM_10_40cm, SoilM_40_100cm, SoilM_100_200cm
- WaterGAP 2.2d — confirmed: doi:10.1594/PANGAEA.918447, variable qrdif (diffuse groundwater recharge), 1901-2016 monthly, 0.5° global, CC BY-NC 4.0

## Resolved (previously open)

- SSEBop — accessed via USGS NHGF STAC catalog (collection `ssebopeta_monthly`, doi:10.5066/P9L2YMV, 2000-2023 monthly, 1km). Aggregated directly to HRU fabric via gdptools — no local download. See PR #34.
- MOD16A2 v061 flat-on-CONUS+ seasonality — root cause was fill-value contamination at the consolidate-time sinusoidal→4 km reprojection: `rioxarray`'s `masked=True` only masks the declared `_FillValue`, leaving the other special codes (water=32761, barren=32762, snow/ice=32763, cloudy=32764, no-data=32766) to be averaged into valid neighbours by `Resampling.average`. Fixed in PR #88 by masking ET_500m fills *before* reprojection. Existing consolidated/aggregated NCs are invalid; re-fetch + re-aggregate to recover real seasonality. See `docs/references/lessons-learned.md` § MOD16A2 v061 flat-on-CONUS+.
