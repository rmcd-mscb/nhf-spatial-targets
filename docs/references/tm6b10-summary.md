# TM 6-B10 crib sheet — NHM-PRMS byHRU calibration targets

Short reference for agents working on `src/nhf_spatial_targets/targets/` and
`catalog/variables.yml`. Authoritative source: Hay et al. (2023),
**Techniques and Methods 6-B10**, doi:[10.3133/tm6B10](https://doi.org/10.3133/tm6B10).

- Full PDF: [`tm6b10.pdf`](./tm6b10.pdf) (62 pp., 13 MB)
- Searchable markdown: [`tm6b10.md`](./tm6b10.md) (pymupdf4llm; figures/equations
  marked `==> picture intentionally omitted <==`)

## What the report actually does

TM 6-B10 documents a three-part CONUS-scale calibration of NHM-PRMS:

- **byHRU** (water-balance volume) — each of 109,951 HRUs calibrated independently
  against five non-streamflow baseline datasets. *This is the part this repo
  produces inputs for.*
- **byHW** (streamflow timing) — 7,265 headwater watersheds ≤3,000 km², using
  ordinary-kriging statistical streamflow series. Out of scope here.
- **byHWobs** — subset with measured streamflow. Out of scope here.

byHRU ranks calibration steps per HRU by FAST (Fourier Amplitude Sensitivity
Test) weights; steps with zero sensitivity for an HRU are skipped (e.g. SCA in
snow-free basins). The objective function is NRMSE, assessed only when the
simulated value falls **outside** the range of the baseline; inside the range,
error is treated as zero.

## The five byHRU targets

For each HRU, per time step, a target file provides a **range** (min, max) — and
for RUN/AET, individual source values too. The Fortran optimizer
(`PRMSobjfun.f`) consumes these files; we produce catalog-driven equivalents.

> **Reminder on periods and versions.** The "Report window" column below shows
> what TM 6-B10 used. In this repo, actual target windows are driven by
> available data and user config (see `CLAUDE.md` → "Relationship to TM 6-B10").
> The "Report sources" column lists the *original* datasets; the parenthetical
> substitutes are what `catalog/sources.yml` actually uses today.

### 1. RUN — Runoff

| Field | Value |
|---|---|
| PRMS variable | `basin_cfs` (streamflow leaving the basin) |
| Time step | monthly |
| Report window | 1982–2010 |
| Units (report) | cubic feet per second |
| Report sources | NHM-MWBM (Bock et al. 2018) — single source, with built-in stochastic uncertainty bounds |
| **This repo** | ERA5-Land `ro` + GLDAS-2.1 NOAH `Qs_acc + Qsb_acc` (replaces NHM-MWBM; issue #41) |
| Range rule (report) | Bock et al. 2018 post-processed CI (source-internal) |
| Range rule (this repo) | multi-source min/max across ERA5-Land and GLDAS-2.1 NOAH |
| Appendix 1 file schema | `Year, Month, Value, Minimum, Maximum` |

**Builder:** `src/nhf_spatial_targets/targets/run.py` · **catalog:** `variables.yml → runoff`

### 2. AET — Actual Evapotranspiration

| Field | Value |
|---|---|
| PRMS variable | `hru_actet` |
| Time step | monthly, mean monthly |
| Report window | 2000–2010 |
| Units | inches |
| Report sources | SSEBop (Senay et al. 2013), MOD16 (v006), NHM-MWBM |
| **This repo** | SSEBop (via USGS NHGF STAC), MOD16A2 **v061** (v006 decommissioned), MOD16 v006 → v061 per CLAUDE.md |
| Range rule | min/max across the three source values per HRU and time step |
| Normalization | **none** — absolute values compared directly |
| Appendix 1 file schema | `Year, Month, AET1, AET2, AET3` (min/max derived inside PRMSobjfun.f; missing values encoded as negatives; at least one source must be present per step) |

**Builder:** `src/nhf_spatial_targets/targets/aet.py` · **catalog:** `variables.yml → aet`

### 3. RCH — Recharge

| Field | Value |
|---|---|
| PRMS variable | `recharge` (to associated groundwater reservoir) |
| Time step | annual |
| Report window | **body text says 2000–2009; Appendix 1 says 1990–1999** — see "Contradictions" below |
| Units | dimensionless (normalized 0–1) |
| Report sources | Reitz et al. 2017 empirical regression, WaterGAP **2.2a** (Döll et al. 2014) |
| **This repo** | Reitz 2017 + WaterGAP **2.2d** (2.2a not publicly available; PANGAEA doi:10.1594/PANGAEA.918447) + ERA5-Land `ssro` (sub-surface runoff as a third proxy) |
| Range rule | each source normalized 0–1 over the calibration window, then per-year min/max across sources |
| Normalization rationale | "substantial differences in recharge magnitude" between sources, driven by conceptual differences — optimizer targets similarity in *relative year-to-year change*, not magnitude |
| Appendix 1 file schema | `Year, Minimum, Maximum` (both values in [0, 1]) |

**Builder:** `src/nhf_spatial_targets/targets/rch.py` · **catalog:** `variables.yml → recharge`

### 4. SOM — Soil Moisture

| Field | Value |
|---|---|
| PRMS variable | `soil_rechr` (upper capillary reservoir — not total soil column) |
| Time step | monthly **and** annual (two separate target files) |
| Report window | 1982–2010 |
| Units | dimensionless (normalized 0–1) |
| Report sources | MERRA-Land, NCEP-NCAR reanalysis, NLDAS-MOSAIC, NLDAS-NOAH |
| **This repo** | MERRA-**2** (`GWETTOP`, 0–0.05 m; MERRA-Land superseded), NCEP-NCAR (`soilw.0-10cm`), NLDAS-MOSAIC (`SoilM_0_10cm`), NLDAS-NOAH (`SoilM_0_10cm`). GLDAS-2.1 also in catalog. |
| Range rule (annual) | normalize each source 0–1 over the full calibration window, then per-year min/max across sources |
| Range rule (monthly) | **per-calendar-month** normalization — all Januaries normalized together, all Februaries together, etc. (confirmed in Appendix 1) — then per-month min/max across sources |
| Comparison note | PRMS `soil_rechr` magnitudes are *not expected* to match the source soil-moisture magnitudes; only *relative change* is compared — hence the normalization |
| Appendix 1 file schema | `SOMann`: `Year, Minimum, Maximum` · `SOMmth`: `Year, Month, Minimum, Maximum` |

**Builder:** `src/nhf_spatial_targets/targets/som.py` · **catalog:** `variables.yml → soil_moisture`

### 5. SCA — Snow-Covered Area

| Field | Value |
|---|---|
| PRMS variable | `snowcov_area` |
| Time step | daily |
| Report window | 2000–2010 |
| Units | fraction (0–1) |
| Report source | MOD10C1 (MODIS/Terra Snow Cover Daily L3 Global 0.05 Deg CMG; Hall & Riggs 2016) |
| **This repo** | MOD10C1 **v061** (v006 decommissioned 2023-07-31) |
| Filtering | use only pixels/days with confidence interval **> 70 %** (`ci > 0.70`) |
| Range rule | "upper and lower range … calculated and treated as an error bound based on the daily snow-covered area value and the associated confidence interval" — **exact formula not disclosed**; lives in `PRMSobjfun.f` which is not publicly available |
| Appendix 1 file schema | `Year, Month, Day, SCA, CI` — bounds are derived inside PRMSobjfun.f, not carried in the target file |

**Builder:** `src/nhf_spatial_targets/targets/sca.py` · **catalog:** `variables.yml → snow_covered_area`

## Contradictions / open questions inside the report

- **RCH normalization window.** Body text (p. 9): "normalized for the range
  from 0 to 1 over the period 2000–2009." Appendix 1 (p. 37): "_RCH_ file …
  covering the time period from 1990 to 1999" and "PRMSobjfun.f … normalizes
  … over the period from 1990 to 1999." The appendix date is almost certainly
  a carry-over typo from an earlier draft — the body text is consistent with
  the Reitz 2017 period. CLAUDE.md currently follows the body (2000–2009).
  Either way: **this repo does not hardcode it** — normalization window is
  driven by user config and recorded in target metadata.
- **SCA CI-bounds formula.** Report describes the intent but not the math; the
  Fortran source for `PRMSobjfun.f` is not published. This is the one
  still-open gap in `CLAUDE.md → Known Gaps`.

## Objective-function mechanics (byHRU)

Useful context when deciding what target files must contain:

- NRMSE per time step, aggregated to per-step objective function, then
  weighted sum across steps using FAST-derived weights per HRU.
- For **range-based** evaluation (runoff, AET, SOM, RCH, SCA): if the simulated
  value is **inside** `[min, max]`, error = 0; if **outside**, NRMSE is computed
  against the nearer bound. So the range is the tolerance band, not the target
  value. This is why sources that disagree widely (wide band) are less
  constraining than sources that agree (narrow band).
- Calibration sequence per HRU: SCA → SOM → RUN → RCH → AET → ALL, **reordered
  by FAST weights** so the most-sensitive target goes first. A zero-weight
  target is skipped.
- SCE (Shuffled Complex Evolution; Duan et al. 1992–94) is the optimizer;
  top 25 % of parameter sets from each step seed the next step's range.

## Where TM 6-B10 does *not* help

- Deriving new target types beyond the original five.
- Consuming modern datasets the report predates (ERA5-Land, GLDAS-2.1, updated
  MODIS versions) — use `catalog/sources.yml` for provenance and the per-fetch
  module docstrings for variable choices.
- Implementation-level details of `PRMSobjfun.f` (e.g. SCA CI-bounds formula).
- Fabric-specific concerns (the report uses the original NHM GeoSpatial Fabric;
  this repo is fabric-agnostic and takes fabric path from project config).

## Cross-references

- `catalog/variables.yml` — per-target source list, range method, units, current periods
- `catalog/sources.yml` — per-source fetch metadata, versions, supersession chains
- `src/nhf_spatial_targets/targets/{run,aet,rch,som,sca}.py` — target builders
- `src/nhf_spatial_targets/normalize/methods.py` — normalization helpers
- `CLAUDE.md → Known Gaps` — live status of each catalog substitution
- `CLAUDE.md → Relationship to TM 6-B10` — policy on time windows and versions
