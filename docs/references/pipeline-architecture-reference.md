# Pipeline Architecture Reference

Comprehensive reference covering all processing stages, source details, design
decisions, and transformation framework. Generated from a full codebase survey.
For a bulleted quick-reference organized by target/dataset, see
`processing-steps-reference.md`. For the target-builder view (recipes + cross-
source unit gotchas), see `calibration-target-recipes.md`.

---

## Part 1: Core Data Inventory

### A. Data Sources Registry (catalog/sources.yml)

#### RUNOFF SOURCES

**era5_land** (ECMWF ERA5-Land Reanalysis)
- **Access:** Copernicus CDS (hourly, clipped to CONUS+contributing-watersheds: 53.0°N, −125.0°W, 24.7°N, −66.0°E, ~10 km buffer)
- **Variables:** `ro` (total runoff, m/month), `sro` (surface runoff, m/month), `ssro` (subsurface runoff, m/month)
- **Time step:** Hourly (aggregated to daily and monthly)
- **Period:** 1979–present
- **Units:** m water equivalent
- **Status:** Current
- **Special handling:** Hourly accumulation resets at 00 UTC. Pipeline handles reset via substituting raw accumulated value when diff is negative, then shifts timestamps back 1 hour before resampling. Both daily and monthly consolidated NCs stored; daily rebuilt only if any hourly input is newer (mtime comparison). Annual downloads split into 12 monthly CDS requests to stay under cost limit; monthly chunks retained for idempotent re-runs.

**gldas_noah_v21_monthly** (GLDAS-2.1 NOAH Monthly Land Surface Runoff)
- **Access:** NASA GES DISC via earthaccess (Earthdata login required)
- **Variables:** `Qs_acc` (storm surface runoff, kg/m²), `Qsb_acc` (baseflow-groundwater, kg/m²), `runoff_total` (derived: `Qs_acc + Qsb_acc`)
- **Time step:** Monthly
- **Period:** 2000–present
- **Units:** kg/m² (mean of 3-hourly accumulations, NOT monthly sum — multiply by 8 × days_in_month to recover mm/month)
- **Status:** Current
- **Special handling:** Downloaded global granules, clipped to bbox at consolidation time. CRITICAL: `_acc` variables are stored as mean of 3-hourly accums, not monthly totals — the `gldas_to_mm_per_month` conversion in `targets/run.py` applies the factor.

**mwbm_climgrid** (USGS Monthly Water Balance Model, ClimGrid-forced, 2024)
- **Access:** ScienceBase manual download (CAPTCHA-gated; operator must place at `<datastore>/mwbm_climgrid/ClimGrid_WBM.nc` before fetch invoked)
- **Variables:** `runoff` (mm), `aet` (mm), `soilstorage` (mm, point-in-time), `swe` (mm, point-in-time)
- **Time step:** Monthly
- **Period:** 1900–2020 (1895–1899 treated as arbitrary spinup)
- **Units:** mm/month (runoff, aet); mm (soilstorage, swe)
- **Status:** Current (successor to retired NHM-MWBM, issue #41)
- **Spatial:** 2.5 arcmin (~0.042°), CONUS only
- **Special handling:** Single ~7.5 GB NetCDF, int16-packed with scale_factor/add_offset (xarray decodes automatically). Fingerprinted on ingest (size + sha256). Full procedure documented in `docs/sources/mwbm_climgrid.md`.

#### EVAPOTRANSPIRATION SOURCES

**mod16a2_v061** (MODIS MOD16A2 Global Terrestrial ET, v061)
- **Access:** LP DAAC via earthaccess
- **Variables:** `ET_500m` (actual evapotranspiration, kg/m²/8-day), `ET_QC_500m` (quality control)
- **Time step:** 8-day composite (aggregated to monthly)
- **Period:** 2000–present
- **Units:** kg/m²/8-day (scale_factor=0.1 in HDF; xarray decodes with decode_cf=True)
- **Spatial:** 500 m sinusoidal → reprojected to WGS84 at fetch
- **Status:** Current (v006 superseded per NASA)
- **Special handling:** Pre-aggregate hook masks flag values >3270 (water=32761, barren=32762, snow/ice=32763, cloudy=32764, no-data=32766, not-processed=32767 in raw int16 scale). **Caution:** July 2000 inspection shows essentially flat seasonality (Jul/Jan = 1.12×) while SSEBop and MWBM swing 6–11×. Inclusion in AET target pending collaborator consensus — see `docs/references/lessons-learned.md`.

**ssebop** (SSEBop Actual Evapotranspiration)
- **Access:** USGS NHGF STAC Zarr catalog (no local download; queried on-the-fly via gdptools NHGFStacZarrData; collection `ssebopeta_monthly`, doi:10.5066/P9L2YMV)
- **Variables:** `et` (mm/month)
- **Time step:** Monthly
- **Period:** 2000–2023
- **Units:** mm/month
- **Spatial:** 1 km, CONUS
- **Status:** Current
- **Special handling:** Aggregation follows different path than file-based sources — uses `aggregate/ssebop.py` directly (not the shared `_driver`); queries STAC at aggregation time.

#### RECHARGE SOURCES

**reitz2017** (Reitz et al. 2017 Empirical Recharge Estimates)
- **Access:** ScienceBase (doi:10.5066/F7PN93P0, item `56c49126e4b0946c65219231`)
- **Variables:** `total_recharge` (m/year), `eff_recharge` (base flow component, m/year)
- **Time step:** Annual
- **Period:** 2000–2013
- **Units:** m/year (catalog originally said inches/year — that was wrong; CONUS-wide mean ≈122 mm/yr validates m/year)
- **Spatial:** 800 m raster, CONUS; CRS: NAD83 geographic (EPSG:4269)
- **Status:** Current

**watergap22d** (WaterGAP 2.2d Global Hydrological Model)
- **Access:** PANGAEA (doi:10.1594/PANGAEA.918447) — open access, replaces 2.2a (registration-gated)
- **Variables:** `qrdif` (diffuse groundwater recharge, kg m⁻² s⁻¹)
- **Time step:** Monthly
- **Period:** 1901–2016
- **Units:** kg m⁻² s⁻¹
- **Spatial:** 0.5°, global
- **Status:** Current (2.2d substituted for original 2.2a)
- **Special handling:** Global grid clipped to CONUS+ bbox. HRU-aggregated CONUS+ mean ~41% below native-grid mean — intrinsic to 0.5° global grid + dry interior US over-representation in CONUS+ window; not a bug. Absolute-magnitude divergence acceptable because recharge target normalizes each source 0–1 before min/max.

**era5_land (ssro)** — recharge proxy
- Uses subsurface runoff (`ssro`) summed monthly→annual as third recharge proxy
- Same fetch/consolidation as runoff source above

#### SOIL MOISTURE SOURCES

**merra2** (MERRA-2, replaces discontinued MERRA-Land)
- **Access:** NASA GES DISC via earthaccess; product M2TMNXLND
- **Variables:**
  - `GWETTOP` (surface soil wetness, 0–0.05m layer) — **preferred**
  - `GWETROOT` (root zone, 0–1.00m integrated)
  - `GWETPROF` (full profile average, surface to bedrock, ~1.3–8.5m varying spatially)
- **Time step:** Monthly
- **Period:** 1980–present
- **Units:** Dimensionless (fraction of plant-available saturation, 0–1)
- **Spatial:** 0.5° × 0.625°, global
- **Status:** Current (MERRA-Land discontinued 2016-02-29)
- **Layer depths:** dzsf=0.05m (constant globally), dzrz=1.00m, dzpr=1.3–8.5m spatially varying

**ncep_ncar** (NCEP/NCAR Reanalysis Soil Moisture)
- **Access:** NOAA PSL direct file download
- **Variables:**
  - `soilw_0_10cm` (volumetric soil moisture, 0–10cm)
  - `soilw_10_200cm` (volumetric soil moisture, 10–200cm)
- **Time step:** Monthly
- **Period:** 1948–present
- **Units:** m³/m³ (volumetric water content) — **upstream mislabels as kg/m²**
- **Spatial:** T62 Gaussian grid (~1.875°), global; **longitudes are 0–360**
- **Status:** Current
- **Special handling:** CRITICAL: GRIB metadata and NetCDF `units` attribute say `kg/m²` but values are m³/m³ (VWC). Valid range [0, 0.43] and `long_name = "Volumetric Soil Moisture"` confirm this. Do NOT divide by layer thickness or water density.

**nldas_mosaic** (NLDAS-2 MOSAIC Land Surface Model)
- **Access:** NASA GES DISC via earthaccess
- **Variables:** `SoilM_0_10cm`, `SoilM_10_40cm`, `SoilM_40_200cm`
- **Time step:** Monthly
- **Period:** 1979–present
- **Units:** kg/m²
- **Spatial:** 0.125°, CONUS
- **Status:** Current

**nldas_noah** (NLDAS-2 NOAH Land Surface Model)
- **Access:** NASA GES DISC via earthaccess
- **Variables:** `SoilM_0_10cm`, `SoilM_10_40cm`, `SoilM_40_100cm`, `SoilM_100_200cm`
- **Time step:** Monthly
- **Period:** 1979–present
- **Units:** kg/m²
- **Spatial:** 0.125°, CONUS
- **Status:** Current
- **Special handling:** Different layer boundaries than MOSAIC (40–100cm and 100–200cm vs MOSAIC's combined 40–200cm)

#### SNOW-COVERED AREA SOURCE

**mod10c1_v061** (MODIS MOD10C1 Daily Snow Cover, v061)
- **Access:** NSIDC via earthaccess
- **Variables:**
  - `Day_CMG_Snow_Cover` (fractional snow cover, 0–100%)
  - `Day_CMG_Clear_Index` (% of cell cloud-free; renamed from `Day_CMG_Confidence_Index` in v006)
  - `Day_CMG_Cloud_Obscured` (% obscured; complementary to Clear_Index)
  - `Snow_Spatial_QA` (categorical 0–4 QA flag — **NOT a percent** despite v061 metadata `units: percent` mislabel)
- **Time step:** Daily
- **Period:** 2000–present
- **Units:** Percent (0–100); convert to fraction 0–1 in targets
- **Spatial:** 0.05° CMG, global
- **Status:** Current (v006 superseded 2023-07-31)
- **Special handling:** SCA and CI carry flag values >100 that must be masked (107=lake ice, 111=night, 237=inland water, 239=ocean, 250=cloud-obscured water, 253=data not mapped, 255=fill). CI threshold: pixels where `CI > 70%` (strict >, not ≥). Pre-aggregate hook applies per-pixel CI gate before area-weighted mean — critical because gating before vs after aggregation gives different results.

---

### B. Variable Definitions (catalog/variables.yml)

#### RUNOFF (runoff / basin_cfs)

- **PRMS variable:** `runoff`
- **Time step:** Monthly
- **Period:** 2000–present (user-configurable)
- **Units:** cfs (cubic feet per second)
- **Sources:** era5_land (`ro`), gldas_noah_v21_monthly (`runoff_total`), mwbm_climgrid (`runoff`)
- **Range method:** multi_source_minmax
- **Normalization:** False

**Range computation:**
1. ERA5-Land `ro` (m/month) → × 1000 → mm/month
2. GLDAS `runoff_total` (kg/m² as mean of 3-hourly accums) → × 8 × days_in_month → mm/month
3. MWBM `runoff` already mm/month
4. Per HRU per month: `(mm × 1e-3 / days_in_month) × hru_area_m2` → m³/day
5. m³/day → cfs via `× 35.3146667 / 86400.0`
6. `lower_bound = min(era5_cfs, gldas_cfs)`, `upper_bound = max(era5_cfs, gldas_cfs)`

**Note:** mwbm_climgrid declared as third source but `targets/run.py` currently consumes only ERA5 and GLDAS; mwbm aggregated to disk but not yet folded into bounds.

#### AET (aet / hru_actet)

- **PRMS variable:** `hru_actet`
- **Time step:** Monthly
- **Period:** 2000–2010 (original TM 6-B10 window)
- **Units:** mm/month (from sources; target in inches/day or equivalent)
- **Sources:** mod16a2_v061, ssebop, mwbm_climgrid
- **Range method:** multi_source_minmax
- **Normalization:** False (absolute values compared directly)

**Range computation:**
1. All sources aggregated to HRU polygons via gdptools
2. Per HRU per month: `lower_bound = min(mod16, ssebop, mwbm)`, `upper_bound = max(...)`
3. No per-source normalization — raw values used to preserve inter-product spread

**Status:** MOD16A2 v061 inclusion pending collaborator consensus. `targets/aet.py` should accept `sources` override from project config until decision finalized.

#### RECHARGE (recharge)

- **PRMS variable:** `recharge`
- **Time step:** Annual
- **Period:** 2000–2009 (body text of TM 6-B10; Appendix 1 says 1990–1999 — that is a typo)
- **Units:** Dimensionless (0–1 after normalization)
- **Sources:** reitz2017 (`total_recharge + eff_recharge`), watergap22d (`qrdif`), era5_land (`ssro` as proxy)
- **Range method:** normalized_minmax
- **Normalization:** True, over 2000–2009

**Range computation:**
1. Each source aggregated independently to HRU polygons
2. Each source normalized 0–1 over 2000–2009 (per-HRU): `norm = (x − min) / (max − min)`
3. ERA5-Land `ssro` summed monthly→annual (m/month → annual total m/year)
4. Per HRU per year: `lower_bound = min(norm_reitz, norm_watergap, norm_era5_ssro)`, `upper_bound = max(...)`

**Rationale:** Substantial magnitude differences between sources driven by conceptual differences in what each measures. Optimizer targets relative year-to-year change, not absolute magnitude. Normalization makes this tractable.

#### SOIL MOISTURE (soil_moisture / soil_rechr)

- **PRMS variable:** `soil_rechr`
- **Time step:** Monthly AND annual (two separate outputs)
- **Period:** 1982–2010 (user-configurable)
- **Units:** Dimensionless (0–1 after normalization)
- **Sources:** merra2 (`GWETTOP`), ncep_ncar (`soilw_0_10cm`), nldas_mosaic (`SoilM_0_10cm`), nldas_noah (`SoilM_0_10cm`)
- **Range method:** normalized_minmax
- **Normalize:** True
- **Normalize_by:** calendar_month (monthly output), full_period (annual output)

**Range computation — Annual:**
1. Each source aggregated to HRU polygons
2. Each source normalized 0–1 over full 1982–2010 period: `norm = (x − min) / (max − min)`
3. Per HRU per year: `lower_bound = min(4 sources)`, `upper_bound = max(4 sources)`

**Range computation — Monthly:**
1. Each source aggregated to HRU polygons
2. Each source normalized per calendar month independently (all Januaries together, all Februaries together, etc.) — confirmed in TM 6-B10 Appendix 1
3. Per HRU per month: `lower_bound = min(4 sources)`, `upper_bound = max(4 sources)`

**Layer depth notes:** ~0–10 cm for most sources; MERRA-2 is 0–5 cm. Normalization removes depth as confound. PRMS `soil_rechr` magnitudes not expected to match source soil-moisture magnitudes.

#### SNOW-COVERED AREA (snow_covered_area / snowcov_area)

- **PRMS variable:** `snowcov_area`
- **Time step:** Daily
- **Period:** 2000–2010
- **Units:** Fraction (0–1)
- **Source:** mod10c1_v061 only
- **Range method:** modis_ci (confidence-interval based)
- **CI threshold:** 0.70 (70%, strict > not ≥)
- **Normalization:** False

**Range computation:**
1. Per pixel per day:
   - `sca = Day_CMG_Snow_Cover / 100` (after masking flag values >100 → NaN)
   - `ci = Day_CMG_Clear_Index / 100` (after masking flag values >100 → NaN)
   - Filter: only include pixels where `ci > 0.70`
2. Area-weight to HRU polygons (pre-aggregation gate ensures only high-CI pixels contribute)
3. Per HRU per day: bounds derived from daily SCA value and associated CI

**Note:** Exact bounds formula not published. TM 6-B10: "upper and lower range calculated and treated as an error bound based on the daily snow-covered area value and the associated confidence interval" — formula lives in PRMSobjfun.f which is not publicly available. One documented open gap.

---

## Part 2: Processing Architecture

### Transformation Pipeline Policy

**Core principle:** "Aggregation is a one-way information bottleneck. Apply each transform at the spatial scale where it is defined."

| Stage | Role | What it does | What it must NOT do |
|---|---|---|---|
| `fetch/<src>.py` | Ingest | Download raw files; consolidate per-year NCs; reproject if needed | Apply scale factors, mask flag values, derive new variables, filter by quality |
| `aggregate/<src>.py` pre-hook | Pre-aggregation transforms | Flag-value masking, sums-of-accumulations, quality gates determining pixel participation in area-weighted mean | Linear scale factors, rename source variables, multi-source combinations |
| `aggregate/<src>.py` post-hook | Cosmetic-only | Attach attrs, rename auxiliary diagnostics (e.g. `valid_mask` → `valid_area_fraction`) | Modify aggregated source values |
| `normalize/methods.py` | Per-source per-HRU | 0–1 normalization, multi-source min/max bounds, NN-fill of NaN HRUs | Mask flags, reach back to pixel-level data |
| `targets/<tgt>.py` | Target assembly | Linear unit conversions, CF-compliant target NetCDF, call into normalize/ | Anything that can't be expressed at HRU scale |
| `notebooks/aggregated/*` | Diagnostic | Mirror targets/ conversions for visual verification | Define new transformations not used in targets/ |

**Why ordering matters — the math:**

Aggregation is area-weighted mean: `x̄_H = Σᵢ aᵢxᵢ / Σᵢ aᵢ` — **linear in x**.

- **Linear scales (× c)** commute with aggregation: `mean(c·x) = c·mean(x)`. Order is architectural convention — put downstream to keep aggregated NC in native units (easier to spot missed conversions).
- **Pixel masking (flags, quality gates)** does NOT commute. Pre-aggregation mask sets `xᵢ = NaN` where quality fails; post-aggregation mask applies threshold to already-mixed HRU mean — wrong answer.
- **Per-HRU normalization** is non-linear (parameters depend on data extremes). Must run post-aggregation at HRU-scale.
- **Multi-source min/max bounds** necessarily post-aggregation (different sources have different grids; must be on common HRU fabric first).

**Worked example: MOD10C1 SCA**
```
fetch/modis.py
  └─ Download HDF tiles, consolidate to per-year NC (no transformation)

aggregate/mod10c1.py pre_aggregate_hook
  ├─ Mask flag values >100 on Day_CMG_Snow_Cover → NaN
  ├─ Mask flag values >100 on Day_CMG_Clear_Index → NaN
  ├─ Apply per-pixel CI > 70 gate to SCA (NaN where fail)
  └─ Emit valid_mask = 1.0 where pixel passes, else 0.0

aggregate/mod10c1.py post_aggregate_hook
  └─ Rename valid_mask → valid_area_fraction
     (per-pixel 0/1 becomes per-HRU [0,1] fraction after area-weighting)

→ aggregated NC contains:
     Day_CMG_Snow_Cover  (native 0–100, CI-filtered)
     Day_CMG_Clear_Index (native 0–100, flag-masked only)
     valid_area_fraction (HRU fraction with CI-passing pixels)

targets/sca.py
  ├─ Read aggregated NC
  ├─ Apply ÷ 100 to Snow Cover and Clear_Index → fractional [0,1]
  ├─ Call normalize.modis_ci_bounds(sca, ci)
  └─ Write CF-compliant per-HRU per-day NetCDF
```

---

## Part 3: Aggregation Framework

### Shared Driver (_driver.py)

**Entry point:** `aggregate_source(adapter, fabric_path, id_col, workdir, batch_size, period)`

**Processing flow:**
1. **Legacy migration:** Move old layout files into per-source subdirs; unlink stale consolidated files
2. **Fail-fast validation:** Check declared variables exist in source NC (unless pre-hook synthesizes them); verify source grid is invariant across files
3. **Enumeration:** Map years covered by source files; filter to `--period` window if specified
4. **Spatial batching:** Load HRU fabric, partition into ~500-HRU spatial batches via KD-tree recursive bisection (`batching.py`)
5. **Per-year loop:**
   - If per-year aggregated NC exists → skip (idempotent)
   - Otherwise: open source file lazily → apply pre-hook → detect coords via CF attrs → for each batch: compute or load cached weights, run gdptools `AggGen(stat_method="mean")` per variable → concatenate batch results on HRU ID → apply post-hook → attach CF-1.6 global attrs → atomic write
6. **Verification:** Check contiguous year coverage in output directory
7. **Manifest update:** Record aggregation provenance (source_key, period, fabric_sha256, output files, weight files, timestamp)

**Weight caching:**
- Cached as CSV under `weights/<source_key>_batch<id>.csv`
- SHA-256 sidecar (`.csv.meta`) fingerprints sorted batch HRU IDs; validates against fabric changes
- Reused across all years for a source (grid invariance enforced)

**Coordinate detection (detect_coords):**
- First pass: CF `axis` attr (`axis='X'`/`'Y'`/`'T'`)
- Second pass: CF `standard_name` (`longitude`, `latitude`, `time`)
- Adapter can override; raises `ValueError` if detection fails and no override provided

**Per-year output naming:** `<source_key>_<YYYY>_agg.nc`

### Source-Specific Adapters

**SourceAdapter fields:**
- `source_key` — catalog entry identifier
- `variables` — tuple of variable names to aggregate
- `x_coord`, `y_coord`, `time_coord` — optional coordinate overrides
- `source_crs` — projection string (default `"EPSG:4326"`)
- `grid_variable` — variable for WeightGen grid inference (default: first in variables)
- `raw_grid_variable` — pre-hook raw variable for grid invariance check
- `files_glob` — pattern for source files (default `"*_consolidated.nc"`)
- `pre_aggregate_hook` / `post_aggregate_hook` — optional transform functions

**ERA5-Land:** No coordinate overrides; CF axis attrs detected automatically. Variables: `ro`, `sro`, `ssro`.

**GLDAS:** Pre-hook synthesizes `runoff_total = Qs_acc + Qsb_acc`; `raw_grid_variable` stays `Qs_acc` for grid-invariance check (exists pre-hook).

**MOD16A2:** Pre-hook masks `ET_500m > 3270` → NaN (flag codes 32761–32767 in raw int16 scale decode to ~3276; threshold 3270 clears all special codes). `source_crs="EPSG:4326"` (tiles reprojected from sinusoidal at fetch).

**MOD10C1:** Pre-hook applies per-pixel CI gate (see worked example above). Post-hook renames `valid_mask` → `valid_area_fraction`.

**SSEBop:** Implements its own aggregation path (`aggregate/ssebop.py`); does not use shared `_driver.py` because source is queried on-the-fly from USGS NHGF STAC Zarr, not from disk files.

---

## Part 4: Fetch Framework

### Consolidation Pattern (consolidate.py)

**Purpose:** Merge per-granule downloads into annual consolidated NCs in native units, with minimal transformation.

**Key utilities:**
- `resolve_license(meta, source_key)` — return license string from catalog; logs WARNING if missing
- `open_consolidated(nc_path)` — safely open with error guidance (corrupt file → delete and re-consolidate)
- `_write_netcdf(ds, out_path)` — atomic write via tempfile + rename; partial NC never persists at final path
- `apply_cf_metadata(ds, source_key)` — attach Conventions, source, institution global attrs

### Per-Source Fetch Modules

#### ERA5-Land (`fetch/era5_land.py`)
- **Download:** Copernicus CDS via `cdsapi`; 12 monthly requests per year
- **Hourly reset handling:** diff with `label="upper"`; negative diff → substitute raw accumulated value; shift timestamps −1 hour
- **Resampling:** hourly increments → daily sums → monthly sums
- **Output:** `monthly/era5_land_monthly_<YYYY>.nc` (primary aggregation input)
- **Locking:** Optional `fcntl` file locking on POSIX (Windows skips) for parallel-fetch safety

#### GLDAS (`fetch/gldas.py`)
- **Download:** earthaccess (NASA GES DISC); global monthly granules
- **Processing:** clip to CONUS+ bbox; derive `runoff_total = Qs_acc + Qsb_acc`; per-year consolidated NC
- **Bbox:** [53.0, −125.0, 24.7, −66.0]
- **Output:** `gldas_noah_v21_monthly_<YYYY>.nc`

#### MWBM ClimGrid (`fetch/mwbm_climgrid.py`)
- **Download:** Manual (CAPTCHA-gated ScienceBase); operator places file at `<datastore>/mwbm_climgrid/ClimGrid_WBM.nc`
- **Validation:** Fingerprint (size + SHA-256); validate CF metadata; record provenance in manifest
- **Output:** `ClimGrid_WBM.nc` (single file; `--period` clips at aggregation time)

#### MERRA-2 (`fetch/merra2.py`)
- **Download:** earthaccess; M2TMNXLND monthly global granules
- **Processing:** clip to CONUS+ bbox; consolidate per year; apply CF metadata
- **Output:** `merra2_<YYYY>.nc`

#### NCEP/NCAR (`fetch/ncep_ncar.py`)
- **Download:** NOAA PSL direct file pattern
- **Processing:** fetch monthly files; clip to CONUS+; consolidate per year
- **Output:** `ncep_ncar_<YYYY>.nc`
- **Units note:** `units: kg/m²` in upstream metadata; values are actually m³/m³ (VWC)

#### Reitz 2017 (`fetch/reitz2017.py`)
- **Download:** ScienceBase via `sciencebasepy`; per-year GeoTIFF zips
- **Processing:** unzip; mosaic GeoTIFFs; consolidate to NC per year
- **CRS:** NAD83 geographic (EPSG:4269) — not reprojected; declared in adapter
- **Output:** `reitz2017_<YYYY>.nc`

#### WaterGAP 2.2d (`fetch/watergap22d.py`)
- **Download:** PANGAEA API (doi:10.1594/PANGAEA.918447)
- **Processing:** parse monthly global data; rewrite to CF-compliant NC (`watergap22d_qrdif_cf.nc`) with decoded time for clean year-based `sel()`
- **Output:** `watergap22d_qrdif_cf.nc` (single file covers 1901–2016)

#### MOD10C1 / MOD16A2 (`fetch/modis.py`)
- **Download:** earthaccess (NSIDC for MOD10C1, LP DAAC for MOD16A2)
- **Processing:** detect and remove 0-byte files; reproject sinusoidal → WGS84; rename coords to `lon`/`lat`; consolidate per year
- **Output MOD10C1:** `mod10c1_v061_<YYYY>.nc` (daily; values in native 0–100 percent)
- **Output MOD16A2:** `mod16a2_v061_<YYYY>.nc` (8-day composites; scale_factor decoded by xarray)

---

## Part 5: Normalization & Target Building

### Normalization Methods (`normalize/methods.py`) — stubs

```python
def normalize_0_1(da, dim="time") -> DataArray:
    """Normalize DataArray to [0,1]: (x - min) / (max - min)."""

def normalize_by_calendar_month(da) -> DataArray:
    """Normalize per calendar month independently.
    For monthly soil moisture targets per TM 6-B10 Appendix 1."""

def multi_source_minmax(datasets) -> tuple[DataArray, DataArray]:
    """Compute (lower, upper) = (min, max) across sources."""

def modis_ci_bounds(sca, ci, ci_threshold=0.70) -> tuple:
    """Construct SCA bounds from CI-filtered snow cover.
    Note: exact formula TBD (PRMSobjfun.f not published)."""
```

### Target Builders

#### Runoff (`targets/run.py`) — **IMPLEMENTED**

```python
era5_to_mm_per_month(da):     return da * 1000.0
gldas_to_mm_per_month(da):    return da * 8.0 * da.time.dt.days_in_month
mm_per_month_to_cfs(da, area): ...  # (mm × 1e-3 / days) × area_m2 × 35.3147 / 86400

_validate_alignment(era5, gldas):
    # Check dims, HRU coords, time overlap; return overlapping time slices

multi_source_runoff_bounds(sources):
    stacked = xr.concat(sources, dim="source")
    return stacked.min("source"), stacked.max("source")
```
- Output: `lower_bound` and `upper_bound` variables, dims `(time, hru)`, units `cfs`, CF-1.6 attrs

#### AET (`targets/aet.py`) — **STUB**
- Sources: mod16a2_v061, ssebop, mwbm_climgrid
- Method: multi_source_minmax over absolute mm/month values
- MOD16A2 8-day → monthly: overlap-weighted sum (`(days_overlap / composite_length) × value`)
- `sources` override from project config (pending MOD16A2 decision)

#### Recharge (`targets/rch.py`) — **STUB**
- Sources: reitz2017, watergap22d, era5_land (ssro)
- Each source normalized 0–1 over 2000–2009 before min/max
- Annual output

#### Soil Moisture (`targets/som.py`) — **STUB**
- Sources: merra2, ncep_ncar, nldas_mosaic, nldas_noah
- Monthly: normalize per calendar month over 1982–2010
- Annual: normalize over full 1982–2010
- Two separate output NCs

#### Snow-Covered Area (`targets/sca.py`) — **STUB**
- Source: mod10c1_v061
- Read aggregated NC (native 0–100); apply ÷ 100; call `normalize.modis_ci_bounds(sca, ci, 0.70)`
- Daily output per HRU

---

## Part 6: Key Design Decisions

### Decision 1: Runoff — ERA5 + GLDAS replaces NHM-MWBM (Issue #41)
- NHM-MWBM was circular (PRMS-derived, used to calibrate PRMS)
- Two independent reanalysis sources provide better constraint and remove circularity
- MWBM kept in catalog; not yet consumed by `targets/run.py`
- GLDAS unit trap: `_acc` = mean of 3-hourly accumulations; requires `× 8 × days_in_month`

### Decision 2: AET — MOD16A2 v061 Inclusion Pending
- July 2000 cross-check: v061 Jul/Jan = 1.12× (flat); SSEBop and MWBM = 6–11× (expected for CONUS ET seasonality)
- Effect: v061 widens bounds in the wrong direction in both January (lower raised) and July (upper pulled down)
- Flatness present at gridded scale — not an aggregation artefact
- `targets/aet.py` to accept `sources` override from project config until resolved

### Decision 3: Recharge — Three Sources + Per-Source 0–1 Normalization
- Sources conceptually measure different things (empirical regression, process model, drainage proxy)
- Absolute magnitudes diverge substantially; optimizer targets relative year-to-year change
- WaterGAP ~41% gap vs native-grid mean: expected for CONUS+ clip of global 0.5° grid; normalization removes bias
- ERA5-Land `ssro` added as third proxy to extend years and add independent signal
- Normalization window: 2000–2009 (body text of TM 6-B10; Appendix 1 "1990–1999" is a typo)

### Decision 4: Soil Moisture — Per-Calendar-Month Normalization
- Monthly output: all Januaries normalized together, all Februaries together, etc. (confirmed TM 6-B10 Appendix 1)
- Annual output: normalized over full 1982–2010 period
- Per-source normalization makes dimensionally incompatible sources combinable (GWETTOP plant-available vs VWC vs kg/m²)
- MERRA-2 `GWETTOP` is 0–5 cm; others 0–10 cm — depth difference cancels under normalization

### Decision 5: SCA — Per-Pixel CI Gate Pre-Aggregation
- Post-aggregation gating gives wrong answer
- Pre-agg: area-weight only CI-passing pixels → `x̄_H = Σ_{i∈V} aᵢxᵢ / Σ_{i∈V} aᵢ` (valid only)
- Post-agg: area-weight all pixels, then gate on HRU-mean CI — irreversibly mixes high and low confidence pixels
- `valid_area_fraction` is a diagnostic (coverage fraction), not a re-applicable threshold

### Decision 6: MOD16A2 Flag Masking at 3270
- Raw int16 flag codes 32761–32767 decode (after scale_factor=0.1) to ~3276; threshold 3270 clears all
- ~37% of January pixels carry flag codes — masking essential
- Applied in both pre-aggregate hook and inspection notebooks

### Decision 7: Units Authority — Catalog Over NetCDF
- `catalog/sources.yml` is authoritative; already-aggregated NCs may carry stale `attrs["units"]` after catalog corrections
- Read units via `nhf_spatial_targets.catalog`, not `ds[var].attrs["units"]`
- But: validate catalog correctness via published CONUS means and spot checks (catalog has been wrong — Reitz, NCEP/NCAR)

### Decision 8: Period Flexibility
- TM 6-B10 windows documented but user-configurable
- Recharge window: 2000–2009 (body text; TM Appendix 1 typo says 1990–1999)
- All target windows recorded in target output metadata and `manifest.json`

---

## Part 7: Variable Transformation Matrix

| Source | Variable | Native Units | Aggregated Units | Normalization | Target Units |
|---|---|---|---|---|---|
| **RUNOFF** | | | | | |
| ERA5-Land | `ro` | m/month | m/month | None | cfs |
| GLDAS | `Qs_acc + Qsb_acc` | kg/m² (mean 3h) | kg/m² | None | cfs |
| MWBM | `runoff` | mm/month | mm/month | None | cfs |
| **AET** | | | | | |
| MOD16A2 | `ET_500m` | kg/m²/8-day | kg/m²/8-day | None | mm/month |
| SSEBop | `et` | mm/month | mm/month | None | mm/month |
| MWBM | `aet` | mm/month | mm/month | None | mm/month |
| **RECHARGE** | | | | | |
| Reitz 2017 | `total_recharge` | m/year | m/year | 0–1 (2000–2009) | [0, 1] |
| WaterGAP 2.2d | `qrdif` | kg m⁻² s⁻¹ | kg m⁻² s⁻¹ | 0–1 (2000–2009) | [0, 1] |
| ERA5-Land | `ssro` (annual sum) | m/year | m/year | 0–1 (2000–2009) | [0, 1] |
| **SOIL MOISTURE** | | | | | |
| MERRA-2 | `GWETTOP` | [0, 1] (plant-avail.) | [0, 1] | 0–1 (annual or per-month) | [0, 1] |
| NCEP/NCAR | `soilw_0_10cm` | m³/m³ (mislabeled kg/m²) | m³/m³ | 0–1 (annual or per-month) | [0, 1] |
| NLDAS MOSAIC | `SoilM_0_10cm` | kg/m² | kg/m² | 0–1 (annual or per-month) | [0, 1] |
| NLDAS NOAH | `SoilM_0_10cm` | kg/m² | kg/m² | 0–1 (annual or per-month) | [0, 1] |
| **SNOW COVER** | | | | | |
| MOD10C1 | `Day_CMG_Snow_Cover` | percent (0–100), CI-gated | percent (0–100, CI-filtered) | None | [0, 1] (÷100) |

---

## Part 8: Known Open Gaps

1. **SCA CI-bounds formula:** TM 6-B10 describes intent; math in PRMSobjfun.f (not public). `targets/sca.py` is a stub pending resolution.
2. **AET target builder:** Stub. Awaiting MOD16A2 v061 collaborator decision + MWBM integration.
3. **Recharge target builder:** Stub. Requires per-source normalization + multi-source min/max.
4. **Soil moisture target builder:** Stub. Requires per-calendar-month normalization (monthly) + annual normalization.
5. **MWBM in runoff bounds:** Aggregated but not yet consumed by `targets/run.py`.
