# Processing Steps Reference

Bulleted quick-reference organized by calibration target and source dataset.
Companion to `calibration-target-recipes.md` (which has the full rationale and
code patterns); this doc is the thing to pull up when someone asks "what exactly
happens to source X on the way to target Y?"

Pipeline stages in order: **fetch → aggregate (pre-hook → driver → post-hook)
→ targets (unit conversions + combination)**.

---

## 1. Runoff (`basin_cfs`)

**PRMS variable:** `runoff`
**Combination rule:** per-HRU per-month `lower = min(sources)`, `upper = max(sources)`
over absolute mm/month values (no normalization); converted to cfs before writing
**Output period:** 2000–present (user-configurable)
**Sources:** ERA5-Land `ro`, GLDAS-2.1 NOAH `runoff_total`, MWBM ClimGrid `runoff` — all three
**Builder:** `targets/run.py` (implemented)

---

### ERA5-Land (`ro`)

**Fetch (`fetch/era5_land.py`)**
- Download hourly CONUS+ data from Copernicus CDS in 12 monthly chunks per year (cost-limit strategy)
- Hourly accumulation increments computed via diff with `label="upper"`; when diff is negative (00 UTC reset), substitute the raw accumulated value
- Shift timestamps back 1 hour so 00 UTC increment lands in the correct calendar day
- Resample hourly increments to daily sums, then daily sums to monthly sums
- Write both a daily-consolidated and a monthly-consolidated NC per year (monthly is the aggregation input)
- Monthly chunk files retained for idempotent re-runs; daily NC rebuilt only if any hourly input is newer (mtime comparison)
- CONUS+ bbox: lon −125 to −66, lat 24.7 to 53

**Aggregate (`aggregate/era5_land.py`)**
- Variables aggregated: `ro`, `sro`, `ssro`
- Pre-hook: none (values are monthly sums in native m/month, no masking needed)
- Driver: area-weighted mean via gdptools per HRU per year; weights cached per batch (KD-tree spatial batching ~500 HRUs)
- Post-hook: none
- Output: `data/aggregated/era5_land/era5_land_<YYYY>_agg.nc`, native units (m/month)

**Target (`targets/run.py`)**
- Convert m/month → mm/month: `ro × 1000`
- Convert mm/month → cfs: `(mm × 1e-3) / days_in_month × hru_area_m2 × 35.3147 / 86400`
- Timestamp convention: end-of-month (e.g. `2000-01-31`)

**Key decisions**
- ERA5-Land chosen as replacement for retired NHM-MWBM runoff (issue #41); removes circularity (MWBM was PRMS-derived)
- Hourly accumulation reset handling (negative diff → substitute raw) prevents double-counting the midnight rollover

---

### GLDAS-2.1 NOAH Monthly (`runoff_total = Qs_acc + Qsb_acc`)

**Fetch (`fetch/gldas.py`)**
- Download global monthly granules from NASA GES DISC via earthaccess
- Clip to CONUS+ bbox at consolidation time
- Derive `runoff_total = Qs_acc + Qsb_acc` (both variables in kg/m²)
- Write per-year consolidated NC

**Aggregate (`aggregate/gldas.py`)**
- Variables aggregated: `Qs_acc`, `Qsb_acc`, `runoff_total`
- Pre-hook: compute `runoff_total` (sum of two components); `raw_grid_variable` stays `Qs_acc` for grid-invariance check
- Driver: area-weighted mean via gdptools per HRU per year
- Post-hook: none
- Output: `data/aggregated/gldas_noah_v21_monthly/gldas_noah_v21_monthly_<YYYY>_agg.nc`

**Target (`targets/run.py`)**
- CRITICAL unit trap: GLDAS `_acc` variables are the **mean of 3-hourly accumulations**, not a monthly sum
- Convert to mm/month: `runoff_total × 8 × days_in_month` (8 three-hourly windows per day)
- Convert mm/month → cfs: same formula as ERA5-Land
- Timestamp convention: start-of-month (e.g. `2000-01-01`) — align by calendar month window, not by date match

**Key decisions**
- Paired with ERA5-Land as the two independent reanalysis sources forming the min/max bound
- `× 8 × days_in_month` factor is non-obvious; omitting it makes GLDAS ~224–248× too small (previously a bug)

---

### MWBM ClimGrid (`runoff`)

**Fetch (`fetch/mwbm_climgrid.py`)**
- Manual download required (CAPTCHA-gated ScienceBase item 64c949bdd34e70357a34c11e)
- Operator places `ClimGrid_WBM.nc` at `<datastore>/mwbm_climgrid/ClimGrid_WBM.nc`
- Fetch invocation fingerprints the file (size + SHA-256) and records provenance in manifest
- Single NC covers 1895–2020 monthly; int16-packed (xarray decodes scale_factor/add_offset automatically)

**Aggregate**
- Variables: `runoff`, `aet`, `soilstorage`, `swe`
- Pre-hook: none (already mm/month in native units)
- Output: per-year NCs; `--period` arg clips to calibration window

**Target**
- Third source in the runoff min/max bound alongside ERA5-Land and GLDAS
- `targets/run.py` consumes all three; lower = min(era5, gldas, mwbm), upper = max(...)

---

## 2. Actual Evapotranspiration (`hru_actet`)

**PRMS variable:** `hru_actet`
**Combination rule:** per-HRU per-month `lower = min(sources)`, `upper = max(sources)` over absolute mm/month (no normalization; inter-source spread is the calibration bound)
**Output period:** 2000–2010 (TM 6-B10 default; user-configurable)
**Sources:** SSEBop `et`, MWBM ClimGrid `aet`, MOD16A2 v061 `ET_500m` — all three
**Builder:** `targets/aet.py` (stub — `NotImplementedError`)

---

### SSEBop (`et`)

**Fetch**
- No local download; accessed directly from USGS NHGF STAC Zarr catalog (collection `ssebopeta_monthly`, doi:10.5066/P9L2YMV)
- No consolidated NC on disk

**Aggregate (`aggregate/ssebop.py`)**
- Implements its own aggregation path (does not use shared `_driver.py`)
- Queries STAC at aggregation time via `gdptools.NHGFStacZarrData`
- Computes or loads cached HRU weights, runs `AggGen` per batch
- Output: per-year NCs in `data/aggregated/ssebop/`

**Target**
- Native units: mm/month — no conversion needed
- Variable name is `et` (earlier code incorrectly wrote `actual_et` — not in zarr)
- Timestamp convention: verify per-file (zarr typically start-of-month)

**Key decisions**
- Accessed on-the-fly, no local storage — avoids ~1 TB download; enables live-catalog updates
- Only AET source that is satellite-ET-derived (Operational Simplified Surface Energy Balance)

---

### MOD16A2 v061 (`ET_500m`)

**Fetch (`fetch/modis.py`)**
- Download 8-day HDF tiles from LP DAAC via earthaccess; CONUS bbox subset
- Detect and remove 0-byte files before open (earthaccess occasionally truncates without error)
- Reproject sinusoidal → WGS84 (EPSG:4326); rename coords to `lon`/`lat`
- Consolidate per year into `mod16a2_v061_<YYYY>.nc`
- `scale_factor=0.1` in HDF metadata; xarray decodes automatically with `decode_cf=True`

**Aggregate (`aggregate/mod16a2.py`)**
- Pre-hook: mask `ET_500m > 3270` → NaN (flag codes 32761–32767 in raw int16 become ~3276 after decode; threshold 3270 clears all flag codes)
  - Flag codes: 32761=water, 32762=barren, 32763=snow/ice, 32764=cloudy, 32766=no-data, 32767=not-processed
  - ~37% of January pixels carry flag codes; masking is essential
- Driver: area-weighted mean per HRU per year
- Post-hook: none
- Output: `data/aggregated/mod16a2_v061/mod16a2_v061_<YYYY>_agg.nc`

**Target**
- Native: kg/m²/8-day composite; 1 kg/m² water = 1 mm depth
- 8-day → monthly conversion: weight each composite by `(days of overlap with month) / composite_length`; sum weighted contributions
  - Year-end composite (DOY 361) covers 5–6 days, not 8; use actual composite length, not 8
  - Treating all composites as 8-day over-counts the year-end composite
- After overlap weighting: result is mm/month

**Key decisions / open issues**
- v006 decommissioned 2023-07-31; v061 is required for all new runs
- **MOD16A2 v061 flat-on-CONUS+:** July 2000 cross-check shows Jul/Jan = 1.12× for v061 vs 6–11× for SSEBop and MWBM; v061 pulls the min/max bound in the wrong direction in both January (lower raised) and July (upper pulled down)
- **Status:** inclusion in AET target pending collaborator consensus; `targets/aet.py` should accept `sources` override from project config until resolved
- See `docs/references/lessons-learned.md` § MOD16A2 v061 flat-on-CONUS+

---

### MWBM ClimGrid (`aet`)

**Fetch / Aggregate:** same single NC as the runoff MWBM source above; `aet` variable aggregated alongside `runoff`

**Target**
- Native units: mm/month — no conversion needed
- CONUS-only at 2.5 arcmin; full spatial coverage within CONUS HRU fabric

---

## 3. Recharge

**PRMS variable:** `recharge`
**Combination rule:** each source normalized 0–1 over 2000–2009 independently per HRU, then `lower = min(normalized)`, `upper = max(normalized)` per year
**Output period:** annual, 2000–2009 (normalization window matches calibration window)
**Builder:** `targets/rch.py` (stub — `NotImplementedError`)

**Normalization rationale:** sources measure conceptually different fluxes (empirical regression vs process model vs soil drainage proxy); absolute magnitudes diverge substantially especially in arid regions. Optimizer targets relative year-to-year change, making normalization tractable and making absolute differences irrelevant.

---

### Reitz 2017 (`total_recharge`)

**Fetch (`fetch/reitz2017.py`)**
- Download per-year GeoTIFF zips from ScienceBase (doi:10.5066/F7PN93P0, item `56c49126e4b0946c65219231`) via `sciencebasepy`
- Unzip and mosaic GeoTIFFs; CRS is NAD83 geographic (EPSG:4269) — not reprojected
- Consolidate to NC per year; apply CF metadata
- Annual, 800 m raster, CONUS only, period 2000–2013

**Aggregate (`aggregate/reitz2017.py`)**
- Variables: `total_recharge`, `eff_recharge`
- Pre-hook: none (values are physical; no flag masking required)
- Driver: area-weighted mean per HRU per year; `source_crs="EPSG:4269"` declared in adapter
- Post-hook: none
- Output: `data/aggregated/reitz2017/reitz2017_<YYYY>_agg.nc`

**Target**
- Native units: m/year — multiply by 1000 → mm/year before normalization
- Unit trap: earlier catalog had `inches/year` (a guess); CONUS mean ~122 mm/yr validates m/year

**What it measures:** empirical regression estimate of total groundwater recharge

---

### WaterGAP 2.2d (`qrdif`)

**Fetch (`fetch/watergap22d.py`)**
- Download from PANGAEA (doi:10.1594/PANGAEA.918447) via `pangaeapy` — open access
- Single NC covers 1901–2016 monthly global at 0.5°; clip to CONUS+ bbox
- Rewritten to CF-compliant NC (`watergap22d_qrdif_cf.nc`) with decoded time so `sel(time=str(year))` works without manual cftime handling

**Aggregate (`aggregate/watergap22d.py`)**
- Variables: `qrdif`
- Pre-hook: none
- Driver: area-weighted mean per HRU per year
- Post-hook: none
- Output: `data/aggregated/watergap22d/watergap22d_<YYYY>_agg.nc`

**Target**
- Native units: kg m⁻² s⁻¹ (rate, monthly mean)
- Convert: for each month, `× seconds_in_month` (= `days_in_month × 86400`) → mm/month; sum 12 months → mm/year
- Known artifact: HRU-aggregated CONUS+ mean ~41% below native-grid mean — expected for 0.5° global grid with dry interior US over-represented after CONUS+ clip; normalization removes this offset

**What it measures:** process-modelled diffuse groundwater recharge (WaterGAP hydrological model)
**Supersedes:** WaterGAP 2.2a (registration-gated, no longer accessible)

---

### ERA5-Land (`ssro`) — recharge proxy

**Fetch / Aggregate:** same consolidated NCs as the runoff ERA5-Land source; `ssro` aggregated alongside `ro` and `sro`

**Target**
- Native: m/month (monthly accumulation)
- Sum 12 months → m/year; multiply by 1000 → mm/year; normalize 0–1 over 2000–2009
- Used as a third recharge proxy (not in TM 6-B10; added to extend available years and provide a third independent signal)

**What it measures:** sub-surface runoff = drainage out the bottom of the ERA5-Land soil column; physically proximal to but not formally equivalent to groundwater recharge

---

## 4. Soil Moisture

**PRMS variable:** `soil_rechr`
**Combination rule:** each source normalized 0–1 independently, then `lower = min(4 sources)`, `upper = max(4 sources)`
**Two outputs required:**
- **Monthly:** normalize per calendar month (all Januaries together, all Februaries together, etc.) over 1982–2010 — confirmed in TM 6-B10 Appendix 1
- **Annual:** normalize over full 1982–2010 period
**Builder:** `targets/som.py` (stub — `NotImplementedError`)

**Normalization rationale:** sources use fundamentally different physical quantities (plant-available wetness, volumetric water content, mass per area); per-source 0–1 normalization cancels constant offsets and makes GWETTOP (0–1 dimensionless) combinable with VWC (0.05–0.45 m³/m³) and kg/m² (0–40 kg/m²).

---

### MERRA-2 (`GWETTOP`)

**Fetch (`fetch/merra2.py`)**
- Download M2TMNXLND monthly global granules from NASA GES DISC via earthaccess
- Clip to CONUS+ bbox; consolidate per year
- Period: 1980–present, 0.5° × 0.625°

**Aggregate (`aggregate/merra2.py`)**
- Variables: `GWETTOP` (preferred), `GWETROOT`, `GWETPROF`
- Pre-hook: none (dimensionless, no flag masking)
- Driver: area-weighted mean per HRU per year
- Post-hook: none
- Output: `data/aggregated/merra2/merra2_<YYYY>_agg.nc`

**Target**
- Native: dimensionless 0–1, plant-available wetness = `(W − W_wilt) / (W_sat − W_wilt)` for 0–5 cm layer
- Do **not** convert to volumetric water content or mm depth via layer thickness — the definition is fundamentally different from VWC; the 0–1 normalization handles this
- Timestamp convention: mid-month (e.g. `1980-01-15`)
- Layer depth: 0–5 cm (MERRA-2 surface skin layer, dzsf = 0.05 m constant globally)

**Key decisions**
- Supersedes MERRA-Land (discontinued 2016-02-29); `GWETTOP` confirmed from GMAO documentation and layer-depth analysis
- Layer is shallower than other SOM sources (0–5 cm vs 0–10 cm); responds faster to precipitation/drying; normalization removes depth as confound

---

### NLDAS-2 NOAH (`SoilM_0_10cm`)

**Fetch (`fetch/nldas.py`)**
- Download monthly CONUS granules from NASA GES DISC via earthaccess
- CONUS only, 0.125° grid, 1979–present

**Aggregate (`aggregate/nldas_noah.py`)**
- Variables: `SoilM_0_10cm`, `SoilM_10_40cm`, `SoilM_40_100cm`, `SoilM_100_200cm`
  - Note: different layer boundaries than MOSAIC (NOAH uses 40–100 cm and 100–200 cm; MOSAIC combines as 40–200 cm)
- Pre-hook: none
- Driver: area-weighted mean per HRU per year
- Post-hook: none
- Output: `data/aggregated/nldas_noah/nldas_noah_<YYYY>_agg.nc`

**Target**
- Native: kg/m² in 0–10 cm layer
- Convert to VWC: `÷ (0.10 m × 1000 kg/m³) = ÷ 100` → m³/m³
- Timestamp convention: start-of-month (e.g. `1980-01-01`)

---

### NLDAS-2 MOSAIC (`SoilM_0_10cm`)

**Fetch (`fetch/nldas.py`)** — same fetch module as NOAH

**Aggregate (`aggregate/nldas_mosaic.py`)**
- Variables: `SoilM_0_10cm`, `SoilM_10_40cm`, `SoilM_40_200cm`
- Pre-hook / driver / post-hook: same as NOAH
- Output: `data/aggregated/nldas_mosaic/nldas_mosaic_<YYYY>_agg.nc`

**Target**
- Native: kg/m² — same ÷ 100 VWC conversion as NOAH
- Timestamp convention: start-of-month

---

### NCEP/NCAR Reanalysis 1 (`soilw_0_10cm`)

**Fetch (`fetch/ncep_ncar.py`)**
- Download via NOAA PSL direct file pattern
- Global at T62 Gaussian grid (~1.875°); **longitudes are 0–360** — wrap to −180–180 before CONUS clip
- Monthly, 1948–present

**Aggregate (`aggregate/ncep_ncar.py`)**
- Variables: `soilw_0_10cm`, `soilw_10_200cm`
- Pre-hook: lon wrap to −180–180; clip to CONUS+ bbox
- Driver: area-weighted mean per HRU per year
- Post-hook: none
- Output: `data/aggregated/ncep_ncar/ncep_ncar_<YYYY>_agg.nc`

**Target**
- CRITICAL unit trap: upstream NetCDF carries `units: kg/m2` but values are **volumetric water content** (m³/m³)
  - Evidence: `long_name = "Volumetric Soil Moisture"`, `valid_range = [0.0, 1.0]`, `actual_range ≈ [0.10, 0.43]`
  - Physical impossibility: 0.43 kg/m² would be physically impossible for a 0–10 cm layer; 0.43 m³/m³ is plausible VWC
  - **Do NOT divide by 100** — pass through as-is
- Timestamp convention: end-of-month (e.g. `1980-01-31`)
  - `time.sel(method="nearest")` against mid-month silently returns December for January target — always slice to calendar-month window and take first hit

---

## 5. Snow-Covered Area (`snowcov_area`)

**PRMS variable:** `snowcov_area`
**Single source:** MOD10C1 v061 only
**Combination rule:** CI-based bounds from daily SCA and associated Clear_Index per HRU
**Output:** daily, per-HRU, fraction [0, 1]
**Output period:** 2000–2010 (user-configurable)
**Builder:** `targets/sca.py` (stub — `NotImplementedError`)
**Open gap:** exact bounds formula not published; PRMSobjfun.f not publicly available

---

### MOD10C1 v061 (`Day_CMG_Snow_Cover` + `Day_CMG_Clear_Index`)

**Fetch (`fetch/modis.py`)**
- Download daily HDF files from NSIDC via earthaccess; CONUS+ bbox
- Detect and remove 0-byte files before open
- Reproject sinusoidal → WGS84; rename coords to `lon`/`lat`
- Consolidate per year into `mod10c1_v061_<YYYY>.nc`
- Variables retained: `Day_CMG_Snow_Cover`, `Day_CMG_Clear_Index`, `Day_CMG_Cloud_Obscured`, `Snow_Spatial_QA`
- **DO NOT use `Snow_Spatial_QA` as the CI filter** — it is a categorical 0–4 quality flag mislabelled `units: percent` in HDF metadata; earlier code did `ci = Snow_Spatial_QA / 100; ci > 0.70` which passes only flag codes and rejects all valid pixels

**Aggregate (`aggregate/mod10c1.py`)**
- Pre-hook (must run before area-weighted mean — see why below):
  1. Mask `Day_CMG_Snow_Cover > 100` → NaN (flag values: 107=lake ice, 111=night, 237=inland water, 239=ocean, 250=cloud-obscured water, 253=not mapped, 255=fill)
  2. Mask `Day_CMG_Clear_Index > 100` → NaN (same flag set)
  3. Compute per-pixel `pass_mask = (ci_masked > 70)` — strict `>`, not `≥`
  4. Set `Day_CMG_Snow_Cover = NaN` where `pass_mask == False`
  5. Emit `valid_mask = 1.0` where pixel passes CI gate, else `0.0`
- Driver: area-weighted mean per HRU per day for all three variables (`Day_CMG_Snow_Cover`, `Day_CMG_Clear_Index`, `valid_mask`)
- Post-hook:
  1. Rename `valid_mask` → `valid_area_fraction` (the per-pixel 0/1 is now a per-HRU fraction [0, 1] after area-weighting)
  2. Warn if >10% of (HRU, time) cells have `valid_area_fraction == 0`
- Output: `data/aggregated/mod10c1_v061/mod10c1_v061_<YYYY>_agg.nc`
  - Variables: `Day_CMG_Snow_Cover` (native 0–100 percent, CI-filtered), `Day_CMG_Clear_Index` (native 0–100, flag-masked only), `valid_area_fraction` ([0, 1] per HRU)

**Why CI gate must be pre-aggregation (not post)**
- Pre-agg: area-weight only CI-passing pixels → HRU gets high-confidence SCA estimate
- Post-agg: area-weight all pixels (mixing high and low CI), then filter by HRU-mean CI → wrong answer; an HRU with 50% high-CI snowy + 50% low-CI cloudy pixels gives the same HRU-mean CI either way, but the pre-agg answer correctly represents only the high-confidence half
- `valid_area_fraction` is a diagnostic (what fraction of the HRU had CI-passing pixels) — it is not a re-applicable threshold; do not gate on it post-aggregation

**Target (`targets/sca.py`)**
- Read aggregated NC (native 0–100 percent)
- Rescale: `sca = Day_CMG_Snow_Cover / 100`, `ci = Day_CMG_Clear_Index / 100` → [0, 1] fractions
- Call `normalize.modis_ci_bounds(sca, ci, threshold=0.70)` → `(lower_bound, upper_bound)` per HRU per day
- Write CF-compliant daily NC
- Timestamp convention: daily, start of day

**Key decisions**
- v006 decommissioned 2023-07-31; v061 required (variable renamed from `Day_CMG_Confidence_Index` to `Day_CMG_Clear_Index`)
- CI threshold: 70%, strict `>` (matches TM 6-B10 language; using `≥` would include borderline cloudy pixels)
- Bounds formula: still an open gap — TM 6-B10 describes intent ("error bound from daily SCA value and CI") but the math is in PRMSobjfun.f which is not public

---

## Cross-cutting conventions

### Aggregation driver

- **`stat_method="mean"`** (not `masked_mean`) — produces NaN for any HRU with no source-grid coverage, making gaps explicit and auditable
- **Spatial batching:** ~500-HRU batches via KD-tree recursive bisection; weights cached per batch as CSV + SHA-256 sidecar
- **Weight reuse:** weights valid across all years for a given source (grid invariance enforced at startup)
- **Idempotent:** per-year output NC skipped if already exists
- **Atomic writes:** tempfile + rename — no partial NC ever lands at final path
- **Manifest:** provenance written after each year (source_key, period, fabric_sha256, output files, weight files, timestamp)

### NaN HRU fill

- NaN HRUs after aggregation (partial or no source coverage) filled by nearest-neighbor in HRU space
- Fill runs as a shared post-processing step in `normalize/` before any combination or normalization
- Keeps aggregation stage honest about coverage; fill location is single auditable place

### Timestamp alignment across sources

- ERA5-Land: end-of-month
- GLDAS: start-of-month
- MERRA-2: mid-month
- NLDAS NOAH/MOSAIC: start-of-month
- NCEP/NCAR: end-of-month
- MOD10C1: daily (start of day)
- **Pattern:** always slice to a calendar-month window and take the first hit; never use `time.sel(method="nearest")` for cross-source alignment

### Units: read from catalog, not from on-disk NetCDF

- `catalog/sources.yml` is authoritative for source units; already-consolidated NCs may carry stale `attrs["units"]` strings after a catalog correction
- Read units via `nhf_spatial_targets.catalog`, not `ds[var].attrs["units"]`
- Still: validate with published CONUS means and spot checks — the catalog itself has been wrong (Reitz m/year vs inches/year; NCEP/NCAR kg/m² vs m³/m³)

### Transformation placement rule

| Operation | Where | Why |
|---|---|---|
| Flag-value masking | Pre-aggregation hook | Does not commute with area-weighted mean |
| Per-pixel quality gates (CI > 70) | Pre-aggregation hook | Same — post-gating gives wrong HRU mean |
| Derive composite variable (Qs_acc + Qsb_acc) | Pre-aggregation hook | Must exist before gdptools runs |
| Rename diagnostic variable | Post-aggregation hook | Cosmetic only; value not touched |
| Linear unit conversion (× 1000, ÷ 100, × 8 × N) | Target builder | Commutes with mean; keep aggregated NC in native units for auditability |
| 0–1 normalization | `normalize/methods.py` | Defined at HRU scale; must run post-aggregation |
| Multi-source min/max bounds | Target builder | Requires both sources on same HRU fabric first |
