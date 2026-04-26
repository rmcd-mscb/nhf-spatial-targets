# Calibration Target Recipes

This is the practical bridge between consolidated source NetCDFs in
`<datastore>/<source_key>/` and the per-target builders in
`src/nhf_spatial_targets/targets/`. The TM 6-B10 summary
(`tm6b10-summary.md`) explains *what* targets to build; this doc
explains *how* each source we have should be read, converted, time-
selected, and combined.

The catalog (`catalog/sources.yml`, `catalog/variables.yml`) carries
per-source / per-variable `notes` blocks that document upstream
mislabelings; this doc is the target-side view that pulls those
together by calibration target so a target-builder author has one
place to look.

Status: most of this is consolidated from the `inspect_consolidated_*`
notebooks (PR #68) and the catalog corrections that fell out of them.
Several target builders are still stubs raising `NotImplementedError`;
the recipes below are guidance for filling them in.

---

## Per-target recipes

### 1. Runoff (RUN)

- **PRMS variable:** `runoff`
- **Sources:** ERA5-Land `ro`, GLDAS-2.1 NOAH `runoff_total = Qs_acc + Qsb_acc`
- **Builder:** `src/nhf_spatial_targets/targets/run.py` (implemented)

**Native-unit conversion to mm/month**

- ERA5-Land `ro`: m of water / month (monthly accumulation, `cell_methods: time: sum`).
  Multiply by 1000 to get mm/month.
- GLDAS NOAH `runoff_total`: kg m⁻², stored as the **mean of 3-hourly accumulations**
  for the month (NOT a monthly sum), per the NASA GES DISC GLDAS-2.1 README.
  Multiply by `8 × days_in_month` to recover mm/month. Implemented in
  `gldas_to_mm_per_month`. Earlier versions of the builder used an identity
  transform here, which made GLDAS values 224–248× too small.

**mm/month → cfs**

`mm/month × 1e-3 m/mm × HRU_area_m² / days_in_month × 35.3147 ÷ 86400`. Implemented
in `mm_per_month_to_cfs`.

**Time conventions (cross-source)**

ERA5-Land timestamps are at end-of-month (e.g. `2000-01-31`); GLDAS at start-of-month
(`2000-01-01`). Both refer to January's accumulation. `time.sel(method="nearest")`
with a mid-month target picks the right month for each, but downstream code aligning
the two sources should slice to a calendar month rather than match exact dates.

**Spatial extent**

ERA5-Land is already subset to CONUS plus contributing watersheds (lon −125 to −66,
lat 24.7 to 53). GLDAS is global and gets aggregated through gdptools onto the same
HRU fabric, so spatial alignment lands at HRU resolution.

**Combination rule**

Per-HRU per-month: `lower_bound = min(era5_cfs, gldas_cfs)`,
`upper_bound = max(era5_cfs, gldas_cfs)`.

**Open / verify**

- ERA5-Land's CONUS-and-contributing-watersheds bbox is in
  `src/nhf_spatial_targets/fetch/era5_land.py`. Confirm that GLDAS aggregation is
  using the same fabric so the multi-source min/max is per-HRU consistent.

---

### 2. Actual evapotranspiration (AET)

- **PRMS variable:** `hru_actet`
- **Sources:** SSEBop `et`, MOD16A2 v061 `ET_500m`
- **Builder:** `src/nhf_spatial_targets/targets/aet.py` (status: stub at last check)

**Native-unit conversion to mm/month**

- SSEBop `et`: native mm/month, monthly time step. No conversion. **Variable name is
  `et`** — earlier code wrote `actual_et`, which doesn't exist in the zarr.
- MOD16A2 `ET_500m`: kg m⁻² per 8-day composite (with units `kg m-2 / 8-day`).
  1 kg m⁻² of water = 1 mm depth, so 1 composite value = mm of ET over 8 days.
  Aggregation to a calendar month: weight each composite by
  `(days of overlap with target month) / 8`, then sum the weighted contributions
  for all composites that intersect the month.

**Time conventions**

- SSEBop monthly, timestamps at start-of-month (typical zarr convention; verify per file).
- MOD16A2 8-day composites with timestamps at the start of the 8-day window
  (`A{year}{doy}` filename pattern).

**Spatial extent**

- SSEBop: CONUS, accessed remotely via the USGS NHGF STAC catalog — no local
  consolidated NC. Aggregation reads the zarr directly.
- MOD16A2: global tiles; we keep yearly consolidated NCs subset around CONUS+.

**Combination rule (TM 6-B10 / per repo)**

Per-HRU per-month: range across both sources defines the calibration error bound.

**Open / verify before implementing**

- **MOD16A2 scale_factor.** The inspect notebook flagged that domain mean for
  Jan 2000 was ~4491 mm/month, vs SSEBop's ~8 mm/month — a 500× offset that
  almost certainly indicates the HDF/NetCDF `scale_factor` (typically 0.1 for
  MOD16A2 ET) is not being applied. Confirm the consolidated NC was opened with
  `decode_cf=True`, or apply the scale factor explicitly in the builder. This is
  the AET-equivalent of the Reitz / NCEP / GLDAS issues — values look "wrong by
  exactly N×" because of an upstream encoding the catalog didn't follow.

---

### 3. Recharge (RCH)

- **PRMS variable:** `recharge`
- **Sources:** Reitz 2017 `total_recharge`, WaterGAP 2.2d `qrdif`, ERA5-Land `ssro`
- **Builder:** `src/nhf_spatial_targets/targets/rch.py` (status: stub)
- **TM 6-B10 originally used Reitz + WaterGAP 2.2a. We replace 2.2a with 2.2d
  (open-access on PANGAEA) and add ERA5-Land ssro as a third proxy.**

**Native-unit conversion to mm/year**

- Reitz `total_recharge`: m/year (annual). Multiply by 1000 → mm/year.
  Earlier catalog versions said `inches/year`; that was wrong, the source
  ScienceBase release stores values in m/year. CONUS mean ~122 mm/year now matches
  the published ~162 mm/year.
- WaterGAP `qrdif`: kg m⁻² s⁻¹ (rate, monthly mean). For each month in the target
  year, `× seconds_in_month` (= `days_in_month × 86400`) → mm/month. Sum 12 months
  → mm/year.
- ERA5-Land `ssro`: m of water / month (monthly accumulation, `cell_methods: time: sum`).
  Sum 12 months → m/year, then × 1000 → mm/year.

**Time conventions**

Reitz: annual, timestamp typically mid-year (July 1). WaterGAP: monthly, start-of-month.
ERA5-Land: monthly, end-of-month.

**Spatial extent**

Reitz is CONUS-only at 0.0083° (1 km). WaterGAP and ERA5-Land are global / CONUS+. For
inspection, the recharge notebook clips all three to ERA5-Land's footprint; the target
builder should aggregate each to HRUs and then operate per-HRU.

**Combination rule (TM 6-B10)**

Each source normalised 0–1 over the calibration window (catalog default is 2000–2009
per the body text). Per-HRU per-year: `lower_bound = min(norm_reitz, norm_watergap, norm_era5_ssro)`,
`upper_bound = max(...)`. Optimisation targets relative year-to-year change, not absolute
magnitude — which is fortunate because the three products measure conceptually different
fluxes.

**Conceptual note on what each source measures**

- Reitz `total_recharge`: empirical regression estimate of total groundwater recharge.
- WaterGAP `qrdif`: process-modelled diffuse groundwater recharge.
- ERA5-Land `ssro`: sub-surface runoff (drainage out the bottom of the soil column);
  used as a proxy for recharge, not formally equivalent.

These will diverge in absolute magnitude, especially in arid regions — the per-source
0–1 normalisation is what makes them combinable.

---

### 4. Soil moisture (SOM)

- **PRMS variable:** `soil_rechr`
- **Sources:** MERRA-2 `GWETTOP`, NLDAS-2 NOAH `SoilM_0_10cm`,
  NLDAS-2 MOSAIC `SoilM_0_10cm`, NCEP/NCAR R1 `soilw_0_10cm`
- **Builder:** `src/nhf_spatial_targets/targets/som.py` (status: stub)

**Native-unit handling**

- **MERRA-2 GWETTOP**: dimensionless 0–1, **plant-available** wetness fraction
  (`(W − W_wilt) / (W_sat − W_wilt)`), 0–5 cm layer. **NOT volumetric water content** —
  do not try to convert to mm depth via layer thickness. Pass through as-is.
- **NLDAS-2 NOAH `SoilM_0_10cm`**: kg m⁻² in the 0–10 cm layer. Divide by
  `(layer_depth_m × ρ_water) = 0.10 × 1000 = 100` to get volumetric water content
  in m³/m³.
- **NLDAS-2 MOSAIC `SoilM_0_10cm`**: same as NOAH (kg m⁻² → ÷ 100 → VWC).
- **NCEP/NCAR R1 `soilw_0_10cm`**: **already volumetric water content** (m³/m³)
  despite the upstream NetCDF labelling its `units` as `kg/m2`. The file's own
  `long_name` ("volumetric soil moisture 0-10 cm"), `var_desc` ("Volumetric Soil
  Moisture"), and `valid_range` ([0.0, 1.0]) all confirm VWC; `actual_range` ~[0.10,
  0.43] is consistent with VWC and physically impossible if the values were mass
  per area. Pass through unchanged. **Do not divide by 100.**

**Time conventions (cross-source)**

This is the messiest set. Every source uses a different convention:

- MERRA-2: mid-month (e.g. `1980-01-15`).
- NLDAS-2 NOAH/MOSAIC: start-of-month (`1980-01-01`, sometimes `1979-12-31` for
  edge cases on a Gaussian grid).
- NCEP/NCAR R1: end-of-month (`1980-01-31`).

`time.sel(method="nearest")` against a mid-month date silently picks December for
NCEP/NCAR (wrong calendar month). **Always slice to a month window and take the
first hit** rather than nearest-to-date, e.g.:

```python
def _select_month(da, year, month):
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    sliced = da.sel(time=slice(start, end))
    return sliced.isel(time=0)
```

**Spatial extent / longitudes**

NLDAS NOAH and MOSAIC are CONUS-only at 0.125°. MERRA-2 is global at ~0.5° × 0.625°.
NCEP/NCAR R1 is global at ~1.875° on a Gaussian grid **with longitudes in 0–360 convention**
— wrap to −180–180 before any CONUS clip:

```python
def _wrap_lon_to_minus180_180(da):
    lon_dim = "lon" if "lon" in da.dims else "x"
    if float(da[lon_dim].max()) > 180:
        new_lon = ((da[lon_dim] + 180) % 360) - 180
        da = da.assign_coords({lon_dim: new_lon}).sortby(lon_dim)
    return da
```

For inspection, NLDAS NOAH's lat/lon is the natural CONUS reference footprint.

**Combination rule (TM 6-B10)**

Each source normalised 0–1 independently. Monthly target: normalise per calendar
month over the 1982–2010 calibration period (all Januaries together, all Februaries
together, etc.) — confirmed in TM 6-B10 Appendix 1. Annual target: normalise over
the full 1982–2010 period. Per-HRU: `lower_bound = min(norm_merra, norm_nldas_noah,
norm_nldas_mosaic, norm_ncep)`, `upper_bound = max(...)`.

The per-source normalisation is what makes GWETTOP (plant-available, 0.1–0.9 typical)
combinable with VWC (0.05–0.45 typical) — the constant offset cancels.

**Conceptual notes**

- Layer depths differ: MERRA-2 0–5 cm, others 0–10 cm. MERRA-2 will respond faster to
  precipitation/drying than the deeper-layer products.
- MERRA-Land was the original TM 6-B10 source; this repo uses MERRA-2 (`GWETTOP`)
  per the catalog `superseded_by` chain.

---

### 5. Snow-covered area (SCA)

- **PRMS variable:** `snowcov_area`
- **Sources:** MOD10C1 v061 — `Day_CMG_Snow_Cover` (SCA) and
  `Day_CMG_Clear_Index` (CI). The consolidated NC also contains
  `Day_CMG_Cloud_Obscured` (kept for QA cross-checks) and
  `Snow_Spatial_QA` (categorical 0–4 quality flag, **not the CI**).
- **Builder:** `src/nhf_spatial_targets/targets/sca.py` (status: stub)

**Native-unit handling**

- `Day_CMG_Snow_Cover`: 0–100 percent of cell with snow. Mask flag values
  (107 = lake ice, 111 = night, 237 = inland water, 239 = ocean,
  250 = cloud-obscured water, 253 = data not mapped, 255 = fill) before any
  quantitative use; valid range is 0–100. Without the mask, CONUS-mean for a
  typical day is dominated by 237/239/250 codes and lands near 100 — physically
  meaningless.
- `Day_CMG_Clear_Index`: 0–100 percent of cell that was cloud-free on that day.
  Same flag values, same mask. **This is what TM 6-B10 calls "confidence interval"**:
  > "a confidence interval equal to 100 percent indicates clear sky conditions
  > with the highest level of confidence"
  In MOD10C1 v006 this variable was named `Day_CMG_Confidence_Index`; renamed to
  `Day_CMG_Clear_Index` in v061. The repo's earlier consolidation dropped this
  variable entirely.
- `Snow_Spatial_QA`: 0–4 categorical (0=best, 1=good, 2=ok, 3=poor, 4=other) plus
  flag values (237/239/250/252/253/254/255). The source HDF labels this `units: percent`
  but it is not a percent. **Do NOT divide by 100; do NOT use as the CI filter.**
  Earlier catalog versions had `ci = Snow_Spatial_QA / 100; ci > 0.70` which would
  pass *only* the special-case flag codes and reject every legitimate QA value.

**Time conventions**

Daily, timestamps at the start of the day (`A2000055` = day 55 of 2000).

**Spatial extent**

0.05° CMG, global. Consolidation subsets to CONUS+ via the project bbox.

**Combination rule (TM 6-B10)**

Per-HRU per-day:

1. Mask flag values.
2. `sca = Day_CMG_Snow_Cover / 100`, `ci = Day_CMG_Clear_Index / 100`.
3. Drop any `(time, lat, lon)` cell where `ci ≤ 0.70`.
4. Aggregate to HRUs via gdptools using `sca` from cells passing the CI filter.
5. Derive lower/upper bounds from the daily SCA value and the associated CI per
   the report's "error bound based on the daily SCA value and the associated
   confidence interval" — **exact formula remains unconfirmed.** PRMSobjfun.f
   is not publicly available; this is the one open methodology gap in
   `CLAUDE.md → Known Gaps`.

---

## Cross-cutting lessons

### 1. Upstream metadata can lie about units

Pattern observed three separate times in this repo:

- **Reitz 2017**: ScienceBase release documented as m/year, our catalog inherited
  `inches/year` (a guess at the time), values came out 25× too small after the
  inches→mm conversion in the inspection notebook.
- **NCEP/NCAR R1**: NetCDF carries `units: kg/m2` for what the same NetCDF's
  `long_name` and `valid_range` clearly identify as volumetric water content.
- **MOD10C1 `Snow_Spatial_QA`**: NetCDF carries `units: percent` for a 0–4
  categorical.

The pattern is consistent: when a source product looks suspicious (mean
several orders of magnitude off, all-NaN after conversion, or values clustered
near zero on a shared colour scale), check at least three of:

1. The file's `valid_range` and `actual_range` attributes (a 0-1 valid_range
   on a "kg/m2"-labelled variable is a strong tell).
2. The file's `long_name` and `var_desc` attributes (these usually describe
   the physical quantity correctly even when `units` doesn't).
3. The published mean from the source paper (Reitz CONUS-mean 6.4 in/yr ≈ 162
   mm/yr; SOM VWC 0.05–0.45; SCA 0–100).
4. Spot-checks at known-magnitude locations (Olympic Peninsula recharge ~30 in/yr,
   Phoenix recharge ~0.1 in/yr).

### 2. Timestamp conventions vary per source

Within the SOM target alone we see end-of-month (NCEP), start-of-month (NLDAS),
and mid-month (MERRA-2) conventions. Cross-source `time.sel(method="nearest",
target="YYYY-MM-15")` against a single date silently picks the wrong calendar
month for end-of-month sources. **Slice to a calendar-month window and take the
first hit** instead. The `_select_month` helper in
`notebooks/inspect_consolidated_soil_moisture.ipynb` is the canonical pattern.

### 3. GLDAS-2.1 `_acc` variables are means, not sums

Monthly products store the *mean* of 3-hourly accumulations. Multiply by
`8 × days_in_month` to recover monthly totals. Documented in NASA GES DISC
GLDAS-2.1 README. The runoff target builder previously had this as an identity
transform, making GLDAS contributions ~248× too small and the multi-source
min/max degenerate.

### 4. earthaccess sometimes writes 0-byte files

NASA Earthdata's server occasionally returns truncated streams that earthaccess
writes as 0-byte files without raising. Detect and remove before any subset/open
step. `_drop_zero_byte_downloads` in `src/nhf_spatial_targets/fetch/modis.py`
does this for both `fetch_mod16a2` and `fetch_mod10c1`.

### 5. MERRA-2 GWETTOP is plant-available wetness, not VWC

`(W − W_wilt) / (W_sat − W_wilt)`. Don't try to convert to mm depth via the
layer thickness — the conceptual definition is fundamentally different from
volumetric water content. The SOM target's per-source 0–1 normalisation handles
this gracefully; absolute-magnitude comparisons do not.

### 6. NCEP/NCAR longitudes are 0–360

Wrap to −180–180 before any CONUS clip. Helper in the SOM notebook.

### 7. Natural CONUS reference footprints

- ERA5-Land's CONUS+buffer (lon −125 to −66, lat 24.7 to 53) — use for the
  runoff and recharge inspections.
- NLDAS-NOAH's CONUS (lon −124.94 to −67.06, lat 25.06 to 52.94) — use for soil
  moisture (since NLDAS is CONUS-only at 0.125° anyway).

For HRU aggregation, gdptools handles spatial alignment per-HRU regardless of
source extent, so footprint clipping is mostly a notebook-side concern for
shared colour scales.

### 8. VSCode notebook renderer drops cells with empty-leading-column tables

`| | A | B |`-style markdown tables silently disappear from VSCode's notebook
view. Use bullet lists instead. Tracked via the project memory.

### 9. Read units from the catalog, not from the on-disk NetCDF

The catalog (`catalog/sources.yml`) is the single source of truth for source
units in this repo. When the catalog is corrected (Reitz inches/year → m/year,
NCEP/NCAR kg/m2 → m3/m3, etc.), already-consolidated and already-aggregated
NetCDFs on disk carry stale `cf_units` strings until they are re-fetched / re-
aggregated. **Target builders, aggregators, and downstream code should read
units from `nhf_spatial_targets.catalog`, not from the NetCDF `attrs["units"]`
they happen to be reading.** Doing otherwise produces silent drift between
what the catalog says and what the data conversion actually applies, with a
window of correctness that depends on the order in which sources were last
re-fetched.

Concretely, prefer:

```python
import nhf_spatial_targets.catalog as cat
units = cat.source(source_key)["variables"][var_idx]["units"]
```

over:

```python
units = ds[var].attrs["units"]
```

This also keeps the unit-conversion logic testable against the catalog
without requiring a real on-disk dataset.

### 10. Don't trust catalog units; trust the math

Every catalog correction in this PR was caught by a notebook plotting absolute
values and a cross-check against published means. Even though lesson 9 says
"read units from the catalog," the catalog itself can be wrong (and was, more
than once). Before opening a target-builder PR, plot the per-HRU result for
one well-known calibration year and confirm:

- CONUS-wide mean is within 30% of the published value for that source.
- Spot values at known-high and known-low locations are physically plausible.
- Source values are within an order of magnitude of each other after unit
  normalisation (any larger gap is a smoking gun for a missed conversion).
