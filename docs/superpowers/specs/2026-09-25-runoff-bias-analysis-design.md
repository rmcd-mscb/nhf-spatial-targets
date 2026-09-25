# Runoff target bias analysis against Oregon gages — design

Issue: #359. Status: approved design, pre-implementation.
Implementation lives in a **separate repo**, `nhf-runoff-bias`; this spec is kept
here because the inputs, the eventual correction step, and the audit trail are
this pipeline's.

## 1. Problem and intent

Colleagues calibrating pywatershed on a small Oregon sub-fabric accumulated the
runoff target's HRU values to a gage and found that **all three members**
(ERA5-Land `ro`, GLDAS-2.1 NOAH `Qs_acc + Qsb_acc`, MWBM ClimGrid `runoff`)
were below observed monthly flow, so the min/max envelope misses the gage.

Intended outcome: determine whether the member bias against gages is
**systematic and explainable by basin characteristics**, and if so produce a
**per-HRU multiplicative correction factor**, predicted from HRU covariates,
that can later be applied to all three members and the bounds (deliverable form
chosen 2026-09-25; widening-only and report-only were the alternatives).

Success criteria:

- A bias estimate per gage per member on the full Oregon fabric, with the
  screening tier of each gage recorded.
- A fitted, cross-validated relationship between bias and covariates, with
  attribution (which covariates matter, in which direction).
- A per-HRU factor table for the Oregon fabric with its uncertainty, or a
  clear negative result that the bias is not explainable at HRU scale.

Not in scope: applying the factor inside `nhf-spatial-targets`; CONUS /
GAGES-II; canopy and ecoregion covariates (second round); any timing
correction.

### What the literature says (see `docs/references/runoff-bias-literature.md`)

- No published evaluation of these three products against Oregon gages
  exists; the project fills a real gap.
- The one PNW large-sample study (Safeeq et al. 2014) finds the **sign of
  LSM bias flips with geology**: under-prediction in groundwater-dominated
  High Cascades basins, over-prediction in runoff-dominated ones. A small
  sub-fabric can sit entirely on one side of that split.
- None of the three products carries a deep aquifer store, so the expected
  monthly shape in the High Cascades is low Jul–Oct and high Dec–Mar with an
  annual total closer to unbiased, unless crest precipitation is under-caught
  (most plausible for 0.25° GLDAS and station-based nClimGrid).
- Regulation and irrigation push the bias the other way at non-reference
  gages, so reference screening is first-order.
- Covariates with support, in order: geology / permeability or baseflow
  index; snow fraction and melt timing; precipitation, runoff ratio, aridity;
  elevation; regulation status. Forest cover has no support as a predictor.

## 2. Framing decisions (fixed)

1. **Bias is measured on long-term mean volume**, as the log ratio of
   observed to accumulated mean flow over the overlap years. Twelve
   climatological monthly ratios are computed alongside to characterise the
   seasonal shape. Timing error is diagnosed, not corrected: a multiplicative
   factor cannot fix a missing aquifer store.
2. **Per member, not per envelope.** Each member gets its own bias analysis.
   Whether one shared factor suffices is a finding.
3. **Approach A then C** (section 5). A is the fast, interpretable baseline;
   C produces the HRU-scale factor directly. Nested-basin differencing (B) is
   diagnostic only.
4. **Separate repo** consuming this pipeline's outputs as data; the only
   thing that later returns here is the factor-application step.

## 3. Inputs and their known defects

| Input | Path (Oregon) | Notes |
|---|---|---|
| Runoff target | `or-spatial-targets/targets/runoff_targets.nc` | CF-1.8, members `era5_land`, `gldas_noah_v21_monthly`, `mwbm_climgrid` in cfs per HRU, monthly 1979–2024, dim `hru_id` (16 814). Carries `fabric_sha256`. |
| Fabric | `or-spatial-targets/fabric/model_layers_9.gpkg` | layers `nhru` (`hru_id`, `nhm_id`, `hru_segment`, `areasqkm`), `nsegment` (`segment_id`, `to_segment`), `npoigages` (`poi_gage_id`, `segment_id`; 851 rows). `drainage_area` is NaN on every row. |
| Gage daily flow | `nhf-spatial-targets/gage_data/sf_efc.nc` | 944 POIs × daily 1979-01-01..2022-12-31, `discharge` in cfs. 846 of 851 fabric POIs present. Not committed (380 MB; `gage_data/` gitignored). |
| Flow management | `nhf-spatial-targets/gage_data/TableA2_FlowManagementIndex.csv` | 631 gages, `storage_index`, `use_index`, `flow_management_index` 0–3, `area_mi2`, `comid`. Source report to be recorded (open question §9). |
| Terrain covariates | `gfv2-params` outputs (`hru_elev`, `hru_slope`, `hru_aspect`, impervious, soils) | keyed on `nhm_id`, which the Oregon `nhru` layer carries. |
| Climate covariates | `nhf-datastore/mwbm_climgrid/ClimGrid_WBM.nc` + cached weights `or-spatial-targets/weights/mwbm_climgrid_batch*.csv` | monthly `prcp`, `pet`, `snow`, `tmean` on the ~5 km nClimGrid, 1895–2020. The Oregon Daymet aggregation holds only `swe`, so ClimGrid is the climate source; climatology window **1980–2020**. |
| Geology | to download: Oregon Geologic Data Compilation, or USGS SGMC, or GLHYMPS permeability | one zonal overlay per HRU. |

Defects in the gage NC that the loader must handle (measured 2026-09-25):

- 214 of 944 POIs have zero valid days; 503 have ≥ 10 years; 451 of those are
  fabric POIs; 430 of those have a flow management index; **49 are index 0,
  181 are index ≤ 1**. The clean sample is ~50–180 gages.
- 50 IDs are non-numeric: 24 suffixed derived series (`13233300-VALO` …),
  24 Washington Ecology IDs (`32A080` …), and two placeholder rows `GAGES`
  / `gages` with no data. 27 series contain negative discharge (down to
  −2 361 cfs): reservoir-outflow / diversion accounting, not measurements.
- Records end 2022-12-31; target runs to 2024. Overlap 1979–2022.
- `efc == -1` on ~54 000 days with finite discharge (flag only, ignored).
- `discharge` is uncompressed and contiguous; `agency_id` is a 121 MB
  per-day string variable. Not a blocker; the sub-project derives a monthly
  product and never re-reads the daily file downstream.
- Median usable gage has ~140 complete months.

## 4. Repo and data contract

- Repo `nhf-runoff-bias`, pixi-managed, src layout, package
  `nhf_runoff_bias`, Cyclopts CLI `nhf-runoff-bias`, ruff + pre-commit as
  here. Python ≥ 3.11. Dependencies: xarray, netCDF4, pandas, geopandas,
  pyarrow, networkx, scikit-learn, statsmodels, scipy, rioxarray (zonal),
  `dataretrieval` (NWIS site service for drainage area), matplotlib.
- `config.yml` names the three inputs by absolute path plus a run directory
  and a `datastore` for downloaded covariates. Nothing is written into the
  targets project.
- Every derived artifact (NetCDF or Parquet) carries a `source_sha256`
  attr/metadata of its inputs so a rebuilt target visibly invalidates
  downstream products. Stages skip when output is newer than inputs and the
  hash matches.

## 5. Components (one module each)

- **`gages.py`** — reads daily NC + CSV. Exclusions: non-numeric IDs other
  than the Ecology pattern, placeholder rows, any series with negative
  discharge. Monthly mean cfs requires ≥ 28 valid days in the month (all
  days for February). Drainage area from the CSV `area_mi2`, then NWIS site
  service, then NaN. Emits `gages_monthly.parquet` (gage, time, q_cfs,
  n_valid_days) and `gage_meta.parquet` (name, agency, lat, lon, area,
  storage/use/flow-management indices, exclusion_reason). **This is the seam
  a GAGES-II / NWIS reader replaces for CONUS.**
- **`network.py`** — directed segment graph from `nsegment`
  (`segment_id → to_segment`), HRUs attached via `nhru.hru_segment`. For each
  POI: upstream HRU set, fabric area (km²), `touches_boundary` (upstream set
  contains a segment with no upstream and `to_segment` links leaving the
  domain, or an inlet), `area_ratio` = fabric area / published area. Emits
  `gage_hrus.parquet` (gage, hru_id) and `gage_network.parquet`.
- **`accumulate.py`** — for each gage and member, sum of the member over the
  HRU set per month (cfs is already a flow, so a straight sum). Emits
  `accumulated.nc` with dims `(gage, member, time)`; NaN-aware, records the
  count of NaN HRUs per gage-month.
- **`covariates.py`** — one builder per group, each emitting a per-HRU
  Parquet on `hru_id`: `terrain` (join gfv2-params on `nhm_id`), `climate`
  (ClimGrid via the cached gdptools weights: annual P and PET, aridity
  PET/P, snow fraction Σsnow/Σprcp from ClimGrid's own partition, mean T), `geology` (majority
  class and area-weighted log-permeability), `derived` (runoff ratio uses
  gage Q/P, so it is basin-scale only). Basin-mean versions are
  area-weighted through `gage_hrus.parquet`. Emits `covariates_hru.parquet`,
  `covariates_basin.parquet`.
- **`bias.py`** — per gage × member over the overlap years present in both
  records: `log_ratio = ln(mean(Q_obs) / mean(Q_acc))`; twelve climatological
  monthly ratios; baseflow index from the daily record (Lyne–Hollick or
  Eckhardt filter); the count of overlap years. Emits `bias.parquet`.
- **`models.py`** —
  *Approach A*: standardise basin covariates, PCA (report loadings and
  variance explained), then (i) OLS on retained components and (ii) random
  forest on raw covariates, both with leave-one-gage-out CV, with Shapley
  attribution for the forest. Predict per-HRU factor `exp(f(x_hru))` from
  the HRU covariates.
  *Approach C*: fit `Q_obs_g = Σ_h A[g,h] · exp(β·x_h) · m_h` for the
  coefficient vector β by nonlinear least squares in log space, with ridge
  regularisation and the PCA basis from A; flag-gated so the first report
  does not require it. Emits `fit.json` (coefficients, CV skill, n gages per
  tier and member) and `factor_hru.parquet` (gage-tier × member × hru_id
  factor with a CV-derived uncertainty).
- **`report/`** — notebooks reading the artifacts: bias map by member,
  bias vs each covariate, seasonal ratio shape by geology class, CV skill,
  predicted factor map per HRU, members-vs-gage hydrographs for a handful
  of representative gages.

## 6. Data flow and CLI

```
nhf-runoff-bias gages      # daily NC + CSV      -> gages_monthly.parquet, gage_meta.parquet
nhf-runoff-bias network    # fabric              -> gage_hrus.parquet, gage_network.parquet
nhf-runoff-bias covariates # gfv2-params, Daymet, geology -> covariates_hru/basin.parquet
nhf-runoff-bias bias       # target + gages + network -> accumulated.nc, bias.parquet
nhf-runoff-bias fit        # covariates + bias   -> fit.json, factor_hru.parquet
```

Gage selection tiers are data, not code:

| Tier | Flow mgmt index | Overlap years | Area check | Boundary | Role |
|---|---|---|---|---|---|
| A | ≤ 1 | ≥ 10 | within 10 % | inside fabric | **primary** — the same screen the pywatershed calibration uses (~180 gages) |
| B | 0 | ≥ 10 | within 10 % | inside fabric | strict sensitivity check (~50 gages) |

Every fit reports both tiers; the factor table and the headline figures come
from tier A, and tier B is shown alongside to confirm the relationship is not
driven by lightly managed basins. A gage failing a check stays in every artifact
with `tier = none` and `exclusion_reason`.

## 7. Quality gates and error handling

- `network` fails loudly on a cycle, an unknown `to_segment`, or an HRU whose
  `hru_segment` is not in `nsegment`.
- `accumulate` refuses to run if the target NC's `fabric_sha256` does not
  match the configured geopackage-derived parquet hash recorded in
  `or-spatial-targets/fabric.json`.
- Covariate joins assert full `hru_id` coverage; a listed exception count is
  logged and stored, never silent NaN.
- `fit.json` records n gages actually used per tier and member.
- Area check uses the CSV or NWIS area; a gage with no published area is
  `tier = none` with reason `no_published_area`.
- **Fabric portability.** The gate above binds an analysis run to one
  fabric, deliberately. When this scales to GFv2 or CONUS the fabric, its
  feature id column, and the POI layer schema will all differ, so the id
  column names (`hru_id`, `nhm_id`, `segment_id`, `to_segment`,
  `hru_segment`, `poi_gage_id`) and the layer names are **config values with
  Oregon defaults**, never literals in `network.py` / `accumulate.py`. Every
  artifact records the fabric path, its sha256, and the id column it is
  keyed on, so an Oregon `factor_hru.parquet` cannot be joined to a GFv2
  fabric by accident.

## 8. Testing

- Synthetic fixture: five HRUs, three segments, two nested gages. Tests:
  traversal incl. nesting; boundary flag; area check; accumulation against
  hand-computed sums; monthly completeness rule; every exclusion rule;
  recovery of a known synthetic bias by A and by C; PCA basis round-trip.
- Integration-marked tests on the real Oregon inputs (skipped in CI).
- Same ruff / pre-commit / CI layout as `nhf-spatial-targets`.

## 9. Open questions carried into implementation

- Provenance of `TableA2_FlowManagementIndex.csv` (which report) for
  citation; meaning of `oregon = N` rows (17 gages). Query is out with the
  table's author (2026-09-25); record the citation in `gage_meta` metadata
  once known.

Resolved 2026-09-25:

- **Geology: SGMC lithology majority class + GLHYMPS log-permeability**, both
  CONUS-consistent so they scale without a product swap.
- **Climate source is ClimGrid, not Daymet.** Daymet on disk is 1980–2025
  but the Oregon aggregation carries only `swe`; the MWBM ClimGrid source
  file already holds `prcp`/`pet`/`snow`/`tmean` and the project has cached
  weights for that grid. Climatology window **1980–2020**; the bias itself
  still uses every overlap year present in both records (1979–2022).
