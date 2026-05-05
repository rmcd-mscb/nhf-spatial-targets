# Lessons Learned

Findings and operational notes accumulated while building the
`nhf-spatial-targets` pipeline that don't fit cleanly into the per-target
recipes (`calibration-target-recipes.md`) or the per-source catalog
(`catalog/sources.yml`). Each entry documents *what we found* and *where
the consequence lives in the code* so a future maintainer can re-evaluate
the finding when the underlying data product changes.

---

## MOD16A2 v061 flat-on-CONUS+

**Status:** resolved — two interacting bugs, both fixed in PR #88:
(1) fill-value contamination at the consolidation reprojection step
(`fetch/consolidate.py`), and (2) double-scaling in the inspection
notebooks' monthly-resample helpers (`notebooks/aggregated/`,
`notebooks/consolidated/`). Re-consolidating + re-aggregating + re-running
the notebooks restores realistic seasonality and absolute magnitudes.

### What we found

The July 2000 cross-check in
`notebooks/aggregated/inspect_aggregated_aet.ipynb` (commit
`cce4ed6`, finding cell `55a87e3a`) compared the three AET sources at
calendar-month cadence on a CONUS+ subset:

| Source | Jan 2000 (mm/month) | Jul 2000 (mm/month) | Jul/Jan |
|---|---|---|---|
| SSEBop | 9.0 | 101.1 | **11.2×** |
| MWBM (ClimGrid) | 12.5 | 77.8 | **6.2×** |
| MOD16A2 v061 (pre-fix) | 33.3 | 37.4 | **1.12×** |

MOD16A2 v061 read as essentially flat across the seasonal cycle while the
other two products swung 6–11× as expected for CONUS ET. The flatness was
present at both the gridded level (Jan 65.5 → Jul 70.0 = 1.07×) and the
HRU-aggregated level, but it is not a property of the consolidated NC's
raw values — it is an artefact of the consolidate-time reprojection.

### Root cause

`_mosaic_and_reproject_timestep` reprojects from the native 500 m
sinusoidal tiles to a 0.04° (≈ 4 km) lat/lon grid using
`Resampling.average`. MOD16A2 v061 reserves raw integer values 32761–32767
for special codes (water=32761, barren=32762, snow/ice=32763, cloudy=32764,
no-data=32766, not-processed=32767). `rioxarray.open_rasterio(masked=True)`
maps **only** the declared `_FillValue` (typically the not-processed code)
to NaN; the other six special codes remain as ordinary numeric pixels.

When `Resampling.average` mixed valid pixels (raw ≈ 0–500) with one of
those un-masked fill codes (raw ≈ 32766), the resulting 4 km cell carried
an arithmetic average like `(31 × 100 + 1 × 32766) / 32 ≈ 1124` raw
(≈ 112 mm/8 day scaled). Such cells appeared along every coastline,
lake margin, and snow boundary — wherever a valid 4 km cell touched a
fill-coded one. They were physically impossible (10× peak summer ET in
January) but fell well below the `<= 3270` post-reprojection mask in
`aggregate/mod16a2.py:_mask_et_fill`, so they survived into HRU
aggregation. With ~10 % of CONUS+ cells contaminated this way and an
average contaminated value of ~1003 mm/8 day, every coastline/lake-margin
HRU was pulled toward the contaminated mean. That eliminates real
spatial variation and flattens the seasonal cycle.

Quantifying with `mod16a2_v061_2010_consolidated.nc` (Jan 1 vs Jul 4
2010, gridded means under different masks):

| Mask threshold | Jan mean | Jul mean | Jul/Jan |
|---|---|---|---|
| `<= 3270` (pre-fix) | 168.1 | 184.1 | **1.10×** |
| `<= 200` | 10.7 | 26.3 | 2.47× |
| `<= 100` | 6.3 | 21.9 | **3.47×** |

A tighter threshold drops the contaminated cells and recovers a
believable seasonal swing.

### The fix

**Pipeline (`fetch/consolidate.py`).** `_mosaic_and_reproject_timestep`
now masks ET_500m values above the valid-data ceiling (raw 32700 / scaled
3270) **before** `Resampling.average` runs, so the area-weighted mean
only sees valid pixels. Boundary cells become either real averages of
their valid neighbours or NaN — never an intermediate contaminated value.
Threshold detection is automatic so the helper works whether rioxarray
returns scaled or raw values (`_mask_modis_et_fills` in
`fetch/consolidate.py`). `aggregate/mod16a2.py:_mask_et_fill` is kept as
a belt-and-suspenders post-aggregate hook; on a freshly-consolidated NC
it is now a no-op.

**Notebooks.** While verifying the fill-mask fix we also discovered the
inspection notebooks were double-scaling MOD16A2 by a factor of 0.1.
The consolidated NC stores raw int-like values on-disk (max 32766) with
`scale_factor=0.1` in attrs; xarray's default `decode_cf=True` applies
the scale on read, so values arriving in the helpers are already in
scaled mm/8 day. The `_mod16a2_to_monthly_mm` helper in
`notebooks/aggregated/inspect_aggregated_aet.ipynb` then multiplied by
`0.1` again, producing figures whose values were 10× too low — visually
the MOD16A2 panel still looked plausible because the contaminated
boundary cells survived the double-scaling and saturated the colourbar
at the high end while the rest of the panel got squashed near zero.
The notebook helper now trusts `decode_cf=True` and refuses to run
against a raw-value DataArray (sanity-check `assert max <= 3300`).
The consolidated notebook's `_mod16a2_monthly_mm` was also missing the
`<= 3270` fill mask (the misleading "Note on scale factor" cell
documented this as "scale_factor not applied" — that was a misdiagnosis).
Both helpers are corrected in PR #88.

### Operational impact

- All consolidated MOD16A2 NCs produced before PR #88 carry the
  contamination. Re-consolidating from the HDF tiles is cheap CPU-wise
  but requires the HDFs (which are deleted after consolidation in normal
  pipeline runs) — for an existing project, a full re-fetch + re-consolidate
  is the cleanest path.
- Re-aggregation is required after re-consolidation. The aggregated
  outputs at `<project>/data/aggregated/mod16a2_v061/` become invalid
  on PR-#88 merge; downstream consumers should re-aggregate before
  rebuilding AET targets.

### How to re-run after PR #88

The fetch command's manifest-based year-skip means a plain re-run does
nothing once a year is recorded. PR #88 adds a `--force` flag to
`nhf-targets fetch mod16a2` (and `mod10c1`) for exactly this case:

```bash
# 1. Re-fetch (overwrites consolidated NCs and the mod16a2_v061
#    manifest entry on completion)
pixi run nhf-targets fetch mod16a2 --project-dir <project> -p YYYY/YYYY --force

# 2. Drop the aggregated NCs so the per-year `out_path.exists()`
#    skip in aggregate/_driver.py doesn't keep the stale outputs
rm <project>/data/aggregated/mod16a2_v061/*_agg.nc

# 3. Re-aggregate
pixi run nhf-targets agg mod16a2 --project-dir <project>
```

The weights cache at `<project>/weights/mod16a2_v061_batch*.csv` does
*not* need to be cleared — its fingerprint is on the fabric batch
geometry, not on source values, so it stays valid across this fix.

### Where to look in the code

- Mask helper + reprojection: `src/nhf_spatial_targets/fetch/consolidate.py`
  (`_mask_modis_et_fills`, `_mosaic_and_reproject_timestep`).
- Post-aggregate safety net: `src/nhf_spatial_targets/aggregate/mod16a2.py`
  (`_mask_et_fill`).
- Regression test: `tests/test_consolidate_modis.py`
  (`test_mask_modis_et_fills_*`,
  `test_mosaic_and_reproject_timestep_no_fill_contamination`).
- Inspection notebooks (both updated by PR #88; re-run after re-aggregation):
  `notebooks/aggregated/inspect_aggregated_aet.ipynb` (`_mod16a2_to_monthly_mm`
  no longer applies `× 0.1`; `× scale_factor` was double-scaling on top of
  xarray's `decode_cf`) and `notebooks/consolidated/inspect_consolidated_aet.ipynb`
  (`_mod16a2_monthly_mm` now masks special codes `> 3270` before the
  monthly sum; the misleading "Note on scale factor" cell is corrected).
- Catalog entry: `catalog/sources.yml` → `mod16a2_v061` (`notes:` block,
  updated alongside this fix).

---

## `mean` vs `masked_mean`, and where NN-fill belongs

**Status:** convention revised. Aggregation now uses `masked_mean` for
sources with explicit per-pixel masking (MOD16A2 fill mask, MOD10C1 CI
gate); `mean` everywhere else. NN-fill, when applied at all, lives at
the *target* stage, not the aggregation stage.

### Where the original convention came from

An earlier feedback note said: "Aggregation uses gdptools
`stat_method='mean'` — never `masked_mean`. Plain mean produces NaN for
partial-coverage HRUs by design; nearest-neighbour-fill those NaN
HRUs as a separate post-processing step in `normalize/methods.py`,
called by each target builder before normalisation/combination."

That was the right rule when no source had per-pixel masking: NaN at
the pixel level meant "no source data here" (geometric partial
coverage), and propagating it to the HRU was honest. The plan was to
fix coverage gaps later via NN-fill, with the imputation step
explicitly visible in `normalize/`.

### What the PR #88 work surfaced

PR #88 added a per-pixel fill mask for MOD16A2 (water/barren/snow/cloud/
no-data codes → NaN before reprojection). MOD10C1 already had similar
per-pixel masking (CI > 70 gate, plus flag-value masks). With
`stat_method="mean"` those pixel-level NaNs poisoned every HRU that
touched even one masked pixel — *most* coastal / lake-margin / snow-
boundary HRUs. The aggregated NCs went from "biased values from
unmasked fills" (the original bug) to "honest but unhelpfully sparse
NaN coverage" (the symptom that surfaced post-fix when the user noticed
MOD16A2 missing in time-series plots for every representative HRU
except Southern Appalachians).

The mechanical question is: when a per-pixel mask sets a pixel to NaN
*on purpose*, should the HRU mean propagate that NaN or skip it?

- Propagating it (the original convention's behaviour with `mean`):
  treats the deliberate mask the same as a true-upstream NaN. Defeats
  the per-pixel mask the moment any masked pixel touches the HRU.
- Skipping it (the new behaviour with `masked_mean`): the HRU value is
  the area-weighted mean of pixels that survived the gate. NaN at the
  HRU level still happens, but only when *every* contributing pixel was
  masked — which is the genuine "no useful source data" state.

For a source with a `pre_aggregate_hook` that sets pixels to NaN,
`masked_mean` is the only way to make the per-pixel mask do its job at
the HRU level. The original convention was right for sources without
per-pixel masking; it was wrong for sources with it.

### The revised convention

- `SourceAdapter.stat_method` (default `"mean"`) controls the
  per-source choice. Sources whose `pre_aggregate_hook` sets pixels to
  NaN must override to `"masked_mean"`; sources without such a hook
  stay on `"mean"` so geometric / true-upstream NaN still propagates.
- `aggregate/mod16a2.py` and `aggregate/mod10c1.py` are the only
  current `masked_mean` users. When a future source adds a NaN-emitting
  pre-hook, it should set `stat_method="masked_mean"` at the same time.
- The aggregated NCs are now the canonical "what each source actually
  covers" record. They preserve NaN HRUs honestly — no imputation.

### NN-fill moves to the target stage

The original plan was for `normalize/methods.py` to NN-fill aggregated
NaN HRUs before the target builder ran. That plan is revised: NN-fill,
*if applied at all*, runs at the target stage on the per-HRU per-time
*bound*, not on the aggregated NCs. The reasoning:

1. **Multi-source min/max bounds are NaN-aware.** With three sources,
   the per-HRU per-month bound (`np.fmin`/`np.fmax`) is well-defined
   whenever ≥1 source is finite. A NaN HRU in MOD16A2 doesn't break the
   AET target's bound — it just means MOD16A2 doesn't contribute that
   month. NN-filling MOD16A2 first would silently introduce an imputed
   value where the source genuinely has nothing to say.
2. **Aggregated NCs as honest record.** Different consumers (calibration
   target, diagnostic notebook, future analyses) may want different
   imputation policies. Preserving NaN at the aggregated layer keeps
   the record canonical and lets each consumer decide.
3. **Target-time NN-fill is the right scale.** If a target needs a
   complete bound at every HRU, NN-fill the *bound* after multi-source
   reduction — that's a single per-HRU per-time imputation step,
   transparent in the target builder and easy to audit.

### Where to look in the code

- `src/nhf_spatial_targets/aggregate/_adapter.py` — `SourceAdapter.stat_method`
  field and validation.
- `src/nhf_spatial_targets/aggregate/_driver.py` — `aggregate_variables_for_batch`
  forwards `adapter.stat_method` to gdptools `AggGen`.
- `src/nhf_spatial_targets/aggregate/mod16a2.py`, `aggregate/mod10c1.py`
  — `stat_method="masked_mean"` set on the adapter, with a comment
  explaining the link to the per-pixel mask in the pre-hook.
- `CLAUDE.md` § "Aggregation Transformation Policy" — concise rule for
  AI assistants.
- `docs/architecture/transformation-pipeline.md` — full architectural
  rule + decision flow.

---

## WaterGAP gridded vs aggregated bbox mismatch

**Status:** documented — no fix needed.

The recharge inspection notebook's validation cell reports a ~41% gap
between WaterGAP's HRU-aggregated mean and its native-grid mean for
2000. This is much larger than the 5–30% gap we treat as "normal" for
gridded vs aggregated cross-checks, but it is *intrinsic to a global
0.5° grid clipped to a CONUS+ window*:

- The unweighted gridded mean still includes a wide non-CONUS buffer.
- After area-weighting to fabric HRUs, the dry interior US (low recharge)
  is over-represented and the value drops well below the gridded number.

A gap above 30% is not automatically a bug for global-grid sources. The
check that *does* matter for absolute-magnitude comparability is the
per-source order-of-magnitude sanity (Reitz CONUS-mean ≈ 123 mm/year vs
the published 162 mm/year; WaterGAP ≈ 61 mm/year plausible for a
diffuse-only flux; ERA5-Land `ssro` ≈ 100 mm/year plausible as a recharge
proxy).

The recharge target normalises each source 0–1 over the calibration
window before computing per-HRU `min/max`, so absolute-magnitude
divergence between the three is *not* a problem for the target — the
optimisation targets relative year-to-year change.

---

## MOD16A2 flag-mask convention

**Status:** stable convention; documented for cross-pipeline consistency.

MOD16A2 v061 ET_500m stores int16-packed values 0–32700 (decoded 0–3270
mm/8-day after `scale_factor=0.1`), with values **above 32700** reserved
for special codes:

| Raw value | Meaning |
|---|---|
| 32761 | water |
| 32762 | barren |
| 32763 | snow / ice |
| 32764 | cloudy |
| 32765 | (reserved) |
| 32766 | no-data |
| 32767 | not-processed |

After `decode_cf=True` these become ~3276.x — not physical ET. The
aggregation pipeline drops them via:

```python
ds["ET_500m"] = ds["ET_500m"].where(ds["ET_500m"] <= 3270.0)
```

(see `src/nhf_spatial_targets/aggregate/mod16a2.py:_mask_et_fill`).

The same threshold is applied in the inspection notebook's gridded
helper (`_mod16a2_target_month_gridded_mm` in
`notebooks/aggregated/inspect_aggregated_aet.ipynb`); without
it, ~37% of CONUS+ pixels in a typical January composite carry flag
codes ≈ 3276 mm and the gridded mean reads ~10× too high. The mask is
the same threshold in both paths so the two columns of the AET
validation table compare like-for-like.

---

## MWBM is gridded, not polygon

**Status:** corrected.

Earlier wording in `calibration-target-recipes.md` §2 implied MWBM was
distributed as a 1 km polygon mesh. It is in fact a regular gridded
NetCDF: CONUS at ~0.042° (2.5 arcminute) on a lat/lon grid, in
`<datastore>/mwbm_climgrid/ClimGrid_WBM.nc`, covering 1895–2020
monthly. The recipe and inspection-notebook code now reflect this.

---

## Validation-cell limits

**Status:** kept as a smoke test, not a quantitative validator.

The `_gridded_mean_*` validation cells in the inspection notebooks
compare the HRU-area-weighted fabric mean against an unweighted mean
over the source's consolidated-NC bbox. Two systematic biases mean
this is *not* like-for-like:

1. **Bbox mismatch.** The consolidated bbox typically extends past the
   fabric (CONUS+, global clip, etc.); the unweighted mean includes
   non-fabric land that has different climate/recharge/ET than the
   fabric mean.
2. **Weighting mismatch.** The aggregated mean is Albers-area-weighted
   over fabric HRUs; the gridded mean is unweighted.

For CONUS-bounded sources whose bbox closely tracks the fabric (SSEBop,
MWBM) the two columns agree to within a few percent — that's a
meaningful correctness check. For global-grid sources clipped to a
CONUS+ window (WaterGAP, MOD16A2 v061) the gap is dominated by the
geometry mismatch and is not informative as a correctness check.

A more diagnostic future cross-check would be a **per-HRU residual
map** comparing the same aggregation run two ways (e.g.
`stat_method="mean"` vs `stat_method="masked_mean"`, or two different
weight CRSs). That would isolate aggregation-pipeline issues from
geometry differences. The current validation cells were sufficient to
catch the bugs we've actually had (MOD16A2 flag mask, recharge path
errors), so the upgrade is a "next time" item, not an urgent fix.

---

## Per-source documentation gap

**Status:** open follow-up.

`docs/sources/` currently contains only `mwbm_climgrid.md`. The other
nine source modules (`era5_land`, `gldas_noah`, `merra2`, `mod10c1_v061`,
`mod16a2_v061`, `ncep_ncar`, `nldas_mosaic`, `nldas_noah`, `reitz2017`,
`watergap22d`) have catalog entries with `notes:` blocks but no
standalone `docs/sources/<source>.md`. This is fine for current
contributors — the catalog is authoritative — but a new operator
onboarding to the project will benefit from per-source quick-start
docs that include manual-download procedures, expected disk size,
license / credentials needed, and known consolidation gotchas.

The `mwbm_climgrid.md` template is the model: title / description,
manual-download steps, fingerprinting / placement, expected file
layout in the datastore, and any gotchas. Track this work in a future
issue rather than blocking on it for the current PR.
