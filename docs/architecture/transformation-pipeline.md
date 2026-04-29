# Transformation pipeline & aggregation policy

This document is the architectural reference for **where** different kinds of
data transformations should live in the pipeline. It exists because several
sources need flag masking, scale factors, quality filtering, unit conversion,
and per-source normalization, and naive choices about ordering can silently
change the answer.

If you are adding a new fetch/aggregate/normalize/target module, this is the
file that tells you which stage your transformation belongs in.

## TL;DR

| Stage | What it does | What it must NOT do |
|---|---|---|
| `fetch/<src>.py` | Download raw files; consolidate per-year NCs from upstream tiles/granules; reproject (e.g. MODIS sinusoidal → WGS84) when the upstream format isn't directly usable. | Apply scale factors, mask flag values, derive new variables, or filter by quality. The datastore should mirror the source's native semantics. |
| `aggregate/<src>.py` `pre_aggregate_hook` | Apply *pixel-defined* operations: flag-value masks, sums-of-accumulations needed to produce a single variable to aggregate, and quality gates that determine which pixels participate in the area-weighted mean. | Apply linear scale factors. Rename source variables. Compute multi-source combinations. Do anything that could equally well live downstream. |
| `aggregate/<src>.py` `post_aggregate_hook` | Cosmetic-only: attach attrs, rename auxiliary diagnostic variables (e.g. `valid_mask` → `valid_area_fraction` after the per-pixel 0/1 mask becomes an HRU-fraction). | Modify aggregated source values. |
| `normalize/methods.py` | Per-source per-HRU operations whose definition is *at HRU scale*: 0–1 normalization, multi-source min/max bounds, NN-fill of NaN HRUs from partial source coverage. | Mask flags. Reach back to pixel-level data. |
| `targets/<tgt>.py` | Linear unit conversions (`× 1000`, `÷ 100`, `× 8 × days_in_month`), CF-compliant target NetCDF assembly, calling into `normalize/` for combination. | Anything that can't be expressed at HRU scale. |
| `notebooks/inspect_aggregated/*` | Diagnostic. Re-implements the same conversions as `targets/` to visually verify the aggregated NCs. | Define new transformations not used in `targets/`. |

The aggregated NC at `<project>/data/aggregated/<source_key>/...` therefore
carries the source's **native variable names** and **native units**, with
flag-masked and quality-gated values. Downstream consumers know what to
expect.

## The principle

> **Aggregation is a one-way information bottleneck. Apply each transform at
> the spatial scale where it is defined.** Pixel-defined operations
> (per-pixel flag masks, per-pixel quality gates) must run pre-aggregation.
> HRU-defined operations (per-HRU time-series normalization, multi-source
> min/max combination) must run post-aggregation. Linear operations commute
> with area-weighted mean and may run on either side, so we put them
> downstream by convention.

Why a "principle" and not just a style choice: for non-linear and threshold-
based operations, doing them on the wrong side of aggregation gives a
different numerical answer.

## Why ordering matters: the math

The aggregator here is gdptools' area-weighted mean over HRU polygons:

$$\bar{x}_H = \frac{\sum_i a_i\, x_i}{\sum_i a_i}$$

where $a_i$ is the source-grid area of pixel $i$ inside HRU $H$ and $x_i$ is
the pixel value. This operation is **linear in $x$**. That single property
determines whether each kind of transform commutes with it.

### Linear scale (`× c`, `÷ c`)

$$\overline{c\,x_i} = c\,\bar{x}_H$$

Scale factors commute. `mean(x / 100)` and `mean(x) / 100` are exactly
equal. Order is purely an architectural choice. **We put linear scales
downstream**, in `targets/` (and mirror them in inspect notebooks), because
that's where the other unit conversions live and because keeping the
aggregated NC in native units makes it easier to spot a missed conversion
factor (catalog has been wrong about `× 0.1`, `inches/year`, and `÷ 100` at
various points — see `docs/references/calibration-target-recipes.md` lessons
1, 9, 10).

### Sums of accumulations (`× 8 × days_in_month`)

These are linear too — multiplying by a per-time-step constant. They
commute. We put them in `targets/run.py` (`gldas_to_mm_per_month`) for the
same reason as above.

### Pixel-defined masking (flag codes, per-pixel quality gates)

These do **not** commute. The whole point of NaN-skipping in the area-
weighted mean is that masked pixels don't contribute. Masking after
aggregation is a different operation:

- **Pre-aggregation mask** (the right answer): set per-pixel
  $x_i = \mathrm{NaN}$ where the per-pixel quality fails, then
  $\bar{x}_H = \frac{\sum_{i \in V} a_i x_i}{\sum_{i \in V} a_i}$
  where $V$ is the set of valid pixels. The HRU mean is over only valid
  pixels.
- **Post-aggregation mask** (a different answer): area-weight all pixels
  including invalid ones, producing some HRU-mean of CI; then gate based on
  whether the HRU-mean CI exceeds the threshold. You've irreversibly mixed
  high- and low-confidence pixels, and you can't recover the
  high-confidence-only mean from the result.

Worked example: HRU with 50% high-CI snowy pixels (snow=80, ci=0.9) and 50%
low-CI cloud pixels (snow=20, ci=0.3):

- Pre-aggregation gating: only high-CI pixels survive, mean snow = 80,
  mean ci = 0.9. ✅ The half of the HRU we can see is snowy.
- Post-aggregation gating with threshold ci > 0.70: HRU-mean snow = 50,
  HRU-mean ci = 0.6 → masked NaN. ❌ We've thrown away the trustworthy
  half *and* the untrustworthy half.

The information loss is asymmetric: pre-aggregation gating preserves the
valid signal even when an HRU is partly contaminated; post-aggregation
gating either contaminates the mean or discards the whole HRU.

Concrete cases in this repo:

- `aggregate/mod16a2.py` — pre-hook masks fill values 32761–32767 (codes
  for water, barren, snow/ice, cloudy, no-data, not-processed) before the
  area-weighted mean.
- `aggregate/mod10c1.py` — pre-hook masks flag values >100 on both
  `Day_CMG_Snow_Cover` and `Day_CMG_Clear_Index`, and applies the per-pixel
  CI > 70 quality gate from TM 6-B10.

### Per-HRU time-series normalization

`(x − x_min) / (x_max − x_min)` is non-linear in $x$ (the parameters
themselves depend on the data). It does not commute with area-weighted
mean, and the HRU-scale and pixel-scale extremes can be very different.

For calibration targets defined at HRU scale, we want the normalization to
reflect HRU-scale extremes. So this lives in `normalize/methods.py`
(`normalize_0_1`, `normalize_by_calendar_month`) and operates on the
already-aggregated per-HRU time series.

### Multi-source min/max bounds

Different sources have different native grids. To take min/max across
sources you must first put both onto the common HRU fabric, which is
aggregation. So this is necessarily post-aggregation by definition. Lives
in `targets/run.py` (`multi_source_runoff_bounds`) and any future per-target
combiner.

## Diagnostic outputs are allowed

Strict reading of the principle would say "the aggregator must not derive
new variables." A useful exception: **aggregation metadata** — variables
that describe the aggregation itself rather than transforming source data
— may be emitted by the aggregator. The canonical example is
`valid_area_fraction` in MOD10C1: it's the area-weighted mean of a
per-pixel 0/1 quality indicator and tells downstream consumers what
fraction of the HRU was actually observed under the CI gate. It is
diagnostic, not a derived source variable, so it stays.

## Worked example: MOD10C1

Concretely, here's how MOD10C1's pipeline lays out under the policy:

```
fetch/mod10c1.py
    └─ download v061 HDF tiles via earthaccess; consolidate to per-year NC
       in <datastore>/mod10c1_v061/. No transformation of values.

aggregate/mod10c1.py     pre_aggregate_hook = build_masked_source
    ├─ Mask flag values (>100) on Day_CMG_Snow_Cover  → NaN
    ├─ Mask flag values (>100) on Day_CMG_Clear_Index → NaN
    ├─ Apply per-pixel CI > 70 gate to Day_CMG_Snow_Cover (NaN where fail)
    └─ Emit valid_mask = 1.0 where pixel passes, else 0.0
       Native 0–100 integer scale preserved on both source variables.

aggregate/mod10c1.py     post_aggregate_hook = _rename_valid_mask
    └─ Rename valid_mask → valid_area_fraction
       (after gdptools area-weighted mean, the per-pixel 0/1 indicator is
        a per-HRU fraction in [0, 1])

→ <project>/data/aggregated/mod10c1_v061/mod10c1_v061_<year>_agg.nc
   carries: Day_CMG_Snow_Cover (native 0–100, CI-filtered),
            Day_CMG_Clear_Index (native 0–100, flag-masked only),
            valid_area_fraction (HRU fraction with CI-passing pixels)

normalize/methods.py
    └─ modis_ci_bounds(sca, ci, ci_threshold) — operates on fractional
       HRU-scale (sca, ci) for whichever target window the user picks.

targets/sca.py
    ├─ Read aggregated NC.
    ├─ Apply ÷ 100 to Day_CMG_Snow_Cover and Day_CMG_Clear_Index → fractional [0, 1].
    ├─ Call normalize.modis_ci_bounds(sca, ci) for the CI-bounds target.
    └─ Write CF-compliant per-HRU per-day NetCDF.
```

## Decision flow for a new source

When you add a new source that needs transformations, ask in order:

1. **Is it required so that the area-weighted mean is computable / well-
   defined?** (Mask invalid pixels, derive a single sum from accumulated
   components.) → `pre_aggregate_hook`.
2. **Is it a per-pixel quality threshold whose definition is at the source
   grid?** → `pre_aggregate_hook`. Note: even though linear rescaling
   commutes, threshold operations do not — apply at the granularity where
   the threshold is defined.
3. **Is it a linear unit conversion?** → `targets/<tgt>.py`. (Mirror it in
   the inspect notebook.)
4. **Is it per-source normalization for combinability with other sources?**
   → `normalize/methods.py`.
5. **Is it multi-source combination (min/max bounds, etc.)?** →
   `targets/<tgt>.py`.
6. **Is it cosmetic (rename, attach attrs)?** → `post_aggregate_hook` if
   the rename is naturally post-aggregation (e.g. the per-pixel mask
   becomes a per-HRU fraction); otherwise leave names native and rename
   at point of use.

If the answer is "it doesn't matter, I could put it anywhere" — that's
the linear case (rule 3). Default to `targets/` for consistency.

## How to verify

The inspect notebooks (`notebooks/inspect_aggregated/inspect_aggregated_*.ipynb`)
each have a **validation cell** that compares the area-weighted-mean over
the HRU fabric against an unweighted gridded mean of the same source over
the consolidated-NC bbox. They differ by 5–30% for sources with significant
out-of-fabric coverage; differences much larger than that are a smoking gun
for either:

- A unit conversion applied on one path but not the other (most common).
- A pre-aggregation mask that doesn't match the gridded-mean-side mask
  (catches policy violations like "the aggregator filtered, the gridded
  comparison didn't").

The validation cell is the field-tested practical check that the
transformation policy is being respected. If you're refactoring how
transformations are split between stages, run the inspect notebooks (via
`inspect_aggregated.slurm` for memory) before and after — the % diff
column in each validation cell should be unchanged within rounding.

## Cross-references

- `CLAUDE.md` — concise policy summary for AI assistants.
- `docs/references/calibration-target-recipes.md` — the per-target recipes
  (units, time selection, formulas) that the policy supports.
- `docs/references/tm6b10-summary.md` — what targets we're building and the
  TM 6-B10 methodological reference.
