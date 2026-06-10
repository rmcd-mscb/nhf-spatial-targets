# https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/docs/references/PRMSobjfun.f90 — crib sheet keyed to this repo

Companion reference for [`https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/docs/references/PRMSobjfun.f90`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/docs/references/PRMSobjfun.f90), the Fortran
calibration objective function used downstream of this pipeline by PRMS-by-HRU
calibration. The file lived outside the previously-public NHM source for years
and is what TM 6-B10 §3 cross-references when it says "see the PRMS objective
function code for the exact bound mechanics."

Source: copied verbatim from a collaborator (2026-05-24) into this repo for
inspection only — **not** compiled or run here. Authoritative copy lives with
the PRMS-by-HRU calibration codebase.

---

## What this file is

`module objfun_mod` containing `subroutine PRMSobjfun(...)` plus 5 per-target
helpers (`calcRUN`, `calcAET`, `calcSCA`, `calcRCH`, `calcSOM`) and 6 reusable
NRMSE / bound-check pure functions. Called once per parameter-set evaluation
during PRMS calibration; reads observed streamflow and per-HRU target text
files from disk; writes a single objective-function value (`prmsOF`).

The file's outputs are inputs to the optimiser; the **inputs** to this file are
what *this* repo (`nhf-spatial-targets`) produces — the per-HRU per-time
`lower_bound` / `upper_bound` values for the 5 implemented targets, which the
operator must export to PRMS-format text files (`./RUN/HRU_<id>`,
`./AET/HRU_<id>`, `./SCA/HRU_<id>`, etc.) before PRMS calibration starts.

## The bound mechanic (every target the same)

Two pure functions in the file express the entire bound semantics that this
pipeline serves:

```fortran
pure function ranged_diff(val, upper, lower) result(res)
  ...
  if (upper == lower) then
    res = (val - upper)**2
  else
    if (val > upper) res = (val - upper)**2
    if (val < lower) res = (val - lower)**2
  end if
end function

pure function nrmse_range(darray, max_vals, min_vals, num_times, weight)
  ! sumdiff = sum over i of ranged_diff(darray(i), max_vals(i), min_vals(i))
  ! rmsd    = sqrt(sumdiff / num_times)
  ! return    (rmsd / (max(max_vals) - min(min_vals))) * weight
end function
```

**Three crisp consequences for the pipeline:**

1. **Inside the bound = zero cost.** A simulated value strictly between
   `lower` and `upper` adds nothing to the objective. This is the
   "soft-constraint with width" framing the decks use.
2. **Outside the bound = squared distance.** The penalty is
   `(val - nearest_bound)**2` — not absolute distance, not signed; the
   optimiser sees a smooth quadratic well shape rising from the bound edge.
3. **When `upper == lower` (bound collapsed) the penalty is *always*
   `(val - bound)**2`.** The collapsed bound becomes a point estimate; the
   optimiser is forced to *match exactly* at that HRU/time.

That third consequence is load-bearing for several pipeline edge cases:

- **SCA July/August (calcSCA, lines 1052-1061).** Hardcoded to
  `lower = upper = 0.0` for months 7 and 8 — the optimiser is *forced* to
  predict zero SCA in July/August, every HRU, every year. Honest in the PNW
  where the snowpack is gone; questionable in deep-snowpack HRUs (Cascades
  high-elevation) where late-summer snow persists. Worth flagging.
- **Recharge `normalized_minmax` saturation.** When both sources hit their
  in-window decadal max for the same HRU/year (e.g. 2006 in the PNW for
  Reitz 2017 + ERA5-Land `ssro`), the normalised bound collapses to
  `(1.0, 1.0)` and the optimiser is forced to match exactly 1.0. This is
  by design — both sources flagged it as "the most-recharge year in the
  period" — but it is the strictest constraint among the 10 years.
- **Single-source bound years** (e.g. AET 2024 once SSEBop ends 2023, or
  Recharge 2014+ once Reitz ends 2013, if the operator extends the period).
  `multi_source_minmax` of one source = `(val, val)` → point estimate
  every step.

## What's there for each target

`PRMSobjfun` computes `prmsOF = sum(NRMSE_streamflow[1..7]) + (ofRUN + ofAET + ofSCA + ofRCH + ofSOM) / 5.0`
each evaluation. The two halves:

### Streamflow (NRMSE[1..7]) — 7 components per gage

These read **observed USGS streamflow** (per-day median / min / max) and
compare against simulated streamflow `sim_vals`. Nothing in this pipeline
produces them — they live in the PRMS calibration project's own
`./CAL_Q/` files.

The 7 split:

| OF | Aggregation | Comparison | Notes |
|---|---|---|---|
| 1 | monthly totals | range (min/max) | `nrmse_range` |
| 2 | per-calendar-month climatology | range (min/max) | `nrmse_range` |
| 3 | monthly totals | median | `nmrse` |
| 4 | per-calendar-month climatology | median | `nmrse` |
| 5 | daily | range (min/max) | `nrmse_range` |
| 6 | daily HIGH EFCs (1-3) | median | `nmrse_median_efc_high` |
| 7 | daily LOW EFCs (4-5) | median | `nmrse_median_efc_low` |

EFC = Environmental Flow Components (large floods / small floods / high
pulses / low / extreme-low; lines 240-247 of the file).

### Per-target HRU bounds (this pipeline's deliverable)

These read per-HRU text files written from this pipeline's NetCDF
`lower_bound` / `upper_bound` outputs:

| Subroutine | Reads | Bound source per HRU/time | Period const |
|---|---|---|---|
| `calcRUN`  | `./RUN/HRU_<id>` (`year month median min max`) | MWBM-derived (in this file's TM-6-B10 vintage); replaced by this repo's multi-source `runoff_targets.nc` | `START_YM_RUN`..`END_YM_RUN` (monthly) |
| `calcAET`  | `./AET/HRU_<id>` | per-row min/max | `START_YR_AET`..`END_YR_AET` |
| `calcSCA`  | `./SCA/HRU_<id>` (`year month day obs CI`) | **computed inside the routine from MOD10C1 obs + CI — see below** | `START_YR_SCA`..`END_YR_SCA` |
| `calcRCH`  | `./RCH/HRU_<id>` | per-row min/max (annual) | `START_YR_RCH`..`END_YR_RCH` |
| `calcSOM`  | `./SOM/HRU_<id>` | per-row min/max | `START_YR_SOM`..`END_YR_SOM` |

All of these except SCA expect the pipeline to deliver `(lower, upper)` for
each HRU/time and use `ranged_diff` directly. SCA is special — see next
section.

### Calibration steps (iSTEP 1..4)

`weight(iSTEP, 1..7)` reweights the 7 streamflow components per step:

| iSTEP | Name | Heaviest weights |
|---|---|---|
| 1 | VOLUME | OF[1..4] (monthly aggregates) = 3.0 |
| 2 | HIGH | OF[6] (HIGH EFCs) = 3.0 |
| 3 | LOW | OF[7] (LOW EFCs) = 3.0 |
| 4 | ALL | OF[5..7] (daily) = 3.0 |

All other weights = 1.0. The 5 per-target HRU OFs are **not** reweighted by
step — they always contribute `(ofRUN + ofAET + ofSCA + ofRCH + ofSOM) / 5.0`.

Mirrors TM 6-B10 §3.6 (the staged calibration procedure).

## SCA bound formula (no longer "missing")

`calcSCA` (lines 1052-1061) constructs its bounds **inside the routine**
from the raw MOD10C1 `(obs, CI)` pair, not from upstream files:

```fortran
if (CI >= 70.0) then
  obsSCA(ihru, LOWER, n) = (CI / 100.0) * obs
  obsSCA(ihru, UPPER, n) = obsSCA(ihru, LOWER, n) + (100.0 - CI) / 100.0

  if (nm == 8 .or. nm == 7) then
    obsSCA(ihru, LOWER, n) = 0.0
    obsSCA(ihru, UPPER, n) = 0.0
  end if
end if
```

Per HRU per day, when MOD10C1 reports clear-index `CI > 70`:

- `lower = (CI/100) * SCA_obs` — the confidence-weighted "definitely
  snow-covered" floor.
- `upper = lower + (1 - CI/100)` — adds back the maximum amount that *could*
  be snow obscured by cloud or canopy, expressed as a fraction.
- **July/August forced to `(0, 0)`** regardless of `obs` — late-summer
  hardcoded as no-snow.
- `CI ≤ 70`: no entry written; the day doesn't contribute to the OF.

Then it walks daily sim values:

```fortran
diff2 = ranged_diff(sim(ihru, cday),
                    obsSCA(ihru, UPPER, i),
                    obsSCA(ihru, LOWER, i)) * (obsSCA(ihru, LOWER, i) + 1.0)
sumdiff = sumdiff + (diff2 * (obsSCA(ihru, LOWER, i) + 1.0))
```

(Note: PAN's comment in the source — `! NOTE: PAN - I don't understand the
math here; should look into it` — flags that the `(LOWER + 1.0)` weight is
applied *twice*, once inside `diff2` and once when accumulating. May be a
typo in the original; downstream PRMS treats it as load-bearing. **For
this pipeline**: `targets/sca.py` (issue #210) should emit `lower_bound`
and `upper_bound` per the formula above and *not* replicate the double-
weight quirk — the weight is an objective-function-side concern that
PRMS/PEST++ applies on top of the bound, not part of the bound itself.)

The bound is then `(LOWER, UPPER)` per HRU/day where CI passed the gate;
days where CI ≤ 70 are dropped from the RMSE denominator (`n` only
increments for valid days).

**Implication for this pipeline:** `targets/sca.py` now implements this
formula (the stub is gone — #237 extended it to a two-source MOD10C1 + UA SWE
bound). The MOD10C1 interval is roughly:

```python
def build_sca_target(mod10c1_aggregated_nc, ci_threshold=0.70):
    ds = xr.open_dataset(mod10c1_aggregated_nc)
    obs = ds["snowcov_area"]           # HRU-aggregated SCA fraction
    ci  = ds["clear_index"]            # HRU-aggregated clear-sky fraction (0-100)

    valid = ci >= ci_threshold * 100   # CI > 70 gate
    lower = xr.where(valid, (ci / 100.0) * obs, np.nan)
    upper = xr.where(valid, lower + (1.0 - ci / 100.0), np.nan)

    # July/August hard zero
    month = ds["time"].dt.month
    summer = (month == 7) | (month == 8)
    lower = xr.where(summer & valid, 0.0, lower)
    upper = xr.where(summer & valid, 0.0, upper)

    return xr.Dataset({"lower_bound": lower, "upper_bound": upper, ...})
```

Watch-outs for that build:

- The `clear_index` HRU aggregate must be computed pre-aggregation per
  the `mod10c1` adapter's existing `pre_aggregate_hook` (it already gates
  pixels with `CI > 70` before averaging — see
  `aggregate/mod10c1.py`). The HRU-mean CI used here is the *area-weighted
  mean clear-sky fraction* after that gate, which is different from
  "all pixels had CI > 70."
- The July/August zero-snow assumption breaks for OR's Cascades high-
  elevation HRUs that hold July snowpack. The PRMS-objfun code accepts
  this loss of fidelity; if the modelling team wants OR-realistic snow
  through August they need to either revisit the formula or accept the
  optimiser pulling those HRUs toward zero in summer.
- `START_YR_SCA` and `END_YR_SCA` are calibration-side constants that
  bracket the SCA contribution. Our pipeline emits the full source
  period; the consumer subsets at read time, so the period decision is
  PRMS-side, not pipeline-side.

## SWE is not in this file

PRMSobjfun has no `calcSWE` and the variable enumeration at lines 135-139
(`AET / RCH / RUN / SCA / SOM`) is explicit about the five-target set.
SWE is a *post-TM-6-B10 extension* added by this pipeline; downstream
calibration would need a `calcSWE` analog added to the Fortran module
before SWE bounds reach the optimiser. Until then, the SWE target
NetCDFs this pipeline produces are useful for inspection and for any
future SWE-aware calibration, but they are not consumed by the current
PRMS objfun.

## Things the file confirms about this pipeline's choices

- **Multi-source min/max is upstream-correct for runoff/AET/SOM.** The
  `nrmse_range` driver expects exactly the `(lower, upper)` shape we emit;
  PRMS calibration was originally built around a single MWBM source for
  runoff and we've widened it without changing the consumer shape.
- **0-1 normalised bounds for recharge + soil moisture are upstream-correct.**
  `calcRCH` and `calcSOM` both compute `min_val = minval(min_vals)`,
  `max_val = maxval(max_vals)` from the per-row bounds and divide RMSE by
  that range. Our normalisation collapses that range to `(0, 1)` per
  source-then-min/max-combined, so the upstream divisor is exactly 1 and
  the NRMSE collapses to RMSE on the normalised values. Mathematically
  equivalent to whatever absolute-units bound width the historical
  calibration used.
- **NaN-aware multi-source combination is essential.** When a year/HRU
  has only one source contributing (1-source-tail extensions, fabric
  coverage gaps), the bound collapses to `(val, val)` which `ranged_diff`
  treats as a point estimate. That's what we want — but the pipeline
  must emit the right HRU as NaN (not as the source's own value) for
  HRUs entirely outside a source's extent, because PRMS would otherwise
  see a fabricated point estimate. The fix in #205 (write-side + read-side
  `mask_netcdf_default_fills`) is what keeps this honest.

## Things this file makes us reconsider

- **SCA is built.** `targets/sca.py` implements this formula (#237), extended
  to a two-source MOD10C1 + UA SWE NaN-aware bound.
- **Bound-collapse to a point estimate is well-defined, not a bug.** When
  the operator sees `upper == lower` in a deck figure, the question is
  whether the resulting point-estimate constraint is *informative* — not
  whether the pipeline broke. A point-estimate constraint is the *strictest*
  type the optimiser sees; if it's spurious (single source, normalisation
  saturation) it over-weights that observation.
- **The hardcoded July/August zero for SCA is a real modelling assumption,
  not just a coding convenience.** Worth a line in the SCA deck slide and a
  callout when we ship SCA bounds to the calibration team.

## See also

- [TM 6-B10 summary](tm6b10-summary.md) — methodological reference; §3.6
  describes the 4-step calibration progression mirrored by `iSTEP 1..4`.
- [Calibration target recipes](calibration-target-recipes.md) — per-target
  unit conversion + multi-source combination decisions the pipeline makes
  before its outputs feed this objective function.
- [Known gaps (resolved)](known-gaps-resolved.md) — the "PRMSobjfun.f not
  publicly available" entry can come off the open list now.
