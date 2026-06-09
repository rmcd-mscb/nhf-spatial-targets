# Design: finish wiring `ua_swe` (consolidate fix + aggregate + SWE + SCA)

**Date:** 2026-06-09
**Umbrella issue:** #237 (reopened — PR-A landed as #238; A2/B/B2/C/D outstanding)
**Author:** Richard McDonald / Claude

## Problem

NSIDC-0719 (University of Arizona Daily 4-km Gridded SWE & Snow Depth,
`ua_swe`) landed its catalog entry + fetch module in PR-A (#238). The
aggregate layer, the SWE-target wiring, and the SCA multi-source refactor
were sliced as PR-B/C/D in #237 but never implemented; #237 was closed
when PR-A merged, so there is currently no open issue tracking the
remaining work. This spec finishes the source end-to-end across both
targets it feeds (SWE and SCA).

During design we found a blocking defect: `ua_swe`'s PR-A consolidator
emits **per-water-year** NCs (`ua_swe_daily_WY<YYYY>.nc`), but the shared
aggregation driver enumerates work by **calendar year** and hard-raises
when a calendar year appears in two files (`enumerate_years`,
`aggregate/_driver.py:307`). Adjacent water years share a calendar year
(WY2000 = Oct 1999–Sep 2000 and WY2001 = Oct 2000–Sep 2001 both contain
CY 2000), so `agg ua-swe` would crash before aggregating anything.

The repo already solved this for its **other** water-year source. Margulis
WUS-SR raw granules are per-water-year, but
`consolidate_calendar_year_margulis_wus_sr`
(`fetch/margulis_wus_sr.py`) re-windows them to calendar-year NCs
(`margulis_wus_sr_daily_<year>.nc`) at consolidate time, so the aggregator
never sees a water year. `ua_swe`'s consolidator simply didn't follow that
precedent. **Fix: convert at the earliest step** — re-window `ua_swe` to
calendar-year consolidated NCs, leaving the generic aggregate and target
workflows untouched.

Delivered as four stacked, independently reviewable PRs (A2 → B → C → D).

## Locked decisions (planning Q&A, 2026-06-09)

1. **WY→CY remapping lives at consolidate** (PR-A2), matching Margulis. No
   change to the aggregate driver or the target builders; the science is
   identical (the SWE/SCA combine is per-(HRU, day) — file windowing never
   touches daily values), so the calendar-year-uniform path is preferred.
2. **PR-D forced-zero (Jul/Aug):** configurable via new
   `snow_covered_area.forced_zero_combined`, **default `true`** — both
   sources forced to `(0, 0)` in July/August, faithful to
   `PRMSobjfun.f90:calcSCA`. `false` applies the forced-zero mask to the
   MOD10C1 contribution only, letting ua_swe's independent summer alpine
   signal reach the combined upper bound.
3. **PR-D zero-width bounds:** configurable via new
   `snow_covered_area.min_sources_for_bound`, **default `1`** — a cell gets
   a bound whenever ≥1 source is finite, so ua_swe alone yields a
   degenerate `[v, v]` point target in low-CI cells. `2` requires both
   sources, so every bound has non-zero width.
4. **Tracking:** reopen #237, keep its checklist, check off A2/B/B2/C/D as
   each PR lands.

---

## PR-A2 — re-window `ua_swe` consolidator to calendar years

**Confidence: high.** Direct port of the Margulis pattern; *simpler*
because `ua_swe` raw files are flat per-WY NCs (no tile mosaic).

### New consolidator

Replace `consolidate_water_year_ua_swe(wy, raw_path, daily_dir)` with
`consolidate_calendar_year_ua_swe(calendar_year, raw_dir, daily_dir)`:

- Calendar year *X* = `[Jan 1 – Sep 30 of X]` from WY *X* raw
  (`4km_SWE_Depth_WY{X}_v01.nc`, which runs Oct *X-1* – Sep *X*) joined
  with `[Oct 1 – Dec 31 of X]` from WY *X+1* raw
  (`4km_SWE_Depth_WY{X+1}_v01.nc`).
- Keep PR-A's existing per-WY decode + EPSG:5070 reprojection + CF logic;
  factor the "open one raw WY → decoded (time, y, x) Dataset" step into a
  helper, then `_calendar_year_slice` each and `xr.concat` along time.
- Output `<datastore>/ua_swe/daily/ua_swe_daily_<year>.nc` (calendar year,
  matching `margulis_wus_sr_daily_<year>.nc` and `snodas_daily_<year>.nc`).
- mtime idempotency + atomic temp-file rename, as today.

### Boundary years

Full calendar coverage is **1982–2022**. CY 1981 (only Oct–Dec, from
WY1982) and CY 2023 (only Jan–Sep, from WY2023 — WY2024 doesn't exist)
are partial. Follow Margulis: a calendar year missing its WY *X+1* raw
**raises `FileNotFoundError`** and is skipped, so only full calendar years
are emitted. The SWE/SCA target `period` is set within 1982–2022; partial
edges contribute nothing and the multi-source combine covers them via
other sources.

### Fetch orchestration

Today the worker loop consolidates per-WY inline after each download. New:
download the needed WYs first (the fetch already resolves a requested CY to
WY *X* **and** *X+1* via `_calendar_years_to_water_years`), then
consolidate per calendar year once both adjacent WY raws are present —
mirroring Margulis's "consolidate synchronously after each year's
downloads complete." Raw WY NCs stay under `ua_swe/raw/` (re-consolidation
needs no re-download).

### Manifest + catalog

- Update the `ua_swe` manifest writer to record **calendar-year**
  consolidated outputs (raw provenance may still note the contributing
  WYs). Mirror Margulis's record shape.
- Update the `catalog/sources.yml[ua_swe]` `access.notes` that describe
  the consolidated layout as `ua_swe_daily_WY<YYYY>.nc` → `..._<year>.nc`.

### Tests

`tests/test_ua_swe.py` (existing): assert (a) CY X output spans Jan 1 –
Dec 31 with days drawn from both WY raws, (b) the Sep 30 → Oct 1 seam has
no gap or duplicate, (c) a boundary CY missing WY *X+1* raises, (d)
CF-1.6 attrs on output, (e) EPSG:5070 coords preserved.

### Docs + notebooks (PR-A2)

- **Breaking:** `notebooks/consolidated/inspect_consolidated_swe.ipynb`
  hardcodes `ua_swe_daily_WY{TARGET_YEAR}.nc` (registry entry) and a
  markdown note describing WY filenames — both must change to the
  calendar-year `ua_swe_daily_{TARGET_YEAR}.nc`. Check
  `inspect_consolidated_snow_covered_area.ipynb` for the same hardcode.
- Add `docs/sources/ua_swe.md` (per-source page; none exists — model on
  `docs/sources/margulis_wus_sr.md`/`snodas.md`) and the
  `docs/sources/index.md` row. Describe the calendar-year consolidated
  layout, not WY.

---

## PR-B — `aggregate/ua_swe.py` + inspect notebook

**Confidence: high.** With PR-A2 done, consolidated NCs are calendar-year,
so the generic driver just works — no WY handling anywhere. Templated on
`aggregate/snodas.py` (pre-projected CONUS SWE) + `aggregate/mod10c1.py`
(binary pre-aggregate hook).

### Adapter

```
ADAPTER = SourceAdapter(
    source_key="ua_swe",
    output_cadence="daily",
    output_name="ua_swe_agg.nc",
    variables=("swe", "snow_depth", "snow_covered_fraction"),
    source_crs="EPSG:5070",          # pre-projected at consolidate; matches WEIGHT_GEN_CRS
    files_glob="daily/ua_swe_daily_*.nc",   # calendar-year files (post-A2)
    pre_aggregate_hook=<binary-derivation closure>,
    stat_method="masked_mean",       # CONUS-masked product; same rationale as SNODAS #151
)
```

### Pre-aggregate hook — derivation, and why it must precede aggregation

The hook derives a third variable that does not exist in the consolidated
NC:

```
scf = (snow_depth > depth_threshold_mm).astype(float).where(snow_depth.notnull())
```

**Why pre-aggregation — the nonlinearity.** Area-weighted mean is linear;
a threshold is nonlinear; the two do not commute:
`mean(depth > t) ≠ mean(depth) > t`. Worked example — one HRU over 10
equal-area pixels, 4 with 50 mm snow, 6 bare:

- **Pre-aggregation (what we do):** per pixel `depth > 1mm` →
  `[1,1,1,1,0,0,0,0,0,0]` → area-mean = **0.40**. Correctly "40% of the
  HRU is snow-covered."
- **Post-aggregation (the trap):** HRU-mean depth = `(4·50)/10` = 20 mm,
  then `20 > 1` = **1.0**. Claims the *whole* HRU is covered when only 40%
  is.

Fractional snow-covered area is *definitionally* a count-of-snowy-pixels-
before-averaging quantity, so the binary lives in the hook, never in
`targets/sca.py`. This is the worked case behind CLAUDE.md's
transformation policy.

**Why `.where(snow_depth.notnull())` is load-bearing.** In xarray
`NaN > 1.0` is `False`, so a naive `.astype(float)` silently turns every
fill / off-CONUS / ocean pixel into a hard `0.0` — a real "no snow" vote
that dilutes the fraction for any HRU touching the CONUS boundary or an
internal gap. The `.where(...notnull())` re-NaNs those pixels so they are
*excluded*, which pairs with `stat_method="masked_mean"` (averages only
finite survivors; NaN only when the whole HRU is unobserved). This is the
**opposite** NaN policy from `mod10c1.py`'s `valid_mask`, which
*deliberately* lets unobserved pixels become `0.0` because its derived
`valid_area_fraction` means "fraction of HRU with usable observations" —
there unobserved *should* count as zero. Same mechanical shape
(binary → float → area-mean), opposite intent: copying mod10c1's hook
verbatim yields a subtly wrong SCA fraction along every coastline. `swe`
and `snow_depth` pass through with native names/units (`kg m-2`, `mm`); no
rescale at agg time.

### Threshold coupling (architectural wrinkle — documented loudly)

`depth_threshold_mm` is *conceptually* an SCA-target knob, so it lives in
config under `snow_covered_area`. But the binary it controls must be
evaluated on the pixel grid (above), so the threshold is applied at
**aggregate** time and its result is **baked into the aggregated NC**.
`aggregate_ua_swe(...)` reads `snow_covered_area.depth_threshold_mm`
(default `1.0`), binds it into the hook via closure (the hook signature
has no `project`, so the module-level `ADAPTER` is parameterized per run),
and stamps it as an attr on `snow_covered_fraction`. Three consequences:

1. The **aggregate stage reads a target-stage config key** — a deliberate
   crossing of the otherwise target-agnostic agg/target boundary.
2. **Changing the threshold invalidates the agg NC, not just the target.**
   Bump 1 → 5 mm and `snow_covered_fraction` on disk is stale; the
   operator must re-run `agg ua-swe` (only the derived var rots;
   `swe`/`snow_depth` are untouched). The stamped attr makes the staleness
   *detectable* rather than silent (see release section).
3. It is an **inherent trade**, not a wart: you cannot have both a
   config-tunable fractional-SCA threshold *and* a threshold-independent
   agg NC, because the tunable parameter sits inside the nonlinearity that
   must precede aggregation.

**The mod10c1 asymmetry a reviewer will ask about.** mod10c1 also bakes a
pixel decision (CI > 70) into its agg NC, yet its configurable
`ci_threshold` (0.70) does *not* force a re-agg. Why? Because mod10c1's
tunable threshold gates the **already-aggregated HRU-mean CI** — a linear
quantity safe to gate post-aggregation — whereas ua_swe's tunable
threshold defines the **binary that must precede** aggregation. Same word
"threshold," different stage, dictated by where the nonlinearity sits.
Rejected alternatives: *aggregate depth only, threshold at target stage*
(the 40%→100% wrong answer); *hardcode the threshold like mod10c1's 70*
(loses the configurability #237 wants — 1 vs 5 vs 25 mm is a real
shallow-snow sensitivity knob).

### Tests + notebook

`tests/test_ua_swe_aggregate.py`: assert (a) an all-NaN-depth HRU yields
NaN `snow_covered_fraction` (not 0), (b) a half-snow HRU yields ~0.5, (c)
threshold attr stamped, (d) CF-1.6 attr set on output.

### Docs + notebooks (PR-B)

- Add `notebooks/aggregated/inspect_aggregated_ua_swe.ipynb` (per-source
  HRU aggregate; one `datasets={}` registry entry per the pattern).
- Add ua_swe to the multi-source aggregated notebooks it feeds:
  `notebooks/aggregated/inspect_aggregated_swe.ipynb` and
  `inspect_aggregated_snow_covered_area.ipynb` (registry entry each).
- Add the `agg ua-swe` line to CLAUDE.md's aggregation command list.

---

## PR-B2 — threshold provenance + operator documentation

**Confidence: medium.** Depends on PR-B's stamped attr. The baked
`depth_threshold_mm` is provenance: a published `snow_covered_fraction` is
meaningless without the threshold that defined it. It must flow into the
release artifact, the publish gate must refuse a stale fraction, and the
pre/post-aggregation rationale must be documented where operators look.

### Release provenance — the threshold travels with the data

The release pipeline already has the seams:

1. **Stamp → attr.** (Done in PR-B: `depth_threshold_mm` on
   `snow_covered_fraction` in the agg NC.)
2. **Projection lifts attr → step params.** Extend the generic
   `rebuild_manifest` projection to read
   `snow_covered_fraction.attrs["depth_threshold_mm"]` from the ua_swe agg
   NC into that aggregate step's `params` dict (`release/lineage.py` steps
   already carry `params`, e.g. `batch_size`). Read from the **NC attr,
   not config** — the rebuild is a pure function of disk and must stay
   deterministic; config could have drifted. Add the case to
   `tests/test_rebuild_manifest.py` (per CLAUDE.md's "new on-disk layout →
   extend the projection + test").
3. **Params → FGDC/ISO automatically.** `release/mcf.py:237-240` already
   renders step `params` into the metadata (`params: depth_threshold_mm=1.0,
   ...`), so once (2) lands the threshold appears in the released FGDC/ISO
   with no metadata-template change.
4. **Publish staleness gate.** Add a preflight (sibling to
   `_preflight_effective_config_current`) that refuses to publish when the
   agg NC's stamped `depth_threshold_mm` ≠ the project config's
   `snow_covered_area.depth_threshold_mm` — i.e. the operator edited the
   threshold without re-aggregating. It is the agg-layer analogue of the
   config-staleness gate and closes the same "edited config, forgot to
   re-run" footgun for this baked param. (First instance of an
   agg-affecting config param; spec it narrowly for ua_swe but name the
   general pattern in the code comment.) Add a `tests/test_release_*`
   case covering the mismatch-rejects / match-passes paths.

### Operator documentation — make the coupling impossible to miss

- `docs/architecture/transformation-pipeline.md`: add the
  `snow_covered_fraction` worked example (40%→100%) as the canonical
  illustration of the pre/post-aggregation gotcha, and document the
  threshold-baking + re-agg coupling.
- `docs/data_release/usgs-process.md`: note that the SCA depth threshold is
  captured in the release manifest step params and surfaced in FGDC/ISO.
- (`docs/sources/ua_swe.md` is owned by PR-A2 and the `depth_threshold_mm`
  config-stub warning by PR-D; both cross-reference this coupling.)

---

## PR-C — SWE target

**Confidence: high.** One shim. `ua_swe.swe` is `kg m-2 ≡ mm`,
identity-to-mm like daymet/snodas.

- Add to `targets/swe.py:SHIMS`:
  `SourceShim(source_key="ua_swe", aggregated_var="swe",
  description="UA SWE (kg/m² ≡ mm, daily)", to_common_units=ua_swe_to_mm,
  expected_cf_units="kg m-2")`.
- Add `ua_swe_to_mm` (identity, sets `units="mm"`).
- Add `ua_swe` to `defaults.py:snow_water_equivalent.sources` and
  `variables.yml:snow_water_equivalent.sources` + `range_notes`.
- Flows through `multi_source_nanminmax` untouched. Update docstrings
  4→5 sources and the period-intersection note: ua_swe (1982–2022 full CY)
  widens the pre-2003 bound — the original motivation for the source.
- Not fabric-scoped (CONUS) → contributes on every fabric.
- `tests/test_swe.py`: assert ua_swe participates in the min/max envelope
  and the n_sources count.

### Docs + notebooks (PR-C)

- `notebooks/targets/inspect_target_swe.ipynb`: bump "up to four sources"
  → five and the `n_sources` legend `0..4` → `0..5`; add ua_swe to the
  per-source aggregate cross-references.
- `notebooks/consolidated/inspect_consolidated_swe.ipynb`: ua_swe is
  already in the registry (path fixed in PR-A2) — verify its description
  lists it as a SWE-bound contributor, not just an SCA proxy.
- README Implementation Status / Calibration Targets: list ua_swe as a
  SWE source (5-source bound; 1982–2022).

---

## PR-D — SCA multi-source refactor

**Confidence: medium.** `targets/sca.py` today hardcodes `_MOD10C1_KEY`
and **asserts** `sources == [mod10c1_v061]`. This is a refactor of working,
implemented code (the calcSCA formula is live — CLAUDE.md's "sca is a stub"
comment is stale and will be corrected here).

### Per-source interval registry

Each source produces a per-(HRU, day) interval; combine NaN-aware:

- **mod10c1** → CI-bounded `[ci·sca, ci·sca + (1−ci)]` where
  `ci ≥ ci_threshold`, else NaN (today's formula, unchanged).
- **ua_swe** → degenerate `[v, v]`, `v = snow_covered_fraction` (a 0–1 HRU
  fraction from PR-B's agg NC). No CI gate — ua_swe is physical.
- **combine:** `lower = nanmin(intervals)`, `upper = nanmax(intervals)`.

### Forced-zero (Jul/Aug) — config-gated `forced_zero_combined` (default `true`)

- `true`: existing behaviour — driver's `forced_zero_months` /
  `forced_zero_validity_var` zeros the **combined** bound in Jul/Aug.
- `false`: zero the **mod10c1 contribution only** before the combine (in
  the loader), leaving driver-level forced-zero off, so ua_swe's summer
  alpine fraction survives into the combined upper bound.

### Zero-width bounds — config-gated `min_sources_for_bound` (default `1`)

- `1`: bound wherever ≥1 source finite (degenerate `[v,v]` allowed).
- `2`: mask the combined bound to NaN where per-cell source count `< 2`
  (threshold on the existing `n_sources` diagnostic), guaranteeing width.

### Config schema additions (4-point checklist ×3)

`snow_covered_area` gains `depth_threshold_mm` (1.0; consumed at **agg**),
`forced_zero_combined` (true), `min_sources_for_bound` (1). For each:
update `init_run.py:_CONFIG_TEMPLATE`, `config/pipeline.yml`,
`tests/test_init_run.py`, `upgrade_config.py:OPTIONAL_CONFIG_FEATURES`.
`sources` default becomes `["mod10c1_v061", "ua_swe"]`.

### Builder validation

Replace the `sources == [mod10c1_v061]` assertion with: every source has a
registered interval loader; `ci_threshold ∈ [0,1]`; `min_sources_for_bound
∈ {1, 2}`; `depth_threshold_mm > 0`.

### Tests

`tests/test_sca.py`: (a) mod10c1-only path unchanged (regression), (b)
two-source combine widens the bound, (c) `forced_zero_combined=false`
preserves a synthetic July ua_swe fraction, (d) `min_sources_for_bound=2`
NaNs a single-source cell, (e) ua_swe-only low-CI cell yields `[v,v]`
under default config.

### Docs + notebooks (PR-D)

- `notebooks/targets/inspect_target_sca.ipynb`: update from single-source
  MOD10C1 to the two-source combine; document the `forced_zero_combined` /
  `min_sources_for_bound` options and their effect on the bound.
- `notebooks/consolidated/inspect_consolidated_snow_covered_area.ipynb`
  and `inspect_aggregated_snow_covered_area.ipynb`: ensure ua_swe's
  `snow_covered_fraction` is shown alongside MOD10C1.
- CLAUDE.md: correct the stale "sca is a stub pending #210" comment
  (line ~22) — the calcSCA formula is implemented; this PR makes it
  multi-source.
- README + `docs/references/tm6b10-summary.md`: note SCA is now a
  two-source bound (MOD10C1 CI-interval ∪ ua_swe depth-derived fraction).
- `docs/references/known-gaps-resolved.md`: record the ua_swe SCA
  second-source closure.

---

## Out of scope

- Re-unifying SCA with the generic year-chunked driver (#230) — already
  done; `sca.py` is `year_chunked=True` today.
- Margulis / other SWE-source behaviour (its CY consolidation is the
  template we copy, not something we change).
- Any change to `ua_swe`'s download path or archive URLs (PR-A, merged).

## Build order

PR-A2 (consolidate re-window + tests) → PR-B (agg + binary hook +
threshold attr + per-source notebook) → PR-B2 (manifest-projection lift +
publish staleness gate + transformation-pipeline.md + usgs-process.md) →
PR-C (one shim + config/catalog + tests) → PR-D (sca refactor + config
schema ×3 + tests). Each rebased on the prior, squash-merged, checked off
in #237.

The four PRs are a **dependency chain** (PR-B reads PR-A2's calendar-year
NCs; PR-D reads PR-B's `snow_covered_fraction`), so they cannot be
developed in parallel. Worktrees are still worth using for **isolation +
review pipelining**: develop each PR in its own git worktree branched off
the previous PR's branch, so the next slice can start while the prior is
in review without disturbing the main checkout. Per project norms, run
`pixi install -e dev` once in each fresh worktree (per-dir `.pixi`) before
committing, and rename the auto-named branch to the
`feature/237-...` convention. The within-PR docs/notebook edits are small
enough that they ride with their code PR rather than a separate worktree.
