# Lessons Learned

Findings and operational notes accumulated while building the
`nhf-spatial-targets` pipeline that don't fit cleanly into the per-target
recipes (`calibration-target-recipes.md`) or the per-source catalog
(`catalog/sources.yml`). Each entry documents *what we found* and *where
the consequence lives in the code* so a future maintainer can re-evaluate
the finding when the underlying data product changes.

---

## MOD16A2 v061 flat-on-CONUS+

**Status:** open — pending collaborator consensus on whether to keep
MOD16A2 v061 in the AET multi-source `min/max` bound.

### What we found

The July 2000 cross-check in
`notebooks/inspect_aggregated/inspect_aggregated_aet.ipynb` (commit
`cce4ed6`, finding cell `55a87e3a`) compared the three AET sources at
calendar-month cadence on a CONUS+ subset:

| Source | Jan 2000 (mm/month) | Jul 2000 (mm/month) | Jul/Jan |
|---|---|---|---|
| SSEBop | 9.0 | 101.1 | **11.2×** |
| MWBM (ClimGrid) | 12.5 | 77.8 | **6.2×** |
| MOD16A2 v061 | 33.3 | 37.4 | **1.12×** |

MOD16A2 v061 is essentially flat across the seasonal cycle while the
other two products swing 6–11× as expected for CONUS ET. The flatness is
present at both the gridded level (Jan 65.5 → Jul 70.0 = 1.07×) and the
HRU-aggregated level, so it is already in the consolidated NC, not an
aggregation artefact.

### Why it matters

The AET target uses `multi_source_minmax` over **absolute** mm/month —
no 0–1 normalisation. In both January and July MOD16A2 sets one end of
the bound and pulls it away from where SSEBop and MWBM agree:

- January: `min = SSEBop (9), max = MOD16A2 (33)` → MOD16A2 sets the upper bound.
- July: `min = MOD16A2 (37), max = SSEBop (101)` → MOD16A2 sets the lower bound.

That is the opposite of what a multi-source error envelope is meant to
do. SSEBop and MWBM agree to within ~30% in both seasons, and dropping
MOD16A2 would give a tighter but more honest bound.

### Possible causes (not yet investigated)

- Over-eager flag masking on the consolidated NC (the `<= 3270` threshold
  in `aggregate/mod16a2.py:_mask_et_fill` may be removing real low-ET
  pixels that the v061 GF gap-fill marks as flag-coded).
- Consolidate-step averaging that homogenises composites across the year.
- Genuine v061 GF behaviour on our CONUS+ tile selection, possibly fixed
  in a future v061 reprocess or a v062 release.

### Where to look in the code

- Inspection notebook: `notebooks/inspect_aggregated/inspect_aggregated_aet.ipynb`
  (cells `38002926` for the time-series analysis, `55a87e3a` for the
  finding).
- Aggregation adapter: `src/nhf_spatial_targets/aggregate/mod16a2.py`.
- Catalog entry: `catalog/sources.yml` → `mod16a2_v061` (`notes:` block).
- Target builder (stub): `src/nhf_spatial_targets/targets/aet.py` —
  should accept a `sources` override from project config until the
  decision is finalised.

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
`notebooks/inspect_aggregated/inspect_aggregated_aet.ipynb`); without
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
