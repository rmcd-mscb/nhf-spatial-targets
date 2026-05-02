---
marp: true
theme: default
paginate: true
size: 16:9
header: '**nhf-spatial-targets** · collaborator overview'
footer: '2026 · NHM calibration targets'
style: |
  section { font-size: 22px; }
  section h1 { font-size: 38px; }
  section h2 { font-size: 32px; }
  table { font-size: 18px; }
  pre { font-size: 16px; }
  img { max-height: 460px; }
  .footnote { font-size: 14px; color: #555; }
---

# nhf-spatial-targets

### Pipeline overview & pre-target inspection findings

Collaborator briefing — calibration-target consensus

Goal of this session: align on **(a) period of record per target group**
and **(b) datasets used in each target group**.

<span class="footnote">
USGS National Hydrologic Model · TM 6-B10 (Hay et al. 2022)
</span>

---

## Today's plan

1. **Pipeline tour** — repo layout, datastore convention, project setup, workflow
2. **Per-target inspection results** — one or two figures per target, the open questions
3. **Decisions** — period of record + datasets, per target group
4. **Open discussion**

Detail-level docs are linked at the end; this deck stays at the
why / what-changed / where-to-look layer.

---

## Project purpose

Build the **five TM 6-B10 calibration targets** for the National Hydrologic
Model by spatially aggregating gridded source datasets to an HRU fabric via
[`gdptools`](https://github.com/rmcd-mscb/gdptools).

| Target | PRMS variable | Method |
|---|---|---|
| Runoff | `basin_cfs` | Multi-source min/max over absolute cfs |
| AET | `hru_actet` | Multi-source min/max over absolute mm/month |
| Recharge | `recharge` | 0–1 normalised min/max |
| Soil moisture | `soil_rechr` | 0–1 normalised min/max |
| Snow-covered area | `snowcov_area` | MOD10C1 CI bounds |

Source-of-truth: `README.md`, `catalog/variables.yml`.

---

## Five sources of "spread" per target

Pipeline emphasises **multi-source bounds** rather than a single best estimate:

- *Different products* see different physics → different answers.
- The *envelope* (min/max) defines the calibration uncertainty.
- Targets are *not absolute* observations — they are *constraints with width*.

Why this matters: a smaller / tighter bound isn't always better.
A bound that's *too tight* under-constrains the optimisation — calibration
will be more sensitive to product bias than to genuine model error.

---

## Repo module map

```
catalog/                          # YAML registry — source-of-truth
  sources.yml                     # per-source: access, variables, period
  variables.yml                   # per-target: sources, range_method, period

src/nhf_spatial_targets/
  fetch/      ← per-source download adapters
  aggregate/  ← gdptools-driven HRU aggregation, per source
  normalize/  ← 0–1 norm + nearest-neighbour HRU fill
  targets/    ← per-variable target builders (some still stubs)
  catalog.py  ← single Python interface to the YAML catalog
  cli.py      ← `nhf-targets` entry point

notebooks/
  inspect_aggregated/             # per-target QC notebooks
  inspect_consolidated/           # source-NC QC, pre-aggregation
```

---

## CLI entry points

```bash
# Project lifecycle
pixi run nhf-targets init             --project-dir /data/my-run
pixi run nhf-targets validate         --project-dir /data/my-run
pixi run nhf-targets fetch <source>   --project-dir /data/my-run
pixi run nhf-targets agg   <source>   --project-dir /data/my-run
pixi run nhf-targets run              --project-dir /data/my-run

# Catalog inspection (no project needed)
pixi run catalog-sources
pixi run catalog-variables
```

All commands take `--project-dir`; the project dir + the catalog are the
two state inputs. Everything else is derived.

---

## Datastore vs project

The pipeline separates two roles:

| **Datastore** | **Project** |
|---|---|
| Raw downloaded source data | Fabric-specific outputs |
| Fabric-independent, expensive | Fabric-tied, reproducible from datastore |
| Re-usable across many fabrics | One per (fabric, run-time) pair |
| `<datastore>/<source_key>/...` | `<project>/data/aggregated/<source>/...` |

**One datastore can serve many projects.** Switching from GFv1.1 → GFv2.0
re-uses the same fetched source data; only weight caches and aggregated
outputs differ.

---

## Datastore layout

```
/caldera/.../nhf-datastore/
  era5_land/                    monthly/era5_land_monthly_{year}.nc
  gldas_noah_v21_monthly/       gldas_consolidated.nc
  merra2/                       merra2_consolidated.nc
  mod10c1_v061/                 mod10c1_v061_{year}_consolidated.nc
  mod16a2_v061/                 mod16a2_v061_{year}_consolidated.nc
  mwbm_climgrid/                ClimGrid_WBM.nc
  ncep_ncar/                    ncep_ncar_consolidated.nc
  nldas_mosaic/                 nldas_mosaic_consolidated.nc
  nldas_noah/                   nldas_noah_consolidated.nc
  reitz2017/                    reitz2017_consolidated.nc
  watergap22d/                  watergap22d_qrdif_cf.nc
```

Source-of-truth: `README.md` § Per-Source Fetch Pipeline.

---

## Project layout

```
/caldera/.../gfv2-spatial-targets/
  config.yml                  ← fabric path, datastore path, target enables
  .credentials.yml            ← gitignored; CDS + Earthdata creds
  fabric.json                 ← computed fabric metadata (from validate)
  manifest.json               ← provenance: fetches, checksums, periods
  data/aggregated/<source>/   ← per-year aggregated NCs (HRU-resolution)
  targets/                    ← final calibration target datasets
  weights/                    ← gdptools weight caches (fabric × source grid)
  logs/                       ← per-run logs
```

Never delete a project directory — it's the audit trail.

---

## Workflow

```
init        → empty project skeleton + config.yml template
   ↓
validate    → preflight checks (fabric, datastore, creds, catalog)
              → writes fabric.json + initial manifest.json
   ↓
fetch <src> → download to <datastore>/<src>/   (incremental, manifest-tracked)
   ↓
agg   <src> → gdptools area-weighted aggregation to HRU fabric
              writes <project>/data/aggregated/<src>/<src>_<year>_agg.nc
   ↓
run         → build calibration targets from aggregated NCs
              writes <project>/targets/...
```

`fetch` and `agg` are idempotent and per-source. Only `run` is end-to-end.

---

## config.yml — the project's contract

```yaml
fabric:
  path: /caldera/.../gfv2_param_v2/gfv2/fabric/gfv2_nhru_merged.gpkg
  id_col: nat_hru_id
  crs: EPSG:4326          # plotting CRS; aggregator re-projects to EPSG:5070

datastore: /caldera/.../nhf-datastore

targets:
  aet:
    enabled: true
    sources: [ssebop, mwbm_climgrid, mod16a2_v061]   # override catalog default
    period: "2000/2010"
  recharge:
    enabled: true
    period: "2000/2009"   # normalisation window
  ...
```

Project config can *override* catalog defaults — used when the consensus on
sources or period changes for a specific run.

---

## manifest.json — provenance record

Records, atomically, *what was fetched / aggregated, when, and from where*:

```json
{
  "fabric_meta": { "path": "...", "id_col": "...", "sha256": "..." },
  "sources": {
    "era5_land": {
      "fetched_periods": ["2000/2010"],
      "files": [
        { "path": "monthly/era5_land_monthly_2000.nc",
          "sha256": "...", "fetched_at": "2026-04-29T..." }
      ]
    },
    ...
  }
}
```

Re-running `fetch` skips periods already recorded — guards against
duplicate downloads when re-running targets.

---

# Per-target inspection results

What follows is the per-target HRU-level QC: what we have aggregated to the
fabric, what looks right, what's open.

Source notebooks live at `notebooks/inspect_aggregated/inspect_aggregated_*.ipynb`.
Each notebook produces ~5–7 figures and saves them to
`docs/figures/inspect_aggregated/` for re-use.

---

## Runoff (RUN)

- **Sources:** ERA5-Land `ro` + GLDAS-2.1 NOAH `Qs_acc + Qsb_acc`
- **Method:** multi-source min/max over absolute cfs
- **Cadence:** monthly

Native-unit map confirms the unit-conversion fix from PR #68:
GLDAS `_acc` fields are *means of 3-hourly accumulations*, not sums — must
be multiplied by `8 × days_in_month` (was 224–248× too small without it).

![w:900](../figures/inspect_aggregated/runoff_native_units_map.png)

---

## Runoff — cross-source comparison

![w:900](../figures/inspect_aggregated/runoff_normalized_comparison.png)

Two sources, mm/month on a shared colour scale (ERA5-Land reference).
Pattern agreement is good in the wet–dry gradient; magnitude offsets
reflect different LSMs (H-TESSEL vs NOAH) and forcing data — exactly the
kind of inter-product spread the min/max bound is meant to capture.

---

## AET

- **Sources:** SSEBop + MWBM (ClimGrid) + MOD16A2 v061¹
- **Method:** multi-source min/max over **absolute mm/month**
  (no 0–1 normalisation — magnitudes propagate to the bound)
- **Cadence:** monthly; MOD16A2's 8-day composites resampled via
  overlap-weighted sum (recipes §2)

![w:900](../figures/inspect_aggregated/aet_normalized_comparison.png)

<span class="footnote">¹ MOD16A2 v061 inclusion is pending today's discussion — see slide 18.</span>

---

## AET — distribution shape

![w:780](../figures/inspect_aggregated/aet_histogram.png)

Cross-source distribution at the chosen target month. Look for:
**mode separation** (where is the typical HRU's ET?), **tail behaviour**
(real high-ET regions vs flag-code leakage), and per-source bias.

---

## AET — open question: MOD16A2 v061

The January-only validation left open whether MOD16A2's high-winter /
low-summer offset was a real product behaviour or a pipeline issue. The
July cross-check answers it:

| Source | Jan 2000 | Jul 2000 | Jul/Jan |
|---|---|---|---|
| SSEBop | 9.0 mm/month | 101.1 mm/month | **11.2×** |
| MWBM | 12.5 | 77.8 | **6.2×** |
| MOD16A2 v061 | 33.3 | 37.4 | **1.12×** |

MOD16A2 v061 is **essentially flat** across the seasonal cycle on CONUS+
while the other two products swing 6–11× as expected for CONUS ET.
Holds at both gridded and aggregated levels — already in the consolidated
NC, not an aggregation artefact.

---

## AET — what the flatness does to the bound

| Month | min | max | bound dominated by |
|---|---|---|---|
| January | SSEBop 9 | MOD16A2 33 | MOD16A2 *upper* (winter overestimate) |
| July | MOD16A2 37 | SSEBop 101 | MOD16A2 *lower* (summer underestimate) |

In **both** seasons MOD16A2 is the outlier and pulls the bound away from
where SSEBop and MWBM agree. That's the *opposite* of what a multi-source
envelope is for — it should widen on real product disagreement, not anchor
to one product's flat baseline.

**Recommendation:** drop MOD16A2 v061 from the AET target until v061's
flat-on-CONUS+ behaviour is understood (possible causes: over-eager flag
masking, GF gap-fill homogenisation, v061 reprocess pending).
SSEBop + MWBM as the min/max pair is a smaller but **honest** bound.

→ **Decision today:** keep, drop, or defer?

---

## Recharge (RCH)

- **Sources:** Reitz 2017 (`total_recharge`) + WaterGAP 2.2d (`qrdif`) +
  ERA5-Land (`ssro`, sub-surface runoff used as proxy)
- **Method:** per-source 0–1 normalised, then per-HRU min/max
- **Cadence:** annual (calibration window default 2000–2009)

The three sources measure *conceptually different fluxes* — empirical total
recharge, process-modelled diffuse recharge, sub-surface runoff. Absolute
magnitudes diverge by design; the 0–1 normalisation is what makes them
combinable. The optimisation targets relative year-to-year change.

![w:850](../figures/inspect_aggregated/recharge_normalized_comparison.png)

---

## Recharge — time series at four climate regimes

![w:900](../figures/inspect_aggregated/recharge_time_series.png)

Inter-annual variability is the signal the calibration target captures.
Olympic Peninsula vs Phoenix span ~5–10× absolute scale; relative phasing
across products drives the bound width.

---

## Soil moisture (SOM)

- **Sources:** MERRA-2 (`GWETTOP`) + NLDAS-2 NOAH + NLDAS-2 MOSAIC + NCEP/NCAR R1
- **Method:** per-source 0–1 normalised, then per-HRU min/max
- **Cadence:** monthly + annual

**Layer-depth caveat:** MERRA-2 is 0–5 cm; the others are 0–10 cm.
**Variable-meaning caveat:** MERRA-2 `GWETTOP` is plant-available wetness
(`(W − Wwilt) / (Wsat − Wwilt)`), *not* volumetric water content. NCEP/NCAR
`soilw_0_10cm` is VWC despite the file labelling its units `kg/m2`.

The 0–1 normalisation handles all of this gracefully — but don't try to
compare absolute values across the four.

![w:780](../figures/inspect_aggregated/soil_moisture_normalized_comparison.png)

---

## Soil moisture — distribution shape

![w:780](../figures/inspect_aggregated/soil_moisture_histogram.png)

Even after each source is *physically* in its own native units, the
*shape* of the per-HRU distribution should be broadly comparable. A
strongly bimodal histogram from one product but unimodal from another
typically signals a regional regime split (wet-coast vs continental) one
captures and the other smooths over.

---

## Snow-covered area (SCA)

- **Source:** MOD10C1 v061 (`Day_CMG_Snow_Cover` + `Day_CMG_Clear_Index` for confidence)
- **Method:** CI-thresholded SCA bounds (CI > 70% pre-aggregation)
- **Cadence:** daily

CI > 0.70 is applied **pre-aggregation** — non-linear, can't post-gate.
Without it, CONUS-mean for a typical day is dominated by flag codes
(237 = inland water, 239 = ocean, 250 = cloud-obscured water) and lands
near 100, which is physically meaningless.

![w:900](../figures/inspect_aggregated/snow_covered_area_normalized_comparison.png)

---

## SCA — open caveat

The TM 6-B10 report describes the calibration-target error bound as
"the daily SCA value and the associated confidence interval", but the
**exact formula remains unconfirmed** — `PRMSobjfun.f` (the source of
truth in the original toolchain) is not publicly available.

Currently we mask SCA where CI ≤ 0.70 and aggregate; the bounding
formula will need a documented assumption (or external reference) before
the SCA target builder is finalised.

This is an existing known gap, not a new finding from this work.
Source-of-truth: `CLAUDE.md` § Known Gaps → "SCA CI-bounds formula".

---

# Decisions to discuss

Two open questions for the group:

1. **(a)** Period of record per target group
2. **(b)** Datasets per target group

Both are tracked in `catalog/variables.yml` per target. Project
`config.yml` can override per-run, so the catalog default reflects the
*default* not the *only* choice.

---

## (a) Period of record — current defaults

| Target | Catalog default | TM 6-B10 reference | Driver |
|---|---|---|---|
| Runoff | `2000/2010` | (n/a — replacement source) | ERA5-Land + GLDAS-2.1 NOAH availability |
| AET | `2000/2010` | 2000/2010 | Matches report; MOD16A2 v006 retired, v061 from 2000 |
| Recharge | `2000/2009` (norm window) | 2000/2009 (body text) | Reitz 2017 covers 2000–2013, others longer |
| Soil moisture | `1982/2010` | 1982/2010 | MERRA-2 from 1980, NLDAS-2 from 1979, NCEP from 1948 |
| Snow-covered area | `2000/2010` | 2000/2010 | MOD10C1 from 2000 |

**Question:** do these still suit, or are shorter / longer / shifted
windows preferred for any target? (E.g., would extending RUN/AET to
2000–2020 change the calibration outcome materially?)

---

## (b) Datasets per target

8 sources documented, working, producing valid HRU-level magnitudes:
- **RUN:** ERA5-Land `ro`, GLDAS-2.1 NOAH `Qs_acc + Qsb_acc`
- **AET:** SSEBop, MWBM (ClimGrid), MOD16A2 v061 (← *open*)
- **RCH:** Reitz 2017, WaterGAP 2.2d, ERA5-Land `ssro`
- **SOM:** MERRA-2, NLDAS-NOAH, NLDAS-MOSAIC, NCEP/NCAR R1
- **SCA:** MOD10C1 v061

**Open question:** MOD16A2 v061 — keep / drop / defer? (See AET decision slide.)

**Out of scope today:** sources not currently aggregated. We can revisit if
the group has a specific addition in mind (e.g., another AET product, or
MWBM family extensions for RCH).

---

## Where things live (pointers for follow-up)

- **Catalog (source-of-truth):** `catalog/sources.yml`, `catalog/variables.yml`
- **Per-target recipes:** `docs/references/calibration-target-recipes.md`
- **Lessons learned:** `docs/references/lessons-learned.md`
  *(MOD16A2 finding, validation-cell limits, mask conventions, etc.)*
- **Inspection notebooks:** `notebooks/inspect_aggregated/inspect_aggregated_*.ipynb`
- **TM 6-B10 crib sheet:** `docs/references/tm6b10-summary.md`
- **CLAUDE.md:** project-level conventions, known gaps, dev workflow

Everything cited here has a path in the repo. After the meeting, the
decisions go into `catalog/variables.yml` (period / sources) and a
follow-up commit to `lessons-learned.md` recording the consensus.

---

# Discussion

(intentionally blank)
