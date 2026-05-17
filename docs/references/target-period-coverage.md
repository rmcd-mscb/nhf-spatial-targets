# Target period coverage

Per-target reference table of source coverage windows and the resulting
**maximum-overlap period** the target can be built over. The
project-level `<target>.period` config knob in each project's
`config.yml` controls the actual output time axis; this doc documents
what's feasible against the data on disk.

Per [CLAUDE.md](../../CLAUDE.md):

> Time windows are not fixed to the report. ... `period` fields in
> `catalog/variables.yml` reflect historical defaults, not hard
> constraints.

The intent of "max overlap" is: produce the longest target output that
keeps every source contributing at every timestep (so `n_sources` stays
at its maximum across the whole period). Beyond the intersection,
`n_sources` drops to a partial bound — still a valid target, just with
fewer source votes.

## Source coverage on disk (gfv2 datastore)

Coverage shown as `<year_first>–<year_last>`, drawn from the per-year
aggregated NCs under `<project>/data/aggregated/<source_key>/` at
`/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets/`
as of 2026-05-17.

| Source | Coverage | Files | Notes |
|---|---|---|---|
| era5_land | 1979–2025 | 47 | full Copernicus archive |
| gldas_noah_v21_monthly | 2000–2025 | 26 | GLDAS-2.1 starts 2000 |
| mwbm_climgrid | 1979–2020 | 42 | Wieczorek et al. 2024 ends 2020 — sets a hard ceiling for runoff + aet multi-source overlaps |
| mod16a2_v061 | 2000–2025 | 26 | MODIS Terra mission start |
| ssebop | 2000–2023 | 24 | NHGF STAC catalog current through 2023 |
| reitz2017 | 2000–2013 | 14 | Reitz et al. 2017 publication window |
| watergap22d | 1901–2016 | 116 | global; gfv2 excludes for recharge (coarse grid) |
| merra2 | 1980–2025 | 46 | MERRA-2 reanalysis start |
| ncep_ncar | 1979–2025 | 47 | gfv2 excludes for soil moisture (T62 coarse) |
| nldas_mosaic | 1979–2025 | 47 | |
| nldas_noah | 1979–2025 | 47 | |
| mod10c1_v061 | 2000–2025 | 26 | MODIS Terra mission start |
| daymet | 1980–2024 | 45 | ORNL distribution |
| snodas | 2004–2012 | 9 | **aggregation-side gap** — fetched 2004–2024 (21 yrs on disk) but aggregator only ran for 2004–2012; needs re-aggregation. Full archive starts 2003-09-30 |
| era5_land_sd | — | — | not yet aggregated (variable available in monthly NCs; aggregator needs the one-line `variables` tuple extension) |
| margulis_wus_sr | — | — | not yet aggregated; daily NCs 1985–2020 fetched; Oregon-only fabric_scope |

## Per-target max-overlap

For each target, the **max-overlap** is the intersection of its active
sources' coverage windows. "Active sources" is the per-target list in
`config.yml` after any per-project exclusions (e.g. gfv2 drops
`watergap22d` from recharge and `ncep_ncar` from soil_moisture
because their coarse grids cannot resolve intermountain-west altitude
gradients — see the inline comments in the gfv2 `config.yml`).

### Runoff (`targets/run.py`)

| Source | Coverage |
|---|---|
| era5_land | 1979–2025 |
| gldas_noah_v21_monthly | 2000–2025 |
| mwbm_climgrid | 1979–2020 |
| **Intersection** | **2000–2020** (21 years × 12 months = 252 monthly timesteps) |

**Ceiling:** mwbm_climgrid's 2020 end. Drop mwbm from `runoff.sources`
to extend past 2020 with a 2-source bound (n_sources=2 from 2021+).

### AET (`targets/aet.py`)

| Source | Coverage |
|---|---|
| mod16a2_v061 | 2000–2025 |
| ssebop | 2000–2023 |
| mwbm_climgrid | 1979–2020 |
| **Intersection** | **2000–2020** (21 years × 12 months = 252 monthly timesteps) |

**Ceiling:** mwbm_climgrid's 2020 end (same as runoff). Drop mwbm to
extend with a 2-source bound through 2023 (ssebop end), then a
1-source bound through 2025 (mod16a2 only).

### Recharge (`targets/rch.py`) — gfv2 sources

| Source | Coverage |
|---|---|
| reitz2017 | 2000–2013 |
| era5_land (`ssro`) | 1979–2025 |
| **Intersection** | **2000–2013** (14 years) |

`watergap22d` excluded for gfv2 (coarse-grid exclusion); if re-enabled
its window 1901–2016 would not change the intersection (still
reitz-capped at 2013).

**Ceiling:** reitz2017's 2013 end. Empirical regression product; no
updated release planned. Drop reitz to extend with a 1-source bound
(era5_land ssro alone), but the resulting target is then a noisy
single-source proxy, not a real cross-source consensus bound.

### Soil moisture (`targets/som.py`) — gfv2 sources

| Source | Coverage |
|---|---|
| merra2 (`GWETTOP`) | 1980–2025 |
| nldas_mosaic (`SoilM_0_10cm`) | 1979–2025 |
| nldas_noah (`SoilM_0_10cm`) | 1979–2025 |
| **Intersection** | **1980–2025** (46 years; 552 monthly + 46 annual timesteps) |

`ncep_ncar` excluded for gfv2 (coarse-grid exclusion); if re-enabled
its window 1979–2025 would not change the intersection (still
merra2-capped at 1980 start).

**Ceiling:** merra2's 1980 start. NLDAS sources start 1979 but
merra2 cuts that off by one year. Drop merra2 to extend back to 1979
with a 2-source bound; tradeoff is losing one LSM with very
different soil-moisture physics (GEOS land surface scheme vs
NOAH/MOSAIC).

### Snow-covered area (`targets/sca.py`)

| Source | Coverage |
|---|---|
| mod10c1_v061 | 2000–2025 |
| **Intersection** | **2000–2025** (26 years × 365 days ≈ 9498 daily timesteps) |

Single-source target — intersection equals coverage. Builder is
currently a stub (CI-bounds formula gap per [CLAUDE.md](../../CLAUDE.md)
Known Gaps); period extends easily once the placeholder formula
lands.

### Snow water equivalent (`targets/swe.py`) — pending

Two source aggregations not yet on disk; max-overlap deferred:

| Source | Coverage (when aggregated) |
|---|---|
| daymet | 1980–2024 |
| snodas | 2004–2012 (aggregated; **2004–2024 if re-aggregated against full fetched archive**) |
| era5_land sd | 1979–2025 (variable on disk; needs aggregator's `variables` tuple extension) |
| margulis_wus_sr | 1985–2020 (daily NCs on disk; needs aggregator + fabric_scope filter) |
| **Intersection (non-OR fabric, post-re-aggregation)** | **2004–2024** (21 years) |
| **Intersection (OR fabric)** | **2004–2020** (margulis caps at 2020) |

**Outstanding work for SWE:**
- Re-aggregate snodas for 2013–2024 (21 fetched yearly archives that
  weren't covered by the initial aggregation pass).
- Implement era5_land sd + margulis aggregators (tracked under
  umbrella issue [#101](https://github.com/rmcd-mscb/nhf-spatial-targets/issues/101)).
- Build `targets/swe.py` with fabric_scope filter (gfv2 is non-OR →
  3-source bound).

## normalize_period vs period (recharge, soil_moisture)

For the two `normalized_minmax` targets, `<target>.normalize_period`
controls the window over which each source's min/max are computed
(the "calibration climatology"). When **`normalize_period == period`**
the output bound stays strictly in `[0, 1]`. When the two differ
(e.g. `period: 2000-2020` with `normalize_period: 2000-2009` to
match a TM 6-B10 climatology while extending the output), values
outside the normalize window may produce normalized values `< 0` or
`> 1` — by design, so out-of-climatology years are visibly
out-of-range, not silently re-scaled. See
[`normalize_0_1_over_window`](../../src/nhf_spatial_targets/normalize/methods.py)
docstrings for the formal contract.

Current gfv2 defaults set `normalize_period == period` for both rch
and som (`normalize_period: null` on som falls back to `period`),
giving a strictly-[0, 1] bound.

## Cross-references

- [`catalog/sources.yml`](../../catalog/sources.yml) — per-source
  `period` and `notes` fields. The `period` field there is the
  source's catalog-declared coverage (e.g. `"2000/present"`); the
  on-disk year-range may be shorter depending on the project's fetch
  / aggregate history.
- [`catalog/variables.yml`](../../catalog/variables.yml) — per-target
  `period` field reflects historical TM 6-B10 defaults; not read by
  the build code.
- [`docs/references/calibration-target-recipes.md`](calibration-target-recipes.md)
  — per-target unit conversions and combination semantics.
- Inline comments in the gfv2 `config.yml` document the per-project
  source exclusions (coarse-grid rationale for dropping `watergap22d`
  / `ncep_ncar` on mountainous fabrics) and the per-target period
  choices.
