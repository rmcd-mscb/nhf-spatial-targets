# Target ↔ model-output mapping

Each calibration target in this repo is one half of a comparison. The other
half is a simulated PRMS / pywatershed variable. This page states, per target,
**what the target's sources actually measure**, **which simulated variable it
should be compared with**, and **what goes wrong when the two don't mean the
same thing**. Read it before changing which model output a target is scored
against, or before widening a bound to make a conflict go away.

Sources for the model side:

- pywatershed variable metadata —
  [`pywatershed/static/metadata/variables.yaml`](https://github.com/EC-USGS/pywatershed/blob/develop/pywatershed/static/metadata/variables.yaml)
  (descriptions, units and dims quoted below).
- pywatershed channel inputs —
  [`hydrology/prms_channel.py`](https://github.com/EC-USGS/pywatershed/blob/develop/pywatershed/hydrology/prms_channel.py)
  `get_inputs()` returns `sroff_vol`, `ssres_flow_vol`, `gwres_flow_vol`: the
  three HRU fluxes that reach the stream.
- The byHRU objective function —
  [`PRMSobjfun.f90`](PRMSobjfun.f90), summarised in
  [prmsobjfun-summary.md](prmsobjfun-summary.md). It reads the five simulated
  series from the PRMS animation file in the column order
  `run, aet, rch, sca, som` (`read_ani_file`). **Which model variable fills
  each column is set in the PRMS control file**, not in the objective
  function — so the mapping below is a convention the calibration setup has to
  follow, not something the code enforces.

## Summary

| Target | Compare with | Model units | What the target's sources measure | How PRMSobjfun treats the simulated series |
|---|---|---|---|---|
| runoff | `hru_outflow` | cfs | **total** runoff (surface + subsurface) | daily → monthly mean, then scaled by the baseline's per-HRU max (same scaling as the bounds) |
| aet | `hru_actet` | inches/day | actual ET | daily → monthly mean |
| recharge | `recharge` (= `soil_to_gw + ssr_to_gw`) | inches | recharge / deep drainage proxies, each normalized 0–1 | daily → annual sum, then normalized 0–1 over its own window |
| soil_moisture | `soil_rechr` | inches | soil water in layers of differing depth, each normalized 0–1 | monthly accumulation, normalized per calendar month |
| snow_covered_area | `snowcov_area` | decimal fraction | fraction of the HRU snow-covered | daily |
| snow_water_equivalent | `pkwater_equiv` | inches | SWE | *(not in PRMSobjfun — an extension beyond TM 6-B10)* |

The catalog field `prms_variable` (`catalog/variables.yml`, `defaults.py`,
`init_run.py:_CONFIG_TEMPLATE`) records the same mapping. It is metadata only;
no pipeline code reads it.

## Runoff — compare with `hru_outflow`, not `sroff`

**What the sources measure.** All three runoff sources are *total* runoff:

| Source | Variable | Components |
|---|---|---|
| ERA5-Land | `ro` | `sro` (surface) + `ssro` (sub-surface) |
| GLDAS-2.1 NOAH | `runoff_total` | `Qs_acc` (surface) + `Qsb_acc` (baseflow-groundwater) |
| MWBM ClimGrid | `runoff` | total runoff from the monthly water balance |

**The matching model variable** is `hru_outflow` — "Total flow leaving each
HRU", in cfs, per HRU. It is `sroff + ssres_flow + gwres_flow` expressed as a
flow over the HRU area: exactly the three fluxes pywatershed's channel takes as
lateral inflow.

- `sroff` — "Surface runoff to the stream network" — is only the first term.
  Comparing it with a total-runoff target is a definitional mismatch.
- `basin_cfs` — "Streamflow leaving the basin through the stream network" — has
  dims `one`: it is a single basin-wide value, not per HRU. TM 6-B10's table
  lists `basin_cfs` for RUN; the per-HRU comparison PRMSobjfun actually makes
  is against a per-HRU series, i.e. `hru_outflow`.

### Worked example: Oregon (2000–2020)

Surface share of total runoff, computed from the aggregated source NCs
(`data/aggregated/era5_land`, `data/aggregated/gldas_noah_v21_monthly`), as the
per-HRU ratio of the 2000–2020 means:

| Source | median HRU | 10th–90th percentile |
|---|---|---|
| ERA5-Land `sro / ro` | 13 % | 8–46 % |
| GLDAS `Qs_acc / runoff_total` | 15 % | 6–47 % |

So about **85 % of the target is sub-surface** at the median HRU. Fitting
`sroff` to this target would push simulated surface runoff up roughly 5–7× —
the monthly numbers can line up while the model's water-balance partitioning
becomes wrong. If `sroff` happens to *fit* a basin, suspect coincidental
magnitude, not a better definition.

The bounds are already wide: `(upper − lower) / ensemble_mean` has a median of
~1.5 (10th–90th percentile ~0.6–2.4) across Oregon (HRU, month) cells, and the
Oregon-mean member values differ substantially (GLDAS ≈ 3.8, ERA5-Land ≈ 6.1,
MWBM ≈ 7.1 cfs per HRU). A large prior-data conflict *despite* that width is a
systematic disagreement worth diagnosing, not something to widen away.

### Diagnosing a runoff prior-data conflict

1. **Which bound, and when?** Is `hru_outflow` above the upper bound all year
   (magnitude), or only in some seasons (timing)?
2. **Annual vs monthly.** The land-surface models drain the subsurface quickly
   and have no deep groundwater storage; PRMS delays water through
   `ssres_flow` and `gwres_flow`. If annual totals agree but months don't, the
   conflict is timing — which argues for a lagged or seasonal comparison, not
   for switching to `sroff`.
3. **Groundwater sink.** `gwres_sink` ("underflow or flow to deep aquifers …
   does not flow to the stream network") is not part of `hru_outflow`, while
   the sources' subsurface runoff has nowhere else to go. HRUs with a
   significant sink will read low against the target.
4. **Check against a gauge.** Convert the basin's observed flow to depth and
   compare with `hru_outflow` and each member. Gauge agrees with the model →
   the target is biased low there (look at the lowest member first; in Oregon
   that is usually GLDAS). Gauge agrees with the target → the model is making
   too much water; calibrate that rather than working around it.
5. **Component targets are available if wanted.** The surface/subsurface
   splits (`sro` / `ssro`, `Qs_acc` / `Qsb_acc`) are already in the aggregated
   NCs. A deliberately built surface-runoff target would be a legitimate
   `sroff` comparison; relabelling the total-runoff target is not.

The per-source members are in the target NC (one variable per source key,
see `member_keys`), so steps 1–4 need no rebuild:

```python
import xarray as xr

ds = xr.open_dataset("<project>/targets/runoff_targets.nc")
basin = ds.sel(hru_id=[...])            # the test basin's HRUs
members = basin[ds.attrs["member_keys"].split(",")]
annual = members.resample(time="YS").mean()   # cfs, annual mean
```

## AET — `hru_actet`

Sources (MOD16A2 v061, SSEBop, MWBM ClimGrid) are actual ET, harmonized to
mm/month and written as **inches/day** — the same units as `hru_actet`, which
PRMSobjfun averages from daily to monthly means. No definitional gap; the
known issue is magnitude spread between remote-sensing and water-balance
products, which is what the min/max bound represents.

## Recharge — `recharge`

`recharge` is "Recharge to the associated GWR as sum of `soil_to_gw` and
`ssr_to_gw`". The sources (Reitz 2017, WaterGAP 2.2d diffuse recharge, ERA5-Land
`ssro` as a proxy) disagree in magnitude for conceptual reasons, so **both
sides are normalized 0–1** — the target per source over its window, the
simulated annual sum over its own min/max in `calcRCH`. Only relative
year-to-year change is compared; absolute magnitude is not.

Note ERA5-Land `ssro` appears in *both* the runoff target (inside `ro`) and the
recharge target. That's deliberate — it is the land-surface model's drainage
term — but it means the two targets are not independent evidence.

## Soil moisture — `soil_rechr`

`soil_rechr` is the **upper** (recharge-zone) part of the capillary reservoir,
not the whole soil column (`soil_moist`, or `soil_moist_tot` including gravity
storage). The sources (MERRA-2, NCEP/NCAR, NLDAS Mosaic, NLDAS Noah) report
layers of differing depths, so magnitudes are not expected to match; both sides
are normalized per calendar month and only relative change is compared.
Switching the model variable to `soil_moist` changes the dynamics being scored
(deeper storage responds more slowly) and should be a deliberate decision.

## Snow-covered area — `snowcov_area`

"Snow-covered area on each HRU prior to melt and sublimation unless snowpack
depleted", decimal fraction, daily. The target's bounds are a MOD10C1
confidence interval combined with ua_swe's depth-derived snow-covered fraction
(see `targets/sca.py`), not a member min/max. PRMSobjfun only uses MOD10C1
values with CI > 70 % and sets July/August to zero.

## Snow water equivalent — `pkwater_equiv`

"Snowpack water equivalent on each HRU", inches, daily. Not scored by
PRMSobjfun; it extends the TM 6-B10 set. Sources are SWE directly (Daymet,
SNODAS, ERA5-Land `sd`, Margulis WUS-SR, UA SWE), so there is no definitional
gap.
