# Inspect-Aggregated Notebooks — Design

**Issue:** [#70 — inspect aggregated results](https://github.com/USGS-WMA-WMA/nhf-spatial-targets/issues/70)
**Date:** 2026-04-27
**Status:** Approved (pre-implementation)

## Goal

Build a parallel set of inspection notebooks at the **HRU-aggregated** level, mirroring the existing `inspect_consolidated_*` notebooks at the gridded-source level. The aggregated notebooks make HRU-level coverage gaps, unit conversions, and cross-source disagreements visible *after* spatial aggregation but *before* downstream NaN-fill / normalisation. They also serve as the live cross-check that HRU-area-weighted aggregated means agree with the gridded means within a few percent.

Five notebooks under a new `notebooks/inspect_aggregated/` folder:

- `inspect_aggregated_runoff.ipynb`
- `inspect_aggregated_aet.ipynb`
- `inspect_aggregated_recharge.ipynb`
- `inspect_aggregated_soil_moisture.ipynb`
- `inspect_aggregated_snow_covered_area.ipynb`

Plus a sibling `notebooks/inspect_aggregated/_helpers.py` carrying shared helpers (not packaged into `nhf_spatial_targets`).

## Non-goals

- Reorganising the existing `inspect_consolidated_*.ipynb` and `visualize_*.ipynb` into per-target folders. Tracked as a follow-up issue.
- Adding the `SAVE_FIGURES` convention to the existing consolidated notebooks. Tracked as the same follow-up issue.
- Building the Reitz 2017 aggregator. Recharge notebook ships with Reitz in scope but skipped-with-reason if its aggregator hasn't landed.
- Building target-side NaN fill (`normalize/`). Notebooks inspect the **unfilled** state; markdown captions point forward to the future fill step.
- Implementing target builders (`targets/aet.py`, `targets/rch.py`, `targets/som.py`, `targets/sca.py`). The notebooks will inform those builds; they don't depend on them.

## Key design decisions

1. **Five fully-parallel notebooks**, not one parameterised notebook. Cell-level structural parity with `inspect_consolidated_*` is the readability goal; abstraction defeats the purpose.
2. **Sibling helpers (`_helpers.py`) in a new folder**, not a packaged module. Notebook-shaping code does not need test-coverage gates and lives next to the notebooks it serves. Tests for the pure-Python helpers still land in `tests/test_inspect_helpers.py`.
3. **Two-tier file opening:** load only the `TARGET_YEAR` per-year NC for everything except the time-series cell; that cell uses `xr.open_mfdataset` over a configurable `TIME_SERIES_YEARS = range(2000, 2011)` window with immediate `.sel(hru=[...])` to keep memory bounded.
4. **Representative HRUs by lat/lon**, not by `nhm_id`. Each notebook declares `REPRESENTATIVE_POINTS = {"<regime name>": (lon, lat), ...}` near the top; helpers resolve to the containing HRU at runtime via `gpd.sjoin`. Works against any fabric.
5. **Live validation cell** comparing per-source HRU-area-weighted means to gridded means computed live from `<datastore>/<source_key>/<consolidated NC>`. Tolerance is reported, not asserted.
6. **Skip-with-reason everywhere.** Each cell that reads source data tolerates missing aggregations (e.g. Reitz pending) without crashing.
7. **Project paths from `config.yml`.** The notebook hardcodes a single caldera-default `PROJECT_DIR` with an "edit me" comment; `_helpers.load_project_paths` reads `<project>/config.yml` to derive `datastore_dir` and `fabric_cfg`. No env-var convention.
8. **Figure-save convention now.** Every plot cell ends with `save_figure(fig, f"{TARGET}_<view>")`, no-op unless `_helpers.SAVE_FIGURES = True`. `.gitignore` gains `docs/figures/`. This makes the figures available to a future `/frontend-slides` pass without committing image binaries.

## Layout

```
notebooks/
  inspect_aggregated/                          # NEW folder
    _helpers.py                                # NEW: shared helpers
    inspect_aggregated_runoff.ipynb            # NEW
    inspect_aggregated_aet.ipynb               # NEW
    inspect_aggregated_recharge.ipynb          # NEW
    inspect_aggregated_soil_moisture.ipynb     # NEW
    inspect_aggregated_snow_covered_area.ipynb # NEW

tests/
  test_inspect_helpers.py                      # NEW: unit tests for helper surface

.gitignore                                     # MODIFIED: add docs/figures/
```

## `_helpers.py` public surface

All helpers private except those listed below. Plotting functions accept an `Axes` and return nothing; data helpers are pure.

- `SAVE_FIGURES: bool` (module-level, default `False`) and `FIGURES_DIR: Path` (default `Path("docs/figures/inspect_aggregated/")`).
- `load_project_paths(project_dir: Path | None = None) -> tuple[Path, Path, dict]`
  Reads `<project>/config.yml`; returns `(project_dir, datastore_dir, fabric_cfg)`. `fabric_cfg` is the `{path, id_col, crs, ...}` block. If `project_dir is None`, defaults to a caldera path defined inside the helper.
- `load_fabric(fabric_cfg: dict) -> gpd.GeoDataFrame`
  `gpd.read_file(fabric_cfg["path"])`, indexed by `fabric_cfg["id_col"]`, retained in EPSG:4326 for plotting.
- `discover_aggregated(project_dir: Path, source_key: str) -> list[Path] | None`
  Globs `<project>/data/aggregated/<source_key>/<source_key>_*_agg.nc`. Returns `None` if the directory is empty or missing — caller prints the skip-with-reason.
- `open_year(project_dir: Path, source_key: str, year: int) -> xr.Dataset`
  Opens a single per-year NC, fully `.load()`ed and detached from the file (per `feedback_rioxarray_close.md`).
- `open_year_range(project_dir: Path, source_key: str, years: range) -> xr.Dataset`
  `xr.open_mfdataset` across the year window, used by the time-series cell only. Caller is responsible for `.sel(hru=[...]).load()` and closing.
- `select_month(da: xr.DataArray, year: int, month: int) -> xr.DataArray`
  Month-window slice helper from the recipes doc, lesson 2 (canonical SOM pattern).
- `plot_hru_choropleth(ax, fabric_gdf, values: pd.Series, *, vmin, vmax, cmap, title, units, nan_color="lightgrey")`
  Joins `values` to `fabric_gdf` on the index, plots polygons coloured by value; NaN HRUs filled with `nan_color` so coverage gaps are visually obvious.
- `area_weighted_mean(values: pd.Series, fabric_gdf) -> float`
  Computes Σ(v · A) / Σ(A) using fabric area in EPSG:5070 (matches the aggregator's `WEIGHT_GEN_CRS`).
- `nan_hru_count(values: pd.Series) -> int`
- `plot_nan_hrus(ax, fabric_gdf, values: pd.Series, *, title)`
  Coverage diagnostic map — boolean choropleth of NaN vs non-NaN HRUs.
- `lookup_hrus_by_points(fabric_gdf, points: dict[str, tuple[float, float]]) -> dict[str, hru_id]`
  Spatial-join lat/lon to the containing HRU. Returns `{regime_name: hru_id}`. Raises if any point falls outside the fabric.
- `unit_from_catalog(source_key: str, var: str) -> str`
  Wraps `nhf_spatial_targets.catalog.source(...)` (lesson 9: read units from the catalog, not from on-disk attrs).
- `save_figure(fig, name: str) -> None`
  No-op unless `SAVE_FIGURES` is True; otherwise writes `FIGURES_DIR / f"{name}.png"` at `dpi=150, bbox_inches="tight"`. Creates `FIGURES_DIR` on first write.

## Per-notebook cell structure

Numbered 1–15 here for clarity; the issue's prose grouping ("10 cells") covers the same content with markdown and code merged. Each notebook has these cells in order:

1. **Markdown — Intro.** Target name, source list, native units (read live from `unit_from_catalog`), pointer to `docs/references/calibration-target-recipes.md`.
2. **Markdown — Per-target conventions in this notebook.** Explicit per-target deviations (see "Per-target deviations" below). One bullet per deviation with a pointer to the recipes doc section.
3. **Code — Setup.** Imports, `from _helpers import *`, `PROJECT_DIR = Path("/caldera/...")` with edit-me comment, `(project_dir, datastore_dir, fabric_cfg) = load_project_paths(PROJECT_DIR)`, `fabric = load_fabric(fabric_cfg)`, `TARGET_TIME = "2000-01-15"`, `TARGET_YEAR = 2000`, `TIME_SERIES_YEARS = range(2000, 2011)`, `REPRESENTATIVE_POINTS = {...}`. Optional `_helpers.SAVE_FIGURES = True` toggle, commented out by default.
4. **Markdown — Load.** Brief explanation of skip-with-reason behaviour.
5. **Code — Load per source.** For each source: `discover_aggregated` → `open_year(project_dir, source_key, TARGET_YEAR)` or print skip-with-reason. Print variable list, time range, HRU count, **NaN-HRU count per source** (the new diagnostic vs the consolidated notebooks).
6. **Code — Dataset reps.** `display(ds)` per opened source.
7. **Code — Native-unit map at TARGET_TIME.** One `plot_hru_choropleth` panel per source, native units in title, NaN HRUs in light grey. Ends with `save_figure(fig, f"{TARGET}_native_units_map")`.
8. **Markdown — Normalized comparison.** Explains canonical-unit conversions per the recipes doc; names the reference source for the colour scale.
9. **Code — Normalized comparison map.** Apply per-target unit conversions inline (varies per target — see deviations); shared colour scale derived from the reference source (per recipes "natural CONUS reference footprints"); one panel per source. Ends with `save_figure(fig, f"{TARGET}_normalized_comparison")`.
10. **Code — Cross-source HRU-level histograms.** Side-by-side or overlaid densities on a shared x-axis, one trace per source, post-conversion units. Ends with `save_figure(fig, f"{TARGET}_histogram")`.
11. **Code — Time series at representative HRUs.** `lookup_hrus_by_points(fabric, REPRESENTATIVE_POINTS)` → for each source, `open_year_range(..., TIME_SERIES_YEARS).sel(hru=[...]).load()` → 2×2 subplot grid, all sources overlaid per panel, post-conversion units. Ends with `save_figure(fig, f"{TARGET}_time_series")`.
12. **Code — Coverage diagnostic.** `nan_hru_count` printed per source; one `plot_nan_hrus` panel per source coloured by NaN/non-NaN. Markdown caption notes these will be NN-filled in `normalize/` (forward-looking; module not yet implemented). Ends with `save_figure(fig, f"{TARGET}_coverage")`.
13. **Code — Validation.** Per source: HRU-area-weighted mean for `TARGET_TIME` (post-conversion) vs gridded mean computed live from `<datastore>/<source_key>/<consolidated NC>`. Print a one-line table per source: source, aggregated mean, gridded mean, abs diff, % diff. Skip-with-reason if the consolidated NC is missing. No assertion — readers investigate `> 5%`.
14. **Markdown — Explanation.** Per-target discussion lifted from the matching `inspect_consolidated_*` notebook, adapted for the HRU view; calibration-target implication.
15. **Code — Cleanup.** Close all opened datasets per `feedback_rioxarray_close.md`.

## Per-target deviations from the template

Captured in cell 2 ("Per-target conventions") of each notebook and applied in cells 7 and 9.

- **RUN (`inspect_aggregated_runoff.ipynb`).** ERA5-Land `ro` × 1000 → mm/month; GLDAS NOAH `runoff_total` × 8 × `days_in_month` → mm/month (recipes §1, lesson 3). Reference source: ERA5-Land.
- **AET (`inspect_aggregated_aet.ipynb`).** SSEBop `et` native mm/month, no conversion. MOD16A2 `ET_500m` × `scale_factor=0.1` (recipes §2). Markdown call-out flags the open `scale_factor` question — the consolidated notebook spotted a 500× offset that almost certainly means the factor was not applied. Reference source: SSEBop.
- **RCH (`inspect_aggregated_recharge.ipynb`).** Cell 7 shows native-unit maps per source (Reitz annual mid-year, WaterGAP/ERA5-Land monthly). Cells 9–13 convert all three to **annual mm/year**: Reitz `total_recharge` × 1000; WaterGAP `qrdif` × `days_in_month × 86400` per month, sum 12 → mm/year; ERA5-Land `ssro` × 1000 per month, sum 12 → mm/year. Reitz skipped-with-reason if its aggregator hasn't landed. Reference source: ERA5-Land.
- **SOM (`inspect_aggregated_soil_moisture.ipynb`).** NLDAS NOAH `SoilM_0_10cm` ÷ 100 → VWC; NLDAS MOSAIC `SoilM_0_10cm` ÷ 100 → VWC; MERRA-2 `GWETTOP` passthrough (plant-available, NOT VWC); NCEP/NCAR `soilw_0_10cm` passthrough (already VWC despite mislabelled `kg/m2` units, recipes §4). Cross-source time alignment via `select_month`, not `time.sel(method="nearest")` (recipes §4 / lesson 2). Reference source: NLDAS NOAH.
- **SCA (`inspect_aggregated_snow_covered_area.ipynb`).** Cell 7 shows a **single-day raw map** (flag values 107/111/237/239/250/253/255 masked) and a **monthly-mean composite map** (CI > 0.70 filter via `Day_CMG_Clear_Index`) side by side. Cells 9–13 use the CI-filtered monthly composite. Note: MOD10C1 must be re-aggregated post-PR-#68 to pick up `Day_CMG_Clear_Index`; notebook still runs (with reduced content) if only `Day_CMG_Snow_Cover` is present. Reference source: the only source — MOD10C1.

## Validation, error handling, testing

**Validation cell behaviour.** No assertion. The cell prints a small table per source and lets the reader judge whether `% diff` is acceptable (issue's tolerance is "within a few percent"). The cell skips a source with a clear reason if the consolidated NC is missing.

**Skip-with-reason pattern, applied at every cell that reads source data.** The `opened` dict is built in cell 5 and reused; downstream cells iterate over `opened`, so a missing source naturally drops out of every panel/trace.

**Memory management.** Cell 5 (load) opens only the `TARGET_YEAR` per-year NC per source, `.load()`s it, `.close()`s the file. Cell 11 (time series) uses `open_year_range(...).sel(hru=[representative_ids]).load()` so only ~4 HRUs across `TIME_SERIES_YEARS` (default 11 years) land in memory. Cell 15 closes everything explicitly per `feedback_rioxarray_close.md`.

**Helper tests (`tests/test_inspect_helpers.py`).** Imports `_helpers` via `importlib.util.spec_from_file_location` so the helper module's notebooks-folder location does not constrain test discovery. Covers:

- `select_month` — start-of-month, end-of-month, mid-month inputs all return the right calendar month.
- `area_weighted_mean` — known-area fixture, hand-computed expected mean.
- `lookup_hrus_by_points` — point inside fabric returns the containing HRU id; point outside raises.
- `discover_aggregated` — returns sorted paths when the dir is populated; returns None when absent or empty (`tmp_path` fixture).
- `unit_from_catalog` — agrees with direct `catalog.source(...)` lookup for two sources.
- `save_figure` — no-op when `SAVE_FIGURES=False`; writes a PNG when True (uses `tmp_path` and a tiny `plt.figure()`).

Plotting helpers (`plot_hru_choropleth`, `plot_nan_hrus`) are excluded from unit tests; their visual correctness is covered by the manual end-to-end validation step.

**End-to-end validation (manual, pre-PR).**

1. Run all five notebooks against `gfv2-spatial-targets` on caldera.
2. Confirm clean VSCode rendering (no silently-dropped cells — bullet lists used in place of `| | A | B |`-style markdown tables, lesson 8).
3. Confirm the validation cell reports HRU-area-weighted means within a few percent of gridded means for every source.
4. Confirm NaN-HRU counts are small and concentrated near coastlines / source-grid edges.
5. Confirm the four representative HRUs show plausible seasonal cycles per target.
6. Regenerate cell IDs as UUIDs after authoring (per issue's "Lessons to apply").

## Build order

1. `_helpers.py` and `tests/test_inspect_helpers.py` first; tests pass.
2. `.gitignore` adds `docs/figures/`.
3. `inspect_aggregated_runoff.ipynb` — full template, validate end-to-end. RUN is the only target with a working builder, so its aggregation is the best-exercised path.
4. Clone the runoff structure into the other four; adapt per-target deviations.
5. Each notebook gets `git add`-ed only after a clean VSCode render and a passing validation cell.

## Lessons to carry through (all in the recipes doc)

- Read units from the catalog, not from the NC `attrs` (lesson 9). Every per-target deviation pulls units via `unit_from_catalog`.
- Use `select_month` rather than `time.sel(method="nearest")` for cross-source time alignment (lesson 2). SOM is the canonical case.
- Mask MOD10C1 flag values > 100 before any quantitative use; `Snow_Spatial_QA` is categorical 0–4, **not** the CI (recipes §5).
- For GLDAS, the `× 8 × days_in_month` conversion happens in the notebook *before* plotting (recipes §1 / lesson 3). The aggregated NC still carries `kg m⁻²` units.
- Avoid markdown tables with empty leading header columns (lesson 8). Use bullet lists.
- Regenerate cell IDs as UUIDs after editing (issue note).

## Open questions / forward-looking

- **Reitz 2017 aggregator** — required for the recharge notebook's third source. Notebook ships with skip-with-reason; aggregator is a separate issue.
- **MOD10C1 re-aggregation** — needed for `Day_CMG_Clear_Index`. SCA notebook ships with a graceful fallback (raw `Day_CMG_Snow_Cover` only) until the re-aggregation lands.
- **NCEP/NCAR aggregated NC stale `cf_units`** — values are correct; only metadata is wrong. Notebook reads units from the catalog (lesson 9) so this does not produce wrong plots; flagged for future re-aggregation.
- **MOD16A2 `scale_factor`** — open question called out in cell 2 markdown of the AET notebook; not blocking for this PR.
- **Frontend-slides follow-up** — the figure-save convention is in place; describing the repo via `/frontend-slides` is a separate task that consumes `docs/figures/inspect_aggregated/*.png` (and `docs/figures/inspect_consolidated/*.png` once the consolidated notebooks are updated in the reorganisation issue).
