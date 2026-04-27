# Reitz 2017 Aggregator — Design

**Date:** 2026-04-27
**Status:** Approved (pre-implementation)

## Goal

Add the missing aggregator for the Reitz 2017 recharge dataset so the recharge calibration target has all three sources (Reitz `total_recharge`, WaterGAP 2.2d `qrdif`, ERA5-Land `ssro`) end-to-end aggregable. Closes the "Reitz aggregator must be built" forward-looking note from PR #72 and unblocks the recharge inspection notebook to load Reitz as a real source instead of skip-with-reason.

## Non-goals

- Re-architecting the Reitz fetch module. The existing `src/nhf_spatial_targets/fetch/reitz2017.py` produces a single multi-year consolidated NC (`reitz2017_consolidated.nc`) and that's fine — the aggregation driver's `enumerate_years` handles multi-year-in-one-file inputs natively (this is also how WaterGAP 2.2d works).
- Changing the recharge target builder (`src/nhf_spatial_targets/targets/rch.py`). Still a stub; landing this aggregator is a prerequisite, not the same task.
- Touching `inspect_consolidated_recharge.ipynb` substantively — the consolidated-side notebook works against the existing fetch and predates this aggregator.

## Key design decisions

1. **Mirror the WaterGAP 2.2d adapter pattern.** WaterGAP is also an annual-or-monthly recharge source consolidated into a single multi-year NC. The aggregator there is a 30-line `SourceAdapter` declaration. Reitz follows the same shape with three differences (variables, source CRS, comment).
2. **Aggregate both `total_recharge` and `eff_recharge`.** The recharge target builder currently consumes only `total_recharge`, but the Reitz catalog block declares both with proper CF metadata, the consolidated NC carries both, and per-variable weight reuse means the marginal cost of aggregating both is one extra `AggGen` call per spatial batch. Future-proof and aligned with the catalog.
3. **Declare `source_crs="EPSG:4269"` (NAD83 geographic).** Reitz GeoTIFFs and the consolidated NC are NAD83 geographic, not WGS84. The `SourceAdapter` default of `"EPSG:4326"` is wrong for Reitz — gdptools needs the actual CRS to compute weights correctly. Reitz is the first non-EPSG:4326 source in the project.
4. **No fetch-side re-projection.** Re-projecting Reitz to EPSG:4326 at fetch time was considered (so the aggregator could use the default CRS) and rejected: NAD83 ↔ WGS84 differs by ~1 m in CONUS for negligible spatial benefit, while a re-projection invalidates every populated datastore.
5. **Unblock the recharge inspection notebook in the same PR.** Drop the "may not yet exist" caveats from cells 1 and 2 of `inspect_aggregated_recharge.ipynb` so the narrative is consistent once the aggregator lands.

## Layout

```
src/nhf_spatial_targets/
  aggregate/
    reitz2017.py                          # NEW

src/nhf_spatial_targets/cli.py            # MODIFIED: import + agg-reitz2017 cmd + agg-all entry

tests/
  test_aggregate_reitz2017.py             # NEW: adapter sanity test
  test_aggregate_integration.py           # MODIFIED: add Reitz end-to-end test

notebooks/inspect_aggregated/
  inspect_aggregated_recharge.ipynb       # MODIFIED: drop "may not yet exist" caveats
```

## `aggregate/reitz2017.py`

```python
"""Reitz 2017 annual recharge adapter (total_recharge + eff_recharge)."""

from __future__ import annotations

from pathlib import Path

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="reitz2017",
    output_name="reitz2017_agg.nc",
    variables=("total_recharge", "eff_recharge"),
    source_crs="EPSG:4269",   # NAD83 geographic — Reitz GeoTIFFs preserve this
)


def aggregate_reitz2017(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> None:
    aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
```

The default `files_glob="*_consolidated.nc"` matches `reitz2017_consolidated.nc` produced by `fetch/reitz2017.py` — no override needed.

## Driver behaviour at run time

- `aggregate_source` enumerates years from `reitz2017_consolidated.nc` via `_driver.enumerate_years` — finds 14 years (2000–2013), each mapping to the same file.
- Per year, `aggregate_year` slices to `("YYYY-01-01", "YYYY-12-31")`, runs `WeightGen` once per spatial batch (cached at `<project>/weights/reitz2017_batch<N>.csv`) and `AggGen` per declared variable, writes `<project>/data/aggregated/reitz2017/reitz2017_<YYYY>_agg.nc`.
- Annual mid-year (July 1) timestamps from the source survive: each per-year output has one timestep at the source's mid-year date.
- Lat-descending source coords are handled by xarray slice semantics — no per-source code needed (WaterGAP is also lat-descending and works through the same driver).

## CLI wiring

In `src/nhf_spatial_targets/cli.py`:

1. Add import alongside the other `aggregate_*` imports:
   ```python
   from nhf_spatial_targets.aggregate.reitz2017 import aggregate_reitz2017
   ```

2. New command:
   ```python
   @agg_app.command(name="reitz2017")
   def agg_reitz2017_cmd(
       workdir: Annotated[Path, Parameter(name=["--project-dir"])],
       batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
   ):
       """Aggregate Reitz 2017 annual recharge to HRU polygons."""
       _run_tier_agg(aggregate_reitz2017, "Reitz 2017", workdir, batch_size)
   ```

3. Append `("reitz2017", aggregate_reitz2017)` to the `sources` list in `agg_all_cmd`.

## Tests

**Adapter sanity** in `tests/test_aggregate_reitz2017.py`:

```python
"""Tests for Reitz 2017 aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets.aggregate.reitz2017 import ADAPTER


def test_adapter_declares_recharge_variables():
    assert ADAPTER.source_key == "reitz2017"
    assert ADAPTER.output_name == "reitz2017_agg.nc"
    assert ADAPTER.variables == ("total_recharge", "eff_recharge")
    assert ADAPTER.source_crs == "EPSG:4269"
```

**Integration test** in `tests/test_aggregate_integration.py` (`@pytest.mark.integration`, runs via `pixi run -e dev test-integration`):

- Build a small synthetic Reitz consolidated NC in `tmp_path`: 50×50 NAD83-geographic grid covering a CONUS bbox (e.g. lon −100 to −95, lat 39 to 41), 2 years (2000, 2001), both `total_recharge` and `eff_recharge` filled with deterministic values (e.g. linearly varying west-to-east). Apply CF metadata via the same shared helper the fetch uses.
- Build a 4-HRU fabric `gpd.GeoDataFrame` covering the bbox; write to `tmp_path/fabric.gpkg`.
- Initialise a project skeleton at `tmp_path/project` (mirrors `test_aggregate_watergap22d_end_to_end`'s harness — copy the helper, don't reinvent).
- Place the synthetic NC at `tmp_path/project/data/raw/reitz2017/reitz2017_consolidated.nc`.
- Call `aggregate_reitz2017(fabric_path, id_col, project_dir)`.
- Assert:
  - Files `<project>/data/aggregated/reitz2017/reitz2017_2000_agg.nc` and `..._2001_agg.nc` both exist.
  - Both files contain both `total_recharge` and `eff_recharge`.
  - HRU dim equals the fabric's row count.
  - Time dim has 1 timestep per file at mid-year July 1.
  - Aggregated values are finite for every HRU (full coverage given the synthetic grid spans the fabric).
  - Per-HRU aggregated `total_recharge` lies between the source min and max for that HRU's footprint (sanity bound on the area-weighted mean).

## Notebook tweak

`notebooks/inspect_aggregated/inspect_aggregated_recharge.ipynb`:

- **Cell 1 (intro):** drop the "**Aggregator may not yet exist** — this notebook skips Reitz with a clear message if its aggregated NCs are absent." sentence from the Reitz bullet. Keep the rest of the bullet (units, annual cadence).
- **Cell 2 (per-target conventions):** drop the "Reitz skipped-with-reason if no aggregated NCs are present." bullet. The generic `discover_aggregated` → skip path stays in cell 5 (handles any source's missing aggregations as a defensive measure), but we no longer flag it as the expected case for Reitz specifically.
- Cell IDs preserved (`--keep-id` is in the pre-commit config since PR #72).
- `inspect_consolidated_recharge.ipynb` is not modified — this PR doesn't change the consolidated-NC story for Reitz.

## Build order

1. `aggregate/reitz2017.py` + `tests/test_aggregate_reitz2017.py` first (TDD-flavoured: adapter sanity test → adapter file → green).
2. CLI wiring (import + `agg-reitz2017` cmd + `agg-all` entry).
3. Integration test in `tests/test_aggregate_integration.py`.
4. Notebook tweak.
5. Pre-PR quality gate: `pixi run -e dev fmt && lint && test`; `test-integration` separately if local fixtures cooperate.

## Open / verify before opening the PR

- **Confirm `apply_cf_metadata` writes axis attributes** on `time`/`lat`/`lon` for the Reitz consolidated NC. The driver's `_find_time_coord_name` requires `axis="T"` or `standard_name="time"`. WaterGAP works through the same code path, so this should already be correct, but worth a single Read of the consolidated NC during implementation.
- **End-to-end run on caldera** before merge. The user (rmcd) will run `nhf-targets agg reitz2017 --project-dir <gfv2-project>` and inspect the resulting per-year NCs to confirm CONUS-mean recharge ≈ 122 mm/yr at the HRU level.
