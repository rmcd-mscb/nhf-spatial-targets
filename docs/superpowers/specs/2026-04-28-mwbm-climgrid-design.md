# MWBM (ClimGrid-driven) Source — Design

**Date:** 2026-04-28
**Status:** Pre-implementation (awaiting user review)

## Goal

Add the USGS Monthly Water Balance Model output driven by ClimGrid
(Wieczorek et al. 2024, doi:10.5066/P9QCLGKM, ScienceBase item
`64c948dbd34e70357a34c11e`) to the catalog, implement a fetch module
that downloads the single consolidated NetCDF, and implement an
aggregation module that produces per-year HRU NetCDFs for four
variables: `runoff`, `aet`, `soilstorage`, `swe`. Declare
`mwbm_climgrid` as a source for the `runoff` and `aet` calibration
targets in `catalog/variables.yml`, and update the four corresponding
inspection notebooks
(`inspect_consolidated_<runoff|aet>.ipynb` and
`inspect_aggregated_<runoff|aet>.ipynb`) so they include
`mwbm_climgrid` alongside existing sources. This restores an
MWBM-family source to the project that was retired with the original
NHM-MWBM (issue #41), but uses the modern ClimGrid-forced product
rather than the decommissioned Maurer/Daymet-forced one.

## Non-goals

- **No target-builder code changes.** `targets/run.py` and
  `targets/aet.py` are not modified. `targets/run.py` continues to
  produce 2-source bounds (ERA5-Land + GLDAS) until the runoff target
  is explicitly extended to 3 sources in a separate PR;
  `targets/aet.py` remains a `NotImplementedError` stub.
  `catalog/variables.yml` records the *intent* that `mwbm_climgrid`
  participates in both targets — actual builder wiring is deferred,
  same reasoning as the original spec scope: each target's
  source-set change deserves its own discussion + PR.
- **No consolidation step in fetch.** The publisher distributes a
  single CF-conformant NetCDF (`ClimGrid_WBM.nc`, ~7.5 GB, NetCDF-4,
  int16-packed). The fetch is a pure download.
- **No re-projection.** The file is on a regular EPSG:4326 grid at
  ~0.042° (2.5 arcmin); gdptools handles this directly.
- **No subsetting on download.** The dataset is already CONUS-only,
  matches every project's spatial extent, and is fabric-independent
  by design — the datastore is shared across projects.
- **No soil-moisture / SCA notebook changes.** `soilstorage` and
  `swe` are aggregated to disk but not declared in `variables.yml`
  for the SOM or SCA targets and not surfaced in those notebooks —
  same deferred-target-decision rationale.

## Key design decisions

1. **Source key: `mwbm_climgrid`.** Disambiguates from the retired
   `mwbm` key (Bock et al., NHM-MWBM, decommissioned with #41). Reads
   "MWBM forced by ClimGrid", which matches how TM 6-B10 and the
   project codebase already refer to the MWBM family.

2. **Variables aggregated: `runoff`, `aet`, `soilstorage`, `swe`.**
   The file contains 9 variables; we keep only those with a plausible
   future calibration-target consumer. `pet` and the climate drivers
   (`tmean`, `prcp`, `rain`, `snow`) are documented in the catalog as
   *available in the source file* but are not aggregated. Adding any of
   them later is a one-line catalog edit + a re-run of
   `agg mwbm-climgrid`.

3. **Catalog `period: "1900/2020"`.** The publisher's stated coverage
   is 1895-01 → 2020-12, but the metadata explicitly notes that
   1895–1899 should be discarded as arbitrary spinup. The catalog
   declares the *usable* extent; consumers read this field to validate
   project time-window requests.

4. **Aggregator processes the full file; 1895–1899 spinup is not
   filtered out.** The shared aggregation driver (`enumerate_years`)
   has no year-filter mechanism today, and the file ships with
   1895-01 → 2020-12 inclusive. We accept ~5 years (~few hundred MB,
   a few minutes of compute) of "wasted" spinup-era aggregation
   rather than expand the driver/adapter contract for a single
   source. The catalog's `period: "1900/2020"` is the *contract*
   target builders honor when they later narrow their time windows;
   the publisher's spinup-discard guidance is documented in the
   catalog `notes` and surfaces wherever someone reads the source
   description. The recommended starting window for any project
   consuming this source is **1979-01 → 2020-12** (modern reanalysis
   era, longest overlap with other current sources), but that
   recommendation lives in target-builder discussions, not in this
   aggregator. If a second source ever needs publisher-driven year
   filtering, extending `SourceAdapter` with an optional
   `year_range: tuple[int, int]` field is the natural follow-up.

5. **Use `sciencebasepy` for download, not the `managerRequestDownload`
   URL.** `sciencebasepy.SbSession` is already a project dependency
   (`fetch/reitz2017.py` uses it), is the USGS-supported entry point,
   and resolves `downloadUri` from the SB API rather than baking an
   opaque S3-redirect URL into the codebase. If ScienceBase reorganizes
   storage, we don't break.

6. **Persist sha256 + size in `manifest.json`.** Cost: one extra
   ~30-second hash pass after a one-time download. Payoff: silent
   datastore corruption is detectable on every subsequent fetch
   (idempotency check skips download only if file size *and* sha256
   match the manifest). No other source in the project hashes its
   download — `mwbm_climgrid` is the largest single download we'll
   ever do (a ~7.5 GB file vs the next largest, NLDAS at ~few hundred
   MB), and silent bit-rot on a 7.5 GB blob held for years is the
   regime where hashing actually pays off.

7. **Aggregation uses the standard `SourceAdapter` / `aggregate_source`
   pattern.** This matches `watergap22d` (also a single-file,
   multi-year monthly source) and `reitz2017` (single-file, multi-year
   annual source) — a ~30-line declarative module. Per-year output
   layout follows PR #51/#55/#56:
   `<project>/data/aggregated/mwbm_climgrid/<YYYY>/mwbm_climgrid_<YYYY>.nc`.
   One weight cache at `<project>/weights/mwbm_climgrid_<fabric>.parquet`
   covers all four variables (they share one grid).

8. **`files_glob="ClimGrid_WBM.nc"` (publisher filename, not a renamed
   `*_consolidated.nc`).** Preserves the publisher-issued file
   identifier in the datastore, matching `watergap22d`'s
   `files_glob="*_cf.nc"` precedent for sources that don't follow the
   default `*_consolidated.nc` naming.

9. **Catalog placement: new `# INTEGRATED WATER BALANCE` section.**
   The source spans four targets (runoff, AET, SOM, SCA-via-SWE);
   filing it under any single target's heading would mislead future
   readers. A dedicated section signals the multi-target role
   honestly.

## Layout

```
catalog/
  sources.yml                                # MODIFIED: new mwbm_climgrid entry
  variables.yml                              # MODIFIED: mwbm_climgrid added to
                                             #          runoff + aet source lists

src/nhf_spatial_targets/
  fetch/
    mwbm_climgrid.py                         # NEW
  aggregate/
    mwbm_climgrid.py                         # NEW
  cli.py                                     # MODIFIED: fetch + agg subcommands,
                                             #          fetch-all + agg-all entries

notebooks/
  inspect_consolidated_runoff.ipynb          # MODIFIED: add mwbm_climgrid section
  inspect_consolidated_aet.ipynb             # MODIFIED: add mwbm_climgrid section
  inspect_aggregated/
    inspect_aggregated_runoff.ipynb          # MODIFIED: add mwbm_climgrid to SOURCES
    inspect_aggregated_aet.ipynb             # MODIFIED: add mwbm_climgrid to SOURCES

tests/
  test_catalog.py                            # MODIFIED: variables.yml assertions
  test_fetch_mwbm_climgrid.py                # NEW
  test_aggregate_mwbm_climgrid.py            # NEW
  test_aggregate_integration.py              # MODIFIED: add mwbm_climgrid case
```

## Catalog entry (`catalog/sources.yml`)

A new section header is added after the recharge block, and the
entry below is appended:

```yaml
  # ---------------------------------------------------------------------------
  # INTEGRATED WATER BALANCE
  # ---------------------------------------------------------------------------
  mwbm_climgrid:
    name: USGS Monthly Water Balance Model (ClimGrid-forced, 2024)
    description: >
      USGS Monthly Water Balance Model (McCabe & Wolock 2011) outputs
      forced by NOAA ClimGrid temperature and precipitation, published
      by Wieczorek et al. (2024). Successor to the retired NHM-MWBM
      (Bock et al. 2017) decommissioned in issue #41. Distributed as a
      single CF-conformant NetCDF covering 1895-01 to 2020-12 monthly,
      CONUS at 2.5 arcminute (~0.042°). The 1895–1899 period is
      treated as arbitrary spinup and excluded from `period`.

      Available in the source file but not aggregated by default:
      `pet`, `tmean`, `prcp`, `rain`, `snow`. Add to the `variables:`
      block below and re-run `agg mwbm-climgrid` if needed.
    citations:
      - "Wieczorek, M.E., Signell, R.P., McCabe, G.J., Wolock, D.M.,
         and Norton, P.A., 2024, doi:10.5066/P9QCLGKM"
      - "McCabe, G.J., and Wolock, D.M., 2011 (model)"
    doi: "10.5066/P9QCLGKM"
    access:
      type: sciencebase
      item_id: "64c948dbd34e70357a34c11e"
      url: https://www.sciencebase.gov/catalog/item/64c948dbd34e70357a34c11e
      filename: ClimGrid_WBM.nc
      notes: >
        Single ~7.5 GB NetCDF-4 file, int16-packed with scale_factor /
        add_offset (xarray decodes automatically on open). Downloaded
        via sciencebasepy.SbSession; integrity verified by size + sha256
        recorded in manifest.json. CONUS bounding box; no spatial
        subsetting at fetch time.
    variables:
      - name: runoff
        long_name: streamflow per unit area (MWBM)
        cf_units: "mm"
        cell_methods: "time: sum"
      - name: aet
        long_name: actual evapotranspiration (MWBM)
        cf_units: "mm"
        cell_methods: "time: sum"
      - name: soilstorage
        long_name: liquid water content of soil layer (MWBM)
        cf_units: "mm"
        cell_methods: "time: point"
      - name: swe
        long_name: liquid water equivalent of snowpack (MWBM)
        cf_units: "mm"
        cell_methods: "time: point"
    time_step: monthly
    period: "1900/2020"
    spatial_extent: CONUS
    spatial_resolution: 2.5 arcmin (~0.042 degree)
    units: mm (tmean: degC; not aggregated)
    license: public domain (USGS)
    status: current
```

`cell_methods` reflect the underlying physics: `runoff` and `aet` are
monthly accumulations (sum), `soilstorage` and `swe` are end-of-month
state (point) — consistent with MWBM-family conventions (Bock et al.
2017). Implementation verifies the actual `cell_methods` stored in
the source NetCDF on first open and raises if the file's encoding
contradicts the catalog declaration; this catches publisher-side
metadata changes before they corrupt downstream targets.

## `fetch/mwbm_climgrid.py`

Modeled on `fetch/reitz2017.py` (which already uses `sciencebasepy`)
with the consolidation logic stripped out, since the publisher file is
already a single CF-1.6 NetCDF.

Public API:

```python
def fetch_mwbm_climgrid(workdir: Path, period: str) -> dict:
    """Download ClimGrid_WBM.nc from ScienceBase to <datastore>/mwbm_climgrid/.

    Idempotent: if the file already exists at the destination AND its
    size + sha256 match the values recorded in manifest.json, the
    download is skipped. If the manifest has no record yet but the
    file is present, the file is hashed and validated against the
    publisher-reported size before being trusted.

    Returns a provenance dict suitable for manifest.json.
    """
```

Key behaviors:

- Reads `access.item_id` and `access.filename` from the catalog. Looks
  up the file via `SbSession.get_item_file_info` and matches by
  `name == "ClimGrid_WBM.nc"`. Errors clearly if the file is missing
  or renamed upstream.
- Streams the download to a `.tmp` path, then atomic-renames on
  successful close. On any exception, deletes the partial.
- Computes sha256 in 8 MB chunks during download (no second-pass read
  of the 7.5 GB file).
- Validates expected variables (`runoff`, `aet`, `soilstorage`, `swe`)
  and the EPSG:4326 grid_mapping by opening the NetCDF lazily after
  download. No data load. Closes the dataset before returning.
- Writes manifest.json provenance under `manifest["sources"]["mwbm_climgrid"]`:
  `source_key`, `access_url`, `doi`, `license`, `period`, `variables`,
  and `file: {path, size_bytes, sha256, downloaded_utc}`.
- Period validation: rejects requests outside `1900/2020`.

The fetch ignores the `period` argument for the actual download
(the file is what it is), but uses it to validate the project's
intended use and to record it in the manifest entry.

## `aggregate/mwbm_climgrid.py`

```python
"""USGS MWBM (ClimGrid-forced) monthly aggregator: runoff, aet, soilstorage, swe."""

from __future__ import annotations

from pathlib import Path

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source


ADAPTER = SourceAdapter(
    source_key="mwbm_climgrid",
    output_name="mwbm_climgrid_agg.nc",  # unused: per-year layout writes to subdirs
    variables=("runoff", "aet", "soilstorage", "swe"),
    files_glob="ClimGrid_WBM.nc",
)


def aggregate_mwbm_climgrid(
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

Output: per-year NetCDFs at
`<project>/data/aggregated/mwbm_climgrid/<YYYY>/mwbm_climgrid_<YYYY>.nc`,
one HRU dimension, four variables, monthly time axis (12 timesteps
per year except possibly the first/last partial year if narrowed).
Driver-derived global attributes record the source key, period, and
fabric metadata. Weight cache at
`<project>/weights/mwbm_climgrid_<fabric>.parquet` is written on the
first run of any variable and reused for the others.

## `catalog/variables.yml` updates

Both target-variable blocks gain `mwbm_climgrid` in their `sources:`
list. No structural changes to the YAML — just an additional entry.

For `runoff` (currently sources `era5_land`, `gldas_noah_v21_monthly`):

```yaml
    sources:
      - era5_land
      - gldas_noah_v21_monthly
      - mwbm_climgrid
```

For `aet` (currently sources `mod16a2_v061`, `ssebop`):

```yaml
    sources:
      - mod16a2_v061
      - ssebop
      - mwbm_climgrid
```

The `range_notes:` block of each is updated to mention that
`mwbm_climgrid` is declared as a source but not yet consumed by the
target builder. The note for `runoff` is also reworded to drop the
"Replaces the original NHM-MWBM source" sentence — that lineage is
now better captured by the `mwbm_climgrid` entry in `sources.yml`.

The variable for the `aet` target is declared as `aet` in the MWBM
catalog (matching the file's variable name); the variable for
`runoff` is `runoff`. Future target-builder wiring will read these
identifiers directly from the per-year aggregated NCs at
`<project>/data/aggregated/mwbm_climgrid/<YYYY>/mwbm_climgrid_<YYYY>_agg.nc`.

## Notebook updates

Four notebooks gain an `mwbm_climgrid` source. Each follows an
established pattern in the corresponding existing source code; the
edit is uniform in shape across notebooks. No new Markdown narrative
beyond a one-line introduction citing Wieczorek et al. (2024).

### `notebooks/inspect_consolidated_runoff.ipynb`

The notebook's "configure source paths" cell holds a list-of-dicts
keyed by source. Add a third entry pointing to
`<datastore>/mwbm_climgrid/ClimGrid_WBM.nc` and reading the `runoff`
variable. The unit conversion to mm/month is a no-op — the file's
`runoff` variable is native mm/month with `cell_methods: time: sum`.
The plotting/comparison cells iterate the source list and need no
per-source branch added.

### `notebooks/inspect_consolidated_aet.ipynb`

Same pattern. New entry reads the `aet` variable from
`ClimGrid_WBM.nc`. Unit conversion to mm/month is a no-op. The
mod16a2-specific composite-handling helper does not need to change;
the new MWBM source uses the existing mm/month path.

### `notebooks/inspect_aggregated/inspect_aggregated_runoff.ipynb`

Append `mwbm_climgrid` to the `SOURCES` dict with
`{"label": "MWBM (ClimGrid, runoff)", "var": "runoff"}`. Add a path
branch to whatever cell resolves per-source aggregated NC paths,
pointing at `<aggregated>/mwbm_climgrid/<YYYY>/mwbm_climgrid_<YYYY>_agg.nc`.
The unit-conversion helper passes MWBM through unchanged (mm/month
on input matches the comparison unit). The mean-trace and
choropleth-comparison cells iterate the SOURCES dict and pick up
the new source automatically.

### `notebooks/inspect_aggregated/inspect_aggregated_aet.ipynb`

Same pattern as runoff. New SOURCES entry
`{"label": "MWBM (ClimGrid, aet)", "var": "aet"}`. New path branch.
Unit conversion: pass-through (MWBM `aet` is mm/month, same as the
notebook's comparison unit).

### Notebook integrity

- Run `pixi run nbstripout --keep-id` on each modified notebook
  before committing (project memory `reference_nbstripout_keep_id`).
- Pre-commit hook will re-run nbstripout; ensure `--keep-id` is in
  the `.pre-commit-config.yaml` invocation (it is, as of PR #72).
- Notebooks reference data that exists only after the aggregator has
  been run against a real datastore. They will raise a clear
  `FileNotFoundError` if the user runs them before
  `agg mwbm-climgrid` finishes — same behavior as the existing
  sources (the helper module's `discover_aggregated` builds explicit
  error messages). This is acceptable; the notebooks are
  inspection tools, not unit-test fixtures.

## CLI wiring (`cli.py`)

Three additions, mirroring the `reitz2017` pattern:

1. New fetch subcommand `nhf-targets fetch mwbm-climgrid` calling
   `fetch_mwbm_climgrid`.
2. New agg subcommand `nhf-targets agg mwbm-climgrid` calling
   `aggregate_mwbm_climgrid`.
3. Append `mwbm_climgrid` to the `fetch all` and `agg all` source
   lists.

CLI flag surface matches the existing pattern: `--project-dir`,
`--period`, `--batch-size` (agg only).

## Tests

### `tests/test_fetch_mwbm_climgrid.py`

- **Catalog round-trip**: catalog entry parses, has the expected
  `access.item_id` / `access.filename` / `variables` shape.
- **Idempotency on size + sha256 match**: pre-seed the datastore with
  a tiny dummy NC + a manifest record, assert `fetch_mwbm_climgrid`
  is a no-op.
- **Idempotency repair on missing manifest**: pre-seed only the file,
  assert the function hashes it, validates, and writes the manifest.
- **Bad period rejection**: `period="1850/1900"` raises with a clear
  message.
- **Network call mocked**: `sciencebasepy.SbSession` is patched so no
  actual network access happens.

### `tests/test_aggregate_mwbm_climgrid.py`

- Adapter constructs without error (catalog typo / source_crs / glob
  validation in `SourceAdapter.__post_init__` all pass).
- Synthetic 3-month, 4-variable, 5×5-cell NC + 3-HRU GeoJSON
  fabric: assert per-year output exists at the expected path,
  contains all four variables, and HRU dimension matches the fabric.
- Reuses the test fixtures established for `watergap22d` / `reitz2017`
  where applicable.

### `tests/test_aggregate_integration.py` (modify)

Add a `mwbm_climgrid` case to the parametrized integration matrix
that exercises the full fetch → aggregate → manifest path against
real-data fixtures.

## Risks and mitigations

- **7.5 GB download time on slow links.** Documented in the catalog
  `notes`; the manifest's sha256 lets a partial/aborted download be
  detected and re-run cleanly.
- **`int16` packing surprises.** xarray decodes scale_factor /
  add_offset transparently as long as `mask_and_scale=True` (the
  default). The aggregation driver opens datasets with defaults, so
  the packed values become physical floats before gdptools sees them.
  No special handling required.
- **`time` axis encoding.** ClimGrid_WBM.nc uses CF-conformant time
  units; the driver's `_find_time_coord_name` already handles "time"
  and CF-decoded `cftime`/`datetime64` axes.
- **Spinup years end up on disk.** Per design decision 4, 1895–1899
  outputs are produced and stored. They are bit-for-bit derived from
  the publisher file and are not "wrong" — just flagged for exclusion.
  Target builders read the catalog `period` to narrow. If a future
  consumer needs the aggregator itself to drop these years, extend
  `SourceAdapter` with `year_range` (deferred).
- **Future re-aggregation if a 5th variable is added.** The single
  weight cache is reused; only the new variable's `AggGen` pass runs.
  Handled natively by the existing driver.

## Out of scope reminders

- No `targets/run.py`, `targets/aet.py`, `targets/som.py`, or any
  other target-builder code changes. `targets/run.py` keeps producing
  2-source bounds; `targets/aet.py` stays a stub.
- No notebook changes for soil moisture, recharge, or snow-covered
  area. Only the four runoff and AET notebooks are touched, matching
  the variables.yml declaration of which targets `mwbm_climgrid`
  participates in.
- No retroactive re-naming of the retired `mwbm` source key (it was
  removed with #41 and is gone from the catalog already).
