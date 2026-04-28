# Reitz 2017 Aggregator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `aggregate_reitz2017` so the recharge calibration target has its third source end-to-end aggregable, following the existing `watergap22d` adapter pattern.

**Architecture:** A 30-line `SourceAdapter` that mirrors the WaterGAP 2.2d adapter, with two deviations: `variables=("total_recharge", "eff_recharge")` (per the catalog block) and `source_crs="EPSG:4269"` (NAD83 geographic — Reitz's first non-EPSG:4326 source in the project). The shared `aggregate_source` driver handles per-year splitting from the single multi-year consolidated NC and emits `<project>/data/aggregated/reitz2017/reitz2017_<YYYY>_agg.nc` per year.

**Tech Stack:** Python 3.11+, gdptools, xarray, pandas, geopandas, pytest, cyclopts. Pixi for environment management.

**Spec:** `docs/superpowers/specs/2026-04-27-reitz2017-aggregator-design.md`

---

## Task 1: Adapter declaration with TDD sanity test

**Files:**
- Create: `src/nhf_spatial_targets/aggregate/reitz2017.py`
- Create: `tests/test_aggregate_reitz2017.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_aggregate_reitz2017.py`:

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

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/test_aggregate_reitz2017.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nhf_spatial_targets.aggregate.reitz2017'`.

- [ ] **Step 3: Implement the adapter module**

Create `src/nhf_spatial_targets/aggregate/reitz2017.py`:

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
    source_crs="EPSG:4269",  # NAD83 geographic — Reitz GeoTIFFs preserve this
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

The default `files_glob="*_consolidated.nc"` matches `reitz2017_consolidated.nc` produced by `fetch/reitz2017.py` — no override needed. CRS validation in `SourceAdapter.__post_init__` will pyproj-parse `"EPSG:4269"` at import time, surfacing typos immediately.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/test_aggregate_reitz2017.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full unit suite to confirm no regressions**

Run: `pixi run -e dev pytest`
Expected: PASS, 481 passed (480 prior + 1 new), 14 deselected.

- [ ] **Step 6: Commit**

```bash
pixi run git commit -m "$(cat <<'EOF'
feat(aggregate): Reitz 2017 adapter (total_recharge + eff_recharge)

30-line SourceAdapter mirroring watergap22d, with two deviations:
variables=("total_recharge", "eff_recharge") to align with the catalog
block, and source_crs="EPSG:4269" for NAD83 geographic. Reitz is the
first non-EPSG:4326 source in the project; gdptools reconciles the
datum mismatch via pyproj at WeightGen time, so no fetch-side
re-projection is needed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Stage `src/nhf_spatial_targets/aggregate/reitz2017.py` and `tests/test_aggregate_reitz2017.py` before committing (or rely on `git commit -a` if no other changes are pending).

---

## Task 2: CLI wiring (`agg reitz2017` + `agg all` entry)

**Files:**
- Modify: `src/nhf_spatial_targets/cli.py`

- [ ] **Step 1: Add the import**

In `src/nhf_spatial_targets/cli.py`, locate the block of `aggregate_*` imports (around line 17–25, after `from nhf_spatial_targets._logging import setup_logging`). Insert this line in alphabetical order — between `from nhf_spatial_targets.aggregate.ncep_ncar import aggregate_ncep_ncar` and `from nhf_spatial_targets.aggregate.nldas_mosaic import aggregate_nldas_mosaic`... wait, alphabetical is `r` after `n`-prefixed and before `s`. Insert between `nldas_noah` and `watergap22d`:

```python
from nhf_spatial_targets.aggregate.reitz2017 import aggregate_reitz2017
```

The block after the edit should be:

```python
from nhf_spatial_targets.aggregate.era5_land import aggregate_era5_land
from nhf_spatial_targets.aggregate.gldas import aggregate_gldas
from nhf_spatial_targets.aggregate.merra2 import aggregate_merra2
from nhf_spatial_targets.aggregate.mod10c1 import aggregate_mod10c1
from nhf_spatial_targets.aggregate.mod16a2 import aggregate_mod16a2
from nhf_spatial_targets.aggregate.ncep_ncar import aggregate_ncep_ncar
from nhf_spatial_targets.aggregate.nldas_mosaic import aggregate_nldas_mosaic
from nhf_spatial_targets.aggregate.nldas_noah import aggregate_nldas_noah
from nhf_spatial_targets.aggregate.reitz2017 import aggregate_reitz2017
from nhf_spatial_targets.aggregate.watergap22d import aggregate_watergap22d
```

- [ ] **Step 2: Add the `agg reitz2017` CLI command**

Locate `agg_watergap22d_cmd` (around line 1126–1132) and insert a new command **immediately after it** (so the help-listing order matches the alphabetical file order in the imports). Add:

```python
@agg_app.command(name="reitz2017")
def agg_reitz2017_cmd(
    workdir: Annotated[Path, Parameter(name=["--project-dir"])],
    batch_size: Annotated[int, Parameter(name="--batch-size")] = 500,
):
    """Aggregate Reitz 2017 annual recharge to HRU polygons."""
    _run_tier_agg(aggregate_reitz2017, "Reitz 2017", workdir, batch_size)
```

The placement after `agg_watergap22d_cmd` keeps recharge sources clustered together in the help output. (WaterGAP and Reitz are both recharge sources.)

- [ ] **Step 3: Append Reitz to the `agg all` source list**

Locate `agg_all_cmd` (around line 1153). Inside it, find the `sources: list[tuple[str, Callable[..., None]]] = [...]` literal (around line 1178). It currently ends with:

```python
        ("watergap22d", aggregate_watergap22d),
        ("mod16a2", aggregate_mod16a2),
        ("mod10c1", aggregate_mod10c1),
    ]
```

Insert `("reitz2017", aggregate_reitz2017),` immediately after the WaterGAP entry, keeping recharge sources together:

```python
        ("watergap22d", aggregate_watergap22d),
        ("reitz2017", aggregate_reitz2017),
        ("mod16a2", aggregate_mod16a2),
        ("mod10c1", aggregate_mod10c1),
    ]
```

- [ ] **Step 4: Smoke-test the CLI registration**

Run: `pixi run nhf-targets agg --help`
Expected output: lists `reitz2017` as one of the agg subcommands alongside `watergap22d`, `mod16a2`, etc., with the description "Aggregate Reitz 2017 annual recharge to HRU polygons."

Run: `pixi run nhf-targets agg reitz2017 --help`
Expected output: shows `--project-dir` and `--batch-size` parameters and the docstring.

- [ ] **Step 5: Run the full unit suite**

Run: `pixi run -e dev pytest`
Expected: PASS, all tests still green.

- [ ] **Step 6: Lint and format**

Run: `pixi run -e dev fmt && pixi run -e dev lint`
Expected: format leaves files unchanged (the import was inserted in an already-sorted block; the new function follows existing style); lint passes.

- [ ] **Step 7: Commit**

```bash
pixi run git commit -m "$(cat <<'EOF'
feat(cli): wire agg reitz2017 + add to agg all

Adds the `agg reitz2017` subcommand and appends reitz2017 to the
agg-all source list, keeping recharge sources (watergap22d + reitz2017)
clustered together in the help output and the all-runner sequence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Stage `src/nhf_spatial_targets/cli.py` before committing.

---

## Task 3: Integration-test stub matching existing convention

**Files:**
- Modify: `tests/test_aggregate_integration.py`

The existing file is uniformly `@pytest.mark.skip` stubs awaiting a fixture infra that hasn't been built. Reitz matches that convention.

- [ ] **Step 1: Add the stub to `tests/test_aggregate_integration.py`**

Locate `test_aggregate_watergap22d_end_to_end` (currently around line 47–49). Insert this stub **immediately after it** so recharge sources stay grouped:

```python
@pytest.mark.skip(reason="fixture datastore + mini-fabric not yet checked in")
def test_aggregate_reitz2017_end_to_end():
    """aggregate_reitz2017 writes per-year NCs with total_recharge + eff_recharge."""
    raise NotImplementedError
```

The result, with surrounding stubs:

```python
@pytest.mark.skip(reason="fixture datastore + mini-fabric not yet checked in")
def test_aggregate_watergap22d_end_to_end():
    raise NotImplementedError


@pytest.mark.skip(reason="fixture datastore + mini-fabric not yet checked in")
def test_aggregate_reitz2017_end_to_end():
    """aggregate_reitz2017 writes per-year NCs with total_recharge + eff_recharge."""
    raise NotImplementedError


@pytest.mark.skip(reason="fixture datastore + mini-fabric not yet checked in")
def test_aggregate_mod16a2_end_to_end():
    raise NotImplementedError
```

- [ ] **Step 2: Verify the stub is registered (and skipped) by pytest**

Run: `pixi run -e dev pytest tests/test_aggregate_integration.py -v`
Expected: every test in the file is reported as `SKIPPED` (no failures, no errors), and the new `test_aggregate_reitz2017_end_to_end` appears in the listing.

- [ ] **Step 3: Verify the integration-test target also discovers it**

Run: `pixi run -e dev test-integration --collect-only 2>&1 | grep reitz2017`
Expected: `test_aggregate_integration.py::test_aggregate_reitz2017_end_to_end` appears in the collected output.

(The `test-integration` pixi task selects `@pytest.mark.integration`-marked tests; the stub inherits the marker via the `pytestmark = pytest.mark.integration` module-level declaration.)

- [ ] **Step 4: Commit**

```bash
pixi run git commit -m "$(cat <<'EOF'
test(aggregate): Reitz 2017 integration-test stub

Matches the existing skip-pattern convention in this file. Real
end-to-end validation happens on caldera against the populated
datastore; the synthetic-fixture harness is a separate effort that
should retrofit all 9 stubs together.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Stage `tests/test_aggregate_integration.py` before committing.

---

## Task 4: Recharge inspection notebook caveat removal

**Files:**
- Modify: `notebooks/inspect_aggregated/inspect_aggregated_recharge.ipynb`

The notebook's cell 1 (intro) and cell 2 (per-target conventions) both flag Reitz as "may not yet exist." Drop those caveats now that the aggregator lands. The generic `discover_aggregated` skip-with-reason path in cell 5 stays — that's a defensive measure for any source missing aggregations on a given project, not a Reitz-specific assumption.

- [ ] **Step 1: Read the current cells to find the exact strings to replace**

Run: `pixi run -e dev python -c "
import json
nb = json.load(open('notebooks/inspect_aggregated/inspect_aggregated_recharge.ipynb'))
for i, c in enumerate(nb['cells'][:2]):
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    print(f'=== cell {i} ({c[\"cell_type\"]}, id={c[\"id\"]}) ===')
    print(src)
    print()
"`

Confirm the output includes:

- In cell 0 (intro markdown): the bullet `- Reitz 2017 (\`total_recharge\`, m/year, annual). **Aggregator may not yet exist** — this notebook skips Reitz with a clear message if its aggregated NCs are absent.`
- In cell 1 (per-target conventions markdown): the bullet `- Reitz skipped-with-reason if no aggregated NCs are present.`

- [ ] **Step 2: Apply the two replacements via a Python script**

Run:

```bash
pixi run -e dev python <<'PY'
import json
from pathlib import Path

p = Path("notebooks/inspect_aggregated/inspect_aggregated_recharge.ipynb")
nb = json.loads(p.read_text())

def get_src(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])

def set_src(cell, new):
    cell["source"] = new.splitlines(keepends=True)

# Cell 0: intro — replace the Reitz bullet
cell0 = nb["cells"][0]
src0 = get_src(cell0)
old0 = (
    "- Reitz 2017 (`total_recharge`, m/year, annual). "
    "**Aggregator may not yet exist** — this notebook skips Reitz "
    "with a clear message if its aggregated NCs are absent."
)
new0 = "- Reitz 2017 (`total_recharge`, m/year, annual)."
assert old0 in src0, "Cell 0 Reitz bullet not found verbatim — read cell 0 first to confirm exact text"
set_src(cell0, src0.replace(old0, new0))

# Cell 1: per-target conventions — drop the skip-with-reason bullet entirely
cell1 = nb["cells"][1]
src1 = get_src(cell1)
old1 = "- Reitz skipped-with-reason if no aggregated NCs are present.\n"
assert old1 in src1, "Cell 1 Reitz skip bullet not found verbatim — read cell 1 first to confirm exact text"
set_src(cell1, src1.replace(old1, ""))

# Cell IDs preserved (--keep-id is in the pre-commit config since PR #72)
p.write_text(json.dumps(nb, indent=1) + "\n")
print("Updated cells 0 and 1; cell IDs preserved.")
PY
```

If either `assert` fails (because the bullet wording in the notebook differs from what's in this plan), STOP and re-read the cell to find the actual text. Do not invent the replacement.

- [ ] **Step 3: Verify cell IDs are unchanged and the notebook is still nbformat-valid**

Run:

```bash
pixi run -e dev python <<'PY'
import json
from pathlib import Path

p = Path("notebooks/inspect_aggregated/inspect_aggregated_recharge.ipynb")
nb = json.loads(p.read_text())
ids = [c["id"] for c in nb["cells"]]
assert len(ids) == 15
assert len(set(ids)) == 15
import re
for i in ids:
    assert re.fullmatch(r"[a-f0-9]{12}", i), f"bad id {i}"

# Confirm no leftover Reitz "may not exist" wording
src = "\n".join(get if isinstance((get := c["source"]), str) else "".join(get) for c in nb["cells"])
assert "may not yet exist" not in src, "leftover 'may not yet exist' in notebook"
assert "skipped-with-reason if no aggregated NCs" not in src, "leftover skip-with-reason caveat"
print("OK: 15 unique 12-char hex IDs, no stale Reitz caveats")
PY
```

Expected: `OK: 15 unique 12-char hex IDs, no stale Reitz caveats`

- [ ] **Step 4: Render-check in VSCode (manual, optional but recommended)**

If you have VSCode access, open `notebooks/inspect_aggregated/inspect_aggregated_recharge.ipynb` and confirm:
- Cell 1 (intro) shows three sources, the Reitz bullet is plain (no bold caveat).
- Cell 2 (per-target conventions) no longer mentions "Reitz skipped-with-reason."
- All 15 cells render — no silently-dropped cells.

If you don't have VSCode access, the JSON-level checks in Step 3 are sufficient for this plan; the user (rmcd) renders on caldera and will catch any visual issue at PR review.

- [ ] **Step 5: Commit**

```bash
pixi run git commit -m "$(cat <<'EOF'
docs(notebooks): remove Reitz "may not yet exist" caveat (#70 follow-up)

The recharge inspection notebook (PR #72) flagged Reitz as a
forward-looking source whose aggregator hadn't been built. Drop those
caveats now that aggregate_reitz2017 lands. The generic
discover_aggregated skip-with-reason path in cell 5 stays — that's a
defensive measure for any source missing aggregations on a given
project, not a Reitz-specific assumption.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Stage `notebooks/inspect_aggregated/inspect_aggregated_recharge.ipynb` before committing. The `nbstripout --keep-id` pre-commit hook will preserve cell UUIDs.

---

## Task 5: Pre-PR quality gate and PR

**Files:** none — verification only.

- [ ] **Step 1: Run the full quality gate**

```bash
pixi run -e dev fmt
pixi run -e dev lint
pixi run -e dev test
```

Expected: format leaves files unchanged; lint passes; all 481 unit tests pass (480 prior + 1 new adapter test); 14 integration tests deselected (now 9 — they were 8 before this PR added the Reitz stub).

- [ ] **Step 2: Confirm the new aggregator is reachable through the CLI**

```bash
pixi run nhf-targets agg --help | grep reitz2017
```

Expected: a line like `reitz2017  Aggregate Reitz 2017 annual recharge to HRU polygons.` (exact format depends on cyclopts; the key check is "reitz2017" appears).

- [ ] **Step 3: Push the branch**

```bash
git push -u origin feature/reitz2017-aggregator
```

Expected: branch is pushed; gh prints a hint URL.

- [ ] **Step 4: Open the PR**

```bash
gh pr create --title "Reitz 2017 aggregator" --body "$(cat <<'EOF'
## Summary

- Adds `aggregate_reitz2017` so the recharge calibration target has its third source end-to-end aggregable
- 30-line `SourceAdapter` mirroring `watergap22d`, with two deviations: `variables=("total_recharge", "eff_recharge")` (per the catalog block) and `source_crs="EPSG:4269"` (NAD83 geographic — Reitz's first non-EPSG:4326 source in the project)
- CLI wires `agg reitz2017` + appends to `agg all`
- Drops the "may not yet exist" caveats from `inspect_aggregated_recharge.ipynb` cells 1 and 2 now that the aggregator lands
- Integration test added as `@pytest.mark.skip` stub matching the existing convention in `test_aggregate_integration.py` (the synthetic-fixture harness is a separate effort that should retrofit all 9 stubs together)

Spec: `docs/superpowers/specs/2026-04-27-reitz2017-aggregator-design.md`. Plan: `docs/superpowers/plans/2026-04-27-reitz2017-aggregator.md`.

## Test plan

- [x] `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test` all pass
- [x] `pixi run nhf-targets agg reitz2017 --help` shows the new subcommand
- [ ] End-to-end run on caldera: `pixi run nhf-targets agg reitz2017 --project-dir <gfv2-project>` produces per-year NCs at `data/aggregated/reitz2017/reitz2017_<YYYY>_agg.nc` with both variables, finite values for in-CONUS HRUs, and CONUS-mean `total_recharge` ≈ 122 mm/yr (matches the published Reitz CONUS mean within the gridded-vs-area-weighted tolerance noted in the recharge inspection notebook)
- [ ] Recharge inspection notebook renders cleanly with Reitz now loading as a real source (no skip-with-reason on the user's project once aggregation runs)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

---

## Self-Review

- **Spec coverage:**
  - Goal: closes the "Reitz aggregator must be built" forward-looking note → Tasks 1–4 cover the aggregator + CLI + tests + notebook tweak.
  - Decision 1 (mirror watergap22d) → Task 1.
  - Decision 2 (both variables) → Task 1's `variables=("total_recharge", "eff_recharge")` line, asserted in Task 1's test.
  - Decision 3 (`source_crs="EPSG:4269"`) → Task 1's adapter, asserted in Task 1's test.
  - Decision 4 (no fetch-side re-projection) → no code change; the spec elaboration documents the rationale; Task 1's commit message reflects it.
  - Decision 5 (notebook caveat removal) → Task 4.
  - "Driver behaviour at run time" section → no task needed; existing driver code already handles per-year splitting from a multi-year consolidated NC (verified by reading `_driver.py::enumerate_years` during exploration).
  - "Open / verify" — `apply_cf_metadata` axis-attribute confirmation is implicitly covered: WaterGAP works through the same code path, so the same CF attrs are present. End-to-end caldera run lives in the PR test plan, not the implementation plan (it's a manual user step post-merge).
- **Placeholder scan:** no TBD/TODO/etc.; every code step shows the actual code; every shell step has the exact command and expected output.
- **Type consistency:** `aggregate_reitz2017` signature `(fabric_path, id_col, workdir, batch_size=500)` matches the watergap22d signature it's modelled on and the `_run_tier_agg` helper's expected call. `ADAPTER` is the `SourceAdapter` instance asserted in the test. The CLI command's `_run_tier_agg(aggregate_reitz2017, "Reitz 2017", workdir, batch_size)` call matches the helper's signature used by every other `agg_*_cmd`.
