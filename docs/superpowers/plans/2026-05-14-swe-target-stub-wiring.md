# Plan: Wire the SWE target category through config schema, CLI, and HPC scripts

## Context

A new **snow water equivalent (SWE)** calibration target was added to the
catalog (`catalog/variables.yml` → `snow_water_equivalent`, `prms_variable:
pkwater_equiv`) and its **fetch + aggregate layers are done** (issue #101
umbrella: Daymet, SNODAS, ERA5-Land, Margulis WUS-SR). But the **target
category itself was never wired into the config schema or CLI** — so
`pixi run init` generates a `config.yml` with only the original five targets
(runoff/aet/recharge/soil_moisture/snow_covered_area). SWE is invisible to new
projects.

The trigger was a report that `pixi run init` does not pick up the SWE
category, plus a question of whether `config/pipeline.yml` is used and an ask
to find everywhere else the category is missing.

**Findings:**
- `pixi run init` writes a hardcoded `_CONFIG_TEMPLATE` string in
  `init_run.py`; the canonical schema is `defaults.py:DEFAULTS["targets"]`.
  Both list only five targets — SWE is missing from both.
- `config/pipeline.yml` is **reference-only** (its header says so; no code
  reads it at runtime) but is kept in sync with the schema as living
  documentation — it is also missing SWE.
- The SWE **target builder does not exist** (`targets/swe.py` is absent), and
  `cli.py:_dispatch()` has no SWE entry. Adding SWE to the schema with
  `enabled: true` without a builder would make `pixi run run` **hard-crash
  (exit 1)** at `_dispatch` — unlike aet/rch/som/sca, which have stub modules
  that `raise NotImplementedError` and are caught + skipped gracefully.

**Decisions (confirmed with user):**
1. **Full stub wiring** — SWE becomes a first-class stub target exactly like
   aet/rch/som/sca: schema + template + reference config + a `targets/swe.py`
   stub + `_dispatch` registration + `pixi` task + tests.
2. **Sync `config/pipeline.yml`** even though it is reference-only.
3. **Update `run_all.slurm`** to a 6-element array so the HPC bulk-run stays
   consistent with `pixi run run`.

**Intended outcome:** `pixi run init` generates a `config.yml` containing the
SWE block; `pixi run run` and `pixi run run-swe` build SWE and skip it
gracefully with a `NotImplementedError` warning (same as the other four
stubs); the HPC array job covers all six targets; the test suite stays green.

## The canonical SWE config block

Field values come from `catalog/variables.yml` (`snow_water_equivalent`,
lines 185-223). Key order mirrors the existing `snow_covered_area` block.
`chunk_months` is **omitted** (it appears only on the *monthly* multi-source
targets; the daily `snow_covered_area` has none). `nn_fill`/`nn_max_candidates`
are **included** (SWE is `multi_source_minmax`, like runoff/aet).
`output_file` uses the short form (`swe_targets.nc`, matching `sca_targets.nc`).

In `defaults.py` (Python dict, `period` is the required-when-enabled sentinel):
```python
        "snow_water_equivalent": {
            "enabled": True,
            "sources": ["daymet", "snodas", "era5_land", "margulis_wus_sr"],
            "time_step": "daily",
            "period": None,
            "prms_variable": "pkwater_equiv",
            "range_method": "multi_source_minmax",
            "output_file": "swe_targets.nc",
            "nn_fill": True,
            "nn_max_candidates": 10,
        },
```

In `init_run.py` `_CONFIG_TEMPLATE` and `config/pipeline.yml` (YAML, concrete
example `period` — SWE cannot start before SNODAS in 2003):
```yaml

  snow_water_equivalent:
    enabled: true
    sources:
      - daymet
      - snodas
      - era5_land
      - margulis_wus_sr
    time_step: daily
    period: "2003-01-01/2010-12-31"
    prms_variable: pkwater_equiv
    range_method: multi_source_minmax
    output_file: swe_targets.nc
    nn_fill: true
    nn_max_candidates: 10
```

## Files to change

### Schema + generated config (the core bug)

**1. `src/nhf_spatial_targets/defaults.py`** — insert the `snow_water_equivalent`
Python block into `DEFAULTS["targets"]`, after the `snow_covered_area` block
(after its closing `},` at line 104, before the `},` that closes `targets`).

**2. `src/nhf_spatial_targets/init_run.py`** — insert the YAML block into the
`_CONFIG_TEMPLATE` string, after the `snow_covered_area:` block (after
`output_file: sca_targets.nc` at line 120, before the closing `"""` at line
121). Keep the blank-line separator between target blocks.

**3. `config/pipeline.yml`** — append the same YAML block at end of file
(after `output_file: sca_targets.nc` at line 109).

### CLI + task wiring

**4. `src/nhf_spatial_targets/targets/swe.py`** *(new file)* — stub builder
mirroring `targets/sca.py` / `targets/aet.py`. Include
`from __future__ import annotations` per the CLAUDE.md convention (as
`targets/rch.py` does). Signature must be the 3-arg stub form used by
`_dispatch` at `cli.py:146`:
```python
"""Build snow water equivalent calibration targets from Daymet, SNODAS, ERA5-Land, and Margulis WUS-SR."""

from __future__ import annotations

# Sources:  daymet, snodas, era5_land, margulis_wus_sr
# Method:   multi_source_minmax (per-HRU per-day nanmin/nanmax across sources)
# Variable: pkwater_equiv
# Timestep: daily


def build(config: dict, fabric_path: str, output_path: str) -> None:
    """Build SWE target dataset."""
    raise NotImplementedError
```

**5. `src/nhf_spatial_targets/cli.py`** — in `_dispatch()` (lines 122-146):
- line 128: add `swe` to the import →
  `from nhf_spatial_targets.targets import aet, rch, run, sca, som, swe`
- `builders` dict (lines 134-139): add `"snow_water_equivalent": swe.build,`
  after the `snow_covered_area` entry.

  `run()` already catches `NotImplementedError` (cli.py:110-115) and logs
  `WARNING ... skipping`, so the stub is safe for both bulk `run` and
  `run --target snow_water_equivalent`.

**6. `pixi.toml`** — after the `run-sca` task (line 76), add:
```toml
run-swe    = { cmd = "nhf-targets run --target snow_water_equivalent" }
```
(Match the aligned-`=` formatting of the `run-runoff`..`run-sca` block.)

### Tests (3 will break without edits + 1 new)

**7. `tests/test_defaults.py`** — `test_defaults_has_all_five_targets`
(lines 15-23) is an **exact-set** assertion → **breaks**. Add
`"snow_water_equivalent"` to the set and rename the function to
`test_defaults_has_all_six_targets`.

**8. `tests/test_validate.py`** — the `_write_config()` fixture (`cfg["targets"]`,
lines 55-61) lists five targets each with a `period`. Once SWE is in `DEFAULTS`
with `enabled: True`, `missing_required` → `validate_workspace` **raises** for
every test in this file unless the fixture supplies SWE's period. Add:
`"snow_water_equivalent": {"period": "2000-01-01/2010-12-31"},`

**9. `tests/test_cli.py`** — `test_run_dispatches_enabled_targets` (lines 68-90)
disables every default target except runoff and asserts
`mock_dispatch.assert_called_once()`. With SWE defaulting to `enabled: True` a
second dispatch fires → **breaks**. Add
`"  snow_water_equivalent:\n    enabled: false\n"` to the `config_extra` disable
list.

**10. `tests/test_swe.py`** *(new file)* — required by the CLAUDE.md Test
Coverage Rule. Mirror `tests/test_rch_target.py`'s shape; cover both the stub
builder and the catalog wiring:
```python
"""Tests for the snow water equivalent calibration target."""

from __future__ import annotations

import pytest


def test_swe_build_is_stub():
    """The SWE target builder is a stub until implemented."""
    from nhf_spatial_targets.targets.swe import build

    with pytest.raises(NotImplementedError):
        build({}, "/fake/fabric.gpkg", "/fake/out")


def test_swe_variable_lists_four_sources():
    """SWE catalog variable carries all four SWE sources."""
    from nhf_spatial_targets import catalog

    v = catalog.variable("snow_water_equivalent")
    assert set(v["sources"]) == {
        "daymet",
        "snodas",
        "era5_land",
        "margulis_wus_sr",
    }
```

### Docs + HPC script

**11. `CLAUDE.md`**
- line 4: the target list is explicitly scoped to *TM 6-B10*; SWE
  (`pkwater_equiv`) references TM 6-B7, not TM 6-B10 — do **not** insert it into
  the TM 6-B10 list. Append a sentence instead, e.g. "... snow-covered area. A
  sixth target, snow water equivalent (SWE), extends the pipeline beyond the
  TM 6-B10 set."
- line 82: `per-variable target builders (run/aet/rch/som/sca)` →
  `(run/aet/rch/som/sca/swe)`.

**12. `README.md`** — line 399: `submits a 5-element array (runoff, aet, rch,
som, sca)` → `submits a 6-element array (runoff, aet, rch, som, sca, swe)`.

**13. `run_all.slurm`** — make it a 6-element array:
- line 5: `#SBATCH --array=0-4` → `#SBATCH --array=0-5`
- line 14: `(5 total)` → `(6 total)`
- line 18: `Stub targets (AET, RCH, SOM, SCA)` →
  `Stub targets (AET, RCH, SOM, SCA, SWE)`
- `RUN_TASKS` array (lines 55-61): add after the `run-sca` line:
  `    "run-swe"      # 5 — SWE (stub)`

## Out of scope (verified, deliberately not touched)

- **`validate.py:_SOURCE_KEYS`** — this convenience list (datastore subdirs
  pre-created by `validate`) already omits `ssebop` and `mwbm_climgrid` and
  those sources work fine; `fetch/snodas.py` self-creates its dirs and
  `fetch/daymet.py` is verify-only with no datastore subdir. Touching it would
  also force a `test_validate.py` change for zero functional gain.
- **`docs/` reference material, `docs/superpowers/`, inspection notebooks** —
  the methodological docs describe the TM 6-B10 five; SWE is a pipeline
  extension. No inspection notebook applies to a stub that produces no output.

## Verification

Branch first (CLAUDE.md git workflow — never commit to `main`): create a
GitHub issue, then `git checkout -b feature/<issue#>-swe-target-stub-wiring`.

1. **Format + lint locally** (fast; CI gate):
   ```
   pixi run -e dev fmt && pixi run -e dev lint
   ```

2. **`pixi run init` smoke test** — the core bug. Generate a fresh project and
   confirm the SWE block is present:
   ```
   pixi run init -- --project-dir /tmp/swe-wiring-check
   grep -A12 'snow_water_equivalent:' /tmp/swe-wiring-check/config.yml
   rm -rf /tmp/swe-wiring-check
   ```
   Expect the full block: 4 sources, `prms_variable: pkwater_equiv`,
   `range_method: multi_source_minmax`, `output_file: swe_targets.nc`,
   `period: "2003-01-01/2010-12-31"`, `nn_fill: true`, `nn_max_candidates: 10`.

3. **Dispatch wiring smoke test** — confirm the builder resolves and the stub
   is caught gracefully (no exit-1 crash):
   ```
   pixi run python -c "from nhf_spatial_targets.targets import swe; import pytest; pytest.raises(NotImplementedError, swe.build, {}, 'f', 'o'); print('swe.build wired + stubbed OK')"
   pixi run pixi task list | grep run-swe
   ```

4. **Targeted tests locally**, then full suite via CI. Local pytest is slow on
   this HPC, so run the directly-affected tests locally and let GitHub Actions
   run the full suite on push:
   ```
   pixi run -e dev pytest tests/test_defaults.py tests/test_cli.py tests/test_validate.py tests/test_swe.py tests/test_catalog.py -q
   ```
   `test_catalog.py` must stay green (CLAUDE.md). Then push and watch CI for
   the full `pixi run -e dev test` run.

5. **SLURM syntax check** — `bash -n run_all.slurm`; visually confirm
   `RUN_TASKS` has 6 entries and `--array=0-5`.

## Critical files

- `src/nhf_spatial_targets/defaults.py` — `DEFAULTS["targets"]` (the canonical schema)
- `src/nhf_spatial_targets/init_run.py` — `_CONFIG_TEMPLATE` (what `init` writes)
- `src/nhf_spatial_targets/cli.py` — `_dispatch()` builders dict (lines 122-146)
- `src/nhf_spatial_targets/targets/swe.py` *(new)* — stub builder
- `tests/test_defaults.py` — breaking exact-set assertion
- `tests/test_cli.py`, `tests/test_validate.py` — breaking fixtures
- `config/pipeline.yml`, `pixi.toml`, `tests/test_swe.py` *(new)*, `CLAUDE.md`,
  `README.md`, `run_all.slurm` — lower-risk wiring/doc edits
