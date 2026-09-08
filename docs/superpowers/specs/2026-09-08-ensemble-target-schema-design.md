# Ensemble calibration target schema — design

- **Issue:** [#338](https://github.com/rmcd-mscb/nhf-spatial-targets/issues/338)
- **Date:** 2026-09-08
- **Status:** approved for implementation planning
- **Motivating artifact:** colleagues' `06_Create_calibration_target_ensembles.ipynb`

## 1. Problem

The notebook writes eight data variables per calibration target; the pipeline
writes three. It also normalizes each source over that source's own period of
record, where the pipeline applies one shared window to every source. Bringing
the pipeline to the notebook's schema is the goal, minus three notebook
behaviors we deliberately decline to copy (§7).

Current pipeline output (`targets/_writers.py:write_bounds_target`):

```
lower_bound(time, id_col)   float32
upper_bound(time, id_col)   float32
n_sources(time, id_col)     int8
```

Target output:

```
<source_key_1>(time, id_col)  float32   # one per configured source
...
ensemble_mean(time, id_col)   float32
ensemble_std(time, id_col)    float32
lower_bound(time, id_col)     float32
upper_bound(time, id_col)     float32
n_sources(time, id_col)       int8
```

## 2. Approach

**Members ride along on the existing reduction.** Every target loader already
builds a `dict[source_key, DataArray]` immediately before calling
`multi_source_nanminmax`. That dict *is* the ensemble. It is passed through to
the writer rather than discarded.

The road not taken: moving the whole reduction into the writer, so that "the
bounds are the member min/max" becomes a structural invariant. Rejected because
SCA's bounds are a MOD10C1 CI interval, not a member min/max, and would need a
bespoke escape hatch — the same over-generalization `targets/_combine.py`'s
docstring records declining once already. The invariant is worth a test
assertion, not a refactor.

## 3. Output schema

### 3.1 `SourceLoaderResult`

One new field:

```python
members: dict[str, xr.DataArray] | None = None
```

Populated by all five loaders (`aet`, `run`, `rch`, `som`, `swe`) with the dict
they already construct. `_driver._apply_forced_zero` carries it through the SCA
rebuild unchanged.

### 3.2 `write_bounds_target`

Two new parameters: `members` and `emit_members`. When emitting:

- Each member becomes a data variable **named by its source key**, carrying the
  bounds' `units` and `coordinates`, and a `long_name` taken from the shim's
  `description`.
- `ensemble_mean` / `ensemble_std` are computed from one `xr.concat` on a
  `source` dim, `skipna=True`, `ddof=0` — the same stack shape
  `multi_source_nanminmax` already builds.
- `ensemble_std` is set to NaN wherever `n_sources < 2`. At one finite source
  the population std is exactly 0, which a downstream calibration weight would
  read as perfect inter-source agreement. Both stats carry
  `ancillary_variables = "n_sources"`.

### 3.3 Constant schema across years

**Every configured member is always emitted, as an all-NaN array when that
source contributed nothing.** This is not cosmetic. `stitch_year_chunks_to_target`
opens per-year files with `join="exact"`; a 1990 SWE year missing a `snodas`
variable would produce a ragged schema across the stitch. Constant schema is
the invariant that lets the generic stitcher stay generic.

### 3.4 Global attributes

- `source_keys` — comma-joined source keys in config order (machine-readable;
  the existing `source` attr keeps its semicolon-joined prose descriptions).
- `members_emitted` — `"true"` / `"false"`, **always present**. Without it a
  reader cannot distinguish "this target never emitted members" from "members
  were emitted and something removed them." The notebook's summary cell
  currently infers member presence from `source_keys ∩ data_vars`, which
  degrades silently to an empty list.
- `Conventions` — bumped to `CF-1.8` to match the notebook and the aggregated
  NCs.

### 3.5 Encoding

The two hard-coded `target_dtypes` dicts (`_writers.py`, `_intermediates.py`)
become "int8 for `n_sources` / `nn_filled`, float32 for everything else." This
is what lets members ride the year-chunked path with no further change — the
stitcher is otherwise fully generic over `data_vars`.

### 3.6 NN-filled companion

The `_nn_filled` companion carries only the filled bounds plus the `nn_filled`
flag — **no members**. NN-fill is defined on the combined bound; filling an
individual member would fabricate a source observation at an HRU that source
never covered. Moot for the snow targets, where `nn_fill` is already off.

## 4. Normalization

`normalize_period` accepts a new sentinel string `per_source_por` alongside the
existing `"start/end"` window form. Under the sentinel, `rch.py` and `som.py`
normalize each source over **its own record, trimmed to complete years**.

New helper in `normalize/methods.py`:

```python
def complete_years_window(da: xr.DataArray, cadence: str) -> slice
```

The trim is load-bearing, not defensive. ERA5-Land's record ends mid-year and
recharge sums months to an annual total, so an untrimmed partial trailing year
produces a spuriously low annual sum that becomes the per-HRU minimum and
compresses every other year toward 1.0.

**The trim applies uniformly to every source and every reduction** (decided
2026-09-08), not only to the annual sums where it is strictly load-bearing.
Reasoning per-reduction about which are safe is more fragile than always
trimming, and a single rule makes two sources' normalized series directly
comparable.

### 4.1 Complete years as the pipeline's definition of "period of record"

`complete_years_window` is the **single definition of a source's period of
record** wherever the pipeline computes one, not a helper local to
normalization. A source's POR is its complete-year span; a partial leading or
trailing year is coverage, not record.

This is a consistency rule, not a change to target contents. Specifically:

- The output time axis is still driven by the target's configured `period`. The
  POR trim narrows *normalization windows*, never the emitted time range.
- The NC records the trimmed window per member as
  `normalize_window_<source_key>`, so the file states the POR it actually used.
- Catalog `period:` fields in `sources.yml` are prose documentation of upstream
  coverage and are **not** authoritative for this computation, which is derived
  from the aggregated NCs on disk. Auditing those fields for complete-year
  consistency is out of scope here (§12).

The NC also records `normalize_period = "per_source_por"` at the global level,
so the mode is visible without inspecting the per-member window attrs.

`parse_period` and `validate` must both recognize the sentinel rather than
attempting to parse it as a date range.

## 5. Config surface

Two additions, each through the three-point checklist in CLAUDE.md
(`init_run._CONFIG_TEMPLATE` stub → `test_init_run.py` assertion →
`upgrade_config.OPTIONAL_CONFIG_FEATURES` entry), plus registration in
`defaults.DEFAULTS` so `validate`'s unknown-key linter does not flag them:

| Key | Default | Scope |
|---|---|---|
| `targets.<t>.emit_members` | `true` (`false` for `snow_covered_area`) | per target |
| `targets.<t>.normalize_period: per_source_por` | n/a (existing key, new value) | recharge, soil moisture |

`snow_covered_area` defaults to `false` because SCA's bounds are a MOD10C1 CI
interval, not a member min/max. Emitting members there would ship variables that
do not reconstruct the bounds, inviting a reader to assume they do. An operator
can still turn it on deliberately for diagnostics.

`emit_members` is an **output switch, not a science switch.** Members are always
computed — they are the input to every reduction. The flag decides only whether
they are also written to disk. Bounds are byte-identical either way.

The `_CONFIG_TEMPLATE` stub must say this in the comment an operator actually
reads, not only here. Required content: (a) members are always computed, the
flag only controls whether they are written; (b) turning it off does not change
`lower_bound` / `upper_bound` / `n_sources` / `ensemble_mean` / `ensemble_std`;
(c) the reason to turn it off is file size on large fabrics, with the gfv2 daily
SWE figure as the concrete example; (d) `members_emitted` records the choice in
the output NC. `tests/test_init_run.py` asserts the stub is present.

It is per-target rather than per-project because the split is between cadences
*within* a project: on gfv2 members are cheap and useful for monthly AET
(592 MB → ~2 GB) and expensive for daily SWE (12 GB → ~48 GB). A project-level
flag would force giving up the cheap case to avoid the expensive one.

Roads not taken: a single project-level flag (above); auto-deciding by estimated
size (a threshold that silently changes a file's schema makes two projects'
outputs incomparable for no visible reason — better that the operator states
intent and `config.effective.yml` records it).

## 6. Oregon project configuration

No code changes; all of this is `config.yml`.

| Target | Change |
|---|---|
| recharge | drop `watergap22d`; `normalize_period: per_source_por` |
| soil_moisture | drop `ncep_ncar`; `normalize_period: per_source_por` |
| snow_water_equivalent | sources → `snodas, era5_land, margulis_wus_sr, ua_swe` (daymet dropped) |
| snow_covered_area | `enabled: false` |
| all | `nn_fill: false` |

SCA code is retained — it is disabled for the Oregon fabric only, and remains
available to other fabrics.

SLURM: bump `slurm/project_or/run_or.slurm` and `slurm/project_gfv2/run_gfv2.slurm`
from `--mem=32G` to `64G`. `write_bounds_target` calls `ds.compute()` (full
materialization, no streaming) and `nn_fill_bounds` then makes a second full
copy; gfv2 soil-moisture monthly with members and NN-fill lands near 11 GB.

Making `write_bounds_target` stream is explicitly **not** proposed — it would be
a real refactor of the monthly path for a problem `--mem` solves.

## 7. Deliberate divergences from the notebook

1. **MOD16A2 8-day → monthly.** Keep `aet.mod16a2_to_mm_per_month`'s true
   overlap-day weighting. The notebook's
   `(ET_500m/8).resample(time="ME").mean() * days_in_month` assigns each
   composite wholly to the month of its *start date* and weights every composite
   in a month equally — a composite starting Dec 27 lands entirely in December.
   The notebook's own markdown already claims to be overlap-weighted, so this is
   a prose/code mismatch on their side.
2. **Spatial metadata.** Keep the minted `centroid_lat` / `centroid_lon` coords
   and the 0-dim WGS84 grid-mapping variable. The notebook reuses the aggregated
   NC's `crs` verbatim, but gdptools writes it as an `(id_col)`-dimensioned
   float64 array, which is not a valid CF grid mapping.
3. **`ensemble_std` at low source counts.** NaN where `n_sources < 2` (§3.2).

Non-blocking notebook feedback to raise separately: the ERA5-Land `sd` prose
describes it as snow depth treated as water-equivalent at density 1000, but
ERA5-Land `sd` *is* snow depth water equivalent in metres — the arithmetic is
right and only the explanation is wrong; and cell 41's print says "all-HRU
min/max" where the code is per-HRU.

## 8. Memory and size

Measured against the two real target files, not estimated.

| | days × HRUs | one float32 slab | today (3 vars) | with members |
|---|---|---|---|---|
| Oregon SWE (4 members) | 16,437 × 16,814 | 1.1 GB | 736 MB on disk | ~2.7 GB on disk |
| gfv2 SWE (4 members) | 16,802 × 361,471 | 24.3 GB | 12 GB on disk | ~36–48 GB on disk |

Per-stage pressure on the paths that do emit members:

- **Per-year SWE build.** The loader already materializes both the members dict
  *and* a full concat copy, so members were always resident; this design only
  keeps them alive through the write. Real delta is `ensemble_mean` /
  `ensemble_std`: +1.1 GB on gfv2, +50 MB on Oregon.
- **Stitch.** Already streams at ~256 MiB per variable chunk, so 3 → 9 variables
  moves peak from ~0.6 GiB to ~1.8 GiB.
- **Monthly targets.** `ds.compute()` is the pressure point; see §6 for the
  `--mem` response.

`maintenance rechunk` is a **backfill** tool for pre-#165 files; freshly built
targets already receive canonical chunking from `build_encoding(layer="target")`
at write time. A members-on gfv2 SWE target would only hit rechunk's eager
`.load()` if the backfill were deliberately run on it.

## 9. Testing

| File | Coverage added |
|---|---|
| `test_targets_common.py` | member emission; mean/std math; `n_sources < 2` std mask; all-NaN member for an absent source; bounds equal member min/max **for `multi_source_minmax` / `normalized_minmax` targets only** (not SCA, whose bounds are a CI interval) |
| `test_normalize_methods.py` | `complete_years_window`; per-source POR window selection |
| `test_targets_rch.py`, `test_targets_som.py` | sentinel path; `normalize_window_<src>` attrs |
| `test_defaults.py`, `test_init_run.py`, `test_upgrade_config.py` | `emit_members` schema; sentinel accepted by `validate` |
| `test_cf_compliance.py` | CF-1.8, `source_keys`, `members_emitted`, `ancillary_variables` |
| `test_targets_swe.py` | per-year schema stability across a year where a source is absent |

## 10. Documentation

Per the CLAUDE.md documentation sync gate: `CLAUDE.md`,
`docs/architecture/transformation-pipeline.md`,
`docs/references/calibration-target-recipes.md` (§2 must now state explicitly
why our MOD16A2 resampler differs from the notebook's),
`docs/architecture/nc-encoding-policy.md`, and the SLURM script headers.

## 11. PR sequence

1. Members + ensemble stats + `emit_members` — the schema change.
2. `per_source_por` + `complete_years_window` — normalization semantics.
3. Oregon project config + SLURM `--mem` bumps — no `src/` changes.
4. Notebook-feedback issues (§7) — filed, not edited, since it is their notebook.

## 12. Out of scope — tracked as separate issues

| Issue | Item |
|---|---|
| [#339](https://github.com/rmcd-mscb/nhf-spatial-targets/issues/339) | Release payload rglobs `targets/` and stages per-year build intermediates as published targets |
| [#340](https://github.com/rmcd-mscb/nhf-spatial-targets/issues/340) | Per-year target file layout for daily targets |
| [#341](https://github.com/rmcd-mscb/nhf-spatial-targets/issues/341) | Notebook feedback: MOD16A2 resampling prose/code mismatch, ERA5-Land `sd` description, per-HRU print string |
| [#342](https://github.com/rmcd-mscb/nhf-spatial-targets/issues/342) | CF: daily targets carry `cell_methods "time: point"` alongside `time_bnds` |

Not yet filed, pending a scope decision (§4.1): auditing `catalog/sources.yml`
`period:` fields for complete-year consistency with the pipeline's POR
definition.

Detail on the two largest deferred items:

**Per-year target file layout (#340).** The year-chunked driver already writes per-year
NCs to `targets/.<target>_intermediates/` and stitches them; publishing those
directly instead would make every gfv2 SWE file independently loadable
(~1.2 GB each) at identical total footprint. Deferred to its own spec because it
is a new on-disk layout: `rebuild_manifest.py:446` globs `targets/*.nc` one level
deep and would not see per-year targets, while `payload._nc_files` uses `rglob`
and would — and `_preflight_provenance_complete` compares those two sets. It
needs the projection extended, its test extended, and the release payload taught
to tell a published per-year target from a build intermediate.

**Release payload stages build intermediates (#339).**
`.swe_intermediates/` and `.sca_intermediates/` live inside `targets/`, and
`payload._nc_files` is `rglob("*.nc")` with no dot-directory filter anywhere in
`release/`. `plan_fabric_child` would therefore stage the OR project's ~45 SWE
and ~44 SCA per-year intermediates as published target files, none of which have
a `target` step in the manifest. Not yet confirmed by a dry-run publish whether
the gate blocks the release or the intermediates ship.
