# `upgrade-config` — report optional-config drift from the init template

## Motivation

PR #193 added two commented-out stubs to the project config template
(`fabric.token`, `representative_points`) after both surfaced as silent
footguns on the Oregon build (umbrella #182). Those stubs help **new**
operators (they see the option at `nhf-targets init` time) but do **nothing**
for operators of **existing** projects, who keep using a config that pre-dates
the addition and never discover the new optional features until something
fails or quietly produces wrong output.

This spec adds:
1. A `nhf-targets upgrade-config -d <project-dir>` CLI that **reports** which
   optional features documented in the init template are missing from a
   project's `config.yml`. **Report-only** — never mutates operator files.
2. A small explicit **registry** of optional features in code so the report
   message and the paste-block stay precise and reviewable.
3. CLAUDE.md guidance that ties the registry into the existing template /
   reference / test update discipline — one new step.

Non-goals: applying changes to operator files; auto-parsing arbitrary YAML
comments; managing required-config schema migration. All deferred.

## Component map

```
src/nhf_spatial_targets/upgrade_config.py     (new)
  OptionalConfigFeature                       (frozen dataclass)
  OPTIONAL_CONFIG_FEATURES                    (list[OptionalConfigFeature])
  check_drift(project_dir) -> list[OptionalConfigFeature]

src/nhf_spatial_targets/cli.py                (extend)
  @app.command("upgrade-config")
  def upgrade_config_cmd(workdir, ...): rich-table report + exit 0/1

tests/test_upgrade_config.py                  (new)
  detection clean + drift cases, CLI exit codes

CLAUDE.md                                     (extend)
  "Config schema additions" — 4-step checklist
```

## Data shape

```python
@dataclass(frozen=True)
class OptionalConfigFeature:
    name: str          # e.g. "fabric.token", "representative_points"
    detect: str        # regex; matches against the project's config.yml TEXT.
                       # Must allow commented-stub form (^\s*#?\s*<key>:) so an
                       # operator who already pasted the stub isn't reported.
    block: str         # literal template block to paste — kept in sync with
                       # _CONFIG_TEMPLATE by CLAUDE.md discipline (step 4).
    added: str         # "2026-05-23 (#193)" — provenance shown in the table.
    why: str           # one-line operator-facing reason (table column).
```

Initial registry entries (both from #193): `fabric.token`, `representative_points`.

## Detection contract

`check_drift(project_dir)` reads `<project_dir>/config.yml` as text and walks
`OPTIONAL_CONFIG_FEATURES`. For each feature, the `detect` regex runs against
the file text in MULTILINE mode. A feature is **in sync** if the regex
matches (either the live key or the commented stub form). Otherwise the
feature is returned as drift.

The regex form `(?m)^\s*#?\s*<key>\s*:` matches all three states an operator
might be in:
- live key (`token: or`)
- commented stub the operator pasted but hasn't enabled (`# token: or`)
- commented stub from the latest template (`# token: or`)

If the project's `config.yml` doesn't exist, `check_drift` raises
`FileNotFoundError` (the project dir is invalid; same error class as the
other CLI commands' "project not found" path).

## CLI contract

```
$ nhf-targets upgrade-config -d <project-dir>

[in sync]   → exit 0, prints "Project config is in sync with the latest
              optional-feature stubs in the init template."

[drift]     → exit 1, prints a rich Table (Feature / Added / Why) followed by
              the literal paste-block for each missing feature.
```

Exit 1 on drift lets CI scripts (e.g. a future "audit our known project dirs"
heartbeat) detect drift without parsing stdout.

## CLAUDE.md addition

A new subsection under **Code Conventions** (after the existing
"Test Coverage Rule" or similar):

```markdown
## Config schema additions

When you add a new optional parameter to the project config (top-level key
or nested commented stub), update all four in the same PR:

1. `src/nhf_spatial_targets/init_run.py:_CONFIG_TEMPLATE` — add the
   parameter as a commented stub (schema + example + reason it's optional).
2. `config/pipeline.yml` — mirror the addition.
3. `tests/test_init_run.py` — assert the new key/stub appears in the
   rendered template.
4. `src/nhf_spatial_targets/upgrade_config.py:OPTIONAL_CONFIG_FEATURES` —
   append an `OptionalConfigFeature` entry so existing-project operators
   see the addition via `nhf-targets upgrade-config -d <dir>`.

Steps 1-3 keep new projects current; step 4 keeps existing projects
discoverable.
```

## Tests

`tests/test_upgrade_config.py`:

- `test_check_drift_reports_missing_when_absent` — tmp project config with
  neither `fabric.token` nor `representative_points` → both reported.
- `test_check_drift_clean_when_live_value_present` — config with a real
  `representative_points: {...}` block → not reported.
- `test_check_drift_clean_when_commented_stub_present` — config with
  `# representative_points:` (operator pasted but hasn't enabled) → not
  reported.
- `test_check_drift_raises_when_config_missing` — bogus project dir →
  FileNotFoundError.
- `test_cli_exits_nonzero_on_drift` — Cyclopts invocation; assert exit code
  1 and that the missing feature's name appears in stdout.
- `test_cli_exits_zero_when_in_sync` — all features matched → exit 0,
  "in sync" message.

## Tradeoffs

- **Discipline-dependent.** A contributor who forgets step 4 of the
  CLAUDE.md checklist leaves drift undetected for their addition. Mitigation
  for a follow-up: a CI check that diffs the template's commented-stub
  lines against `OPTIONAL_CONFIG_FEATURES` and fails the build on
  mismatch. Out of scope for v1.
- **String matching, not YAML parsing.** Comment-preserving YAML round-trip
  is fragile; the recent #192 regex bug is fresh evidence. The registry
  pattern keeps the detection narrow and reviewable. Deeply-nested optional
  features (beyond the simple `fabric.token` pattern) would warrant
  revisiting.
- **No `--apply`.** Operators copy-paste manually. Trades convenience for
  zero risk of clobbering their customizations or comments.

## References

- #193 (init template stubs — the trigger for needing this)
- #182 (Oregon umbrella — operator-experience source for the registry's
  initial entries)
- #194 (validate silent-skip warnings — complementary preventive infra)
