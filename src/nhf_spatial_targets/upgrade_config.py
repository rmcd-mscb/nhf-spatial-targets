"""Report optional-config drift between a project's config.yml and the latest
init template.

Existing projects don't pick up new optional features (e.g. ``fabric.token``,
``representative_points``) added to the init template, because they were
created before those features existed. This module is the operator-facing
discovery path: ``nhf-targets maintenance check-config -d <dir>`` lists what's missing,
with the literal commented block to paste.

**Report-only.** This module never mutates the operator's config.yml.

When you add a new optional config parameter (update
``init_run.py:_CONFIG_TEMPLATE`` + ``tests/test_init_run.py`` per the
CLAUDE.md "Config schema additions" checklist), also append an
:class:`OptionalConfigFeature` entry to :data:`OPTIONAL_CONFIG_FEATURES`
below so existing-project operators see it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from nhf_spatial_targets.defaults import DEFAULTS


@dataclass(frozen=True)
class OptionalConfigFeature:
    """A single optional-config addition tracked for drift reporting.

    Attributes
    ----------
    name
        The dotted key path as the operator should think of it
        (``fabric.token``, ``representative_points``). Shown in the table.
    detect
        A regex applied to the project's ``config.yml`` text in MULTILINE
        mode. The feature is considered in-sync if it matches **any** form:
        the live value, the commented-stub form the operator pasted, or the
        commented stub left from a fresh init. The canonical pattern is
        ``(?m)^\\s*#?\\s*<key>\\s*:``.
    block
        The literal commented stub from ``_CONFIG_TEMPLATE`` — what the
        operator sees in the ``check-config`` paste output. Kept in
        sync with the init template by the CLAUDE.md discipline.
    added
        Provenance shown in the report table (``"2026-05-23 (#193)"``).
    why
        One-line operator-facing reason this feature exists. Shown in the
        report table.
    """

    name: str
    detect: str
    block: str
    added: str
    why: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Append new entries here whenever a new optional config parameter lands in
# init_run.py:_CONFIG_TEMPLATE. See CLAUDE.md "Config schema additions" for
# the full update checklist.
#
# Detection regexes use ``(?m)^\s*#?\s*<key>\s*:`` so commented-stub forms
# (operator pasted but hasn't enabled) and live values both count as in-sync.

OPTIONAL_CONFIG_FEATURES: list[OptionalConfigFeature] = [
    OptionalConfigFeature(
        name="fabric.token",
        # Nested under `fabric:`; we match the leaf key only because there is
        # no other `token:` field in the schema today. If that changes,
        # tighten this to require fabric: context (e.g. multi-line lookbehind).
        detect=r"(?m)^\s*#?\s*token\s*:",
        block=(
            "  # Optional fabric-scope token. Sources tagged with `fabric_scope` in\n"
            "  # catalog/sources.yml (e.g. margulis_wus_sr -> [or]) are silently skipped\n"
            "  # at both agg and target stages when this is unset. Set to one of\n"
            '  # catalog.FABRIC_SCOPE_TOKENS (currently {"or"}) on fabric-restricted\n'
            "  # projects to opt in. Paste under the `fabric:` block.\n"
            "  # token: or\n"
        ),
        added="2026-05-23 (#193)",
        why="opt-in for fabric_scope sources (e.g. margulis_wus_sr on the OR fabric)",
    ),
    OptionalConfigFeature(
        name="representative_points",
        detect=r"(?m)^\s*#?\s*representative_points\s*:",
        block=(
            "# Per-target representative HRU points used by the inspect_* notebooks'\n"
            "# time-series cells. When unset, notebooks fall back to hardcoded CONUS\n"
            "# defaults that lie outside regional fabrics (lookup_hrus_by_points\n"
            '# raises). Set per-target lists of `"label": [lon, lat]` inside the\n'
            "# fabric to override. YAML anchors keep one set across targets.\n"
            "#\n"
            "# representative_points:\n"
            "#   aet: &rep_pts\n"
            '#     "Region A": [-121.7, 45.4]\n'
            '#     "Region B": [-123.0, 44.6]\n'
            '#     "Region C": [-118.6, 42.7]\n'
            '#     "Region D": [-123.5, 45.0]\n'
            "#   runoff: *rep_pts\n"
            "#   recharge: *rep_pts\n"
            "#   soil_moisture: *rep_pts\n"
            "#   snow_covered_area: *rep_pts\n"
            "#   swe: *rep_pts\n"
        ),
        added="2026-05-23 (#193)",
        why="per-target REPRESENTATIVE_POINTS for inspect_* notebooks (#192)",
    ),
    OptionalConfigFeature(
        name="release",
        # Top-level `release:` key (no other `release:` in the schema).
        detect=r"(?m)^\s*#?\s*release\s*:",
        block=(
            "# Per-project metadata for the `nhf-targets release` workflow. Most\n"
            "# defaults live in catalog/release_defaults.yml (committed repo-wide).\n"
            "# This block carries the per-project bits: authors, IPDS number, DOI\n"
            "# (filled after ScienceBase staff mints it post-IPDS approval), and an\n"
            "# optional fabric-label override. Leave commented until ready to publish.\n"
            "# See docs/data_release/ for the full workflow.\n"
            "#\n"
            "# release:\n"
            "#   authors:\n"
            "#     - given: Jane\n"
            "#       family: Doe\n"
            "#       orcid: 0000-0000-0000-0000\n"
            "#       affiliation: U.S. Geological Survey\n"
            "#   ipds_number: IP-XXXXXX\n"
            "#   doi: null               # populated after SB staff mints DOI\n"
            "#   abstract_notes: null    # optional paragraph appended to fabric abstract\n"
            "#   fabric_label: null      # human-readable name; defaults to Path(fabric.path).stem\n"
        ),
        added="2026-05-26 (#243)",
        why="ScienceBase data-release metadata for `nhf-targets release` (#241)",
    ),
    OptionalConfigFeature(
        name="snow_covered_area.depth_threshold_mm",
        # Leaf key, unique to snow_covered_area in the schema.
        detect=r"(?m)^\s*#?\s*depth_threshold_mm\s*:",
        block=(
            "    # Pixel snow-depth cutoff (mm) for ua_swe's snow_covered_fraction\n"
            "    # binary. Consumed at the ua_swe AGGREGATE stage (changing it requires\n"
            "    # re-aggregating ua_swe; the publish gate refuses a stamp that\n"
            "    # disagrees with this value). Paste under `targets.snow_covered_area:`.\n"
            "    # depth_threshold_mm: 1.0\n"
        ),
        added="2026-06-10 (#237)",
        why="pixel snow-depth cutoff for ua_swe's SCA snow_covered_fraction (#237)",
    ),
    OptionalConfigFeature(
        name="snow_covered_area.forced_zero_combined",
        detect=r"(?m)^\s*#?\s*forced_zero_combined\s*:",
        block=(
            "    # true: zero the COMBINED SCA bound in Jul/Aug (calcSCA). false: zero\n"
            "    # only the mod10c1 contribution pre-combine, so ua_swe's summer alpine\n"
            "    # fraction survives into the combined upper bound. Paste under\n"
            "    # `targets.snow_covered_area:`.\n"
            "    # forced_zero_combined: true\n"
        ),
        added="2026-06-10 (#237)",
        why="July/August forced-zero policy for the multi-source SCA bound (#237)",
    ),
    OptionalConfigFeature(
        name="snow_covered_area.min_sources_for_bound",
        detect=r"(?m)^\s*#?\s*min_sources_for_bound\s*:",
        block=(
            "    # 1: keep an SCA bound wherever >=1 source is finite (a single\n"
            "    # source's degenerate [v, v] is allowed). 2: NaN the combined bound\n"
            "    # where fewer than 2 sources are finite. Paste under\n"
            "    # `targets.snow_covered_area:`.\n"
            "    # min_sources_for_bound: 1\n"
        ),
        added="2026-06-10 (#237)",
        why="zero-width-bound policy for the multi-source SCA bound (#237)",
    ),
]


def check_drift(project_dir: Path) -> list[OptionalConfigFeature]:
    """Return the optional features missing from ``<project_dir>/config.yml``.

    A feature is considered present if its ``detect`` regex matches anywhere
    in the project's config text (live value or commented stub).

    Raises
    ------
    FileNotFoundError
        If ``<project_dir>/config.yml`` does not exist — the project dir is
        invalid for upgrade purposes (init it first, or fix the path).
    """
    cfg_path = Path(project_dir) / "config.yml"
    text = cfg_path.read_text()  # raises FileNotFoundError naturally
    return [
        feat for feat in OPTIONAL_CONFIG_FEATURES if not re.search(feat.detect, text)
    ]


# ---------------------------------------------------------------------------
# Whole-target + new-source hints (issue #279, PR-5)
# ---------------------------------------------------------------------------
#
# Beyond the enumerated optional params above, check-config also surfaces two
# coarser-grained drifts, both report-only:
#   - a whole target in the defaults schema that the operator's config omits;
#   - a catalog source eligible for a target (per variables.yml) that the
#     operator has not pinned in targets.<t>.sources[].

# Config target key -> catalog/variables.yml top-level key. Currently identity
# for every target: the config schema (defaults.py "targets") and the variable
# registry (variables.yml) both use the long names (recharge, soil_moisture,
# snow_covered_area, ...). The rch/som/sca/swe divergence lives only in the
# targets/ builder module names, NOT in these keys. Kept explicit so a future
# rename of either side is a one-line change here rather than a silent mis-map;
# consistency (coverage of DEFAULTS["targets"] + each value resolving in
# variables()) is asserted by
# tests/test_upgrade_config.py::test_target_to_variable_mapping_is_consistent.
_TARGET_TO_VARIABLE: dict[str, str] = {
    "runoff": "runoff",
    "aet": "aet",
    "recharge": "recharge",
    "soil_moisture": "soil_moisture",
    "snow_covered_area": "snow_covered_area",
    "snow_water_equivalent": "snow_water_equivalent",
}


def _load_user_targets(project_dir: Path) -> dict:
    """Return the operator's literal ``targets:`` mapping from config.yml.

    Reads the on-disk config verbatim (NOT merged with defaults) so the report
    reflects what the operator actually wrote. Returns ``{}`` when there is no
    ``targets:`` mapping. Raises ``FileNotFoundError`` if config.yml is absent
    (mirrors :func:`check_drift`).
    """
    cfg_path = Path(project_dir) / "config.yml"
    raw = yaml.safe_load(cfg_path.read_text())  # raises FileNotFoundError
    if not isinstance(raw, dict):
        return {}
    targets = raw.get("targets")
    return targets if isinstance(targets, dict) else {}


def check_missing_targets(project_dir: Path) -> list[str]:
    """Return target keys in the defaults schema absent from the operator config.

    A whole target present in :data:`~nhf_spatial_targets.defaults.DEFAULTS`
    but missing from the operator's ``config.yml`` ``targets:`` mapping is a
    report-only hint — the operator may want to add and configure it. Order
    follows the defaults schema for stable output.

    Raises
    ------
    FileNotFoundError
        If ``<project_dir>/config.yml`` does not exist.
    """
    user_targets = _load_user_targets(project_dir)
    return [t for t in DEFAULTS["targets"] if t not in user_targets]


def check_available_sources(project_dir: Path) -> dict[str, list[str]]:
    """Return, per target, eligible catalog sources not pinned in the config.

    For each target where the operator pinned an explicit ``sources:`` list,
    report the catalog sources that ``catalog/variables.yml`` lists as eligible
    for that target's variable but which are absent from the operator's list —
    i.e. "a catalog source is available for target X that you are not using."
    A report-only hint, never a failure.

    Targets without an explicit ``sources:`` list are skipped: they track the
    defaults and so cannot fossilize against a newly-added catalog source, so
    reporting them would be noise.

    Eligibility is taken **verbatim** from ``variables.yml`` with no
    ``superseded_by`` filtering (unlike the loud check in ``validate``). This is
    correct today because the per-variable ``sources`` lists name only current
    keys; if a superseded key ever lands in one, this hint would suggest adding
    a dead source — fix the catalog, not this function.

    Raises
    ------
    FileNotFoundError
        If ``<project_dir>/config.yml`` does not exist.
    """
    from nhf_spatial_targets.catalog import variables

    user_targets = _load_user_targets(project_dir)
    all_vars = variables()
    out: dict[str, list[str]] = {}
    for tgt_name, var_key in _TARGET_TO_VARIABLE.items():
        tgt_cfg = user_targets.get(tgt_name)
        if not isinstance(tgt_cfg, dict):
            continue
        configured = tgt_cfg.get("sources")
        if not isinstance(configured, list):
            continue  # on defaults for sources -> nothing pinned to drift
        # _TARGET_TO_VARIABLE values are guaranteed to resolve in variables()
        # by test_target_to_variable_mapping_is_consistent; the guard is purely
        # defensive against an unsynced future edit.
        var_def = all_vars.get(var_key)
        if var_def is None:
            continue
        eligible = var_def.get("sources") or []
        missing = [s for s in eligible if s not in configured]
        if missing:
            out[tgt_name] = missing
    return out
