"""Report optional-config drift between a project's config.yml and the latest
init template.

Existing projects don't pick up new optional features (e.g. ``fabric.token``,
``representative_points``) added to the init template, because they were
created before those features existed. This module is the operator-facing
discovery path: ``nhf-targets upgrade-config -d <dir>`` lists what's missing,
with the literal commented block to paste.

**Report-only.** This module never mutates the operator's config.yml.

When you add a new optional config parameter (update
``init_run.py:_CONFIG_TEMPLATE`` + ``config/pipeline.yml`` +
``tests/test_init_run.py`` per the CLAUDE.md "Config schema additions"
checklist), also append an :class:`OptionalConfigFeature` entry to
:data:`OPTIONAL_CONFIG_FEATURES` below so existing-project operators see it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


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
        operator sees in the ``--upgrade-config`` paste output. Kept in
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
