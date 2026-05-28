"""Load + validate ``catalog/release_defaults.yml``.

``release_defaults.yml`` holds the repo-wide FGDC / ScienceBase boilerplate
(publisher, distribution contact, license + keyword templates, abstract
templates) shared by every project's release. Per-project bits (authors,
IPDS number, DOI) live in each project's ``config.yml`` ``release:`` block
and are handled by :mod:`nhf_spatial_targets.release.config`.

Validation here is **shape only**: the eight canonical top-level sections
must be present and be mappings, and no unknown sections may appear. It
deliberately does *not* require the sections to be populated -- the
committed file ships as an empty scaffold and is filled in over later phases.
The hard "fully populated" gate is a publish-time pre-flight check, not a
load-time one, so an operator can build + dry-run a payload before the
boilerplate is final.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Repo root is four parents up: release/ -> nhf_spatial_targets/ -> src/ -> root.
_CATALOG_DIR = Path(__file__).resolve().parents[3] / "catalog"
_DEFAULT_PATH = _CATALOG_DIR / "release_defaults.yml"

# Canonical top-level sections. Empty mappings are allowed (the scaffold
# ships them empty); unknown sections raise so a typo can't silently drop
# boilerplate from the rendered metadata.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "metadata",
    "contacts",
    "distribution",
    "keywords",
    "umbrella",
    "source",
    "fabric",
    "spatial_reference",
)

# Dotted paths into a defaults document that must be *populated* (truthy) for a
# release to be publishable. This is the publish-time gate the loader's
# shape-only check deliberately omits: the committed scaffold may ship empty
# sections and an operator builds/dry-runs against them, but publishing
# metadata with an empty title template or no distributor contact would emit a
# broken FGDC/ISO record. Every path here is read by a
# :mod:`nhf_spatial_targets.release.mcf` builder; an empty value renders an
# empty (and invalid) metadata field.
#
# A path segment navigates nested mappings; the leaf must be truthy after the
# same normalization the renderers see (non-empty string, list, or mapping; a
# nonzero number such as the EPSG code). Kept beside REQUIRED_SECTIONS so the
# two can't drift.
REQUIRED_POPULATED_PATHS: tuple[str, ...] = (
    "contacts.distributor",
    "contacts.point_of_contact",
    "distribution.use_constraints",
    "distribution.liability_statement",
    "distribution.fallback_landing_url",
    "distribution.distribution_protocol",
    "keywords.topiccategory",
    "keywords.umbrella.theme",
    "keywords.source.theme",
    "keywords.fabric.theme",
    "umbrella.title_template",
    "umbrella.abstract_template",
    "umbrella.purpose",
    "umbrella.lineage_statement",
    "umbrella.versioning_policy",
    "source.abstract_suffix",
    "source.purpose_template",
    "source.lineage_statement_template",
    "fabric.title_template",
    "fabric.abstract_template",
    "fabric.purpose_template",
    "fabric.lineage_statement_template",
    "spatial_reference.epsg",
)


def validate_release_defaults(
    data: dict, *, source: str = "release_defaults.yml"
) -> None:
    """Raise ``ValueError`` if *data* is not a well-shaped defaults document.

    Parameters
    ----------
    data
        Parsed contents of a ``release_defaults.yml`` document.
    source
        Path or label used in error messages (for pinpointing which file
        failed when several are validated in one run).

    Raises
    ------
    ValueError
        *data* is not a mapping; a required section is missing; a section
        is present but not a mapping; or an unknown section appears.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"{source}: expected a YAML mapping at the top level; "
            f"got {type(data).__name__}."
        )
    missing = [s for s in REQUIRED_SECTIONS if s not in data]
    if missing:
        raise ValueError(
            f"{source}: missing required section(s) {missing}. "
            f"Expected all of: {list(REQUIRED_SECTIONS)}."
        )
    unknown = sorted(set(data) - set(REQUIRED_SECTIONS))
    if unknown:
        raise ValueError(
            f"{source}: unknown top-level section(s) {unknown}. "
            f"Allowed: {list(REQUIRED_SECTIONS)}."
        )
    # An empty section may be written as `{}` or as a bare key (None).
    for section in REQUIRED_SECTIONS:
        value = data[section]
        if value is not None and not isinstance(value, dict):
            raise ValueError(
                f"{source}: section '{section}' must be a mapping; "
                f"got {type(value).__name__}."
            )


def _resolve_path(data: dict, dotted: str) -> object | None:
    """Navigate *dotted* (e.g. ``"umbrella.title_template"``) into *data*.

    Returns the leaf value, or ``None`` if any intermediate segment is absent
    or not a mapping. A literal ``None`` leaf is indistinguishable from a
    missing one here, which is fine: both are "not populated".
    """
    node: object = data
    for segment in dotted.split("."):
        if not isinstance(node, dict) or segment not in node:
            return None
        node = node[segment]
    return node


def _is_populated(value: object) -> bool:
    """Whether *value* counts as filled in for publish purposes.

    A string is populated only if it has non-whitespace content (folded YAML
    scalars can leave a bare newline); a list/mapping must be non-empty; a
    number (the EPSG code) is populated when nonzero. ``None`` is never
    populated.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return True


def missing_release_defaults(data: dict) -> list[str]:
    """Return the dotted paths in :data:`REQUIRED_POPULATED_PATHS` left empty.

    Pure inspection -- no I/O, no raising. An empty list means every
    publish-required field is populated. Used by both the publish pre-flight
    (which raises on a non-empty result) and the read-only dry-run report
    (which surfaces it without failing).
    """
    return [
        p for p in REQUIRED_POPULATED_PATHS if not _is_populated(_resolve_path(data, p))
    ]


def require_populated_release_defaults(
    data: dict, *, source: str = "release_defaults.yml"
) -> None:
    """Raise ``ValueError`` if any publish-required default is unpopulated.

    The hard "fully populated" gate the module docstring defers to publish
    time. Distinct from :func:`validate_release_defaults` (shape only): this
    fails when a required field is *present-but-empty*, which the loader
    deliberately tolerates so a half-filled scaffold can still build/dry-run.

    Raises
    ------
    ValueError
        One or more paths in :data:`REQUIRED_POPULATED_PATHS` are empty; the
        message lists every empty path so an operator fills them in one pass.
    """
    missing = missing_release_defaults(data)
    if missing:
        raise ValueError(
            f"{source}: release defaults are not fully populated; publishing "
            f"would emit incomplete metadata. Fill in: {missing}."
        )


def load_release_defaults(path: Path | None = None) -> dict:
    """Read + validate a ``release_defaults.yml`` document.

    Parameters
    ----------
    path
        Path to the defaults file. Defaults to the committed
        ``catalog/release_defaults.yml``. Tests pass a fixture path.

    Returns
    -------
    dict
        The validated document. Sections written as a bare key (parsed to
        ``None``) are normalized to an empty mapping so callers can index
        every section unconditionally.
    """
    if path is None:
        path = _DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(f"release defaults not found: {path}")
    data = yaml.safe_load(path.read_text())
    if data is None:
        data = {}
    validate_release_defaults(data, source=str(path))
    return {section: (data[section] or {}) for section in REQUIRED_SECTIONS}
