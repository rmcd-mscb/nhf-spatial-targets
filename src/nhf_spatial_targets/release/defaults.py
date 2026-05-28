"""Load + validate ``catalog/release_defaults.yml``.

``release_defaults.yml`` holds the repo-wide FGDC / ScienceBase boilerplate
(publisher, distribution contact, license + keyword templates, abstract
templates) shared by every project's release. Per-project bits (authors,
IPDS number, DOI) live in each project's ``config.yml`` ``release:`` block
and are handled by :mod:`nhf_spatial_targets.release.config`.

Validation here is **shape only**: the eight canonical top-level sections
must be present and be mappings, and no unknown sections may appear. It
deliberately does *not* require the sections to be populated -- the
committed file ships as an empty scaffold and is filled in over PR-D/PR-E.
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
