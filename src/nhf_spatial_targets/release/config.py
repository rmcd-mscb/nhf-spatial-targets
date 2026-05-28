"""Per-project release config: validation + the scaffolded ``release.yml``.

The canonical per-project release metadata (authors, IPDS number, DOI,
abstract notes, fabric label) lives in each project's ``config.yml`` under
the ``release:`` key, default-merged by
:func:`nhf_spatial_targets.defaults.apply_defaults`.

``nhf-targets release init`` also scaffolds a standalone
``<project>/release/release.yml`` carrying the same block. It is written as
a separate, comment-headed file purely for git-friendly review: an operator
edits authors / IPDS number there and the diff is small and self-contained
rather than buried in the larger ``config.yml``. The scaffolded file
round-trips -- loading it back yields exactly the block that was written.

Validation here is **shape only** (types + recognized keys). ``authors``
being non-empty is required to *publish*, not to load; that gate lives in
the publish pre-flight, matching the note in
:data:`nhf_spatial_targets.defaults.DEFAULTS`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from nhf_spatial_targets.defaults import apply_defaults

# Recognized keys in a per-project ``release:`` block, mirroring
# ``defaults.DEFAULTS["release"]``. Unknown keys raise so a typo (e.g.
# ``ipds_no``) surfaces instead of silently doing nothing.
RELEASE_CONFIG_KEYS: tuple[str, ...] = (
    "authors",
    "ipds_number",
    "doi",
    "abstract_notes",
    "fabric_label",
)

# Recognized keys for a single author entry.
_AUTHOR_REQUIRED: tuple[str, ...] = ("given", "family")
_AUTHOR_OPTIONAL: tuple[str, ...] = ("orcid", "affiliation")
_AUTHOR_KEYS: frozenset[str] = frozenset(_AUTHOR_REQUIRED + _AUTHOR_OPTIONAL)

_SCALAR_KEYS: tuple[str, ...] = ("ipds_number", "doi", "abstract_notes", "fabric_label")


def _validate_authors(authors: object, *, source: str) -> None:
    if not isinstance(authors, list):
        raise ValueError(
            f"{source}: release.authors must be a list; got {type(authors).__name__}."
        )
    for i, author in enumerate(authors):
        if not isinstance(author, dict):
            raise ValueError(
                f"{source}: release.authors[{i}] must be a mapping with "
                f"keys {list(_AUTHOR_REQUIRED)} (+ optional "
                f"{list(_AUTHOR_OPTIONAL)}); got {type(author).__name__}."
            )
        unknown = sorted(set(author) - _AUTHOR_KEYS)
        if unknown:
            raise ValueError(
                f"{source}: release.authors[{i}] has unknown key(s) "
                f"{unknown}. Allowed: {sorted(_AUTHOR_KEYS)}."
            )
        for key in _AUTHOR_REQUIRED:
            value = author.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{source}: release.authors[{i}].{key} is required and "
                    f"must be a non-empty string."
                )
        for key in _AUTHOR_OPTIONAL:
            value = author.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"{source}: release.authors[{i}].{key} must be a string "
                    f"or null; got {type(value).__name__}."
                )


def validate_release_config(block: dict, *, source: str = "config.yml") -> None:
    """Raise ``ValueError`` if a per-project ``release:`` block is malformed.

    Checks types and recognized keys only. An empty ``authors`` list is
    permitted (publishing later requires it; loading does not).

    Parameters
    ----------
    block
        The ``release:`` mapping (already default-merged is fine).
    source
        Label used in error messages.

    Raises
    ------
    ValueError
        ``block`` is not a mapping, contains an unknown key, or any field
        has the wrong type / shape.
    """
    if not isinstance(block, dict):
        raise ValueError(
            f"{source}: release block must be a mapping; got {type(block).__name__}."
        )
    unknown = sorted(set(block) - set(RELEASE_CONFIG_KEYS))
    if unknown:
        raise ValueError(
            f"{source}: release block has unknown key(s) {unknown}. "
            f"Allowed: {list(RELEASE_CONFIG_KEYS)}."
        )
    _validate_authors(block.get("authors", []), source=source)
    for key in _SCALAR_KEYS:
        value = block.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"{source}: release.{key} must be a string or null; "
                f"got {type(value).__name__}."
            )


def _empty_block() -> dict:
    """Return the default (all-empty) release block."""
    return {
        "authors": [],
        "ipds_number": None,
        "doi": None,
        "abstract_notes": None,
        "fabric_label": None,
    }


def load_release_config(project_dir: Path) -> dict:
    """Load + validate the ``release:`` block from a project's ``config.yml``.

    Reads ``<project_dir>/config.yml``, applies the package defaults (so a
    project that omits ``release:`` still yields the full block), validates
    the result, and returns just the ``release`` sub-mapping.

    Raises
    ------
    FileNotFoundError
        ``config.yml`` is absent.
    ValueError
        ``config.yml`` is unparseable or the ``release:`` block is malformed.
    """
    config_path = project_dir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yml not found in {project_dir}. "
            f"Run 'nhf-targets init --project-dir {project_dir}' first."
        )
    try:
        user = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Cannot parse config.yml in {project_dir}: {exc}") from exc
    merged = apply_defaults(user)
    block = merged.get("release") or _empty_block()
    validate_release_config(block, source=str(config_path))
    return block


def load_release_yml(path: Path) -> dict:
    """Load + validate a scaffolded ``release.yml`` file.

    Returns the block with missing keys filled from the empty defaults, so
    a partially-edited file still yields the full set of recognized keys.

    Raises
    ------
    FileNotFoundError
        *path* does not exist.
    ValueError
        The file is unparseable or the block is malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"release.yml not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Cannot parse {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected a YAML mapping at the top level; "
            f"got {type(data).__name__}."
        )
    validate_release_config(data, source=str(path))
    block = _empty_block()
    block.update(data)
    return block


_RELEASE_YML_HEADER = """\
# Per-project ScienceBase release metadata for `nhf-targets release`.
#
# This file mirrors the `release:` block in config.yml but lives on its own
# so edits to authors / IPDS number / DOI get a small, reviewable diff.
# Fill `authors` and `ipds_number` before publishing; `doi` is populated
# after ScienceBase staff mints it post-IPDS approval. See docs/data_release/.
#
# authors is a list of mappings; `given` and `family` are required, `orcid`
# and `affiliation` optional. Example:
#
#   authors:
#     - given: Jane
#       family: Doe
#       orcid: 0000-0000-0000-0000
#       affiliation: U.S. Geological Survey
"""


def scaffold_release_yml(
    release_dir: Path,
    *,
    block: dict | None = None,
    overwrite: bool = False,
) -> Path:
    """Write ``<release_dir>/release.yml`` and return its path.

    The file is comment-headed for operator guidance, but the data keys
    are written as real YAML so :func:`load_release_yml` round-trips them.

    Parameters
    ----------
    release_dir
        The ``<project>/release`` directory. Created if missing.
    block
        Release block to serialize. Defaults to the empty block. Validated
        before writing so a malformed block fails loudly here rather than
        on the next load.
    overwrite
        When ``False`` (default) and the file already exists, leave it
        untouched and return its path -- never clobber an operator's edits.

    Raises
    ------
    ValueError
        *block* is malformed.
    """
    if block is None:
        block = {}
    validate_release_config(block, source="release.yml (scaffold)")

    release_dir.mkdir(parents=True, exist_ok=True)
    path = release_dir / "release.yml"
    if path.exists() and not overwrite:
        return path

    # Fill unspecified keys from the empty defaults so a partial block
    # serializes `authors: []` rather than `authors: null` (which would
    # fail re-validation on load). Serialize in canonical key order so the
    # scaffold reads top-to-bottom the way the docs describe it.
    full = _empty_block()
    full.update(block)
    ordered = {key: full[key] for key in RELEASE_CONFIG_KEYS}
    body = yaml.safe_dump(ordered, sort_keys=False, default_flow_style=False)
    path.write_text(_RELEASE_YML_HEADER + "\n" + body)
    return path
