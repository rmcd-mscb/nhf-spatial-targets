"""Data catalog interface — loads sources.yml and variables.yml."""

from __future__ import annotations

from pathlib import Path

import yaml

_CATALOG_DIR = Path(__file__).parent.parent.parent / "catalog"


def _load(filename: str) -> dict:
    """Load a YAML catalog file and return its contents."""
    path = _CATALOG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog file not found: {path}  (expected in {_CATALOG_DIR})"
        )
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


def sources() -> dict:
    """Return the full sources registry."""
    return _load("sources.yml")["sources"]


def variables() -> dict:
    """Return the full variable definitions."""
    return _load("variables.yml")["variables"]


def source(name: str) -> dict:
    """Return a single source entry by key."""
    all_sources = sources()
    if name not in all_sources:
        raise KeyError(f"Source '{name}' not found in catalog/sources.yml")
    return all_sources[name]


def variable(name: str) -> dict:
    """Return a single variable entry by key."""
    all_vars = variables()
    if name not in all_vars:
        raise KeyError(f"Variable '{name}' not found in catalog/variables.yml")
    return all_vars[name]


def source_var_cf_units(source_key: str, var_name: str) -> str:
    """Return the ``cf_units`` string for ``var_name`` under source ``source_key``.

    Resolves the variable entry by ``name`` first and falls back to
    ``file_variable`` (the on-disk name carried by sources like Reitz 2017
    whose distribution variables differ from the catalog name). Only the
    dict-form variables shape is supported — flat-string variable lists
    have no per-variable units field and are kept only on superseded
    sources; target builders that pin units must consume sources that
    declare ``cf_units`` explicitly.

    Used by target builders at startup to assert that catalog-declared
    per-pixel units still match the per-source unit-shim's expectation
    (issue #130). Catching drift here turns a silent magnitude bug into
    a loud startup failure with the source-and-units context the
    operator needs to fix it.

    Parameters
    ----------
    source_key
        Catalog source key (e.g. ``"era5_land"``).
    var_name
        Variable name to look up; matched against each entry's ``name``,
        then ``file_variable``.

    Raises
    ------
    KeyError
        Source has no variable matching ``var_name``, or the matching
        entry is in flat-string form (no ``cf_units`` available).
    """
    src = source(source_key)
    for entry in src.get("variables", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("name") == var_name or entry.get("file_variable") == var_name:
            cf_units = entry.get("cf_units")
            if cf_units is None:
                raise KeyError(
                    f"Catalog source {source_key!r} variable {var_name!r} "
                    f"has no 'cf_units' field. Add cf_units to "
                    f"catalog/sources.yml so target builders can validate "
                    f"unit drift at startup."
                )
            return cf_units
    raise KeyError(
        f"Variable {var_name!r} not found (as 'name' or 'file_variable') "
        f"in catalog source {source_key!r}."
    )


def source_var_cell_methods(source_key: str, var_name: str) -> str | None:
    """Return the ``cell_methods`` string for ``var_name`` under ``source_key``.

    Mirrors :func:`source_var_cf_units` but returns the CF ``cell_methods``
    string (e.g. ``"time: sum"`` for accumulated fields,
    ``"time: point"`` for instantaneous snapshots). Returns ``None`` when
    the variable entry exists but does not declare ``cell_methods`` —
    callers decide whether to raise on missing metadata, since some
    superseded sources omit it.

    Used by fetchers and aggregators that need to dispatch on whether
    a variable is accumulated vs instantaneous (e.g. ERA5-Land's
    hourly→daily and daily→monthly reducers) without duplicating the
    accumulated/instantaneous mapping in Python.

    Parameters
    ----------
    source_key
        Catalog source key (e.g. ``"era5_land"``).
    var_name
        Variable name to look up; matched against each entry's ``name``,
        then ``file_variable``.

    Returns
    -------
    str or None
        The ``cell_methods`` string, or ``None`` when the entry exists
        but has no ``cell_methods`` field.

    Raises
    ------
    KeyError
        Source has no variable matching ``var_name``, or the matching
        entry is in flat-string form (no per-variable metadata).
    """
    src = source(source_key)
    for entry in src.get("variables", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("name") == var_name or entry.get("file_variable") == var_name:
            return entry.get("cell_methods")
    raise KeyError(
        f"Variable {var_name!r} not found (as 'name' or 'file_variable') "
        f"in catalog source {source_key!r}."
    )


# Allowed `fabric_scope.fabrics` tokens. Keep this set in sync with the
# fabrics this pipeline targets — adding a new fabric should be
# accompanied by extending this allow-list and the matching
# enforcement in the target builders.
FABRIC_SCOPE_TOKENS: frozenset[str] = frozenset({"or"})

# Allowed `release.distribution_kind` tokens.
#
# - "data": the consolidated-source child item carries the source's
#   consolidated NetCDF files (the default; most sources).
# - "metadata_only": the child item carries README + FGDC + an upstream
#   pointer but no data payload. Used for sources whose primary
#   distribution is elsewhere (e.g. daymet, hosted on ORNL DAAC as
#   multi-TB zarr stores that we do not redistribute).
DISTRIBUTION_KIND_TOKENS: frozenset[str] = frozenset({"data", "metadata_only"})

# Release-block default values applied when a source omits one or more keys
# from its optional ``release:`` block. Kept in one place so the FGDC
# builder and the publishability filter agree on defaults.
_RELEASE_DEFAULTS: dict = {
    "publishable": True,
    "distribution_kind": "data",
    "notes": None,
}


def validate_fabric_scope(source_key: str, scope: dict | None) -> None:
    """Raise ValueError if a source's ``fabric_scope`` block is malformed.

    The check is intentionally narrow:

    - ``scope`` may be ``None`` (no scope, source available everywhere).
    - When present, ``scope`` must be a mapping with a non-empty list
      ``fabrics`` whose entries are all in :data:`FABRIC_SCOPE_TOKENS`.

    This catches typos (`fabrics: [oregon]`) that would otherwise
    silently disable scoping at the target-builder boundary.

    Parameters
    ----------
    source_key : str
        Catalog source key (for error messages).
    scope : dict or None
        The value of ``sources[source_key].fabric_scope``.

    Raises
    ------
    ValueError
        ``scope`` is not a mapping, ``fabrics`` is missing/empty/not a
        list, or contains tokens outside :data:`FABRIC_SCOPE_TOKENS`.
    """
    if scope is None:
        return
    if not isinstance(scope, dict):
        raise ValueError(
            f"sources[{source_key!r}].fabric_scope must be a mapping; "
            f"got {type(scope).__name__}."
        )
    fabrics = scope.get("fabrics")
    if not isinstance(fabrics, list) or not fabrics:
        raise ValueError(
            f"sources[{source_key!r}].fabric_scope.fabrics must be a "
            f"non-empty list; got {fabrics!r}."
        )
    unknown = [f for f in fabrics if f not in FABRIC_SCOPE_TOKENS]
    if unknown:
        raise ValueError(
            f"sources[{source_key!r}].fabric_scope.fabrics contains "
            f"unknown token(s) {unknown}. Allowed: "
            f"{sorted(FABRIC_SCOPE_TOKENS)}. If you intended to add a "
            f"new fabric, extend FABRIC_SCOPE_TOKENS in catalog.py."
        )


def validate_release_block(source_key: str, release: dict | None) -> None:
    """Raise ValueError if a source's ``release`` block is malformed.

    The check is intentionally narrow:

    - ``release`` may be ``None`` (no block, defaults apply).
    - When present, ``release`` must be a mapping. Recognized keys are
      ``publishable`` (bool), ``distribution_kind`` (member of
      :data:`DISTRIBUTION_KIND_TOKENS`), and ``notes`` (string).
    - Unknown keys raise — catches typos like ``publishible`` that would
      otherwise silently fall through to the default.

    Parameters
    ----------
    source_key : str
        Catalog source key (for error messages).
    release : dict or None
        The value of ``sources[source_key].release``.

    Raises
    ------
    ValueError
        ``release`` is not a mapping; ``publishable`` is not a bool;
        ``distribution_kind`` is not in :data:`DISTRIBUTION_KIND_TOKENS`;
        ``notes`` is not a string; or there are unknown keys.
    """
    if release is None:
        return
    if not isinstance(release, dict):
        raise ValueError(
            f"sources[{source_key!r}].release must be a mapping; "
            f"got {type(release).__name__}."
        )
    unknown = set(release) - set(_RELEASE_DEFAULTS)
    if unknown:
        raise ValueError(
            f"sources[{source_key!r}].release contains unknown key(s) "
            f"{sorted(unknown)}. Allowed: {sorted(_RELEASE_DEFAULTS)}."
        )
    pub = release.get("publishable")
    if pub is not None and not isinstance(pub, bool):
        raise ValueError(
            f"sources[{source_key!r}].release.publishable must be a bool; "
            f"got {type(pub).__name__}."
        )
    kind = release.get("distribution_kind")
    if kind is not None and kind not in DISTRIBUTION_KIND_TOKENS:
        raise ValueError(
            f"sources[{source_key!r}].release.distribution_kind must be one of "
            f"{sorted(DISTRIBUTION_KIND_TOKENS)}; got {kind!r}."
        )
    notes = release.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(
            f"sources[{source_key!r}].release.notes must be a string; "
            f"got {type(notes).__name__}."
        )


def release_block(source_key: str) -> dict:
    """Return the ``release`` block for ``source_key`` with defaults applied.

    Returns a fresh dict with the four canonical keys (``publishable``,
    ``distribution_kind``, ``notes``) populated from the source's
    optional ``release:`` block plus :data:`_RELEASE_DEFAULTS` fallbacks.
    Validates the on-disk block on every call — catches typos at the same
    boundary the FGDC builder reads from.

    Raises
    ------
    ValueError
        The source declares a malformed ``release`` block (see
        :func:`validate_release_block`).
    """
    src = source(source_key)
    raw = src.get("release")
    validate_release_block(source_key, raw)
    out = dict(_RELEASE_DEFAULTS)
    if isinstance(raw, dict):
        out.update({k: v for k, v in raw.items() if v is not None})
    return out


def publishable_sources() -> dict:
    """Return the subset of sources eligible for the ScienceBase release.

    A source is eligible iff:

    - it carries no ``superseded_by`` field — superseded sources never
      ship as catalog-archive entries only (this is the same marker
      :func:`test_every_current_source_has_status_field` uses, robust
      against status-string variants like ``"superseded_registration_required"``); AND
    - its ``release.publishable`` is True (the default).

    The returned dict mirrors :func:`sources` but excludes filtered keys.
    Order is preserved from the underlying YAML to keep deterministic
    output across release builds.

    Note: this filter is *informational* -- it does not raise on a
    non-publishable source. Downstream release stages (payload staging,
    FGDC emission, publish) own the hard gates; callers that need a
    hard gate should compose with those rather than assume this
    function's return value is acted on yet.
    """
    out: dict = {}
    for key, src in sources().items():
        if src.get("superseded_by") is not None:
            continue
        rb = release_block(key)
        if rb["publishable"]:
            out[key] = src
    return out
