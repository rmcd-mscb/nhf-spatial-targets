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


# Allowed `fabric_scope.fabrics` tokens. Keep this set in sync with the
# fabrics this pipeline targets — adding a new fabric should be
# accompanied by extending this allow-list and the matching
# enforcement in the target builders.
FABRIC_SCOPE_TOKENS: frozenset[str] = frozenset({"or"})


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
