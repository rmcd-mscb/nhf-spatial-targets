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
