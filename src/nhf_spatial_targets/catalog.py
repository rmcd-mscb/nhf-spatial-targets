"""Data catalog interface — loads sources.yml and variables.yml."""

from __future__ import annotations

from pathlib import Path

import yaml

_CATALOG_DIR = Path(__file__).parent.parent.parent / "catalog"


def _load(filename: str) -> dict:
    path = _CATALOG_DIR / filename
    with path.open() as f:
        return yaml.safe_load(f)


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
