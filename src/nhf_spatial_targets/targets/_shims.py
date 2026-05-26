"""Per-source shim contract for multi-source target builders.

A :class:`SourceShim` co-locates the four facts each target needs about
a contributing source: the catalog/storage key, the variable name in the
aggregated NC, the human-readable label for the output NC's global
``source`` attr, and the per-source unit-shim function that brings the
native variable into the target's common intermediate units (e.g.
mm/month for runoff and AET).

Target modules declare a tuple of :class:`SourceShim` instances as their
``SHIMS`` constant; the driver calls :func:`shims_by_config_label`
(user-facing) or :func:`shims_by_key` (on-disk) to look one up by key
inside the build loop. Adding a future source is a single edit — there
is no parallel-dict drift surface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import xarray as xr


@dataclass(frozen=True)
class SourceShim:
    """Per-source contract for a multi-source target builder.

    Attributes
    ----------
    source_key
        On-disk storage key. Used as the subdirectory name under
        ``<project>/data/aggregated/`` and as the prefix of the per-year
        NC filename. Matches the catalog source key in the common case,
        but may be a synthetic key (e.g. ``"era5_land_sd"``) when one
        upstream source produces multiple aggregated cadences in
        separate subdirs — see ``aggregate/era5_land.py``'s
        ``ADAPTER_SD``.
    aggregated_var
        Variable name to extract from the aggregated NC (e.g. ``"et"``).
    description
        Human-readable label for the output NC's global ``source`` attr.
    to_common_units
        Callable that converts the source's native variable to the
        target's common intermediate units. For multi_source_minmax
        targets the common intermediate is usually mm/month; the per-
        target unit chain then applies a final linear conversion (e.g.
        mm/month → cfs for runoff, mm/month → inches/day for AET).
    config_label
        Optional user-facing alias used in the project config's
        ``<target>.sources`` list. Defaults to ``source_key``. Set this
        when the storage key is a synthetic disambiguation (e.g.
        ``"era5_land_sd"``) but the config should keep the canonical
        catalog name (``"era5_land"``) — this keeps target builders
        free of any parallel label→storage dict that could drift from
        the SHIMS registry.
    catalog_source_key
        Optional ``catalog/sources.yml`` key for the source. Defaults
        to ``config_label`` if set, else ``source_key``. Used by
        :func:`validate_source_units` to look up the catalog
        ``cf_units`` for ``aggregated_var``. Set this only when the
        catalog key is genuinely independent of both the on-disk
        storage key and the user-facing config alias — e.g. a shim
        named ``foo_v2`` with ``config_label="legacy_foo"`` (a config
        alias) but catalog data still under ``foo_v2``.
    expected_cf_units
        Optional CF-style units string this shim was written against.
        When set, :func:`validate_source_units` (called by every
        target builder at startup) asserts the catalog's ``cf_units``
        for ``aggregated_var`` matches exactly. Catches drift between
        the catalog and the shim's hardcoded conversion factor — e.g.
        a post-hoc cf_units correction that the shim hasn't been
        updated for. ``None`` opts out of the check (used for
        synthetic source keys whose aggregated variable is not the
        same name as anything in the catalog).
    """

    source_key: str
    aggregated_var: str
    description: str
    to_common_units: Callable[[xr.DataArray], xr.DataArray]
    config_label: str | None = None
    catalog_source_key: str | None = None
    expected_cf_units: str | None = None


def shims_by_key(shims: "tuple[SourceShim, ...]") -> "dict[str, SourceShim]":
    """Index a ``SHIMS`` tuple by ``source_key`` for lookup at build time.

    Raises ``ValueError`` if two shims share the same ``source_key`` —
    that would silently shadow one entry, which is exactly the kind of
    drift this refactor prevents.
    """
    out: dict[str, SourceShim] = {}
    for shim in shims:
        if shim.source_key in out:
            raise ValueError(
                f"Duplicate SourceShim.source_key={shim.source_key!r} in "
                f"target SHIMS registry. Each source may appear at most once."
            )
        out[shim.source_key] = shim
    return out


def shims_by_config_label(
    shims: "tuple[SourceShim, ...]",
) -> "dict[str, SourceShim]":
    """Index a ``SHIMS`` tuple by ``config_label`` (defaulted to ``source_key``).

    Used by target builders to resolve a user-facing source name from
    the project config (e.g. ``"era5_land"``) to the SourceShim that
    encodes its storage key, aggregated variable, and unit shim. Raises
    ``ValueError`` on duplicate labels — two shims wanting the same
    config label is the same drift class :func:`shims_by_key` guards.
    """
    out: dict[str, SourceShim] = {}
    for shim in shims:
        label = shim.config_label or shim.source_key
        if label in out:
            raise ValueError(
                f"Duplicate SourceShim config_label={label!r} in target "
                f"SHIMS registry. Each label may appear at most once "
                f"(set distinct config_label values to disambiguate)."
            )
        out[label] = shim
    return out


def validate_source_units(
    shims: "tuple[SourceShim, ...]",
    sources: "list[str] | tuple[str, ...]",
) -> None:
    """Assert each requested source's catalog ``cf_units`` matches the shim.

    Resolves each entry in ``sources`` (a list of config labels, e.g.
    ``["era5_land", "gldas_noah_v21_monthly"]`` from the project config)
    to its :class:`SourceShim`, then looks up the catalog ``cf_units``
    for the shim's ``aggregated_var`` under the shim's catalog source
    key (``catalog_source_key`` if set, else ``config_label``, else
    ``source_key``) and raises if the strings differ.

    Called at the top of every target builder's ``build()`` so a
    post-hoc catalog correction (PR #68 caught four such corrections)
    fails loud at startup rather than silently producing values off by
    the missed conversion factor. Per-source unit shims encode an
    assumption — e.g. ``gldas_to_mm_per_month`` assumes ``cf_units:
    "kg m-2"`` so it can apply ``× 8 × days_in_month``. If the catalog
    is corrected to a different string (or the shim's conversion is
    updated without touching ``expected_cf_units``) this validator
    raises with both strings so the operator can decide which side
    needs to follow the other.

    Shims with ``expected_cf_units=None`` opt out (e.g. legacy stubs
    pending unit harmonisation). Unknown source labels raise too —
    catching a project-config typo at startup is the same drift class
    this validator guards.

    Parameters
    ----------
    shims
        The target's ``SHIMS`` tuple.
    sources
        Project-config-side source labels for which the build is
        about to read data. Typically ``runoff_cfg["sources"]``,
        ``aet_cfg["sources"]``, etc.

    Raises
    ------
    ValueError
        Catalog ``cf_units`` differs from ``shim.expected_cf_units``
        for any requested source, or a requested source label has no
        matching shim.
    """
    from nhf_spatial_targets import catalog as cat

    by_label = shims_by_config_label(shims)
    for label in sources:
        if label not in by_label:
            raise ValueError(
                f"validate_source_units: no matching SourceShim for source "
                f"{label!r}. Known labels: {sorted(by_label)}."
            )
        shim = by_label[label]
        if shim.expected_cf_units is None:
            continue
        catalog_key = shim.catalog_source_key or shim.config_label or shim.source_key
        try:
            actual = cat.source_var_cf_units(catalog_key, shim.aggregated_var)
        except KeyError as exc:
            raise ValueError(
                f"validate_source_units: cannot resolve cf_units for "
                f"{catalog_key!r}/{shim.aggregated_var!r}: {exc}. Either "
                f"add cf_units to catalog/sources.yml or set "
                f"expected_cf_units=None on the shim to opt out."
            ) from exc
        if actual != shim.expected_cf_units:
            raise ValueError(
                f"Catalog cf_units drift for {catalog_key!r}/"
                f"{shim.aggregated_var!r}: catalog has {actual!r}, "
                f"shim {shim.source_key!r} expects "
                f"{shim.expected_cf_units!r}. Update the shim if the "
                f"units changed intentionally, or correct the catalog."
            )
