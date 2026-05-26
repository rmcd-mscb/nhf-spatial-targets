"""Declarative adapter for calibration-target builders.

A :class:`TargetAdapter` describes a target's framing — its config key,
the cadence at which bounds are emitted, the units / cell_methods /
long_name template the writer needs, optional year-chunking +
intermediates layout, and a builder-supplied ``source_loader`` callable
that produces the per-source contributions. The companion driver
(``targets/_driver.py``) consumes the adapter and runs the shared
pipeline so each per-target module stays focused on the source-specific
parts (unit shims, normalization, source filtering).

Analog of ``aggregate/_adapter.py:SourceAdapter`` on the target side.
There the adapter declared the aggregation framing (variables, CRS,
hooks, stat_method) and the driver ran the gdptools loop; here the
adapter declares the target framing (cadence, units, year-chunking) and
the driver runs the read → combine → write pipeline.

Field rationale
---------------

Every field below is justified by an existing builder. Where two
builders disagree on a behavior, the adapter encodes the choice as a
named field rather than special-casing in the driver — this keeps "what
does this target actually do" answerable by reading the adapter alone.

- ``target_key`` / ``config_key`` — the short label used in log messages
  / CLI dispatch ("sca") and the key under ``project.config["targets"]``
  ("snow_covered_area"). These are decoupled because the CLI uses the
  short form (``pixi run run-sca``) but the project config uses the
  full TM 6-B10 name; the adapter is the one place both forms appear
  together.

- ``cadence`` — drives the master-index frequency, the ``time_bnds``
  offset, and the read-chunk hint. Picking a single ``Literal`` here
  rather than three separate fields keeps the three coupled values from
  drifting (a "daily" cadence wired to ``MonthBegin(1)`` would silently
  produce a year-wide time_bnds for every day).

- ``bounds_units`` / ``bounds_long_name_kind`` / ``cell_methods`` —
  forwarded to :func:`_writers.write_bounds_target`. The triple lives
  on the adapter because every builder hard-coded it identically in
  per-builder constants before this refactor; collecting them here is
  what makes "add a new target" a config-file-like exercise rather than
  a code-copy exercise.

- ``title`` / ``nn_title`` — the unfilled + NN-filled NC's CF ``title``
  attr. SOM is the only builder that overrides on a per-variant basis
  (monthly vs annual); for that one case the SOM builder builds the
  adapter twice with different titles rather than the adapter growing
  a variant-aware title template.

- ``references`` — every existing builder hard-codes
  ``"Hay et al. 2022, doi:10.3133/tm6B10"`` in its ``extra_attrs``;
  the adapter defaults to that string and SCA / SWE extend it with
  their additional reference (PRMSobjfun.f90 and Markstrom et al.
  respectively).

- ``needs_hru_area`` — only runoff needs per-HRU area for the
  mm/month → cfs unit conversion; every other target computes centroids
  only. The driver consults this flag to pick
  :func:`compute_hru_area_and_centroids` vs :func:`compute_hru_centroids`.

- ``year_chunked`` + ``intermediates_subdir`` + ``intermediate_base`` +
  ``per_year`` — gate the year-chunked code path. Without
  ``year_chunked=True`` the single-shot build runs (runoff / AET /
  recharge / SOM). With it, the driver runs the per-year loop with
  fingerprint-cached skip, orphan pruning, and a stitch pass. SCA and
  SWE set this; the daily cadence × multi-decade × multi-source product
  would otherwise materialise O(100 GB).

- ``forced_zero_months`` — SCA's PRMSobjfun ``calcSCA`` rule: July /
  August (months 7, 8) are forced to ``(lower, upper) = (0, 0)`` for
  every CI-passing HRU. Encoded as a tuple on the adapter (with the
  ``()`` default meaning "no forced zero") rather than as a special
  case in the driver: any future target that wants a seasonal forced-
  zero rule sets the months and the driver applies them uniformly.

- ``source_loader`` — the one place builder-specific code is required.
  The callable receives the project + adapter + per-year context (when
  year-chunked) and returns the ``(lower, upper, n_sources,
  n_sources_count, time_index, time_offset_unit, extra_attrs)``
  tuple the driver hands to :func:`_writers.write_bounds_target`. This
  keeps unit shims, normalization, and per-source filtering inside the
  builder where they belong while the driver owns reading the fabric,
  running the year loop (when applicable), pruning orphans, stitching,
  and writing.

Worked example: SCA
-------------------

Bare excerpt from ``targets/sca.py``::

    ADAPTER = TargetAdapter(
        target_key="sca",
        config_key="snow_covered_area",
        cadence="daily",
        bounds_units="1",
        bounds_long_name_kind="daily fractional snow-covered area",
        cell_methods="time: point",
        title=(
            "NHM SCA calibration target (CI-bounded fractional 0-1; "
            "lower/upper from MOD10C1 v061)"
        ),
        nn_title="NHM SCA calibration target (NN-filled, fractional 0-1)",
        references=(
            "Hay et al. 2022, doi:10.3133/tm6B10; "
            "PRMSobjfun.f90 calcSCA (docs/references/PRMSobjfun.f90)"
        ),
        year_chunked=True,
        intermediates_subdir=".sca_intermediates",
        intermediate_base="sca_targets",
        forced_zero_months=(7, 8),
        source_loader=_load_sca_year,
    )

    def build(project):
        from nhf_spatial_targets.targets._driver import build as run_driver
        run_driver(ADAPTER, project)

The driver inspects ``year_chunked`` and runs the per-year build loop
with the cache + orphan-prune + stitch contract. ``_load_sca_year`` is
the SCA-specific code that knows about the CI gate, the July/August
rule (which the driver applies via ``forced_zero_months`` after the
loader returns), and the binary single-source ``n_sources`` semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import xarray as xr

# Allowed cadence values. ``daily`` and the year-chunked targets cohabit:
# SCA and SWE are daily AND year-chunked, while ``monthly`` and ``annual``
# are always single-shot in current targets.
Cadence = Literal["monthly", "annual", "daily"]


@dataclass(frozen=True)
class TargetAdapter:
    """Declarative description of a calibration-target build for the driver.

    All fields are required unless documented as optional. The adapter
    is frozen so it can be declared at module scope and cached safely
    across invocations.

    Parameters
    ----------
    target_key
        Short label for log messages / CLI dispatch (``"sca"``,
        ``"swe"``, ``"runoff"``, ``"aet"``, ``"recharge"``,
        ``"soil_moisture"``).
    config_key
        Key into ``project.config["targets"]`` (``"snow_covered_area"``,
        ``"snow_water_equivalent"``, etc.). Distinct from ``target_key``
        because the CLI uses the short form but the on-disk config uses
        the TM 6-B10 name.
    cadence
        ``"monthly"`` / ``"annual"`` / ``"daily"``. Drives the
        master-index ``freq``, the ``time_bnds`` offset, and (when
        relevant) the per-year build loop's day-level resolution.
    bounds_units
        Units string for ``lower_bound`` / ``upper_bound`` variable
        attrs (e.g. ``"cfs"``, ``"inches/day"``, ``"inches"``,
        ``"1"``).
    bounds_long_name_kind
        Substituted into the writer's ``long_name`` template:
        ``"lower bound of {kind} (NaN-aware min across sources)"``
        Examples: ``"monthly runoff"``, ``"annual recharge"``.
    cell_methods
        CF ``cell_methods`` attr value (``"time: sum"``,
        ``"time: mean"``, ``"time: point"``).
    title
        Global ``title`` attr for the unfilled target NC.
    nn_title
        Global ``title`` attr for the ``_nn_filled.nc`` companion. Only
        used when the target builds NN-fill (i.e. when the project
        config's ``nn_fill`` is True for this target).
    source_loader
        Callable invoked by the driver to produce per-source
        contributions. Signature::

            (project, adapter, year_context=None) -> SourceLoaderResult

        ``year_context`` is ``None`` for single-shot builds and a
        ``(year, year_start_iso, year_end_iso)`` tuple for the
        year-chunked path. Returns the SourceLoaderResult namedtuple
        that bundles the combined bounds, the master time index, the
        ``time_bnds`` offset unit, and any per-target ``extra_attrs``.
    references
        ``references`` global attr. Defaults to the TM 6-B10 citation;
        SCA / SWE extend it with their secondary references.
    needs_hru_area
        ``True`` only for runoff (the mm/month → cfs conversion needs
        per-HRU area). Drives the choice between
        :func:`compute_hru_area_and_centroids` and
        :func:`compute_hru_centroids` in the driver setup.
    year_chunked
        ``True`` for SCA and SWE. Switches the driver from single-shot
        build to the per-year loop + fingerprint cache + orphan-prune +
        stitch pipeline. Requires ``intermediates_subdir`` and
        ``intermediate_base`` when set.
    intermediates_subdir
        Directory name under ``<project>/targets/`` for per-year
        intermediates (e.g. ``".sca_intermediates"``). Required when
        ``year_chunked=True``.
    intermediate_base
        Filename prefix for per-year intermediates without the trailing
        ``_<YYYY>`` (e.g. ``"sca_targets"`` →
        ``sca_targets_2005.nc``). Required when ``year_chunked=True``.
    forced_zero_months
        Tuple of calendar month integers (1-12) whose CI-passing cells
        are forced to ``(lower, upper) = (0, 0)`` after the loader
        runs. Only SCA uses this (``(7, 8)``); the empty default means
        no forced zero. Encoded on the adapter rather than special-
        cased in the driver so future targets with a seasonal floor can
        opt in declaratively.
    forced_zero_validity_var
        Name of an extra DataArray returned in the source-loader's
        ``extras`` dict that selects which cells to force to zero.
        ``"valid"`` for SCA — only force July/August zero where the CI
        gate passed, leaving sub-threshold cells NaN. Default ``None``
        means apply forced_zero to every month-7/8 cell unconditionally.
    on_year_skip
        Optional callable invoked when the year-chunked driver decides
        to skip a year because its cached intermediate is healthy.
        Signature ``(project, adapter, year, cached_attrs) -> None``.
        SCA uses this to re-emit its per-year low-valid-coverage WARNING
        from the cached ``frac_valid_bound`` attr, so an operator
        re-running against a healthy cache continues to see the alert
        that fired on the original build (#213). Default ``None`` is a
        no-op skip.
    per_year_title_template
        ``str.format``-style template used to title each per-year
        intermediate NC. Must contain ``{year}``. Defaults to a short
        ``"<short label> calibration target year {year} (intermediate)"``
        synthesized from ``target_key``; year-chunked adapters
        (SCA / SWE) override this so the intermediate's title matches
        the original builders' format.
    per_year_nn_title_template
        Companion template for the NN-filled intermediate. Same
        constraints as ``per_year_title_template``.
    """

    target_key: str
    config_key: str
    cadence: Cadence
    bounds_units: str
    bounds_long_name_kind: str
    cell_methods: str
    title: str
    nn_title: str
    source_loader: Callable[..., "SourceLoaderResult"] = field(repr=False)
    references: str = "Hay et al. 2022, doi:10.3133/tm6B10"
    needs_hru_area: bool = False
    year_chunked: bool = False
    intermediates_subdir: str | None = None
    intermediate_base: str | None = None
    forced_zero_months: tuple[int, ...] = ()
    forced_zero_validity_var: str | None = None
    on_year_skip: Callable[..., None] | None = field(default=None, repr=False)
    per_year_title_template: str | None = None
    per_year_nn_title_template: str | None = None

    def __post_init__(self) -> None:
        _ALLOWED_CADENCES = {"monthly", "annual", "daily"}
        if self.cadence not in _ALLOWED_CADENCES:
            raise ValueError(
                f"TargetAdapter.cadence={self.cadence!r} is not recognized; "
                f"expected one of {sorted(_ALLOWED_CADENCES)}."
            )
        if self.year_chunked:
            if self.intermediates_subdir is None:
                raise ValueError(
                    f"TargetAdapter({self.target_key!r}): year_chunked=True "
                    "requires intermediates_subdir to be set."
                )
            if self.intermediate_base is None:
                raise ValueError(
                    f"TargetAdapter({self.target_key!r}): year_chunked=True "
                    "requires intermediate_base to be set."
                )
        for month in self.forced_zero_months:
            if not 1 <= int(month) <= 12:
                raise ValueError(
                    f"TargetAdapter({self.target_key!r}): "
                    f"forced_zero_months entry {month!r} is not 1-12."
                )
        for tmpl_field in ("per_year_title_template", "per_year_nn_title_template"):
            tmpl = getattr(self, tmpl_field)
            if tmpl is not None and "{year}" not in tmpl:
                raise ValueError(
                    f"TargetAdapter({self.target_key!r}): {tmpl_field}={tmpl!r} "
                    "must contain '{year}' as a format placeholder."
                )


@dataclass(frozen=True)
class SourceLoaderResult:
    """Return shape from a target's source_loader callable.

    The driver hands this to :func:`_writers.write_bounds_target`. Each
    field is the obvious one a multi-source build produces — the
    factored shape exists so the loader can do whatever per-source
    arithmetic it needs (unit shims, normalization, NaN-aware min/max,
    fabric-scope filtering) without the driver needing to know about
    those details.

    Attributes
    ----------
    lower, upper
        The combined-source bounds. Aligned on ``(time, id_col)`` with
        ``time`` equal to ``time_index``.
    n_sources
        Per-cell finite-source-count diagnostic, int8 dtype.
    n_sources_count
        Total contributor count (drives the ``n_sources`` flag_values
        length).
    time_index
        Master ``DatetimeIndex`` ``lower`` / ``upper`` are aligned to.
    time_offset_unit
        Offset added to each ``time_index`` entry to form
        ``time_bnds``'s upper edge (e.g. ``pd.offsets.MonthBegin(1)``).
    extra_attrs
        Per-target global attrs (``source``, ``period``,
        ``normalize_period``, etc.). Merged with the driver's
        adapter-level defaults (``references``, ``area_crs``,
        ``fabric_sha256``, etc.) before writing.
    extras
        Optional extra DataArrays the driver may need for post-loader
        transforms — currently only used by SCA to pass the per-cell
        ``valid`` mask under ``forced_zero_validity_var=="valid"``.
        Empty dict by default.
    """

    lower: "xr.DataArray"
    upper: "xr.DataArray"
    n_sources: "xr.DataArray"
    n_sources_count: int
    time_index: "object"  # pd.DatetimeIndex; quoted to keep this file pandas-free
    time_offset_unit: "object"  # pd.offsets.*
    extra_attrs: dict
    extras: dict = field(default_factory=dict)
