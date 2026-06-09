"""University of Arizona daily SWE / snow-depth aggregator (CONUS).

Reads per-calendar-year consolidated NCs at
``<datastore>/ua_swe/daily/ua_swe_daily_<year>.nc`` (written by PR-A2's
``fetch.ua_swe.consolidate_calendar_year_ua_swe``), pre-projected to
EPSG:5070 at consolidate time, and emits area-weighted HRU means via the
shared aggregation driver.

Output variables:

- ``swe`` (``kg m-2`` ≡ mm) and ``snow_depth`` (mm): native pass-through.
- ``snow_covered_fraction`` (``1``): **derived** in the
  ``pre_aggregate_hook`` as the per-pixel binary
  ``snow_depth > depth_threshold_mm``, then area-weighted to the HRU. The
  binary must precede aggregation because a threshold is nonlinear and does
  not commute with the area-weighted mean (``mean(d > t) != mean(d) > t``).
  See ``docs/architecture/transformation-pipeline.md`` and the design spec
  ``docs/superpowers/specs/2026-06-09-ua-swe-wiring-design.md``.

``.where(snow_depth.notnull())`` is load-bearing: ``NaN > t`` is ``False``,
so a naive ``.astype(float)`` would turn every off-CONUS / fill pixel into a
hard ``0.0`` ("no snow") that dilutes the fraction for any HRU touching the
CONUS boundary or an internal gap. Re-NaNing those pixels and pairing with
``stat_method="masked_mean"`` excludes them (the HRU is NaN only when its
whole footprint is unobserved). This is the **opposite** NaN policy from
``mod10c1.py``'s ``valid_mask``, which deliberately lets unobserved pixels
become ``0.0`` because its derived ``valid_area_fraction`` means "fraction of
HRU with usable observations." Same mechanical shape, opposite intent —
copying mod10c1's hook verbatim yields a subtly wrong SCA fraction along
every coastline.

``depth_threshold_mm`` is conceptually an SCA-target knob
(``snow_covered_area.depth_threshold_mm`` in project config) but must be
evaluated on the pixel grid, so it is read at agg time, bound into the hooks
via closure, and stamped on ``snow_covered_fraction`` so the agg NC records
the threshold that defined it. Changing the threshold invalidates the agg
NC's ``snow_covered_fraction`` (re-run ``agg ua-swe``); ``swe`` /
``snow_depth`` are untouched. The ``depth_threshold_mm`` config key is wired
by PR-D; this module reads it defensively, defaulting to
``DEFAULT_DEPTH_THRESHOLD_MM`` when absent.

The attrs the ``pre_aggregate_hook`` stamps on the synthesized variable do
not reliably survive gdptools' aggregation, so a closure-bound
``post_aggregate_hook`` re-asserts them on the HRU-scale result — the same
pattern ``mod10c1.py`` uses to re-stamp its derived ``valid_area_fraction``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import xarray as xr
import yaml

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source

_SOURCE_KEY = "ua_swe"
_OUTPUT_NAME = "ua_swe_agg.nc"
_DEPTH_VAR = "snow_depth"
_SCF_VAR = "snow_covered_fraction"

#: Pixel snow-depth threshold (mm) for the snow-covered binary, used when the
#: project config carries no ``snow_covered_area.depth_threshold_mm`` (that
#: key is introduced by PR-D, so this module must read it defensively).
DEFAULT_DEPTH_THRESHOLD_MM = 1.0


def _scf_cf_attrs() -> dict[str, str]:
    """CF-1.6 variable attrs for ``snow_covered_fraction``, from the catalog.

    Read from ``catalog/sources.yml`` (the single source of truth) rather
    than hardcoded, so a units / long_name correction flows through on the
    next aggregation.
    """
    from nhf_spatial_targets import catalog

    for var in catalog.source(_SOURCE_KEY)["variables"]:
        if var.get("name") == _SCF_VAR:
            return {
                "long_name": var["long_name"],
                "units": var["cf_units"],
                "cell_methods": var["cell_methods"],
            }
    raise KeyError(
        f"{_SOURCE_KEY}: catalog has no {_SCF_VAR!r} variable entry; cannot "
        f"stamp CF attrs on the derived snow-covered fraction."
    )


def make_pre_aggregate_hook(
    depth_threshold_mm: float,
) -> Callable[[xr.Dataset], xr.Dataset]:
    """Return a pre-aggregate hook that derives ``snow_covered_fraction``.

    The hook adds the per-pixel binary ``snow_depth > depth_threshold_mm`` as
    a float field, re-NaNing fill pixels (``.where(snow_depth.notnull())``) so
    ``masked_mean`` excludes them. ``swe`` and ``snow_depth`` pass through
    untouched. ``depth_threshold_mm`` is closed over so the module-level
    adapter can be re-parameterized per run.
    """
    cf_attrs = _scf_cf_attrs()

    def _hook(ds: xr.Dataset) -> xr.Dataset:
        if _DEPTH_VAR not in ds.data_vars:
            raise KeyError(
                f"{_SOURCE_KEY}: pre_aggregate_hook expected raw variable "
                f"{_DEPTH_VAR!r}, found {list(ds.data_vars)}."
            )
        depth = ds[_DEPTH_VAR]
        scf = (depth > depth_threshold_mm).astype("float64").where(depth.notnull())
        scf.attrs = {**cf_attrs, "depth_threshold_mm": float(depth_threshold_mm)}
        return ds.assign({_SCF_VAR: scf})

    return _hook


def make_post_aggregate_hook(
    depth_threshold_mm: float,
) -> Callable[[xr.Dataset], xr.Dataset]:
    """Return a post-aggregate hook that re-stamps ``snow_covered_fraction`` attrs.

    gdptools does not reliably carry a synthesized variable's attrs through
    aggregation (the same reason ``mod10c1.py`` re-stamps its derived
    ``valid_area_fraction`` post-aggregation), so re-assert the CF attrs and
    the ``depth_threshold_mm`` provenance stamp on the HRU-scale result.
    """
    cf_attrs = _scf_cf_attrs()

    def _hook(year_ds: xr.Dataset) -> xr.Dataset:
        if _SCF_VAR in year_ds.data_vars:
            year_ds[_SCF_VAR].attrs = {
                **cf_attrs,
                "depth_threshold_mm": float(depth_threshold_mm),
            }
        return year_ds

    return _hook


def build_adapter(
    depth_threshold_mm: float = DEFAULT_DEPTH_THRESHOLD_MM,
) -> SourceAdapter:
    """Build the ua_swe ``SourceAdapter`` with hooks bound to the threshold."""
    return SourceAdapter(
        source_key=_SOURCE_KEY,
        output_cadence="daily",
        output_name=_OUTPUT_NAME,
        variables=("swe", "snow_depth", "snow_covered_fraction"),
        # Pre-projected at consolidate time (PR-A2); matches the driver's
        # WEIGHT_GEN_CRS so gdptools skips reprojection during weight gen.
        source_crs="EPSG:5070",
        files_glob="daily/ua_swe_daily_*.nc",
        # grid_variable defaults to variables[0] = "swe" (a genuine raw var),
        # so raw_grid_variable resolves without an explicit override.
        pre_aggregate_hook=make_pre_aggregate_hook(depth_threshold_mm),
        post_aggregate_hook=make_post_aggregate_hook(depth_threshold_mm),
        # CONUS-masked product + a masked binary -> masked_mean (mirrors
        # snodas #151). Under the default "mean", one NaN border pixel would
        # poison every HRU it touches.
        stat_method="masked_mean",
    )


#: Module-level adapter at the default threshold, for the static contract and
#: tests. ``aggregate_ua_swe`` re-builds a per-run adapter from project config.
ADAPTER = build_adapter()


def _resolve_depth_threshold_mm(workdir: Path) -> float:
    """Read ``snow_covered_area.depth_threshold_mm`` from the project config.

    Defaults to ``DEFAULT_DEPTH_THRESHOLD_MM`` when absent — the config key is
    wired by PR-D, so PR-B must not assume it exists.
    """
    from nhf_spatial_targets.defaults import apply_defaults

    cfg = apply_defaults(yaml.safe_load((Path(workdir) / "config.yml").read_text()))
    sca_cfg = (cfg.get("targets") or {}).get("snow_covered_area") or {}
    return float(sca_cfg.get("depth_threshold_mm", DEFAULT_DEPTH_THRESHOLD_MM))


def aggregate_ua_swe(
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
    period: str | None = None,
    *,
    worker_index: int = 0,
    n_workers: int = 1,
) -> None:
    """Aggregate UA daily SWE / snow-depth / snow-covered-fraction to HRUs.

    Reads ``snow_covered_area.depth_threshold_mm`` (default
    ``DEFAULT_DEPTH_THRESHOLD_MM``) from the project config, binds it into the
    derive/stamp hooks, and runs the shared driver. ``worker_index`` /
    ``n_workers`` enable SLURM-array year sharding (default ``(0, 1)`` serial).
    """
    adapter = build_adapter(_resolve_depth_threshold_mm(workdir))
    aggregate_source(
        adapter,
        fabric_path,
        id_col,
        workdir,
        batch_size,
        period=period,
        worker_index=worker_index,
        n_workers=n_workers,
    )
