"""Project-config defaults: single source of truth for every settable key.

The merge is non-modifying: ``workspace.load()`` overlays user config from
``config.yml`` onto :data:`DEFAULTS` so old project directories continue to
work after new keys land. The merged dict is what every downstream consumer
sees; the on-disk ``config.yml`` is never touched.

Lists in the user config replace the default list wholesale (no item-wise
merge), so e.g. ``targets.runoff.sources: [era5_land]`` means just that one
source — no surprise inclusion of the default three.

A leaf value of ``None`` in :data:`DEFAULTS` means "no default — required."
"""

from __future__ import annotations

import copy
from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DEFAULTS: dict = {
    "fabric": {
        "path": None,  # required
        "id_col": "nhm_id",
        "crs": "EPSG:4326",
        "buffer_deg": 0.1,
        # Equal-area CRS used for HRU area + NN-fill distances. Override for
        # AK / HI / PR (e.g. EPSG:3338 for Alaska Albers).
        "area_crs": "EPSG:5070",
    },
    "datastore": None,  # required
    "dir_mode": "2775",
    "aggregation": {"engine": "gdptools", "method": "area_weighted"},
    "output": {"dir": "outputs", "format": "netcdf", "compress": True},
    "targets": {
        "runoff": {
            "enabled": True,
            "sources": [
                "era5_land",
                "gldas_noah_v21_monthly",
                "mwbm_climgrid",
            ],
            "time_step": "monthly",
            "period": None,  # required when enabled
            "prms_variable": "basin_cfs",
            "range_method": "multi_source_minmax",
            "output_file": "runoff_targets.nc",
            "nn_fill": True,
            "nn_max_candidates": 10,
            "chunk_months": 12,
        },
        "aet": {
            "enabled": True,
            "sources": ["mod16a2_v061", "ssebop", "mwbm_climgrid"],
            "time_step": "monthly",
            "period": None,
            "prms_variable": "hru_actet",
            "range_method": "multi_source_minmax",
            "output_file": "aet_targets.nc",
            "nn_fill": True,
            "nn_max_candidates": 10,
            "chunk_months": 12,
        },
        "recharge": {
            "enabled": True,
            "sources": ["reitz2017", "watergap22d", "era5_land"],
            "time_step": "annual",
            "period": None,
            "prms_variable": "recharge",
            "range_method": "normalized_minmax",
            "normalize": True,
            "normalize_period": "2000-01-01/2009-12-31",
            "output_file": "recharge_targets.nc",
            "nn_fill": True,
            "nn_max_candidates": 10,
        },
        "soil_moisture": {
            "enabled": True,
            "sources": ["merra2", "ncep_ncar", "nldas_mosaic", "nldas_noah"],
            "time_step": ["monthly", "annual"],
            "period": None,
            "prms_variable": "soil_rechr",
            "range_method": "normalized_minmax",
            "normalize": True,
            "normalize_by": "calendar_month",
            "output_file": "soil_moisture_targets.nc",
            "nn_fill": True,
            "nn_max_candidates": 10,
        },
        "snow_covered_area": {
            "enabled": True,
            "sources": ["mod10c1_v061"],
            "time_step": "daily",
            "period": None,
            "prms_variable": "snowcov_area",
            "range_method": "modis_ci",
            "ci_threshold": 0.70,
            "output_file": "sca_targets.nc",
            "nn_fill": True,
            "nn_max_candidates": 10,
        },
    },
}


# Always-required dotted paths (independent of any toggle).
REQUIRED: list[tuple[str, ...]] = [
    ("datastore",),
    ("fabric", "path"),
]

# Per-target paths that are required iff the target is enabled.
_REQUIRED_PER_TARGET: list[tuple[str, ...]] = [
    ("period",),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_defaults(user: dict | None) -> dict:
    """Return ``user`` deep-merged onto :data:`DEFAULTS`.

    User values win at every leaf. Nested dicts recurse. Lists in ``user``
    replace the default list wholesale (no item-wise merge), so a user-set
    ``sources: [era5_land]`` yields exactly that one element.

    A ``None`` value in the user config is treated as "not set" and falls
    through to the default; pass an explicit empty string or empty dict if
    you genuinely want to suppress a default.
    """
    return _deep_merge(DEFAULTS, user or {})


def iter_default_diff(user: dict | None) -> Iterator[tuple[str, object]]:
    """Yield ``(dotted_path, default_value)`` for every key the user did not set.

    Walks the merged dict and reports leaves where ``user`` had no value.
    Used by ``validate`` to print which defaults took effect.
    """
    user = user or {}
    yield from _walk_diff(DEFAULTS, user, prefix=())


def find_unknown_keys(user: dict | None) -> list[str]:
    """Return dotted paths for user-set keys not present in :data:`DEFAULTS`.

    Catches typos like ``runoff.nn_fil``. Lists are leaves (their items are
    not introspected). Top-level keys not in DEFAULTS are reported as-is.
    """
    user = user or {}
    return list(_walk_unknown(DEFAULTS, user, prefix=()))


def missing_required(merged: dict) -> list[str]:
    """Return dotted paths for required keys missing or ``None`` in ``merged``.

    Includes always-required paths (``datastore``, ``fabric.path``) plus
    per-target ``period`` for targets where ``enabled`` is True.
    """
    missing: list[str] = []
    for path in REQUIRED:
        if _get(merged, path) in (None, ""):
            missing.append(".".join(path))
    targets = merged.get("targets", {}) or {}
    for tname, tcfg in targets.items():
        if not isinstance(tcfg, dict) or not tcfg.get("enabled", False):
            continue
        for path in _REQUIRED_PER_TARGET:
            if _get(tcfg, path) in (None, ""):
                missing.append(".".join(("targets", tname, *path)))
    return missing


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _walk_diff(
    defaults: dict, user: dict, prefix: tuple[str, ...]
) -> Iterator[tuple[str, object]]:
    for k, dv in defaults.items():
        path = (*prefix, k)
        uv = user.get(k) if isinstance(user, dict) else None
        if isinstance(dv, dict):
            if isinstance(uv, dict):
                yield from _walk_diff(dv, uv, path)
            else:
                # User did not set this whole subtree; report each leaf default.
                yield from _walk_diff(dv, {}, path)
        else:
            if uv is None:
                yield (".".join(path), dv)


def _walk_unknown(defaults: dict, user: dict, prefix: tuple[str, ...]) -> Iterator[str]:
    if not isinstance(user, dict):
        return
    for k, uv in user.items():
        path = (*prefix, k)
        if k not in defaults:
            yield ".".join(path)
            continue
        dv = defaults[k]
        if isinstance(dv, dict) and isinstance(uv, dict):
            yield from _walk_unknown(dv, uv, path)


def _get(d: dict, path: tuple[str, ...]) -> object:
    cur: object = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur
