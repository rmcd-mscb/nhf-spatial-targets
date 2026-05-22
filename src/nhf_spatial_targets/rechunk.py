"""Backfill existing aggregated/target NCs to the io_nc chunked layout (#165 ST5).

The #165 writers only chunk+compress NCs they newly write; projects built
before ST2/ST3 still hold contiguous, uncompressed NetCDFs. ``rechunk_project``
rewrites those in place to the canonical chunked+zlib layout produced by
:func:`io_nc.build_encoding`, reclaiming disk (≈85% on sparse daily sources)
without re-aggregating.

Guarantees:

- **Idempotent** — a file whose data variables are already chunked+compressed
  is skipped (detected via ``netCDF4.Variable.chunking()`` / ``.filters()``).
- **Atomic** — each file is rewritten to ``<file>.rechunk.tmp`` then renamed,
  so an interrupted run never leaves a half-written NC at the canonical path.
- **Bit-identical** — every data variable is compared (NaN-aware for floats)
  against the original before the rename; a mismatch aborts that file. Native
  dtypes are preserved, so rechunking changes storage layout only.
- **Scoped** — operates only on a project's ``data/aggregated/`` and
  ``targets/`` trees. It never touches the shared datastore's *consolidated*
  NCs (those use issue #158's per-source tile sizes, a different policy), and
  it skips the daymet/ssebop aggregated outputs left as-is by ST3a.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import xarray as xr

from nhf_spatial_targets.io_nc import build_encoding
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)

_VALID_LAYERS = ("aggregated", "target")

#: Aggregated sources intentionally left unchunked by ST3a (already chunked,
#: remote-sourced) — rechunk skips them for the same reason.
_SKIP_SOURCES = frozenset({"daymet", "ssebop"})


def _first_field_var(nc: netCDF4.Dataset) -> netCDF4.Variable | None:
    """The first ``>=2D`` data variable — representative for chunk detection."""
    for v in nc.variables.values():
        if v.ndim >= 2:
            return v
    return None


def is_rechunked(path: Path) -> bool:
    """True if *path*'s field variables are already chunked + zlib-compressed."""
    with netCDF4.Dataset(path) as nc:
        v = _first_field_var(nc)
        if v is None:
            return False
        return v.chunking() != "contiguous" and bool(v.filters().get("zlib"))


def _arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact equality, NaN-aware for floating dtypes."""
    if np.issubdtype(a.dtype, np.floating):
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def _iter_layer_files(
    project: Project, layer: str | None, source: str | None
) -> list[tuple[Path, str]]:
    """Per-file work list of ``(path, layer)`` for the requested scope."""
    files: list[tuple[Path, str]] = []
    if layer in (None, "aggregated"):
        agg = project.aggregated_dir()
        if agg.is_dir():
            for src_dir in sorted(p for p in agg.iterdir() if p.is_dir()):
                key = src_dir.name
                if source is not None and key != source:
                    continue
                if key in _SKIP_SOURCES:
                    continue
                files.extend(
                    (f, "aggregated") for f in sorted(src_dir.glob(f"{key}_*_agg.nc"))
                )
    if layer in (None, "target") and source is None:
        tdir = project.targets_dir()
        if tdir.is_dir():
            files.extend((f, "target") for f in sorted(tdir.glob("*.nc")))
    return files


def rechunk_file(
    path: Path, layer: str, id_col: str, *, dry_run: bool = False
) -> dict[str, Any]:
    """Rewrite one NC to the io_nc chunked layout; return a result record.

    ``status`` is one of ``"skipped"`` (already chunked), ``"would-rechunk"``
    (dry run), or ``"rechunked"``. Raises if the bit-identity check fails.
    """
    size_before = path.stat().st_size
    if is_rechunked(path):
        return {
            "path": path,
            "status": "skipped",
            "size_before": size_before,
            "size_after": size_before,
        }
    if dry_run:
        return {
            "path": path,
            "status": "would-rechunk",
            "size_before": size_before,
            "size_after": None,
        }

    with xr.open_dataset(path) as ds_lazy:
        ds = ds_lazy.load()
    encoding = build_encoding(
        ds, layer=layer, hru_dim=id_col, timesteps_per_file=ds.sizes.get("time")
    )
    tmp = path.with_suffix(path.suffix + ".rechunk.tmp")
    try:
        ds.to_netcdf(tmp, format="NETCDF4", encoding=encoding)
        with xr.open_dataset(tmp) as got:
            for name, da in ds.data_vars.items():
                if not _arrays_equal(da.values, got[name].values):
                    raise RuntimeError(
                        f"rechunk altered values for '{name}' in {path}; "
                        f"aborting (no file replaced)."
                    )
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    size_after = path.stat().st_size
    logger.info(
        "rechunked %s: %.1f MB -> %.1f MB (%.0f%% smaller)",
        path.name,
        size_before / 1e6,
        size_after / 1e6,
        100.0 * (1 - size_after / size_before) if size_before else 0.0,
    )
    return {
        "path": path,
        "status": "rechunked",
        "size_before": size_before,
        "size_after": size_after,
    }


def rechunk_project(
    project: Project,
    *,
    layer: str | None = None,
    source: str | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Rechunk every in-scope NC for *project*; return per-file result records.

    Parameters
    ----------
    project:
        Loaded project.
    layer:
        ``"aggregated"``, ``"target"``, or ``None`` for both.
    source:
        Restrict the aggregated layer to a single source key. Ignored for the
        target layer (targets are not per-source).
    dry_run:
        Report planned work without writing.
    """
    if layer is not None and layer not in _VALID_LAYERS:
        raise ValueError(
            f"invalid layer {layer!r}; expected one of {_VALID_LAYERS} or None"
        )
    id_col = project.id_col
    results = [
        rechunk_file(path, file_layer, id_col, dry_run=dry_run)
        for path, file_layer in _iter_layer_files(project, layer, source)
    ]
    return results
