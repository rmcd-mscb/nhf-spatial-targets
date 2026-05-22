"""Backfill existing aggregated/target NCs to the io_nc chunked layout (#165 ST5).

The #165 writers only chunk+compress NCs they newly write; projects built
before ST2/ST3 still hold contiguous, uncompressed NetCDFs. ``rechunk_project``
rewrites those in place to the canonical chunked+zlib layout produced by
:func:`io_nc.build_encoding`, reclaiming substantial disk (largest on
sparse/NaN-heavy daily sources, where contiguous storage dominated) without
re-aggregating.

Guarantees:

- **Idempotent** — a file is skipped only when *every* field variable is
  already chunked+compressed (detected via ``netCDF4.Variable.chunking()`` /
  ``.filters()``); a mixed-state file is rewritten.
- **Atomic** — each file is rewritten to ``<file>.rechunk.tmp`` then renamed,
  so an interrupted run never leaves a half-written NC at the canonical path.
- **Value-preserving** — every variable (data vars *and* coordinates,
  including the re-encoded ``time`` / ``time_bnds``) is compared decoded and
  NaN-aware against the original before the rename; a mismatch aborts that file
  with no replacement. Native dtypes are preserved, so the change is
  storage-layout only (the on-disk bytes differ — chunking, zlib, and an added
  ``_FillValue`` — but the decoded values do not).
- **Scoped** — operates only on a project's ``data/aggregated/`` and
  ``targets/`` trees. It never touches the shared datastore's *consolidated*
  NCs (those use issue #158's per-source tile sizes, a different policy), and
  it skips the daymet/ssebop aggregated outputs left as-is by ST3.
- **Per-file isolated** — one file's failure (bit-identity mismatch, disk
  full, a stray NC missing the HRU dim) is recorded and the run continues; the
  CLI surfaces failures and exits non-zero.
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

#: Aggregated sources intentionally left unchunked by ST3 (already chunked,
#: remote-sourced) — rechunk skips them for the same reason.
_SKIP_SOURCES = frozenset({"daymet", "ssebop"})


def is_rechunked(path: Path) -> bool:
    """True only if *every* field (>=2D) variable is chunked + zlib-compressed.

    Checking all field variables — not just the first — is essential: a file
    where one variable is chunked but another is still contiguous (e.g. an
    interrupted earlier run) must be rewritten, not silently skipped. A
    mixed-chunk-state file is logged at WARNING and reported as not-yet-rechunked.
    """
    with netCDF4.Dataset(path) as nc:
        field_vars = [v for v in nc.variables.values() if v.ndim >= 2]
        if not field_vars:
            return False
        states = [
            v.chunking() != "contiguous" and bool(v.filters().get("zlib"))
            for v in field_vars
        ]
        if any(states) and not all(states):
            logger.warning(
                "%s has mixed chunk state across field variables; will rechunk.",
                path,
            )
        return all(states)


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

    ``status`` is one of ``"skipped"`` (already chunked), ``"skipped-no-hru"``
    (no HRU dim — not a fabric-aligned NC), ``"would-rechunk"`` (dry run), or
    ``"rechunked"``. Raises if the write fails or the value-preservation check
    detects a mismatch (the original is left untouched in that case).
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
        # A stray, non-fabric NC (no HRU dim) can't be chunked by the
        # aggregated/target formula — skip it rather than crash build_encoding.
        if id_col not in ds_lazy.dims:
            return {
                "path": path,
                "status": "skipped-no-hru",
                "size_before": size_before,
                "size_after": size_before,
            }
        ds = ds_lazy.load()
    encoding = build_encoding(
        ds, layer=layer, hru_dim=id_col, timesteps_per_file=ds.sizes.get("time")
    )
    tmp = path.with_suffix(path.suffix + ".rechunk.tmp")
    try:
        ds.to_netcdf(tmp, format="NETCDF4", encoding=encoding)
        with xr.open_dataset(tmp) as got:
            # Compare every variable — data vars AND coordinates. time/time_bnds
            # are re-encoded (float64 epoch), so verifying coords is not optional.
            for name in ds.variables:
                if not _arrays_equal(
                    np.asarray(ds[name].values), np.asarray(got[name].values)
                ):
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
    results: list[dict[str, Any]] = []
    for path, file_layer in _iter_layer_files(project, layer, source):
        try:
            results.append(rechunk_file(path, file_layer, id_col, dry_run=dry_run))
        except Exception as exc:
            # Per-file isolation: one bad file (mismatch, disk full, corrupt
            # source) is recorded and the run continues. The original is left
            # untouched by rechunk_file's atomic guard; the CLI surfaces the
            # failure and exits non-zero so it can't pass silently.
            logger.error("rechunk failed for %s: %s", path, exc)
            results.append(
                {
                    "path": path,
                    "status": "failed",
                    "error": str(exc),
                    "size_before": path.stat().st_size if path.exists() else 0,
                    "size_after": None,
                }
            )
    return results
