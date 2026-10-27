"""Find and back-fill netCDF default-fill sentinels stored as data (issue #361).

gdptools' ``AggGen`` writes ``NC_FILL_DOUBLE`` (``9.969209968386869e+36``)
for HRUs whose geometry has no overlap with the source grid, without a
matching ``_FillValue`` attribute. Issue #204 / PR #205 masks that sentinel
at the aggregation **write** boundary and at the target-builder **read**
boundary (:func:`io_nc.mask_netcdf_default_fills`), but every aggregated NC
written *before* #205 still carries the sentinel on disk, decoding as
~1e36 of real data for anyone who reads it directly (inspect notebooks,
collaborators, the release payload, ``maintenance rechunk``).

This module is the on-disk backfill, split along the ``maintenance`` verb
grammar:

- :func:`scan_file` / :func:`scan_project` **report** sentinel counts per
  float data var (``maintenance check-fills``; also the publish pre-flight).
- :func:`mask_file` / :func:`mask_project` **rewrite** affected NCs in place
  (``maintenance mask-fills``).

Guarantees of the rewrite, mirroring :mod:`rechunk`:

- **Idempotent** — a file with no sentinel cells is reported ``clean`` and
  never rewritten.
- **Atomic** — written to ``<file>.maskfills.tmp`` then renamed; an
  interrupted run never leaves a half-written NC at the canonical path.
- **Value-preserving except the sentinel** — every variable (data vars and
  coordinates) is compared NaN-aware against the masked in-memory dataset
  before the rename; a mismatch aborts that file with no replacement. A
  rewritten file lands in the canonical :func:`io_nc.build_encoding` layout
  (chunked + zlib, ``_FillValue=NaN``), so the backfill also rechunks.
- **Scoped** — a project's ``data/aggregated/`` and ``targets/`` trees only.
  Unlike ``rechunk`` it does **not** skip daymet/ssebop: a sentinel is wrong
  wherever it is (those sources declare the sentinel as ``_FillValue`` and so
  decode clean — they are scanned, never rewritten).
- **Per-file isolated** — one file's failure is recorded and the run
  continues; the CLI surfaces failures and exits non-zero.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from netCDF4 import default_fillvals

from nhf_spatial_targets.io_nc import build_encoding, mask_netcdf_default_fills
from nhf_spatial_targets.rechunk import _arrays_equal, _iter_layer_files
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)

_VALID_LAYERS = ("aggregated", "target")


def netcdf_default_fill(dtype: Any) -> float | None:
    """Return netCDF's default fill for a floating *dtype*, else ``None``.

    ``default_fillvals`` is keyed by endian-less dtype strings (``'f4'``,
    ``'f8'``); ``np.dtype.str`` carries a byte-order prefix that is stripped.
    Only floats are handled — the fabric layers emit no integer data vars from
    gdptools (same scope as :func:`io_nc.mask_netcdf_default_fills`).
    """
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        return None
    return default_fillvals.get(dtype.str.lstrip("<>=|"))


def scan_file(path: Path) -> dict[str, int]:
    """Count sentinel cells per float data var in one NC; ``{}`` when clean.

    Opens lazily on the on-disk chunks and streams the comparison, so a large
    target NC is never loaded whole. Equality is exact: the sentinel is a
    bit-pattern constant. A file whose declared ``_FillValue`` *is* the
    sentinel decodes those cells to NaN before the comparison and is
    therefore clean — nothing on disk needs fixing there.
    """
    counts: dict[str, int] = {}
    with xr.open_dataset(path, chunks={}) as ds:
        for name, da in ds.data_vars.items():
            if "grid_mapping_name" in da.attrs:
                continue
            sentinel = netcdf_default_fill(da.dtype)
            if sentinel is None:
                continue
            n = int((da == sentinel).sum().compute())
            if n:
                counts[name] = n
    return counts


def scan_project(
    project: Project,
    *,
    layer: str | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Scan every in-scope NC; return one ``{path, layer, counts, error}``
    record per file (``counts`` is ``{}`` for clean or unreadable files).

    An unreadable file is recorded with its ``error`` rather than aborting the
    scan, so the report names it next to the sentinel counts of its siblings.
    """
    if layer is not None and layer not in _VALID_LAYERS:
        raise ValueError(
            f"invalid layer {layer!r}; expected one of {_VALID_LAYERS} or None"
        )
    results: list[dict[str, Any]] = []
    for path, file_layer in _iter_layer_files(
        project, layer, source, skip_sources=frozenset()
    ):
        rec: dict[str, Any] = {"path": path, "layer": file_layer, "counts": {}}
        try:
            rec["counts"] = scan_file(path)
        except Exception as exc:
            logger.error("check-fills could not read %s: %s", path, exc)
            rec["error"] = str(exc)
        results.append(rec)
    return results


def mask_file(
    path: Path, layer: str, id_col: str, *, dry_run: bool = False
) -> dict[str, Any]:
    """Rewrite one NC with its sentinel cells set to NaN; return a result record.

    ``status`` is one of ``"clean"`` (no sentinel, untouched),
    ``"skipped-no-hru"`` (no HRU dim — not a fabric-aligned NC),
    ``"would-mask"`` (dry run) or ``"masked"``. ``counts`` carries the
    per-variable sentinel counts found before the rewrite. Raises if the write
    fails or the value-preservation check detects a mismatch (the original is
    left untouched in that case).
    """
    rec: dict[str, Any] = {"path": path, "layer": layer, "counts": {}}
    # A stray, non-fabric NC (no HRU dim) can't be encoded by the
    # aggregated/target formula — skip it rather than crash build_encoding.
    with xr.open_dataset(path) as ds_lazy:
        if id_col not in ds_lazy.dims:
            return {**rec, "status": "skipped-no-hru"}
    counts = scan_file(path)
    rec["counts"] = counts
    if not counts:
        return {**rec, "status": "clean"}
    if dry_run:
        return {**rec, "status": "would-mask"}

    with xr.open_dataset(path) as ds_lazy:
        ds = ds_lazy.load()
    masked = mask_netcdf_default_fills(ds)
    encoding = build_encoding(
        masked, layer=layer, hru_dim=id_col, timesteps_per_file=masked.sizes.get("time")
    )
    tmp = path.with_suffix(path.suffix + ".maskfills.tmp")
    try:
        masked.to_netcdf(tmp, format="NETCDF4", encoding=encoding)
        with xr.open_dataset(tmp) as got:
            # Compare against the *masked* in-memory dataset: the only
            # permitted change is sentinel -> NaN, and that has already been
            # applied to `masked`, so anything else that differs is a defect.
            for name in masked.variables:
                if not _arrays_equal(
                    np.asarray(masked[name].values), np.asarray(got[name].values)
                ):
                    raise RuntimeError(
                        f"mask-fills altered values for '{name}' in {path}; "
                        f"aborting (no file replaced)."
                    )
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    logger.info(
        "masked %s: %s sentinel cell(s) -> NaN",
        path.name,
        ", ".join(f"{v}={n}" for v, n in counts.items()),
    )
    return {**rec, "status": "masked"}


def mask_project(
    project: Project,
    *,
    layer: str | None = None,
    source: str | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Mask every in-scope NC for *project*; return per-file result records.

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
    for path, file_layer in _iter_layer_files(
        project, layer, source, skip_sources=frozenset()
    ):
        try:
            results.append(mask_file(path, file_layer, id_col, dry_run=dry_run))
        except Exception as exc:
            # Per-file isolation: one bad file is recorded and the run
            # continues. The original is left untouched by mask_file's atomic
            # guard; the CLI surfaces the failure and exits non-zero.
            logger.error("mask-fills failed for %s: %s", path, exc)
            results.append(
                {
                    "path": path,
                    "layer": file_layer,
                    "status": "failed",
                    "error": str(exc),
                    "counts": {},
                }
            )
        finally:
            # mask_file eagerly `.load()`s the whole dataset (same footprint
            # story as rechunk_project): collect between files so peak RSS
            # stays at ~one file rather than accumulating.
            gc.collect()
    return results
