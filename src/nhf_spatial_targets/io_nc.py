"""Shared NetCDF encoding + atomic-write policy (issue #165).

Three pipeline stages write NetCDFs — consolidate, aggregate, targets — and
historically each picked its own (or no) chunking/compression. This module is
the single place that policy lives, so a future codec change is a one-file edit
rather than three.

Two public entry points:

- :func:`build_encoding` returns an xarray ``encoding`` dict for a Dataset,
  parameterized by *layer*. For the ``aggregated`` and ``target`` layers it
  chunks ``(time, hru)`` data variables so a single HRU's full time series
  lands in one ~1 MiB HDF5 chunk (the dominant calibration read pattern), and
  applies zlib compression. The ``consolidated`` layer is a seam owned by
  issue #158 (per-source spatial tiling) and raises until that lands.
- :func:`atomic_to_netcdf` writes via a sibling tempfile then renames, so a
  partial NetCDF never appears at the destination path.

The HDF5 partial-chunk constraint motivates the chunk shape: HDF5 cannot read
part of a chunk, so a per-HRU time-series read must pull every chunk the column
touches. Chunking the HRU dimension narrowly (and the time dimension fully)
makes that read one chunk instead of the whole file. See
``docs/architecture/transformation-pipeline.md`` and issue #165 for the full
rationale.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

#: HRU/time layers that share the per-HRU-time-series chunking formula.
_FABRIC_LAYERS = ("aggregated", "target")

#: Default per-chunk byte budget (1 MiB) — netCDF4's default HDF5 chunk-cache
#: size, and a good balance between read granularity and per-chunk overhead.
DEFAULT_TARGET_CHUNK_BYTES = 1_048_576

#: Pinned CF time encoding applied to ``time`` / ``time_bnds`` on every layer.
_TIME_ENCODING = {
    "dtype": "float64",
    "units": "days since 1970-01-01 00:00:00",
    "calendar": "proleptic_gregorian",
}


def _fill_value_for(dtype: np.dtype) -> Any:
    """Layer-agnostic ``_FillValue`` policy keyed on the encoded dtype.

    Floats use NaN and non-``int16`` integers (notably the ``int8``
    diagnostics) carry no fill value — matching the pre-#165
    ``targets/_common.py`` writer, which only ever wrote float32 and int8
    encodings. Packed ``int16`` uses the project's ``-9999`` sentinel (the
    SNODAS/aggregate-layer convention); the targets writer had no int16 path.
    """
    if np.issubdtype(dtype, np.floating):
        return dtype.type(np.nan)
    if dtype == np.int16:
        return np.int16(-9999)
    return None


def _chunk_hru(
    timesteps_per_file: int,
    n_hru: int,
    itemsize: int,
    target_chunk_bytes: int,
) -> int:
    """HRU-dim chunk length giving ~``target_chunk_bytes`` per chunk, capped.

    ``chunk_hru = ceil(target_chunk_bytes / (timesteps_per_file * itemsize))``,
    never larger than the fabric's HRU count.
    """
    denom = max(timesteps_per_file, 1) * itemsize
    return min(math.ceil(target_chunk_bytes / denom), n_hru)


def build_encoding(
    ds: xr.Dataset,
    layer: str,
    *,
    hru_dim: str | None = None,
    spatial_dims: tuple[str, str] | None = None,
    timesteps_per_file: int | None = None,
    var_dtype: dict[str, str] | None = None,
    compression_level: int = 4,
    target_chunk_bytes: int = DEFAULT_TARGET_CHUNK_BYTES,
) -> dict[str, dict[str, Any]]:
    """Build an xarray ``encoding`` dict for *ds* per the given *layer*.

    Parameters
    ----------
    ds:
        Dataset about to be written. Inspected for data-variable dtypes,
        dimension order, and sizes; not mutated.
    layer:
        ``"aggregated"`` or ``"target"`` apply the per-HRU-time-series chunk
        formula. ``"consolidated"`` is the seam for issue #158 and raises
        ``NotImplementedError`` until that work lands.
    hru_dim:
        Name of the HRU/``id_col`` dimension (required for fabric layers).
    spatial_dims:
        ``(y, x)`` dims for the consolidated layer (accepted for forward
        compatibility with #158; unused here).
    timesteps_per_file:
        Length of the time chunk. Defaults to the full ``time`` axis of *ds*.
    var_dtype:
        Optional per-variable on-disk dtype override, e.g.
        ``{"ro": "float32"}``. Variables absent from the map keep their
        in-memory dtype.
    compression_level:
        zlib ``complevel`` (default 4).
    target_chunk_bytes:
        Per-chunk byte budget driving the HRU chunk length.

    Returns
    -------
    dict
        ``{var_name: {encoding...}}`` suitable for ``ds.to_netcdf(encoding=)``.
        Includes pinned time encoding for ``time`` / ``time_bnds`` when present.
    """
    if layer == "consolidated":
        raise NotImplementedError(
            "build_encoding(layer='consolidated') is a seam owned by issue "
            "#158 (per-source spatial tiling); not implemented here."
        )
    if layer not in _FABRIC_LAYERS:
        raise ValueError(
            f"unknown layer {layer!r}; expected one of "
            f"{_FABRIC_LAYERS + ('consolidated',)}"
        )
    if hru_dim is None:
        raise ValueError(f"hru_dim is required for layer={layer!r}")
    # Fail loudly on a drifted/mistyped dim name. Without this guard the
    # data-var loop below silently skips every variable and returns a dict
    # with only time encoding — producing an unchunked, uncompressed NetCDF
    # with no error, defeating this module's entire purpose.
    if hru_dim not in ds.sizes:
        raise ValueError(
            f"hru_dim={hru_dim!r} not found in dataset dims {tuple(ds.sizes)} "
            f"for layer={layer!r}; cannot build per-HRU chunking. Check that "
            f"the project id_col matches the dataset."
        )

    var_dtype = var_dtype or {}
    n_hru = int(ds.sizes[hru_dim])
    time_size = int(ds.sizes["time"]) if "time" in ds.sizes else None
    if timesteps_per_file is None and time_size is not None:
        timesteps_per_file = time_size
    if time_size is not None:
        time_chunk = min(timesteps_per_file, time_size)
    else:
        time_chunk = timesteps_per_file or 1

    encoding: dict[str, dict[str, Any]] = {}

    for name, da in ds.data_vars.items():
        if hru_dim not in da.dims:
            continue
        # A CF grid-mapping container (e.g. ``crs``) may carry the hru_dim in
        # some aggregated NCs, but it is metadata, not a data field — adding
        # chunksizes/zlib/_FillValue would corrupt it. Leave it untouched.
        if "grid_mapping_name" in da.attrs:
            continue
        dtype = np.dtype(var_dtype.get(name, da.dtype))
        # Size the HRU chunk against this variable's own time extent: a static
        # per-HRU var (no time dim) chunks on HRU alone rather than carrying a
        # phantom time factor that would under-fill the byte budget.
        var_time_chunk = time_chunk if "time" in da.dims else 1
        chunk_hru = _chunk_hru(
            var_time_chunk, n_hru, dtype.itemsize, target_chunk_bytes
        )
        chunksizes = tuple(
            time_chunk
            if dim == "time"
            else chunk_hru
            if dim == hru_dim
            else ds.sizes[dim]
            for dim in da.dims
        )
        var_enc: dict[str, Any] = {
            "dtype": str(dtype),
            "zlib": True,
            "complevel": compression_level,
            "chunksizes": chunksizes,
            "_FillValue": _fill_value_for(dtype),
            # Set shuffle explicitly for both kinds: netCDF4-python defaults
            # shuffle=True whenever zlib is on, so omitting the key for floats
            # would silently enable it on disk, contradicting the policy.
            "shuffle": bool(np.issubdtype(dtype, np.integer)),
        }
        encoding[name] = var_enc

    for tvar in ("time", "time_bnds"):
        if tvar in ds.variables:
            encoding[tvar] = dict(_TIME_ENCODING)

    return encoding


def atomic_to_netcdf(
    ds: xr.Dataset,
    path: Path | str,
    *,
    encoding: dict[str, dict[str, Any]] | None = None,
    format: str = "NETCDF4",
) -> None:
    """Write *ds* to *path* atomically: tempfile in the same dir, then rename.

    A sibling tempfile keeps the rename on the same filesystem (atomic). On any
    exception the tempfile is removed so no partial NetCDF is left behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".nc.tmp")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        ds.to_netcdf(tmp_path, format=format, encoding=encoding)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
