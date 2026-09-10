"""Relabel a project's fabric ``id_col`` in place across derived artifacts.

Issue #353. A fabric may carry several HRU identifiers -- a national one
(``nhm_id``) and the model's own dense index (``hru_id`` /
``model_hru_idx``). Choosing the wrong one couples the project to an
identifier someone else controls: the Oregon fabric's ``nhm_id`` was
renumbered between two releases while ``hru_id`` was untouched.

When the fabric geometry does not change, switching ``id_col`` moves
labels only -- every aggregated and target value is identical. This
module performs that relabel so the pipeline need not re-run
aggregation (~a day) to reproduce numbers it already has.

**Why this cannot be an in-place patch.** The id is a NetCDF coordinate
variable whose name matches its dimension. ``netCDF4.Dataset.
renameDimension`` on such a dimension silently destroys the coordinate's
data -- the values come back as the dtype fill value with no error
raised and the data variables still reporting correct dims. Every file
is therefore fully rewritten through
:func:`~nhf_spatial_targets.io_nc.build_encoding` +
:func:`~nhf_spatial_targets.io_nc.atomic_to_netcdf`, which also keeps
the chunking/compression policy intact.

**Why the map is value-based rather than positional.** The new ids
happen to be a dense ``1..N`` matching row order, so ``arange`` would
"work". It is not used: a positional write is correct only while every
file really is in the order it was spot-checked in, and it fails
*silently* onto the wrong rows when one is not. Mapping each id through
an explicit bijection is order-independent and aborts on an id it has
never seen.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "build_id_map",
    "relabel_dataset",
    "relabel_weight_frame",
    "relabel_project",
]


def build_id_map(
    fabric_gdf: gpd.GeoDataFrame, from_col: str, to_col: str
) -> dict[int, int]:
    """Build the ``{old_id: new_id}`` bijection from the fabric itself.

    The fabric is the source of truth: both identifiers sit on the same
    row, so pairing them row-wise cannot disagree with the geometry.
    Deriving the map here rather than accepting a hand-supplied
    crosswalk removes a whole class of stale-file error.

    Both directions must be unique. A non-unique source cannot be
    inverted; a non-unique destination would silently merge two HRUs
    into one row.
    """
    for col in (from_col, to_col):
        if col not in fabric_gdf.columns:
            raise KeyError(
                f"build_id_map: column {col!r} is not in the fabric "
                f"(have: {sorted(fabric_gdf.columns)})"
            )
    src = fabric_gdf[from_col]
    dst = fabric_gdf[to_col]
    if not src.is_unique:
        raise ValueError(f"build_id_map: {from_col!r} is not unique in the fabric")
    if not dst.is_unique:
        raise ValueError(f"build_id_map: {to_col!r} is not unique in the fabric")
    return {int(a): int(b) for a, b in zip(src, dst)}


def _mapped(values: np.ndarray, id_map: dict[int, int], what: str) -> np.ndarray:
    missing = sorted({int(v) for v in values} - id_map.keys())
    if missing:
        head = missing[:5]
        raise ValueError(
            f"{what}: {len(missing)} id(s) not in the id map (first: {head}). "
            "Refusing to relabel -- the fabric and this artifact disagree."
        )
    return np.array([id_map[int(v)] for v in values], dtype="int64")


def relabel_dataset(
    ds: xr.Dataset, from_col: str, to_col: str, id_map: dict[int, int]
) -> xr.Dataset:
    """Return *ds* with its id dimension renamed and its ids mapped.

    Returns *ds* unchanged (identity) when it already carries *to_col* --
    the relabel is idempotent, so a partially migrated project can be
    re-run without double-mapping.

    The result is sorted ascending on *to_col*, preserving the canonical
    row order invariant (issue #93) even if the map is not
    order-preserving.
    """
    if to_col in ds.dims:
        return ds
    if from_col not in ds.dims:
        raise ValueError(
            f"relabel_dataset: dataset has neither {from_col!r} nor {to_col!r} "
            f"among its dims {tuple(ds.dims)}"
        )

    new_ids = _mapped(ds[from_col].values, id_map, "relabel_dataset")
    out = ds.assign_coords({from_col: new_ids}).rename({from_col: to_col})
    # sortby moves the data with its row; a bare coordinate overwrite would not.
    if not bool(np.all(np.diff(out[to_col].values) > 0)):
        out = out.sortby(to_col)
    return out


def relabel_weight_frame(
    df: pd.DataFrame, from_col: str, to_col: str, id_map: dict[int, int]
) -> pd.DataFrame:
    """Return *df* (a gdptools weight cache) with its id column relabelled.

    Weight rows are (hru, source-cell, weight) triples; the id column
    repeats per HRU, so this is a plain value map with no ordering
    requirement. Idempotent, like :func:`relabel_dataset`.
    """
    if to_col in df.columns:
        return df
    if from_col not in df.columns:
        raise ValueError(
            f"relabel_weight_frame: frame has neither {from_col!r} nor "
            f"{to_col!r} among its columns {list(df.columns)}"
        )
    out = df.copy()
    out[from_col] = _mapped(out[from_col].to_numpy(), id_map, "relabel_weight_frame")
    return out.rename(columns={from_col: to_col})


#: Encoding keys carried over verbatim when a file is rewritten. Anything
#: outside this set (``source``, ``original_shape``, ``coordinates``, ...)
#: is xarray bookkeeping that ``to_netcdf`` either rejects or recomputes.
_PRESERVED_ENCODING_KEYS = frozenset(
    {
        "zlib",
        "complevel",
        "shuffle",
        "fletcher32",
        "contiguous",
        "chunksizes",
        "dtype",
        "_FillValue",
        "scale_factor",
        "add_offset",
        "units",
        "calendar",
        "endian",
        "least_significant_digit",
    }
)


def _preserved_encoding(ds: xr.Dataset, rename: dict[str, str]) -> dict[str, dict]:
    """Harvest the on-disk encoding of *ds* so a rewrite reproduces it.

    A relabel must change labels and nothing else. Re-deriving the
    encoding from :func:`~nhf_spatial_targets.io_nc.build_encoding`
    would silently re-encode files the project deliberately leaves
    alone -- ``rechunk._SKIP_SOURCES`` keeps the daymet and ssebop
    aggregated outputs unchunked -- and would flip ``shuffle`` on every
    float variable besides. Carrying the source file's own encoding
    forward keeps the diff to the id.
    """
    out: dict[str, dict] = {}
    for name, var in {**ds.variables}.items():
        enc = {k: v for k, v in var.encoding.items() if k in _PRESERVED_ENCODING_KEYS}
        if enc:
            out[rename.get(str(name), str(name))] = enc
    return out


def _relabel_nc(path: Path, from_col: str, to_col: str, id_map: dict[int, int]) -> str:
    from nhf_spatial_targets.io_nc import atomic_to_netcdf

    with xr.open_dataset(path) as ds:
        if to_col in ds.dims:
            return "skipped-already-relabelled"
        encoding = _preserved_encoding(ds, {from_col: to_col})
        out = relabel_dataset(ds, from_col, to_col, id_map).load()

    atomic_to_netcdf(out, path, encoding=encoding)
    return "relabelled"


def _relabel_csv(path: Path, from_col: str, to_col: str, id_map: dict[int, int]) -> str:
    df = pd.read_csv(path)
    if to_col in df.columns:
        return "skipped-already-relabelled"
    out = relabel_weight_frame(df, from_col, to_col, id_map)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(path)
    return "relabelled"


def relabel_project(
    workdir: Path,
    from_col: str,
    to_col: str,
    id_map: dict[int, int],
    *,
    dry_run: bool = False,
) -> list[dict]:
    """Relabel every aggregated NC and weight CSV under *workdir*.

    Target NCs are deliberately **not** touched: they are the published
    deliverable and are cheap to rebuild (~20 min via ``nhf-targets
    run``), so they should be produced natively under the new id rather
    than relabelled. Only the expensive intermediates are migrated.

    Each file is independent -- a failure is recorded and the walk
    continues, so one bad file cannot abort a long migration midway and
    leave the operator guessing which files were reached. Every entry is
    reported, and callers should treat any ``failed`` status as fatal
    for the migration as a whole.
    """
    workdir = Path(workdir)
    results: list[dict] = []

    targets: list[tuple[Path, str]] = [
        *((p, "nc") for p in sorted((workdir / "data" / "aggregated").rglob("*.nc"))),
        *((p, "csv") for p in sorted((workdir / "weights").glob("*.csv"))),
    ]

    for path, kind in targets:
        if dry_run:
            results.append({"path": path, "kind": kind, "status": "would-relabel"})
            continue
        try:
            status = (
                _relabel_nc(path, from_col, to_col, id_map)
                if kind == "nc"
                else (_relabel_csv(path, from_col, to_col, id_map))
            )
            results.append({"path": path, "kind": kind, "status": status})
        except Exception as exc:  # noqa: BLE001 - reported per file, fatal to the run
            results.append(
                {"path": path, "kind": kind, "status": "failed", "error": str(exc)}
            )
    return results
