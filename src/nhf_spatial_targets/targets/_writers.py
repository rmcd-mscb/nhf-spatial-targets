"""Atomic CF-1.6 target-NC writers (single-file + bounds-with-NN-fill).

Two writers, both invoked from the per-target driver:

- :func:`write_target_nc` writes a pre-assembled Dataset atomically with
  CF-1.6 global attrs, the canonical encoding from
  :func:`io_nc.build_encoding`, and an explicit sort-on-emission for the
  HRU dimension (issue #93).
- :func:`write_bounds_target` is the higher-level helper used by every
  multi-source-minmax builder: it assembles ``lower_bound`` /
  ``upper_bound`` / ``n_sources`` / centroid coords / time_bnds, calls
  :func:`write_target_nc`, then optionally writes the NN-filled
  companion via :func:`normalize.methods.nn_fill_bounds`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nhf_spatial_targets.targets._combine import build_n_sources_attrs
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


def _target_encoding_without_chunks(
    ds: xr.Dataset, var_dtype: dict[str, str]
) -> dict[str, dict]:
    """Target encoding (dtype/zlib/fill/time) minus per-HRU ``chunksizes``.

    Only used by :func:`write_target_nc` when no ``sort_dim``/``id_col`` is
    supplied, so the per-HRU chunk dimension is unknown and HDF5 auto-chunks.
    Reuses ``io_nc``'s fill-value and time policy so this fallback stays
    consistent with the chunked ``build_encoding`` path.
    """
    from nhf_spatial_targets.io_nc import _TIME_ENCODING, _fill_value_for

    encoding: dict[str, dict] = {}
    for name, dt in var_dtype.items():
        dtype = np.dtype(dt)
        enc: dict = {
            "dtype": str(dtype),
            "zlib": True,
            "complevel": 4,
            "_FillValue": _fill_value_for(dtype),
            # Explicit shuffle: netCDF4 defaults it True under zlib (see
            # io_nc.build_encoding), so omitting it for floats would enable it.
            "shuffle": bool(np.issubdtype(dtype, np.integer)),
        }
        encoding[name] = enc
    for tvar in ("time", "time_bnds"):
        if tvar in ds.variables:
            encoding[tvar] = dict(_TIME_ENCODING)
    # time_bnds is a CF boundary variable and must not carry _FillValue
    # (CF §7.1); mirror io_nc.build_encoding and pin it off.
    if "time_bnds" in ds.variables:
        encoding["time_bnds"]["_FillValue"] = None
    return encoding


def write_target_nc(
    ds: xr.Dataset,
    output_path: Path,
    title: str,
    extra_global_attrs: dict | None = None,
    sort_dim: str | None = None,
) -> None:
    """Write a target Dataset to NetCDF atomically with CF-1.6 metadata.

    The Dataset is expected to already carry the data variables, ancillary
    coordinates (``time_bnds``, ``centroid_lat``, ``centroid_lon``), and
    per-variable attrs (``units``, ``long_name``, ``cell_methods``, etc.).
    This helper sets the global ``Conventions`` / ``title`` / ``history`` /
    ``software_version`` attrs, then delegates encoding to
    :func:`io_nc.build_encoding` (``layer="target"``): float32+zlib bounds,
    int8+zlib diagnostics, the pinned ``proleptic_gregorian`` time axis, and
    per-HRU-time-series ``chunksizes`` so a single HRU's calibration read is
    one ~1 MiB chunk (issue #165 ST2). The write goes through
    :func:`io_nc.atomic_to_netcdf` (tempfile + rename) so a partial NetCDF
    never lands at the final path. When ``sort_dim`` is omitted the HRU
    chunk dim is unknown, so the same dtype/compression/time policy is applied
    without per-HRU chunking.

    When ``sort_dim`` is given, the Dataset is sorted ascending on that
    dimension before write. Target builders pass ``project.id_col`` here
    to enforce the canonical HRU row order at the emission boundary
    (issue #93). Upstream helpers (``read_aggregated_source``,
    ``compute_hru_area_and_centroids``) already produce sorted data; the
    explicit sort here makes the invariant unmistakable at the file boundary.
    """
    from datetime import datetime, timezone

    from nhf_spatial_targets import __version__

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = ds.copy()
    if sort_dim is not None:
        ds = ds.sortby(sort_dim)
    ds.attrs.setdefault("Conventions", "CF-1.6")
    ds.attrs["title"] = title
    ds.attrs["history"] = (
        f"{datetime.now(timezone.utc).isoformat()} created by "
        f"nhf_spatial_targets v{__version__}"
    )
    ds.attrs.setdefault("institution", "USGS")
    ds.attrs.setdefault("software_version", __version__)
    if extra_global_attrs:
        ds.attrs.update(extra_global_attrs)

    # CF §5.6: anchor the spatial reference. Target bound vars carry
    # grid_mapping="crs" (inherited from the aggregated sources), but no crs
    # container flows through the multi-source combine, leaving the reference
    # dangling. The HRU centroid coords (centroid_lat/lon) are EPSG:4326, so
    # mint a 0-dim WGS84 latitude_longitude grid-mapping variable to match.
    from pyproj import CRS as _CRS

    _wgs84 = _CRS.from_epsg(4326)
    _crs_attrs = dict(_wgs84.to_cf())
    _crs_attrs.setdefault("crs_wkt", _wgs84.to_wkt())
    ds["crs"] = xr.DataArray(np.int32(0), attrs=_crs_attrs)
    for _v in ("lower_bound", "upper_bound", "n_sources", "nn_filled"):
        if _v in ds.data_vars:
            ds[_v].attrs["grid_mapping"] = "crs"
    # CF §3: the HRU index coordinate is an identifier, not a measurement —
    # label it and carry no units.
    if sort_dim is not None and sort_dim in ds.variables:
        ds[sort_dim].attrs.pop("units", None)
        ds[sort_dim].attrs["long_name"] = "HRU Index"

    from nhf_spatial_targets.io_nc import atomic_to_netcdf, build_encoding

    # Pin the on-disk dtype for each known target var: float32 bounds, int8
    # diagnostics. build_encoding derives _FillValue / shuffle from the dtype
    # (NaN + no-shuffle for floats, no-fill + shuffle for the int8 diagnostics).
    target_dtypes = {
        v: "float32" for v in ("lower_bound", "upper_bound") if v in ds.data_vars
    }
    target_dtypes.update(
        {v: "int8" for v in ("n_sources", "nn_filled") if v in ds.data_vars}
    )

    if sort_dim is not None:
        encoding = build_encoding(
            ds, layer="target", hru_dim=sort_dim, var_dtype=target_dtypes
        )
    else:
        # No id_col known — production target builders always pass sort_dim, so
        # this is only the bare ``write_target_nc(ds, out, title=...)`` path.
        # Apply the same dtype/compression/time policy minus per-HRU chunking
        # (HDF5 auto-chunks), since the HRU dim name is unavailable here.
        encoding = _target_encoding_without_chunks(ds, target_dtypes)

    atomic_to_netcdf(ds, output_path, encoding=encoding)
    logger.info("Wrote %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


def write_bounds_target(
    *,
    project: Project,
    lower: xr.DataArray,
    upper: xr.DataArray,
    n_sources: xr.DataArray,
    n_sources_count: int,
    time_index: pd.DatetimeIndex,
    time_offset_unit: object,
    bounds_units: str,
    bounds_long_name_kind: str,
    cell_methods: str,
    output_path: Path,
    title: str,
    nn_title: str,
    extra_global_attrs: dict,
    hru_meta: "pd.DataFrame",
    nn_fill: bool,
    nn_max_candidates: int,
    id_col: str,
) -> None:
    """Assemble + write a bounds-target Dataset, with optional NN-fill companion.

    Consolidates the assemble-and-write pipeline shared by every target
    builder (runoff, AET, recharge, soil moisture): centroid coords,
    ``time_bnds``, per-variable attrs (units / long_name / cell_methods /
    coordinates), global attrs, atomic write via ``write_target_nc``, a
    coverage-summary log line, and the optional ``nn_fill_bounds``
    companion file.

    Parameters
    ----------
    project
        Loaded :class:`~nhf_spatial_targets.workspace.Project`.
    lower, upper, n_sources
        The three combined-source DataArrays from
        :func:`multi_source_nanminmax`.
    n_sources_count
        Total number of source contributors (an int); drives the
        ``n_sources`` diagnostic's ``flag_values`` length.
    time_index
        Master ``DatetimeIndex`` that ``lower`` / ``upper`` are aligned to.
    time_offset_unit
        Offset added to each ``time_index`` entry to form ``time_bnds``'s
        upper edge (e.g. ``pd.offsets.MonthBegin(1)`` for monthly,
        ``pd.offsets.YearBegin(1)`` for annual).
    bounds_units
        Units string for the lower/upper variable attrs (e.g. ``"cfs"``,
        ``"inches/day"``, ``"1"``).
    bounds_long_name_kind
        Substituted into the ``long_name`` template: ``"lower bound of
        {kind} (NaN-aware min across sources)"``. Examples: ``"monthly
        runoff"``, ``"annual recharge"``.
    cell_methods
        CF ``cell_methods`` attr value for both bounds (e.g.
        ``"time: sum"``, ``"time: mean"``).
    output_path
        Final NetCDF path for the unfilled target.
    title
        ``title`` global attr for the unfilled target.
    nn_title
        ``title`` global attr for the NN-filled companion (only used when
        ``nn_fill`` is True).
    extra_global_attrs
        Per-target metadata (``source``, ``period``, ``fabric_sha256``,
        etc.) — passed through to ``write_target_nc``.
    hru_meta
        DataFrame returned by ``compute_hru_centroids`` (or the combined
        helper). Must contain ``centroid_lat``, ``centroid_lon``,
        ``centroid_x``, ``centroid_y`` columns.
    nn_fill
        If True, additionally write ``<output>_nn_filled.nc`` via
        :func:`nn_fill_bounds`.
    nn_max_candidates
        Forwarded to :func:`nn_fill_bounds`.
    id_col
        HRU id column name (e.g. ``"nhm_id"``); the dataset is sorted
        ascending on this dim at emission per the #93 canonical-row-order
        invariant.
    """
    # Avoid a circular-import by deferring this helper-internal import.
    from nhf_spatial_targets.normalize.methods import nn_fill_bounds

    lower.name = "lower_bound"
    upper.name = "upper_bound"
    n_sources.name = "n_sources"

    time_bnds = xr.DataArray(
        list(zip(time_index.values, (time_index + time_offset_unit).values)),
        dims=("time", "nv"),
        coords={"time": time_index.values},
        name="time_bnds",
    )
    centroid_lat = xr.DataArray(
        hru_meta["centroid_lat"].values,
        dims=(id_col,),
        coords={id_col: hru_meta.index.values},
        attrs={
            "units": "degrees_north",
            "standard_name": "latitude",
            "long_name": "HRU centroid latitude",
        },
    )
    centroid_lon = xr.DataArray(
        hru_meta["centroid_lon"].values,
        dims=(id_col,),
        coords={id_col: hru_meta.index.values},
        attrs={
            "units": "degrees_east",
            "standard_name": "longitude",
            "long_name": "HRU centroid longitude",
        },
    )

    lower.attrs.update(
        {
            "units": bounds_units,
            "long_name": (
                f"lower bound of {bounds_long_name_kind} (NaN-aware min across sources)"
            ),
            "cell_methods": cell_methods,
            "coordinates": "centroid_lat centroid_lon",
        }
    )
    upper.attrs.update(
        {
            "units": bounds_units,
            "long_name": (
                f"upper bound of {bounds_long_name_kind} (NaN-aware max across sources)"
            ),
            "cell_methods": cell_methods,
            "coordinates": "centroid_lat centroid_lon",
        }
    )
    n_sources.attrs.update(build_n_sources_attrs(n_sources_count))

    ds = xr.Dataset(
        {
            "lower_bound": lower,
            "upper_bound": upper,
            "n_sources": n_sources,
        },
        coords={
            "time": time_index,
            id_col: lower[id_col],
            "time_bnds": time_bnds,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
        },
    )
    ds["time"].attrs["bounds"] = "time_bnds"
    ds["time"].attrs["axis"] = "T"
    ds["time"].attrs["standard_name"] = "time"
    ds[id_col].attrs["long_name"] = "HRU identifier"
    ds[id_col].attrs["cf_role"] = "timeseries_id"

    ds_loaded = ds.compute()

    write_target_nc(
        ds_loaded,
        output_path,
        title=title,
        extra_global_attrs=extra_global_attrs,
        sort_dim=id_col,
    )
    _append_target_step(
        project=project,
        output_path=output_path,
        kind="target",
        params={
            "bounds_long_name_kind": bounds_long_name_kind,
            "n_sources_count": int(n_sources_count),
            "id_col": id_col,
        },
        extra_global_attrs=extra_global_attrs,
    )

    n = ds_loaded["n_sources"].values
    total = n.size
    none = int((n == 0).sum())
    logger.info(
        "%s coverage: %d/%d cells have >=1 finite source (%.2f%% all-NaN)",
        bounds_long_name_kind,
        total - none,
        total,
        100.0 * none / total if total else 0.0,
    )

    if not nn_fill:
        return

    centroids_xy = hru_meta[["centroid_x", "centroid_y"]].values
    filled_ds, nn_diag = nn_fill_bounds(
        ds_loaded, centroids_xy, max_candidates=nn_max_candidates
    )
    nn_diag.attrs.update(
        {
            "units": "1",
            "long_name": "nearest-neighbor fill flag",
            # int8 to match the on-disk nn_filled dtype (CF §3.5).
            "flag_values": np.array([0, 1], dtype="int8"),
            "flag_meanings": "not_filled filled",
            "coordinates": "centroid_lat centroid_lon",
        }
    )
    filled_ds["nn_filled"] = nn_diag
    filled_attrs = dict(extra_global_attrs)
    filled_attrs["nn_fill_max_candidates"] = nn_max_candidates
    filled_attrs["nn_fill_distance_crs"] = project.area_crs
    nn_path = output_path.with_name(
        output_path.stem + "_nn_filled" + output_path.suffix
    )
    write_target_nc(
        filled_ds,
        nn_path,
        title=nn_title,
        extra_global_attrs=filled_attrs,
        sort_dim=id_col,
    )
    _append_target_step(
        project=project,
        output_path=nn_path,
        kind="nn_fill",
        params={
            "bounds_long_name_kind": bounds_long_name_kind,
            "nn_fill_max_candidates": int(nn_max_candidates),
            "nn_fill_distance_crs": project.area_crs,
            "id_col": id_col,
        },
        extra_global_attrs=filled_attrs,
    )


def _append_target_step(
    *,
    project: Project,
    output_path: Path,
    kind: str,
    params: dict,
    extra_global_attrs: dict,
) -> None:
    """Append one target-stage lineage step for *output_path*.

    Helper for :func:`write_bounds_target` (release PR-B). Lifts the
    sha256 + size + mtime fingerprint of the just-written NC onto
    ``manifest.json.steps[]`` so PR-D's FGDC generator can populate
    ``dataquality.lineage.processstep[]`` without re-walking the
    targets dir. The catalog ``source`` (multi-source-derived target's
    upstream key list) is forwarded into ``params["source"]`` so the
    step record stays linkable to the parent aggregate steps.
    """
    from nhf_spatial_targets.release.lineage import append_step, output_file_entry

    step_params = dict(params)
    # ``extra_global_attrs["source"]`` carries the comma-separated upstream
    # source list (e.g. ``"era5_land, gldas_noah_v21_monthly, mwbm_climgrid"``)
    # built by ``_common_global_attrs``. Forwarding it onto the step record
    # makes the multi-source provenance explicit without re-reading the NC.
    if "source" in extra_global_attrs:
        step_params["source"] = extra_global_attrs["source"]
    if "period" in extra_global_attrs:
        step_params["period"] = extra_global_attrs["period"]
    append_step(
        project.manifest_path,
        kind=kind,
        source_key=None,
        outputs=[output_file_entry(output_path)],
        params=step_params,
        command=f"run-{step_params.get('bounds_long_name_kind', 'target').split()[-1]}",
    )
