"""Atomic CF-1.8 target-NC writers (single-file + bounds-with-NN-fill).

Two writers, both invoked from the per-target driver:

- :func:`write_target_nc` writes a pre-assembled Dataset atomically with
  CF-1.8 global attrs, the canonical encoding from
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

from nhf_spatial_targets.release.lineage import StepKind
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
    """Write a target Dataset to NetCDF atomically with CF-1.8 metadata.

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
    ds.attrs.setdefault("Conventions", "CF-1.8")
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
    # Every data variable except the grid-mapping container itself points at
    # the crs variable. Iterating data_vars rather than a fixed name list
    # keeps ensemble members and statistics covered as the schema grows.
    for _v in ds.data_vars:
        if _v != "crs":
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
    # int8 for the two flag diagnostics, float32 for every other data
    # variable (bounds, ensemble members, ensemble statistics). Derived from
    # data_vars rather than a fixed list so a new variable cannot silently
    # fall through to float64 on disk. `crs` is a 0-dim int32 grid-mapping
    # container minted above and carries no encoding.
    target_dtypes = {v: "int8" for v in ("n_sources", "nn_filled") if v in ds.data_vars}
    target_dtypes.update(
        {v: "float32" for v in ds.data_vars if v not in target_dtypes and v != "crs"}
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
    target_key: str | None = None,
    members: dict[str, xr.DataArray] | None = None,
    emit_members: bool = False,
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
    target_key
        Canonical target identifier from the adapter (``"runoff"``,
        ``"aet"``, ``"rch"``, ``"som"``, ``"sca"``, ``"swe"``); used to
        build the lineage step ``command`` field as ``run-<target_key>``.
        Falls back to ``"target"`` when ``None`` so out-of-pipeline callers
        without an adapter still work.
    members
        Per-source contributions keyed by source key, from
        ``SourceLoaderResult.members``. Recorded as the ``member_keys``
        global attr whenever present (the EFFECTIVE, post-availability-
        filter member list), and written as named data variables when
        ``emit_members`` is True. Deliberately distinct from
        ``source_keys``, which ``_driver._common_global_attrs`` already
        stamps from the CONFIGURED ``targets.<t>.sources`` list and which
        the publish gate (``release.publish._config_product_problems``)
        compares against config with strict equality — overwriting it
        here would silently redefine it to the effective subset and break
        that gate the moment a source is unavailable for a given build.
    emit_members
        Whether to write the members and the derived ``ensemble_mean`` /
        ``ensemble_std`` variables. Purely an output switch: the bounds
        and ``n_sources`` are byte-identical either way. Raises when
        True and ``members`` is empty, rather than silently ignoring the
        operator's config.
    """
    if emit_members and not members:
        raise ValueError(
            "write_bounds_target: emit_members is True but no members were "
            "supplied by the target's source_loader. Set emit_members=False "
            "for targets without a member decomposition (e.g. SCA, whose "
            "bounds are a CI interval rather than a member min/max)."
        )

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

    data_vars: dict[str, xr.DataArray] = {
        "lower_bound": lower,
        "upper_bound": upper,
        "n_sources": n_sources,
    }
    if emit_members:
        from nhf_spatial_targets.targets._combine import ensemble_stats

        mean, std = ensemble_stats(members, n_sources)
        mean.name = "ensemble_mean"
        std.name = "ensemble_std"
        mean.attrs = {
            "units": bounds_units,
            "long_name": f"ensemble mean of {bounds_long_name_kind}",
            "cell_methods": cell_methods,
            "coordinates": "centroid_lat centroid_lon",
            "ancillary_variables": "n_sources",
        }
        std.attrs = {
            "units": bounds_units,
            "long_name": (
                f"ensemble standard deviation of {bounds_long_name_kind} "
                "(population, NaN where n_sources < 2)"
            ),
            "cell_methods": cell_methods,
            "coordinates": "centroid_lat centroid_lon",
            "ancillary_variables": "n_sources",
        }
        data_vars["ensemble_mean"] = mean
        data_vars["ensemble_std"] = std
        for key, member_da in members.items():
            # rename(key) alone is NOT enough: DataArray.rename(str) with
            # no attrs given reuses the SAME underlying Variable object, so
            # `member.attrs = {...}` below would mutate member_da.attrs in
            # place -- i.e. corrupt the loader's own member DataArray that
            # the caller still holds a reference to (issue #338 fix round
            # 2, finding 4). .copy(deep=False) after rename() decouples the
            # Variable (and its attrs dict) while still sharing the
            # underlying data buffer, keeping the memory win .copy()
            # (a full deep copy) would have given up.
            member = member_da.rename(key).copy(deep=False)
            # long_name always describes the TARGET-units quantity the
            # emitted variable actually holds (issue #338 fix round 3,
            # finding 1) -- the shim's own description (e.g. "ERA5-Land
            # ssro (m/month, summed to mm/year)") documents the SOURCE's
            # native units, which contradicts `units` below once the
            # value has been converted. Preserve that description under
            # a separate provenance attr instead of discarding it.
            source_description = member_da.attrs.get("long_name")
            member.attrs = {
                "units": bounds_units,
                "long_name": f"{key} contribution to {bounds_long_name_kind}",
                "cell_methods": cell_methods,
                "coordinates": "centroid_lat centroid_lon",
            }
            if source_description:
                member.attrs["source_description"] = source_description
            data_vars[key] = member

    extra_global_attrs = dict(extra_global_attrs)
    extra_global_attrs["members_emitted"] = "true" if emit_members else "false"
    if members:
        # member_keys is the EFFECTIVE (post-availability-filter) member
        # list -- deliberately NOT source_keys, which
        # _driver._common_global_attrs already stamps from the CONFIGURED
        # targets.<t>.sources list and which the publish gate compares
        # against config with strict equality (release/publish.py
        # _config_product_problems). Overwriting source_keys here would
        # silently redefine it to the effective subset and break that gate.
        extra_global_attrs["member_keys"] = ",".join(members)

    ds = xr.Dataset(
        data_vars,
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
        target_key=target_key,
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
    if emit_members:
        # nn_fill_bounds returns ds.copy() with only lower_bound/upper_bound
        # overwritten, so members + ensemble_mean/ensemble_std would
        # otherwise ride along into the companion completely unfilled. Drop
        # them: the NN-filled companion carries only the filled bounds,
        # n_sources, and the nn_filled flag (spec Sec 3.6) — NN-filling an
        # individual member would fabricate a source observation at an HRU
        # that source never covered.
        filled_ds = filled_ds.drop_vars([*members, "ensemble_mean", "ensemble_std"])
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
    if emit_members:
        # The companion carries no member data vars (dropped above), so it
        # must not claim members_emitted="true" -- that would read as
        # "emitted and then lost" instead of "never emitted here", exactly
        # the ambiguity the attr exists to resolve. member_keys names
        # variables that are not on this file, so drop it too.
        filled_attrs["members_emitted"] = "false"
        filled_attrs.pop("member_keys", None)
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
        target_key=target_key,
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
    kind: StepKind,
    params: dict,
    extra_global_attrs: dict,
    target_key: str | None,
) -> None:
    """Append one target-stage lineage step for *output_path*.

    Forwards the comma-separated upstream-source list from
    ``extra_global_attrs["source"]`` onto the step so consumers don't
    need to re-open the NC to recover provenance.
    """
    from nhf_spatial_targets.release.lineage import append_step, output_file_entry

    step_params = dict(params)
    if "source" in extra_global_attrs:
        step_params["source"] = extra_global_attrs["source"]
    if "period" in extra_global_attrs:
        step_params["period"] = extra_global_attrs["period"]
    command = f"run-{target_key}" if target_key else "run-target"
    append_step(
        project.manifest_path,
        kind=kind,
        source_key=None,
        outputs=[output_file_entry(output_path)],
        params=step_params,
        command=command,
    )
