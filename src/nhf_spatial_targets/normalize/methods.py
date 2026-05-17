"""Normalization and range-bound construction methods."""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def normalize_0_1(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Min/max normalize a DataArray to [0, 1] along ``dim``.

    Each non-``dim`` cell is independently normalized using its own min/max
    over ``dim``::

        norm = (x - x.min(dim)) / (x.max(dim) - x.min(dim))

    NaNs along ``dim`` are ignored when computing min/max (``skipna=True``).
    Cells where the range is zero (constant series or all-NaN) yield NaN —
    a zero-range normalization is undefined and a sentinel like 0.5 would
    fabricate spurious "middle of the range" coverage that the
    multi-source min/max combiner would then propagate as a real bound.

    The original variable's ``units`` attr is replaced with ``"1"``
    (dimensionless); other attrs are preserved.

    Used by normalized_minmax targets (recharge, soil_moisture) to bring
    sources with different native units onto a common 0-1 scale before
    multi-source min/max combination, per TM 6-B10 §3 and §4.
    """
    if dim not in da.dims:
        raise ValueError(
            f"normalize_0_1: dim {dim!r} not in DataArray dims {tuple(da.dims)!r}"
        )
    mn = da.min(dim=dim, skipna=True)
    mx = da.max(dim=dim, skipna=True)
    rng = mx - mn
    # Zero-range mask: cells where every value is the same (or all-NaN).
    # xr arithmetic with a True mask propagates NaN through the division.
    safe_rng = rng.where(rng > 0)
    norm = (da - mn) / safe_rng
    norm.attrs = dict(da.attrs)
    norm.attrs["units"] = "1"
    return norm


def normalize_0_1_by_calendar_month(da: xr.DataArray) -> xr.DataArray:
    """Min/max normalize a DataArray to [0, 1] **per calendar month**.

    Each calendar month is normalized independently — all Januaries across
    the input period are pooled and normalized to [0, 1]; all Februaries
    are pooled separately and normalized to [0, 1]; etc. This is the
    monthly soil-moisture target's normalization per TM 6-B10 §4 Appendix 1:
    cross-month seasonality is removed so the bound reflects relative wet/dry
    *within* each month rather than absolute differences between January and
    July.

    Implementation: groupby ``time.month`` and apply :func:`normalize_0_1`
    along ``time`` within each group. The output preserves the input's
    ``(time, ...)`` shape and order; only ``units`` is replaced with
    ``"1"``.

    Requires a monotonic ``time`` coordinate (no shuffling required, but
    the groupby relies on the time dim being addressable).
    """
    if "time" not in da.dims:
        raise ValueError(
            f"normalize_0_1_by_calendar_month: expected 'time' dim, got "
            f"{tuple(da.dims)!r}"
        )
    grouped = da.groupby("time.month")
    norm = grouped.map(lambda g: normalize_0_1(g, dim="time"))
    # `groupby.map` drops the synthetic 'month' coord automatically when the
    # callable returns same-dim output; the result is back on the original
    # time index. Re-stamp units to be unambiguous.
    norm.attrs = dict(da.attrs)
    norm.attrs["units"] = "1"
    return norm


def normalize_0_1_over_window(
    da: xr.DataArray, window: xr.DataArray, dim: str = "time"
) -> xr.DataArray:
    """Normalize ``da`` to [0, 1] using ``window``'s min/max along ``dim``.

    Like :func:`normalize_0_1` but the min/max are computed over a subset
    of ``da`` (the calibration / normalization period). Values in ``da``
    that fall outside ``window``'s climatology may produce values < 0 or
    > 1 — by design: TM 6-B10 §3 and §4 say the bound reflects the
    calibration-period climatology, and a future year outside that
    climatology should be visibly out-of-range, not silently re-normalized
    against extended data.

    Cells where the window's range is zero (constant series or all-NaN
    over the window) yield NaN; see :func:`normalize_0_1` for the
    reasoning.

    Parameters
    ----------
    da
        DataArray to normalize. Must have ``dim`` as one of its dimensions.
    window
        Subset of ``da`` along ``dim`` (typically a time-slice) used to
        compute the min/max. The remaining (non-``dim``) dims must match
        ``da``.
    dim
        Dimension to compute min/max over. Defaults to ``"time"``.
    """
    if dim not in da.dims:
        raise ValueError(
            f"normalize_0_1_over_window: dim {dim!r} not in DataArray dims "
            f"{tuple(da.dims)!r}"
        )
    mn = window.min(dim=dim, skipna=True)
    mx = window.max(dim=dim, skipna=True)
    rng = mx - mn
    safe_rng = rng.where(rng > 0)
    norm = (da - mn) / safe_rng
    norm.attrs = dict(da.attrs)
    norm.attrs["units"] = "1"
    return norm


def normalize_0_1_by_calendar_month_over_window(
    da: xr.DataArray, window: xr.DataArray
) -> xr.DataArray:
    """Per-calendar-month [0, 1] normalize using ``window``'s monthly min/max.

    Like :func:`normalize_0_1_by_calendar_month` but the per-calendar-month
    min/max are computed over a subset (the normalization period). All
    Januaries within ``window`` pool to give the January min/max, which
    is then applied to every January in ``da`` — including Januaries
    outside ``window``, which may produce values < 0 or > 1 by the same
    "visibly out-of-range" design as :func:`normalize_0_1_over_window`.

    Per TM 6-B10 §4 Appendix 1, this is the monthly soil moisture
    target's normalization when ``period`` and ``normalize_period``
    differ; when they are equal, the behavior collapses to
    :func:`normalize_0_1_by_calendar_month` applied to the full period.
    """
    if "time" not in da.dims:
        raise ValueError(
            f"normalize_0_1_by_calendar_month_over_window: expected 'time' "
            f"dim, got {tuple(da.dims)!r}"
        )
    if "time" not in window.dims:
        raise ValueError(
            f"normalize_0_1_by_calendar_month_over_window: window must also "
            f"have 'time' dim; got {tuple(window.dims)!r}"
        )
    # Per-calendar-month min/max over the window. Result is dim ``month``.
    window_grouped = window.groupby("time.month")
    mn_by_month = window_grouped.min("time", skipna=True)
    mx_by_month = window_grouped.max("time", skipna=True)
    # For each timestep in da, look up its calendar month's window min/max.
    months_da = da["time"].dt.month
    mn = mn_by_month.sel(month=months_da)
    mx = mx_by_month.sel(month=months_da)
    rng = mx - mn
    safe_rng = rng.where(rng > 0)
    norm = (da - mn) / safe_rng
    norm.attrs = dict(da.attrs)
    norm.attrs["units"] = "1"
    return norm


def multi_source_minmax(datasets: list) -> tuple:
    """Compute lower/upper bounds as min/max across a list of DataArrays."""
    raise NotImplementedError


def modis_ci_bounds(sca, ci, ci_threshold: float = 0.70) -> tuple:
    """Construct SCA calibration bounds from MODIS confidence interval."""
    raise NotImplementedError


def nn_fill_bounds(
    ds: xr.Dataset,
    centroids_xy: np.ndarray,
    max_candidates: int = 10,
) -> tuple[xr.Dataset, xr.DataArray]:
    """Fill NaN bound cells with the nearest *finite* HRU at the same time step.

    For every HRU position that is NaN in *both* ``lower_bound`` and
    ``upper_bound`` at any time step, this walks ``cKDTree`` neighbors in
    increasing-distance order and adopts the bound values of the first
    donor that is finite *at that time step*. If no donor among the first
    ``max_candidates`` neighbors is finite, the cell stays NaN.

    Cells where both bounds are already finite are untouched.

    Parameters
    ----------
    ds
        Dataset with ``lower_bound(time, id_col)`` and
        ``upper_bound(time, id_col)`` float vars.
    centroids_xy
        Array of shape ``(n_hrus, 2)`` with HRU centroids in an equal-area
        CRS (matching ``ds[id_col]`` order).
    max_candidates
        Maximum number of donor neighbors to consider per (time, hru)
        before giving up.

    Returns
    -------
    filled_ds, nn_filled
        ``filled_ds`` is a copy of ``ds`` with ``lower_bound`` /
        ``upper_bound`` updated; ``nn_filled`` is an int8
        ``(time, id_col)`` flag array (0 = not filled, 1 = filled).
    """
    from scipy.spatial import cKDTree

    if "lower_bound" not in ds or "upper_bound" not in ds:
        raise ValueError(
            "nn_fill_bounds requires 'lower_bound' and 'upper_bound' in ds"
        )
    id_col = next(d for d in ds["lower_bound"].dims if d != "time")
    if centroids_xy.shape != (ds.sizes[id_col], 2):
        raise ValueError(
            f"centroids_xy shape {centroids_xy.shape} does not match "
            f"({ds.sizes[id_col]}, 2)"
        )

    lower = ds["lower_bound"].values.copy()
    upper = ds["upper_bound"].values.copy()
    n_time, n_hru = lower.shape

    # Use up to (1 + max_candidates) neighbors so that index 0 (the cell itself)
    # can be skipped without losing donor budget.
    k = min(1 + max_candidates, n_hru)
    tree = cKDTree(centroids_xy)
    _, neighbor_idx = tree.query(centroids_xy, k=k)
    if k == 1:
        neighbor_idx = neighbor_idx[:, None]

    diag = np.zeros((n_time, n_hru), dtype=np.int8)
    nan_mask = np.isnan(lower) & np.isnan(upper)

    if not nan_mask.any():
        nn_diag = xr.DataArray(
            diag,
            dims=ds["lower_bound"].dims,
            coords={d: ds[d] for d in ds["lower_bound"].dims},
            name="nn_filled",
        )
        return ds.copy(), nn_diag

    # Iterate only HRUs that ever go NaN.
    nan_hrus = np.where(nan_mask.any(axis=0))[0]
    n_unfilled = 0
    for h in nan_hrus:
        candidates = neighbor_idx[h]
        # Skip self (index 0 in the kNN result).
        candidates = candidates[candidates != h]
        for t in np.where(nan_mask[:, h])[0]:
            for cand in candidates[:max_candidates]:
                lo = lower[t, cand]
                up = upper[t, cand]
                if np.isfinite(lo) and np.isfinite(up):
                    lower[t, h] = lo
                    upper[t, h] = up
                    diag[t, h] = 1
                    break
            else:
                n_unfilled += 1
    if n_unfilled:
        logger.warning(
            "nn_fill_bounds: %d (time, hru) cells stayed NaN after exhausting "
            "%d donor candidates.",
            n_unfilled,
            max_candidates,
        )

    out = ds.copy()
    out["lower_bound"] = (ds["lower_bound"].dims, lower)
    out["upper_bound"] = (ds["upper_bound"].dims, upper)
    out["lower_bound"].attrs = dict(ds["lower_bound"].attrs)
    out["upper_bound"].attrs = dict(ds["upper_bound"].attrs)
    nn_diag = xr.DataArray(
        diag,
        dims=ds["lower_bound"].dims,
        coords={d: ds[d] for d in ds["lower_bound"].dims},
        name="nn_filled",
    )
    return out, nn_diag
