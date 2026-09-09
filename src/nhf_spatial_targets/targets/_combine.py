"""Multi-source combiners for target builders.

Three combiners are currently in use:

- :func:`multi_source_nanminmax`: NaN-aware per-cell min/max with an int8
  finite-source count diagnostic. Used by runoff, AET, recharge, soil
  moisture, and SWE. The bound is NaN only when *every* source is NaN at
  that (HRU, time).
- The CI-bounded combiner is inlined in ``targets/sca.py`` because it is
  single-source and intermixed with the July/August forced-zero rule;
  exposing it here would be an over-generalization for the only caller.
- The normalized-minmax combiner is the same :func:`multi_source_nanminmax`
  composed downstream of ``normalize/methods.py:normalize_0_1_*``; recharge
  and SOM call those helpers directly from their builders.

:func:`build_n_sources_attrs` constructs the per-variable attrs dict for
the ``n_sources`` diagnostic so the flag_values / flag_meanings pair stays
in sync with the source count.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def multi_source_nanminmax(
    sources: dict[str, xr.DataArray],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """NaN-aware per-cell min, max, and finite-source count.

    All input DataArrays must share dims and coords (typically
    ``(time, id_col)``). They are stacked on a new ``source`` dim and
    reduced with ``skipna=True``.

    A bound is defined whenever ≥1 source is finite at that cell; the result
    is NaN only when *every* source is NaN there, which is exactly when
    ``n_sources == 0``.

    Parameters
    ----------
    sources
        Mapping from source key to per-source DataArray.

    Returns
    -------
    lower, upper, n_sources
        ``(time, id_col)`` arrays. ``n_sources`` is int8 with values in
        ``[0, len(sources)]``; ``lower`` / ``upper`` preserve the input
        dtype (typically float32).

    Raises
    ------
    ValueError
        If any two sources have different HRU coords (different fabrics).
    """
    keys = list(sources.keys())
    if not keys:
        raise ValueError("multi_source_nanminmax: empty sources dict")
    ref = sources[keys[0]]
    hru_dim = next(d for d in ref.dims if d != "time")
    for k in keys[1:]:
        other = sources[k]
        if not other[hru_dim].equals(ref[hru_dim]):
            raise ValueError(
                f"HRU coords differ between sources '{keys[0]}' and '{k}'. "
                "All sources must be aggregated to the same fabric."
            )

    stacked = xr.concat([sources[k] for k in keys], dim=xr.Variable("source", keys))
    lower = stacked.min(dim="source", skipna=True)
    upper = stacked.max(dim="source", skipna=True)
    n_sources = stacked.notnull().sum(dim="source").astype(np.int8)
    return lower, upper, n_sources


def build_n_sources_attrs(
    n_sources_count: int,
    ancillary_coords: str = "centroid_lat centroid_lon",
) -> dict:
    """Build the per-variable attrs dict for an ``n_sources`` diagnostic var.

    Parameters
    ----------
    n_sources_count
        Number of source contributors (an integer ≥ 0 and ≤ 5). Determines
        the length of the ``flag_values`` list and the matching
        ``flag_meanings`` labels (``none one two three four five``,
        truncated to ``n_sources_count + 1`` entries).
    ancillary_coords
        Space-separated list of ancillary coordinate variable names to
        record under CF's ``coordinates`` attr. Defaults to the centroid
        pair used by every target builder.
    """
    flag_labels = ["none", "one", "two", "three", "four", "five"]
    if n_sources_count + 1 > len(flag_labels):
        raise ValueError(
            f"build_n_sources_attrs: n_sources_count={n_sources_count} exceeds "
            f"the {len(flag_labels) - 1}-source label vocabulary."
        )
    return {
        "units": "1",
        "long_name": "number of finite source contributions",
        # int8 to match the on-disk n_sources dtype (CF §3.5 requires
        # flag_values dtype == parent variable dtype).
        "flag_values": np.array(range(0, n_sources_count + 1), dtype="int8"),
        "flag_meanings": " ".join(flag_labels[: n_sources_count + 1]),
        "coordinates": ancillary_coords,
    }


def ensemble_stats(
    members: dict[str, xr.DataArray],
    n_sources: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """NaN-aware ensemble mean and population standard deviation.

    Both are reduced over a stacked ``source`` dim with ``skipna=True``, so
    the mean is defined wherever at least one source is finite.

    ``std`` is masked to NaN wherever ``n_sources < 2``. With a single finite
    source the population standard deviation is exactly 0, which a downstream
    calibration weight would read as perfect inter-source agreement rather
    than as "only one source was available here". Masking makes the
    distinction explicit; ``n_sources`` remains the authoritative coverage
    diagnostic.

    Parameters
    ----------
    members
        Mapping from source key to per-source DataArray. All must share
        dims and coords (typically ``(time, id_col)``).
    n_sources
        Per-cell finite-source count from
        :func:`multi_source_nanminmax`, used for the ``std`` mask.

    Returns
    -------
    mean, std
        ``(time, id_col)`` arrays matching the members' shape.

    Raises
    ------
    ValueError
        If ``members`` is empty.
    """
    keys = list(members.keys())
    if not keys:
        raise ValueError("ensemble_stats: empty members dict")
    stacked = xr.concat([members[k] for k in keys], dim=xr.Variable("source", keys))
    mean = stacked.mean(dim="source", skipna=True)
    std = stacked.std(dim="source", skipna=True, ddof=0).where(n_sources >= 2)
    return mean, std
