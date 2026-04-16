"""CF-based coordinate detection for source datasets."""

from __future__ import annotations

import xarray as xr

_X_STANDARD_NAMES = frozenset({"longitude", "projection_x_coordinate"})
_Y_STANDARD_NAMES = frozenset({"latitude", "projection_y_coordinate"})
_T_STANDARD_NAMES = frozenset({"time"})


def _find_axis(
    ds: xr.Dataset,
    dims: tuple[str, ...],
    axis_letter: str,
    standard_names: frozenset[str],
) -> str | None:
    # First pass: CF axis attribute.
    for name in dims:
        if name in ds.coords and ds.coords[name].attrs.get("axis") == axis_letter:
            return name
    # Second pass: CF standard_name.
    for name in dims:
        if (
            name in ds.coords
            and ds.coords[name].attrs.get("standard_name") in standard_names
        ):
            return name
    return None


def detect_coords(
    ds: xr.Dataset,
    var: str,
    x_override: str | None = None,
    y_override: str | None = None,
    time_override: str | None = None,
) -> tuple[str, str, str]:
    """Return (x_coord, y_coord, time_coord) for ``ds[var]``.

    Resolution order per axis:
      1. Explicit override (must be one of ``ds[var].dims``).
      2. Coordinate whose ``axis`` attr is 'X' / 'Y' / 'T'.
      3. Coordinate whose ``standard_name`` attr matches a CF name for that axis.

    Raises KeyError if ``var`` is not in ``ds``.
    Raises ValueError if an override is not in ``ds[var].dims`` or if any axis
    cannot be resolved after overrides + CF passes.
    """
    if var not in ds.data_vars:
        raise KeyError(f"Variable {var!r} not in dataset (have {list(ds.data_vars)})")
    dims = tuple(ds[var].dims)

    def _resolve(
        override: str | None,
        axis_letter: str,
        standard_names: frozenset[str],
        label: str,
    ) -> str:
        if override is not None:
            if override not in dims:
                raise ValueError(
                    f"{label} override {override!r} is not a dim of {var!r} "
                    f"(dims={dims})"
                )
            return override
        found = _find_axis(ds, dims, axis_letter, standard_names)
        if found is None:
            attrs_by_dim = {d: dict(ds.coords[d].attrs) for d in dims if d in ds.coords}
            raise ValueError(
                f"Could not detect {label} coord for {var!r}. "
                f"No dim has axis={axis_letter!r} or standard_name in "
                f"{sorted(standard_names)}. "
                f"dims={dims}, coord attrs={attrs_by_dim}"
            )
        return found

    x = _resolve(x_override, "X", _X_STANDARD_NAMES, "x")
    y = _resolve(y_override, "Y", _Y_STANDARD_NAMES, "y")
    t = _resolve(time_override, "T", _T_STANDARD_NAMES, "time")
    return x, y, t
