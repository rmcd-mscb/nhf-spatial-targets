"""Margulis Western US Snow Reanalysis daily SWE aggregator.

Reads per-year consolidated NCs at
``<datastore>/margulis_wus_sr/daily/margulis_wus_sr_daily_<year>.nc``
(written by ``fetch.margulis_wus_sr`` / its consolidator) and emits
area-weighted HRU means via the shared aggregation driver.

The consolidated NCs store ``SWE`` as float32 in **metres water
equivalent** on a regular WGS84 lat/lon grid (the consolidator
reprojects from the native ~90 m EASE-Grid). Per-pixel NaN over
unmodelled cells decodes naturally via ``mask_and_scale=True``; the
default ``stat_method="mean"`` is the right choice since the source
does not apply any per-pixel quality gate in a ``pre_aggregate_hook``
— HRUs that straddle the WUS domain edge become honest NaN, which the
SWE target builder handles at multi-source combination time.

The source covers only the Western US. The aggregation driver's
geometry-driven coverage guard (#309) skips fabric batches outside the
grid and emits honest NaN rows for them, so on a CONUS fabric the
aggregated NC carries data only at WUS HRUs; the SWE target's NaN-aware
combine uses it wherever it is finite.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from nhf_spatial_targets import catalog
from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source
from nhf_spatial_targets.io_nc import atomic_to_netcdf
from nhf_spatial_targets.release.lineage import iso_from_mtime

logger = logging.getLogger(__name__)

_SOURCE_KEY = "margulis_wus_sr"
_MASK_RELPATH = Path(_SOURCE_KEY) / "derived" / "perennial_snow_mask.nc"
_EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class MaskRule:
    """Perennial-snow rule parameters, read from the catalog (#363)."""

    swe_threshold_m: float
    check_month_day: str
    min_year_fraction: float

    @classmethod
    def from_catalog(cls) -> MaskRule:
        """Load the rule from ``margulis_wus_sr.perennial_snow_mask``.

        Raises ``KeyError`` if the block or a field is missing and
        ``ValueError`` if a field is invalid.
        """
        block = catalog.source(_SOURCE_KEY)["perennial_snow_mask"]
        thr = float(block["swe_threshold_m"])
        md = str(block["check_month_day"])
        frac = float(block["min_year_fraction"])
        if thr <= 0:
            raise ValueError(f"swe_threshold_m must be > 0, got {thr}")
        if not 0.0 <= frac < 1.0:
            raise ValueError(f"min_year_fraction must be in [0, 1), got {frac}")
        try:
            pd.Timestamp(f"2001-{md}")
        except ValueError as exc:
            raise ValueError(f"check_month_day {md!r} is not a valid MM-DD") from exc
        return cls(swe_threshold_m=thr, check_month_day=md, min_year_fraction=frac)


def mask_fingerprint(rule: MaskRule, years: list[int]) -> str:
    """12-hex-char sha256 of the rule parameters plus the sorted input years."""
    payload = {
        "swe_threshold_m": rule.swe_threshold_m,
        "check_month_day": rule.check_month_day,
        "min_year_fraction": rule.min_year_fraction,
        "years": sorted(int(y) for y in years),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def classify_perennial(
    held: ArrayLike, valid: ArrayLike, min_year_fraction: float
) -> np.ndarray:
    """Return int8 0/1: perennial iff ``held / valid > min_year_fraction``.

    ``valid == 0`` (never finite) is never perennial. The comparison is
    strict, so exactly half the valid years does not qualify at 0.5.
    """
    h = np.asarray(held, dtype="float64")
    v = np.asarray(valid, dtype="float64")
    # held > frac * valid avoids division; valid == 0 -> 0 > 0 is False.
    return (h > min_year_fraction * v).astype(np.int8)


def _year_files(datastore: Path) -> list[tuple[int, Path]]:
    daily = datastore / _SOURCE_KEY / "daily"
    out = []
    for f in sorted(daily.glob("margulis_wus_sr_daily_*.nc")):
        out.append((int(f.stem.rsplit("_", 1)[1]), f))
    return out


def _check_slice(path: Path, year: int, month_day: str) -> xr.DataArray | None:
    """Return the check-date SWE slice (lazy-loaded), or None if absent."""
    with xr.open_dataset(path) as ds:
        da = ds["SWE"].sel(time=slice(f"{year}-{month_day}", f"{year}-{month_day}"))
        if da.sizes["time"] == 0:
            return None
        return da.isel(time=0, drop=True).load()


def _read_fingerprint(path: Path) -> str | None:
    try:
        with xr.open_dataset(path) as ds:
            return ds.attrs.get("perennial_snow_mask_fingerprint")
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable perennial-snow mask %s (%s); rebuilding", path, exc)
        return None


def _cell_area_km2(lat: np.ndarray, lon: np.ndarray, flag: np.ndarray) -> float:
    if lat.size < 2 or lon.size < 2:
        return 0.0
    dlat = np.deg2rad(np.median(np.abs(np.diff(lat))))
    dlon = np.deg2rad(np.median(np.abs(np.diff(lon))))
    lat_r = np.deg2rad(lat)
    band = (
        _EARTH_RADIUS_KM**2
        * dlon
        * (np.sin(lat_r + dlat / 2) - np.sin(lat_r - dlat / 2))
    )
    return float((flag.astype("float64") * np.abs(band)[:, None]).sum())


def ensure_perennial_snow_mask(datastore: Path, rule: MaskRule | None = None) -> Path:
    """Return the perennial-snow mask path, (re)building it if stale.

    The mask lives at
    ``<datastore>/margulis_wus_sr/derived/perennial_snow_mask.nc`` and is
    rebuilt only when its stored fingerprint differs from the fingerprint of
    the current rule plus the years on disk that have a check-date timestep.
    """
    rule = rule or MaskRule.from_catalog()
    datastore = Path(datastore)
    path = datastore / _MASK_RELPATH

    slices: dict[int, tuple[Path, xr.DataArray]] = {}
    for year, f in _year_files(datastore):
        da = _check_slice(f, year, rule.check_month_day)
        if da is None:
            logger.info(
                "Margulis %s has no %s-%s timestep; left out of the "
                "perennial-snow mask",
                f.name,
                year,
                rule.check_month_day,
            )
            continue
        slices[year] = (f, da)
    if not slices:
        raise FileNotFoundError(
            f"No Margulis daily files with a {rule.check_month_day} timestep "
            f"under {datastore / _SOURCE_KEY / 'daily'}; run "
            f"`nhf-targets fetch {_SOURCE_KEY}` first."
        )

    years = sorted(slices)
    fingerprint = mask_fingerprint(rule, years)
    if path.exists() and _read_fingerprint(path) == fingerprint:
        return path

    first = slices[years[0]][1]
    lat, lon = first["lat"].values, first["lon"].values
    for year in years[1:]:
        da = slices[year][1]
        for name, ref in (("lat", lat), ("lon", lon)):
            if da[name].shape != ref.shape or not np.array_equal(da[name].values, ref):
                raise ValueError(
                    f"Margulis {year} {name} coordinates differ from {years[0]}; "
                    "the perennial-snow mask needs one common grid."
                )

    held = np.zeros((lat.size, lon.size), dtype="int16")
    valid = np.zeros_like(held)
    for year in years:
        arr = slices[year][1].transpose("lat", "lon").values
        finite = np.isfinite(arr)
        valid += finite
        # NaN >= thr is False (and warns); mask first.
        held += finite & (np.where(finite, arr, 0.0) >= rule.swe_threshold_m)
    perennial = classify_perennial(held, valid, rule.min_year_fraction)

    newest = max((slices[y][0] for y in years), key=lambda p: p.stat().st_mtime)
    from pyproj import CRS

    crs_attrs = dict(CRS.from_epsg(4326).to_cf())
    crs_attrs.setdefault("crs_wkt", CRS.from_epsg(4326).to_wkt())
    ds = xr.Dataset(
        {
            "perennial": (
                ("lat", "lon"),
                perennial,
                {
                    "long_name": "perennial snow / glacier pixel flag",
                    "flag_values": np.array([0, 1], dtype="int8"),
                    "flag_meanings": "seasonal perennial",
                    "grid_mapping": "crs",
                },
            ),
            "n_years_held": (
                ("lat", "lon"),
                held,
                {
                    "long_name": "years with SWE at or above threshold on check date",
                    "units": "1",
                    "grid_mapping": "crs",
                },
            ),
            "n_years_valid": (
                ("lat", "lon"),
                valid,
                {
                    "long_name": "years with finite SWE on check date",
                    "units": "1",
                    "grid_mapping": "crs",
                },
            ),
            "crs": xr.DataArray(np.int32(0), attrs=crs_attrs),
        },
        coords={
            "lat": (
                "lat",
                lat,
                {
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "axis": "Y",
                },
            ),
            "lon": (
                "lon",
                lon,
                {
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "axis": "X",
                },
            ),
        },
        attrs={
            "Conventions": "CF-1.6",
            "title": "Margulis WUS-SR perennial snow / glacier mask",
            "source": _SOURCE_KEY,
            "history": (
                f"{iso_from_mtime(newest)} derived from Margulis WUS-SR daily "
                f"files (#363)"
            ),
            "perennial_snow_mask_fingerprint": fingerprint,
            "perennial_snow_mask_swe_threshold_m": float(rule.swe_threshold_m),
            "perennial_snow_mask_check_month_day": rule.check_month_day,
            "perennial_snow_mask_min_year_fraction": float(rule.min_year_fraction),
            "perennial_snow_mask_years": ",".join(str(y) for y in years),
            "perennial_snow_mask_n_cells": int(perennial.sum()),
            "perennial_snow_mask_area_km2": _cell_area_km2(lat, lon, perennial),
        },
    )
    chunks = (min(lat.size, 512), min(lon.size, 512))
    enc_int = {
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
        "chunksizes": chunks,
        "_FillValue": None,
    }
    encoding = {
        "perennial": dict(enc_int),
        "n_years_held": dict(enc_int),
        "n_years_valid": dict(enc_int),
        "lat": {"_FillValue": None},
        "lon": {"_FillValue": None},
    }
    atomic_to_netcdf(ds, path, encoding=encoding)
    logger.info(
        "Built perennial-snow mask %s (%d cells, fingerprint %s)",
        path,
        int(perennial.sum()),
        fingerprint,
    )
    return path


def make_perennial_weight_filter(
    mask_path: Path,
) -> Callable[[pd.DataFrame, np.ndarray, np.ndarray], pd.DataFrame]:
    """Build a ``SourceAdapter.weight_filter`` dropping perennial-snow cells.

    Loads the mask once. The filter matches weight rows to mask cells by exact
    ``(lat, lon)`` value (both come from the same consolidated files) and
    raises ``ValueError`` if any row references a cell absent from the mask.
    """
    with xr.open_dataset(mask_path) as ds:
        perennial = ds["perennial"].transpose("lat", "lon").values.astype(bool)
        lat_index = pd.Index(ds["lat"].values)
        lon_index = pd.Index(ds["lon"].values)

    def _filter(
        weights: pd.DataFrame, cell_y: np.ndarray, cell_x: np.ndarray
    ) -> pd.DataFrame:
        iy = lat_index.get_indexer(cell_y)
        ix = lon_index.get_indexer(cell_x)
        missing = int(((iy < 0) | (ix < 0)).sum())
        if missing:
            raise ValueError(
                f"{missing} weight rows reference cells absent from the "
                f"perennial mask grid ({mask_path}); the mask and the "
                "consolidated Margulis files are on different grids."
            )
        return weights.loc[~perennial[iy, ix]]

    return _filter


def mask_stamp_attrs(mask_path: Path) -> dict[str, str | float]:
    """Return the four ``perennial_snow_mask_*`` rule attrs from the mask file."""
    with xr.open_dataset(mask_path) as ds:
        a = ds.attrs
        return {
            "perennial_snow_mask_fingerprint": str(
                a["perennial_snow_mask_fingerprint"]
            ),
            "perennial_snow_mask_swe_threshold_m": float(
                a["perennial_snow_mask_swe_threshold_m"]
            ),
            "perennial_snow_mask_check_month_day": str(
                a["perennial_snow_mask_check_month_day"]
            ),
            "perennial_snow_mask_min_year_fraction": float(
                a["perennial_snow_mask_min_year_fraction"]
            ),
        }


ADAPTER = SourceAdapter(
    source_key="margulis_wus_sr",
    output_cadence="daily",
    output_name="margulis_wus_sr_agg.nc",
    variables=("SWE",),
    source_crs="EPSG:4326",
    files_glob="daily/margulis_wus_sr_daily_*.nc",
)


def aggregate_margulis_wus_sr(
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
    period: str | None = None,
    *,
    worker_index: int = 0,
    n_workers: int = 1,
) -> None:
    """Aggregate Margulis WUS-SR daily SWE to HRU polygons; emit per-year NCs."""
    aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
        period=period,
        worker_index=worker_index,
        n_workers=n_workers,
    )
