"""Margulis Western US Snow Reanalysis daily SWE aggregator.

Reads per-year consolidated NCs at
``<datastore>/margulis_wus_sr/daily/margulis_wus_sr_daily_<year>.nc``
(written by ``fetch.margulis_wus_sr`` / its consolidator) and emits
area-weighted HRU means via the shared aggregation driver.

The consolidated NCs store ``SWE`` as float32 in **metres water
equivalent** on a regular WGS84 lat/lon grid (the consolidator
reprojects from the native ~90 m EASE-Grid). Per-pixel NaN over
unmodelled cells decodes naturally via ``mask_and_scale=True``.

Perennial snow / glacier mask (#363)
------------------------------------
PRMS cannot carry snow across water years, so glacier and perennial-snow
pixels are removed before HRU aggregation. ``ensure_perennial_snow_mask``
builds a per-pixel flag in the datastore
(``margulis_wus_sr/derived/perennial_snow_mask.nc``): a pixel is perennial
when its SWE is at or above a threshold on the water-year-end check date in
more than a given fraction of the years on disk (rule in the catalog,
``margulis_wus_sr.perennial_snow_mask``). The mask is rebuilt when the rule
or the set of years changes, and its fingerprint is stamped on every
per-year aggregated NC so pre-mask or stale years are re-aggregated. The
fingerprint covers the rule and the year set, not file contents: after
re-consolidating a year with corrected data the operator must delete the mask,
the aggregated dir and the SWE intermediates by hand (see
``docs/sources/margulis_wus_sr.md``).

The mask is applied as ``SourceAdapter.weight_filter``: perennial cells are
dropped from the gdptools weight table, so the area-weighted ``mean`` is
taken over the surviving (seasonal) cells only. An HRU with no perennial
cell is bit-identical to the unmasked run. An HRU whose cells are all
perennial has no weights left and is NaN, never 0.0 (gdptools yields
NC_FILL_DOUBLE, masked to NaN at the aggregation write boundary).

``stat_method`` stays ``"mean"`` (NaN propagates) and ``masked_mean`` is
deliberately avoided: gdptools returns 0.0, not NaN, for an HRU whose
pixels are all NaN under ``masked_mean`` (#364), which would turn a fully
perennial HRU into a false zero-SWE value. Dropping weight rows gives the
survivors-only mean without that failure mode.

Without the mask, HRUs that straddle the WUS domain edge or contain an
unmodelled (NaN) pixel become honest NaN under ``mean``; the SWE target
builder handles that at multi-source combination time.

The source covers only the Western US. The aggregation driver's
geometry-driven coverage guard (#309) skips fabric batches outside the
grid and emits honest NaN rows for them, so on a CONUS fabric the
aggregated NC carries data only at WUS HRUs; the SWE target's NaN-aware
combine uses it wherever it is finite.
"""

from __future__ import annotations

import dataclasses
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
from nhf_spatial_targets.workspace import load as load_project

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
        if not (np.isfinite(thr) and thr > 0):
            raise ValueError(f"swe_threshold_m must be finite and > 0, got {thr}")
        if not (np.isfinite(frac) and 0.0 <= frac < 1.0):
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
    # Spec form: held / valid > frac, with valid == 0 -> ratio 0 (never perennial).
    ratio = np.divide(h, v, out=np.zeros(np.broadcast(h, v).shape), where=v > 0)
    return (ratio > min_year_fraction).astype(np.int8)


def _year_files(datastore: Path) -> list[tuple[int, Path]]:
    daily = datastore / _SOURCE_KEY / "daily"
    out = []
    for f in sorted(daily.glob("margulis_wus_sr_daily_*.nc")):
        out.append((int(f.stem.rsplit("_", 1)[1]), f))
    return out


def _has_check_date(path: Path, year: int, month_day: str) -> bool:
    """True if the file's ``time`` index contains the check date (no SWE read)."""
    with xr.open_dataset(path) as ds:
        times = ds.indexes["time"]
        day = pd.Timestamp(f"{year}-{month_day}")
        return bool(((times >= day) & (times < day + pd.Timedelta(days=1))).any())


def _load_check_slice(path: Path, year: int, month_day: str) -> xr.DataArray:
    """Load the check-date SWE slice (time dim dropped)."""
    with xr.open_dataset(path) as ds:
        da = ds["SWE"].sel(time=slice(f"{year}-{month_day}", f"{year}-{month_day}"))
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

    files: dict[int, Path] = {}
    for year, f in _year_files(datastore):
        if not _has_check_date(f, year, rule.check_month_day):
            logger.info(
                "Margulis %s has no %s-%s timestep; left out of the "
                "perennial-snow mask",
                f.name,
                year,
                rule.check_month_day,
            )
            continue
        files[year] = f
    if not files:
        raise FileNotFoundError(
            f"No Margulis daily files with a {rule.check_month_day} timestep "
            f"under {datastore / _SOURCE_KEY / 'daily'}; run "
            f"`nhf-targets fetch {_SOURCE_KEY}` first."
        )

    years = sorted(files)
    fingerprint = mask_fingerprint(rule, years)
    if path.exists() and _read_fingerprint(path) == fingerprint:
        return path

    # Rebuild: stream one year's slice at a time into the accumulators.
    lat = lon = held = valid = None
    for year in years:
        da = _load_check_slice(files[year], year, rule.check_month_day)
        if lat is None:
            lat, lon = da["lat"].values, da["lon"].values
            held = np.zeros((lat.size, lon.size), dtype="int16")
            valid = np.zeros_like(held)
        else:
            for name, ref in (("lat", lat), ("lon", lon)):
                if da[name].shape != ref.shape or not np.array_equal(
                    da[name].values, ref
                ):
                    raise ValueError(
                        f"Margulis {year} {name} coordinates differ from "
                        f"{years[0]}; the perennial-snow mask needs one common grid."
                    )
        arr = da.transpose("lat", "lon").values
        finite = np.isfinite(arr)
        valid += finite
        # NaN >= thr is False (and warns); mask first.
        held += finite & (np.where(finite, arr, 0.0) >= rule.swe_threshold_m)
        del da, arr, finite
    perennial = classify_perennial(held, valid, rule.min_year_fraction)

    newest = max(files.values(), key=lambda p: p.stat().st_mtime)
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
    """Aggregate Margulis WUS-SR daily SWE to HRU polygons; emit per-year NCs.

    Builds (or reuses) the perennial-snow mask in the datastore, then
    aggregates with perennial cells dropped from the weights (#363).
    """
    datastore = load_project(Path(workdir)).datastore
    mask_path = ensure_perennial_snow_mask(datastore)
    adapter = dataclasses.replace(
        ADAPTER,
        weight_filter=make_perennial_weight_filter(mask_path),
        stamp_attrs=mask_stamp_attrs(mask_path),
    )
    aggregate_source(
        adapter,
        fabric_path,
        id_col,
        workdir,
        batch_size,
        period=period,
        worker_index=worker_index,
        n_workers=n_workers,
    )
