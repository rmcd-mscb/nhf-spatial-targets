"""Fetch GLDAS-2.1 NOAH monthly runoff from NASA GES DISC via earthaccess."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import parse_period
from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata
from nhf_spatial_targets.workspace import load as _load_project

logger = logging.getLogger(__name__)

_SOURCE_KEY = "gldas_noah_v21_monthly"

# Lat/lon bbox matches the ERA5-Land bbox: N=53.0, W=-125.0, S=24.7, E=-66.0
BBOX_NWSE = [53.0, -125.0, 24.7, -66.0]


def derive_runoff_total(ds: xr.Dataset) -> xr.Dataset:
    """Add ``runoff_total = Qs_acc + Qsb_acc`` to the dataset.

    Both inputs are kg m-2 (mm equivalent); their sum has the same units.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing ``Qs_acc`` and ``Qsb_acc`` variables.

    Returns
    -------
    xr.Dataset
        Input dataset with ``runoff_total`` added.
    """
    total = ds["Qs_acc"] + ds["Qsb_acc"]
    total.attrs = {
        "long_name": "total runoff (Qs_acc + Qsb_acc, derived)",
        "units": "kg m-2",
        "cell_methods": "time: sum",
    }
    return ds.assign(runoff_total=total)


def clip_to_bbox(ds: xr.Dataset, bbox_nwse: list[float]) -> xr.Dataset:
    """Clip a global lat/lon dataset to a [N, W, S, E] bounding box.

    Supports both ``lat``/``lon`` and ``latitude``/``longitude`` coordinate
    names, and handles both ascending and descending latitude ordering.
    If the longitude coordinate is 0–360 (detected by max > 180), it is
    converted to -180–180 before slicing.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with geographic coordinates.
    bbox_nwse : list[float]
        Bounding box as ``[north, west, south, east]`` in decimal degrees.

    Returns
    -------
    xr.Dataset
        Subset of *ds* clipped to the bounding box.

    Raises
    ------
    ValueError
        If the dataset has neither ``lat``/``lon`` nor
        ``latitude``/``longitude`` coordinate names, or if the clipped
        result is empty (e.g. due to a longitude-convention mismatch).
    """
    n, w, s, e = bbox_nwse

    if "lat" in ds.coords and "lon" in ds.coords:
        lat_name, lon_name = "lat", "lon"
    elif "latitude" in ds.coords and "longitude" in ds.coords:
        lat_name, lon_name = "latitude", "longitude"
    else:
        raise ValueError(
            f"Dataset must have lat/lon or latitude/longitude coordinates; "
            f"got {list(ds.coords)}"
        )

    # Convert 0-360 longitude to -180-180 if necessary
    if float(ds[lon_name].max()) > 180:
        lon_vals = ds[lon_name].values.copy()
        lon_vals = (lon_vals + 180) % 360 - 180
        ds = ds.assign_coords({lon_name: lon_vals}).sortby(lon_name)

    lat = ds[lat_name]
    # sel with slice requires the slice direction to match the coord order
    if float(lat[0]) < float(lat[-1]):
        lat_slice = slice(s, n)
    else:
        lat_slice = slice(n, s)
    clipped = ds.sel({lat_name: lat_slice, lon_name: slice(w, e)})

    if clipped.sizes[lat_name] == 0 or clipped.sizes[lon_name] == 0:
        lat_range = (float(ds[lat_name].min()), float(ds[lat_name].max()))
        lon_range = (float(ds[lon_name].min()), float(ds[lon_name].max()))
        raise ValueError(
            f"Clipping produced an empty dataset. "
            f"Input {lat_name} range: {lat_range}, {lon_name} range: {lon_range}. "
            f"Requested bbox: N={n}, W={w}, S={s}, E={e}. "
            f"Check that the dataset and bbox use the same longitude convention "
            f"(-180/180 vs 0/360)."
        )

    return clipped


def fetch_gldas(workdir: Path, period: str) -> dict:
    """Download GLDAS-2.1 NOAH monthly granules and consolidate.

    Uses earthaccess (NASA EDL) to download monthly granules covering
    ``period``, then concatenates, derives ``runoff_total``, clips to the
    project bbox, applies CF metadata, and writes a single consolidated NC
    to the datastore.

    Parameters
    ----------
    workdir : Path
        Project directory. Reads ``fabric.json`` for the project bbox;
        writes consolidated output to the shared datastore.
    period : str
        Temporal range as ``"YYYY/YYYY"`` (start/end years inclusive).

    Returns
    -------
    dict
        Provenance record for ``manifest.json``.
    """
    import earthaccess

    ws = _load_project(workdir)
    meta = _catalog.source(_SOURCE_KEY)
    start_str, end_str = parse_period(period)

    raw_dir = ws.raw_dir(_SOURCE_KEY) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cf_path = ws.raw_dir(_SOURCE_KEY) / "gldas_noah_v21_monthly.nc"
    now_utc = datetime.now(timezone.utc).isoformat()

    earthaccess.login(strategy="netrc")
    results = earthaccess.search_data(
        short_name=meta["access"]["short_name"],
        version=meta["access"]["version"],
        temporal=(start_str, end_str),
    )
    logger.info("Found %d GLDAS granules for period %s", len(results), period)

    if not results:
        raise ValueError(
            f"No GLDAS granules found for short_name="
            f"{meta['access']['short_name']!r}, period={period!r}. "
            "Check network connectivity and Earthdata credentials."
        )

    downloaded = earthaccess.download(results, str(raw_dir))
    logger.info("Downloaded %d GLDAS granules to %s", len(downloaded), raw_dir)

    if not downloaded:
        raise RuntimeError(
            "earthaccess.download() returned no files. "
            "Check network connectivity and Earthdata credentials."
        )
    if len(downloaded) < len(results):
        raise RuntimeError(
            f"Partial GLDAS download: got {len(downloaded)}/{len(results)} granules. "
            f"Re-run to retry, or inspect earthaccess logs."
        )

    with xr.open_mfdataset(sorted(downloaded), combine="by_coords") as raw_ds:
        ds = raw_ds[["Qs_acc", "Qsb_acc"]].load()

    ds = derive_runoff_total(ds)
    ds = clip_to_bbox(ds, BBOX_NWSE)
    ds = apply_cf_metadata(ds, _SOURCE_KEY, "monthly")

    # Add source-level global attributes
    ds.attrs.update(
        {
            "title": "GLDAS-2.1 NOAH monthly runoff (CONUS+ buffered)",
            "institution": "NASA GES DISC",
            "source": "GLDAS_NOAH025_M v2.1",
            "references": "doi:10.1175/BAMS-85-3-381",
            "frequency": "month",
        }
    )

    tmp = cf_path.with_suffix(".nc.tmp")
    try:
        ds.to_netcdf(tmp, format="NETCDF4")
        tmp.rename(cf_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    bbox = ws.fabric["bbox_buffered"]
    file_info = {
        "path": str(cf_path),
        "size_bytes": cf_path.stat().st_size,
        "downloaded_utc": now_utc,
        "n_granules": len(downloaded),
    }
    _update_manifest(workdir, period, bbox, meta, file_info)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": meta["access"]["url"],
        "license": meta.get("license", "public domain (NASA)"),
        "variables": [v["name"] for v in meta["variables"]],
        "period": period,
        "bbox": bbox,
        "download_timestamp": now_utc,
        "file": file_info,
    }


def _update_manifest(
    workdir: Path,
    period: str,
    bbox: dict,
    meta: dict,
    file_info: dict,
) -> None:
    """Merge GLDAS-2.1 provenance into manifest.json."""
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {workdir} is corrupted and cannot be "
                f"parsed. Inspect the file manually or restore from backup. "
                f"Detail: {exc}"
            ) from exc
    else:
        manifest = {"sources": {}, "steps": []}

    if "sources" not in manifest:
        manifest["sources"] = {}

    entry = manifest["sources"].get(_SOURCE_KEY, {})
    entry.update(
        {
            "source_key": _SOURCE_KEY,
            "access_url": meta["access"]["url"],
            "license": meta.get("license", "public domain (NASA)"),
            "period": period,
            "bbox": bbox,
            "variables": [v["name"] for v in meta["variables"]],
            "file": file_info,
        }
    )
    manifest["sources"][_SOURCE_KEY] = entry

    fd, tmp = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp).replace(manifest_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    logger.info("Updated manifest.json with GLDAS-2.1 provenance")
