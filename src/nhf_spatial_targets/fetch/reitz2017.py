"""Fetch Reitz et al. (2017) recharge estimates from USGS ScienceBase."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import parse_period, years_in_period

logger = logging.getLogger(__name__)

_SOURCE_KEY = "reitz2017"
_CONSOLIDATED_FILENAME = "reitz2017_consolidated.nc"
_DATA_PERIOD = (2000, 2013)


def _year_from_filename(path: Path) -> int:
    """Extract year from filenames like TotalRecharge_2005.tif."""
    stem = path.stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot extract year from filename: {path.name}")
    try:
        return int(parts[1])
    except ValueError:
        raise ValueError(f"Cannot extract year from filename: {path.name}") from None


def _consolidate(output_dir: Path, period: str) -> Path:
    """Read annual GeoTIFFs and consolidate into a single NetCDF.

    Parameters
    ----------
    output_dir : Path
        Directory containing TotalRecharge_YYYY.tif and EffRecharge_YYYY.tif files.
    period : str
        Period as "YYYY/YYYY" — only files within this range are included.

    Returns
    -------
    Path
        Path to the consolidated NetCDF file.
    """
    years = years_in_period(period)
    nc_path = output_dir / _CONSOLIDATED_FILENAME
    tmp_path = nc_path.with_suffix(".nc.tmp")

    var_configs = [
        ("TotalRecharge", "total_recharge"),
        ("EffRecharge", "eff_recharge"),
    ]

    # Collect per-variable DataArrays keyed by year
    var_arrays: dict[str, dict[int, xr.DataArray]] = {
        ds_name: {} for _, ds_name in var_configs
    }

    for file_prefix, ds_name in var_configs:
        for tif_path in sorted(output_dir.glob(f"{file_prefix}_*.tif")):
            year = _year_from_filename(tif_path)
            if year not in years:
                continue
            da = rioxarray.open_rasterio(tif_path, masked=True)
            da = da.squeeze("band", drop=True)
            var_arrays[ds_name][year] = da

    # Validate pairing: every year must have both variables
    total_years = set(var_arrays["total_recharge"].keys())
    eff_years = set(var_arrays["eff_recharge"].keys())
    if total_years != eff_years:
        only_total = total_years - eff_years
        only_eff = eff_years - total_years
        raise RuntimeError(
            f"Mismatched years between TotalRecharge and EffRecharge. "
            f"Only in TotalRecharge: {sorted(only_total)}. "
            f"Only in EffRecharge: {sorted(only_eff)}."
        )

    if not total_years:
        raise RuntimeError(
            f"No GeoTIFF files found in {output_dir} for period {period}."
        )

    sorted_years = sorted(total_years)
    time_coords = pd.to_datetime([f"{y}-07-01" for y in sorted_years])

    # Stack into Dataset
    ds = xr.Dataset()
    for _, ds_name in var_configs:
        stacked = xr.concat(
            [var_arrays[ds_name][y] for y in sorted_years],
            dim="time",
        )
        stacked = stacked.assign_coords(time=time_coords)
        ds[ds_name] = stacked

    # Write atomically with compression
    encoding = {ds_name: {"zlib": True, "complevel": 4} for _, ds_name in var_configs}
    try:
        ds.to_netcdf(tmp_path, format="NETCDF4", encoding=encoding)
        tmp_path.rename(nc_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        ds.close()

    logger.info("Wrote consolidated file: %s", nc_path)
    return nc_path


def fetch_reitz2017(run_dir: Path, period: str) -> dict:
    """Download Reitz 2017 recharge estimates from USGS ScienceBase.

    Downloads annual zipped GeoTIFFs for total and effective recharge,
    extracts them, consolidates into a single NetCDF, and updates
    manifest.json. Skips download if consolidated file already exists.

    Parameters
    ----------
    run_dir : Path
        Run workspace directory.
    period : str
        Temporal range as "YYYY/YYYY".

    Returns
    -------
    dict
        Provenance record for manifest.json.
    """
    import zipfile

    meta = _catalog.source(_SOURCE_KEY)
    parse_period(period)  # validate format

    access = meta["access"]
    child_item_id = access["child_item_id"]
    file_patterns = access["file_patterns"]
    license_str = meta.get("license", "unknown")

    # Validate period is within data range
    requested_years = years_in_period(period)
    for y in requested_years:
        if y < _DATA_PERIOD[0] or y > _DATA_PERIOD[1]:
            raise ValueError(
                f"Year {y} is outside the Reitz 2017 data range "
                f"({_DATA_PERIOD[0]}–{_DATA_PERIOD[1]}). "
                f"Adjust the --period argument."
            )

    fabric_path = run_dir / "fabric.json"
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"fabric.json not found in {run_dir}. "
            f"Run 'nhf-targets init' to create a run workspace first."
        )

    output_dir = run_dir / "data" / "raw" / _SOURCE_KEY
    output_dir.mkdir(parents=True, exist_ok=True)
    nc_path = output_dir / _CONSOLIDATED_FILENAME
    now_utc = datetime.now(timezone.utc).isoformat()

    if nc_path.exists():
        logger.info("Consolidated file already exists, skipping: %s", nc_path)
    else:
        # Determine which years need downloading
        years_to_download = []
        for year in requested_years:
            total_tif = output_dir / f"TotalRecharge_{year}.tif"
            eff_tif = output_dir / f"EffRecharge_{year}.tif"
            if total_tif.exists() and eff_tif.exists():
                logger.info("Year %d already downloaded, skipping", year)
            else:
                years_to_download.append(year)

        if years_to_download:
            from sciencebasepy import SbSession

            logger.info(
                "Connecting to ScienceBase (child item %s)...",
                child_item_id,
            )
            try:
                sb = SbSession()
                item = sb.get_item(child_item_id)
                file_infos = sb.get_item_file_info(item)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to connect to ScienceBase item {child_item_id}: {exc}"
                ) from exc

            # Build lookup: filename -> file_info dict
            file_lookup = {fi["name"]: fi for fi in file_infos}

            for i, year in enumerate(years_to_download, 1):
                logger.info(
                    "Downloading year %d (%d/%d)...",
                    year,
                    i,
                    len(years_to_download),
                )
                for var_key, pattern in file_patterns.items():
                    zip_name = pattern.replace("{year}", str(year))
                    fi = file_lookup.get(zip_name)
                    if fi is None:
                        raise RuntimeError(
                            f"File '{zip_name}' not found in ScienceBase "
                            f"item {child_item_id}. Available files: "
                            f"{sorted(file_lookup.keys())}"
                        )

                    try:
                        sb.download_file(fi["url"], fi["name"], str(output_dir))
                    except Exception as exc:
                        raise RuntimeError(
                            f"ScienceBase download failed for '{zip_name}': {exc}"
                        ) from exc

                    zip_path = output_dir / zip_name
                    # Extract .tif from zip
                    try:
                        with zipfile.ZipFile(zip_path) as zf:
                            tif_names = [
                                n for n in zf.namelist() if n.lower().endswith(".tif")
                            ]
                            if not tif_names:
                                raise RuntimeError(f"No .tif file found in {zip_name}")
                            zf.extract(tif_names[0], output_dir)
                    finally:
                        zip_path.unlink(missing_ok=True)

        # Consolidate
        logger.info("Consolidating GeoTIFFs into NetCDF...")
        _consolidate(output_dir, period)

    # Build provenance
    file_info = {
        "path": str(nc_path.relative_to(run_dir)),
        "size_bytes": nc_path.stat().st_size,
        "downloaded_utc": now_utc,
        "years": requested_years,
    }

    _update_manifest(run_dir, period, meta, license_str, file_info)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": access["url"],
        "doi": access["doi"],
        "license": license_str,
        "variables": [v["name"] for v in meta["variables"]],
        "period": period,
        "spatial_extent": "CONUS",
        "download_timestamp": now_utc,
        "file": file_info,
    }


def _update_manifest(
    run_dir: Path,
    period: str,
    meta: dict,
    license_str: str,
    file_info: dict,
) -> None:
    """Merge Reitz 2017 provenance into manifest.json."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {run_dir} is corrupt and cannot be "
                f"parsed. You may need to delete it and re-run the fetch "
                f"step. Original error: {exc}"
            ) from exc
    else:
        manifest = {"sources": {}, "steps": []}

    if "sources" not in manifest:
        manifest["sources"] = {}

    access = meta["access"]
    entry = manifest["sources"].get(_SOURCE_KEY, {})
    entry.update(
        {
            "source_key": _SOURCE_KEY,
            "access_url": access["url"],
            "doi": access["doi"],
            "license": license_str,
            "period": period,
            "spatial_extent": "CONUS",
            "variables": [v["name"] for v in meta["variables"]],
            "file": file_info,
        }
    )
    manifest["sources"][_SOURCE_KEY] = entry

    tmp_fd, tmp_path = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp_path).replace(manifest_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    logger.info("Updated manifest.json with Reitz 2017 provenance")
