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
from nhf_spatial_targets.workspace import load as _load_workspace

logger = logging.getLogger(__name__)

_SOURCE_KEY = "reitz2017"
_CONSOLIDATED_FILENAME = "reitz2017_consolidated.nc"
_DATA_PERIOD = (2000, 2013)  # doi:10.5066/F7PN93P0


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
    src_crs_wkt: str | None = None

    for file_prefix, ds_name in var_configs:
        for tif_path in sorted(output_dir.glob(f"{file_prefix}_*.tif")):
            year = _year_from_filename(tif_path)
            if year not in years:
                continue
            try:
                da = rioxarray.open_rasterio(tif_path, masked=True)
                da = da.squeeze("band", drop=True).load()
                if src_crs_wkt is None and da.rio.crs is not None:
                    src_crs_wkt = da.rio.crs.to_wkt()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read GeoTIFF '{tif_path.name}' for variable "
                    f"'{ds_name}', year {year}. The file may be corrupt — "
                    f"delete it and re-run the fetch. Original error: {exc}"
                ) from exc
            finally:
                if "da" in locals():
                    da.close()
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
    # Annual data: use mid-year (July 1) as representative timestamp
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

    # Apply CF-1.6 metadata via shared helper (handles spatial_ref removal,
    # CRS variable, grid_mapping, coordinate renaming, variable attrs, etc.)
    from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata

    ds = apply_cf_metadata(ds, _SOURCE_KEY, "annual", crs_wkt=src_crs_wkt)

    # Write atomically with compression
    encoding = {ds_name: {"zlib": True, "complevel": 4} for _, ds_name in var_configs}
    try:
        ds.to_netcdf(tmp_path, format="NETCDF4", encoding=encoding)
        tmp_path.rename(nc_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        try:
            ds.close()
        except Exception:
            pass

    logger.info("Wrote consolidated file: %s", nc_path)
    return nc_path


def fetch_reitz2017(workdir: Path, period: str) -> dict:
    """Download Reitz 2017 recharge estimates from USGS ScienceBase.

    Downloads annual zipped GeoTIFFs for total and effective recharge,
    extracts them, consolidates into a single NetCDF, and updates
    manifest.json. Skips download if consolidated file already exists.

    Parameters
    ----------
    workdir : Path
        Workspace directory.
    period : str
        Temporal range as "YYYY/YYYY".

    Returns
    -------
    dict
        Provenance record for manifest.json.
    """
    import zipfile

    ws = _load_workspace(workdir)
    meta = _catalog.source(_SOURCE_KEY)
    parse_period(period)  # validate format

    access = meta.get("access", {})
    for key in ("child_item_id", "file_patterns"):
        if key not in access:
            raise ValueError(
                f"Catalog entry for '{_SOURCE_KEY}' is missing "
                f"'access.{key}'. Check catalog/sources.yml."
            )
    child_item_id = access["child_item_id"]
    file_patterns = access["file_patterns"]
    license_str = meta.get("license", "unknown")

    # Validate period is within data range
    requested_years = years_in_period(period)
    for y in requested_years:
        if y < _DATA_PERIOD[0] or y > _DATA_PERIOD[1]:
            raise ValueError(
                f"Year {y} is outside the Reitz 2017 data range "
                f"({_DATA_PERIOD[0]}-{_DATA_PERIOD[1]}). "
                f"Adjust the --period argument."
            )

    logger.info(
        "Reitz 2017 data covers all of CONUS; spatial subsetting "
        "is applied during the aggregation step."
    )

    output_dir = ws.raw_dir(_SOURCE_KEY)
    output_dir.mkdir(parents=True, exist_ok=True)
    nc_path = output_dir / _CONSOLIDATED_FILENAME
    now_utc = datetime.now(timezone.utc).isoformat()

    if nc_path.exists():
        # Verify existing file covers the requested period
        try:
            with xr.open_dataset(nc_path) as ds_check:
                existing_years = set(int(y) for y in ds_check.time.dt.year.values)
        except Exception as exc:
            raise RuntimeError(
                f"Consolidated file {nc_path} exists but cannot be read. "
                f"It may be corrupt. Delete it and re-run the fetch. "
                f"Original error: {exc}"
            ) from exc
        missing = set(requested_years) - existing_years
        if missing:
            raise RuntimeError(
                f"Consolidated file {nc_path} exists but is missing years "
                f"{sorted(missing)}. Delete it and re-run to rebuild with "
                f"the requested period {period}."
            )
        logger.info("Consolidated file already exists, skipping: %s", nc_path)
    else:
        # Determine which years need downloading
        years_to_download = []
        for year in requested_years:
            total_tif = output_dir / f"TotalRecharge_{year}.tif"
            eff_tif = output_dir / f"EffRecharge_{year}.tif"
            both_exist = total_tif.exists() and eff_tif.exists()
            if (
                both_exist
                and total_tif.stat().st_size > 0
                and eff_tif.stat().st_size > 0
            ):
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
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to connect to ScienceBase item {child_item_id}: {exc}"
                ) from exc

            if not item or "id" not in item:
                raise RuntimeError(
                    f"ScienceBase item {child_item_id} returned an invalid "
                    f"response. The item may have been deleted or moved."
                )

            try:
                file_infos = sb.get_item_file_info(item)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to retrieve file list from ScienceBase item "
                    f"{child_item_id}: {exc}"
                ) from exc
            if not file_infos:
                raise RuntimeError(
                    f"ScienceBase item {child_item_id} has no downloadable "
                    f"files. Check the item at {access['url']}."
                )

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
                    if not zip_path.exists() or zip_path.stat().st_size == 0:
                        zip_path.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"Download of '{zip_name}' produced no file or a "
                            f"zero-byte file. ScienceBase may be experiencing "
                            f"issues. Try again later."
                        )

                    # Extract .tif from zip
                    try:
                        with zipfile.ZipFile(zip_path) as zf:
                            tif_names = [
                                n for n in zf.namelist() if n.lower().endswith(".tif")
                            ]
                            if not tif_names:
                                raise RuntimeError(f"No .tif file found in {zip_name}")
                            if len(tif_names) > 1:
                                raise RuntimeError(
                                    f"Zip '{zip_name}' contains {len(tif_names)} "
                                    f".tif files: {tif_names}. Expected exactly "
                                    f"one. The zip structure may have changed."
                                )
                            # Extract and rename to match the expected
                            # convention (e.g. TotalRecharge_2005.tif).
                            # The actual names inside the zips differ
                            # (e.g. RC_2005.tif, RC_eff_2005.tif).
                            tif_member = tif_names[0]
                            expected_name = Path(zip_name).with_suffix(".tif").name
                            (output_dir / expected_name).write_bytes(
                                zf.read(tif_member)
                            )
                    except zipfile.BadZipFile as exc:
                        raise RuntimeError(
                            f"Corrupt zip file '{zip_name}' for year {year}. "
                            f"The download may have been truncated. "
                            f"Delete it and retry. Original error: {exc}"
                        ) from exc
                    finally:
                        zip_path.unlink(missing_ok=True)

        # Consolidate
        logger.info("Consolidating GeoTIFFs into NetCDF...")
        _consolidate(output_dir, period)

    # Build provenance
    file_info = {
        "path": str(nc_path),
        "size_bytes": nc_path.stat().st_size,
        "downloaded_utc": now_utc,
        "years": requested_years,
    }

    _update_manifest(workdir, period, meta, license_str, file_info)

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
    workdir: Path,
    period: str,
    meta: dict,
    license_str: str,
    file_info: dict,
) -> None:
    """Merge Reitz 2017 provenance into manifest.json."""
    ws = _load_workspace(workdir)
    manifest_path = ws.manifest_path
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.json in {workdir} is corrupt and cannot be "
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
