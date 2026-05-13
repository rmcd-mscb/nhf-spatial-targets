"""Fetch Margulis Western US Snow Reanalysis (NSIDC-0719) via earthaccess.

NSIDC-0719 is a daily 90 m posterior SWE reanalysis over the Western US
for water years 1985-2021. This source is **fabric-scoped to Oregon
only** in this pipeline (see ``catalog/sources.yml ->
margulis_wus_sr.fabric_scope``). The fetch step does not enforce that
scope — raw downloads stay in the shared datastore and remain usable
by any project pointing at the same store — but the manifest entry
records the scope so downstream aggregate/target code can honour it.

The module has two stages:

1. **Download** (``fetch_margulis_wus_sr``) — search NSIDC via
   earthaccess and write per-water-year per-tile NetCDF granules to
   ``<datastore>/margulis_wus_sr/raw/<year>/``.
2. **Consolidate** (``consolidate_calendar_year_margulis_wus_sr``) —
   stitch the per-water-year per-tile granules into one
   CF-1.6-compliant per-calendar-year NetCDF at
   ``<datastore>/margulis_wus_sr/daily/margulis_wus_sr_daily_<year>.nc``
   covering the full Western US source domain. The consolidator runs
   synchronously after each year's downloads complete.

The calendar-year rebuild assembles year *X* from the Jan–Sep portion
of WY *X* (the water year ending Sep 30 of *X*) joined with the
Oct–Dec portion of WY *X+1* (the water year starting Oct 1 of *X*).
The existing fetch layout places **both** water years adjacent to a
calendar year in the same ``raw/<X>/`` directory, so the consolidator
only needs to look at one raw directory per output file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xarray as xr

try:
    import fcntl as _fcntl

    _HAVE_FLOCK = True
except ImportError:  # Windows fallback.
    _HAVE_FLOCK = False

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import (
    parse_period,
    period_bounds,
    years_in_period,
)
from nhf_spatial_targets.fetch.consolidate import (
    apply_cf_metadata,
    _write_netcdf,
)
from nhf_spatial_targets.workspace import load as _load_project

logger = logging.getLogger(__name__)

_SOURCE_KEY = "margulis_wus_sr"
_CONSOLIDATED_FILENAME_TEMPLATE = "margulis_wus_sr_daily_{year}.nc"
_SWE_GRANULE_GLOB = "*_WY*_SWE_SCA_POST.nc"
# Index of the posterior mean along the granule's `Stats` dim. NSIDC-0719
# v01 documents the Stats axis as [mean, std, median, p25, p75] (1-indexed).
# `_select_posterior_mean` asserts Stats has the expected size before
# selecting so a future re-ordering surfaces as a loud failure rather
# than silently consuming the wrong layer.
_STATS_MEAN_IDX = 0
_STATS_EXPECTED_SIZE = 5
# Match WY tag inside a granule filename: e.g. "WY1999_00" -> prev_year=1999,
# next_year=2000. The two-digit suffix is the *century-rolled* end year of
# the water year (Oct prev_year → Sep next_year).
_WY_TAG_RE = re.compile(r"WY(?P<prev>\d{4})_(?P<curr>\d{2})_SWE_SCA_POST\.nc$")


def _wy_for_filename(name: str) -> tuple[int, int] | None:
    """Return ``(wy_prev_year, wy_next_year)`` parsed from a SWE granule name.

    NSIDC-0719 granules carry a tag like ``WY1999_00`` (= water year
    2000 = Oct 1999 – Sep 2000). The two-digit ``_00`` is the
    century-rolled end year. Returns ``None`` for filenames that do not
    match the SWE granule pattern (e.g. SD_POST granules, which carry
    snow depth and are ignored by the SWE consolidator).
    """
    m = _WY_TAG_RE.search(name)
    if not m:
        return None
    prev = int(m.group("prev"))
    curr_2d = int(m.group("curr"))
    # Roll the 2-digit suffix to the same century as the previous year.
    # WY1999_00 -> 1999/2000; WY2020_21 -> 2020/2021.
    century = (prev // 100) * 100
    candidate = century + curr_2d
    if candidate <= prev:
        candidate += 100
    return prev, candidate


def _group_swe_granules_by_wy(raw_dir: Path) -> dict[int, list[Path]]:
    """Return ``{wy_next_year: [tile granule path, ...]}`` for one raw dir.

    Filters to the ``*SWE_SCA_POST.nc`` granules (the SWE+SCA product);
    ignores ``*SD_POST.nc`` granules (snow depth) since SWE is the only
    variable declared for this source in the catalog.
    """
    grouped: dict[int, list[Path]] = {}
    for p in sorted(raw_dir.glob(_SWE_GRANULE_GLOB)):
        parsed = _wy_for_filename(p.name)
        if parsed is None:
            continue
        _prev, wy_next = parsed
        grouped.setdefault(wy_next, []).append(p)
    return grouped


def _select_posterior_mean(ds: xr.Dataset) -> xr.Dataset:
    """Drop the ``Stats`` dim by selecting the posterior mean layer.

    Asserts the ``Stats`` axis has its expected size — a re-ordering or
    addition of stats layers in a future NSIDC release should surface as
    a loud failure rather than silently selecting the wrong slice.
    """
    if "Stats" not in ds.dims:
        return ds
    if ds.sizes["Stats"] != _STATS_EXPECTED_SIZE:
        raise ValueError(
            f"margulis_wus_sr: granule has Stats dim of size "
            f"{ds.sizes['Stats']}, expected {_STATS_EXPECTED_SIZE} "
            f"(mean, std, median, p25, p75). Source layout may have "
            f"changed; verify before consolidating."
        )
    return ds.isel(Stats=_STATS_MEAN_IDX, drop=True)


def _load_swe_granule_with_time(path: Path, wy_prev_year: int) -> xr.Dataset:
    """Open one tile-WY granule and produce a ``(time, lat, lon)`` SWE dataset.

    Renames variables and coordinates to the catalog-declared / CF
    canonical names: ``SWE_Post`` → ``SWE``, ``Latitude`` → ``lat``,
    ``Longitude`` → ``lon``, ``Day`` → ``time``. Discards the
    co-bundled ``SCA_Post`` variable. Builds a daily time coordinate
    starting Oct 1 of ``wy_prev_year``.
    """
    ds = xr.open_dataset(path, chunks={"Day": 30})
    if "SWE_Post" not in ds.data_vars:
        raise ValueError(
            f"margulis_wus_sr: granule {path.name} has no `SWE_Post` "
            f"variable (got {list(ds.data_vars)}). Source schema may "
            f"have changed; re-run characterisation notebook to verify."
        )
    ds = ds[["SWE_Post"]]
    ds = _select_posterior_mean(ds)
    n_days = int(ds.sizes["Day"])
    times = pd.date_range(
        pd.Timestamp(f"{wy_prev_year}-10-01"), periods=n_days, freq="D"
    )
    ds = ds.assign_coords(Day=times).rename(
        {"Day": "time", "SWE_Post": "SWE", "Latitude": "lat", "Longitude": "lon"}
    )
    return ds


def _calendar_year_slice(ds: xr.Dataset, calendar_year: int) -> xr.Dataset:
    """Slice a WY dataset down to whichever portion overlaps the calendar year."""
    cy_start = pd.Timestamp(f"{calendar_year}-01-01")
    cy_end = pd.Timestamp(f"{calendar_year}-12-31")
    return ds.sel(time=slice(cy_start, cy_end))


def _mosaic_tiles(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Mosaic per-tile datasets along ``lat``/``lon`` into one WUS-wide dataset.

    NSIDC-0719 tiles abut at integer-degree boundaries with the boundary
    coordinate present in *both* neighbours (e.g. tile N31 has lat[0]=32.0
    and tile N32 has lat[-1]=32.0). ``combine_by_coords`` with
    ``compat='override'`` is the right call: it merges by coordinate
    value and keeps the first occurrence on overlap. Tiles share a
    common regular grid (1° / 224 ≈ 0.00446°), so no resampling is
    needed.
    """
    return xr.combine_by_coords(
        datasets,
        combine_attrs="drop_conflicts",
        compat="override",
        data_vars="minimal",
        coords="minimal",
        join="outer",
    )


def consolidate_calendar_year_margulis_wus_sr(
    calendar_year: int, raw_root: Path, daily_dir: Path
) -> Path:
    """Mosaic raw NSIDC-0719 tile granules into one CF NetCDF for ``calendar_year``.

    Mirrors the SNODAS consolidation pattern (PR #109): per-year synchronous
    consolidation immediately after the download loop, with mtime-based
    idempotency, atomic temp-file + rename, and CF-1.6 compliance via
    :func:`nhf_spatial_targets.fetch.consolidate.apply_cf_metadata`.

    Calendar year *X* is assembled from:

    - Days 93..N of WY *X* (Jan 1 – Sep 30, where WY *X* = Oct *X-1* – Sep *X*)
    - Days 1..92 of WY *X+1* (Oct 1 – Dec 31)

    Both source water years are typically already present in
    ``raw_root/<calendar_year>/`` because the fetch step downloads the
    two adjacent water years into the same calendar-year directory.

    Parameters
    ----------
    calendar_year : int
        Target calendar year (Jan 1 – Dec 31). Must have **both** WY
        granule sets available in ``raw_root``; the final source year
        (e.g. 2021 with the v01 dataset that ends at WY2021) lacks the
        WY *X+1* granules and will raise ``FileNotFoundError``.
    raw_root : Path
        ``<datastore>/margulis_wus_sr/raw/<calendar_year>/`` —
        directory containing the per-water-year per-tile NSIDC-0719
        granule NetCDFs.
    daily_dir : Path
        Output directory (typically
        ``<datastore>/margulis_wus_sr/daily/``). Created if missing.

    Returns
    -------
    Path
        ``daily_dir / "margulis_wus_sr_daily_<calendar_year>.nc"``.

    Raises
    ------
    FileNotFoundError
        ``raw_root`` is missing, or one of the two required water years
        is absent (typical at source-domain boundaries — e.g. the final
        catalogued year lacks the *next* water year).
    ValueError
        A granule's schema (variable name, Stats dim size) does not
        match the expected NSIDC-0719 v01 layout.
    """
    raw_root = Path(raw_root)
    daily_dir = Path(daily_dir)
    daily_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise FileNotFoundError(
            f"margulis_wus_sr: raw directory {raw_root} does not exist; "
            f"run `nhf-targets fetch margulis-wus-sr` for calendar year "
            f"{calendar_year} first."
        )

    grouped = _group_swe_granules_by_wy(raw_root)
    wy_a = calendar_year  # WY ending Sep of CY; contributes Jan–Sep
    wy_b = calendar_year + 1  # WY starting Oct of CY; contributes Oct–Dec
    missing = [str(w) for w in (wy_a, wy_b) if w not in grouped]
    if missing:
        raise FileNotFoundError(
            f"margulis_wus_sr: calendar year {calendar_year} consolidation "
            f"needs water years {wy_a} and {wy_b} in {raw_root}, but the "
            f"following WY granules are absent: {missing}. This is expected "
            f"at the source-domain boundary (NSIDC-0719 v01 ends at WY 2021, "
            f"so CY 2021 cannot be completed); otherwise run "
            f"`nhf-targets fetch margulis-wus-sr --period "
            f"{calendar_year}/{calendar_year + 1}` to fill the gap."
        )

    out_path = daily_dir / _CONSOLIDATED_FILENAME_TEMPLATE.format(year=calendar_year)
    all_inputs = grouped[wy_a] + grouped[wy_b]

    if out_path.exists():
        out_mtime = out_path.stat().st_mtime
        newest_input_mtime = max(p.stat().st_mtime for p in all_inputs)
        if newest_input_mtime <= out_mtime:
            logger.info(
                "margulis_wus_sr: daily NC up-to-date for %d (%d tiles older "
                "than NC); skipping: %s",
                calendar_year,
                len(all_inputs),
                out_path,
            )
            return out_path
        logger.info(
            "margulis_wus_sr: raw tiles newer than daily NC for %d; "
            "re-consolidating: %s",
            calendar_year,
            out_path,
        )

    logger.info(
        "margulis_wus_sr: consolidating CY %d from %d WY-%d tiles + %d "
        "WY-%d tiles -> %s",
        calendar_year,
        len(grouped[wy_a]),
        wy_a,
        len(grouped[wy_b]),
        wy_b,
        out_path,
    )

    # Load each tile-WY, time-stamp it, then slice to the CY window.
    # Slicing per-tile keeps the in-memory footprint bounded by one CY
    # of one tile (~74 MB float32 at agg_16) rather than two full WYs.
    cy_tiles: list[xr.Dataset] = []
    for wy_next, wy_prev in ((wy_a, wy_a - 1), (wy_b, wy_b - 1)):
        for tile_path in grouped[wy_next]:
            ds_tile = _load_swe_granule_with_time(tile_path, wy_prev)
            ds_tile_cy = _calendar_year_slice(ds_tile, calendar_year)
            if ds_tile_cy.sizes["time"] == 0:
                # Defensive: shouldn't happen with correct WY pairing, but
                # surface it rather than silently dropping the tile.
                raise ValueError(
                    f"margulis_wus_sr: tile {tile_path.name} contributed "
                    f"0 days to CY {calendar_year} (WY {wy_next}); "
                    f"check WY tagging logic in _wy_for_filename."
                )
            cy_tiles.append(ds_tile_cy)

    # Mosaic all (per-tile, per-WY-half) datasets into one WUS-wide
    # (time, lat, lon) dataset. ``combine_by_coords`` already returns
    # coords sorted ascending — no `sortby` needed (and explicit sorts
    # over a 732-tile dask graph trigger O(n) chunk-rebalancing
    # blockwise ops, with `dask` emitting PerformanceWarnings about
    # 14x chunk-count growth). Flip latitude to descending via a cheap
    # `isel` view; the time axis already lands in chronological order
    # because we appended WY *X* (Jan–Sep) before WY *X+1* (Oct–Dec)
    # and combine_by_coords preserves that.
    merged = _mosaic_tiles(cy_tiles)
    if float(merged.lat.values[0]) < float(merged.lat.values[-1]):
        merged = merged.isel(lat=slice(None, None, -1))

    # Sanity-check the assembled time axis: every day of the calendar
    # year must be present, no duplicates.
    expected_n_days = (
        366 if pd.Timestamp(f"{calendar_year}-12-31").is_leap_year else 365
    )
    actual_n_days = int(merged.sizes["time"])
    if actual_n_days != expected_n_days:
        raise ValueError(
            f"margulis_wus_sr: assembled CY {calendar_year} has "
            f"{actual_n_days} daily steps, expected {expected_n_days}. "
            f"Some tiles may carry a non-standard `Day` axis length; "
            f"re-run the inspect_margulis_wus_sr notebook."
        )

    # CF-1.6 metadata via the shared helper, then add Margulis-specific
    # provenance global attrs.
    merged = apply_cf_metadata(merged, _SOURCE_KEY, "daily")
    meta = _catalog.source(_SOURCE_KEY)
    now_utc = datetime.now(timezone.utc).isoformat()
    merged.attrs.update(
        {
            "title": meta.get("name", _SOURCE_KEY),
            "source": meta.get("name", _SOURCE_KEY),
            "institution": "UCLA (data) / USGS NHGF (pipeline)",
            "references": f"doi:{meta.get('doi', '')}".rstrip(":"),
            "history": (
                f"{now_utc}: consolidated by nhf_spatial_targets "
                f"consolidate_calendar_year_margulis_wus_sr (issue #111)"
            ),
            "margulis_source_water_years": json.dumps([wy_a, wy_b]),
        }
    )

    # Encoding: int-friendly storage is overkill for posterior mean SWE
    # (already small floats); use zlib level 4 on the native float dtype
    # with per-timestep chunking so downstream consumers can stream
    # single-day reads. The shared `_write_netcdf` handles atomic
    # temp-file + rename.
    #
    # Rechunk dask blocks to match the NetCDF encoding chunks BEFORE
    # write. With ~732 lazily-opened tile granules feeding into
    # ``combine_by_coords``, the resulting dask array carries thousands
    # of small (per-tile, per-WY-time-slice) blocks. Writing that to
    # NetCDF encoding ``(1, nlat, nlon)`` without an intermediate
    # rechunk drives dask to do a blockwise rebalance over the whole
    # graph (the 14x-chunk-growth PerformanceWarning). Rechunking to
    # ``(time=1, lat=-1, lon=-1)`` produces one dask block per
    # timestep, aligned 1:1 with the output NetCDF chunks.
    nlat = int(merged.sizes["lat"])
    nlon = int(merged.sizes["lon"])
    merged = merged.chunk({"time": 1, "lat": -1, "lon": -1})
    encoding = {
        "SWE": {
            "zlib": True,
            "complevel": 4,
            "chunksizes": (1, nlat, nlon),
        }
    }
    _write_netcdf(merged, out_path, encoding=encoding)
    logger.info(
        "margulis_wus_sr: wrote consolidated CY %d -> %s", calendar_year, out_path
    )
    return out_path


def _attempt_consolidation_into_record(
    rec: dict, raw_root: Path, daily_dir: Path
) -> None:
    """Run consolidation for ``rec['year']``, mutating ``rec`` with the result.

    On success, sets ``daily_path``, ``consolidated_utc``,
    ``source_water_years``. On failure (boundary year, schema drift,
    I/O), sets ``consolidate_error`` and leaves any prior values
    in place. Never raises — the fetch wrapper records errors and
    keeps going.
    """
    rec_year = int(rec["year"])
    year_dir = raw_root / f"{rec_year}"
    try:
        out_path = consolidate_calendar_year_margulis_wus_sr(
            rec_year, year_dir, daily_dir
        )
    except FileNotFoundError as exc:
        logger.warning(
            "margulis_wus_sr: consolidation skipped for CY %d: %s", rec_year, exc
        )
        rec["consolidate_error"] = str(exc)
        return
    except Exception as exc:  # noqa: BLE001 — record + continue
        logger.exception("margulis_wus_sr: consolidation failed for CY %d", rec_year)
        rec["consolidate_error"] = repr(exc)
        return
    rec["daily_path"] = str(out_path)
    rec["consolidated_utc"] = datetime.now(timezone.utc).isoformat()
    rec["source_water_years"] = [rec_year, rec_year + 1]
    rec.pop("consolidate_error", None)


def fetch_margulis_wus_sr(workdir: Path, period: str) -> dict:
    """Download Margulis WUS-SR daily granules and consolidate into per-CY NCs.

    Two-stage path:

    1. Search NSIDC for granules covering ``period`` within the project's
       fabric bbox; download to ``<datastore>/margulis_wus_sr/raw/<year>/``.
    2. For each calendar year with **both** adjacent water years on disk,
       call :func:`consolidate_calendar_year_margulis_wus_sr` to produce a
       single CF-1.6 NetCDF at
       ``<datastore>/margulis_wus_sr/daily/margulis_wus_sr_daily_<year>.nc``.
       A consolidation failure (e.g. boundary year missing its WY *X+1*) is
       logged + recorded in the manifest as ``consolidate_error`` but does
       not abort the rest of the fetch.

    The flock-protected manifest entry records the fabric scope from the
    catalog plus per-calendar-year ``daily_path`` / ``consolidated_utc`` /
    ``source_water_years`` for completed consolidations.

    The source is fabric-scoped to Oregon via the catalog's
    ``fabric_scope`` block; calling on a non-Oregon project still runs
    (raw downloads are reusable across projects sharing a datastore)
    but emits a warning that the search bbox is unlikely to overlap
    NSIDC-0719's Western US domain.

    Parameters
    ----------
    workdir : Path
        Project directory. The project's fabric bbox is used as the
        ``bounding_box`` constraint on the CMR search.
    period : str
        Temporal window ``"YYYY/YYYY"``. Clamped to the publisher window
        recorded in ``catalog/sources.yml`` — out-of-range years raise
        ValueError.

    Returns
    -------
    dict
        Provenance summary.

    Raises
    ------
    ValueError
        Period falls outside the publisher window, or the project
        fabric lacks a buffered bbox.
    RuntimeError
        ``earthaccess.download`` returned fewer files than the CMR
        result count.
    """
    import earthaccess

    parse_period(period)
    ws = _load_project(workdir)
    meta = _catalog.source(_SOURCE_KEY)
    access = meta["access"]
    fabric_scope = meta.get("fabric_scope", {})
    data_lo, data_hi = period_bounds(meta["period"])
    years = years_in_period(period)
    for y in years:
        if y < data_lo or y > data_hi:
            raise ValueError(
                f"Year {y} is outside the Margulis WUS-SR publisher window "
                f"({data_lo}-{data_hi}, from catalog `sources.yml[{_SOURCE_KEY}]"
                f".period`). Adjust --period."
            )

    # Operator hint: warn (don't fail) if the project's fabric isn't in
    # the source's `fabric_scope.fabrics` list. Raw downloads are still
    # useful (reusable by any project pointing at the same datastore),
    # but a non-Oregon CMR search will return 0 granules on every year
    # and the operator deserves a hint about why.
    scope_fabrics = list(fabric_scope.get("fabrics") or [])
    fabric_id = (ws.config.get("fabric") or {}).get("id") or (
        (ws.config.get("fabric") or {}).get("name")
    )
    if scope_fabrics and fabric_id and fabric_id not in scope_fabrics:
        logger.warning(
            "margulis_wus_sr is scoped to fabrics %s in the catalog; this "
            "project's fabric is %r, which is outside that scope. "
            "Fetching anyway (raw downloads are datastore-shared), but "
            "expect zero granules per year. See docs/sources/"
            "margulis_wus_sr.md.",
            scope_fabrics,
            fabric_id,
        )

    raw_root = ws.raw_dir(_SOURCE_KEY) / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    bbox_buffered = ws.fabric.get("bbox_buffered") or {}
    if not bbox_buffered:
        raise ValueError(
            "Margulis WUS-SR fetch needs a buffered fabric bbox; "
            "fabric.json has no 'bbox_buffered' key. Re-run "
            "`nhf-targets validate` to regenerate fabric.json."
        )
    search_bbox = (
        float(bbox_buffered["minx"]),
        float(bbox_buffered["miny"]),
        float(bbox_buffered["maxx"]),
        float(bbox_buffered["maxy"]),
    )

    now_utc = datetime.now(timezone.utc).isoformat()
    earthaccess.login(strategy="netrc")

    # Pre-filter against the manifest: years already fully downloaded
    # in a prior run are skipped before any CMR search work.
    completed = _completed_years_from_manifest(workdir)
    pending = [y for y in years if y not in completed]
    if completed:
        logger.info(
            "margulis_wus_sr: skipping %d year(s) already in manifest: %s",
            len(completed & set(years)),
            sorted(completed & set(years)),
        )

    year_records: list[dict] = []
    for year in pending:
        year_dir = raw_root / f"{year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        logger.info("margulis_wus_sr: searching CMR for year %d", year)
        results = earthaccess.search_data(
            short_name=access["short_name"],
            version=access.get("version"),
            temporal=(f"{year}-01-01", f"{year}-12-31"),
            bounding_box=search_bbox,
        )
        n_found = len(results)
        if n_found == 0:
            logger.warning(
                "margulis_wus_sr: no granules found for year %d (bbox %s); skipping",
                year,
                search_bbox,
            )
            year_records.append(
                {
                    "year": year,
                    "raw_dir": str(year_dir),
                    "n_granules": 0,
                    "downloaded_utc": now_utc,
                    "note": "no_granules_in_CMR",
                }
            )
            continue
        downloaded = earthaccess.download(results, str(year_dir))
        if not downloaded:
            raise RuntimeError(
                f"margulis_wus_sr: earthaccess.download returned no files for "
                f"year {year}; check network connectivity and Earthdata "
                f"credentials."
            )
        # Drop zero-byte downloads (NSIDC occasionally returns truncated
        # granules; the consolidation step would later trip on them).
        usable = []
        for f in downloaded:
            p = Path(f)
            if p.exists() and p.stat().st_size > 0:
                usable.append(str(p))
            else:
                logger.warning("margulis_wus_sr: dropping zero-byte download %s", p)
                if p.exists():
                    p.unlink()
        if len(usable) < n_found:
            raise RuntimeError(
                f"margulis_wus_sr: partial download for year {year}: "
                f"{len(usable)}/{n_found} usable granules. Re-run to retry."
            )
        year_records.append(
            {
                "year": year,
                "raw_dir": str(year_dir),
                "n_granules": len(usable),
                "downloaded_utc": now_utc,
            }
        )

    # --- Consolidation pass ---------------------------------------------
    # Build the per-calendar-year CF NetCDF for every year we either
    # just downloaded *or* previously downloaded but never consolidated
    # (the backfill case — raws on disk, daily NC missing). Failures
    # are recorded in the manifest as `consolidate_error` but never
    # abort the fetch; the boundary-year case (final source WY has no
    # successor, e.g. CY 2021 with the v01 dataset) is the canonical
    # benign failure.
    daily_dir = raw_root.parent / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    # 1) Years we downloaded this run: consolidate inline so the manifest
    # entry for the same record carries both `n_granules` and
    # `daily_path`.
    for rec in year_records:
        if int(rec.get("n_granules", 0)) == 0:
            continue
        _attempt_consolidation_into_record(rec, raw_root, daily_dir)

    # 2) Backfill: years already in the manifest with raws on disk but
    # no daily NC. Append a fresh record per year so the manifest merge
    # updates the existing entry's `daily_path` without disturbing the
    # original download timestamp.
    downloaded_this_run = {int(r["year"]) for r in year_records}
    for backfill_year in _years_needing_consolidation(workdir):
        if backfill_year in downloaded_this_run:
            continue
        year_dir = raw_root / f"{backfill_year}"
        if not year_dir.exists():
            continue
        rec: dict = {"year": backfill_year, "raw_dir": str(year_dir)}
        _attempt_consolidation_into_record(rec, raw_root, daily_dir)
        year_records.append(rec)

    _update_manifest(workdir, period, meta, year_records, fabric_scope, search_bbox)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": access["url"],
        "doi": meta.get("doi"),
        "license": meta.get("license", "public domain (NSIDC)"),
        "variables": [v["name"] for v in meta["variables"]],
        "period": period,
        "search_bbox": search_bbox,
        "fabric_scope": fabric_scope,
        "years": year_records,
        "download_timestamp": now_utc,
    }


def _load_margulis_manifest_entry(workdir: Path) -> dict:
    """Return the ``manifest['sources']['margulis_wus_sr']`` entry, or ``{}``.

    Returns an empty dict on missing / unparseable manifest so callers can
    treat both cases uniformly.
    """
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        logger.warning(
            "margulis_wus_sr: manifest.json could not be parsed; treating "
            "all years as pending."
        )
        return {}
    return manifest.get("sources", {}).get(_SOURCE_KEY, {})


def _completed_years_from_manifest(workdir: Path) -> set[int]:
    """Return years with ``n_granules > 0`` recorded in the manifest.

    This is the **download-skip** gate: a year is treated as already
    downloaded once it has ≥1 granule recorded. The download-skip
    semantics are intentionally independent of consolidation status —
    deleting a consolidated NC should trigger a re-consolidation, not a
    re-download. The consolidation-backfill gate lives separately in
    :func:`_years_needing_consolidation`.

    Years recorded with ``n_granules: 0`` (no CMR hits) are NOT
    considered complete — re-runs will retry them, which is intentional
    because CMR coverage can fill in retroactively.
    """
    entry = _load_margulis_manifest_entry(workdir)
    return {
        int(rec["year"])
        for rec in entry.get("years", [])
        if int(rec.get("n_granules", 0)) > 0
    }


def _years_needing_consolidation(workdir: Path) -> set[int]:
    """Return years that have raw downloads but no consolidated NC on disk.

    A "backfill" set: years where ``n_granules > 0`` but either
    ``daily_path`` is missing from the manifest or the referenced file
    has been deleted. The fetch loop uses this to re-consolidate
    without re-downloading — supports the case where the raw store
    pre-dates the consolidator (raw NCs are intact, daily NCs need to
    be built).
    """
    entry = _load_margulis_manifest_entry(workdir)
    needs: set[int] = set()
    for rec in entry.get("years", []):
        if int(rec.get("n_granules", 0)) <= 0:
            continue
        daily_path = rec.get("daily_path")
        if not daily_path or not Path(daily_path).exists():
            needs.add(int(rec["year"]))
    return needs


def _update_manifest(
    workdir: Path,
    period: str,
    meta: dict,
    year_records: list[dict],
    fabric_scope: dict,
    search_bbox: tuple[float, float, float, float],
) -> None:
    """Merge Margulis provenance into manifest.json (flock-protected)."""
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    def _do_update() -> None:
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"manifest.json in {workdir} is corrupt and cannot be "
                    f"parsed. Delete it and re-run the fetch step. "
                    f"Original error: {exc}"
                ) from exc
        else:
            manifest = {"sources": {}, "steps": []}

        manifest.setdefault("sources", {})
        entry = manifest["sources"].get(_SOURCE_KEY, {})
        existing_by_year = {int(y["year"]): y for y in entry.get("years", [])}
        for rec in year_records:
            existing_by_year[int(rec["year"])] = rec
        merged_years = [existing_by_year[y] for y in sorted(existing_by_year)]
        access = meta["access"]
        entry.update(
            {
                "source_key": _SOURCE_KEY,
                "access_url": access["url"],
                "doi": meta.get("doi"),
                "license": meta.get("license", "public domain (NSIDC)"),
                "period": period,
                "search_bbox": list(search_bbox),
                "fabric_scope": fabric_scope,
                "variables": [v["name"] for v in meta["variables"]],
                "years": merged_years,
            }
        )
        manifest["sources"][_SOURCE_KEY] = entry

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=manifest_path.parent, suffix=".json.tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(manifest, f, indent=2)
            Path(tmp_path).replace(manifest_path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    if _HAVE_FLOCK:
        with open(lock_path, "a") as _lock_f:
            _fcntl.flock(_lock_f, _fcntl.LOCK_EX)
            try:
                _do_update()
            finally:
                _fcntl.flock(_lock_f, _fcntl.LOCK_UN)
    else:
        _do_update()
    logger.info(
        "Updated manifest.json with margulis_wus_sr provenance for years %s",
        [r["year"] for r in year_records],
    )
