"""Fetch SNODAS daily snow water equivalent from NSIDC via earthaccess.

This is the **fetch-only** scaffolding for SNODAS (NSIDC collection
G02158). It downloads raw daily granules (tar/gz bundles of flat int16
binary fields plus ENVI ``.Hdr`` headers) into
``<datastore>/snodas/raw/<year>/`` and records them in ``manifest.json``.

Consolidation of the raw bundles into per-year daily and monthly CF
NetCDFs is **deferred** to the SNODAS aggregate follow-up issue. The
SNODAS native format (flat binary + ENVI ``.Hdr``) needs operator
characterisation in a notebook before a robust parser can be written,
per CLAUDE.md "characterise the data first" guidance. Until that work
lands, the fetch module records the raw paths and lets downstream
aggregate code raise loudly when it tries to consume them.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl as _fcntl

    _HAVE_FLOCK = True
except ImportError:  # Windows fallback (not used on HPC).
    _HAVE_FLOCK = False

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import (
    parse_period,
    period_bounds,
    years_in_period,
)
from nhf_spatial_targets.workspace import load as _load_project

logger = logging.getLogger(__name__)

_SOURCE_KEY = "snodas"


def _assign_worker_years(
    all_years: list[int],
    worker_index: int,
    n_workers: int,
) -> list[int]:
    """Round-robin slice of ``all_years`` for ``worker_index`` of ``n_workers``.

    Mirrors ``era5_land._assign_worker_years`` (without the manifest-merge
    behaviour, which is not needed here — SNODAS does no per-month chunk
    pre-staging). Each worker takes a deterministic slice of ``all_years``
    keyed by ``worker_index``; slicing the full list (not a remaining
    set) keeps each worker's assignment independent of sibling progress.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if not 0 <= worker_index < n_workers:
        raise ValueError(
            f"worker_index must be in [0, {n_workers}); got {worker_index}"
        )
    return list(all_years[worker_index::n_workers])


def fetch_snodas(
    workdir: Path,
    period: str,
    *,
    worker_index: int = 0,
    n_workers: int = 1,
) -> dict:
    """Download SNODAS daily granules to the shared datastore.

    This is the fetch-only path: search NSIDC via ``earthaccess`` for daily
    granules covering each requested year and download the tar/gz bundles
    into ``<datastore>/snodas/raw/<year>/``. The bundles are NOT decoded
    into CF NetCDFs in this PR (see module docstring); the consolidation
    step lives in the SNODAS aggregate follow-up issue.

    Parameters
    ----------
    workdir : Path
        Project directory.
    period : str
        Temporal window ``"YYYY/YYYY"`` (inclusive). Validated against the
        SNODAS publisher start (2003).
    worker_index, n_workers : int
        Round-robin year sharding for parallel workers (mirrors
        ``fetch_era5_land``). Defaults to a single-worker run.

    Returns
    -------
    dict
        Provenance summary.

    Raises
    ------
    ValueError
        Period falls outside the publisher window or no granules match.
    RuntimeError
        ``earthaccess.download`` returned fewer files than the search
        result count, or zero files at all.
    """
    import earthaccess

    parse_period(period)
    ws = _load_project(workdir)
    meta = _catalog.source(_SOURCE_KEY)
    access = meta["access"]
    data_lo, data_hi = period_bounds(meta["period"])
    years = years_in_period(period)
    for y in years:
        if y < data_lo or y > data_hi:
            raise ValueError(
                f"Year {y} is outside the SNODAS publisher window "
                f"({data_lo}-{data_hi}, from catalog `sources.yml[{_SOURCE_KEY}]"
                f".period`). Adjust --period."
            )
    # Use the project's buffered fabric bbox as the CMR search bounding
    # box (matches the pattern in merra2.py / nldas.py / margulis_wus_sr.py).
    # The catalog's `bbox_nwse` is CDS-convention metadata (kept for the
    # ERA5-Land fetch); CMR / earthaccess want `(W, S, E, N)`, which is
    # exactly the (minx, miny, maxx, maxy) order fabric.json records.
    bbox_buffered = ws.fabric.get("bbox_buffered") or {}
    if not bbox_buffered:
        raise ValueError(
            "SNODAS fetch needs a buffered fabric bbox; fabric.json has "
            "no 'bbox_buffered' key. Re-run `nhf-targets validate` to "
            "regenerate fabric.json."
        )
    search_bbox = (
        float(bbox_buffered["minx"]),
        float(bbox_buffered["miny"]),
        float(bbox_buffered["maxx"]),
        float(bbox_buffered["maxy"]),
    )

    raw_root = ws.raw_dir(_SOURCE_KEY) / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc).isoformat()

    earthaccess.login(strategy="netrc")
    # Pre-filter against the manifest: years already fully downloaded
    # in a prior run are skipped before any CMR search/download work.
    completed = _completed_years_from_manifest(workdir)
    pending = [y for y in years if y not in completed]
    if completed:
        logger.info(
            "snodas: skipping %d year(s) already in manifest: %s",
            len(completed & set(years)),
            sorted(completed & set(years)),
        )
    assigned = _assign_worker_years(pending, worker_index, n_workers)
    if not assigned:
        logger.info(
            "snodas: worker %d/%d has no years to process for period %s",
            worker_index,
            n_workers,
            period,
        )
        return {
            "source_key": _SOURCE_KEY,
            "access_url": access["url"],
            "doi": meta.get("doi"),
            "license": meta.get("license", "public domain (NSIDC)"),
            "variables": [v["name"] for v in meta["variables"]],
            "period": period,
            "search_bbox": list(search_bbox),
            "worker_index": worker_index,
            "n_workers": n_workers,
            "years": [],
            "download_timestamp": now_utc,
        }

    year_records: list[dict] = []
    for year in assigned:
        year_dir = raw_root / f"{year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        logger.info("snodas: searching CMR for year %d", year)
        results = earthaccess.search_data(
            short_name=access["short_name"],
            version=access.get("version"),
            temporal=(f"{year}-01-01", f"{year}-12-31"),
            bounding_box=search_bbox,
        )
        n_found = len(results)
        if n_found == 0:
            logger.warning("snodas: no granules found for year %d; skipping", year)
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
                f"snodas: earthaccess.download returned no files for year {year}; "
                f"check network connectivity and Earthdata credentials."
            )
        if len(downloaded) < n_found:
            raise RuntimeError(
                f"snodas: partial download for year {year}: "
                f"{len(downloaded)}/{n_found} granules. Re-run to retry."
            )
        year_records.append(
            {
                "year": year,
                "raw_dir": str(year_dir),
                "n_granules": len(downloaded),
                "downloaded_utc": now_utc,
            }
        )

    _update_manifest(workdir, period, meta, year_records, search_bbox)

    return {
        "source_key": _SOURCE_KEY,
        "access_url": access["url"],
        "doi": meta.get("doi"),
        "license": meta.get("license", "public domain (NSIDC)"),
        "variables": [v["name"] for v in meta["variables"]],
        "period": period,
        "search_bbox": list(search_bbox),
        "worker_index": worker_index,
        "n_workers": n_workers,
        "years": year_records,
        "download_timestamp": now_utc,
    }


def _completed_years_from_manifest(workdir: Path) -> set[int]:
    """Return the set of years already recorded in manifest.json.

    A year counts as complete when its manifest entry has ``n_granules > 0``;
    years recorded with ``n_granules: 0`` (no CMR hits) are NOT considered
    complete — re-runs will retry them, which is intentional because CMR
    coverage can fill in retroactively.

    Returns an empty set (with a warning) if the manifest is absent or
    unparseable so the caller falls through to a clean fresh fetch.
    """
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    if not manifest_path.exists():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        logger.warning(
            "snodas: manifest.json could not be parsed; treating all years as pending."
        )
        return set()
    entry = manifest.get("sources", {}).get(_SOURCE_KEY, {})
    return {
        int(rec["year"])
        for rec in entry.get("years", [])
        if int(rec.get("n_granules", 0)) > 0
    }


def _update_manifest(
    workdir: Path,
    period: str,
    meta: dict,
    year_records: list[dict],
    search_bbox: tuple[float, float, float, float],
) -> None:
    """Merge SNODAS provenance into manifest.json (flock-protected).

    Parallel workers may call this concurrently; the file lock makes the
    read-modify-write atomic. Years from the new ``year_records`` overwrite
    any earlier entries for the same year (latest write wins per year).
    """
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
        "Updated manifest.json with snodas provenance for years %s",
        [r["year"] for r in year_records],
    )
