"""Fetch Margulis Western US Snow Reanalysis (NSIDC-0719) via earthaccess.

NSIDC-0719 is a daily 90 m posterior SWE reanalysis over the Western US
for water years 1985-2021. This source is **fabric-scoped to Oregon
only** in this pipeline (see ``catalog/sources.yml ->
margulis_wus_sr.fabric_scope``). The fetch step does not enforce that
scope — raw downloads stay in the shared datastore and remain usable
by any project pointing at the same store — but the manifest entry
records the scope so downstream aggregate/target code can honour it.

This is the **fetch-only** scaffolding: search NSIDC via earthaccess,
download granules to ``<datastore>/margulis_wus_sr/raw/<year>/``,
flock-protect the manifest update. Per-year consolidation into CF
NetCDFs is deferred to the Margulis aggregate follow-up issue.
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
except ImportError:  # Windows fallback.
    _HAVE_FLOCK = False

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets.fetch._period import (
    parse_period,
    period_bounds,
    years_in_period,
)
from nhf_spatial_targets.workspace import load as _load_project

logger = logging.getLogger(__name__)

_SOURCE_KEY = "margulis_wus_sr"


def fetch_margulis_wus_sr(workdir: Path, period: str) -> dict:
    """Download Margulis WUS-SR daily granules to the shared datastore.

    Fetch-only path: search NSIDC for granules covering ``period`` within
    the project's fabric bbox, download to
    ``<datastore>/margulis_wus_sr/raw/<year>/``, and write a flock-protected
    manifest entry recording the fabric scope from the catalog. NSIDC-0719
    granules are per-water-year per-tile NetCDFs; the consolidation step
    that concatenates and re-projects them is deferred to a follow-up.

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


def _completed_years_from_manifest(workdir: Path) -> set[int]:
    """Return the set of years with ``n_granules > 0`` in manifest.json.

    Years recorded with ``n_granules: 0`` (no CMR hits) are NOT
    considered complete — re-runs will retry them, which is intentional
    because CMR coverage can fill in retroactively. Missing/corrupt
    manifest yields an empty set so the caller falls through to a clean
    fresh fetch.
    """
    ws = _load_project(workdir)
    manifest_path = ws.manifest_path
    if not manifest_path.exists():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        logger.warning(
            "margulis_wus_sr: manifest.json could not be parsed; treating "
            "all years as pending."
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
