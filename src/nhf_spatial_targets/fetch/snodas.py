"""Fetch SNODAS daily snow water equivalent from NSIDC's HTTPS archive.

This is the **fetch-only** scaffolding for SNODAS (NSIDC collection
G02158). NSIDC's CMR record for G02158 is a metadata-only stub with
**zero granule-level records** (verified in issue #107):
``earthaccess.search_data(short_name='G02158')`` returns 0 hits, no
matter the bbox or temporal filter. The data actually lives behind
NSIDC's Earthdata-Login-gated HTTPS archive at:

    https://noaadata.apps.nsidc.org/NOAA/G02158/masked/YYYY/MM_Mon/
        SNODAS_YYYYMMDD.tar

The fetch module constructs daily URLs from each date and streams the
``.tar`` bundle via the earthaccess HTTPS auth session
(``earthaccess.login(strategy='netrc').get_session()``). Each bundle
contains flat int16 binary fields plus ENVI-style ``.Hdr`` headers;
decoding to a CF NetCDF is **deferred** to the SNODAS aggregate
follow-up issue.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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
# Per-request HTTP timeout (connect + first-byte). SNODAS .tar bundles
# are O(10-30 MB); 60s is generous for a healthy network.
_HTTP_TIMEOUT_SECONDS = 60
# Chunk size for streaming downloads.
_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
# Defense in depth against truncated bodies (issue #107 review): a real
# SNODAS .tar is 10-30 MB, never below ~5 MB even for the early sparse
# years. Anything smaller is either a 404 HTML body that slipped past
# the status check or a mid-stream connection drop. The Content-Length
# integrity check in `_download_tar` is the primary defense; this
# minimum is a fallback for cases where the server omits Content-Length.
_MIN_VALID_TAR_BYTES = 1 * 1024 * 1024  # 1 MiB


def _assign_worker_years(
    all_years: list[int],
    worker_index: int,
    n_workers: int,
) -> list[int]:
    """Round-robin slice of ``all_years`` for ``worker_index`` of ``n_workers``.

    Mirrors ``era5_land._assign_worker_years`` (without the manifest-merge
    behaviour, which is not needed here — SNODAS does no per-month chunk
    pre-staging). Slicing the full list (not a remaining set) keeps each
    worker's assignment independent of sibling progress.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if not 0 <= worker_index < n_workers:
        raise ValueError(
            f"worker_index must be in [0, {n_workers}); got {worker_index}"
        )
    return list(all_years[worker_index::n_workers])


def _daily_urls(archive_url: str, year: int) -> list[tuple[pd.Timestamp, str]]:
    """Yield (date, URL) pairs for every calendar day in ``year``.

    Archive layout is ``<archive_url>/<year>/MM_Mon/SNODAS_YYYYMMDD.tar``,
    e.g. ``.../2020/01_Jan/SNODAS_20200101.tar``. Not every date carries
    a file — partial-year boundaries (2003 starts late September) and
    occasional gaps are normal; 404s are handled upstream.
    """
    base = archive_url.rstrip("/")
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="1D")
    out: list[tuple[pd.Timestamp, str]] = []
    for d in days:
        month_dir = f"{d.month:02d}_{d.strftime('%b')}"
        fname = f"SNODAS_{d.strftime('%Y%m%d')}.tar"
        out.append((d, f"{base}/{year}/{month_dir}/{fname}"))
    return out


def _download_tar(
    session, url: str, out_path: Path, *, timeout: int = _HTTP_TIMEOUT_SECONDS
) -> str:
    """Stream a single .tar from *url* to *out_path*. Returns a status code.

    Status codes (manifest-friendly):
      - ``"downloaded"`` — file streamed AND verified against Content-Length
        (or, if the server omits Content-Length, written through to a
        non-zero non-truncated-looking length) and written this call.
      - ``"already_present"`` — file exists on disk with size above
        :data:`_MIN_VALID_TAR_BYTES`; skipped.
      - ``"missing_404"`` — server returned 404; not an error (partial years
        and gaps are normal in SNODAS).
      - ``"error"`` — any other failure; logged, not raised.

    Atomic: streams to a ``.tar.tmp`` sibling, then renames on success.
    The ``.tar.tmp`` is unlinked on any failure path (including a
    Content-Length mismatch) so resume sees a clean directory.

    Assumes the caller has partitioned writers so only one worker
    targets any given year directory (mirrors ``fetch_era5_land``'s
    contract). The atomic rename is per-file race-free even within a
    worker.
    """
    # Treat existing files smaller than _MIN_VALID_TAR_BYTES as suspect
    # — SNODAS .tars are 10-30 MB; a few-hundred-byte stub is the
    # signature of a pre-fix #107 truncated write that the now-stricter
    # path would have rejected. Re-download on the next run.
    if out_path.exists():
        sz = out_path.stat().st_size
        if sz >= _MIN_VALID_TAR_BYTES:
            return "already_present"
        logger.warning(
            "snodas: existing %s is suspiciously small (%d bytes < %d); redownloading.",
            out_path.name,
            sz,
            _MIN_VALID_TAR_BYTES,
        )
        out_path.unlink(missing_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        resp = session.get(url, timeout=timeout, stream=True, allow_redirects=True)
    except Exception as exc:
        logger.warning("snodas: GET failed for %s: %s", url, exc)
        tmp.unlink(missing_ok=True)
        return "error"
    try:
        if resp.status_code == 404:
            return "missing_404"
        if resp.status_code != 200:
            logger.warning(
                "snodas: unexpected status %d for %s; skipping",
                resp.status_code,
                url,
            )
            return "error"

        # Capture the advertised body size BEFORE streaming so we can
        # verify integrity on close. NSIDC's archive sets Content-Length
        # on tarball responses; if it ever stops doing so (gzip transfer
        # encoding, etc.) we fall back to a minimum-size check.
        declared_len = resp.headers.get("Content-Length")
        try:
            declared_bytes: int | None = int(declared_len) if declared_len else None
        except ValueError:
            declared_bytes = None

        bytes_written = 0
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)

        # Integrity gate: requests' iter_content does NOT raise on a
        # short read — a mid-stream connection close that delivers
        # fewer bytes than promised leaves a truncated file looking
        # "complete". Prefer Content-Length when the server provided
        # one (NSIDC's archive does); fall back to a minimum-size
        # check otherwise. Either way, a failed check unlinks the tmp
        # and returns "error" so the next run retries the day cleanly.
        if declared_bytes is not None:
            if bytes_written != declared_bytes:
                logger.warning(
                    "snodas: short read for %s — wrote %d of %d declared bytes",
                    url,
                    bytes_written,
                    declared_bytes,
                )
                tmp.unlink(missing_ok=True)
                return "error"
        elif bytes_written < _MIN_VALID_TAR_BYTES:
            logger.warning(
                "snodas: truncated response for %s — wrote %d bytes "
                "(< %d minimum and no Content-Length header to confirm); "
                "discarding",
                url,
                bytes_written,
                _MIN_VALID_TAR_BYTES,
            )
            tmp.unlink(missing_ok=True)
            return "error"

        tmp.replace(out_path)
        return "downloaded"
    except Exception as exc:
        logger.warning("snodas: write failed for %s → %s: %s", url, out_path, exc)
        tmp.unlink(missing_ok=True)
        return "error"
    finally:
        try:
            resp.close()
        except Exception:
            pass


def _earthaccess_session():
    """Return an authenticated HTTPS session for noaadata.apps.nsidc.org.

    Wrapped for monkeypatch in tests. Production calls
    ``earthaccess.login(strategy='netrc').get_session()``.
    """
    import earthaccess

    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        raise RuntimeError(
            "earthaccess login failed for SNODAS; check ~/.netrc has a "
            "machine urs.earthdata.nasa.gov entry. Run "
            "`nhf-targets materialize-credentials --project-dir <project>` "
            "to refresh from .credentials.yml."
        )
    return auth.get_session()


def fetch_snodas(
    workdir: Path,
    period: str,
    *,
    worker_index: int = 0,
    n_workers: int = 1,
) -> dict:
    """Download SNODAS daily .tar bundles to the shared datastore.

    Fetch-only path: for each assigned year, walk the daily URLs at
    ``<archive_url>/<year>/MM_Mon/SNODAS_YYYYMMDD.tar`` and stream each
    bundle via the earthaccess HTTPS auth session. Per-day 404s are
    recorded but do not fail the year (partial-year boundaries and
    occasional gaps are normal). Decoding the int16 binary into CF
    NetCDFs is deferred to the SNODAS aggregate follow-up.

    Parameters
    ----------
    workdir : Path
        Project directory.
    period : str
        Temporal window ``"YYYY/YYYY"`` (inclusive). Validated against the
        catalog's ``period``.
    worker_index, n_workers : int
        Round-robin year sharding for parallel workers. Default
        single-worker (serial).

    Returns
    -------
    dict
        Provenance summary including a per-year ``years`` list with
        ``n_downloaded_this_run``, ``n_already_present``,
        ``n_missing_404``, ``n_errors``, and the cumulative
        ``n_granules`` (downloaded + already on disk).

    Raises
    ------
    ValueError
        Period falls outside the catalog window or
        ``access.archive_url`` is missing from the catalog.
    RuntimeError
        Earthdata login failed.
    """
    parse_period(period)
    ws = _load_project(workdir)
    meta = _catalog.source(_SOURCE_KEY)
    access = meta["access"]
    archive_url = access.get("archive_url")
    if not archive_url:
        raise ValueError(
            f"catalog `sources.yml[{_SOURCE_KEY}].access.archive_url` is "
            f"missing. Set it to the NSIDC HTTPS archive root, e.g. "
            f"https://noaadata.apps.nsidc.org/NOAA/G02158/masked/"
        )
    data_lo, data_hi = period_bounds(meta["period"])
    years = years_in_period(period)
    for y in years:
        if y < data_lo or y > data_hi:
            raise ValueError(
                f"Year {y} is outside the SNODAS publisher window "
                f"({data_lo}-{data_hi}, from catalog "
                f"`sources.yml[{_SOURCE_KEY}].period`). Adjust --period."
            )

    raw_root = ws.raw_dir(_SOURCE_KEY) / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc).isoformat()

    session = _earthaccess_session()

    # Pre-filter against the manifest: years already fully downloaded
    # are skipped before any HTTP work.
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
        return _build_summary(
            meta, period, archive_url, worker_index, n_workers, [], now_utc
        )

    year_records: list[dict] = []
    for year in assigned:
        year_dir = raw_root / f"{year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        rec = _fetch_year(session, archive_url, year, year_dir)
        rec["downloaded_utc"] = now_utc
        year_records.append(rec)
        logger.info(
            "snodas: year %d — downloaded=%d, already_present=%d, "
            "missing_404=%d, errors=%d",
            year,
            rec["n_downloaded_this_run"],
            rec["n_already_present"],
            rec["n_missing_404"],
            rec["n_errors"],
        )

    _update_manifest(workdir, period, meta, year_records, archive_url)
    return _build_summary(
        meta, period, archive_url, worker_index, n_workers, year_records, now_utc
    )


def _fetch_year(session, archive_url: str, year: int, year_dir: Path) -> dict:
    """Download every available daily .tar for ``year`` into ``year_dir``.

    Returns a per-year record dict. Per-day 404s are common at year
    boundaries and not treated as errors.
    """
    n_downloaded = 0
    n_already = 0
    n_404 = 0
    n_err = 0
    for date, url in _daily_urls(archive_url, year):
        fname = f"SNODAS_{date.strftime('%Y%m%d')}.tar"
        out_path = year_dir / fname
        status = _download_tar(session, url, out_path)
        if status == "downloaded":
            n_downloaded += 1
        elif status == "already_present":
            n_already += 1
        elif status == "missing_404":
            n_404 += 1
        else:
            n_err += 1
    n_granules = n_downloaded + n_already
    return {
        "year": year,
        "raw_dir": str(year_dir),
        "n_granules": n_granules,
        "n_downloaded_this_run": n_downloaded,
        "n_already_present": n_already,
        "n_missing_404": n_404,
        "n_errors": n_err,
    }


def _build_summary(
    meta: dict,
    period: str,
    archive_url: str,
    worker_index: int,
    n_workers: int,
    year_records: list[dict],
    now_utc: str,
) -> dict:
    access = meta["access"]
    return {
        "source_key": _SOURCE_KEY,
        "access_url": access["url"],
        "archive_url": archive_url,
        "doi": meta.get("doi"),
        "license": meta.get("license", "public domain (NSIDC)"),
        "variables": [v["name"] for v in meta["variables"]],
        "period": period,
        "worker_index": worker_index,
        "n_workers": n_workers,
        "years": year_records,
        "download_timestamp": now_utc,
    }


def _completed_years_from_manifest(workdir: Path) -> set[int]:
    """Return the set of years already recorded with ``n_granules > 0``.

    Years recorded with ``n_granules: 0`` are NOT considered complete —
    re-runs retry them (NSIDC coverage can fill in retroactively, and
    a year of all 404s usually means a transient archive issue).
    Missing/corrupt manifest yields an empty set so the caller falls
    through to a fresh fetch.
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
    archive_url: str,
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
                "archive_url": archive_url,
                "doi": meta.get("doi"),
                "license": meta.get("license", "public domain (NSIDC)"),
                "period": period,
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
