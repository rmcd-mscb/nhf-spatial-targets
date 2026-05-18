"""Fetch SNODAS daily snow water equivalent from NSIDC's HTTPS archive.

This module both downloads the raw ``SNODAS_YYYYMMDD.tar`` bundles from
NSIDC and decodes them into per-year daily CF NetCDFs. NSIDC's CMR
record for G02158 is a metadata-only stub with **zero granule-level
records** (verified in issue #107):
``earthaccess.search_data(short_name='G02158')`` returns 0 hits, no
matter the bbox or temporal filter. The data actually lives behind
NSIDC's Earthdata-Login-gated HTTPS archive at:

    https://noaadata.apps.nsidc.org/NOAA/G02158/masked/YYYY/MM_Mon/
        SNODAS_YYYYMMDD.tar

The fetch module constructs daily URLs from each date and streams the
``.tar`` bundle via the earthaccess HTTPS auth session
(``earthaccess.login(strategy='netrc').get_session()``). Each bundle
contains flat int16 binary fields plus ENVI-style ``.Hdr`` headers.
After all of a year's ``.tars`` are on disk, :func:`consolidate_year_snodas`
decodes product 1034 (SWE) from each day, **reprojects from native
WGS84 to EPSG:5070 (NAD83 / CONUS Albers Equal Area) at 1000 m using
nearest-neighbour resampling**, and writes a single per-year CF NetCDF
at ``<datastore>/snodas/daily/snodas_daily_<year>.nc``. The
pre-projection (issue #121) eliminates ~5-8× of gdptools weight-gen
cost in the aggregator by matching the aggregator's WEIGHT_GEN_CRS;
nearest-neighbour preserves the integer-like ``-9999`` fill code
without bleed into neighbouring cells (SWE is instantaneous, not a
flux). Raw ``.tars`` are preserved on disk after consolidation for
provenance and future re-decode if catalog metadata changes.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    import fcntl as _fcntl

    _HAVE_FLOCK = True
except ImportError:  # Windows fallback (not used on HPC).
    _HAVE_FLOCK = False

import nhf_spatial_targets.catalog as _catalog
from nhf_spatial_targets import __version__
from nhf_spatial_targets.fetch._period import (
    parse_period,
    period_bounds,
    years_in_period,
)
from nhf_spatial_targets.fetch.consolidate import apply_cf_metadata
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

# Per-day .tar member regex for the SWE product (NSIDC code 1034). The
# pattern is stable across all years 2003-2024; verified by inspecting
# headers from 2003, 2009, 2013, 2014, 2024 (notebook
# inspect_consolidated_snodas.ipynb).
_SWE_PRODUCT_REGEX = re.compile(r"us_ssmv11034.*\.dat\.gz$")
_SWE_HEADER_REGEX = re.compile(r"us_ssmv11034.*\.txt\.gz$")
# Per the NSIDC user guide: -9999 is the only documented fill code for
# the masked product. No "saturated" sentinel is documented; pixels at
# the int16 max of 32767 are valid (typically glaciated peaks).
_SNODAS_FILL = -9999
# Within-year header coord tolerance. Cells are ~0.00833° (30 arcsec)
# wide; 1e-6 deg (~0.1 m) is well below cell width and absorbs the
# sub-µdeg text-representation noise observed between headers within a
# year, while still flagging real mid-year format changes loudly.
_SNODAS_GRID_TOL_DEG = 1e-6
# Sub-pixel drift tolerance for within-year header comparison. SNODAS
# masked CONUS is on a 30-arcsec grid (1/120 °); ~half a pixel
# (0.5/120 ≈ 0.00417 °) absorbs the observed CY 2013 mid-year origin
# drift (~4×10⁻⁴ ° = ~5% of a pixel) without silently accepting a
# real format/resolution change. Drifts in this band are recorded as
# provenance global attrs but do not abort consolidation. Used only
# when row/column counts are unchanged — see ``consolidate_year_snodas``.
_SNODAS_GRID_DRIFT_TOL_DEG = 0.5 / 120
_SNODAS_DAILY_FILENAME_TEMPLATE = "snodas_daily_{year}.nc"
# Pre-projection target CRS + resolution (issue #121). EPSG:5070 matches
# the aggregator's WEIGHT_GEN_CRS so gdptools' weight gen does not need
# to reproject the source grid at all (~5-8× speedup on CONUS fabrics).
# 1000 m approximately preserves the native 30 arcsec (~1 km) sampling
# without over-densifying.
_SNODAS_DST_CRS = "EPSG:5070"
_SNODAS_DST_RESOLUTION_M = 1000.0

# Locale-independent month abbreviations for URL construction. The NSIDC
# archive uses fixed English short names (e.g. ``01_Jan``); using
# ``date.strftime('%b')`` would emit ``01_Jän`` on a German LC_TIME and
# every URL would 404. Index 0 unused so month integer maps directly.
_MONTH_NAMES: tuple[str, ...] = (
    "",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


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
    # Intentionally probe-by-GET rather than HEAD-then-GET: 404s are
    # cheap (single round trip, no body), and HTML directory parsing
    # would add complexity and locale assumptions about the index page.
    base = archive_url.rstrip("/")
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="1D")
    out: list[tuple[pd.Timestamp, str]] = []
    for d in days:
        month_dir = f"{d.month:02d}_{_MONTH_NAMES[d.month]}"
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

    Wraps ``earthaccess.login(strategy='netrc').get_session()`` and
    mounts a ``urllib3.Retry`` adapter so transient flakes (502/503/504
    and connection resets) don't immediately turn into permanent
    ``"error"`` records on the per-day status counter. With ~8k daily
    files per full run, even a 99.9% per-request success rate produces
    a handful of stragglers; the retry budget of 3 attempts with
    exponential backoff turns most of those into successful downloads
    rather than operator-driven re-runs.

    Wrapped (rather than inlined into ``fetch_snodas``) for
    monkeypatching in tests.
    """
    import earthaccess
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        raise RuntimeError(
            "earthaccess login failed for SNODAS; check ~/.netrc has a "
            "machine urs.earthdata.nasa.gov entry. Run "
            "`nhf-targets materialize-credentials --project-dir <project>` "
            "to refresh from .credentials.yml."
        )
    session = auth.get_session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=(500, 502, 503, 504),
        # Idempotent GETs only — safe to retry.
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _parse_snodas_header(text: str) -> dict[str, str]:
    """Parse a SNODAS ENVI ``.Hdr`` header text into a flat ``dict``.

    Headers are key/value lines joined by ``:``. Trailing whitespace and
    the trailing-comment style ``Not applicable`` placeholders are left
    as-is; callers reach for the specific keys they need (rows, cols,
    bbox, resolution).
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        out[key.strip()] = val.strip()
    return out


def _date_from_tar_filename(tar_path: Path) -> pd.Timestamp:
    """Extract ``YYYYMMDD`` from ``SNODAS_YYYYMMDD.tar``."""
    m = re.search(r"SNODAS_(\d{8})\.tar$", tar_path.name)
    if not m:
        raise ValueError(
            f"{tar_path.name}: cannot parse YYYYMMDD from filename "
            f"(expected SNODAS_YYYYMMDD.tar)."
        )
    return pd.Timestamp(m.group(1))


def _read_snodas_swe_header(tar_path: Path) -> dict[str, str]:
    """Open ``tar_path`` in-memory and return the SWE product's parsed header.

    Cheap by comparison to the binary decode: only the ``us_ssmv11034*.txt.gz``
    member (typically <1 KB compressed) is read. Used by the streaming
    consolidator to validate grid stability across an entire year before
    decoding any binary.

    Raises
    ------
    ValueError
        No SWE header member, or header missing the rows/cols keys.
    """
    with tarfile.open(tar_path, "r") as tf:
        members = tf.getnames()
        hdr = next((m for m in members if _SWE_HEADER_REGEX.search(m)), None)
        if hdr is None:
            raise ValueError(
                f"{tar_path.name}: no SWE header (code 1034) member. "
                f"Tar members: {members}"
            )
        hdr_member = tf.extractfile(hdr)
        if hdr_member is None:
            raise ValueError(f"{tar_path.name}: cannot extract {hdr}")
        header_text = gzip.decompress(hdr_member.read()).decode("latin-1")
    header = _parse_snodas_header(header_text)
    for required in ("Number of rows", "Number of columns"):
        if required not in header:
            raise ValueError(
                f"{tar_path.name}: SWE header missing required key {required!r}"
            )
    return header


def _read_snodas_swe_array(
    tar_path: Path, expected_rows: int, expected_cols: int
) -> np.ndarray:
    """Decode just the SWE binary from ``tar_path`` to an ``int16`` array.

    Called once per day during the dask-streaming consolidation write.
    Memory footprint per call: ~one day's worth (≈ 46 MB at native CONUS
    resolution); released as soon as the chunk is compressed and written.

    Raises
    ------
    ValueError
        Member missing, binary size disagrees with ``(expected_rows, expected_cols)``.
    """
    with tarfile.open(tar_path, "r") as tf:
        members = tf.getnames()
        dat = next((m for m in members if _SWE_PRODUCT_REGEX.search(m)), None)
        if dat is None:
            raise ValueError(
                f"{tar_path.name}: no SWE product (code 1034) member. "
                f"Tar members: {members}"
            )
        dat_member = tf.extractfile(dat)
        if dat_member is None:
            raise ValueError(f"{tar_path.name}: cannot extract {dat}")
        raw = gzip.decompress(dat_member.read())

    expected_bytes = expected_rows * expected_cols * 2
    if len(raw) != expected_bytes:
        raise ValueError(
            f"{tar_path.name}: SWE binary has {len(raw)} bytes, expected "
            f"{expected_bytes} for ({expected_rows}, {expected_cols}) big-endian int16."
        )
    # `.astype(np.int16)` already returns a new owned array (with native
    # byte order on little-endian hosts), so the gzip buffer is released
    # as soon as this function returns.
    return (
        np.frombuffer(raw, dtype=">i2")
        .reshape(expected_rows, expected_cols)
        .astype(np.int16)
    )


def _decode_snodas_swe_tar(
    tar_path: Path,
) -> tuple[pd.Timestamp, np.ndarray, dict[str, str]]:
    """Decode the SWE product (NSIDC code 1034) out of one daily ``.tar``.

    Returns ``(date, swe_int16_2d, header_dict)``. Convenience wrapper
    around :func:`_read_snodas_swe_header` + :func:`_read_snodas_swe_array`
    for callers that want both halves in one round trip. The streaming
    consolidator uses the two helpers separately so the (large) binary
    decode is lazy.

    Raises
    ------
    ValueError
        Member missing, binary size mismatch, or unparseable date.
    """
    date = _date_from_tar_filename(tar_path)
    header = _read_snodas_swe_header(tar_path)
    rows = int(header["Number of rows"])
    cols = int(header["Number of columns"])
    arr = _read_snodas_swe_array(tar_path, rows, cols)
    return date, arr, header


def _coords_from_snodas_header(
    header: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (lat_descending, lon_ascending) cell-center coords."""
    rows = int(header["Number of rows"])
    cols = int(header["Number of columns"])
    lon_min = float(header["Minimum x-axis coordinate"])
    lat_max = float(header["Maximum y-axis coordinate"])
    dx = float(header["X-axis resolution"])
    dy = float(header["Y-axis resolution"])
    lon = lon_min + dx * (np.arange(cols) + 0.5)
    lat = lat_max - dy * (np.arange(rows) + 0.5)
    return lat.astype(np.float64), lon.astype(np.float64)


def _grids_match(
    h1: dict[str, str],
    h2: dict[str, str],
    tol: float = _SNODAS_GRID_TOL_DEG,
) -> bool:
    """Whether two SNODAS headers describe the same grid within ``tol`` deg."""
    if h1.get("Number of rows") != h2.get("Number of rows"):
        return False
    if h1.get("Number of columns") != h2.get("Number of columns"):
        return False
    for k in (
        "Minimum x-axis coordinate",
        "Maximum y-axis coordinate",
        "X-axis resolution",
        "Y-axis resolution",
    ):
        try:
            d = abs(float(h1[k]) - float(h2[k]))
        except KeyError:
            return False
        if d > tol:
            return False
    return True


def _max_grid_drift_deg(h1: dict[str, str], h2: dict[str, str]) -> float | None:
    """Return the max absolute origin/resolution drift between two headers.

    Returns ``None`` if ``(rows, cols)`` differ (a structural mismatch
    that cannot be a sub-pixel drift), or if a required key is missing.
    Otherwise returns the max ``|h1[k] - h2[k]|`` over the four
    grid-defining numeric fields (``Minimum x-axis coordinate``,
    ``Maximum y-axis coordinate``, ``X-axis resolution``,
    ``Y-axis resolution``).
    """
    if h1.get("Number of rows") != h2.get("Number of rows"):
        return None
    if h1.get("Number of columns") != h2.get("Number of columns"):
        return None
    drifts: list[float] = []
    for k in (
        "Minimum x-axis coordinate",
        "Maximum y-axis coordinate",
        "X-axis resolution",
        "Y-axis resolution",
    ):
        try:
            drifts.append(abs(float(h1[k]) - float(h2[k])))
        except KeyError:
            return None
    return max(drifts)


def _build_wgs84_dataarray(
    swe_int16: np.ndarray,
    wgs84_lat: np.ndarray,
    wgs84_lon: np.ndarray,
) -> "xr.DataArray":
    """Wrap a raw SNODAS day in a rio-tagged WGS84 DataArray for reprojection."""
    import rioxarray  # noqa: F401  (registers ``.rio`` accessor)

    da = xr.DataArray(
        swe_int16,
        coords={"y": wgs84_lat, "x": wgs84_lon},
        dims=("y", "x"),
        name="swe",
    )
    # write_crs + write_nodata teach rioxarray the source CRS and the
    # fill code to honour during resampling. The raw SNODAS array
    # carries -9999 in band as int16 (no scaling applied), so
    # ``encoded=False`` (default) is correct — the value is in the
    # array, not hidden behind a mask_and_scale decode.
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.write_nodata(_SNODAS_FILL)
    return da


def _compute_dst_grid(
    sample_day: np.ndarray,
    wgs84_lat: np.ndarray,
    wgs84_lon: np.ndarray,
) -> tuple[object, tuple[int, int], np.ndarray, np.ndarray]:
    """One-shot reprojection of the first day, used to lock the year's grid.

    The full year's per-day reprojections all use the same destination
    transform/shape so the stacked output has a single, stable EPSG:5070
    grid. Returns ``(transform, (rows, cols), y_metres, x_metres)``.
    """
    from rasterio.enums import Resampling

    template = _build_wgs84_dataarray(sample_day, wgs84_lat, wgs84_lon)
    reprojected = template.rio.reproject(
        _SNODAS_DST_CRS,
        resolution=_SNODAS_DST_RESOLUTION_M,
        resampling=Resampling.nearest,
        nodata=_SNODAS_FILL,
    )
    dst_transform = reprojected.rio.transform()
    dst_shape = (int(reprojected.sizes["y"]), int(reprojected.sizes["x"]))
    y_metres = np.asarray(reprojected["y"].values, dtype=np.float64)
    x_metres = np.asarray(reprojected["x"].values, dtype=np.float64)
    return dst_transform, dst_shape, y_metres, x_metres


def _decode_and_reproject_day(
    tar_path: Path,
    rows: int,
    cols: int,
    wgs84_lat: np.ndarray,
    wgs84_lon: np.ndarray,
    dst_transform: object,
    dst_shape: tuple[int, int],
) -> np.ndarray:
    """Decode one .tar's SWE binary, reproject to the locked EPSG:5070 grid.

    Combines the two operations in one delayed task so the WGS84 source
    array is released as soon as the projected destination is built;
    peak per-task memory is ~source + ~destination (≈ 60-70 MB for full
    CONUS), independent of the year length.
    """
    from rasterio.enums import Resampling

    raw = _read_snodas_swe_array(tar_path, rows, cols)
    src = _build_wgs84_dataarray(raw, wgs84_lat, wgs84_lon)
    dst = src.rio.reproject(
        _SNODAS_DST_CRS,
        shape=dst_shape,
        transform=dst_transform,
        resampling=Resampling.nearest,
        nodata=_SNODAS_FILL,
    )
    return np.asarray(dst.values, dtype=np.int16)


def _compute_lat_lon_2d(
    y_metres: np.ndarray, x_metres: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D (lat, lon) auxiliary coords at EPSG:5070 cell centres."""
    from pyproj import Transformer

    xx, yy = np.meshgrid(x_metres, y_metres)
    transformer = Transformer.from_crs(_SNODAS_DST_CRS, "EPSG:4326", always_xy=True)
    lon2d, lat2d = transformer.transform(xx, yy)
    return lat2d.astype(np.float32), lon2d.astype(np.float32)


def consolidate_year_snodas(
    year: int,
    raw_dir: Path,
    daily_dir: Path,
    *,
    grid_drift_tol_deg: float = _SNODAS_GRID_DRIFT_TOL_DEG,
) -> Path:
    """Decode every ``SNODAS_YYYYMMDD.tar`` in ``raw_dir`` into one year NC.

    Mirrors :func:`nhf_spatial_targets.fetch.era5_land.consolidate_year`:

    - **Idempotent**: skips rebuild when the output exists and is newer
      than every input ``.tar`` (mtime check, same pattern as ERA5-Land).
      *Schema-change caveat (issue #121)*: the on-disk schema flipped
      from WGS84 lat/lon to EPSG:5070 projected y/x. Operators upgrading
      from pre-#121 daily NCs must ``rm <datastore>/snodas/daily/*.nc``
      manually before re-running; otherwise mtime says "up to date" and
      the schema change never takes effect.
    - **Atomic**: writes to ``.nc.tmp`` then renames; an interrupted run
      never leaves a half-written NC at the canonical path.
    - **Pre-projection (issue #121)**: every day's SWE binary is decoded
      as native WGS84, then immediately reprojected to EPSG:5070
      (NAD83 / CONUS Albers, 1000 m) using ``Resampling.nearest`` so the
      ``-9999`` fill code does not bleed into neighbouring cells. The
      whole year's per-day reprojections target a single (transform,
      shape) computed from the first day, so the stacked output has a
      stable EPSG:5070 grid. The aggregator's ``WEIGHT_GEN_CRS`` is also
      EPSG:5070, so gdptools skips reprojection entirely (~5-8× speedup).
    - **CF metadata**: applies :func:`apply_cf_metadata` with
      ``time_step="daily"`` and ``coord_type="projected"``, so the
      variable carries ``units`` from catalog ``cf_units`` (``kg m-2``),
      ``cell_methods``, ``grid_mapping``, and a CF-correct EPSG:5070
      ``crs`` ancillary variable. 2D ``lat``/``lon`` auxiliary
      coordinates (computed from the EPSG:5070 cell centres) are
      attached as CF auxiliary coordinate variables so downstream tools
      that need geographic coords still have them — but note these
      describe the *new* projected cell centres, not the original SNODAS
      30-arcsec pixel positions.
    - **Storage**: SWE stored as native ``int16`` with
      ``_FillValue=-9999`` plus zlib (level 4). xarray's default
      ``mask_and_scale=True`` on read converts fills to NaN
      transparently for downstream consumers.

    The per-year NC carries the first day's grid metadata as a global
    attribute (``snodas_first_day_header``) for provenance — SNODAS has
    re-georeferenced the masked product at least twice (between
    2003/2004 and 2013/2014), with sub-pixel shifts (< 1e-3 deg). The
    consolidator verifies every day within the year shares the same
    ``(rows, cols)`` and the same grid origin to within
    ``grid_drift_tol_deg`` of the first day:

    - Exact match (within the strict 1e-6 ° tolerance) — pass silently.
    - Drift within ``grid_drift_tol_deg`` (default ~half a SNODAS pixel)
      AND identical ``(rows, cols)`` — accept the day, log a warning,
      and record ``snodas_grid_drift_days_count`` /
      ``snodas_grid_drift_max_deg`` global attrs on the consolidated NC.
      The day's pixel array is inserted as-is and labelled with the
      first-day's lat/lon (zero-shift nearest-neighbour, since the
      drift is sub-pixel and shape is unchanged).
    - Beyond ``grid_drift_tol_deg`` or any ``(rows, cols)`` change —
      raise ``ValueError``; cross-year shifts are expected and live at
      the file boundary.

    Parameters
    ----------
    year : int
    raw_dir : Path
        Directory containing the ``SNODAS_YYYYMMDD.tar`` files for one year
        (typically ``<datastore>/snodas/raw/<year>/``).
    daily_dir : Path
        Output directory (typically ``<datastore>/snodas/daily/``). Created
        if missing.
    grid_drift_tol_deg : float, optional
        Maximum within-year header drift accepted as sub-pixel noise.
        Defaults to half a SNODAS pixel (0.5/120 ° ≈ 0.00417 °).

    Returns
    -------
    Path
        ``daily_dir / "snodas_daily_<year>.nc"``.

    Raises
    ------
    FileNotFoundError
        No ``SNODAS_*.tar`` files in ``raw_dir``.
    ValueError
        Any day fails to decode, or a within-year header mismatch is
        detected beyond ``grid_drift_tol_deg``.
    """
    raw_dir = Path(raw_dir)
    daily_dir = Path(daily_dir)
    daily_dir.mkdir(parents=True, exist_ok=True)
    out_path = daily_dir / _SNODAS_DAILY_FILENAME_TEMPLATE.format(year=year)

    tar_paths = sorted(raw_dir.glob("SNODAS_*.tar"))
    if not tar_paths:
        raise FileNotFoundError(
            f"No SNODAS_*.tar files found in {raw_dir}. "
            f"Run 'nhf-targets fetch snodas' for year {year} first."
        )

    if out_path.exists():
        out_mtime = out_path.stat().st_mtime
        newest_tar_mtime = max(p.stat().st_mtime for p in tar_paths)
        if newest_tar_mtime <= out_mtime:
            logger.info(
                "snodas: daily NC up-to-date for %d (%d tars older than NC); skipping: %s",
                year,
                len(tar_paths),
                out_path,
            )
            return out_path
        logger.info(
            "snodas: raw .tars newer than daily NC for %d; re-consolidating: %s",
            year,
            out_path,
        )

    logger.info(
        "snodas: consolidating %d day(s) for year %d -> %s",
        len(tar_paths),
        year,
        out_path,
    )

    # Phase 1 — eager, cheap: read every day's header to validate the
    # within-year grid invariant before touching any binary. A SNODAS
    # header is ~1 KB compressed; reading 366 of them is sub-second.
    # `tar_paths` is already chronologically sorted (zero-padded
    # YYYYMMDD filenames lex-sort = chronologically-sort), so we don't
    # need a separate np.argsort after decode.
    first_header = _read_snodas_swe_header(tar_paths[0])
    rows = int(first_header["Number of rows"])
    cols = int(first_header["Number of columns"])
    wgs84_lat, wgs84_lon = _coords_from_snodas_header(first_header)
    times = np.array(
        [_date_from_tar_filename(p).to_datetime64() for p in tar_paths],
        dtype="datetime64[ns]",
    )

    # Track sub-pixel drift days: SNODAS has been re-georeferenced
    # mid-year on at least one occasion (CY 2013, ~4×10⁻⁴ ° origin
    # shift between Jan 1 and Oct 1), and the array shape stays the
    # same across the shift. Accept these as zero-shift NN snapped to
    # the first-day grid; surface a real format change as ValueError.
    drift_days: list[tuple[str, float]] = []  # (tar name, max drift deg)
    for p in tar_paths[1:]:
        header = _read_snodas_swe_header(p)
        if _grids_match(first_header, header):
            continue
        drift = _max_grid_drift_deg(first_header, header)
        if drift is not None and drift <= grid_drift_tol_deg:
            drift_days.append((p.name, drift))
            continue
        raise ValueError(
            f"snodas: within-year grid mismatch in {year}: {p.name} "
            f"differs from {tar_paths[0].name} (first day). "
            f"first: rows={first_header.get('Number of rows')}, "
            f"cols={first_header.get('Number of columns')}, "
            f"min_x={first_header.get('Minimum x-axis coordinate')}, "
            f"max_y={first_header.get('Maximum y-axis coordinate')}. "
            f"mismatching: rows={header.get('Number of rows')}, "
            f"cols={header.get('Number of columns')}, "
            f"min_x={header.get('Minimum x-axis coordinate')}, "
            f"max_y={header.get('Maximum y-axis coordinate')}. "
            f"Drift beyond grid_drift_tol_deg={grid_drift_tol_deg:g} "
            f"(or shape change). Re-consolidate after removing the "
            f"mismatching .tar; cross-year shifts are expected at year "
            f"boundaries, but within-year mismatches beyond sub-pixel "
            f"drift indicate a corrupt or mid-year-changed download "
            f"that should not be silently absorbed."
        )

    if drift_days:
        max_drift = max(d for _, d in drift_days)
        first_drift_name, _ = drift_days[0]
        logger.warning(
            "snodas: %d day(s) in %d show within-year sub-pixel grid drift "
            "≤ %g deg (max %g deg); accepting as zero-shift NN snap to "
            "first-day grid. First drift day: %s. Recorded in NC global "
            "attrs `snodas_grid_drift_days_count` and "
            "`snodas_grid_drift_max_deg`.",
            len(drift_days),
            year,
            grid_drift_tol_deg,
            max_drift,
            first_drift_name,
        )

    # Phase 2a — eager but cheap: lock the destination EPSG:5070 grid by
    # reprojecting the first day once. Every subsequent day's reprojection
    # targets this same (transform, shape) so the stacked output has a
    # single stable projected grid for the year. Decoding one day's binary
    # to drive the computation is unavoidable but small (~46 MB int16).
    sample_day = _read_snodas_swe_array(tar_paths[0], rows, cols)
    dst_transform, dst_shape, dst_y, dst_x = _compute_dst_grid(
        sample_day, wgs84_lat, wgs84_lon
    )
    del sample_day  # release the WGS84 source array before the dask stage
    dst_rows, dst_cols = dst_shape
    lat2d, lon2d = _compute_lat_lon_2d(dst_y, dst_x)

    # Phase 2b — lazy: build a dask-delayed stack of (decode + reproject)
    # per day. The combined task releases its WGS84 source array as soon
    # as the EPSG:5070 destination is built, so peak per-task memory is
    # ~source + ~destination (≈ 60-70 MB for full CONUS), independent of
    # the year length. xarray materializes one chunk at a time during
    # `to_netcdf` (synchronous scheduler below) and the zlib-compressed
    # chunk is written and released before the next is decoded — keeps
    # the full year off the heap (the old eager stack was ~17 GB int16
    # for a CONUS year and caused OOM under --mem=32G in SLURM job
    # 17553331). The chunked storage layout `(1, dst_rows, dst_cols)`
    # matches exactly one day per dask block.
    import dask
    import dask.array as da

    delayed_days = [
        dask.delayed(_decode_and_reproject_day)(
            p, rows, cols, wgs84_lat, wgs84_lon, dst_transform, dst_shape
        )
        for p in tar_paths
    ]
    swe_stack = da.stack(
        [da.from_delayed(d, shape=dst_shape, dtype=np.int16) for d in delayed_days],
        axis=0,
    )

    ds = xr.Dataset(
        {"swe": (("time", "y", "x"), swe_stack)},
        coords={
            "time": times,
            "y": dst_y,
            "x": dst_x,
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
        },
    )
    ds = apply_cf_metadata(ds, _SOURCE_KEY, "daily", coord_type="projected")

    # Record the first-day (WGS84) grid metadata as a global attr so future
    # readers can detect cross-year drift without re-opening a .tar. These
    # are the *source* SNODAS grid parameters; the on-disk NC is in EPSG:5070.
    first_header_subset = {
        k: first_header[k]
        for k in (
            "Minimum x-axis coordinate",
            "Maximum x-axis coordinate",
            "Minimum y-axis coordinate",
            "Maximum y-axis coordinate",
            "X-axis resolution",
            "Y-axis resolution",
            "Number of rows",
            "Number of columns",
        )
        if k in first_header
    }
    ds.attrs.update(
        {
            "title": (f"SNODAS daily snow water equivalent (CONUS, masked) {year}"),
            "institution": "NOAA NOHRSC / NSIDC",
            "source": "SNODAS (NSIDC G02158)",
            "references": "doi:10.7265/N5TB14TC",
            "frequency": "day",
            "history": f"Consolidated by nhf-spatial-targets v{__version__}",
            "snodas_first_day_header": json.dumps(first_header_subset),
            "snodas_reprojection": (
                f"native WGS84 -> {_SNODAS_DST_CRS} at "
                f"{_SNODAS_DST_RESOLUTION_M:g} m, Resampling.nearest "
                f"(pre-projected at consolidate time per issue #121)"
            ),
        }
    )
    if drift_days:
        ds.attrs["snodas_grid_drift_days_count"] = len(drift_days)
        ds.attrs["snodas_grid_drift_max_deg"] = max(d for _, d in drift_days)
        ds.attrs["snodas_grid_drift_tol_deg"] = grid_drift_tol_deg

    encoding = {
        "swe": {
            "zlib": True,
            "complevel": 4,
            "_FillValue": np.int16(_SNODAS_FILL),
            "chunksizes": (1, dst_rows, dst_cols),
            "dtype": "int16",
        },
    }

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        # Force the synchronous dask scheduler during write so chunks are
        # decoded ONE AT A TIME instead of in parallel. xarray's default
        # `to_netcdf` uses dask's threaded scheduler, which dispatches
        # multiple chunk-decode tasks concurrently and balloons RAM with
        # in-flight decoded chunks. For SNODAS each chunk is ~46 MB int16
        # at native CONUS resolution; parallel decode of even 4 chunks
        # would put ~200 MB in flight per chunk × ~3-4× working overhead
        # ≈ multi-GB peak per year. Synchronous keeps peak bounded to
        # ~1-2 chunks (~hundreds of MB total) regardless of n_days.
        # The serialization cost is small for SNODAS (~1 s decode per
        # day, ~6 min serial vs ~2 min parallel for a full year) and the
        # memory savings turn 32 GB SLURM grants from "OOM at year 2003"
        # into "comfortably fits in 4-8 GB".
        with dask.config.set(scheduler="synchronous"):
            ds.to_netcdf(tmp, format="NETCDF4", encoding=encoding)
        tmp.replace(out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    logger.info("snodas: wrote daily NC: %s", out_path)
    return out_path


def fetch_snodas(
    workdir: Path,
    period: str,
    *,
    worker_index: int = 0,
    n_workers: int = 1,
) -> dict:
    """Download SNODAS daily .tar bundles and consolidate to per-year CF NetCDFs.

    Two-phase per year: (1) walk the daily URLs at
    ``<archive_url>/<year>/MM_Mon/SNODAS_YYYYMMDD.tar`` and stream each
    bundle via the earthaccess HTTPS auth session — per-day 404s are
    recorded but do not fail the year (partial-year boundaries and
    occasional gaps are normal). (2) Once the year's .tars are on disk,
    :func:`consolidate_year_snodas` decodes the SWE product out of every
    bundle into a single per-year CF NetCDF at
    ``<datastore>/snodas/daily/snodas_daily_<year>.nc``. Consolidation
    failures are logged and stored on the per-year manifest record as
    ``consolidate_error`` rather than aborting the run (the raw .tars are
    preserved and the consolidation can be retried).

    Backfill: years already downloaded by a prior run (manifest entry has
    ``n_granules > 0`` but no ``daily_path``) are re-entered on the next
    run; the new completion check requires ``daily_path`` to exist, so
    only the consolidation step runs — no re-download.

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
    daily_dir = ws.raw_dir(_SOURCE_KEY) / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
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
        logger.info(
            "snodas: year %d — downloaded=%d, already_present=%d, "
            "missing_404=%d, errors=%d",
            year,
            rec["n_downloaded_this_run"],
            rec["n_already_present"],
            rec["n_missing_404"],
            rec["n_errors"],
        )
        # Consolidation step: decode every .tar into a per-year daily NC
        # and stash the path on the per-year record. Data-level failures
        # (corrupt tar, missing inputs, header drift, disk full) are caught
        # and recorded as `consolidate_error` so the run continues for
        # other years — raw .tars are preserved for retry. Programming
        # errors (AttributeError, ImportError, TypeError, ...) are NOT
        # caught here so a real bug aborts the run loudly rather than
        # silently degrading every year's record.
        if rec["n_granules"] > 0:
            try:
                daily_path = consolidate_year_snodas(year, year_dir, daily_dir)
                rec["daily_path"] = str(daily_path)
                rec["consolidated_utc"] = datetime.now(timezone.utc).isoformat()
            except (
                ValueError,
                FileNotFoundError,
                OSError,
                RuntimeError,
                tarfile.TarError,
            ) as exc:
                logger.warning(
                    "snodas: consolidation failed for year %d: %s",
                    year,
                    exc,
                )
                rec["consolidate_error"] = str(exc)
        else:
            logger.info(
                "snodas: year %d has no granules on disk; skipping consolidation.",
                year,
            )
        year_records.append(rec)

    _update_manifest(workdir, period, meta, year_records, archive_url)
    return _build_summary(
        meta, period, archive_url, worker_index, n_workers, year_records, now_utc
    )


def _fetch_year(session, archive_url: str, year: int, year_dir: Path) -> dict:
    """Download every available daily .tar for ``year`` into ``year_dir``.

    Returns a per-year record dict. Per-day 404s are common at year
    boundaries and not treated as errors. When ``n_errors > 0`` the
    manifest record also carries ``first_error_url`` / ``last_error_url``
    and a ``.failed_urls.txt`` sidecar lists every failed URL — both
    purely for operator diagnostics, no functional behaviour depends on
    them.
    """
    n_downloaded = 0
    n_already = 0
    n_404 = 0
    n_err = 0
    failed_urls: list[str] = []
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
            failed_urls.append(url)
    n_granules = n_downloaded + n_already
    rec: dict = {
        "year": year,
        "raw_dir": str(year_dir),
        "n_granules": n_granules,
        "n_downloaded_this_run": n_downloaded,
        "n_already_present": n_already,
        "n_missing_404": n_404,
        "n_errors": n_err,
    }
    if failed_urls:
        rec["first_error_url"] = failed_urls[0]
        rec["last_error_url"] = failed_urls[-1]
        # Sidecar makes the full list available for `curl -I`-style
        # triage without bloating the manifest. Overwritten each run so
        # the file reflects only the most recent run's failures (which
        # is what operators expect when investigating).
        sidecar = year_dir / ".failed_urls.txt"
        sidecar.write_text("\n".join(failed_urls) + "\n")
    return rec


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
    """Return the set of years recorded as fully downloaded AND consolidated.

    A year is considered complete only when its manifest record carries
    both ``n_granules > 0`` AND a ``daily_path`` that exists on disk.
    This dual check is the backfill mechanism: pre-existing manifest
    entries from the download-only fetch (PRs #100 / #103 / #106 / #108)
    have ``n_granules > 0`` but no ``daily_path``, so a re-run picks
    them up and only the new consolidation step runs (no re-download).

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
    completed: set[int] = set()
    for rec in entry.get("years", []):
        try:
            n_granules = int(rec.get("n_granules", 0))
        except (TypeError, ValueError):
            continue
        if n_granules <= 0:
            continue
        daily_path = rec.get("daily_path")
        if not daily_path or not Path(daily_path).exists():
            continue
        try:
            completed.add(int(rec["year"]))
        except (KeyError, TypeError, ValueError):
            continue
    return completed


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
