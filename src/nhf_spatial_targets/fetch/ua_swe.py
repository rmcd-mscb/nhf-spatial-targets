"""Fetch UA Daily 4-km Gridded SWE and Snow Depth (NSIDC-0719) from NSIDC.

The Broxton/Zeng/Dawson University of Arizona product publishes one
NetCDF per water year at the NSIDC archive root:

    https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0719_SWE_Snow_Depth_v1/
        4km_SWE_Depth_WY<YYYY>_v01.nc       # one per water year, ~90 MB
        SWE_Mask_v01.nc                     # one CONUS/water/ice mask, ~2 MB

Each per-WY file carries float32 ``SWE`` (millimeters water equivalent)
and ``DEPTH`` (millimeters snow thickness) on a CONUS NAD83 (EPSG:4269)
lat-lon grid of 621 × 1405 (~0.04167° = ~4 km). Time is encoded as
float32 days since ``1900-01-01`` without a ``units`` attribute; the
companion ``time_str`` variable carries the same dates as
``dd-mmm-yyyy`` byte strings. Fill is ``NaN`` (the dataset uses no
integer sentinel), so a ``where(>=0)`` mask is unnecessary at consolidate
time — but the consolidator asserts ``min >= -1.0`` (raising on any
value below that) to catch a future format change to e.g. -9999 loudly.

Filenames are deterministic, so granule discovery is direct URL
construction by water year rather than CMR search. CMR may carry a
collection record for NSIDC-0719 but historically NSIDC's CMR records
for HTTPS-archive products have been granule-less stubs (issue #107
documented this for SNODAS/G02158); the deterministic-URL approach
sidesteps the question entirely.

After download, :func:`consolidate_water_year_ua_swe` decodes time,
renames the native ``SWE`` / ``DEPTH`` variables to the pipeline's
``swe`` / ``snow_depth`` (matching ``catalog/sources.yml``), and
**reprojects from NAD83 lat-lon to EPSG:5070 (NAD83 / CONUS Albers
Equal Area) at 4000 m using nearest-neighbour resampling**, mirroring
the SNODAS pre-projection (issue #121) so the aggregator's
``WEIGHT_GEN_CRS`` matches and gdptools' weight generation does not
need to reproject the source grid at all. Output is a single per-WY
CF-1.6 NetCDF at
``<datastore>/ua_swe/daily/ua_swe_daily_WY<YYYY>.nc``. Raw downloaded
NCs are preserved under ``<datastore>/ua_swe/raw/`` for provenance and
to enable re-decode after a catalog metadata change.

Backwards-compat note: ``catalog/sources.yml[ua_swe].period`` is the
calendar-year range ``1981/2023`` for compatibility with
:func:`fetch._period.parse_period`; the actual archive runs water years
1982-2023 (1981-10-01 through 2023-09-30). A user-requested calendar
year may touch one or two adjacent water-year files; the fetch module
resolves the set of WY files needed.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr
from rasterio.enums import Resampling

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

_SOURCE_KEY = "ua_swe"

# Per-request HTTP timeout (connect + first-byte). UA SWE per-WY files
# are ~90 MB compressed; 120 s is generous for a healthy network.
_HTTP_TIMEOUT_SECONDS = 120
_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
# Defense in depth against truncated bodies. A real per-WY file is
# 70-100 MB; anything below ~10 MB is a stub. Content-Length integrity
# is the primary defense; this is the fallback when the server omits
# the header.
_MIN_VALID_NC_BYTES = 10 * 1024 * 1024  # 10 MiB

# Filenames are deterministic at the NSIDC archive root.
_WY_FILENAME_TEMPLATE = "4km_SWE_Depth_WY{wy}_v01.nc"
_MASK_FILENAME = "SWE_Mask_v01.nc"
_CONSOLIDATED_FILENAME_TEMPLATE = "ua_swe_daily_WY{wy}.nc"

# Source-side variable names; renamed at consolidate time to match the
# lowercase pipeline convention (and the catalog `variables` block).
_NATIVE_SWE_VAR = "SWE"
_NATIVE_DEPTH_VAR = "DEPTH"
_DEST_SWE_VAR = "swe"
_DEST_DEPTH_VAR = "snow_depth"

# Time encoding in the native NetCDFs is float32 days since 1900-01-01
# without a `units` attribute; the consolidator decodes via this epoch
# explicitly. The companion `time_str` byte variable carries the same
# dates as `dd-mmm-yyyy` for cross-check, but we don't depend on it.
_SRC_TIME_EPOCH = pd.Timestamp("1900-01-01")

# Source CRS as advertised in the per-WY NCs' `crs` ancillary variable.
_SRC_CRS = "EPSG:4269"  # NAD83 geographic

# Pre-projection target — matches the aggregator's WEIGHT_GEN_CRS.
# Nearest-neighbour resampling preserves NaN fill without bleed.
_DST_CRS = "EPSG:5070"  # NAD83 / CONUS Albers
_DST_RESOLUTION_M = 4000.0


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------


def _calendar_years_to_water_years(years: list[int]) -> list[int]:
    """Map calendar years to the set of water years whose files contain them.

    Water year ``Y`` runs from ``Oct 1, Y-1`` through ``Sep 30, Y``. A
    calendar year ``C`` therefore touches:

    - WY ``C`` (covers Jan-Sep of CY ``C``)
    - WY ``C+1`` (covers Oct-Dec of CY ``C``)

    The returned list is sorted and deduplicated.
    """
    touched: set[int] = set()
    for cy in years:
        touched.add(cy)  # Jan-Sep of CY lives in WY = CY
        touched.add(cy + 1)  # Oct-Dec of CY lives in WY = CY+1
    return sorted(touched)


def _wy_url(archive_url: str, wy: int) -> str:
    """Construct the deterministic per-WY file URL at the NSIDC archive."""
    base = archive_url.rstrip("/")
    return f"{base}/{_WY_FILENAME_TEMPLATE.format(wy=wy)}"


def _mask_url(archive_url: str) -> str:
    """Construct the deterministic CONUS/water/ice mask URL."""
    base = archive_url.rstrip("/")
    return f"{base}/{_MASK_FILENAME}"


# ---------------------------------------------------------------------------
# HTTPS session + download
# ---------------------------------------------------------------------------


def _earthaccess_session():
    """Return an authenticated HTTPS session for daacdata.apps.nsidc.org.

    Wraps ``earthaccess.login(strategy='netrc').get_session()``. Callers
    are responsible for ensuring ``~/.netrc`` carries valid Earthdata
    credentials (the ``nhf-targets materialize-credentials`` command
    populates this from ``.credentials.yml``).
    """
    try:
        import earthaccess
    except ImportError as exc:  # pragma: no cover - import-time guarantee
        raise RuntimeError(
            "earthaccess is required for ua_swe fetch but is not installed."
        ) from exc

    auth = earthaccess.login(strategy="netrc")
    if auth is None or not getattr(auth, "authenticated", False):
        raise RuntimeError(
            "NASA Earthdata login failed. Run `nhf-targets "
            "materialize-credentials --project-dir <project>` to populate "
            "~/.netrc from your project's .credentials.yml, or visit "
            "https://urs.earthdata.nasa.gov/users/new to register."
        )
    return auth.get_session()


def _download_file(
    session,
    url: str,
    out_path: Path,
    *,
    timeout: int = _HTTP_TIMEOUT_SECONDS,
    min_bytes: int = _MIN_VALID_NC_BYTES,
) -> str:
    """Stream a single file from *url* to *out_path*. Returns a status code.

    Status codes (manifest-friendly):

    - ``"downloaded"`` — file streamed AND verified against Content-Length
      (or, if the server omits Content-Length, the body cleared
      ``min_bytes``) and written this call.
    - ``"already_present"`` — file exists on disk with size at or above
      ``min_bytes``; skipped.
    - ``"missing_404"`` — server returned 404; logged as a warning.
    - ``"error"`` — any other failure; logged, not raised.

    Atomic: streams to a ``.tmp`` sibling, then renames on success. The
    ``.tmp`` is unlinked on any failure path (including a Content-Length
    mismatch) so resume sees a clean directory.
    """
    if out_path.exists():
        sz = out_path.stat().st_size
        if sz >= min_bytes:
            return "already_present"
        logger.warning(
            "ua_swe: existing %s is suspiciously small (%d < %d bytes); redownloading.",
            out_path.name,
            sz,
            min_bytes,
        )
        out_path.unlink(missing_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        resp = session.get(url, timeout=timeout, stream=True, allow_redirects=True)
    except Exception as exc:
        logger.warning("ua_swe: GET failed for %s: %s", url, exc)
        tmp.unlink(missing_ok=True)
        return "error"
    try:
        if resp.status_code == 404:
            logger.warning("ua_swe: 404 for %s", url)
            return "missing_404"
        if resp.status_code != 200:
            logger.warning("ua_swe: unexpected status %d for %s", resp.status_code, url)
            return "error"

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

        if declared_bytes is not None:
            if bytes_written != declared_bytes:
                logger.warning(
                    "ua_swe: short read for %s — wrote %d of %d declared bytes",
                    url,
                    bytes_written,
                    declared_bytes,
                )
                tmp.unlink(missing_ok=True)
                return "error"
        elif bytes_written < min_bytes:
            logger.warning(
                "ua_swe: truncated response for %s — wrote %d bytes "
                "(< %d minimum and no Content-Length to confirm); discarding",
                url,
                bytes_written,
                min_bytes,
            )
            tmp.unlink(missing_ok=True)
            return "error"

        tmp.replace(out_path)
        return "downloaded"
    except Exception as exc:
        logger.warning("ua_swe: write failed for %s → %s: %s", url, out_path, exc)
        tmp.unlink(missing_ok=True)
        return "error"
    finally:
        try:
            resp.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def consolidate_water_year_ua_swe(wy: int, raw_path: Path, daily_dir: Path) -> Path:
    """Consolidate a per-WY raw NC into a CF-1.6 pre-projected daily NC.

    Steps:

    1. Open the raw file with ``decode_times=False`` (the source omits
       the ``time:units`` attribute, so xarray cannot auto-decode).
    2. Decode the ``time`` float32 axis as days since
       :data:`_SRC_TIME_EPOCH` (= 1900-01-01) into ``pd.Timestamp``.
    3. Build a new :class:`xr.Dataset` containing only the reprojected
       ``swe`` / ``snow_depth`` variables — the source's ``time_str``
       provenance variable and inline ``crs`` scalar are left behind by
       construction (not via ``drop_vars``); :func:`apply_cf_metadata`
       in step 7 writes a fresh grid-mapping ancillary.
    4. Rename ``SWE`` → ``swe``, ``DEPTH`` → ``snow_depth`` to match the
       catalog ``variables`` block.
    5. Attach the source CRS (NAD83 / EPSG:4269) via rioxarray.
    6. Reproject both 3-D variables to EPSG:5070 (NAD83 / CONUS Albers)
       at 4-km resolution using nearest-neighbour resampling
       (preserves NaN fill).
    7. Apply CF-1.6 metadata via
       :func:`fetch.consolidate.apply_cf_metadata` (sets variable
       ``units`` / ``long_name`` / ``cell_methods`` from the catalog
       and emits the EPSG:5070 grid-mapping ancillary).
    8. Atomic write via ``.nc.tmp`` rename.

    Idempotency: if the output NC exists and is newer than the raw NC,
    the rebuild is skipped.

    Parameters
    ----------
    wy : int
        Water year. The output filename embeds this label.
    raw_path : Path
        Source raw NC, e.g. ``<datastore>/ua_swe/raw/4km_SWE_Depth_WY1982_v01.nc``.
    daily_dir : Path
        Output directory; the file is written as
        ``ua_swe_daily_WY<wy>.nc`` inside.

    Returns
    -------
    Path
        Path to the consolidated NC.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"ua_swe raw NC not found: {raw_path}")
    daily_dir.mkdir(parents=True, exist_ok=True)
    out_path = daily_dir / _CONSOLIDATED_FILENAME_TEMPLATE.format(wy=wy)

    # Mtime-based idempotency. A rebuild is forced only when the raw
    # NC is newer than the consolidated output (e.g. after re-download).
    if out_path.exists() and out_path.stat().st_mtime >= raw_path.stat().st_mtime:
        logger.info(
            "ua_swe: WY %d already consolidated and current (%s); skipping.",
            wy,
            out_path.name,
        )
        return out_path

    # Per-day reproject pattern: keep peak memory bounded for environments
    # without a per-job SLURM allocation (login node cgroups, etc.). For a
    # 365-day WY on a 621x1405 source and ~1500x900 EPSG:5070 destination,
    # peak working set stays at a few hundred MB instead of the ~10 GB the
    # whole-year reproject can demand.
    ds_raw = xr.open_dataset(raw_path, decode_times=False)
    try:
        # Decode time axis manually (the source omits the `time:units`
        # attribute, so xarray cannot auto-decode).
        time_days = ds_raw["time"].values.astype("float64")
        times = _SRC_TIME_EPOCH + pd.to_timedelta(time_days, unit="D")
        n_time = ds_raw.sizes["time"]

        # Defensive: a real format change to integer sentinels (e.g.
        # -9999 added in a future v02) must fail loudly rather than
        # silently producing nonsense aggregates. Use chunked min across
        # time so we never materialise the full year in memory just to
        # check.
        for vname in (_NATIVE_SWE_VAR, _NATIVE_DEPTH_VAR):
            v = ds_raw[vname]
            running_min = np.inf
            for t in range(n_time):
                slab = v.isel(time=t).values
                m = np.nanmin(slab)
                if np.isfinite(m) and m < running_min:
                    running_min = float(m)
            if running_min < -1.0:
                raise ValueError(
                    f"ua_swe WY {wy}: variable {vname!r} contains values "
                    f"< -1.0 (min={running_min:.3f}); the publisher may "
                    f"have introduced an integer sentinel. Inspect the raw "
                    f"NC and update fetch/ua_swe.py before consolidating "
                    f"further."
                )

        # Reproject one day at a time. Each day is a small 2-D array
        # (~3.5 MB raw), so reprojection is cheap. The resulting
        # per-day DataArrays are accumulated into lists and concatenated
        # at the end; this trades a tiny accumulator-list footprint for
        # a much lower reproject-step peak.
        #
        # Lock the destination grid on the first day's reprojection and
        # force every subsequent day onto exactly that ``(transform,
        # shape)`` via ``shape=`` / ``transform=`` (rather than re-deriving
        # from ``resolution=``). This mirrors the SNODAS pattern in
        # ``_compute_dst_grid`` / ``_decode_and_reproject_day`` and
        # eliminates any chance of floating-point drift in the destination
        # bounds producing mismatched per-day shapes that would either
        # break the final ``xr.concat`` or (worse) silently broadcast.
        #
        # ``nodata=np.float32("nan")`` is pinned explicitly so a future
        # rioxarray default change can't quietly alter fill-handling on
        # the projected output. The publisher's NaN-fill convention is
        # preserved across reprojection.
        reprojected_per_var: dict[str, list[xr.DataArray]] = {
            _DEST_SWE_VAR: [],
            _DEST_DEPTH_VAR: [],
        }
        var_map = (
            (_NATIVE_SWE_VAR, _DEST_SWE_VAR),
            (_NATIVE_DEPTH_VAR, _DEST_DEPTH_VAR),
        )
        dst_transform = None
        dst_shape: tuple[int, int] | None = None
        nan_f32 = np.float32("nan")
        for t in range(n_time):
            for src_name, dst_name in var_map:
                da_day = ds_raw[src_name].isel(time=t)
                da_day = da_day.rio.set_spatial_dims(
                    x_dim="lon", y_dim="lat", inplace=False
                )
                da_day = da_day.rio.write_crs(_SRC_CRS, inplace=False)
                if dst_transform is None:
                    # First reprojection: derive (and capture) the
                    # destination grid from the requested resolution.
                    da_day_reproj = da_day.rio.reproject(
                        dst_crs=_DST_CRS,
                        resolution=(_DST_RESOLUTION_M, _DST_RESOLUTION_M),
                        resampling=Resampling.nearest,
                        nodata=nan_f32,
                    )
                    dst_transform = da_day_reproj.rio.transform()
                    dst_shape = (
                        int(da_day_reproj.sizes["y"]),
                        int(da_day_reproj.sizes["x"]),
                    )
                else:
                    # Subsequent reprojections: force onto the locked
                    # day-0 grid so all days are guaranteed-conformable.
                    da_day_reproj = da_day.rio.reproject(
                        dst_crs=_DST_CRS,
                        shape=dst_shape,
                        transform=dst_transform,
                        resampling=Resampling.nearest,
                        nodata=nan_f32,
                    )
                # Drop any inherited time-related scalar coords that
                # snuck through `isel(time=t)`; we re-attach the proper
                # time vector via `expand_dims` next.
                da_day_reproj = da_day_reproj.drop_vars(
                    [c for c in da_day_reproj.coords if c == "time"],
                    errors="ignore",
                )
                da_day_reproj = da_day_reproj.expand_dims(time=[times[t]])
                da_day_reproj.name = dst_name
                reprojected_per_var[dst_name].append(da_day_reproj)

        ds_reproj = xr.Dataset(
            {
                dst_name: xr.concat(days, dim="time")
                for dst_name, days in reprojected_per_var.items()
            }
        )
    finally:
        ds_raw.close()

    # Step 7: catalog-driven CF-1.6 metadata. `coord_type="projected"`
    # keeps the (y, x) dims as metres and emits the projected grid
    # mapping into the `crs` ancillary.
    ds_out = apply_cf_metadata(
        ds_reproj,
        _SOURCE_KEY,
        time_step="daily",
        coord_type="projected",
    )

    # Step 8: atomic write with explicit encoding (build_encoding(
    # layer="consolidated") is the issue #158 seam and currently
    # raises). Encoding mirrors the SNODAS consolidator: zlib 4, time
    # pinned to days since 1970-01-01 in a proleptic_gregorian
    # calendar so downstream tools agree on the epoch.
    encoding = _build_consolidated_encoding(ds_out)
    _atomic_write_dataset(ds_out, out_path, encoding=encoding)
    return out_path


def _build_consolidated_encoding(ds: xr.Dataset) -> dict:
    """Hand-rolled encoding for the consolidated NC (issue #158 workaround).

    ``io_nc.build_encoding(layer="consolidated", ...)`` currently
    raises ``NotImplementedError`` because the per-source spatial
    tiling policy is open work; until it lands, fetch modules carry
    their own encoding. The defaults here match the SNODAS
    consolidator: zlib level 4, integer shuffle off (we have float
    data), float32 ``_FillValue=NaN``, and a pinned time epoch.
    """
    encoding: dict[str, dict] = {}
    for vname in (_DEST_SWE_VAR, _DEST_DEPTH_VAR):
        if vname in ds.data_vars:
            encoding[vname] = {
                "zlib": True,
                "complevel": 4,
                "shuffle": False,
                "_FillValue": np.float32("nan"),
                "dtype": "float32",
            }
    encoding["time"] = {
        "units": "days since 1970-01-01",
        "calendar": "proleptic_gregorian",
        "dtype": "float64",
        # CF §9.6 strongly discourages _FillValue on coordinate variables
        # (coordinates should be exhaustive). xarray auto-attaches one
        # otherwise; setting None suppresses it.
        "_FillValue": None,
    }
    for coord in ("x", "y"):
        if coord in ds.coords:
            encoding[coord] = {"_FillValue": None}
    return encoding


def _atomic_write_dataset(ds: xr.Dataset, out_path: Path, encoding: dict) -> None:
    """Write *ds* to *out_path* via ``.nc.tmp`` rename. Cleans up on failure."""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        ds.to_netcdf(tmp, encoding=encoding, format="NETCDF4")
        tmp.replace(out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Manifest (read-merge-write, flock-protected)
# ---------------------------------------------------------------------------


def _update_manifest(
    workdir: Path,
    period: str,
    meta: dict,
    wy_records: list[dict],
    archive_url: str,
    mask_record: dict | None,
) -> None:
    """Merge ua_swe provenance into ``manifest.json`` under file lock.

    Read-merge-write semantics so concurrent worker invocations cannot
    clobber each other (mirrors the SNODAS pattern). Records for the
    same water year overwrite earlier entries on later calls.
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
        existing_by_wy = {int(y["water_year"]): y for y in entry.get("water_years", [])}
        for rec in wy_records:
            existing_by_wy[int(rec["water_year"])] = rec
        merged_wys = [existing_by_wy[w] for w in sorted(existing_by_wy)]
        access = meta["access"]
        entry.update(
            {
                "source_key": _SOURCE_KEY,
                "access_url": access["url"],
                "archive_url": archive_url,
                "doi": meta.get("doi"),
                "license": meta.get("license", "public domain (NASA NSIDC)"),
                "period": period,
                "variables": [v["name"] for v in meta["variables"]],
                "water_years": merged_wys,
                "fetched_by": f"nhf_spatial_targets {__version__}",
            }
        )
        if mask_record is not None:
            # Preserve any prior mask record if this run didn't touch it.
            entry["mask"] = mask_record
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
        with open(lock_path, "a") as lock_f:
            _fcntl.flock(lock_f, _fcntl.LOCK_EX)
            try:
                _do_update()
            finally:
                _fcntl.flock(lock_f, _fcntl.LOCK_UN)
    else:
        _do_update()
    logger.info(
        "ua_swe: updated manifest.json for water years %s",
        [r["water_year"] for r in wy_records],
    )


# ---------------------------------------------------------------------------
# Worker sharding
# ---------------------------------------------------------------------------


def _assign_worker_water_years(
    all_wys: list[int],
    worker_index: int,
    n_workers: int,
) -> list[int]:
    """Round-robin slice of ``all_wys`` for one worker.

    Mirrors :func:`snodas._assign_worker_years`. Slicing the full list
    (not a remaining-work set) keeps each worker's assignment
    independent of sibling progress, which is what manifests with
    concurrent file locks expect.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if not 0 <= worker_index < n_workers:
        raise ValueError(
            f"worker_index must be in [0, {n_workers}); got {worker_index}"
        )
    return list(all_wys[worker_index::n_workers])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fetch_ua_swe(
    workdir: Path,
    period: str,
    *,
    worker_index: int = 0,
    n_workers: int = 1,
) -> dict:
    """Download UA SWE per-WY NCs and consolidate to CF NetCDFs.

    Two-phase per water year: (1) stream
    ``4km_SWE_Depth_WY<YYYY>_v01.nc`` from the NSIDC HTTPS archive into
    ``<datastore>/ua_swe/raw/`` via the earthaccess auth session.
    (2) Decode, rename, pre-project to EPSG:5070, and write a
    CF-compliant per-WY NC at
    ``<datastore>/ua_swe/daily/ua_swe_daily_WY<YYYY>.nc``. The CONUS
    water/ice mask (``SWE_Mask_v01.nc``) is downloaded once at the
    datastore root and reused across runs.

    Parameters
    ----------
    workdir : Path
        Project directory.
    period : str
        Calendar-year window ``"YYYY/YYYY"`` (inclusive). The actual WY
        files needed are CY and CY+1 for each requested calendar year;
        the fetcher resolves the union internally. Validated against
        the catalog's ``period``.
    worker_index, n_workers : int
        Round-robin water-year sharding for parallel workers.

    Returns
    -------
    dict
        Provenance summary with the per-WY records appended.

    Raises
    ------
    ValueError
        Requested period falls outside the catalog window, or
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
            f"missing. Set it to the NSIDC HTTPS archive root."
        )
    data_lo, data_hi = period_bounds(meta["period"])
    requested_cys = years_in_period(period)
    for cy in requested_cys:
        if cy < data_lo or cy > data_hi:
            raise ValueError(
                f"Calendar year {cy} is outside the ua_swe publisher window "
                f"({data_lo}-{data_hi}, from catalog "
                f"`sources.yml[{_SOURCE_KEY}].period`). Adjust --period."
            )
    # Resolve calendar years to the water-year files that contain them.
    # For each WY needed, also clamp to the publisher's available WY
    # range. WY 1982 is the earliest published; WY 2023 the latest.
    candidate_wys = _calendar_years_to_water_years(requested_cys)
    publisher_wy_lo = data_lo + 1  # WY 1982 for catalog "1981/2023"
    publisher_wy_hi = data_hi  # WY 2023
    wys = [w for w in candidate_wys if publisher_wy_lo <= w <= publisher_wy_hi]
    if not wys:
        logger.warning(
            "ua_swe: requested calendar years %s map to no published water "
            "years in [%d, %d]; nothing to fetch.",
            requested_cys,
            publisher_wy_lo,
            publisher_wy_hi,
        )

    raw_dir = ws.raw_dir(_SOURCE_KEY) / "raw"
    daily_dir = ws.raw_dir(_SOURCE_KEY) / "daily"
    raw_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc).isoformat()

    session = _earthaccess_session()

    # Mask is downloaded once per datastore. Only worker 0 attempts it
    # to avoid concurrent writes to the same path; siblings just see
    # `already_present` on their next pass.
    mask_record: dict | None = None
    if worker_index == 0:
        mask_path = raw_dir / _MASK_FILENAME
        mask_status = _download_file(
            session,
            _mask_url(archive_url),
            mask_path,
            min_bytes=128 * 1024,  # mask is small (~2 MB); use a smaller floor
        )
        mask_record = {
            "filename": _MASK_FILENAME,
            "url": _mask_url(archive_url),
            "path": str(mask_path),
            "status": mask_status,
            "downloaded_utc": now_utc,
        }
        logger.info("ua_swe: CONUS mask status=%s (%s)", mask_status, mask_path)

    assigned = _assign_worker_water_years(wys, worker_index, n_workers)
    if not assigned:
        logger.info(
            "ua_swe: worker %d/%d has no water years to process for period %s",
            worker_index,
            n_workers,
            period,
        )
        return {
            "source_key": _SOURCE_KEY,
            "period": period,
            "archive_url": archive_url,
            "worker_index": worker_index,
            "n_workers": n_workers,
            "water_years": [],
            "fetched_at": now_utc,
            "mask": mask_record,
        }

    wy_records: list[dict] = []
    for wy in assigned:
        url = _wy_url(archive_url, wy)
        raw_path = raw_dir / _WY_FILENAME_TEMPLATE.format(wy=wy)
        status = _download_file(session, url, raw_path)
        rec: dict = {
            "water_year": wy,
            "url": url,
            "raw_path": str(raw_path),
            "status": status,
            "downloaded_utc": now_utc,
        }
        if status in ("downloaded", "already_present"):
            # Data-level failures are recorded and continue; programming
            # errors are not caught.
            try:
                daily_path = consolidate_water_year_ua_swe(wy, raw_path, daily_dir)
                rec["daily_path"] = str(daily_path)
                rec["consolidated_utc"] = datetime.now(timezone.utc).isoformat()
            except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
                logger.warning("ua_swe: consolidation failed for WY %d: %s", wy, exc)
                rec["consolidate_error"] = str(exc)
        elif status == "missing_404":
            logger.warning(
                "ua_swe: WY %d returned 404 at %s; skipping consolidation. "
                "Confirm the catalog `archive_url` and the v01 filename "
                "convention if this persists.",
                wy,
                url,
            )
        else:  # "error"
            logger.warning(
                "ua_swe: WY %d download failed (%s); skipping consolidation.",
                wy,
                status,
            )
        wy_records.append(rec)
        logger.info(
            "ua_swe: WY %d — status=%s%s",
            wy,
            status,
            (f", daily={Path(rec['daily_path']).name}" if "daily_path" in rec else ""),
        )

    _update_manifest(workdir, period, meta, wy_records, archive_url, mask_record)
    return {
        "source_key": _SOURCE_KEY,
        "period": period,
        "archive_url": archive_url,
        "worker_index": worker_index,
        "n_workers": n_workers,
        "water_years": wy_records,
        "fetched_at": now_utc,
        "mask": mask_record,
    }
