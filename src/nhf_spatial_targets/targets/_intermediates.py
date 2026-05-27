"""Year-chunked target intermediates: fingerprinting, pruning, stitching.

Used by the daily-cadence multi-year builders (SCA, SWE) where a single
in-memory build would materialise multi-decade datasets that exceed even
the large-memory partition. The year-chunked path bounds peak memory to
one year regardless of total period length, at the cost of an extra
stitch pass.

Three concerns are encapsulated here:

- :func:`iter_period_years` — drives the per-year build loop, clipping
  the first/last year boundaries to the user's period when they fall
  mid-year.
- Cache invalidation (issue #213): per-year intermediates carry
  ``config_fingerprint`` / ``code_version`` global attrs.
  :func:`should_skip_year_build` compares the cached attrs against
  :func:`target_config_fingerprint` / :func:`code_version_fingerprint`
  and auto-deletes stale intermediates so the operator does not have to
  ``rm`` after a config edit or version bump.
- Orphan pruning (issue #211): :func:`prune_orphan_year_intermediates`
  deletes per-year files whose year falls outside the active period, so
  a downward period change does not silently leak years into the stitched
  output. :func:`stitch_year_chunks_to_target` then builds the canonical
  single-file output via a column-streamed write.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import xarray as xr

from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)


def iter_period_years(
    period_start: str, period_end: str
) -> "list[tuple[int, str, str]]":
    """Iterate ``(year, year_start_iso, year_end_iso)`` over the period.

    The first and last yields are clipped to ``period_start`` / ``period_end``
    when those fall mid-year. Used by year-chunked target builders to drive
    the per-year build loop.

    Examples
    --------
    >>> iter_period_years("1980-06-15", "1982-03-20")
    [(1980, '1980-06-15', '1980-12-31'),
     (1981, '1981-01-01', '1981-12-31'),
     (1982, '1982-01-01', '1982-03-20')]
    """
    import pandas as pd

    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end)
    if end < start:
        raise ValueError(f"period end {period_end!r} precedes start {period_start!r}")
    out: list[tuple[int, str, str]] = []
    for year in range(start.year, end.year + 1):
        ys = max(start, pd.Timestamp(f"{year}-01-01"))
        ye = min(end, pd.Timestamp(f"{year}-12-31"))
        out.append((year, ys.strftime("%Y-%m-%d"), ye.strftime("%Y-%m-%d")))
    return out


# Names of global attrs written to every per-year intermediate so the skip
# branch can detect a stale cache. Kept as module-level constants so callers
# (sca.build, swe.build) can pass the values into ``extra_global_attrs`` and
# the on-disk schema is canonical.
INTERMEDIATE_CONFIG_FINGERPRINT_ATTR = "config_fingerprint"
INTERMEDIATE_CODE_VERSION_ATTR = "code_version"


def target_config_fingerprint(project: Project, target_name: str) -> str:
    """Stable hash of a target's config block + fabric identity.

    Combines the target's config block from ``project.config["targets"]``,
    the fabric path, and the fabric sha256 into a deterministic
    sha256-truncated-to-12-hex-chars fingerprint. Used by the year-chunked
    builders' skip predicate so that any change to the target's knobs
    (sources, periods, thresholds, etc.) or a fabric swap invalidates the
    on-disk per-year intermediates automatically.

    The fabric identity is included because a fabric swap changes the HRU
    set and CRS the target was built against — the same target config
    against a different fabric produces semantically different output that
    must not be silently reused.

    Returns
    -------
    A 12-character hex string. Collisions across realistic config-edit
    volumes are negligible.
    """
    target_cfg = project.config["targets"][target_name]
    fabric_cfg = project.config.get("fabric") or {}
    fabric_sha = (project.fabric or {}).get("sha256", "")
    payload = {
        "target": target_cfg,
        "fabric_path": fabric_cfg.get("path", ""),
        "fabric_sha256": fabric_sha,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def code_version_fingerprint() -> str:
    """The package ``__version__`` used as a code-version cache tag.

    Operators bumping ``__version__`` between runs get automatic
    invalidation; commits that change builder logic without a version
    bump do not. Mid-version edits therefore require a manual ``rm``
    of the intermediates directory — a known limitation documented in
    each year-chunked builder's module docstring.
    """
    from nhf_spatial_targets import __version__

    return __version__


def should_skip_year_build(
    paths: "list[Path]",
    *,
    active_config_fingerprint: str,
    active_code_version: str,
    target_label: str,
    year: int,
    logger: logging.Logger,
) -> "tuple[bool, dict | None]":
    """Decide whether to skip rebuilding a year given existing intermediates.

    Replaces the path-existence-only skip predicates that ``targets/sca.py``
    and ``targets/swe.py`` carried before issue #213. Compares the cached
    intermediate's ``config_fingerprint`` / ``code_version`` global attrs
    against the active values; on mismatch (or on missing-attr / corrupt-
    file), logs a WARNING naming both fingerprints, deletes the stale
    intermediate(s), and falls through to a rebuild.

    Parameters
    ----------
    paths
        All per-year intermediate paths the caller would expect to find on
        disk for a complete year build (the unfilled NC plus, when
        ``nn_fill`` is on, the ``_nn_filled.nc`` companion). The first
        path is treated as the primary attrs source; companions are read
        only via ``.exists()``.
    active_config_fingerprint
        Output of :func:`target_config_fingerprint` for the current run.
    active_code_version
        Output of :func:`code_version_fingerprint` for the current run.
    target_label
        Short string used in log messages (e.g. ``"sca"``, ``"swe"``).
    year
        Calendar year being built; used in log messages.
    logger
        Caller's module logger so warnings appear under the right name.

    Returns
    -------
    (skip, cached_attrs)
        Four distinct return shapes, each with a specific caller contract:

        - ``(False, None)`` — at least one expected path missing. No cache
          to examine; caller builds the year fresh.
        - ``(False, {})`` — cached intermediate exists but could not be
          opened (corrupt / truncated NetCDF). All paths unlinked; caller
          rebuilds.
        - ``(False, file_attrs)`` — cached intermediate opened cleanly but
          its fingerprint attrs were missing (pre-#213 file) or differed
          from the active values. All paths unlinked; caller rebuilds.
          ``file_attrs`` is the dict that was read off disk before the
          unlink, kept for forensic logging.
        - ``(True, file_attrs)`` — cache is valid; caller may skip the
          rebuild and use ``file_attrs`` to re-emit any per-year
          diagnostic warnings (e.g. low-coverage) so the operator sees
          them on every run, not just the first.

    Callers should defensively guard attr lookups with
    ``(cached_attrs or {}).get(...)`` to handle the corrupt-file shape
    without a special case.

    Treats a cached intermediate that is missing the fingerprint attrs
    as stale (pre-#213 intermediate) and triggers a rebuild — safe
    default for the one-time upgrade transition. The cleanup pass uses
    ``unlink(missing_ok=True)`` so a partial intermediate set (unfilled
    present, _nn_filled missing) does not raise.
    """
    if not all(p.exists() for p in paths):
        return False, None

    primary = paths[0]
    try:
        with xr.open_dataset(primary) as cached:
            cached_attrs = dict(cached.attrs)
    except (OSError, RuntimeError, ValueError) as exc:
        # netCDF4-backed xr.open_dataset on a corrupt/truncated file may
        # raise OSError (HDF5 layer), RuntimeError (libnetcdf bindings),
        # or ValueError (CF decode failure). All three are "the cache is
        # unusable" from this function's perspective — recovery is the
        # same: warn, unlink, let the caller rebuild.
        logger.warning(
            "%s year %d: cached intermediate %s could not be opened "
            "(%s); deleting and rebuilding.",
            target_label,
            year,
            primary,
            exc,
        )
        for p in paths:
            p.unlink(missing_ok=True)
        return False, {}

    cached_fp = cached_attrs.get(INTERMEDIATE_CONFIG_FINGERPRINT_ATTR)
    cached_ver = cached_attrs.get(INTERMEDIATE_CODE_VERSION_ATTR)
    # ``not`` (rather than ``is None``) also catches the empty-string case,
    # which a malformed upstream writer could produce — treat as missing
    # rather than equal-to-active-if-active-is-also-empty.
    if not cached_fp or not cached_ver:
        logger.warning(
            "%s year %d: cached intermediate %s predates issue #213 "
            "fingerprint support (missing %s/%s attrs); deleting and "
            "rebuilding.",
            target_label,
            year,
            primary,
            INTERMEDIATE_CONFIG_FINGERPRINT_ATTR,
            INTERMEDIATE_CODE_VERSION_ATTR,
        )
        for p in paths:
            p.unlink(missing_ok=True)
        return False, cached_attrs

    if cached_fp != active_config_fingerprint or cached_ver != active_code_version:
        logger.warning(
            "%s year %d: cache fingerprint mismatch — cached config=%s "
            "code=%s, active config=%s code=%s. Deleting stale "
            "intermediates and rebuilding.",
            target_label,
            year,
            cached_fp,
            cached_ver,
            active_config_fingerprint,
            active_code_version,
        )
        for p in paths:
            p.unlink(missing_ok=True)
        return False, cached_attrs

    return True, cached_attrs


# The 4-digit anchor is load-bearing — a 5-digit "year" must NOT match
# so that misnamed intermediates fall through to the stitch's
# computed-from-iter_period_years input list as the second defense layer
# (verified by test_stitch_input_computed_from_period_not_glob).
_YEAR_INTERMEDIATE_PATTERN = re.compile(r"_(\d{4})(?:_nn_filled)?\.nc$")


def prune_orphan_year_intermediates(
    intermediates_dir: Path,
    base: str,
    expected_years: "set[int]",
    *,
    target_label: str,
    logger: logging.Logger,
) -> "list[int]":
    """Delete per-year intermediate NCs whose year falls outside ``expected_years``.

    Globs both ``<base>_<YYYY>.nc`` and ``<base>_<YYYY>_nn_filled.nc``
    under ``intermediates_dir``, parses the 4-digit year from each
    filename, and unlinks files whose year is not in ``expected_years``.
    Mirrors the auto-invalidation pattern from
    :func:`should_skip_year_build` (#213): the operator sees a WARNING
    naming the years pruned but does not need to ``rm`` by hand.

    Used by year-chunked target builders (sca, swe) to defend the stitch
    step against orphan intermediates left behind by a previous build
    against a wider period (#211). Without this pass the stitcher's
    directory glob would silently include them, leaking out-of-period
    years into the final canonical NC. Files matching the glob but
    failing the strict year regex (e.g. operator-renamed debris) are
    logged at DEBUG and left untouched.

    Parameters
    ----------
    intermediates_dir
        ``<project>/targets/.<target>_intermediates/`` for the current
        build. Must exist (caller's responsibility).
    base
        Filename prefix matching the per-year writer's pattern, without
        the trailing ``_<YYYY>``. E.g. ``"sca_targets"`` matches
        ``sca_targets_2005.nc`` and ``sca_targets_2005_nn_filled.nc``.
    expected_years
        Set of calendar years the active config period covers. Files
        whose parsed year is NOT in this set are pruned.
    target_label
        Short string used in log messages (e.g. ``"sca"``, ``"swe"``).
    logger
        Caller's module logger so the WARNING appears under the right
        name.

    Returns
    -------
    Sorted list of years pruned. Empty when no orphans were found.
    """
    pruned: dict[int, list[Path]] = {}
    skipped: list[str] = []
    for path in intermediates_dir.glob(f"{base}_*.nc"):
        match = _YEAR_INTERMEDIATE_PATTERN.search(path.name)
        if not match:
            # Filename didn't match the canonical pattern — skip rather
            # than risk unlinking an unrelated file. Recorded for the
            # DEBUG breadcrumb so operators auditing the intermediates
            # dir know prune saw them and chose not to act.
            skipped.append(path.name)
            continue
        year = int(match.group(1))
        if year in expected_years:
            continue
        pruned.setdefault(year, []).append(path)

    if skipped:
        logger.debug(
            "%s: %d files in %s matched the glob but not the year "
            "regex; left untouched (names: %s)",
            target_label,
            len(skipped),
            intermediates_dir,
            ", ".join(sorted(skipped)),
        )

    if not pruned:
        # Audit breadcrumb — operators tailing the build log can confirm
        # the prune pass actually ran when nothing was found.
        logger.debug(
            "%s: no orphan year intermediates to prune (active period covers %d years)",
            target_label,
            len(expected_years),
        )
        return []

    for year_paths in pruned.values():
        for path in year_paths:
            path.unlink(missing_ok=True)

    pruned_years = sorted(pruned)
    logger.warning(
        "%s: pruned %d orphan year intermediates outside the active "
        "period (years: %s). Caused by a downward period change since "
        "the previous build; without pruning the stitched output would "
        "silently include these years. This is automatic recovery; no "
        "action required.",
        target_label,
        sum(len(v) for v in pruned.values()),
        ", ".join(str(y) for y in pruned_years),
    )
    return pruned_years


def stitch_year_chunks_to_target(
    intermediate_files: "list[Path]",
    output_path: Path,
    *,
    title: str,
    extra_global_attrs: dict | None,
    sort_dim: str,
    project: Project | None = None,
    lineage_kind: str = "target",
) -> None:
    """Lazily open per-year target NCs and stream them into one canonical NC.

    Streams as **full-time HRU columns**: the dataset is rechunked to
    ``{"time": -1, sort_dim: chunk_hru}`` so a single dask chunk spans the
    whole time axis for a block of HRUs. This matches the columnar on-disk
    layout that :func:`io_nc.build_encoding` writes (``(full_time, chunk_hru)``)
    — each on-disk chunk is filled in one pass, never re-compressed across year
    boundaries (issue #165 ST3b). Peak memory is one ~256 MiB column block, not
    the whole target.

    (This replaces the earlier year-slab streaming, which bounded memory the
    same way but produced a time-slab on-disk layout that made per-HRU
    calibration reads pull every time chunk.)

    Encoding is delegated to :func:`io_nc.build_encoding` (``layer="target"``):
    float32+zlib bounds, int8+zlib diagnostics, pinned ``proleptic_gregorian``
    time. Global attrs are taken from the first per-year file, then overridden
    by ``extra_global_attrs`` plus a fresh ``history`` line. The output is
    sorted ascending on ``sort_dim`` (issue #93) and written atomically via
    :func:`io_nc.atomic_to_netcdf`.

    Parameters
    ----------
    intermediate_files
        Per-year NC paths to stitch (already sorted by year via filename glob).
    output_path
        Final canonical NC path (e.g. ``<project>/targets/swe_targets.nc``).
    title
        Global ``title`` attr for the stitched file.
    extra_global_attrs
        Per-target metadata to overlay on the per-year files' global attrs.
    sort_dim
        Dimension to sort ascending on before write (typically
        ``project.id_col``); also the HRU chunk dimension.
    project
        Optional :class:`~nhf_spatial_targets.workspace.Project` used to
        append a lineage step to ``project.manifest_path`` after the
        stitch completes (release PR-B). When ``None`` no step is
        appended -- the pre-PR-B call shape is preserved so tests and
        external callers that don't have a project remain functional.
    lineage_kind
        Step ``kind`` for the appended record; the year-chunked driver
        passes ``"nn_fill"`` for the nn-filled stitch and ``"target"``
        (default) for the honest-NaN stitch.
    """
    from datetime import datetime, timezone

    from nhf_spatial_targets import __version__
    from nhf_spatial_targets.io_nc import atomic_to_netcdf, build_encoding
    from nhf_spatial_targets.targets._io import _read_chunk_hru

    if not intermediate_files:
        raise ValueError(
            "stitch_year_chunks_to_target: intermediate_files is empty; "
            "the year loop produced no per-year NCs."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.open_mfdataset(
        [str(p) for p in intermediate_files],
        combine="by_coords",
        # `join="exact"` mirrors the project's defensive-failure pattern
        # in `check_hru_coords`: a coord-set mismatch across per-year
        # files indicates per-year-build corruption (fabric drift,
        # weight-cache poisoning) and should fail loud, not silently
        # broadcast NaN on the union as `join="outer"` would.
        join="exact",
        chunks={},  # native per-file chunks; rechunked to columns below
        engine="netcdf4",
    )
    try:
        ds = ds.sortby(sort_dim)
        # Rechunk to full-time HRU columns so the columnar on-disk write fills
        # each chunk in one pass (no cross-year re-compression). chunk_hru here
        # (~256 MiB read block) is intentionally wider than the ~1 MiB on-disk
        # chunk: it bounds streaming memory while still spanning the full time
        # axis, so every on-disk chunk it covers is written exactly once.
        n_time = int(ds.sizes.get("time", 1))
        ds = ds.chunk({"time": -1, sort_dim: _read_chunk_hru(n_time, 4)})

        # `combine_attrs="override"` (the open_mfdataset default) keeps
        # the first file's attrs, which include per-year-only attrs
        # like ``year_chunk`` and ``frac_valid_bound`` (SCA's per-year
        # low-coverage diagnostic) that don't apply to the stitched
        # multi-year output. Strip them before the canonical attrs are
        # applied so the final NC reports its actual scope honestly —
        # otherwise an operator inspecting the stitched NC's attrs
        # would trust a single-year statistic as if it were
        # multi-year.
        for per_year_attr in ("year_chunk", "frac_valid_bound"):
            ds.attrs.pop(per_year_attr, None)
        ds.attrs.setdefault("Conventions", "CF-1.6")
        ds.attrs["title"] = title
        ds.attrs["history"] = (
            f"{datetime.now(timezone.utc).isoformat()} stitched from "
            f"{len(intermediate_files)} per-year NCs by "
            f"nhf_spatial_targets v{__version__}"
        )
        ds.attrs.setdefault("institution", "USGS")
        ds.attrs.setdefault("software_version", __version__)
        if extra_global_attrs:
            ds.attrs.update(extra_global_attrs)

        target_dtypes = {
            v: "float32" for v in ("lower_bound", "upper_bound") if v in ds.data_vars
        }
        target_dtypes.update(
            {v: "int8" for v in ("n_sources", "nn_filled") if v in ds.data_vars}
        )
        encoding = build_encoding(
            ds,
            layer="target",
            hru_dim=sort_dim,
            var_dtype=target_dtypes,
            timesteps_per_file=n_time,
        )
        expected_sizes = {"time": n_time, sort_dim: int(ds.sizes[sort_dim])}
        atomic_to_netcdf(ds, output_path, encoding=encoding)
    finally:
        ds.close()

    # Cheap post-write coverage check: a truncated or mis-shaped stream would
    # change the dim sizes. The per-year intermediates remain on disk, so a
    # loud failure here is recoverable (re-run regenerates the stitch).
    with xr.open_dataset(output_path) as _check:
        got_sizes = {
            "time": _check.sizes.get("time"),
            sort_dim: _check.sizes.get(sort_dim),
        }
    if got_sizes != expected_sizes:
        raise RuntimeError(
            f"stitched output {output_path} has dims {got_sizes}, expected "
            f"{expected_sizes}; the stitch did not write the full extent."
        )

    logger.info(
        "Stitched %d per-year NCs -> %s (%.1f MB)",
        len(intermediate_files),
        output_path,
        output_path.stat().st_size / 1e6,
    )

    if project is not None:
        # Release PR-B: the stitched NC is the canonical target file that
        # gets published. Record one lineage step per stitch with the per-
        # year intermediates as inputs (path+size+mtime, no sha256 -- they
        # are the year-chunked builder's own outputs and were fingerprinted
        # in their own ``kind=target`` steps already) and the stitched NC
        # as the output (with sha256, since this is what FGDC consumers
        # will hash-check).
        from nhf_spatial_targets.release.lineage import (
            append_step,
            input_file_entry,
            output_file_entry,
        )

        append_step(
            project.manifest_path,
            kind=lineage_kind,
            source_key=None,
            inputs=[input_file_entry(p) for p in intermediate_files if p.exists()],
            outputs=[output_file_entry(output_path)],
            params={
                "stitched_from": len(intermediate_files),
                "sort_dim": sort_dim,
            },
            command=f"stitch-{lineage_kind}",
        )
