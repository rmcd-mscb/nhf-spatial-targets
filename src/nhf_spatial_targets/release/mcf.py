"""Build pygeometa-shaped MCF dicts for the ScienceBase data release.

An MCF (Metadata Control File) dict is the single intermediate
representation rendered twice in a later phase: to FGDC CSDGM 2.0 XML via
Jinja2 templates, and to ISO 19139 XML via
``pygeometa.schemas.iso19139``. Building one dict and rendering it two ways
keeps FGDC and ISO from drifting.

The dicts produced here conform to pygeometa's core MCF schema (``mcf``,
``metadata``, ``spatial``, ``identification``, ``contact``,
``distribution``, ``dataquality``) so the pygeometa ISO emitter consumes
them directly. They additionally carry a few FGDC-flavored keys that
pygeometa ignores but the FGDC Jinja templates read:

- ``identification.purpose`` -- FGDC ``<purpose>``
- ``identification.useconstraints`` -- FGDC ``<useconst>`` (richer than the
  pygeometa-native ``identification.rights`` that ISO reads)
- ``identification.doi`` -- the DOI URL for the citation ``<onlink>``
- ``identification.citation`` -- ``{authors, doi}`` for ``<citeinfo>``
- ``identification.entityandattribute`` -- per-variable / per-target
  ``<eainfo>`` records
- ``dataquality.lineage.processstep`` -- ``<procstep>`` blocks built from
  ``manifest.json.steps[]``
- ``distribution_liability`` (top level) -- FGDC ``<distliab>`` boilerplate

The builders are **pure**: every input is passed in (the populated
``catalog/release_defaults.yml`` via :mod:`release.defaults`, the per-source
catalog entries via :mod:`catalog`, the loaded ``manifest.json`` /
``fabric.json`` / project ``config.yml`` dicts). They do no disk walking and
no registry I/O -- the build orchestrator (a later phase) supplies
registry-derived values (DOI, version, staged file list) as keyword
arguments. A ``now``
argument is injectable so golden tests are byte-stable, mirroring
:func:`nhf_spatial_targets.release.lineage.append_step`.

The module never enumerates a hard-coded source list: source children flow
from :func:`catalog.publishable_sources`, fabric participation from the
project ``manifest.json.sources`` intersected with the same filter, so a new
dataset appears with zero edits here.
"""

from __future__ import annotations

from datetime import datetime, timezone

import jinja2

from nhf_spatial_targets import catalog

# pygeometa core MCF format version (schemas/mcf/core.yaml `mcf.version`).
MCF_VERSION = 2.0

# Structural spatial-representation hints per item type. These describe how
# the artifacts encode geography (gridded NetCDF vs HRU-polygon vector); they
# are intrinsic to the pipeline's output, not catalog metadata.
_SPATIAL = {
    "umbrella": {"datatype": "grid", "geomtype": "surface"},
    "source": {"datatype": "grid", "geomtype": "surface"},
    "fabric": {"datatype": "vector", "geomtype": "surface"},
}

# Cat-Interop function codes for a distribution entry (pygeometa enum).
_FN_DOWNLOAD = "download"
_FN_INFO = "information"

# license-substring -> attribution-snippet key in
# release_defaults.distribution.attribution_snippets. Keyed on the source's
# catalog `license` text so attribution flows by data, not by hard-coded
# source key (era5_land's "Copernicus license ..." selects "copernicus").
# Order matters: first match wins. "ornl" precedes "nasa" so daymet
# ("public domain (NASA / ORNL DAAC)") is credited to ORNL DAAC, not the
# NASA Earthdata system; "nsidc" precedes "nasa" so NSIDC-distributed NASA
# products (margulis_wus_sr) are credited to NSIDC.
_LICENSE_SNIPPETS: tuple[tuple[str, str], ...] = (
    ("copernicus", "copernicus"),
    ("ornl", "ornl"),
    ("nsidc", "nsidc"),
    ("nasa", "earthdata"),
)

_JINJA = jinja2.Environment(autoescape=False, undefined=jinja2.Undefined)


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _now_dt(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(timezone.utc)


def _collapse_ws(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and strip ends.

    Folded YAML scalars plus inline Jinja ``{% if %}`` blocks leave ragged
    whitespace; normalizing here keeps rendered abstracts/titles stable and
    readable in golden files regardless of template line wrapping.
    """
    return " ".join(text.split())


def _render(template_str: str, /, **ctx: object) -> str:
    """Render a Jinja2 template string, whitespace-collapsed."""
    return _collapse_ws(_JINJA.from_string(template_str).render(**ctx))


def _period_bound(token: str, *, end: bool) -> str | None:
    """Map one side of a ``"YYYY/YYYY"``-style period token to an ISO date.

    ``"present"`` (open-ended) -> ``None``. A bare 4-digit year expands to
    Jan 1 (begin) or Dec 31 (end). Anything already date-shaped is returned
    as-is.
    """
    token = token.strip()
    if not token or token.lower() == "present":
        return None
    if len(token) == 4 and token.isdigit():
        return f"{token}-12-31" if end else f"{token}-01-01"
    return token


def _parse_period(period: str | None) -> tuple[str | None, str | None]:
    if not period or "/" not in period:
        return (None, None)
    lo, hi = period.split("/", 1)
    return (_period_bound(lo, end=False), _period_bound(hi, end=True))


def _union_bbox(bboxes: list[list[float] | None]) -> list[float] | None:
    """Union a list of ``[west, south, east, north]`` boxes (None-skipping)."""
    valid = [b for b in bboxes if b]
    if not valid:
        return None
    return [
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    ]


def _union_temporal(
    spans: list[tuple[str | None, str | None]],
) -> tuple[str | None, str | None]:
    """Union (begin, end) spans. Any open end (``None``) makes the union open."""
    begins = [b for b, _ in spans if b]
    begin = min(begins) if begins else None
    has_open_end = any(b is not None and e is None for b, e in spans)
    ends = [e for _, e in spans if e]
    end = None if has_open_end else (max(ends) if ends else None)
    return begin, end


def _bbox_from_nwse(nwse: list[float] | None) -> list[float] | None:
    """Convert a catalog ``bbox_nwse`` ``[N, W, S, E]`` to ``[W, S, E, N]``.

    The catalog stores Copernicus-convention north/west/south/east; FGDC and
    pygeometa want west/south/east/north (xmin, ymin, xmax, ymax).
    """
    if not nwse or len(nwse) != 4:
        return None
    n, w, s, e = nwse
    return [float(w), float(s), float(e), float(n)]


# ---------------------------------------------------------------------------
# Shared block builders
# ---------------------------------------------------------------------------


def _contact_block(defaults: dict) -> dict:
    """Build the MCF ``contact`` block from ``release_defaults.contacts``.

    The default contact entries already use pygeometa field names; only the
    ``point_of_contact`` -> ``pointOfContact`` role key is remapped.
    """
    contacts = defaults.get("contacts", {})
    block: dict = {}
    if contacts.get("point_of_contact"):
        block["pointOfContact"] = dict(contacts["point_of_contact"])
    if contacts.get("distributor"):
        block["distributor"] = dict(contacts["distributor"])
    return block


def _distribution_entry(
    *, url: str, name: str, description: str, function: str, protocol: str
) -> dict:
    return {
        "url": url,
        "type": protocol,
        "name": name,
        "description": description,
        "function": function,
    }


def _process_steps(steps: list[dict] | None, *, source_key: str | None) -> list[dict]:
    """Project ``manifest.json.steps[]`` to FGDC ``<procstep>`` records.

    For a source child (``source_key`` given) only steps tagged with that
    key are kept; for a fabric child (``source_key`` None) all steps are
    kept. Each record carries the CSDGM-ordered fields the FGDC template
    emits: ``procdesc``, ``procdate`` (YYYYMMDD), ``proctime`` (HHMMSS),
    ``srcused`` (input basenames), ``srcprod`` (output basenames).
    """
    out: list[dict] = []
    for step in steps or []:
        if source_key is not None and step.get("source_key") != source_key:
            continue
        ts = step.get("timestamp_utc", "")
        procdate, proctime = "", ""
        if "T" in ts:
            date_part, _, time_part = ts.partition("T")
            procdate = date_part.replace("-", "")
            proctime = time_part[:8].replace(":", "")
        out.append(
            {
                "procdesc": _step_description(step),
                "procdate": procdate,
                "proctime": proctime,
                "srcused": _basenames(step.get("inputs")),
                "srcprod": _basenames(step.get("outputs")),
            }
        )
    return out


def _step_description(step: dict) -> str:
    parts = [
        f"{step.get('kind', 'step')} via {step.get('tool', 'nhf-targets')} "
        f"{step.get('software_version', '')}".strip()
    ]
    if step.get("command"):
        parts.append(f"command: {step['command']}")
    params = step.get("params") or {}
    if params:
        rendered = ", ".join(f"{k}={params[k]}" for k in sorted(params))
        parts.append(f"params: {rendered}")
    return "; ".join(parts)


def _basenames(entries: list[dict] | None) -> list[str]:
    names = set()
    for entry in entries or []:
        path = entry.get("path")
        if path:
            names.add(path.rsplit("/", 1)[-1])
    return sorted(names)


# ---------------------------------------------------------------------------
# Source-child MCF
# ---------------------------------------------------------------------------


def build_source_mcf(
    source_key: str,
    *,
    defaults: dict,
    manifest: dict | None = None,
    distribution_files: list[str] | None = None,
    identifier: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Build the MCF dict for one consolidated-source child item.

    Reads the source's catalog entry (name, description, citations, license,
    period, bbox, variables) and ``release`` block, plus repo-level
    ``defaults``. ``manifest`` (loaded ``manifest.json``) supplies the
    process steps filtered to ``source_key``. ``distribution_files`` are the
    staged consolidated NetCDF filenames (passed by the build orchestrator;
    ``None`` pre-build yields only the upstream landing entry).
    ``metadata_only`` sources
    (e.g. daymet) carry only the upstream landing entry regardless.

    Parameters
    ----------
    source_key
        Catalog key (e.g. ``"era5_land"``).
    defaults
        Validated ``catalog/release_defaults.yml`` mapping.
    manifest, distribution_files, identifier, now
        See module docstring. ``identifier`` overrides the deterministic
        ``urn:usgs:nhf-release:source:<key>`` (the build orchestrator passes
        the ScienceBase id).
    """
    src = catalog.source(source_key)
    release = catalog.release_block(source_key)
    dist_defaults = defaults.get("distribution", {})
    build_date = _now_dt(now).date().isoformat()

    title = f"{src.get('name', source_key)} (consolidated for NHF)"
    abstract = _collapse_ws(
        f"{src.get('description', '').strip()} "
        f"{defaults.get('source', {}).get('abstract_suffix', '').strip()}"
    )
    purpose = _render(
        defaults.get("source", {}).get("purpose_template", ""),
        source_name=src.get("name", source_key),
    )

    bbox = _bbox_from_nwse(src.get("access", {}).get("bbox_nwse"))
    begin, end = _parse_period(src.get("period"))
    upstream_url = src.get("access", {}).get("url") or dist_defaults.get(
        "fallback_landing_url", ""
    )
    doi = _doi_url(src.get("doi") or src.get("access", {}).get("doi"))

    identification = _identification(
        defaults=defaults,
        scope_keywords=defaults.get("keywords", {}).get("source", {}),
        extra_theme=[
            v["long_name"]
            for v in src.get("variables", [])
            if isinstance(v, dict) and v.get("long_name")
        ],
        place_keywords=_place_keywords(src.get("spatial_extent")),
        title=title,
        abstract=abstract,
        purpose=purpose,
        bbox=bbox,
        temporal=(begin, end),
        url=doi or upstream_url,
        doi=doi,
        useconstraints=_source_useconstraints(src, release, defaults),
        citation={"authors": list(src.get("citations", [])), "doi": doi},
        entityandattribute=_source_eainfo(src),
        build_date=build_date,
    )

    distribution = _source_distribution(
        release=release,
        upstream_url=upstream_url,
        files=distribution_files,
        protocol=dist_defaults.get("distribution_protocol", "WWW:LINK"),
    )

    lineage_stmt = _render(
        defaults.get("source", {}).get("lineage_statement_template", ""),
        source_name=src.get("name", source_key),
    )

    return _assemble(
        identifier=identifier or f"urn:usgs:nhf-release:source:{source_key}",
        hierarchylevel="dataset",
        spatial=_SPATIAL["source"],
        defaults=defaults,
        identification=identification,
        distribution=distribution,
        lineage_statement=lineage_stmt,
        process_steps=_process_steps(
            manifest.get("steps") if manifest else None, source_key=source_key
        ),
        scope_level="dataset",
        build_date=build_date,
    )


def _source_useconstraints(src: dict, release: dict, defaults: dict) -> str:
    parts = [defaults.get("distribution", {}).get("use_constraints", "").strip()]
    if src.get("license"):
        parts.append(f"Source dataset license: {src['license']}.")
    if release.get("notes"):
        parts.append(_collapse_ws(release["notes"]))
    snippet = _attribution_snippet(src.get("license", ""), defaults)
    if snippet:
        parts.append(snippet)
    return "\n\n".join(p for p in parts if p)


def _attribution_snippet(license_text: str, defaults: dict) -> str | None:
    snippets = defaults.get("distribution", {}).get("attribution_snippets", {})
    lic = (license_text or "").lower()
    for needle, key in _LICENSE_SNIPPETS:
        if needle in lic and snippets.get(key):
            return _collapse_ws(snippets[key])
    return None


def _source_eainfo(src: dict) -> list[dict]:
    out = []
    for var in src.get("variables", []):
        if not isinstance(var, dict):
            continue
        out.append(
            {
                "name": var.get("name", ""),
                "definition": var.get("long_name", ""),
                "units": var.get("cf_units") or var.get("units"),
                "cell_methods": var.get("cell_methods"),
            }
        )
    return out


def _source_distribution(
    *, release: dict, upstream_url: str, files: list[str] | None, protocol: str
) -> dict:
    """One landing entry (upstream) plus one download entry per staged file.

    ``metadata_only`` sources (daymet) carry only the upstream landing entry;
    their file payload lives at the upstream archive, not in the release.
    """
    block: dict = {
        "upstream": _distribution_entry(
            url=upstream_url,
            name="Upstream source archive",
            description="Original data provider's landing page.",
            function=_FN_INFO,
            protocol="WWW:LINK",
        )
    }
    if release.get("distribution_kind") == "metadata_only":
        return block
    for name in files or []:
        block[name] = _distribution_entry(
            url=name,
            name=name,
            description="Consolidated CF-1.6 NetCDF.",
            function=_FN_DOWNLOAD,
            protocol=protocol,
        )
    return block


# ---------------------------------------------------------------------------
# Fabric-child MCF
# ---------------------------------------------------------------------------


def build_fabric_mcf(
    *,
    config: dict,
    fabric: dict,
    defaults: dict,
    manifest: dict | None = None,
    distribution_files: list[str] | None = None,
    doi: str | None = None,
    identifier: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Build the MCF dict for one fabric child item.

    ``config`` is the merged project config (``apply_defaults`` applied);
    ``fabric`` is the loaded ``fabric.json``. Enabled targets come from
    ``config['targets']``; participating sources from
    ``manifest['sources']`` intersected with
    :func:`catalog.publishable_sources` (catalog/manifest-driven, never a
    hard-coded list). ``doi`` is the umbrella DOI children share (``None``
    pre-mint); ``distribution_files`` are the staged file names (supplied by
    the build orchestrator).
    """
    release_cfg = config.get("release") or {}
    fab_defaults = defaults.get("fabric", {})
    dist_defaults = defaults.get("distribution", {})
    build_date = _now_dt(now).date().isoformat()

    label = _fabric_label(config, fabric)
    enabled = _enabled_targets(config)
    target_names = [name for name, _ in enabled]
    source_names = _participating_sources(manifest)
    doi_url = _doi_url(doi)

    title = _render(fab_defaults.get("title_template", ""), fabric_label=label)
    abstract = _render(
        fab_defaults.get("abstract_template", ""),
        fabric_label=label,
        hru_count=fabric.get("hru_count"),
        target_names=target_names,
        source_names=source_names,
        abstract_notes=release_cfg.get("abstract_notes"),
    )
    purpose = _render(fab_defaults.get("purpose_template", ""), fabric_label=label)
    lineage_stmt = _render(
        fab_defaults.get("lineage_statement_template", ""), fabric_label=label
    )

    bbox = _fabric_bbox(fabric)
    temporal = _union_temporal([_parse_period(p) for _, p in enabled])

    identification = _identification(
        defaults=defaults,
        scope_keywords=defaults.get("keywords", {}).get("fabric", {}),
        extra_theme=target_names,
        place_keywords=[label],
        title=title,
        abstract=abstract,
        purpose=purpose,
        bbox=bbox,
        temporal=temporal,
        url=doi_url or dist_defaults.get("fallback_landing_url", ""),
        doi=doi_url,
        useconstraints=_fabric_useconstraints(source_names, defaults),
        citation={"authors": list(release_cfg.get("authors", [])), "doi": doi_url},
        entityandattribute=_fabric_eainfo(target_names),
        build_date=build_date,
    )

    distribution = _fabric_distribution(
        files=distribution_files,
        landing_url=doi_url or dist_defaults.get("fallback_landing_url", ""),
        protocol=dist_defaults.get("distribution_protocol", "WWW:LINK"),
    )

    return _assemble(
        identifier=identifier or f"urn:usgs:nhf-release:fabric:{label}",
        hierarchylevel="dataset",
        spatial=_SPATIAL["fabric"],
        defaults=defaults,
        identification=identification,
        distribution=distribution,
        lineage_statement=lineage_stmt,
        process_steps=_process_steps(
            manifest.get("steps") if manifest else None, source_key=None
        ),
        scope_level="dataset",
        build_date=build_date,
    )


def _fabric_label(config: dict, fabric: dict) -> str:
    label = (config.get("release") or {}).get("fabric_label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    path = (config.get("fabric") or {}).get("path") or fabric.get("path") or ""
    stem = path.rsplit("/", 1)[-1]
    return stem.rsplit(".", 1)[0] if "." in stem else stem


def _enabled_targets(config: dict) -> list[tuple[str, str | None]]:
    out = []
    for name, cfg in (config.get("targets") or {}).items():
        if isinstance(cfg, dict) and cfg.get("enabled"):
            out.append((name, cfg.get("period")))
    return out


def _participating_sources(manifest: dict | None) -> list[str]:
    # Manifest-driven (provenance of record), in contrast to the disk-driven
    # payload.sources_used (which aggregated/ subdirs are actually staged).
    # The two can disagree if the manifest lists a source whose aggregated NCs
    # are no longer on disk; the build orchestrator is responsible for
    # reconciling the metadata source list with the staged payload.
    if not manifest:
        return []
    publishable = catalog.publishable_sources()
    return sorted(k for k in (manifest.get("sources") or {}) if k in publishable)


def _fabric_bbox(fabric: dict) -> list[float] | None:
    # None means "no fabric extent supplied"; a present-but-malformed bbox is
    # a corrupted fabric.json (validate.py always writes the four float keys),
    # so raise loudly rather than silently shrinking a published bounding box.
    bbox = fabric.get("bbox")
    if bbox is None:
        return None
    if not isinstance(bbox, dict) or not {"minx", "miny", "maxx", "maxy"} <= set(bbox):
        raise ValueError(f"fabric.json bbox is malformed: {bbox!r}")
    return [
        float(bbox["minx"]),
        float(bbox["miny"]),
        float(bbox["maxx"]),
        float(bbox["maxy"]),
    ]


def _fabric_useconstraints(source_names: list[str], defaults: dict) -> str:
    parts = [defaults.get("distribution", {}).get("use_constraints", "").strip()]
    licenses = []
    for key in source_names:
        lic = catalog.source(key).get("license")
        if lic and lic not in licenses:
            licenses.append(lic)
    if licenses:
        parts.append(
            "Participating source dataset licenses: " + "; ".join(licenses) + "."
        )
    seen: set[str] = set()
    for key in source_names:
        snippet = _attribution_snippet(catalog.source(key).get("license", ""), defaults)
        if snippet and snippet not in seen:
            seen.add(snippet)
            parts.append(snippet)
    return "\n\n".join(p for p in parts if p)


def _fabric_eainfo(target_names: list[str]) -> list[dict]:
    out = []
    for name in target_names:
        try:
            var = catalog.variable(name)
        except KeyError:
            continue
        out.append(
            {
                "name": name,
                "definition": _collapse_ws(var.get("description", "")),
                "units": var.get("units"),
                "range_method": var.get("range_method"),
                "range_notes": _collapse_ws(var.get("range_notes", "")) or None,
            }
        )
    return out


def _fabric_distribution(
    *, files: list[str] | None, landing_url: str, protocol: str
) -> dict:
    block: dict = {
        "landing": _distribution_entry(
            url=landing_url,
            name="ScienceBase landing page",
            description="Data release landing page.",
            function=_FN_INFO,
            protocol="WWW:LINK",
        )
    }
    for name in files or []:
        block[name] = _distribution_entry(
            url=name,
            name=name,
            description=_file_description(name),
            function=_FN_DOWNLOAD,
            protocol=protocol,
        )
    return block


def _file_description(name: str) -> str:
    if name.endswith(".gpkg"):
        return "Hydrologic fabric GeoPackage."
    if name.startswith("aggregated/"):
        return "Source dataset area-weighted to the fabric (CF-1.6 NetCDF)."
    if name.startswith("targets/"):
        return "Calibration-target dataset (CF-1.6 NetCDF)."
    if name == "manifest.json":
        return "Provenance manifest."
    if name.endswith(".csv") or name == "SHA256SUMS":
        return "Per-file checksums."
    if name.endswith(".md"):
        return "Human-readable README."
    return "Release file."


# ---------------------------------------------------------------------------
# Umbrella MCF
# ---------------------------------------------------------------------------


def build_umbrella_mcf(
    *,
    defaults: dict,
    children: list[dict] | None = None,
    authors: list[dict] | None = None,
    doi: str | None = None,
    version: str | None = None,
    identifier: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Build the metadata-only umbrella (DOI parent / series) MCF dict.

    ``children`` is the list of already-built child MCF dicts (source +
    fabric); the umbrella unions their bboxes and temporal extents and lists
    their titles in the abstract. ``authors`` is the project release-author
    list; ``version`` defaults to ``"1.0"`` and ``doi`` is the minted DOI
    (``None`` pre-mint). All registry-derived values are passed in -- this
    builder does no registry I/O.
    """
    children = children or []
    umb_defaults = defaults.get("umbrella", {})
    dist_defaults = defaults.get("distribution", {})
    build_date = _now_dt(now).date().isoformat()
    version = version or "1.0"
    doi_url = _doi_url(doi)

    child_titles = [
        c.get("identification", {}).get("title", "")
        for c in children
        if c.get("identification", {}).get("title")
    ]
    bbox = _union_bbox([_child_bbox(c) for c in children])
    temporal = _union_temporal([_child_temporal(c) for c in children])

    title = _render(umb_defaults.get("title_template", ""), version=version)
    abstract = _render(
        umb_defaults.get("abstract_template", ""),
        version=version,
        n_children=len(children),
        child_titles=child_titles,
    )

    identification = _identification(
        defaults=defaults,
        scope_keywords=defaults.get("keywords", {}).get("umbrella", {}),
        extra_theme=[],
        place_keywords=defaults.get("keywords", {})
        .get("umbrella", {})
        .get("place", []),
        title=title,
        abstract=abstract,
        purpose=_collapse_ws(umb_defaults.get("purpose", "")),
        bbox=bbox,
        temporal=temporal,
        url=doi_url or dist_defaults.get("fallback_landing_url", ""),
        doi=doi_url,
        useconstraints=_collapse_ws(dist_defaults.get("use_constraints", "")),
        citation={"authors": list(authors or []), "doi": doi_url},
        entityandattribute=[],
        build_date=build_date,
        revision_date=build_date,
    )

    distribution = {
        "landing": _distribution_entry(
            url=doi_url or dist_defaults.get("fallback_landing_url", ""),
            name="ScienceBase landing page",
            description="Data release landing page (DOI).",
            function=_FN_INFO,
            protocol="WWW:LINK",
        )
    }

    return _assemble(
        identifier=identifier or "urn:usgs:nhf-release:umbrella",
        hierarchylevel="series",
        spatial=_SPATIAL["umbrella"],
        defaults=defaults,
        identification=identification,
        distribution=distribution,
        lineage_statement=_collapse_ws(umb_defaults.get("lineage_statement", "")),
        process_steps=[],
        scope_level="series",
        build_date=build_date,
        versioning_policy=_collapse_ws(umb_defaults.get("versioning_policy", "")),
        version=version,
    )


def _child_bbox(child: dict) -> list[float] | None:
    extents = child.get("identification", {}).get("extents", {})
    spatial = extents.get("spatial") or [{}]
    return spatial[0].get("bbox")


def _child_temporal(child: dict) -> tuple[str | None, str | None]:
    extents = child.get("identification", {}).get("extents", {})
    temporal = extents.get("temporal") or [{}]
    span = temporal[0]
    return span.get("begin"), span.get("end")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _identification(
    *,
    defaults: dict,
    scope_keywords: dict,
    extra_theme: list[str],
    place_keywords: list[str],
    title: str,
    abstract: str,
    purpose: str,
    bbox: list[float] | None,
    temporal: tuple[str | None, str | None],
    url: str,
    doi: str | None,
    useconstraints: str,
    citation: dict,
    entityandattribute: list[dict],
    build_date: str,
    revision_date: str | None = None,
) -> dict:
    theme = list(scope_keywords.get("theme", [])) + [
        t for t in extra_theme if t not in scope_keywords.get("theme", [])
    ]
    dist_defaults = defaults.get("distribution", {})
    begin, end = temporal

    dates = {"creation": build_date, "publication": build_date}
    if revision_date:
        dates["revision"] = revision_date

    block: dict = {
        "language": "eng",
        "charset": "utf8",
        "title": title,
        "abstract": abstract,
        "purpose": purpose,
        "dates": dates,
        "keywords": {
            "theme": {"keywords": theme, "keywords_type": "theme"},
            "place": {"keywords": list(place_keywords), "keywords_type": "place"},
        },
        "topiccategory": list(defaults.get("keywords", {}).get("topiccategory", [])),
        "extents": {
            "spatial": [
                {
                    "bbox": bbox,
                    "crs": defaults.get("spatial_reference", {}).get("epsg", 4326),
                }
            ],
            "temporal": [{"begin": begin, "end": end}],
        },
        "fees": dist_defaults.get("fees", "None."),
        "accessconstraints": dist_defaults.get(
            "access_constraints", "otherRestrictions"
        ),
        "rights": useconstraints,
        "useconstraints": useconstraints,
        "url": url,
        "status": "completed",
        "maintenancefrequency": "asNeeded",
        "doi": doi,
        "citation": citation,
        "entityandattribute": entityandattribute,
    }
    return block


def _assemble(
    *,
    identifier: str,
    hierarchylevel: str,
    spatial: dict,
    defaults: dict,
    identification: dict,
    distribution: dict,
    lineage_statement: str,
    process_steps: list[dict],
    scope_level: str,
    build_date: str,
    versioning_policy: str | None = None,
    version: str | None = None,
) -> dict:
    metadata = {
        "identifier": identifier,
        "language": defaults.get("metadata", {}).get("language", "en"),
        "charset": defaults.get("metadata", {}).get("charset", "utf8"),
        "hierarchylevel": hierarchylevel,
        "datestamp": build_date,
        "dates": {"creation": build_date},
    }
    dataquality = {
        "scope": {"level": scope_level},
        "lineage": {"statement": lineage_statement, "processstep": process_steps},
    }
    mcf: dict = {
        "mcf": {"version": MCF_VERSION},
        "metadata": metadata,
        "spatial": dict(spatial),
        "identification": identification,
        "contact": _contact_block(defaults),
        "distribution": distribution,
        "dataquality": dataquality,
        "spatial_reference": dict(defaults.get("spatial_reference", {})),
        # FGDC <distliab> is mandatory within <distinfo>; pygeometa/ISO ignore
        # this key. Sourced from release_defaults so the boilerplate is edited
        # in one place and flows to every rendered FGDC document.
        "distribution_liability": _collapse_ws(
            defaults.get("distribution", {}).get("liability_statement", "")
        ),
    }
    if version is not None:
        mcf["identification"]["edition"] = version
    if versioning_policy:
        mcf["identification"]["versioning_policy"] = versioning_policy
    return mcf


def _doi_url(doi: str | None) -> str | None:
    """Normalize a bare DOI or ``doi:`` form to a full ``https://doi.org/`` URL."""
    if not doi:
        return None
    doi = doi.strip()
    if doi.startswith("http"):
        return doi
    if doi.lower().startswith("doi:"):
        doi = doi[4:]
    return f"https://doi.org/{doi}"


def _place_keywords(spatial_extent: str | None) -> list[str]:
    if not spatial_extent:
        return []
    return [_collapse_ws(spatial_extent)]
