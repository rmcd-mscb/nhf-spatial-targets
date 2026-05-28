"""Render FGDC CSDGM 2.0 XML from an MCF dict via Jinja2 templates.

pygeometa ships no FGDC output schema, so the Federal Geographic Data
Committee *Content Standard for Digital Geospatial Metadata* (CSDGM,
FGDC-STD-001-1998) is rendered here from the same MCF dict that
:mod:`nhf_spatial_targets.release.mcf` builds for ISO and README. Building one
MCF and rendering it three ways keeps FGDC, ISO and README from drifting.

The renderer is a **pure function of the MCF dict**: it flattens the dict into
a template context (mirroring :mod:`nhf_spatial_targets.release.readme`) and
renders one of the per-item-type templates under ``templates/fgdc/``. Every
date already lives in the MCF (``now`` was injected at build time), so the
output is byte-stable -- no ``datetime.now()`` is called here.

Two FGDC subtleties are handled in this module rather than the templates:

- **CSDGM element order is strict** -- ``mp`` (the USGS Metadata Parser)
  rejects out-of-order elements. The base template fixes the order; the
  ``<procstep>`` order in particular is ``procdesc -> srcused* -> procdate ->
  proctime -> srcprod*`` with ``srcused``/``srcprod`` repeatable.
- **Source Citation Abbreviations.** ``<srcused>``/``<srcprod>`` carry short
  abbreviations that must resolve to ``<srcinfo>`` blocks in the same
  document. :func:`_srcinfo_entries` derives one ``<srcinfo>`` per unique
  process-step basename (abbreviation == basename), so the references always
  resolve.

After Jinja renders the tree it is parsed and pretty-printed with ``lxml``
(``remove_blank_text`` + ``pretty_print``) so indentation is deterministic and
the result is guaranteed well-formed regardless of template whitespace.
"""

from __future__ import annotations

from pathlib import Path

import jinja2
from lxml import etree

_TEMPLATE_DIR = Path(__file__).parent / "templates" / "fgdc"

_KINDS = ("umbrella", "source", "fabric")

# autoescape=True is essential here (unlike the README renderer): abstracts and
# use-constraints contain ``&``, ``<``, ``>`` (e.g. "kg/m²", ">=1",
# "min(a, b)") that must become XML entities. Jinja macros return Markup under
# autoescape, so the cntinfo() macro output is spliced without double-escaping.
_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=jinja2.StrictUndefined,
)

# ISO maintenance / status / access codelist values -> FGDC free-text phrases.
_PROGRESS = {"completed": "Complete", "onGoing": "In work", "planned": "Planned"}
_UPDATE = {
    "asNeeded": "As needed",
    "continual": "Continually",
    "annually": "Annually",
}
_ACCCONST = {"otherRestrictions": "None. Please see Use Constraints."}

# spatial.datatype -> FGDC citation <geoform>.
_GEOFORM = {"grid": "raster digital data", "vector": "vector digital data"}

# Filename extension -> FGDC <formname>.
_FORMNAME = {
    ".nc": "NetCDF",
    ".gpkg": "GeoPackage",
    ".json": "JSON",
    ".csv": "CSV",
    ".md": "Text",
}

# Per-item-type entity-type labels for the <eainfo><detailed><enttyp> block.
_ENTTYP = {
    "source": (
        "Consolidated NetCDF variables",
        "Variables carried in the consolidated CF-1.6 NetCDF files for this "
        "source dataset.",
    ),
    "fabric": (
        "Calibration target variables",
        "Calibration-target variables provided for this hydrologic fabric.",
    ),
}


def _fgdc_date(iso: str | None) -> str | None:
    """Convert an ISO date to the CSDGM ``YYYYMMDD`` form.

    ``"2026-05-28"`` -> ``"20260528"``; ``None`` -> ``None``. A partial value
    (a bare ``"1979"``) keeps its precision (dashes are simply removed). The
    CSDGM ``"Unknown"`` / ``"Present"`` fallbacks are supplied by
    :func:`_context`, not here -- this function only ever sees a real ISO date
    or ``None``.
    """
    if not iso:
        return None
    return iso.replace("-", "")


def _origins(citation: dict) -> list[str]:
    """Originator names for the citation ``<origin>`` elements.

    Structured authors (``{given, family, ...}``) render ``Family, Given``;
    free-text source citations pass through. CSDGM requires at least one
    ``<origin>``, so an empty author list falls back to the USGS.
    """
    out: list[str] = []
    for author in citation.get("authors", []):
        if isinstance(author, dict):
            name = f"{author.get('family', '')}, {author.get('given', '')}".strip(", ")
            out.append(name or "U.S. Geological Survey")
        else:
            out.append(str(author))
    return out or ["U.S. Geological Survey"]


def _onlinks(ident: dict) -> list[str]:
    """Online-linkage URLs: DOI, item URL, and any informational dist entries."""
    out: list[str] = []
    for url in (ident.get("doi"), ident.get("url")):
        if url and url not in out:
            out.append(url)
    return out


def _info_onlinks(distribution: dict, existing: list[str]) -> list[str]:
    """Landing/upstream (function=information) URLs not already linked."""
    out: list[str] = []
    for entry in (distribution or {}).values():
        url = entry.get("url")
        if entry.get("function") == "information" and url and url not in existing:
            if url not in out:
                out.append(url)
    return out


def _supplinf(ident: dict, lineage: dict) -> str | None:
    """Supplemental-information prose: edition + versioning + lineage summary.

    CSDGM lineage has no free-text statement element, so the MCF
    ``dataquality.lineage.statement`` (and the umbrella's edition / versioning
    policy) live here.
    """
    parts: list[str] = []
    if ident.get("edition"):
        parts.append(f"Edition: {ident['edition']}.")
    if ident.get("versioning_policy"):
        parts.append(ident["versioning_policy"])
    if lineage.get("statement"):
        parts.append(lineage["statement"])
    return "\n\n".join(parts) or None


def _cntinfo(contact: dict) -> dict:
    """Flatten an MCF contact entry into the cntinfo() macro's field names."""
    return {
        "cntper": contact.get("individualname", ""),
        "cntorg": contact.get("organization", ""),
        "cntpos": contact.get("positionname", ""),
        "address": contact.get("address", ""),
        "city": contact.get("city", ""),
        "state": contact.get("administrativearea", ""),
        "postal": contact.get("postalcode", ""),
        "country": contact.get("country", ""),
        "cntvoice": contact.get("phone", ""),
        "cntfax": contact.get("fax", ""),
        "cntemail": contact.get("email", ""),
    }


def _digforms(distribution: dict) -> list[dict]:
    """One digital-form entry per downloadable distribution file."""
    out: list[dict] = []
    for name, entry in (distribution or {}).items():
        if entry.get("function") != "download":
            continue
        suffix = Path(name).suffix
        out.append(
            {
                "formname": _FORMNAME.get(suffix, "Digital data"),
                "networkr": entry.get("url", name),
            }
        )
    return out


def _producing_procdate(procsteps: list[dict], basename: str) -> str | None:
    for step in procsteps:
        if basename in (step.get("srcprod") or []):
            return step.get("procdate") or None
    return None


def _srcinfo_entries(procsteps: list[dict]) -> list[dict]:
    """One ``<srcinfo>`` per unique process-step basename.

    ``<srcused>``/``<srcprod>`` carry Source Citation Abbreviations that must
    resolve to a ``<srcinfo>`` block; we use the basename itself as the
    abbreviation so the mapping is 1:1 and stable. A basename's source date is
    the process date of the step that produced it (it appears in that step's
    ``srcprod``); pure inputs that the pipeline never produced fall back to the
    CSDGM ``"Unknown"`` token.
    """
    names: set[str] = set()
    for step in procsteps:
        names.update(step.get("srcused") or [])
        names.update(step.get("srcprod") or [])
    entries: list[dict] = []
    for name in sorted(names):
        caldate = _producing_procdate(procsteps, name) or "Unknown"
        entries.append(
            {
                "srccitea": name,
                "title": name,
                "origin": "U.S. Geological Survey",
                "caldate": caldate,
                "typesrc": "Digital data",
                "srccurr": "Process date",
                "srccontr": (
                    "Intermediate or source dataset processed by the "
                    "nhf-spatial-targets pipeline."
                ),
            }
        )
    return entries


def _context(mcf: dict, *, kind: str) -> dict:
    ident = mcf.get("identification", {})
    extents = ident.get("extents", {})
    spatial = (extents.get("spatial") or [{}])[0]
    temporal = (extents.get("temporal") or [{}])[0]
    keywords = ident.get("keywords", {})
    contact = mcf.get("contact", {})
    point_of_contact = contact.get("pointOfContact", {})
    distributor = contact.get("distributor", {})
    lineage = mcf.get("dataquality", {}).get("lineage", {})
    procsteps = lineage.get("processstep", [])
    srcinfo_entries = _srcinfo_entries(procsteps)

    onlinks = _onlinks(ident)
    onlinks += _info_onlinks(mcf.get("distribution", {}), onlinks)

    enttypl, enttypd = _ENTTYP.get(kind, ("", ""))

    return {
        "metadata_date": _fgdc_date(mcf.get("metadata", {}).get("datestamp")),
        # idinfo / citation
        "origins": _origins(ident.get("citation", {})),
        "pubdate": _fgdc_date(ident.get("dates", {}).get("publication")),
        "title": ident.get("title", ""),
        "geoform": _GEOFORM.get(mcf.get("spatial", {}).get("datatype"), "digital data"),
        "pub_place": ", ".join(
            p
            for p in (distributor.get("city"), distributor.get("administrativearea"))
            if p
        ),
        "publisher": distributor.get("organization", ""),
        "onlinks": onlinks,
        # idinfo / descript
        "abstract": ident.get("abstract", ""),
        "purpose": ident.get("purpose", ""),
        "supplinf": _supplinf(ident, lineage),
        # idinfo / timeperd
        "begdate": _fgdc_date(temporal.get("begin")) or "Unknown",
        "enddate": _fgdc_date(temporal.get("end")) or "Present",
        # idinfo / status
        "progress": _PROGRESS.get(ident.get("status"), "Complete"),
        "update": _UPDATE.get(ident.get("maintenancefrequency"), "As needed"),
        # idinfo / spdom
        "bbox": spatial.get("bbox"),
        # idinfo / keywords
        "theme_keywords": keywords.get("theme", {}).get("keywords", []),
        "topic_categories": ident.get("topiccategory", []),
        "place_keywords": keywords.get("place", {}).get("keywords", []),
        # idinfo / constraints
        "accconst": _ACCCONST.get(
            ident.get("accessconstraints"), ident.get("accessconstraints", "None.")
        ),
        "useconst": ident.get("useconstraints", ""),
        "point_of_contact": _cntinfo(point_of_contact),
        # dataqual / lineage
        "has_lineage": bool(procsteps or srcinfo_entries),
        "srcinfo_entries": srcinfo_entries,
        "procsteps": procsteps,
        # spref
        "spref": mcf.get("spatial_reference", {}),
        # eainfo (rendered by the per-kind child template)
        "entityandattribute": ident.get("entityandattribute", []),
        "enttypl": enttypl,
        "enttypd": enttypd,
        # distinfo
        "distributor": _cntinfo(distributor),
        "distliab": mcf.get("distribution_liability", ""),
        "digforms": _digforms(mcf.get("distribution", {})),
        "fees": ident.get("fees", "None."),
    }


def render_fgdc(mcf: dict, *, kind: str) -> str:
    """Render the FGDC CSDGM 2.0 XML for one release item.

    Parameters
    ----------
    mcf
        The MCF dict from :func:`nhf_spatial_targets.release.mcf.build_*_mcf`.
    kind
        One of ``"umbrella"``, ``"source"``, ``"fabric"`` -- selects the
        template and the entity/attribute flavor.

    Returns
    -------
    Pretty-printed, well-formed FGDC XML with an XML declaration.

    Raises
    ------
    ValueError
        *kind* is not a recognized item type, or the item has no bounding box
        (CSDGM ``<spdom>`` is mandatory, so an item with no extent cannot
        produce valid FGDC).
    """
    if kind not in _KINDS:
        raise ValueError(f"Unknown FGDC kind {kind!r}; expected one of {_KINDS}.")
    context = _context(mcf, kind=kind)
    if context["bbox"] is None:
        # CSDGM <spdom><bounding> is mandatory; emitting the record without it
        # produces XML that mp rejects. Fail loudly here (pointing at the
        # catalog fix) rather than shipping silently-invalid metadata.
        raise ValueError(
            f"Cannot render FGDC for {kind!r} item {mcf.get('identification', {}).get('title', '')!r}: "
            "it has no bounding box, but CSDGM <spdom> is mandatory. Add "
            "`access.bbox_nwse` to the source's catalog entry (see the snodas "
            "entry) or ensure fabric.json carries a bbox."
        )
    raw = _ENV.get_template(f"{kind}.xml.j2").render(**context)
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.fromstring(raw.encode("utf-8"), parser=parser)
    return etree.tostring(
        tree, pretty_print=True, xml_declaration=True, encoding="UTF-8"
    ).decode("utf-8")
