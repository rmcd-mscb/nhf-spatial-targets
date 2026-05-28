"""Render human-readable ``README.md`` files for release items.

One Jinja2 template per item type (umbrella / source / fabric) under
``templates/readme/``, rendered against the same MCF dict that
:mod:`nhf_spatial_targets.release.mcf` builds for FGDC + ISO. Building one
MCF and rendering it three ways (FGDC, ISO, README) keeps the human-readable
file from drifting against the machine-readable metadata.

The renderer is pure: it reads only the passed MCF dict and derives a flat
template context from it (download-file listing, citation, extents, contacts,
lineage summary). No disk walking, no catalog or registry I/O.
"""

from __future__ import annotations

from pathlib import Path

import jinja2

_TEMPLATE_DIR = Path(__file__).parent / "templates" / "readme"

_KINDS = ("umbrella", "source", "fabric")

_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    # StrictUndefined turns a template typo (a variable not in _context)
    # into a loud UndefinedError instead of silently rendering a blank.
    undefined=jinja2.StrictUndefined,
)


def _download_files(mcf: dict) -> list[dict]:
    """Staged download entries from the MCF distribution block.

    Returns ``[{"name", "description"}, ...]`` for every distribution entry
    whose function is ``download`` (i.e. an actual staged file), skipping the
    landing/upstream information entries.
    """
    out = []
    for name, entry in (mcf.get("distribution") or {}).items():
        if entry.get("function") == "download":
            out.append({"name": name, "description": entry.get("description", "")})
    return out


def _citation_lines(citation: dict) -> list[str]:
    """One display string per author.

    Structured authors (umbrella / fabric -- ``{given, family, orcid,
    affiliation}``) format as ``Family, Given (Affiliation) [ORCID: ...]``;
    source citations are already free-text strings and pass through. Lines
    are precomputed in Python rather than the template so the Jinja
    ``trim_blocks`` setting can't eat the newline after an inline ``{% endif
    %}`` and run two authors together.
    """
    lines = []
    for author in citation.get("authors", []):
        if isinstance(author, dict):
            text = f"{author.get('family', '')}, {author.get('given', '')}".strip(", ")
            if author.get("affiliation"):
                text += f" ({author['affiliation']})"
            if author.get("orcid"):
                text += f" [ORCID: {author['orcid']}]"
            lines.append(text)
        else:
            lines.append(str(author))
    return lines


def _contact_line(contact: dict) -> str:
    text = contact.get("organization", "")
    if contact.get("email"):
        text += f" ({contact['email']})"
    return text


def _context(mcf: dict) -> dict:
    ident = mcf.get("identification", {})
    extents = ident.get("extents", {})
    spatial = (extents.get("spatial") or [{}])[0]
    temporal = (extents.get("temporal") or [{}])[0]
    contact = mcf.get("contact", {})
    lineage = mcf.get("dataquality", {}).get("lineage", {})
    citation = ident.get("citation", {})
    return {
        "title": ident.get("title", ""),
        "abstract": ident.get("abstract", ""),
        "purpose": ident.get("purpose", ""),
        "useconstraints": ident.get("useconstraints", ""),
        "doi": ident.get("doi"),
        "url": ident.get("url", ""),
        "edition": ident.get("edition"),
        "begin": temporal.get("begin"),
        "end": temporal.get("end"),
        "bbox": spatial.get("bbox"),
        "theme_keywords": ident.get("keywords", {})
        .get("theme", {})
        .get("keywords", []),
        "place_keywords": ident.get("keywords", {})
        .get("place", {})
        .get("keywords", []),
        "citation_lines": _citation_lines(citation),
        "citation_doi": citation.get("doi"),
        "entityandattribute": ident.get("entityandattribute", []),
        "point_of_contact_line": _contact_line(contact.get("pointOfContact", {})),
        "distributor_line": _contact_line(contact.get("distributor", {})),
        "lineage_statement": lineage.get("statement", ""),
        "process_step_count": len(lineage.get("processstep", [])),
        "download_files": _download_files(mcf),
    }


def render_readme(mcf: dict, *, kind: str) -> str:
    """Render the ``README.md`` body for one release item.

    Parameters
    ----------
    mcf
        The MCF dict from :func:`nhf_spatial_targets.release.mcf.build_*_mcf`.
    kind
        One of ``"umbrella"``, ``"source"``, ``"fabric"`` -- selects the
        template.

    Returns
    -------
    The rendered Markdown as a string.

    Raises
    ------
    ValueError
        *kind* is not a recognized item type.
    """
    if kind not in _KINDS:
        raise ValueError(f"Unknown README kind {kind!r}; expected one of {_KINDS}.")
    template = _ENV.get_template(f"{kind}.md.j2")
    return template.render(**_context(mcf))
