"""Render ISO 19139 XML from an MCF dict via pygeometa.

Unlike FGDC (which pygeometa cannot emit, see :mod:`nhf_spatial_targets.
release.fgdc`), ISO 19139 is produced directly by pygeometa's output schema.
This module is a thin, pure wrapper so the release pipeline has one symmetric
``render_iso(mcf)`` / ``render_fgdc(mcf, kind=...)`` surface and so the ISO
render path is covered by a golden test alongside FGDC.

The MCF dict is the contract: every date is already baked in (``now`` injected
at build time), so the output is byte-stable for a pinned pygeometa version.
"""

from __future__ import annotations

from pygeometa.schemas.iso19139 import ISO19139OutputSchema


def render_iso(mcf: dict) -> str:
    """Render the ISO 19139 XML for one release item from its MCF dict.

    Parameters
    ----------
    mcf
        The MCF dict from :func:`nhf_spatial_targets.release.mcf.build_*_mcf`.

    Returns
    -------
    The ISO 19139 XML as a string (includes the XML declaration).
    """
    return ISO19139OutputSchema().write(mcf)
