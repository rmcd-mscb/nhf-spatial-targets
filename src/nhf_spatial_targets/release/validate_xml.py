"""Validate rendered metadata XML with the USGS Metadata Parser (``mp``).

``mp`` is the canonical FGDC CSDGM validator but is an operator-installed
external tool, not a Python dependency. This module wraps it defensively:

- ``mp`` on PATH -> run it, return parsed ``(ok, warnings, errors)``.
- ``mp`` absent -> fall back to an ``lxml`` well-formedness check (warn-only),
  so a missing validator never hard-fails a build.

The result is a :class:`MpResult` NamedTuple, so callers can unpack it as
``ok, warnings, errors = validate_xml_file(path)`` (the tuple shape the design
plan specifies) or use the attribute names.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

from lxml import etree

_MP_TIMEOUT_S = 120

_ERROR_HEADERS = {"error", "errors"}
_WARNING_HEADERS = {"warning", "warnings"}


class MpResult(NamedTuple):
    """Outcome of an XML validation: overall pass flag plus message lists."""

    ok: bool
    warnings: list[str]
    errors: list[str]


def mp_available(mp_path: str = "mp") -> bool:
    """True if the ``mp`` executable is resolvable on PATH (or as given)."""
    return shutil.which(mp_path) is not None


def _wellformed(path: Path) -> MpResult:
    """lxml-only fallback when ``mp`` is unavailable: check well-formedness."""
    try:
        etree.parse(str(path))
    except etree.XMLSyntaxError as exc:
        return MpResult(False, [], [f"XML is not well-formed: {exc}"])
    return MpResult(
        True,
        ["mp not found on PATH; validated XML well-formedness only (no CSDGM check)."],
        [],
    )


def _parse_mp_report(report: str) -> tuple[list[str], list[str]]:
    """Split an ``mp`` error report into (warnings, errors).

    ``mp`` groups its messages under ``Errors:`` / ``Warnings:`` section
    headers, one message per (indented) line. A line is treated as a section
    header only when it is *exactly* that header token (``"Errors:"`` /
    ``"Warnings:"``, case-insensitive, trailing colon optional) -- so a
    message body that happens to begin with the word "error"/"warning" stays
    in its section instead of being mistaken for a header and dropped. Lines
    before the first header (mp's banner) carry no section and are ignored.

    This categorization is best-effort and used only for human-readable
    reporting: :func:`validate_xml_file` keys the pass/fail decision on mp's
    exit status, so a format mismatch here cannot turn an mp failure into a
    silent pass.
    """
    warnings: list[str] = []
    errors: list[str] = []
    bucket: list[str] | None = None
    for line in report.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        token = stripped.rstrip(":").strip().lower()
        if token in _ERROR_HEADERS:
            bucket = errors
            continue
        if token in _WARNING_HEADERS:
            bucket = warnings
            continue
        if bucket is not None:
            bucket.append(stripped)
    return warnings, errors


def validate_xml_file(path: str | Path, *, mp_path: str = "mp") -> MpResult:
    """Validate an FGDC XML file with ``mp``, or fall back to well-formedness.

    Parameters
    ----------
    path
        Path to the XML file to validate.
    mp_path
        Name or path of the ``mp`` executable (default ``"mp"``).

    Returns
    -------
    MpResult
        ``ok`` is False when ``mp`` exits non-zero or reports errors (or, in
        fallback mode, the XML is not well-formed). ``mp``-absent is a warning,
        not a failure.
    """
    path = Path(path)
    if not mp_available(mp_path):
        return _wellformed(path)

    with tempfile.TemporaryDirectory() as tmp:
        err_file = Path(tmp) / "mp_errors.txt"
        try:
            proc = subprocess.run(
                [mp_path, "-e", str(err_file), str(path)],
                capture_output=True,
                text=True,
                timeout=_MP_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return MpResult(False, [], [f"mp timed out after {_MP_TIMEOUT_S}s."])
        report = err_file.read_text() if err_file.exists() else (proc.stdout or "")
    warnings, errors = _parse_mp_report(report)
    # mp's exit status is authoritative for pass/fail; the parsed lists are
    # only for human-readable categorization. A non-zero exit we could not
    # attribute to a parsed error (mp aborted, or wrote a report format we did
    # not recognize) must surface as a failure rather than a silent pass.
    if proc.returncode != 0 and not errors:
        errors = [
            f"mp exited with status {proc.returncode}; "
            f"report: {report.strip() or '(empty)'}"
        ]
    ok = proc.returncode == 0 and not errors
    return MpResult(ok, warnings, errors)


def validate_xml_string(xml: str, *, mp_path: str = "mp") -> MpResult:
    """Validate an XML string by writing it to a temp file and running ``mp``."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metadata.xml"
        path.write_text(xml, encoding="utf-8")
        return validate_xml_file(path, mp_path=mp_path)
