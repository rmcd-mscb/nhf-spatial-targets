"""Credential materialisation helpers for nhf-spatial-targets.

Copies credential sections from ``.credentials.yml`` into the dotfiles
read by ``cdsapi`` (``~/.cdsapirc``) and ``earthaccess`` (``~/.netrc``).
Both writes are atomic (tmp + rename) and the resulting files are mode 0600.

The netrc helper uses a **line-based** approach to preserve all existing
content verbatim — macdef blocks (and their blank-line terminators),
``default`` entries, ``#`` comments, and other machine entries are kept
byte-for-byte in their original order.

.. note::
    Concurrent external edits to ``~/.netrc`` between the read and the
    atomic replace will be overwritten silently.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import tempfile
from pathlib import Path

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def materialize_cdsapirc(creds: dict, home: Path | None = None) -> Path:
    """Write ``~/.cdsapirc`` from the ``cds`` section of ``.credentials.yml``.

    Overwrites any existing file atomically (tmp-file + rename).  The file
    is written with mode 0600 so that only the owning user can read it.

    Parameters
    ----------
    creds:
        Parsed ``.credentials.yml`` dictionary.  Must contain a ``cds``
        section with non-empty ``url`` and ``key`` keys.
    home:
        Override the home directory (useful in tests).  Defaults to
        ``Path.home()``.

    Returns
    -------
    Path
        Absolute path of the written file.

    Raises
    ------
    ValueError
        If the ``cds`` section is absent or if ``url`` or ``key`` is empty.
    """
    section = (creds or {}).get("cds") or {}
    url = section.get("url", "").strip()
    key = section.get("key", "").strip()
    if not url or not key:
        raise ValueError(
            "cds credentials incomplete — "
            "cds.url and cds.key must be non-empty in .credentials.yml"
        )

    target = _home_dir(home) / ".cdsapirc"
    content = f"url: {url}\nkey: {key}\n"
    _atomic_write(target, content, mode=0o600)
    return target


def materialize_netrc_earthdata(creds: dict, home: Path | None = None) -> Path:
    """Merge an earthdata entry into ``~/.netrc`` from ``.credentials.yml``.

    Behaviour:

    - If ``~/.netrc`` does not exist it is created with just the earthdata
      entry.
    - If one or more ``urs.earthdata.nasa.gov`` entries already exist they
      are all removed and a single updated entry is appended at the end.
    - All other content — machine entries, macdef blocks (including their
      blank-line terminators), ``default`` blocks, ``#`` comments, and blank
      lines — is preserved **line-for-line** in its original order.
    - A backup of the pre-existing ``~/.netrc`` is written to ``~/.netrc.bak``
      (mode 0600) before any changes are made.
    - The file is written atomically (tmp in the same directory + rename)
      and set to mode 0600.

    Parameters
    ----------
    creds:
        Parsed ``.credentials.yml`` dictionary.  Must contain a
        ``nasa_earthdata`` section with non-empty ``username`` and
        ``password`` keys.
    home:
        Override the home directory (useful in tests).  Defaults to
        ``Path.home()``.

    Returns
    -------
    Path
        Absolute path of the written file.

    Raises
    ------
    ValueError
        If the ``nasa_earthdata`` section is absent or if ``username`` or
        ``password`` is empty.
    """
    section = (creds or {}).get("nasa_earthdata") or {}
    username = section.get("username", "").strip()
    password = section.get("password", "").strip()
    if not username or not password:
        raise ValueError(
            "nasa_earthdata credentials incomplete — "
            "nasa_earthdata.username and nasa_earthdata.password "
            "must be non-empty in .credentials.yml"
        )

    target = _home_dir(home) / ".netrc"

    # Back up the existing file before modifying it
    if target.exists():
        bak = target.parent / ".netrc.bak"
        shutil.copy2(str(target), str(bak))
        os.chmod(bak, 0o600)
        _logger.debug("Backed up existing ~/.netrc to %s", bak)

    # Build the new earthdata line (single-line format is widely supported)
    earthdata_line = (
        f"machine urs.earthdata.nasa.gov login {username} password {password}\n"
    )

    # Read existing file as lines, removing any previous earthdata blocks
    if target.exists():
        existing_lines = target.read_text().splitlines(keepends=True)
    else:
        existing_lines = []

    kept_lines = _remove_earthdata_blocks(existing_lines)

    # Append the new entry at the end
    content = "".join(kept_lines) + earthdata_line

    _atomic_write(target, content, mode=0o600)
    return target


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _home_dir(home: Path | None) -> Path:
    return home if home is not None else Path.home()


def _atomic_write(target: Path, content: str, mode: int) -> None:
    """Write *content* to *target* atomically; set file permissions to *mode*.

    After the rename, the final file mode is verified.  If the filesystem
    silently ignores ``chmod`` and the resulting mode differs from *mode*,
    an ``OSError`` is raised rather than leaving credentials potentially
    world-readable.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp_")
    try:
        try:
            os.write(fd, content.encode())
        finally:
            os.close(fd)
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, target)
    except Exception:
        # Clean up the tmp file on failure
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    # Post-condition: verify the mode was applied.  Some filesystems (FAT,
    # CIFS mounts) silently ignore chmod; refuse to continue in that case.
    actual_mode = stat.S_IMODE(target.stat().st_mode)
    if actual_mode != mode:
        raise OSError(
            f"File mode mismatch after write: expected {oct(mode)} but got "
            f"{oct(actual_mode)} for {target}.  The filesystem may not support "
            "POSIX permissions.  Refusing to leave credentials potentially "
            "world-readable."
        )


def _remove_earthdata_blocks(lines: list[str]) -> list[str]:
    """Return lines with all 'machine urs.earthdata.nasa.gov' blocks removed.

    Preserves all other content verbatim, including macdef bodies, default
    blocks, comments, and blank lines.

    A ``machine urs.earthdata.nasa.gov`` block starts at the line whose
    first non-whitespace token is ``machine`` followed by
    ``urs.earthdata.nasa.gov``.  The block extends through subsequent
    continuation lines (lines that do not start a new top-level keyword)
    until one of the following is encountered:

    - Another ``machine <name>`` line
    - A ``default`` line
    - A ``macdef <name>`` line (the macdef body through the next blank line
      is **not** considered part of the earthdata block)
    - End of file
    """
    _EARTHDATA_HOST = "urs.earthdata.nasa.gov"

    result: list[str] = []
    skip = False

    for line in lines:
        stripped = line.strip()
        tokens = stripped.split()

        # Detect top-level keyword lines
        is_machine = len(tokens) >= 2 and tokens[0] == "machine"
        is_default = len(tokens) >= 1 and tokens[0] == "default"
        is_macdef = len(tokens) >= 2 and tokens[0] == "macdef"

        if is_machine:
            if tokens[1] == _EARTHDATA_HOST:
                # Start of an earthdata block — skip it
                skip = True
            else:
                # A different machine block — keep it
                skip = False
                result.append(line)
        elif is_default or is_macdef:
            # Top-level default/macdef — always keep; end any earthdata skip
            skip = False
            result.append(line)
        else:
            if not skip:
                result.append(line)

    return result
