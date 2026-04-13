"""Credential materialisation helpers for nhf-spatial-targets.

Copies credential sections from ``.credentials.yml`` into the dotfiles
read by ``cdsapi`` (``~/.cdsapirc``) and ``earthaccess`` (``~/.netrc``).
Both writes are atomic (tmp + rename) and the resulting files are mode 0600.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


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
    - If an ``urs.earthdata.nasa.gov`` entry already exists it is replaced
      in place; all other machine entries are preserved verbatim.
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

    # Build the new earthdata line (single-line format is widely supported)
    earthdata_line = (
        f"machine urs.earthdata.nasa.gov login {username} password {password}"
    )

    # Parse existing file, excluding any previous earthdata entry
    existing_blocks = _parse_netrc_excluding(target, "urs.earthdata.nasa.gov")

    # Append the new entry
    all_lines = existing_blocks + [earthdata_line]
    content = "\n".join(all_lines) + "\n"

    _atomic_write(target, content, mode=0o600)
    return target


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _home_dir(home: Path | None) -> Path:
    return home if home is not None else Path.home()


def _atomic_write(target: Path, content: str, mode: int) -> None:
    """Write *content* to *target* atomically; set file permissions to *mode*."""
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


def _parse_netrc_excluding(netrc_path: Path, exclude_machine: str) -> list[str]:
    """Return lines from *netrc_path* that do not belong to *exclude_machine*.

    Uses simple line-based parsing.  Each ``machine`` keyword starts a new
    block; the block ends at the next ``machine`` keyword (or EOF).  Blocks
    whose machine name matches *exclude_machine* are dropped.  Blocks for
    other machines are returned as single-line entries (normalised) to avoid
    accumulating blank lines on repeated materialise calls.

    If *netrc_path* does not exist an empty list is returned.
    """
    if not netrc_path.exists():
        return []

    raw = netrc_path.read_text()
    tokens = raw.split()

    # Reconstruct machine blocks from the token stream
    blocks: list[dict] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "machine":
            if i + 1 >= len(tokens):
                break
            machine = tokens[i + 1]
            attrs: dict[str, str] = {}
            i += 2
            while i < len(tokens) and tokens[i] != "machine":
                key = tokens[i]
                if i + 1 < len(tokens) and tokens[i + 1] != "machine":
                    attrs[key] = tokens[i + 1]
                    i += 2
                else:
                    i += 1
            blocks.append({"machine": machine, "attrs": attrs})
        else:
            i += 1

    lines: list[str] = []
    for block in blocks:
        if block["machine"] == exclude_machine:
            continue
        parts = [f"machine {block['machine']}"]
        for k, v in block["attrs"].items():
            parts.append(f"{k} {v}")
        lines.append(" ".join(parts))

    return lines


def _file_mode(path: Path) -> int:
    """Return the permission bits of *path* as an integer (e.g. 0o600)."""
    return stat.S_IMODE(path.stat().st_mode)
