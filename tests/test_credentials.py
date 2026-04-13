"""Tests for nhf_spatial_targets.credentials — credential materialisation helpers."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from nhf_spatial_targets.credentials import (
    _remove_earthdata_blocks,
    materialize_cdsapirc,
    materialize_netrc_earthdata,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

VALID_CREDS: dict = {
    "cds": {
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "12345:abc-def",
    },
    "nasa_earthdata": {
        "username": "myuser",
        "password": "mypassword",
    },
}


def _file_mode(path: Path) -> int:
    """Return permission bits of *path* as an integer."""
    return stat.S_IMODE(path.stat().st_mode)


# ---------------------------------------------------------------------------
# materialize_cdsapirc — content
# ---------------------------------------------------------------------------


def test_materialize_cdsapirc_writes_expected_content(tmp_path):
    """Written file must follow the two-line cdsapirc format."""
    written = materialize_cdsapirc(VALID_CREDS, home=tmp_path)

    assert written == tmp_path / ".cdsapirc"
    content = written.read_text()
    assert content == (
        "url: https://cds.climate.copernicus.eu/api\nkey: 12345:abc-def\n"
    )


# ---------------------------------------------------------------------------
# materialize_cdsapirc — file mode
# ---------------------------------------------------------------------------


def test_materialize_cdsapirc_mode_600(tmp_path):
    """~/.cdsapirc must be written with mode 0600."""
    written = materialize_cdsapirc(VALID_CREDS, home=tmp_path)
    assert _file_mode(written) == 0o600


# ---------------------------------------------------------------------------
# materialize_cdsapirc — overwrites existing
# ---------------------------------------------------------------------------


def test_materialize_cdsapirc_overwrites_existing(tmp_path):
    """A stale ~/.cdsapirc must be atomically replaced."""
    cdsapirc = tmp_path / ".cdsapirc"
    cdsapirc.write_text("url: https://old.example.com\nkey: stale\n")

    materialize_cdsapirc(VALID_CREDS, home=tmp_path)

    content = cdsapirc.read_text()
    assert "old.example.com" not in content
    assert "stale" not in content
    assert "cds.climate.copernicus.eu" in content


# ---------------------------------------------------------------------------
# materialize_cdsapirc — missing / incomplete section raises
# ---------------------------------------------------------------------------


def test_materialize_cdsapirc_missing_section_raises(tmp_path):
    """Empty / missing cds section must raise ValueError."""
    with pytest.raises(ValueError, match="cds"):
        materialize_cdsapirc({}, home=tmp_path)


def test_materialize_cdsapirc_missing_key_raises(tmp_path):
    """cds.key missing must raise ValueError."""
    creds = {"cds": {"url": "https://cds.climate.copernicus.eu/api"}}
    with pytest.raises(ValueError, match="cds"):
        materialize_cdsapirc(creds, home=tmp_path)


def test_materialize_cdsapirc_missing_url_raises(tmp_path):
    """cds.url missing must raise ValueError."""
    creds = {"cds": {"key": "uid:key"}}
    with pytest.raises(ValueError, match="cds"):
        materialize_cdsapirc(creds, home=tmp_path)


# ---------------------------------------------------------------------------
# materialize_netrc_earthdata — creates when absent
# ---------------------------------------------------------------------------


def test_materialize_netrc_creates_when_absent(tmp_path):
    """~/.netrc must be created from scratch if it does not exist."""
    assert not (tmp_path / ".netrc").exists()

    written = materialize_netrc_earthdata(VALID_CREDS, home=tmp_path)

    assert written == tmp_path / ".netrc"
    content = written.read_text()
    assert "urs.earthdata.nasa.gov" in content
    assert "myuser" in content
    assert "mypassword" in content


# ---------------------------------------------------------------------------
# materialize_netrc_earthdata — preserves other machine entries
# ---------------------------------------------------------------------------


def test_materialize_netrc_preserves_other_machines(tmp_path):
    """Pre-existing machine entries for other hosts must be preserved."""
    netrc = tmp_path / ".netrc"
    netrc.write_text("machine example.com login x password y\n")

    materialize_netrc_earthdata(VALID_CREDS, home=tmp_path)

    content = netrc.read_text()
    assert "example.com" in content
    assert "urs.earthdata.nasa.gov" in content
    assert "myuser" in content


# ---------------------------------------------------------------------------
# materialize_netrc_earthdata — replaces existing earthdata entry
# ---------------------------------------------------------------------------


def test_materialize_netrc_replaces_existing_earthdata(tmp_path):
    """An existing urs.earthdata entry must be replaced, not duplicated."""
    netrc = tmp_path / ".netrc"
    netrc.write_text("machine urs.earthdata.nasa.gov login olduser password oldpass\n")

    materialize_netrc_earthdata(VALID_CREDS, home=tmp_path)

    content = netrc.read_text()
    # Only one occurrence of the machine name
    assert content.count("urs.earthdata.nasa.gov") == 1
    assert "olduser" not in content
    assert "oldpass" not in content
    assert "myuser" in content
    assert "mypassword" in content


# ---------------------------------------------------------------------------
# materialize_netrc_earthdata — file mode
# ---------------------------------------------------------------------------


def test_materialize_netrc_mode_600(tmp_path):
    """~/.netrc must be written with mode 0600."""
    written = materialize_netrc_earthdata(VALID_CREDS, home=tmp_path)
    assert _file_mode(written) == 0o600


# ---------------------------------------------------------------------------
# materialize_netrc_earthdata — missing / incomplete section raises
# ---------------------------------------------------------------------------


def test_materialize_netrc_missing_section_raises(tmp_path):
    """Empty / missing nasa_earthdata section must raise ValueError."""
    with pytest.raises(ValueError, match="nasa_earthdata"):
        materialize_netrc_earthdata({}, home=tmp_path)


def test_materialize_netrc_missing_password_raises(tmp_path):
    """nasa_earthdata.password missing must raise ValueError."""
    creds = {"nasa_earthdata": {"username": "u"}}
    with pytest.raises(ValueError, match="nasa_earthdata"):
        materialize_netrc_earthdata(creds, home=tmp_path)


# ---------------------------------------------------------------------------
# materialize_netrc_earthdata — idempotency
# ---------------------------------------------------------------------------


def test_materialize_netrc_idempotent(tmp_path):
    """Calling materialize twice must yield the same single entry."""
    materialize_netrc_earthdata(VALID_CREDS, home=tmp_path)
    materialize_netrc_earthdata(VALID_CREDS, home=tmp_path)

    content = (tmp_path / ".netrc").read_text()
    assert content.count("urs.earthdata.nasa.gov") == 1


# ---------------------------------------------------------------------------
# _remove_earthdata_blocks — line-based parser edge cases
# ---------------------------------------------------------------------------


def test_materialize_netrc_preserves_macdef(tmp_path):
    """macdef block body must survive materialize unchanged."""
    existing = (
        "macdef init\ncd /foo\nls\n\nmachine other.example.com login a password b\n"
    )
    netrc = tmp_path / ".netrc"
    netrc.write_text(existing)

    materialize_netrc_earthdata(VALID_CREDS, home=tmp_path)

    content = netrc.read_text()
    assert "macdef init" in content
    assert "cd /foo" in content
    assert "ls" in content
    assert "other.example.com" in content
    assert "urs.earthdata.nasa.gov" in content


def test_materialize_netrc_preserves_default_block(tmp_path):
    """default entry must survive materialize unchanged."""
    existing = "default login anonymous password guest\n"
    netrc = tmp_path / ".netrc"
    netrc.write_text(existing)

    materialize_netrc_earthdata(VALID_CREDS, home=tmp_path)

    content = netrc.read_text()
    assert "default login anonymous password guest" in content
    assert "urs.earthdata.nasa.gov" in content


def test_materialize_netrc_preserves_comments(tmp_path):
    """Comment lines must survive materialize unchanged."""
    existing = "# this is a comment\nmachine a.com login b password c\n"
    netrc = tmp_path / ".netrc"
    netrc.write_text(existing)

    materialize_netrc_earthdata(VALID_CREDS, home=tmp_path)

    content = netrc.read_text()
    assert "# this is a comment" in content
    assert "machine a.com" in content
    assert "urs.earthdata.nasa.gov" in content


def test_materialize_netrc_handles_multiple_earthdata_blocks(tmp_path):
    """Two pre-existing earthdata blocks must be collapsed to exactly one."""
    existing = (
        "machine urs.earthdata.nasa.gov login old1 password old1\n"
        "machine other.com login x password y\n"
        "machine urs.earthdata.nasa.gov login old2 password old2\n"
    )
    netrc = tmp_path / ".netrc"
    netrc.write_text(existing)

    materialize_netrc_earthdata(VALID_CREDS, home=tmp_path)

    content = netrc.read_text()
    assert content.count("urs.earthdata.nasa.gov") == 1
    assert "old1" not in content
    assert "old2" not in content
    assert "myuser" in content
    assert "other.com" in content


def test_materialize_netrc_preserves_multiline_machine_entry(tmp_path):
    """Multi-line machine block (login/password on separate lines) is kept."""
    existing = "machine foo.com\n  login x\n  password y\n"
    netrc = tmp_path / ".netrc"
    netrc.write_text(existing)

    materialize_netrc_earthdata(VALID_CREDS, home=tmp_path)

    content = netrc.read_text()
    assert "machine foo.com" in content
    assert "  login x" in content
    assert "  password y" in content
    assert "urs.earthdata.nasa.gov" in content


def test_materialize_netrc_preserves_blank_lines(tmp_path):
    """Blank lines between entries must be kept."""
    existing = "machine a.com login x password y\n\nmachine b.com login p password q\n"
    netrc = tmp_path / ".netrc"
    netrc.write_text(existing)

    materialize_netrc_earthdata(VALID_CREDS, home=tmp_path)

    content = netrc.read_text()
    # Both machines survive
    assert "machine a.com" in content
    assert "machine b.com" in content
    # There is at least one blank line preserved
    assert "\n\n" in content


# ---------------------------------------------------------------------------
# _remove_earthdata_blocks — unit tests
# ---------------------------------------------------------------------------


def test_remove_earthdata_blocks_empty():
    assert _remove_earthdata_blocks([]) == []


def test_remove_earthdata_blocks_no_earthdata():
    lines = ["machine foo.com login x password y\n"]
    assert _remove_earthdata_blocks(lines) == lines


def test_remove_earthdata_blocks_only_earthdata():
    lines = ["machine urs.earthdata.nasa.gov login u password p\n"]
    assert _remove_earthdata_blocks(lines) == []


def test_remove_earthdata_blocks_keeps_other_machines():
    lines = [
        "machine urs.earthdata.nasa.gov login u password p\n",
        "machine other.com login a password b\n",
    ]
    result = _remove_earthdata_blocks(lines)
    assert result == ["machine other.com login a password b\n"]


def test_remove_earthdata_blocks_keeps_macdef_after_earthdata():
    lines = [
        "machine urs.earthdata.nasa.gov login u password p\n",
        "macdef init\n",
        "ls\n",
        "\n",
    ]
    result = _remove_earthdata_blocks(lines)
    assert result == ["macdef init\n", "ls\n", "\n"]


def test_remove_earthdata_blocks_skips_continuation_lines():
    lines = [
        "machine urs.earthdata.nasa.gov\n",
        "  login u\n",
        "  password p\n",
        "machine other.com login a password b\n",
    ]
    result = _remove_earthdata_blocks(lines)
    assert result == ["machine other.com login a password b\n"]


# ---------------------------------------------------------------------------
# materialize_netrc_earthdata — backup tests
# ---------------------------------------------------------------------------


def test_materialize_netrc_creates_backup(tmp_path):
    """Existing ~/.netrc must be backed up to ~/.netrc.bak with mode 0600."""
    original_content = "machine pre-existing.com login old password old\n"
    netrc = tmp_path / ".netrc"
    netrc.write_text(original_content)

    materialize_netrc_earthdata(VALID_CREDS, home=tmp_path)

    bak = tmp_path / ".netrc.bak"
    assert bak.exists(), "~/.netrc.bak should be created"
    assert bak.read_text() == original_content, "backup must contain original content"
    assert _file_mode(bak) == 0o600, "backup must be mode 0600"


def test_materialize_netrc_no_backup_when_absent(tmp_path):
    """No backup file should be created when ~/.netrc does not exist."""
    materialize_netrc_earthdata(VALID_CREDS, home=tmp_path)

    bak = tmp_path / ".netrc.bak"
    assert not bak.exists(), "No backup when original file absent"


# ---------------------------------------------------------------------------
# _atomic_write — mode verification
# ---------------------------------------------------------------------------


def test_atomic_write_raises_when_mode_silently_ignored(tmp_path, monkeypatch):
    """_atomic_write must raise OSError if the written file has wrong mode."""
    import nhf_spatial_targets.credentials as _creds_mod

    # Monkeypatch stat.S_IMODE in the credentials module to always return 0o644
    # regardless of what chmod set — simulating a filesystem that ignores chmod.
    monkeypatch.setattr(_creds_mod.stat, "S_IMODE", lambda mode: 0o644)

    target = tmp_path / "secret"
    with pytest.raises(OSError, match="mode mismatch"):
        _creds_mod._atomic_write(target, "content", mode=0o600)
