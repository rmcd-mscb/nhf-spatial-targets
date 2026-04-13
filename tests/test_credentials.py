"""Tests for nhf_spatial_targets.credentials — credential materialisation helpers."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from nhf_spatial_targets.credentials import (
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
