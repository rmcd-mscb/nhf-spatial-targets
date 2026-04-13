"""CLI-level tests for the 'materialize-credentials' command.

Tests call the CLI function directly (not via subprocess) and monkeypatch
Path.home() so that the test suite never writes to the real $HOME.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

import nhf_spatial_targets.credentials as _creds_mod
from nhf_spatial_targets.cli import materialize_credentials_cmd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_CREDS_CONTENT = """\
cds:
  url: https://cds.climate.copernicus.eu/api
  key: 12345:abc-def
nasa_earthdata:
  username: myuser
  password: mypassword
"""

_CDS_ONLY_CONTENT = """\
cds:
  url: https://cds.climate.copernicus.eu/api
  key: 12345:abc-def
"""


def _file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _make_project(project_dir: Path, creds_content: str | None) -> None:
    """Create a minimal project skeleton with optional .credentials.yml."""
    project_dir.mkdir(parents=True, exist_ok=True)
    if creds_content is not None:
        (project_dir / ".credentials.yml").write_text(creds_content)


def _patch_home(monkeypatch, home: Path) -> None:
    """Redirect Path.home() and _home_dir() to *home* for all helpers."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))


# ---------------------------------------------------------------------------
# Exit code tests
# ---------------------------------------------------------------------------


def test_missing_project_dir_exits_2(tmp_path):
    """A non-existent project directory must exit with code 2."""
    project_dir = tmp_path / "no-such-project"

    with pytest.raises(SystemExit) as exc_info:
        materialize_credentials_cmd(project_dir)

    assert exc_info.value.code == 2


def test_missing_credentials_yml_exits_1(tmp_path):
    """A project without .credentials.yml must exit with code 1."""
    project_dir = tmp_path / "project"
    _make_project(project_dir, creds_content=None)

    with pytest.raises(SystemExit) as exc_info:
        materialize_credentials_cmd(project_dir)

    assert exc_info.value.code == 1


def test_malformed_yaml_exits_1(tmp_path, capsys):
    """Unparseable .credentials.yml must exit with code 1."""
    project_dir = tmp_path / "project"
    _make_project(project_dir, creds_content="cds: [\nbad yaml")

    with pytest.raises(SystemExit) as exc_info:
        materialize_credentials_cmd(project_dir)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "parse" in captured.err.lower() or "yaml" in captured.err.lower()


def test_empty_yaml_exits_1(tmp_path, capsys):
    """An empty .credentials.yml (yaml.safe_load returns None) must exit 1."""
    project_dir = tmp_path / "project"
    _make_project(project_dir, creds_content="")

    with pytest.raises(SystemExit) as exc_info:
        materialize_credentials_cmd(project_dir)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "save your edits" in captured.err or "empty" in captured.err


def test_cds_only_exits_1_netrc_skipped(tmp_path, monkeypatch):
    """Only cds section populated: exit 1; cdsapirc written, netrc skipped."""
    home = tmp_path / "home"
    home.mkdir()
    project_dir = tmp_path / "project"
    _make_project(project_dir, creds_content=_CDS_ONLY_CONTENT)
    _patch_home(monkeypatch, home)

    with pytest.raises(SystemExit) as exc_info:
        materialize_credentials_cmd(project_dir)

    # cdsapirc should be written
    assert (home / ".cdsapirc").exists(), "~/.cdsapirc should be written"
    # netrc should NOT be written (missing nasa_earthdata section)
    assert not (home / ".netrc").exists(), "~/.netrc should not be written"
    # Exit 1 because nasa_earthdata section was missing (user error)
    assert exc_info.value.code == 1


def test_both_sections_valid_exits_0(tmp_path, monkeypatch):
    """Both sections valid: exit 0, both files written with correct contents."""
    home = tmp_path / "home"
    home.mkdir()
    project_dir = tmp_path / "project"
    _make_project(project_dir, creds_content=_VALID_CREDS_CONTENT)
    _patch_home(monkeypatch, home)

    # Should NOT raise SystemExit (exit 0)
    try:
        materialize_credentials_cmd(project_dir)
    except SystemExit as exc:
        pytest.fail(f"Expected exit 0 but got SystemExit({exc.code})")

    cdsapirc = home / ".cdsapirc"
    netrc = home / ".netrc"

    assert cdsapirc.exists()
    assert netrc.exists()

    cdsapirc_text = cdsapirc.read_text()
    assert "cds.climate.copernicus.eu" in cdsapirc_text
    assert "12345:abc-def" in cdsapirc_text

    netrc_text = netrc.read_text()
    assert "urs.earthdata.nasa.gov" in netrc_text
    assert "myuser" in netrc_text
    assert "mypassword" in netrc_text

    # Both files must be mode 0600
    assert _file_mode(cdsapirc) == 0o600
    assert _file_mode(netrc) == 0o600


def test_permission_error_exits_3(tmp_path, monkeypatch):
    """A simulated PermissionError (OSError) must exit with code 3."""
    home = tmp_path / "home"
    home.mkdir()
    project_dir = tmp_path / "project"
    _make_project(project_dir, creds_content=_VALID_CREDS_CONTENT)
    _patch_home(monkeypatch, home)

    # Monkeypatch materialize_cdsapirc to raise PermissionError
    monkeypatch.setattr(
        _creds_mod,
        "materialize_cdsapirc",
        lambda creds, home=None: (_ for _ in ()).throw(
            PermissionError("read-only filesystem")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        materialize_credentials_cmd(project_dir)

    assert exc_info.value.code == 3
