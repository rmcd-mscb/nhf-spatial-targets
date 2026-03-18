"""Tests for shared earthdata credential handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nhf_spatial_targets.fetch._auth import earthdata_login


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    rd = tmp_path / "run"
    rd.mkdir()
    return rd


def _write_credentials(workdir: Path, username: str, password: str) -> None:
    import yaml

    creds = {
        "nasa_earthdata": {"username": username, "password": password},
        "sciencebase": {"username": "", "password": ""},
    }
    (workdir / ".credentials.yml").write_text(yaml.dump(creds))


@patch("nhf_spatial_targets.fetch._auth.earthaccess.login")
def test_credentials_file_used(mock_login, workdir):
    """Credentials from .credentials.yml are set as env vars."""
    _write_credentials(workdir, "myuser", "mypass")
    mock_login.return_value = MagicMock(authenticated=True)

    earthdata_login(workdir)

    mock_login.assert_called_once_with(strategy="environment")


@patch("nhf_spatial_targets.fetch._auth.earthaccess.login")
def test_empty_credentials_falls_back(mock_login, workdir):
    """Empty credentials in .credentials.yml trigger default login."""
    _write_credentials(workdir, "", "")
    mock_login.return_value = MagicMock(authenticated=True)

    earthdata_login(workdir)

    mock_login.assert_called_once_with()


@patch("nhf_spatial_targets.fetch._auth.earthaccess.login")
def test_no_credentials_file_falls_back(mock_login, workdir):
    """Missing .credentials.yml triggers default login."""
    mock_login.return_value = MagicMock(authenticated=True)

    earthdata_login(workdir)

    mock_login.assert_called_once_with()


@patch("nhf_spatial_targets.fetch._auth.earthaccess.login")
def test_login_failure_raises(mock_login, workdir):
    """RuntimeError raised when all login strategies fail."""
    mock_login.return_value = MagicMock(authenticated=False)

    with pytest.raises(RuntimeError, match="Earthdata"):
        earthdata_login(workdir)


@patch("nhf_spatial_targets.fetch._auth.earthaccess.login")
def test_login_returns_none_raises(mock_login, workdir):
    """RuntimeError raised when login returns None."""
    mock_login.return_value = None

    with pytest.raises(RuntimeError, match="Earthdata"):
        earthdata_login(workdir)


@patch("nhf_spatial_targets.fetch._auth.earthaccess.login")
def test_credentials_fallback_on_env_failure(mock_login, workdir):
    """Falls back to default login if environment strategy fails."""
    _write_credentials(workdir, "myuser", "mypass")
    # First call (strategy="environment") fails, second (default) succeeds
    mock_login.side_effect = [
        MagicMock(authenticated=False),
        MagicMock(authenticated=True),
    ]

    result = earthdata_login(workdir)

    assert result.authenticated
    assert mock_login.call_count == 2


def test_malformed_yaml_raises(workdir):
    """Malformed .credentials.yml raises ValueError."""
    (workdir / ".credentials.yml").write_text(": invalid: yaml: [")

    with pytest.raises(ValueError, match="Cannot parse"):
        earthdata_login(workdir)


@patch("nhf_spatial_targets.fetch._auth.earthaccess.login")
def test_env_vars_cleaned_up_on_fallback(mock_login, workdir, monkeypatch):
    """Environment variables are removed after environment-based login fails."""
    import os

    _write_credentials(workdir, "myuser", "mypass")
    mock_login.side_effect = [
        MagicMock(authenticated=False),
        MagicMock(authenticated=True),
    ]
    # Ensure env vars don't pre-exist
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)

    earthdata_login(workdir)

    assert "EARTHDATA_USERNAME" not in os.environ
    assert "EARTHDATA_PASSWORD" not in os.environ
