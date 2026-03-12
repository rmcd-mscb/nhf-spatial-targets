"""Shared NASA Earthdata authentication for fetch modules."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import earthaccess
import yaml

logger = logging.getLogger(__name__)


def earthdata_login(run_dir: Path) -> earthaccess.Auth:
    """Authenticate with NASA Earthdata.

    Reads credentials from ``run_dir/.credentials.yml`` if available,
    otherwise falls back to earthaccess default login strategies
    (netrc, interactive prompt, etc.).

    Parameters
    ----------
    run_dir : Path
        Run workspace directory containing ``.credentials.yml``.

    Returns
    -------
    earthaccess.Auth
        Authenticated session object.

    Raises
    ------
    RuntimeError
        If all login strategies fail.
    """
    creds_path = run_dir / ".credentials.yml"
    username, password = "", ""

    if creds_path.exists():
        try:
            data = yaml.safe_load(creds_path.read_text()) or {}
            nasa = data.get("nasa_earthdata", {})
            username = nasa.get("username", "") or ""
            password = nasa.get("password", "") or ""
        except (yaml.YAMLError, AttributeError):
            logger.warning("Could not parse %s, using default login", creds_path)

    if username and password:
        os.environ["EARTHDATA_USERNAME"] = username
        os.environ["EARTHDATA_PASSWORD"] = password
        logger.info("Using credentials from .credentials.yml")
        auth = earthaccess.login(strategy="environment")
        if auth is not None and auth.authenticated:
            return auth
        logger.warning("Environment-based login failed, falling back to default")

    auth = earthaccess.login()
    if auth is None or not auth.authenticated:
        raise RuntimeError(
            "NASA Earthdata login failed. Either fill in "
            f"{creds_path} or register at "
            "https://urs.earthdata.nasa.gov/users/new"
        )
    return auth
