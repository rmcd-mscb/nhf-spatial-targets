"""Logging configuration for nhf-spatial-targets CLI."""

from __future__ import annotations

import logging


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger with RichHandler.

    Parameters
    ----------
    verbose : bool
        If True, set level to DEBUG and show Python source file/line in log output.
        If False, set level to INFO.
    """
    from rich.logging import RichHandler

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=verbose)],
        force=True,
    )
    # Suppress noisy third-party loggers
    for name in ("earthaccess", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
