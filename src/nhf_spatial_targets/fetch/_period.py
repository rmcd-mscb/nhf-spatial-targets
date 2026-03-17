"""Shared period-parsing utilities for fetch modules."""

from __future__ import annotations


def parse_period(period: str) -> tuple[str, str]:
    """Parse ``"YYYY/YYYY"`` into ``("YYYY-01-01", "YYYY-12-31")``."""
    parts = period.split("/")
    if len(parts) != 2:
        raise ValueError(f"period must be 'YYYY/YYYY', got: {period!r}")
    start_year, end_year = parts
    try:
        start_int, end_int = int(start_year), int(end_year)
    except ValueError:
        raise ValueError(f"period years must be integers, got: {period!r}") from None
    if end_int < start_int:
        raise ValueError(
            f"period end year ({end_year}) is before start year "
            f"({start_year}). Use 'YYYY/YYYY' with start <= end."
        )
    return (f"{start_year}-01-01", f"{end_year}-12-31")


def months_in_period(period: str) -> list[str]:
    """Return list of 'YYYY-MM' strings for every month in the period.

    All 12 months are included for each year (no sub-annual start/end).
    """
    parse_period(period)  # validate format
    parts = period.split("/")
    start_year, end_year = int(parts[0]), int(parts[1])
    months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            months.append(f"{year}-{month:02d}")
    return months


def years_in_period(period: str) -> list[int]:
    """Return list of year integers for every year in the period."""
    parse_period(period)  # validate format
    parts = period.split("/")
    start_year, end_year = int(parts[0]), int(parts[1])
    return list(range(start_year, end_year + 1))


def clamp_period(requested: str, available: str) -> str | None:
    """Clamp *requested* period to the *available* range from the catalog.

    Parameters
    ----------
    requested : str
        User-requested period as ``"YYYY/YYYY"``.
    available : str
        Catalog period, e.g. ``"2000/2013"`` or ``"1980/present"``.
        ``"present"`` is treated as year 9999 (no upper bound).

    Returns
    -------
    str or None
        Clamped period as ``"YYYY/YYYY"``, or ``None`` if there is no
        overlap between the requested and available ranges.
    """
    parse_period(requested)
    req_parts = requested.split("/")
    req_start, req_end = int(req_parts[0]), int(req_parts[1])

    avail_parts = available.split("/")
    if len(avail_parts) != 2:
        raise ValueError(f"available period must be 'YYYY/YYYY', got: {available!r}")
    avail_start = int(avail_parts[0])
    avail_end = (
        9999 if avail_parts[1] == "present" else int(avail_parts[1].split("-")[0])
    )

    clamped_start = max(req_start, avail_start)
    clamped_end = min(req_end, avail_end)

    if clamped_start > clamped_end:
        return None

    return f"{clamped_start}/{clamped_end}"
