"""Tests for shared period parsing utilities."""

from __future__ import annotations

import pytest

from nhf_spatial_targets.fetch._period import (
    clamp_period,
    months_in_period,
    parse_period,
    years_in_period,
)


def test_parse_period_valid():
    assert parse_period("2005/2006") == ("2005-01-01", "2006-12-31")


def test_parse_period_single_year():
    assert parse_period("2010/2010") == ("2010-01-01", "2010-12-31")


def test_parse_period_missing_slash():
    with pytest.raises(ValueError, match="YYYY/YYYY"):
        parse_period("2010")


def test_parse_period_non_numeric():
    with pytest.raises(ValueError, match="integers"):
        parse_period("abc/def")


def test_parse_period_reversed():
    with pytest.raises(ValueError, match="before start year"):
        parse_period("2015/2010")


def test_months_in_period():
    months = months_in_period("2010/2010")
    assert len(months) == 12
    assert months[0] == "2010-01"
    assert months[-1] == "2010-12"


def test_months_in_period_multi_year():
    months = months_in_period("2009/2010")
    assert len(months) == 24
    assert months[0] == "2009-01"
    assert months[-1] == "2010-12"


def test_years_in_period():
    years = years_in_period("2005/2008")
    assert years == [2005, 2006, 2007, 2008]


def test_years_in_period_single():
    years = years_in_period("2010/2010")
    assert years == [2010]


def test_clamp_period_within_range():
    assert clamp_period("2000/2010", "1980/2020") == "2000/2010"


def test_clamp_period_clamps_end():
    assert clamp_period("2000/2020", "2000/2013") == "2000/2013"


def test_clamp_period_clamps_start():
    assert clamp_period("1990/2010", "2000/2013") == "2000/2010"


def test_clamp_period_clamps_both():
    assert clamp_period("1990/2020", "2000/2013") == "2000/2013"


def test_clamp_period_no_overlap():
    assert clamp_period("1990/1995", "2000/2013") is None


def test_clamp_period_present():
    assert clamp_period("2000/2025", "1980/present") == "2000/2025"


def test_clamp_period_present_clamps_start():
    assert clamp_period("1970/2010", "1980/present") == "1980/2010"


def test_clamp_period_handles_date_suffix():
    """Available period like '1980/2016-02-29' uses the year portion."""
    assert clamp_period("2000/2020", "1980/2016-02-29") == "2000/2016"
