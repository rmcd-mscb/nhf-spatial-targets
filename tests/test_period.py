"""Tests for shared period parsing utilities."""

from __future__ import annotations

import pytest

from nhf_spatial_targets.fetch._period import (
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
