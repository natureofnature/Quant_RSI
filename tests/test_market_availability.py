"""Date-only availability follows the source's timezone, conservatively when unknown."""

from datetime import UTC, date, datetime, timedelta

import pytest

from quant_rsi.market.availability import DATE_ONLY_RULE, date_only_available_at


@pytest.mark.parametrize("day", ["2024-01-01", "2024-02-29", "2024-03-10", "2024-12-31"])
def test_unknown_timezone_is_next_day_noon_utc_as_before(day):
    # The rule discovery and realization used before this module, unchanged.
    previous = datetime.combine(
        date.fromisoformat(day) + timedelta(days=1), datetime.min.time(), UTC
    ) + timedelta(hours=12)
    assert date_only_available_at(day) == previous
    assert date_only_available_at(date.fromisoformat(day)) == previous


@pytest.mark.parametrize(
    ("day", "zone", "expected"),
    [
        ("2024-01-15", "America/New_York", "2024-01-16T05:00:00+00:00"),  # EST
        ("2024-07-15", "America/New_York", "2024-07-16T04:00:00+00:00"),  # EDT
        ("2024-03-09", "America/New_York", "2024-03-10T05:00:00+00:00"),  # day before DST
        ("2024-07-15", "Asia/Taipei", "2024-07-15T16:00:00+00:00"),
        ("2024-07-15", "UTC", "2024-07-16T00:00:00+00:00"),
    ],
)
def test_declared_timezone_uses_its_next_local_midnight(day, zone, expected):
    assert date_only_available_at(day, zone).isoformat() == expected


def test_unknown_timezone_is_never_earlier_than_any_declared_zone():
    unknown = date_only_available_at("2024-07-15")
    for zone in ("Pacific/Pago_Pago", "America/Los_Angeles", "Europe/London", "Asia/Tokyo"):
        assert date_only_available_at("2024-07-15", zone) <= unknown


def test_rule_is_versioned():
    assert DATE_ONLY_RULE == "date_only_next_day_start_v1"


def test_lookback_window_is_half_open_before_the_instant():
    from quant_rsi.market.availability import in_lookback_window

    instant = datetime(2024, 7, 15, 20, tzinfo=UTC)
    window = timedelta(days=3)
    assert in_lookback_window(instant - window, instant, window)
    assert in_lookback_window(instant - timedelta(seconds=1), instant, window)
    assert not in_lookback_window(instant, instant, window)
    assert not in_lookback_window(instant - window - timedelta(seconds=1), instant, window)
