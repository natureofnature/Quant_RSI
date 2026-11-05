"""Exact retained trading-session windows; offline calendar fixtures, no model fits."""

from copy import deepcopy
from datetime import date

import numpy as np
import pytest
from test_monthly_rolling import retained as retained_fixture
from test_session_rolling_windows import subset

from quant_rsi.core.artifacts import canonical
from quant_rsi.core.data import MarketData
from quant_rsi.research.rolling_windows import monthly_plan, session_plan, sessions

retained = retained_fixture


def test_126_session_windows_keep_the_same_training_count_across_calendar_months(retained):
    _, data, _, calendar = retained
    plan = session_plan(data, calendar, "2026-08", "2026-10", train_sessions=126)
    assert len(plan) > 15
    for fold in plan:
        expected = [
            row["session_id"]
            for row in sessions(
                calendar,
                date.fromisoformat(fold["training_window"]["start"]),
                date.fromisoformat(fold["origin_session"]),
            )
        ]
        assert expected == fold["history_sessions"]
        assert len(expected) == fold["training_window_sessions"] == 126
        assert expected[-1] == fold["origin_session"] == fold["training_window"]["end"]
        assert expected[0] == fold["training_window"]["start"]
        assert len(fold["training_indices"]) == len(fold["training_origins"]) == 93
        assert fold["training_indices"] == list(range(30, 123))
        assert fold["training_targets"][-1] == fold["origin_session"]
        assert fold["unmatured_origins"] == expected[-3:]
    for previous, current in zip(plan[:-1], plan[1:], strict=True):
        assert current["history_sessions"][:-3] == previous["history_sessions"][3:]
        assert current["origin_session"] == previous["target_session"]
    assert "2026-07-03" not in plan[0]["history_sessions"]
    assert any(
        "2026-09-07" > f["history_sessions"][0] and "2026-09-07" < f["origin_session"] for f in plan
    )
    assert all("2026-09-07" not in f["history_sessions"] for f in plan)


@pytest.mark.parametrize("length,horizon", [(61, 1), (63, 3), (80, 20), (126, 3)])
def test_session_count_and_horizon_define_actual_mature_fit_rows(retained, length, horizon):
    _, data, _, calendar = retained
    plan = session_plan(
        data, calendar, "2026-08", "2026-08", horizon_sessions=horizon, train_sessions=length
    )
    assert plan
    for fold in plan:
        assert len(fold["history_sessions"]) == length
        assert len(fold["training_indices"]) == length - 30 - horizon
        assert len(fold["unmatured_origins"]) == horizon


def test_explicit_session_window_does_not_require_unneeded_calendar_years(retained):
    _, data, _, calendar = retained
    assert set(calendar["holidays"]) == {2026}
    narrow = session_plan(data, calendar, "2026-07", "2026-07", train_months=24, train_sessions=63)
    assert narrow[0]["training_window_sessions"] == 63
    assert narrow[0]["training_window"]["start"] >= "2026-01-01"


@pytest.mark.parametrize("missing", ["2026-04-01", "2026-06-30", "2026-07-31"])
def test_missing_session_cannot_be_replaced_by_an_extra_older_row(retained, missing):
    _, data, _, calendar = retained
    incomplete = subset(data, [i for i, day in enumerate(data.dates) if day != missing])
    with pytest.raises(ValueError, match="Incomplete training history/calendar"):
        session_plan(incomplete, calendar, "2026-08", "2026-08", train_sessions=126)


def test_missing_data_before_the_requested_window_does_not_invalidate_it(retained):
    _, data, _, calendar = retained
    original = session_plan(data, calendar, "2026-08", "2026-08", train_sessions=63)
    start = original[0]["training_window"]["start"]
    reduced = subset(data, [i for i, day in enumerate(data.dates) if day >= start])
    assert session_plan(reduced, calendar, "2026-08", "2026-08", train_sessions=63) == original


@pytest.mark.parametrize("length", [126, 445])
def test_not_enough_retained_history_is_an_explicit_error(retained, length):
    _, data, _, calendar = retained
    short = subset(data, [i for i, day in enumerate(data.dates) if day >= "2026-05-01"])
    with pytest.raises(ValueError, match="Insufficient retained training sessions"):
        session_plan(short, calendar, "2026-08", "2026-08", train_sessions=length)


@pytest.mark.parametrize("length", [0, 60, 62, 446, True, 126.0, "126"])
def test_training_count_must_be_a_bounded_integer_with_enough_mature_labels(retained, length):
    _, data, _, calendar = retained
    with pytest.raises(ValueError, match="61..445.*60 \\+ horizon"):
        session_plan(data, calendar, "2026-08", "2026-08", train_sessions=length)


def test_future_prices_do_not_change_first_session_window_or_its_training_labels(retained):
    _, data, _, calendar = retained
    original = session_plan(data, calendar, "2026-08", "2026-08", train_sessions=126)
    opens, closes = data.opens.copy(), data.closes.copy()
    future = [i for i, day in enumerate(data.dates) if day > original[0]["origin_session"]]
    opens[future] *= 4
    closes[future] *= 7
    changed = MarketData(data.dates, data.instruments, opens, closes)
    assert session_plan(changed, calendar, "2026-08", "2026-08", train_sessions=126) == original
    indices = [data.dates.index(day) for day in original[0]["history_sessions"]]
    train = np.asarray(original[0]["training_indices"])
    before = np.log(data.closes[indices][train + 3] / data.closes[indices][train])
    after = np.log(changed.closes[indices][train + 3] / changed.closes[indices][train])
    np.testing.assert_array_equal(before, after)


def test_explicit_none_preserves_calendar_mode_and_monthly_output(retained):
    _, data, _, calendar = retained
    monthly = canonical(monthly_plan(data, calendar, "2026-07", "2026-07"))
    calendar_plan = session_plan(data, calendar, "2026-07", "2026-07")
    assert session_plan(data, calendar, "2026-07", "2026-07", train_sessions=None) == calendar_plan
    assert all("training_window_sessions" not in fold for fold in calendar_plan)
    assert canonical(monthly_plan(data, calendar, "2026-07", "2026-07")) == monthly


def test_session_window_checks_its_published_calendar_not_just_row_count(retained):
    _, data, _, calendar = retained
    changed = deepcopy(calendar)
    changed["holidays"][2026].add(date(2026, 4, 1))
    with pytest.raises(ValueError, match="Incomplete training history/calendar"):
        session_plan(data, changed, "2026-08", "2026-08", train_sessions=126)
