"""Offline calendar and label-isolation checks for fixed-session research."""

from copy import deepcopy
from datetime import date

import numpy as np
import pytest
from test_monthly_rolling import retained as retained_fixture

from quant_rsi.core.artifacts import canonical
from quant_rsi.core.data import MarketData
from quant_rsi.research.rolling_windows import monthly_plan, period_id, session_plan, sessions

retained = retained_fixture


def subset(data, indices):
    indices = np.asarray(indices, dtype=int)
    return MarketData(
        tuple(data.dates[i] for i in indices),
        data.instruments,
        data.opens[indices],
        data.closes[indices],
    )


def test_three_session_plan_respects_holiday_contiguity_and_mature_labels(retained):
    _, data, _, calendar = retained
    plan = session_plan(data, calendar, "2026-07", "2026-08")
    first = plan[0]
    assert first["origin_session"] == "2026-06-30"
    assert first["target_session"] == "2026-07-06"  # July 3 is an exchange holiday.
    assert first["training_window"] == {"start": "2026-01-01", "end": "2026-06-30"}
    assert plan[1]["training_window"] == {"start": "2026-01-07", "end": "2026-07-06"}
    assert plan[1]["target_session"] == "2026-07-09"
    assert any(f["origin_session"][:7] != f["target_session"][:7] for f in plan[1:])
    for previous, current in zip(plan[:-1], plan[1:], strict=True):
        assert current["origin_session"] == previous["target_session"]
        assert current["information_cutoff"] == previous["target_close_at"]
    for fold in plan:
        assert "month" not in fold and fold["period_kind"] == "trading_sessions"
        assert period_id(fold) == fold["target_session"]
        assert fold["horizon_sessions"] == 3 and fold["warmup_sessions"] == 30
        history = fold["history_sessions"]
        assert fold["training_indices"] == list(range(30, len(history) - 3))
        assert fold["training_targets"][-1] == fold["origin_session"]
        assert fold["unmatured_origins"] == history[-3:]
        assert all(target <= fold["origin_session"] for target in fold["training_targets"])
        assert (
            data.dates.index(fold["target_session"]) - data.dates.index(fold["origin_session"]) == 3
        )


def test_only_complete_requested_intervals_are_kept_and_monthly_output_is_unchanged(retained):
    _, data, _, calendar = retained
    monthly = monthly_plan(data, calendar, "2026-07", "2026-07")
    original = canonical(monthly)
    plan = session_plan(data, calendar, "2026-07", "2026-07")
    july = sessions(calendar, date(2026, 7, 1), date(2026, 7, 31))
    assert len(plan) == len(july) // 3 == 7
    assert plan[-1]["target_session"] == "2026-07-30"
    assert all(f["horizon_sessions"] == 3 for f in plan)
    assert period_id(monthly[0]) == "2026-07"
    assert canonical(monthly_plan(data, calendar, "2026-07", "2026-07")) == original


def test_one_pending_target_stops_planning_without_shortening_it(retained):
    _, data, _, calendar = retained
    partial = subset(data, [i for i, day in enumerate(data.dates) if day <= "2026-07-07"])
    plan = session_plan(partial, calendar, "2026-07", "2026-09")
    assert [f["target_session"] for f in plan] == ["2026-07-06", "2026-07-09"]
    assert plan[-1]["origin_session"] == "2026-07-06"
    assert plan[-1]["target_session"] not in partial.dates
    assert plan[-1]["horizon_sessions"] == 3


def test_unobserved_price_values_cannot_change_windows_or_training_labels(retained):
    _, data, _, calendar = retained
    plan = session_plan(data, calendar, "2026-07", "2026-08")
    closes, opens = data.closes.copy(), data.opens.copy()
    future = [i for i, day in enumerate(data.dates) if day > "2026-06-30"]
    closes[future] *= 7
    opens[future] *= 11
    changed = MarketData(data.dates, data.instruments, opens, closes)
    assert session_plan(changed, calendar, "2026-07", "2026-08") == plan
    first = plan[0]
    indices = [data.dates.index(day) for day in first["history_sessions"]]
    train = np.asarray(first["training_indices"])
    original_labels = np.log(data.closes[indices][train + 3] / data.closes[indices][train])
    changed_labels = np.log(changed.closes[indices][train + 3] / changed.closes[indices][train])
    np.testing.assert_array_equal(original_labels, changed_labels)


@pytest.mark.parametrize("missing", ["2026-05-04", "2026-07-02", "2026-07-07"])
def test_missing_training_or_observed_forecast_prices_are_rejected(retained, missing):
    _, data, _, calendar = retained
    rows = [i for i, day in enumerate(data.dates) if day != missing and day <= "2026-07-08"]
    partial = subset(data, rows)
    with pytest.raises(ValueError, match="Incomplete"):
        session_plan(partial, calendar, "2026-07", "2026-07")


def test_half_open_calendar_window_clips_short_month_start_day(retained):
    _, data, _, calendar = retained
    plan = session_plan(data, calendar, "2026-08", "2026-08", horizon_sessions=1)
    fold = next(f for f in plan if f["origin_session"] == "2026-08-28")
    assert fold["training_window"] == {"start": "2026-02-28", "end": "2026-08-28"}
    assert fold["history_sessions"][0] == "2026-03-02"


@pytest.mark.parametrize("horizon", [0, 21, True, 3.0])
def test_invalid_horizon_is_rejected(retained, horizon):
    _, data, _, calendar = retained
    with pytest.raises(ValueError, match="1..20 horizon"):
        session_plan(data, calendar, "2026-07", "2026-07", horizon_sessions=horizon)


@pytest.mark.parametrize("months", [0, 25, True, 6.0])
def test_invalid_training_width_is_rejected(retained, months):
    _, data, _, calendar = retained
    with pytest.raises(ValueError, match="1..24 training"):
        session_plan(data, calendar, "2026-07", "2026-07", train_months=months)


def test_calendar_range_and_operation_bounds_fail_before_training(retained):
    _, data, _, calendar = retained
    with pytest.raises(ValueError, match="Ordered months"):
        session_plan(data, calendar, "2026-08", "2026-07")
    with pytest.raises(ValueError, match="24 calendar"):
        session_plan(data, calendar, "2026-07", "2028-07")
    with pytest.raises(ValueError, match="240 session forecasts"):
        session_plan(data, calendar, "2026-01", "2026-12", horizon_sessions=1)
    with pytest.raises(ValueError, match="No complete forecast"):
        session_plan(data, calendar, "2026-02", "2026-02", horizon_sessions=20)
    with pytest.raises(ValueError, match="Insufficient mature"):
        session_plan(data, calendar, "2026-07", "2026-07", train_months=1)


def test_missing_published_calendar_year_is_not_invented(retained):
    _, data, _, calendar = retained
    changed = deepcopy(calendar)
    changed["holidays"].pop(2026)
    with pytest.raises(ValueError, match="calendar"):
        session_plan(data, changed, "2026-07", "2026-07")
