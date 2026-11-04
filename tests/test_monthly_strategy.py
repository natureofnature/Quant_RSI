"""One costed account across observed variable-length calendar-month forecasts."""

from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pytest

from quant_rsi.core.data import MarketData
from quant_rsi.research.demo_strategy import _risk_metrics, evaluate_monthly_strategy


def market():
    day = date(2026, 1, 30)
    dates = []
    while day <= date(2026, 4, 30):
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += timedelta(days=1)
    return MarketData(
        tuple(dates),
        ("SOXX",),
        np.full((len(dates), 1), 100.0),
        np.full((len(dates), 1), 100.0),
    )


def forecasts(data, scores=(0.1, 0.1, 0.1)):
    dates = ["2026-01-30", "2026-02-27", "2026-03-31", "2026-04-30"]
    return [
        {
            "origin_session": origin,
            "target_session": target,
            "reference_price": float(data.closes[data.dates.index(origin), 0]),
            "predicted_price": float(data.closes[data.dates.index(origin), 0] * np.exp(score)),
            "observed_price": float(data.closes[data.dates.index(target), 0]),
            "predicted_log_return": score,
        }
        for origin, target, score in zip(dates, dates[1:], scores, strict=False)
    ]


def test_positive_monthly_signals_preserve_one_account_and_one_purchase():
    data = market()
    result = evaluate_monthly_strategy(data, "SOXX", forecasts(data))
    state = result["strategy"]
    assert state == result["buy_and_hold"]
    assert len(state["fills"]) == 1
    assert state["orders"][0]["target_session"] == "2026-02-02"
    assert state["cash"] == "90.10"
    assert state["holdings"]["SOXX"]["quantity"] == 99
    assert state["metrics"]["fees"] == {"commission": "4.95", "slippage": "4.95", "tax": "0.00"}
    assert Decimal(state["metrics"]["net_return"]) == Decimal("-0.00099")
    assert len(state["valuations"]) == len(data.dates)
    assert result["prediction_horizon_sessions"] == [20, 22, 22]
    assert result["period"] == {"start": "2026-01-30", "end": "2026-04-30"}
    expected = _risk_metrics(state["valuations"])
    assert state["metrics"]["sharpe_ratio"] == expected["sharpe_ratio"]
    assert state["metrics"]["risk_statistics"]["daily_return_count"] == len(data.dates) - 1


def test_monthly_cash_switches_fill_next_open_without_resetting_cash():
    data = market()
    result = evaluate_monthly_strategy(data, "SOXX", forecasts(data, (0.1, -0.1, 0.1)))
    state = result["strategy"]
    assert [row["side"] for row in state["fills"]] == ["buy", "sell", "buy"]
    assert [row["target_session"] for row in state["orders"]] == [
        "2026-02-02",
        "2026-03-02",
        "2026-04-01",
    ]
    assert Decimal(state["metrics"]["net_return"]) == Decimal("-0.00297")
    assert state["cash"] == "70.30"
    assert state["metrics"]["fees"]["commission"] == "14.85"
    assert len(result["buy_and_hold"]["fills"]) == 1
    assert state["valuations"][-1]["session_id"] == "2026-04-30"


@pytest.mark.parametrize("mutation", ["duplicate", "reverse", "gap", "midmonth", "future"])
def test_invalid_month_sequences_are_rejected_before_replay(mutation):
    data = market()
    rows = forecasts(data)
    if mutation == "duplicate":
        rows.insert(1, deepcopy(rows[0]))
    elif mutation == "reverse":
        rows.reverse()
    elif mutation == "gap":
        del rows[1]
    elif mutation == "midmonth":
        rows[0]["target_session"] = "2026-02-26"
    else:
        rows[-1]["target_session"] = "2026-05-29"
    with pytest.raises(ValueError, match="Monthly forecasts"):
        evaluate_monthly_strategy(data, "SOXX", rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_price", 101.0),
        ("observed_price", 101.0),
        ("predicted_price", 150.0),
        ("predicted_log_return", float("nan")),
    ],
)
def test_monthly_forecasts_must_match_retained_prices_and_return(field, value):
    data = market()
    rows = forecasts(data)
    rows[0][field] = value
    with pytest.raises(ValueError, match="Predictions must be chronological"):
        evaluate_monthly_strategy(data, "SOXX", rows)


def test_unobserved_month_is_not_silently_shortened_to_current_price():
    data = market()
    rows = forecasts(data)
    rows[-1]["observed_price"] = None
    with pytest.raises(ValueError, match="Invalid prediction fields"):
        evaluate_monthly_strategy(data, "SOXX", rows)
