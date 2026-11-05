"""Offline continuous-account checks; synthetic fixtures do not demonstrate alpha."""

from copy import deepcopy
from decimal import Decimal

import numpy as np
import pytest
from test_monthly_strategy import market

from quant_rsi.research.demo_strategy import (
    _risk_metrics,
    evaluate_monthly_strategy,
    evaluate_session_strategy,
)


def forecasts(data, scores=(0.1, 0.1, 0.1)):
    return [
        {
            "origin_session": data.dates[index],
            "target_session": data.dates[index + 3],
            "reference_price": float(data.closes[index, 0]),
            "predicted_price": float(data.closes[index, 0] * np.exp(score)),
            "observed_price": float(data.closes[index + 3, 0]),
            "predicted_log_return": score,
        }
        for index, score in zip(range(0, 3 * len(scores), 3), scores, strict=True)
    ]


def test_three_day_positive_signals_use_one_costed_account_across_month_boundary():
    data = market()
    rows = forecasts(data)
    result = evaluate_session_strategy(data, "SOXX", rows)
    state = result["strategy"]
    assert state == result["buy_and_hold"]
    assert len(state["fills"]) == 1
    assert state["orders"][0]["target_session"] == data.dates[1] == "2026-02-02"
    assert state["holdings"]["SOXX"]["quantity"] == 99
    assert state["cash"] == "90.10"
    assert state["metrics"]["fees"] == {"commission": "4.95", "slippage": "4.95", "tax": "0.00"}
    assert Decimal(state["metrics"]["net_return"]) == Decimal("-0.00099")
    assert len(state["valuations"]) == 10
    assert result["period"] == {"start": data.dates[0], "end": data.dates[9]}
    assert result["prediction_horizon_sessions"] == 3
    assert result["prediction_horizon"] == "fixed_trading_sessions"
    assert state["metrics"]["sharpe_ratio"] == _risk_metrics(state["valuations"])["sharpe_ratio"]
    assert state["metrics"]["risk_statistics"]["daily_return_count"] == 9
    with pytest.raises(ValueError, match="Monthly forecasts"):
        evaluate_monthly_strategy(data, "SOXX", rows)


def test_revisions_trade_next_open_with_costs_and_without_resetting_capital():
    data = market()
    result = evaluate_session_strategy(data, "SOXX", forecasts(data, (0.1, -0.1, 0.1)))
    state = result["strategy"]
    assert [row["side"] for row in state["fills"]] == ["buy", "sell", "buy"]
    assert [row["target_session"] for row in state["orders"]] == [data.dates[i] for i in (1, 4, 7)]
    assert Decimal(state["metrics"]["net_return"]) == Decimal("-0.00297")
    assert state["cash"] == "70.30"
    assert state["metrics"]["fees"]["commission"] == "14.85"
    assert len(result["buy_and_hold"]["fills"]) == 1


@pytest.mark.parametrize("mutation", ["duplicate", "reverse", "gap", "overlap", "short", "future"])
def test_noncontiguous_or_unobserved_intervals_are_not_replayed(mutation):
    data = market()
    rows = forecasts(data)
    if mutation == "duplicate":
        rows.insert(1, deepcopy(rows[0]))
    elif mutation == "reverse":
        rows.reverse()
    elif mutation == "gap":
        del rows[1]
    elif mutation == "overlap":
        rows[1]["origin_session"] = data.dates[2]
        rows[1]["target_session"] = data.dates[5]
    elif mutation == "short":
        rows[-1]["target_session"] = data.dates[8]
    else:
        rows[-1]["target_session"] = "2026-05-01"
    with pytest.raises(ValueError, match="Session forecasts"):
        evaluate_session_strategy(data, "SOXX", rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_price", 101),
        ("observed_price", None),
        ("predicted_price", 150),
        ("predicted_log_return", float("nan")),
    ],
)
def test_prices_and_return_must_match_retained_inputs(field, value):
    data = market()
    rows = forecasts(data)
    rows[0][field] = value
    with pytest.raises(ValueError, match="prediction|Predictions"):
        evaluate_session_strategy(data, "SOXX", rows)


@pytest.mark.parametrize("horizon", [0, 21, True, 3.0])
def test_invalid_horizons_do_not_silently_reinterpret_predictions(horizon):
    data = market()
    with pytest.raises(ValueError, match="1..20"):
        evaluate_session_strategy(data, "SOXX", forecasts(data), horizon_sessions=horizon)
