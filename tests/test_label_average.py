"""The 30-session label is the trailing five-close average, not the single close."""

from datetime import date, timedelta

import numpy as np
import pytest

from quant_rsi.core.data import MarketData
from quant_rsi.research.demo_features import label_levels
from quant_rsi.research.demo_strategy import evaluate_strategy


def test_average_label_is_the_five_closes_ending_at_the_endpoint():
    closes = np.arange(1, 41, dtype=float)
    levels = label_levels(closes, [0, 2], 30, 5)
    assert levels[0] == pytest.approx(np.mean(closes[26:31]))
    assert levels[1] == pytest.approx(np.mean(closes[28:33]))
    assert label_levels(closes, [4], 5, 1)[0] == pytest.approx(closes[9])
    assert label_levels(closes, [0], 6, 3)[0] == pytest.approx(np.mean(closes[4:7]))


def test_strategy_accepts_the_average_and_rejects_the_endpoint_close():
    sessions = 40
    dates = tuple((date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(sessions))
    closes = np.arange(10, 10 + sessions, dtype=float).reshape(-1, 1)
    data = MarketData(dates, ("KO",), closes.copy(), closes.copy())
    level = float(label_levels(closes[:, 0], [0], 30, 5)[0])
    score = 0.01
    row = dict(
        origin_session=dates[0],
        target_session=dates[30],
        reference_price=float(closes[0, 0]),
        predicted_price=float(closes[0, 0] * np.exp(score)),
        observed_price=level,
        predicted_log_return=score,
    )
    result = evaluate_strategy(data, "KO", [row], horizon_sessions=30, label_average=5)
    assert result["prediction_horizon_sessions"] == 30
    with pytest.raises(ValueError):
        evaluate_strategy(
            data,
            "KO",
            [row | {"observed_price": float(closes[30, 0])}],
            horizon_sessions=30,
            label_average=5,
        )
