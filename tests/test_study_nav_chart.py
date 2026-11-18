"""The hypothetical NAV chart is a display of retained accounts, not a new account."""

import re

from quant_rsi.research.study_report import _equity_chart, _selected_prediction


def _account(points, fills=()):
    return {
        "valuations": [{"session_id": day, "nav": nav} for day, nav in points],
        "fills": [{"resolved_at": day, "side": side} for day, side in fills],
    }


def _bundle(strategy, hold, fills=()):
    return {"strategy": _account(strategy, fills), "buy_and_hold": _account(hold)}


WINDOWS = {
    "training": {"start": "2025-03-02", "end": "2025-12-31"},
    "development": {"start": "2026-01-05", "end": "2026-04-16"},
    "test": {"start": "2026-04-27", "end": "2026-07-22"},
}


def _chart():
    selected = {
        "strategy": _bundle(
            (("2026-01-05", 10000), ("2026-04-24", 11000)),
            (("2026-01-05", 10000), ("2026-04-24", 10500)),
        )
    }
    terminal = {
        "status": "completed",
        "strategy": _bundle(
            (
                ("2026-04-27", 10000),
                ("2026-05-01", 10000),
                ("2026-07-30", 10100),
            ),
            (
                ("2026-04-27", 10000),
                ("2026-05-01", 9800),
                ("2026-07-30", 9700),
            ),
            fills=(("2026-05-01T13:30:00Z", "buy"), ("2026-07-30T13:30:00Z", "sell")),
        ),
    }
    return _equity_chart(selected, terminal, WINDOWS)


def _polylines(html):
    return re.findall(
        r"<polyline data-account='([^']+)' data-name='([^']+)' points='([^']+)'",
        html,
    )


def _coords(points):
    return [tuple(float(part) for part in pair.split(",")) for pair in points.split()]


def test_span_chart_keeps_three_bands_and_separate_accounts():
    html = _chart()
    assert "Hypothetical account" in html
    assert "not a live quote" in html
    assert "not evidence of a tradable edge" in html
    assert "candle" not in html.lower()
    span, test = html.split("Reserved test account", 1)
    assert "data-side" not in span
    for name in ("training", "development", "test"):
        assert f"data-window='{name}'" in span
    train = re.search(r"<rect data-window='training' x='([0-9.]+)'[^>]*width='([0-9.]+)'", span)
    strategy = [points for account, name, points in _polylines(span) if name == "Selected strategy"]
    accounts = [
        account for account, name, _points in _polylines(span) if name == "Selected strategy"
    ]
    assert accounts == ["development", "test"]
    development, held_out = (_coords(points) for points in strategy)
    assert len(development) == 2
    assert len(held_out) == 3
    assert float(train.group(1)) + float(train.group(2)) < development[0][0] < held_out[0][0]
    assert development[1][1] < development[0][1]
    assert held_out[0][1] == held_out[1][1]
    assert held_out[2][1] < held_out[0][1]
    assert "<script>" not in html
    assert "class='hit'" in span and "hit:hover .cross" in span
    assert "Selected strategy 11000.00" in span
    assert "data-grid='date'" in span and "data-grid='nav'" in span


def test_reserved_test_panel_marks_buys_and_sells():
    _span, test = _chart().split("Reserved test account", 1)
    assert "data-side='buy'" in test
    assert "data-side='sell'" in test
    assert "Cash days stay flat" in test
    assert "without refitting" in test
    strategy = [
        points for _account, name, points in _polylines(test) if name == "Selected strategy"
    ]
    assert len(strategy) == 1
    assert len(_coords(strategy[0])) == 3
    assert "Buy" in test and "Sell" in test


def test_missing_terminal_keeps_the_test_band_without_a_test_line():
    selected = {
        "strategy": _bundle(
            (("2026-01-05", 10000), ("2026-04-24", 11000)),
            (("2026-01-05", 10000), ("2026-04-24", 10500)),
        )
    }
    html = _equity_chart(selected, {"status": "unavailable"}, WINDOWS)
    assert "data-window='test'" in html
    assert "data-account='test'" not in html
    assert "No completed reserved-test account" in html
    assert "data-side" not in html


def test_chart_uses_the_selected_round_not_the_last_trial():
    attempts = [{"prediction": {"id": 1}}, {"prediction": {"id": 2}}, {"prediction": {"id": 3}}]
    chosen = _selected_prediction(
        attempts[0]["prediction"],
        attempts[-1]["prediction"],
        {"attempts": attempts, "selected_round": 2},
        {"selected_attempt": "revised"},
    )
    assert chosen["id"] == 2
    assert (
        _selected_prediction(
            {"id": "initial"},
            {"id": "revised"},
            None,
            {"selected_attempt": "initial"},
        )["id"]
        == "initial"
    )
