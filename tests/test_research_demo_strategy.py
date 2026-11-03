"""Small accounting oracles; invented prices are not evidence of investment value."""

from copy import deepcopy
from datetime import date, timedelta
from decimal import ROUND_HALF_EVEN, Decimal

import numpy as np
import pytest

from quant_rsi.core.data import MarketData, synthetic_data
from quant_rsi.core.schemas import DataSpec
from quant_rsi.research.demo_strategy import _risk_metrics, evaluate_strategy


def market():
    return MarketData(
        tuple((date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(12)),
        ("KO",),
        np.full((12, 1), 100.0),
        np.full((12, 1), 100.0),
    )


def prediction(data, origin=0, score=0.1):
    reference = float(data.closes[origin, 0])
    return dict(
        origin_session=data.dates[origin],
        target_session=data.dates[origin + 5],
        reference_price=reference,
        predicted_price=reference * np.exp(score),
        observed_price=float(data.closes[origin + 5, 0]),
        predicted_log_return=score,
    )


def test_constant_positive_matches_buyhold_costs_and_preserves_end_inventory():
    data = market()
    result = evaluate_strategy(data, "KO", [prediction(data, i) for i in range(3)])
    strategy, baseline = result["strategy"], result["buy_and_hold"]
    assert strategy == baseline
    assert strategy["cash"] == "90.10"
    assert strategy["holdings"]["KO"]["quantity"] == 99
    assert strategy["metrics"]["fees"] == {"commission": "4.95", "tax": "0.00", "slippage": "4.95"}
    assert Decimal(strategy["metrics"]["net_return"]) == Decimal("-0.00099")
    assert Decimal(strategy["metrics"]["turnover"]) == Decimal("0.99")
    assert len(strategy["fills"]) == 1
    assert len(strategy["valuations"]) == 8
    assert strategy["orders"][0]["target_session"] == data.dates[1]
    assert result["period"] == {"start": data.dates[0], "end": data.dates[7]}
    assert "hypothetical" in result["evidence_mode"]
    # One cost loss followed by six flat days; the initial cash mark is not a return.
    assert strategy["metrics"]["sharpe_ratio"] == pytest.approx(-6)
    assert strategy["metrics"]["annualized_volatility"] == pytest.approx(0.00099 * 6)
    assert strategy["metrics"]["risk_statistics"]["daily_return_count"] == 7


def test_negative_then_positive_then_negative_changes_next_open_positions():
    data = market()
    result = evaluate_strategy(
        data, "KO", [prediction(data, i, score) for i, score in enumerate((-0.1, 0.1, -0.1))]
    )
    state = result["strategy"]
    assert [fill["side"] for fill in state["fills"]] == ["buy", "sell"]
    assert [order["target_session"] for order in state["orders"]] == [data.dates[2], data.dates[3]]
    assert state["holdings"]["KO"]["quantity"] == 0
    assert state["cash"] == "9980.20"
    assert Decimal(state["metrics"]["net_return"]) == Decimal("-0.00198")
    assert len(result["buy_and_hold"]["fills"]) == 1


@pytest.mark.parametrize("score", [-0.1, 0.0])
def test_nonpositive_remains_cash_but_buyhold_still_enters(score):
    data = market()
    result = evaluate_strategy(data, "KO", [prediction(data, score=score)])
    assert result["strategy"]["fills"] == []
    assert Decimal(result["strategy"]["metrics"]["net_return"]) == 0
    assert len(result["buy_and_hold"]["fills"]) == 1
    metrics = result["strategy"]["metrics"]
    assert metrics["sharpe_ratio"] is None
    assert metrics["annualized_volatility"] == 0
    assert metrics["risk_statistics"]["status"] == "zero_volatility"


def test_sharpe_uses_daily_nav_between_sparse_signals_not_overlapping_forecast_returns():
    data = market()
    data.closes[1:, 0] = 110
    result = evaluate_strategy(data, "KO", [prediction(data)], commission_bps="0", slippage_bps="0")
    # Daily net returns are 10%, 0%, 0%, 0%, 0%, even though there is only one forecast.
    metrics = result["strategy"]["metrics"]
    assert metrics["sharpe_ratio"] == pytest.approx(np.sqrt(50.4))
    assert metrics["annualized_volatility"] == pytest.approx(np.sqrt(0.504))
    assert metrics["risk_statistics"]["daily_return_count"] == 5
    assert metrics["risk_statistics"]["annual_risk_free_rate"] == 0
    assert metrics["risk_statistics"]["annualization_sessions"] == 252
    assert metrics["risk_statistics"]["standard_deviation_ddof"] == 1
    assert metrics == result["buy_and_hold"]["metrics"]


@pytest.mark.parametrize(
    "nav,status,count",
    [
        ([], "insufficient_observations", 0),
        ([100, 101], "insufficient_observations", 1),
        ([100, None, 101], "missing_or_nonpositive_nav", 0),
        ([100, 0, 101], "missing_or_nonpositive_nav", 0),
    ],
)
def test_unavailable_risk_statistics_do_not_bridge_gaps_or_emit_infinity(nav, status, count):
    metrics = _risk_metrics([{"nav": value} for value in nav])
    assert metrics["sharpe_ratio"] is None
    assert metrics["annualized_volatility"] is None
    assert metrics["risk_statistics"]["status"] == status
    assert metrics["risk_statistics"]["daily_return_count"] == count


def test_next_open_gap_changes_fill_without_changing_close_sized_order():
    data = market()
    before = evaluate_strategy(data, "KO", [prediction(data)])
    data.opens[1, 0] = 200
    after = evaluate_strategy(data, "KO", [prediction(data)])
    assert before["strategy"]["orders"][0]["submitted_quantity"] == 99
    assert after["strategy"]["orders"][0]["submitted_quantity"] == 99
    assert before["strategy"]["fills"][0]["quantity"] == 99
    assert after["strategy"]["fills"][0]["quantity"] == 49
    assert Decimal(after["strategy"]["fills"][0]["price"]) == 200
    assert Decimal(after["strategy"]["cash"]) >= 0


def test_drawdown_uses_every_close_between_signals_and_includes_starting_capital():
    data = market()
    data.closes[2, 0] = 90
    result = evaluate_strategy(data, "KO", [prediction(data)], commission_bps="0", slippage_bps="0")
    assert Decimal(result["strategy"]["metrics"]["net_return"]) == 0
    assert Decimal(result["strategy"]["metrics"]["max_drawdown"]) == Decimal("-0.1")
    assert [row["session_id"] for row in result["strategy"]["valuations"]] == list(data.dates[:6])


@pytest.mark.parametrize(
    "changes",
    [
        {"predicted_log_return": float("nan")},
        {"predicted_price": -1},
        {"reference_price": 101},
        {"observed_price": 101},
        {"target_session": "2026-01-07"},
        {"origin_session": "absent"},
        {"predicted_log_return": 0.2},
    ],
)
def test_invalid_or_misaligned_predictions_are_rejected(changes):
    data = market()
    with pytest.raises(ValueError):
        evaluate_strategy(data, "KO", [prediction(data) | changes])


def test_duplicate_reversed_empty_or_wrong_symbol_inputs_are_rejected():
    data = market()
    row = prediction(data)
    for rows in ([], [row, deepcopy(row)], [prediction(data, 1), row]):
        with pytest.raises(ValueError):
            evaluate_strategy(data, "KO", rows)
    with pytest.raises(ValueError):
        evaluate_strategy(data, "OTHER", [row])


def test_actual_synthetic_prices_round_only_account_quotes_and_preserve_model_inputs():
    data = synthetic_data(
        DataSpec(source="synthetic", sessions=128, instruments=1, train_cutoff=100, seed=7)
    )
    original_opens, original_closes = data.opens.copy(), data.closes.copy()
    result = evaluate_strategy(
        data, data.instruments[0], [prediction(data, origin) for origin in range(81, 101)]
    )
    assert len(result["strategy"]["valuations"]) == 25
    fill = result["strategy"]["fills"][0]
    expected = Decimal(str(float(data.opens[82, 0]))).quantize(
        Decimal("0.000000000001"), rounding=ROUND_HALF_EVEN
    )
    assert Decimal(fill["price"]) == expected
    assert Decimal(fill["price"]).as_tuple().exponent >= -12
    assert Decimal(result["strategy"]["cash"]) >= 0
    assert "12 decimal places" in result["execution_note"]
    np.testing.assert_array_equal(data.opens, original_opens)
    np.testing.assert_array_equal(data.closes, original_closes)


def test_quote_that_rounds_to_zero_is_rejected_by_account_contract():
    data = market()
    data.opens[:] = data.closes[:] = 1e-14
    with pytest.raises(ValueError, match="must be positive"):
        evaluate_strategy(data, "KO", [prediction(data)])


@pytest.mark.parametrize("cost", ["5", "0", "400"])
def test_cost_stress_replays_fixed_predictions_and_preserves_bounds(monkeypatch, cost):
    from quant_rsi.core.artifacts import canonical, digest
    from quant_rsi.research import study_cost_stress

    data = market()
    rows = [prediction(data, i) for i in range(3)]
    policy = dict(
        commission_bps=cost,
        slippage_bps=cost,
        initial_cash="10000",
        threshold=0.0,
        horizon_sessions=5,
    )
    baseline = evaluate_strategy(data, "KO", rows, **policy)
    selected = {"factor_ids": ["momentum_3"], "development": {"rows": rows}, "strategy": baseline}
    before = canonical(selected)
    replays = []

    def counted(data, symbol, predictions, **settings):
        assert canonical(predictions) == canonical(rows)
        replays.append(settings)
        return evaluate_strategy(data, symbol, predictions, **settings)

    monkeypatch.setattr(study_cost_stress, "evaluate_strategy", counted)
    result = study_cost_stress.evaluate_cost_stress(
        data, "KO", selected, None, selected_round=2, policy=policy
    )
    assert canonical(selected) == before
    assert result["extra_fits"] == 0 and result["selected_round"] == 2
    assert result["segments"]["terminal"]["status"] == "unavailable"
    segment = result["segments"]["development"]
    assert segment["prediction_rows_sha256"] == digest(canonical(rows))
    scenarios = segment["scenarios"]
    assert scenarios[0]["account"] == baseline
    assert len(replays) == (1 if cost == "400" else 2)
    for scenario in scenarios:
        if scenario["status"] == "completed":
            account = scenario["account"]
            assert account["period"] == baseline["period"]
            assert account["strategy"] == account["buy_and_hold"]
            assert account["account_rules"]["buy_costs"]["commission_bps"] == str(
                int(cost) * scenario["multiplier"]
            )
    table = study_cost_stress.cost_stress_rows(result)
    assert len(table) == 6
    if cost == "5":
        # Commission uses the slipped notional; each component rounds to cents.
        assert [
            row["account"]["strategy"]["metrics"]["fees"]["commission"] for row in scenarios
        ] == ["4.95", "9.91", "14.87"]
        assert all(row["daily_return_count"] == 7 for row in table)
    elif cost == "0":
        assert all(row["sharpe_ratio"] is None for row in table)
        assert all(row["risk_status"] == "zero_volatility" for row in table)
    else:
        assert scenarios[-1]["status"] == "unavailable"
        assert scenarios[-1]["reason"] == "scaled_cost_above_supported_1000_bps_bound"
        assert table[-1]["net_return"] is None


def test_cost_stress_replays_exit_and_reentry_with_asymmetric_rates():
    from quant_rsi.research.study_cost_stress import evaluate_cost_stress

    data = market()
    rows = [prediction(data, index, score) for index, score in enumerate((0.1, -0.1, 0.1))]
    policy = dict(
        commission_bps="2",
        slippage_bps="7",
        initial_cash="10000",
        threshold=0.0,
        horizon_sessions=5,
    )
    selected = {
        "factor_ids": ["momentum_3"],
        "development": {"rows": rows},
        "strategy": evaluate_strategy(data, "KO", rows, **policy),
    }
    result = evaluate_cost_stress(data, "KO", selected, None, selected_round=1, policy=policy)
    for scenario in result["segments"]["development"]["scenarios"]:
        account = scenario["account"]
        multiplier = scenario["multiplier"]
        for side in ("buy_costs", "sell_costs"):
            assert account["account_rules"][side]["commission_bps"] == str(2 * multiplier)
            assert account["account_rules"][side]["slippage_bps"] == str(7 * multiplier)
        assert [fill["side"] for fill in account["strategy"]["fills"]] == ["buy", "sell", "buy"]
        assert len(account["buy_and_hold"]["fills"]) == 1
        assert Decimal(account["strategy"]["metrics"]["fees"]["commission"]) > 0
        assert Decimal(account["strategy"]["metrics"]["fees"]["slippage"]) > Decimal(
            account["strategy"]["metrics"]["fees"]["commission"]
        )
