"""Hypothetical close-signal/next-open replay through the existing account helpers."""

import math
from decimal import ROUND_HALF_EVEN, Decimal
from statistics import mean, stdev

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.data import MarketData
from quant_rsi.execution import execution_account as account
from quant_rsi.market.market_contracts import AccountRules, decimal_text


def evaluate_strategy(
    data: MarketData,
    symbol: str,
    predictions: list[dict],
    *,
    commission_bps="5",
    slippage_bps="5",
    initial_cash="10000",
    threshold=0.0,
    horizon_sessions=5,
):
    """Review each supplied close signal; unchanged long signals preserve held units."""
    if symbol not in data.instruments or not predictions or not math.isfinite(threshold):
        raise ValueError("A supported symbol, predictions and finite threshold are required")
    column = data.instruments.index(symbol)
    if type(horizon_sessions) is not int or horizon_sessions not in (3, 5, 20, 60):
        raise ValueError("Prediction horizon must be 3, 5, 20 or 60 trading sessions")
    signals = _signals(data, column, predictions, horizon_sessions)
    return _evaluate_signals(
        data,
        symbol,
        signals,
        max(signals) + horizon_sessions,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        initial_cash=initial_cash,
        threshold=threshold,
    ) | {
        "prediction_horizon_sessions": horizon_sessions,
        "execution_note": (
            f"{horizon_sessions}-session labels differ from next-open trade intervals. "
            "Account quotes are rounded half-even to 12 decimal places; "
            "original model prices are unchanged."
        ),
    }


def evaluate_monthly_strategy(
    data: MarketData,
    symbol: str,
    predictions: list[dict],
    *,
    commission_bps="5",
    slippage_bps="5",
    initial_cash="10000",
    threshold=0.0,
):
    """Replay observed contiguous monthly forecasts in one costed account.

    Endpoints must be the last retained session of their respective months;
    the caller must verify those endpoints against the published exchange calendar.
    Unknown targets are rejected, never silently shortened to available prices.
    """
    if symbol not in data.instruments or not predictions or not math.isfinite(threshold):
        raise ValueError("A supported symbol, predictions and finite threshold are required")
    column = data.instruments.index(symbol)
    positions = {session: index for index, session in enumerate(data.dates)}
    month_ends = {session[:7]: index for index, session in enumerate(data.dates)}
    signals, horizons, previous_target = {}, [], None
    for row in predictions:
        try:
            origin, target = positions[row["origin_session"]], positions[row["target_session"]]
        except (KeyError, TypeError):
            raise ValueError(
                "Monthly forecasts require observed origin and target sessions"
            ) from None
        origin_month = data.dates[origin][:7]
        year, month = map(int, origin_month.split("-"))
        next_month = f"{year + (month == 12):04d}-{month % 12 + 1:02d}"
        if (
            origin != month_ends[origin_month]
            or data.dates[target][:7] != next_month
            or target != month_ends[next_month]
            or (previous_target is not None and origin != previous_target)
        ):
            raise ValueError("Monthly forecasts must form contiguous month-end intervals")
        horizon = target - origin
        signals.update(_signals(data, column, [row], horizon))
        horizons.append(horizon)
        previous_target = target
    return _evaluate_signals(
        data,
        symbol,
        signals,
        previous_target,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        initial_cash=initial_cash,
        threshold=threshold,
    ) | {
        "evidence_mode": "historical_monthly_walk_forward_hypothetical_account",
        "prediction_horizon_sessions": horizons,
        "prediction_horizon": "next_calendar_month_end",
        "execution_note": (
            "One continuous account across observed monthly forecasts; orders execute at "
            "the next retained open and daily NAV determines risk statistics. Month-end "
            "price returns differ from executable next-open returns. Quotes are rounded "
            "half-even to 12 decimal places; no per-fold reset or averaging of Sharpes."
        ),
    }


def evaluate_session_strategy(
    data: MarketData,
    symbol: str,
    predictions: list[dict],
    *,
    horizon_sessions=3,
    commission_bps="5",
    slippage_bps="5",
    initial_cash="10000",
    threshold=0.0,
):
    """Replay contiguous, non-overlapping fixed-session forecasts in one account.

    The planner verifies the exchange calendar. This boundary also rejects gaps,
    overlaps, unknown targets and intervals shorter than the requested horizon.
    """
    if symbol not in data.instruments or not predictions or not math.isfinite(threshold):
        raise ValueError("A supported symbol, predictions and finite threshold are required")
    if type(horizon_sessions) is not int or not 1 <= horizon_sessions <= 20:
        raise ValueError("Prediction horizon must be 1..20 trading sessions")
    column = data.instruments.index(symbol)
    positions = {session: index for index, session in enumerate(data.dates)}
    previous_target = None
    for row in predictions:
        try:
            origin, target = positions[row["origin_session"]], positions[row["target_session"]]
        except (KeyError, TypeError):
            raise ValueError(
                "Session forecasts require observed origin and target sessions"
            ) from None
        if target - origin != horizon_sessions or (
            previous_target is not None and origin != previous_target
        ):
            raise ValueError("Session forecasts must form contiguous fixed-horizon intervals")
        previous_target = target
    signals = _signals(data, column, predictions, horizon_sessions)
    return _evaluate_signals(
        data,
        symbol,
        signals,
        previous_target,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        initial_cash=initial_cash,
        threshold=threshold,
    ) | {
        "evidence_mode": "historical_session_walk_forward_hypothetical_account",
        "prediction_horizon_sessions": horizon_sessions,
        "prediction_horizon": "fixed_trading_sessions",
        "execution_note": (
            "One continuous account across observed, non-overlapping session forecasts; "
            "orders execute at the next retained open and daily NAV determines risk statistics. "
            "Close-to-close forecast returns differ from executable next-open returns. "
            "Quotes are rounded half-even to 12 decimal places; no per-period reset or "
            "averaging of Sharpes. Pending intervals are excluded."
        ),
    }


def _evaluate_signals(
    data, symbol, signals, last, *, commission_bps, slippage_bps, initial_cash, threshold
):
    column = data.instruments.index(symbol)
    first = min(signals)
    costs = dict(commission_bps=commission_bps, tax_bps="0", slippage_bps=slippage_bps)
    rules = AccountRules(
        currency="USD",
        currency_precision=2,
        rounding_policy="half_even_per_component",
        initial_cash=initial_cash,
        cash_settlement="immediate_simulation",
        inventory_unlock="immediate_simulation",
        sizing="long_only_topk",
        allocation="1",
        max_positions=1,
        lot_size=1,
        cost_policy="linear_side_bps",
        buy_costs=costs,
        sell_costs=costs,
        cost_claim="explicit_observation_assumption",
        fill_policy="next_registered_open_market",
        fill_cap_units={symbol: 1_000_000_000_000},
        funding_policy="pre_funded_no_unfilled_sale_credit",
        valuation="close_fresh_quote_only",
        benchmark="cash_zero_interest",
        action_policy="notice_suspend_unresolved",
        termination="preserve_inventory",
        stop_policy="cancel_unelapsed_targets",
        max_drawdown="1",
    )
    results = {}
    for name in ("strategy", "buy_and_hold"):
        state = account.new_account(rules)
        for index in range(first, last + 1):
            if index > first:
                state = account.resolve_open(
                    state,
                    _quote(data, column, index, "open"),
                    rules,
                    session_id=data.dates[index],
                    resolution_at=data.timestamp(index, "open"),
                )
            quotes = _quote(data, column, index, "close")
            state = account.value_close(
                state,
                quotes,
                rules,
                session_id=data.dates[index],
                close_at=data.timestamp(index, "close"),
            )
            if (name == "buy_and_hold" and index == first) or (
                name == "strategy" and index in signals
            ):
                long = name == "buy_and_hold" or signals[index] > threshold
                state = _submit(state, data, index, symbol, long, quotes[symbol]["price"], rules)
        results[name] = {"metrics": _metrics(state, rules), **state}
    return {
        "evidence_mode": "historical_development_hypothetical_account",
        "clock_basis": "MarketData.timestamp simulation, not live or durable submission evidence",
        "policy": "log_return_above_threshold_long_otherwise_cash; hold units until signal changes",
        "threshold": threshold,
        "period": {"start": data.dates[first], "end": data.dates[last]},
        "account_rules": rules.model_dump(mode="json"),
        **results,
    }


def _signals(data, column, predictions, horizon_sessions=5):
    positions = {session: index for index, session in enumerate(data.dates)}
    result = {}
    for row in predictions:
        try:
            origin, target = positions[row["origin_session"]], positions[row["target_session"]]
            reference, predicted, observed, score = (
                float(row[key])
                for key in (
                    "reference_price",
                    "predicted_price",
                    "observed_price",
                    "predicted_log_return",
                )
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError("Invalid prediction fields or session alignment") from None
        if (
            target != origin + horizon_sessions
            or (result and origin <= max(result))
            or not all(math.isfinite(value) for value in (reference, predicted, observed, score))
            or min(reference, predicted, observed) <= 0
            or not math.isclose(reference, float(data.closes[origin, column]), rel_tol=1e-9)
            or not math.isclose(observed, float(data.closes[target, column]), rel_tol=1e-9)
            or not math.isclose(math.log(predicted / reference), score, abs_tol=1e-9)
        ):
            raise ValueError(
                "Predictions must be chronological and match "
                f"{horizon_sessions}-session price inputs"
            )
        result[origin] = score
    return result


def _quote(data, column, index, phase):
    values = data.opens if phase == "open" else data.closes
    rounded = Decimal(str(float(values[index, column]))).quantize(
        Decimal("0.000000000001"), rounding=ROUND_HALF_EVEN
    )
    quote = {"price": decimal_text(format(rounded, "f"), positive=True)}
    quote["payload_ref"] = digest(canonical([data.dates[index], phase, quote["price"]]))
    return {data.instruments[column]: quote}


def _submit(state, data, index, symbol, long, price, rules):
    held = state["holdings"].get(symbol, {}).get("quantity", 0)
    if state["suspended_reasons"] or (long and held) or (not long and not held):
        return state
    quantity = int(Decimal(state["cash"]) / Decimal(price)) if long else held
    decision_at = data.timestamp(index, "close")
    instruction = dict(
        instrument=symbol, side="buy" if long else "sell", quantity=quantity, reference_price=price
    )
    state, identifiers = account.prepare_orders(
        state,
        [instruction],
        rules,
        decision_ref=digest(canonical([data.dates[index], long, instruction])),
        target_session=data.dates[index + 1],
        target_at=data.timestamp(index + 1, "open"),
        prediction_finished_at=decision_at,
    )
    # This witness is hypothetical, never a claim of persisted or prospective submission.
    return account.confirm_submission(state, identifiers, decision_at)


def _metrics(state, rules):
    initial = Decimal(rules.initial_cash)
    return {
        **_risk_metrics(state["valuations"]),
        "net_return": account.amount(Decimal(state["last_nav"]) / initial - 1),
        "max_drawdown": min((row["drawdown"] for row in state["valuations"]), key=Decimal),
        "fees": {key: state["cumulative_" + key] for key in ("commission", "tax", "slippage")},
        "turnover": account.amount(
            sum((Decimal(fill["raw_notional"]) for fill in state["fills"]), Decimal(0)) / initial
        ),
        "turnover_definition": "sum_absolute_raw_fill_notional_divided_by_initial_cash",
        "ending_inventory": state["holdings"],
        "ending_cash": state["cash"],
    }


def _risk_metrics(valuations):
    """Use consecutive costed closes, excluding the initial pre-trade cash mark."""
    result = {
        "sharpe_ratio": None,
        "annualized_volatility": None,
        "risk_statistics": {
            "return_basis": "consecutive_daily_net_account_nav_simple_returns",
            "annualization_sessions": 252,
            "annual_risk_free_rate": 0.0,
            "standard_deviation_ddof": 1,
            "daily_return_count": 0,
            "status": "insufficient_observations",
            "interpretation": "Short retrospective development sample; not independent evidence.",
        },
    }
    info = result["risk_statistics"]
    nav = [float(row["nav"]) if row["nav"] is not None else None for row in valuations]
    if any(value is None or not math.isfinite(value) or value <= 0 for value in nav):
        info["status"] = "missing_or_nonpositive_nav"
        return result
    returns = [current / previous - 1 for previous, current in zip(nav, nav[1:], strict=False)]
    info["daily_return_count"] = len(returns)
    if len(returns) < 2:
        return result
    deviation = stdev(returns)
    result["annualized_volatility"] = deviation * math.sqrt(252)
    info["status"] = "zero_volatility" if deviation == 0 else "available"
    if deviation:
        result["sharpe_ratio"] = mean(returns) / deviation * math.sqrt(252)
    return result
