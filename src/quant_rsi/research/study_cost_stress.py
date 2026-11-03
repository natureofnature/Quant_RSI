"""Fixed post-selection transaction-cost scenarios using the existing account replay."""

from copy import deepcopy
from decimal import Decimal

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.market.market_contracts import decimal_text
from quant_rsi.research.demo_strategy import evaluate_strategy

MULTIPLIERS = (1, 2, 3)
NOTICE = (
    "Selected predictions replayed at the configured, double and triple commission/slippage "
    "rates. Signals, prices, initial cash and trading policy are fixed; position sizes may "
    "change with available cash. Strategy and buy-and-hold pay the same scenario costs. "
    "No additional fits, selection feedback or promotion decision. These hypothetical "
    "post-selection diagnostics are not statistical robustness or independent evidence. "
    "Zero configured costs remain zero; missing metrics remain unavailable."
)


def evaluate_cost_stress(data, symbol, selected, terminal, *, selected_round, policy):
    result = {
        "status": "completed",
        "notice": NOTICE,
        "selected_round": selected_round,
        "factor_ids": list(selected["factor_ids"]),
        "extra_fits": 0,
        "multipliers": list(MULTIPLIERS),
        "base_policy": deepcopy(policy),
        "segments": {},
    }
    for name in ("development", "terminal"):
        if name == "terminal" and (terminal or {}).get("status") != "completed":
            result["segments"][name] = {
                "status": "unavailable",
                "reason": "terminal_evaluation_disabled_or_unavailable",
            }
            continue
        evaluation = selected["development"] if name == "development" else terminal["evaluation"]
        baseline = selected["strategy"] if name == "development" else terminal["strategy"]
        predictions = evaluation["rows"]
        segment = {
            "status": "completed",
            "prediction_rows_sha256": digest(canonical(predictions)),
            "sessions": [row["origin_session"] for row in predictions],
            "scenarios": [],
        }
        for multiplier in MULTIPLIERS:
            costs = {
                key: decimal_text(str(Decimal(policy[key]) * multiplier))
                for key in ("commission_bps", "slippage_bps")
            }
            scenario = {"multiplier": multiplier, **costs}
            if any(Decimal(value) > 1000 for value in costs.values()):
                scenario.update(
                    status="unavailable", reason="scaled_cost_above_supported_1000_bps_bound"
                )
            else:
                account = (
                    deepcopy(baseline)
                    if multiplier == 1
                    else evaluate_strategy(data, symbol, predictions, **(policy | costs))
                )
                if name == "terminal":
                    account["evidence_mode"] = "historical_terminal_hypothetical_account"
                scenario.update(status="completed", account=account)
            segment["scenarios"].append(scenario)
        result["segments"][name] = segment
    return result


def cost_stress_rows(result):
    """Shared report/UI rows; keep unavailable scenarios and risk metrics visible."""
    rows = []
    for window, segment in result["segments"].items():
        if segment["status"] != "completed":
            continue
        for scenario in segment["scenarios"]:
            for name in ("strategy", "buy_and_hold"):
                metrics = (
                    scenario["account"][name]["metrics"]
                    if scenario["status"] == "completed"
                    else {}
                )
                rows.append(
                    {
                        "window": window,
                        "multiplier": scenario["multiplier"],
                        "commission_bps": scenario["commission_bps"],
                        "slippage_bps": scenario["slippage_bps"],
                        "account": name,
                        "status": scenario["status"],
                        "reason": scenario.get("reason"),
                        "risk_status": metrics.get("risk_statistics", {}).get("status"),
                        "daily_return_count": metrics.get("risk_statistics", {}).get(
                            "daily_return_count"
                        ),
                        **{
                            key: metrics.get(key)
                            for key in ("net_return", "sharpe_ratio", "max_drawdown")
                        },
                    }
                )
    return rows
