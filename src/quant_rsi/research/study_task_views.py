"""Read-only task settings and conservative comparisons of retained study results."""

import math

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research.study_queue import study_options
from quant_rsi.research.study_ui import _read_json

COMPARISON_NOTICE = (
    "Descriptive comparison of selected retrospective results. Comparable samples and policies "
    "do not make selection histories, research budgets or information sets independent or equal. "
    "No winner or predictive-advantage claim is inferred. Differences are right minus left."
)


def settings_from_task(task):
    manifest = _read_json(task / "task.json", {})
    if manifest.get("action", "research") != "research":
        raise ValueError("Only research tasks have reusable study settings")
    configuration = _read_json(task / "study/config.json", {})
    defaults = study_options({})
    values = {key: manifest[key] for key in defaults if key in manifest}
    for key in defaults:
        source = "prices" if key == "data" else key
        if source in configuration:
            value = configuration[source]
            if key == "llm_model" and value is None:
                value = ""
            if value is not None or key in {
                "max_fits",
                "patience",
                "historical_sessions",
                "train_dates",
                "development_dates",
                "terminal_dates",
            }:
                values[key] = value
    return {
        "source_task_id": task.name,
        "symbols": [manifest["symbol"]],
        "options": study_options(values),
        "notice": (
            "Settings copied for editing only. Queue research to create a new task with fresh "
            "inputs and the current runtime; this does not replay or resume the old study. "
            "The original artifacts are unchanged. Unchanged-evidence checks still apply."
        ),
    }


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _metric_number(value):
    # Account contracts serialize Decimal values as strings; display them as numbers.
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    return value if _finite(value) else None


def _segment(result, inputs, prediction, name):
    if name == "development":
        evaluation, strategy = prediction.get("development"), prediction.get("strategy")
    else:
        terminal = result.get("terminal_evaluation", {})
        if terminal.get("status") != "completed":
            return {"status": "unavailable", "reason": "Terminal evaluation is not available."}
        evaluation, strategy = terminal.get("evaluation"), terminal.get("strategy")
    if not evaluation or not strategy:
        return {
            "status": "unavailable",
            "reason": "Retained evaluation/account evidence is missing.",
        }
    rows = evaluation.get("rows", [])
    period = strategy.get("period", {})
    rules = strategy.get("account_rules")
    observations = []
    try:
        account = strategy["strategy"]["metrics"]
        buy_and_hold = strategy["buy_and_hold"]["metrics"]
        metrics = evaluation["metrics"]
        if (
            not evaluation.get("metric_units")
            or any(
                key not in strategy
                for key in ("policy", "threshold", "prediction_horizon_sessions")
            )
            or (
                result.get("configuration", {}).get("prices") == "daily"
                and not evaluation.get("currency")
            )
        ):
            raise ValueError("Missing metric units or execution policy")
        for row in rows:
            observation = {
                key: row[key]
                for key in ("origin_session", "target_session", "reference_price", "observed_price")
            }
            if any(
                not _finite(observation[key]) or observation[key] <= 0
                for key in ("reference_price", "observed_price")
            ):
                raise ValueError("Invalid observation")
            observations.append(observation)
        dates, opens, closes = (inputs[key] for key in ("dates", "opens", "closes"))
        if len(dates) != len(opens) or len(dates) != len(closes):
            raise ValueError("Incomplete prices")
        quotes = [
            (day, opening, close)
            for day, opening, close in zip(dates, opens, closes, strict=True)
            if period["start"] <= day <= period["end"]
        ]
        if (
            not observations
            or not rules
            or not quotes
            or inputs["target"] != result.get("target", result.get("symbol"))
            or quotes[0][0] != period["start"]
            or quotes[-1][0] != period["end"]
            or any(not _finite(price) or price <= 0 for _, *prices in quotes for price in prices)
        ):
            raise ValueError("Incomplete evidence")
    except (KeyError, TypeError, ValueError):
        return {
            "status": "unavailable",
            "reason": "Observation, price, unit or policy evidence is incomplete.",
        }
    baseline = evaluation.get("baselines", {}).get("unchanged_price", {}).get("metrics", {})
    return {
        "status": "completed",
        "period": period,
        "currency": evaluation.get("currency"),
        "metrics": {
            key: _metric_number(value)
            for key, value in {
                "mae_price": metrics.get("mae_price"),
                "unchanged_price_mae": baseline.get("mae_price"),
                "direction_hit_rate": metrics.get("direction_hit_rate"),
                "sample_count": metrics.get("sample_count"),
                "net_return": account.get("net_return"),
                "sharpe_ratio": account.get("sharpe_ratio"),
                "daily_return_count": account.get("risk_statistics", {}).get("daily_return_count"),
                "max_drawdown": account.get("max_drawdown"),
                "buy_and_hold_return": buy_and_hold.get("net_return"),
            }.items()
        },
        "account_assumptions": {
            "initial_cash": rules.get("initial_cash"),
            "buy_costs": rules.get("buy_costs"),
            "sell_costs": rules.get("sell_costs"),
            "threshold": strategy.get("threshold"),
        },
        "signatures": {
            "Evaluation observations": digest(canonical(observations)),
            "Actual account price path": digest(canonical(quotes)),
            "Execution policy and costs": digest(
                canonical(
                    {
                        "rules": rules,
                        "threshold": strategy.get("threshold"),
                        "policy": strategy.get("policy"),
                        "horizon": strategy.get("prediction_horizon_sessions"),
                    }
                )
            ),
            "Metric units": digest(
                canonical(
                    {
                        "currency": evaluation.get("currency"),
                        "units": evaluation.get("metric_units"),
                    }
                )
            ),
        },
    }


def retained_task_view(task):
    result = _read_json(task / "study/result.json", {})
    inputs = _read_json(task / "study/inputs.json", {})
    manifest = _read_json(task / "task.json", {})
    configuration = result.get("configuration", {})
    iterations = result.get("iterations", {})
    selected = iterations.get("selected_round")
    attempts = iterations.get("attempts", [])
    discovery = result.get("discovery")
    if type(selected) is not int or not 1 <= selected <= len(attempts):
        if discovery is None or selected is not None:
            raise ValueError("Completed study has no retained selected result")
        prediction = {}
    else:
        prediction = attempts[selected - 1]["prediction"]
    selection_note = (discovery or {}).get("selection_note")
    experiment_groups = []
    for key, group in (discovery or {}).get("groups", {}).items():
        incumbent = group.get("incumbent", {}).get("result", {})
        experiment_groups.append(
            {
                "comparison_key": key,
                "incumbent_round": incumbent.get("round"),
                "candidate_count": len(group.get("candidates", [])),
                "development": _segment(
                    result, inputs, incumbent.get("prediction", {}), "development"
                ),
            }
        )
    return {
        "id": task.name,
        "symbol": result.get("target", result.get("symbol")),
        "created_at": manifest.get("created_at"),
        "data": configuration.get("prices"),
        "horizon_sessions": prediction.get(
            "horizon_sessions", configuration.get("horizon_sessions")
        ),
        "information_mode": configuration.get("information_mode", "retrospective"),
        "question": configuration.get("question", ""),
        "model": configuration.get("model"),
        "researcher": configuration.get("llm"),
        "research_mode": configuration.get("research_mode", "legacy"),
        "selected_round": selected,
        "selection_note": selection_note,
        "experiment_groups": experiment_groups,
        "completed_rounds": len(attempts),
        "fit_budget": configuration.get("fit_budget"),
        "factor_ids": prediction.get("factor_ids", []),
        "training_rows": prediction.get("training_rows"),
        "input_coverage": {
            "price_provider": inputs.get("price_provider"),
            "retained_price_sessions": len(inputs.get("dates", [])),
            "news_snapshots": inputs.get("news_snapshot_count"),
            "financial_provider": result.get("study", {}).get("financial_data", {}).get("provider"),
        },
        "segments": {
            name: (
                _segment(result, inputs, prediction, name)
                if prediction
                else {
                    "status": "unavailable",
                    "reason": "No single discovery experiment selected. "
                    + (selection_note or "Inspect separately grouped experiment conclusions."),
                }
            )
            for name in ("development", "terminal")
        },
    }


def compare_retained_tasks(left, right):
    views = [retained_task_view(task) for task in (left, right)]
    results = {}
    common = [
        key.replace("_", " ") + " differs"
        for key in ("symbol", "data", "horizon_sessions", "information_mode")
        if views[0][key] != views[1][key]
    ]
    common += [
        key.replace("_", " ") + " is unavailable"
        for key in ("symbol", "data", "horizon_sessions")
        if any(view[key] is None for view in views)
    ]
    for name in ("development", "terminal"):
        a, b = (v["segments"][name] for v in views)
        reasons = list(common)
        if a["status"] != "completed" or b["status"] != "completed":
            reasons += [
                f"{side}: {value['reason']}"
                for side, value in (("Left", a), ("Right", b))
                if value["status"] != "completed"
            ]
        else:
            reasons += [
                label + " differs"
                for label in a["signatures"]
                if a["signatures"][label] != b["signatures"][label]
            ]
        comparable = not reasons
        deltas = {
            key: b["metrics"][key] - value
            for key, value in a.get("metrics", {}).items()
            if comparable
            and key not in {"sample_count", "daily_return_count"}
            and _finite(value)
            and _finite(b["metrics"].get(key))
        }
        results[name] = {"comparable": comparable, "reasons": reasons, "differences": deltas}
    return {"studies": views, "comparison": results, "notice": COMPARISON_NOTICE}
