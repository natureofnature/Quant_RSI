"""Small visual projections of retained study outputs; never used by research selection."""

import math


def _pick(value, *keys):
    return {key: value.get(key) for key in keys}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _news(items):
    return [
        {
            **_pick(
                item,
                "id",
                "title",
                "url",
                "published_at",
                "published_date",
                "received_at",
                "status",
                "content_extent",
                "symbol",
            ),
            "excerpt": str(item.get("text") or item.get("summary") or "")[:600],
            "excerpt_shortened": len(str(item.get("text") or item.get("summary") or "")) > 600,
        }
        for item in items[:24]
    ]


def _evidence(ids, previous):
    available = {}
    for stage, value in previous.items():
        items = (
            value
            if stage.startswith("article_reads_")
            else value.get("items", [])
            if stage == "news"
            else []
        )
        for item in items:
            if item.get("id"):
                available[item["id"]] = item
    return _news(
        [
            available.get(key, {"id": key, "title": "Evidence not present in this trace"})
            for key in ids
        ]
    )


def _factors(items):
    return [
        _pick(
            item,
            "factor_id",
            "kind",
            "description",
            "input_kind",
            "recipe",
            "runnable",
            "availability_reason",
        )
        for item in items[:64]
    ]


def factor_changes(before, after):
    """IDs define identity. Pair unique same-operator replacements only for recipe comparison."""
    old = {item["factor_id"]: item for item in _factors(before)}
    new = {item["factor_id"]: item for item in _factors(after)}
    changes = []
    for key in sorted(old.keys() & new.keys()):
        changes.append(
            {
                "type": "retained" if old[key]["recipe"] == new[key]["recipe"] else "changed",
                "before": old.pop(key),
                "after": new.pop(key),
            }
        )

    def operator(item):
        return item["kind"], item["input_kind"], (item["recipe"] or {}).get("op")

    for key, item in list(old.items()):
        same_old = [v for v in old.values() if operator(v) == operator(item)]
        same_new = [v for v in new.values() if operator(v) == operator(item)]
        if len(same_old) == len(same_new) == 1 and operator(item)[-1]:
            replacement = same_new[0]
            changes.append(
                {
                    "type": "replaced",
                    "before": old.pop(key),
                    "after": new.pop(replacement["factor_id"]),
                }
            )
    changes.extend({"type": "removed", "before": item, "after": None} for item in old.values())
    changes.extend({"type": "added", "before": None, "after": item} for item in new.values())
    return changes


def prediction_rows(current, initial=None):
    old = {
        (row["origin_session"], row["target_session"]): row
        for row in (initial or {}).get("development", {}).get("rows", [])
    }
    rows = []
    for row in current.get("development", {}).get("rows", []):
        prior = old.get((row["origin_session"], row["target_session"]), {})
        # Never join by row position or compare attempts with different observed outcomes.
        comparable = prior.get("observed_price") == row.get("observed_price")
        rows.append(
            {
                "origin": row["origin_session"],
                "date": row["target_session"],
                "observed": row.get("observed_price"),
                "prediction": row.get("predicted_price"),
                "initial": prior.get("predicted_price") if comparable else None,
                "baseline": row.get("reference_price"),
            }
        )
    return rows


def _accounts(previous):
    first = previous.get("first_prediction", {}).get("strategy", {})
    second = previous.get("second_prediction", {}).get("strategy", {})
    accounts = [
        ("Initial", first.get("strategy", {})),
        ("Revised", second.get("strategy", {})),
        ("Buy & hold", first.get("buy_and_hold", {})),
    ]
    metrics = [
        {
            "label": label,
            **_pick(
                account.get("metrics", {}),
                "net_return",
                "sharpe_ratio",
                "max_drawdown",
                "risk_statistics",
            ),
        }
        for label, account in accounts
    ]
    by_date = {}
    for label, account in accounts:
        for valuation in account.get("valuations", []):
            row = by_date.setdefault(valuation["session_id"], {"date": valuation["session_id"]})
            row[label] = valuation.get("nav") if valuation.get("status") == "complete" else None
    return metrics, [by_date[key] for key in sorted(by_date)]


def visual_output(stage, output, previous, symbol):
    if stage == "load_prices":
        return {"kind": "facts", "facts": _pick(output, "symbol", "provider", "sessions")}
    if stage in {"collect_news_context", "load_related_prices"}:
        return {"kind": "relationships", "target": symbol, **output["related"]}
    if stage == "target":
        return {
            "kind": "facts",
            "facts": _pick(
                output,
                "symbol",
                "reference_price",
                "origin_session",
                "evaluation_sessions",
                "model_mode",
            ),
            "note": output.get("model_note"),
        }
    if stage == "related_stocks":
        return {"kind": "relationships", "target": symbol, **output}
    if stage == "news" or stage.startswith(("news_search_", "article_reads_")):
        items = (
            output if isinstance(output, list) else output.get("items", output.get("results", []))
        )
        return {
            "kind": "news",
            "items": _news(items),
            "count": len(items),
            "query": output.get("query") if isinstance(output, dict) else None,
            "note": output.get("rationale") if isinstance(output, dict) else None,
        }
    if stage.startswith("news_review_"):
        return {
            "kind": "findings",
            "findings": [
                {
                    "text": item["text"],
                    "evidence": _evidence(item.get("evidence_ids", []), previous),
                }
                for item in output.get("findings", [])
            ],
            "related": output.get("related_companies", []),
            "followup": output.get("followup_query"),
        }
    if stage in {"initial_hypothesis", "revised_hypothesis"}:
        return {
            "kind": "hypothesis",
            "text": output.get("hypothesis"),
            "factors": output.get("factor_ids", []),
            "evidence": _evidence(output.get("evidence_news_ids", []), previous),
            "feedback_used": output.get("feedback_used"),
        }
    if stage in {"factor_library", "initial_factor_library", "revised_factor_library"}:
        return {
            "kind": "factors",
            "factors": _factors(output.get("definitions", [])),
            "count": len(output.get("definitions", [])),
            "note": output.get("execution_note"),
            "catalog": True,
        }
    if stage in {"initial_factors", "revised_factors"}:
        revised = stage == "revised_factors"
        hypothesis = previous.get("revised_hypothesis" if revised else "initial_hypothesis", {})
        return {
            "kind": "factor_changes" if revised else "factors",
            "factors": _factors(output),
            "changes": factor_changes(previous.get("initial_factors", []), output)
            if revised
            else [],
            "note": hypothesis.get("hypothesis"),
            "count": len(output),
        }
    if stage in {"first_prediction", "second_prediction"}:
        initial = previous.get("first_prediction") if stage == "second_prediction" else None
        development = output.get("development", {})
        return {
            "kind": "prediction",
            "attempt": "revised" if stage == "second_prediction" else "initial",
            "rows": prediction_rows(output, initial),
            "metrics": development.get("metrics", {}),
            "initial_metrics": (initial or {}).get("development", {}).get("metrics"),
            "baseline_metrics": development.get("baselines", {})
            .get("unchanged_price", {})
            .get("metrics", {}),
            "currency": development.get("currency"),
            "horizon": output.get("horizon_sessions"),
            "note": development.get("evaluation_note"),
            "input_usage": output.get("input_usage", []),
        }
    if stage == "feedback":
        rows = prediction_rows(previous.get("first_prediction", {}))
        errors = [
            {**row, "error": row["prediction"] - row["observed"]}
            for row in rows
            if _number(row["prediction"]) and _number(row["observed"])
        ]
        return {
            "kind": "feedback",
            "metrics": output.get("development", {}).get("metrics", {}),
            "errors": sorted(errors, key=lambda row: abs(row["error"]), reverse=True)[:5],
            "latest": _pick(
                output, "predicted_price", "observed_price", "error", "observed_session"
            ),
            "note": output.get("development", {}).get("evaluation_note"),
        }
    if stage == "summary":
        accounts, nav = _accounts(previous)
        return {
            "kind": "summary",
            "text": output.get("text"),
            "selection": output.get("selection"),
            "accounts": accounts,
            "nav": nav,
        }
    return {"kind": "unknown"}
