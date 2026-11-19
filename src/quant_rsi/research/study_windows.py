"""Resolve fixed date ranges and restrict historical research context before LLM calls."""

from copy import deepcopy
from datetime import date
from typing import Literal

import numpy as np

from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.market.availability import (
    AT_INSTANT_RULE,
    DATE_ONLY_RULE,
    LOOKBACK_WINDOW_RULE,
    MAX_AGE_RULE,
    usable_at,
)
from quant_rsi.market.daily_data import CLOSE_BUFFER_SECONDS
from quant_rsi.market.market_contracts import timestamp, utc_text


class StudyTiming(StrictModel):
    """A study's declared decision instant, label window and execution (#159 D3.1).

    The at-instant rule follows from these: under a close-to-close label decided at the
    session close, information stamped exactly at the close describes the post-close move
    the label measures, so it is usable only if available strictly before the close.
    """

    decision_instant: Literal["session_close"] = "session_close"
    label: Literal["close_to_close"] = "close_to_close"
    label_window: str
    horizon_sessions: int
    execution: Literal["next_session_open"] = "next_session_open"
    at_instant_rule: Literal["strictly_before_decision_instant_v1"] = AT_INSTANT_RULE
    max_age_rule: str = MAX_AGE_RULE
    date_only_rule: str = DATE_ONLY_RULE
    lookback_window_rule: str = LOOKBACK_WINDOW_RULE
    close_readiness_seconds: int = CLOSE_BUFFER_SECONDS


def timing_rows(timing):
    """Report rows for a recorded StudyTiming; older results without one get none."""
    if not timing:
        return []
    return [
        ["Decision instant", timing["decision_instant"]],
        ["Label", f"{timing['label']}: {timing['label_window']}"],
        ["Execution", timing["execution"]],
        ["Usable at the decision instant", timing["at_instant_rule"]],
        ["Freshness (max age)", timing["max_age_rule"]],
        ["Date-only publication", timing["date_only_rule"]],
        ["News and guidance windows", timing["lookback_window_rule"]],
        ["Daily close readiness", f"close + {timing['close_readiness_seconds']} s"],
    ]


def study_timing(horizon_sessions):
    return StudyTiming(
        label_window=(
            f"log(close[origin + {horizon_sessions} sessions] / close[origin]); "
            "origin = decision session"
        ),
        horizon_sessions=horizon_sessions,
    ).model_dump(mode="json")


def configure_windows(model, args):
    dates, horizon = model.data.dates, model.horizon
    excluded = {"train": [], "development": [], "terminal": []}

    def resolve(name, default):
        requested = getattr(args, name + "_dates", None)
        if not requested:
            return default.copy()
        start, end = requested
        date.fromisoformat(start)
        date.fromisoformat(end)
        if start > end or start < dates[0] or end > dates[-1]:
            raise ValueError(f"{name} date range must be ordered and inside the retained calendar")
        rows = np.array([i for i, day in enumerate(dates) if start <= day <= end], dtype=int)
        if not len(rows):
            raise ValueError(f"{name} date range contains no retained sessions")
        return rows

    terminal = resolve("terminal", model.terminal_origins)
    if getattr(args, "terminal_dates", None) and not args.terminal_sessions:
        raise ValueError("--terminal-dates conflicts with --terminal-sessions 0")
    dev = resolve("development", model.origins)
    train = resolve("train", model.train)
    if len(terminal) and dev[-1] >= terminal[0]:
        raise ValueError("Development origins must precede terminal origins")
    if train[-1] >= dev[0]:
        raise ValueError("Training origins must precede development origins")
    # Targets must be observed strictly before the next partition starts.
    for name, rows, boundary in (
        ("terminal", terminal, len(dates)),
        ("development", dev, terminal[0] if len(terminal) else len(dates)),
        ("train", train, dev[0]),
    ):
        mask = (rows >= 30) & (rows + horizon < boundary)
        excluded[name] = [dates[int(i)] for i in rows[~mask]]
        if name == "train":
            train = rows[mask]
        elif name == "development":
            dev = rows[mask]
        else:
            terminal = rows[mask]
    if len(train) < 12 or len(dev) < 2 or (args.terminal_sessions and not len(terminal)):
        raise ValueError("Date ranges leave insufficient mature training/development/terminal rows")
    model.train, model.origins, model.terminal_origins = train, dev, terminal
    model.origin, model.evaluation_sessions = int(dev[-1]), len(dev)
    model._planned_development_end = dates[int(dev[-1]) + horizon]
    closes = (model.price_info or {}).get("session_close_times") or [
        model.data.timestamp(i, "close") for i in range(len(dates))
    ]
    mode = getattr(args, "information_mode", "retrospective")
    cutoff = getattr(args, "research_cutoff", None) or closes[int(dev[-1]) + horizon]
    cutoff = utc_text(cutoff)
    if mode == "retained":
        if timestamp(cutoff) < timestamp(closes[int(dev[-1]) + horizon]):
            raise ValueError("Research cutoff precedes matured development feedback")
        if len(terminal) and timestamp(cutoff) >= timestamp(closes[int(terminal[0])]):
            raise ValueError("Retained research cutoff must precede the terminal window")
    return {
        "information_mode": mode,
        "research_cutoff": cutoff if mode == "retained" else None,
        "decision_time": "retained exchange session close; timezone-aware UTC timestamps",
        "timing": study_timing(horizon),
        "horizon_sessions": horizon,
        "partitions": {
            name: {
                "origin_sessions": [dates[int(i)] for i in rows],
                "target_sessions": [dates[int(i) + horizon] for i in rows],
                "first_decision_time": closes[int(rows[0])] if len(rows) else None,
                "last_target_time": closes[int(rows[-1]) + horizon] if len(rows) else None,
            }
            for name, rows in (("train", train), ("development", dev), ("terminal", terminal))
        },
        "excluded_boundary_origins": excluded,
        "notice": "Fixed development feedback, not independent historical validation. "
        "Retained mode restricts supplied evidence; LLM pretrained knowledge is not filtered. "
        "Public historical price/filing reconstruction is distinct "
        "from receipt-based news archives.",
    }


def retained_news(snapshots, cutoff, symbol, limit):
    """Filter before constructing context; never substitute current search or rewritten pages."""
    instant = timestamp(cutoff)
    eligible = []
    for snapshot in snapshots:
        if not snapshot.get("received_at") or not usable_at(
            timestamp(snapshot["received_at"]), instant
        ):
            continue
        eligible.append(snapshot)
    items = {}
    for snapshot in sorted(eligible, key=lambda s: timestamp(s["received_at"])):
        for item in snapshot["items"]:
            if (
                item.get("received_at")
                and item.get("published_at")
                and usable_at(
                    max(timestamp(item["received_at"]), timestamp(item["published_at"])), instant
                )
            ):
                items[item["id"]] = deepcopy(item)
    return {
        "provider": "retained_historical_news",
        "symbol": symbol,
        "items": list(items.values())[-limit:],
        "research_cutoff": cutoff,
        "coverage_status": "available" if items else "missing_eligible_historical_material",
        "notice": (
            "Eligible retained snapshots only; no current-search fallback. "
            "Original receipt times preserved."
        ),
    }


def restrict_context(context, model, plan):
    result = deepcopy(context)
    result["information_contract"] = plan
    if plan["information_mode"] != "retained":
        return result
    result["price_evidence"] = {
        "scope": "retained historical prices; only development feedback exposed"
    }
    result["prior_outcomes"] = None
    result["factor_memory"] = {
        "entries": [],
        "notice": (
            "Cross-run memory disabled in retained mode; no timestamp-proven historical view."
        ),
    }
    result["factor_ledger"] = None
    result["guidance"] = None
    if "financial_features" in result:
        # The latest retained row can belong to terminal, so never use evidence[-1].
        result["financial_features"]["latest_visible_facts"] = deepcopy(
            model.development_financial_facts()
        )
    return result
