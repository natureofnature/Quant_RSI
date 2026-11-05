"""Calendar-month folds with cutoff-only, fixed-width training histories."""

import json
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path

from quant_rsi.core.artifacts import digest
from quant_rsi.market.daily_data import _session


def month_start(value):
    parsed = date.fromisoformat(value + "-01")
    if parsed.strftime("%Y-%m") != value:
        raise ValueError("Months must use YYYY-MM")
    return parsed


def shift_month(day, amount):
    year, month = divmod(day.year * 12 + day.month - 1 + amount, 12)
    return date(year, month + 1, 1)


def read_calendar(directory):
    root = Path(directory)
    receipt = json.loads((root / "collection_receipt.json").read_bytes())
    raw = (root / "calendar_evidence.json").read_bytes()
    if digest(raw) != receipt["result"]["calendar_evidence_ref"]:
        raise ValueError("Published calendar differs from collection receipt")
    value = json.loads(raw)
    return {
        "holidays": {
            int(y): {date.fromisoformat(d) for d in days} for y, days in value["holidays"].items()
        },
        "early": {date.fromisoformat(d) for d in value["early_closes_1300"]},
        "sha256": digest(raw),
    }


def sessions(calendar, start, end):
    result = []
    day = start
    while day <= end:
        session = _session(day, calendar["holidays"], calendar["early"])
        if session:
            result.append(session)
        day += timedelta(days=1)
    return result


def monthly_plan(data, calendar, first_month, last_month, train_months=6):
    first, last = month_start(first_month), month_start(last_month)
    if first > last or not 1 <= train_months <= 24:
        raise ValueError("Ordered months and a training window of 1..24 months are required")
    if (last.year - first.year) * 12 + last.month - first.month >= 24:
        raise ValueError("At most 24 monthly forecasts per task")
    result = []
    month = first
    while month <= last:
        start, end = shift_month(month, -train_months), month - timedelta(days=1)
        history = sessions(calendar, start, end)
        future = sessions(calendar, month, shift_month(month, 1) - timedelta(days=1))
        expected = [s["session_id"] for s in history]
        actual = [day for day in data.dates if start.isoformat() <= day <= end.isoformat()]
        if not expected or actual != expected or not future:
            raise ValueError(f"Incomplete six-month history/calendar for {month:%Y-%m}")
        h = len(future)
        # 30 rows are warm-up, and every admitted H-session label has matured at origin.
        training = list(range(30, len(history) - h))
        if len(training) < 30:
            raise ValueError(f"Insufficient mature daily training samples for {month:%Y-%m}")
        result.append(
            {
                "month": month.strftime("%Y-%m"),
                "training_window": {"start": start.isoformat(), "end": end.isoformat()},
                "history_sessions": expected,
                "origin_session": expected[-1],
                "target_session": future[-1]["session_id"],
                "information_cutoff": history[-1]["close_at"],
                "target_close_at": future[-1]["close_at"],
                "horizon_sessions": h,
                "training_indices": training,
                "training_origins": [expected[i] for i in training],
                "training_targets": [expected[i + h] for i in training],
                "warmup_sessions": 30,
                "unmatured_origins": expected[-h:],
                "calendar_sha256": calendar["sha256"],
            }
        )
        month = shift_month(month, 1)
    return result


def period_id(fold):
    """Use a session target ID without changing retained monthly identifiers."""
    return fold["period_id"] if "period_id" in fold else fold["month"]


def session_plan(
    data,
    calendar,
    first_month,
    last_month,
    train_months=6,
    horizon_sessions=3,
    *,
    train_sessions=None,
):
    """Plan contiguous H-session targets within the requested calendar-month range.

    Training is the half-open calendar interval [origin + 1 day - M months,
    origin + 1 day), clipping the start day when its month is shorter. Only full
    H-session intervals are forecast; a trailing shorter interval is omitted.
    The first unavailable target is retained as pending, then planning stops.
    An explicit train_sessions instead selects exactly N published sessions
    through the origin, using retained dates to bound calendar verification.
    """
    first, last = month_start(first_month), month_start(last_month)
    if (
        first > last
        or (
            train_sessions is None
            and (type(train_months) is not int or not 1 <= train_months <= 24)
        )
        or type(horizon_sessions) is not int
        or not 1 <= horizon_sessions <= 20
    ):
        raise ValueError(
            "Ordered months, 1..24 training months and 1..20 horizon sessions required"
        )
    if train_sessions is not None and (
        type(train_sessions) is not int
        or not 61 <= train_sessions <= 445
        or train_sessions < 60 + horizon_sessions
    ):
        raise ValueError("Training sessions must be 61..445 and at least 60 + horizon sessions")
    if (last.year - first.year) * 12 + last.month - first.month >= 24:
        raise ValueError("At most 24 calendar months per session task")
    future = sessions(calendar, first, shift_month(last, 1) - timedelta(days=1))
    if len(future) // horizon_sessions > 240:
        raise ValueError("At most 240 session forecasts per task")
    before = sessions(calendar, shift_month(first, -1), first - timedelta(days=1))
    if not before or len(future) < horizon_sessions:
        raise ValueError("No complete forecast interval in the requested calendar range")
    origin = before[-1]
    result = []
    for offset in range(0, len(future) - horizon_sessions + 1, horizon_sessions):
        block = future[offset : offset + horizon_sessions]
        target = block[-1]
        end = date.fromisoformat(origin["session_id"])
        if train_sessions is None:
            exclusive_end = end + timedelta(days=1)
            shifted = shift_month(exclusive_end, -train_months)
            start = shifted.replace(
                day=min(exclusive_end.day, monthrange(shifted.year, shifted.month)[1])
            )
        else:
            retained_history = [day for day in data.dates if day <= origin["session_id"]]
            if len(retained_history) < train_sessions:
                raise ValueError(
                    f"Insufficient retained training sessions for {target['session_id']}"
                )
            start = date.fromisoformat(retained_history[-train_sessions])
        history = sessions(calendar, start, end)
        expected = [row["session_id"] for row in history]
        actual = [day for day in data.dates if start.isoformat() <= day <= end.isoformat()]
        if (
            not expected
            or actual != expected
            or (train_sessions is not None and len(expected) != train_sessions)
        ):
            raise ValueError(f"Incomplete training history/calendar for {target['session_id']}")
        observed = [row["session_id"] for row in block if row["session_id"] <= data.dates[-1]]
        retained = [day for day in data.dates if origin["session_id"] < day <= target["session_id"]]
        if retained != observed:
            raise ValueError(f"Incomplete observed forecast interval for {target['session_id']}")
        training = list(range(30, len(history) - horizon_sessions))
        if len(training) < 30:
            raise ValueError(
                f"Insufficient mature daily training samples for {target['session_id']}"
            )
        result.append(
            {
                "period_id": target["session_id"],
                "period_kind": "trading_sessions",
                "training_window": {"start": start.isoformat(), "end": end.isoformat()},
                "history_sessions": expected,
                "origin_session": origin["session_id"],
                "target_session": target["session_id"],
                "information_cutoff": history[-1]["close_at"],
                "target_close_at": target["close_at"],
                "horizon_sessions": horizon_sessions,
                "training_indices": training,
                "training_origins": [expected[i] for i in training],
                "training_targets": [expected[i + horizon_sessions] for i in training],
                "warmup_sessions": 30,
                "unmatured_origins": expected[-horizon_sessions:],
                "calendar_sha256": calendar["sha256"],
            }
        )
        if train_sessions is not None:
            result[-1]["training_window_sessions"] = train_sessions
        if target["session_id"] > data.dates[-1]:
            break
        origin = target
    return result
