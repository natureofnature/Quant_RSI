"""Calendar-month folds with cutoff-only, fixed-width training histories."""

import json
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
