"""Parse and validate local source-event CSV batches (neutral market data layer).

Shared by the research study's daily price collections and the frozen forecast
workflow's event store (#159 D2); imports only core and market contracts.
"""

import csv
import io
import re

from quant_rsi.core.artifacts import canonical
from quant_rsi.market.market_contracts import SourceEvent, timestamp


def validate_event(contract, event, received_at):
    if event.instrument not in contract.instruments:
        raise ValueError("event instrument lies outside the registered identity grid")
    sessions = {session.session_id: session for session in contract.calendar.sessions}
    if event.kind in {"OPEN_QUOTE", "CLOSE_QUOTE"}:
        if event.session_id not in sessions:
            raise ValueError("quote session is absent from the registered calendar")
        expected = (
            sessions[event.session_id].open_at
            if event.kind == "OPEN_QUOTE"
            else sessions[event.session_id].close_at
        )
        if event.event_at != expected:
            raise ValueError("quote event time differs from the registered session phase")
        if event.event_at > timestamp(received_at):
            raise ValueError("future market prices cannot be imported as received observations")
    elif event.session_id is not None and event.session_id not in sessions:
        raise ValueError("status/action references an unknown calendar session")
    if (
        contract.knowledge_basis == "documented_historical_availability"
        and event.source_available_at is None
    ):
        raise ValueError("documented history requires source availability on every revision")


def parse_csv(contract, raw, received_at):
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("local import exceeds the supported 64 MiB batch limit")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if (
        reader.fieldnames is None
        or len(set(reader.fieldnames)) != len(reader.fieldnames)
        or set(reader.fieldnames) != set(contract.columns.values())
    ):
        raise ValueError("CSV headers must exactly match the explicit one-to-one mapping")
    events, seen = [], set()
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("CSV row has a missing or extra column")
        value = {name: (row[column].strip() or None) for name, column in contract.columns.items()}
        revision = value.get("revision")
        if revision is None or not re.fullmatch(r"[0-9]+", revision):
            raise ValueError("revision must be an explicit nonnegative integer")
        value["revision"] = int(revision)
        for field in ("session_id", "source_available_at", "price", "status", "action"):
            value.setdefault(field, None)
        event = SourceEvent.model_validate_json(canonical(value))
        validate_event(contract, event, received_at)
        key = (event.event_id, event.revision)
        if key in seen:
            raise ValueError("duplicate event revision within an import")
        seen.add(key)
        events.append(event)
    if not events:
        raise ValueError("input contains no source events")
    return events
