"""History indexing preserves tied evidence, temporal visibility and quote conflicts."""

from copy import deepcopy
from datetime import timedelta, timezone
from types import SimpleNamespace

import pytest
from market_helpers import contract_and_events

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.forecast.forecast_features import _history


def append_event(data, event, *, received=None):
    ref = digest(canonical(event))
    data.snapshot["receipt_watermark"] += 1
    data.snapshot["events"].append(
        dict(
            event=event,
            payload_ref=ref,
            receipt_seq=data.snapshot["receipt_watermark"],
            first_seen_at=received or event["source_available_at"],
        )
    )
    return ref


@pytest.fixture
def source():
    contract, events, _ = contract_and_events(instruments=("A", "B"), count=4)
    data = SimpleNamespace(
        instruments=["A"], contract=contract, snapshot=dict(receipt_watermark=0, events=[])
    )
    for event in events:
        append_event(data, event)
    return data, SimpleNamespace(notices=[])


def event(data, event_id):
    return deepcopy(
        next(x["event"] for x in data.snapshot["events"] if x["event"]["event_id"] == event_id)
    )


@pytest.mark.parametrize("second_status", ["active", "inactive"])
def test_latest_membership_keeps_all_equal_time_evidence(source, second_status):
    data, _ = source
    closes = [s.close_at for s in data.contract.calendar.sessions]
    first = event(data, "A.membership")
    first.update(
        event_id="first", event_at=closes[1].isoformat(), source_available_at=closes[1].isoformat()
    )
    ref_a = append_event(data, first)
    second = first | dict(
        event_id="second",
        status=second_status,
        event_at=closes[1].astimezone(timezone(timedelta(hours=-4))).isoformat(),
    )
    ref_b = append_event(data, second)
    prices, reasons, refs = _history(source, closes[1:3], closes[3], None)
    assert {ref_a, ref_b}.issubset(refs)
    assert digest(canonical(event(data, "A.membership"))) not in refs
    if second_status == "active":
        assert prices == [110, 121] and reasons == []
    else:
        assert prices is None and reasons == ["history_membership_unknown_or_inactive"]


def test_late_membership_does_not_contaminate_earlier_calls(source):
    data, _ = source
    closes = [s.close_at for s in data.contract.calendar.sessions]
    inactive = event(data, "A.membership") | dict(
        event_id="inactive",
        status="inactive",
        event_at=closes[1].isoformat(),
        source_available_at=closes[3].isoformat(),
    )
    ref = append_event(data, inactive)
    before = _history(source, closes[1:3], closes[2], None)
    assert before[0] == [110, 121] and ref not in before[2]
    later = _history(source, closes[1:3], closes[3], None)
    assert later[0] is None and ref in later[2]
    assert _history(source, closes[1:3], closes[2], None) == before


def test_duplicate_quote_is_not_silently_overwritten(source):
    data, _ = source
    closes = [s.close_at for s in data.contract.calendar.sessions]
    quote = event(data, "A.CLOSE_QUOTE.1")
    duplicate = append_event(data, quote | dict(event_id="conflict", price="999"))
    prices, reasons, refs = _history(source, closes[1:3], closes[3], None)
    assert prices is None and reasons == ["history_quote_missing_invalid_or_conflicting"]
    assert {duplicate, digest(canonical(quote))}.issubset(refs)


@pytest.mark.parametrize("basis,early_price", [(None, 123), ("observed_receipts", 110)])
def test_quote_revision_uses_the_requested_knowledge_boundary(source, basis, early_price):
    data, _ = source
    closes = [s.close_at for s in data.contract.calendar.sessions]
    quote = event(data, "A.CLOSE_QUOTE.1")
    revised = quote | dict(revision=1, price="123", source_available_at=closes[2].isoformat())
    ref = append_event(data, revised, received=closes[3].isoformat())
    before = _history(source, closes[1:3], closes[2], basis)
    assert before[0] == [early_price, 121]
    later = _history(source, closes[1:3], closes[3], basis)
    assert later[0] == [123, 121] and ref in later[2]
    assert digest(canonical(quote)) not in later[2]
    assert _history(source, closes[1:3], closes[2], basis) == before


def test_missing_calendar_close_cannot_borrow_a_nearby_quote(source):
    data, _ = source
    closes = [s.close_at for s in data.contract.calendar.sessions]
    assert _history(source, [closes[1] + timedelta(minutes=30)], closes[3], None) == (
        None,
        ["context_calendar_close_missing"],
        [],
    )
